import json
import subprocess
import sys

import numpy as np
import pymc as pm
import pytest
import torch

import pymc_pfn as pfn
from pymc_pfn.model import PFNModule, from_grid, integrate, to_grid, token_mask
from pymc_pfn.predict import prepare

KINDS = {"x": "cell", "y": "row", "beta": "entity", "effect": "cell", "sigma": "global"}
OBSERVED, TARGETS = ["x", "y"], ["beta", "effect", "sigma"]


def regression(n_obs: int, n_groups: int) -> pm.Model:
    with pm.Model(coords={"obs": range(n_obs), "group": range(n_groups)}) as m:
        x = pm.Normal("x", dims=("obs", "group"))
        beta = pm.Normal("beta", dims="group")
        sigma = pm.HalfNormal("sigma", 0.5)
        pm.Deterministic("effect", x * beta, dims=("obs", "group"))
        pm.Normal("y", (x * beta).sum(axis=-1), sigma, dims="obs")
    return m


def make_prior() -> pfn.Prior:
    return pfn.Prior(
        regression,
        sizes={"n_obs": [12, 24], "n_groups": [2, 3]},
        observed=OBSERVED,
        targets=TARGETS,
        row_dim="obs",
        entity_dim="group",
        prefix_rows="n_obs",
    )


TINY = pfn.ModelConfig(d_model=32, n_heads=2, encoder_layers=2, decoder_layers=1)


def config(**overrides) -> pfn.FitConfig:
    base = {
        "steps": 300,
        "batch_size": 32,
        "eval_every": 100,
        "warmup": 50,
        "lr": 2e-3,
        "model": TINY,
        "norm_tasks": 256,
        "holdout_tasks": 64,
        "prefetch": 0,
        "device": "cpu",
    }
    return pfn.FitConfig(**{**base, **overrides})


@pytest.fixture(scope="module")
def prior() -> pfn.Prior:
    return make_prior()


@pytest.fixture(scope="module")
def trained(prior, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return pfn.fit(prior, config(), out=out, verbose=False), out


def random_inputs(b=3, r=5, e=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    arrays = {
        "x": torch.randn(b, r, e, generator=g),
        "y": torch.randn(b, r, generator=g),
        "beta": torch.randn(b, e, generator=g),
        "effect": torch.randn(b, r, e, generator=g),
        "sigma": torch.randn(b, generator=g),
    }
    return (
        arrays,
        torch.ones(b, r, dtype=torch.bool),
        torch.ones(b, e, dtype=torch.bool),
    )


def run_module(module, arrays, row_mask, entity_mask):
    """Deterministic pass: encode, then one velocity evaluation with z = targets."""
    r, e = row_mask.shape[1], entity_mask.shape[1]
    mask = token_mask(row_mask, entity_mask)
    obs, present = to_grid(arrays, OBSERVED, KINDS, r, e)
    tgt, tpresent = to_grid(arrays, TARGETS, KINDS, r, e)
    valid = tpresent[None] & mask[..., None]
    encoded = module.encode(obs, present[None] & mask[..., None], mask)
    v = module.velocity(encoded, tgt, torch.full((len(obs),), 0.3), valid, mask)
    return from_grid(v, TARGETS, KINDS, r, e)


def test_grid_roundtrip():
    arrays, _, _ = random_inputs()
    names = OBSERVED + TARGETS
    grid, present = to_grid(arrays, names, KINDS, 5, 4)
    assert grid.shape == (3, 6, 5, 5)
    assert present.sum(dim=(0, 1)).tolist() == [20, 5, 4, 20, 1]
    back = from_grid(grid, names, KINDS, 5, 4)
    for name in names:
        torch.testing.assert_close(back[name], arrays[name])


@torch.no_grad()
def test_padding_does_not_change_real_outputs():
    module = PFNModule(2, 3, TINY).eval()
    arrays, rows, ents = random_inputs()
    reference = run_module(module, arrays, rows, ents)

    def pad(t, kind):
        shape = {"cell": (3, 8, 6), "row": (3, 8), "entity": (3, 6), "global": (3,)}[
            kind
        ]
        out = torch.full(shape, 99.0)  # garbage in the padding
        out[tuple(slice(0, s) for s in t.shape)] = t
        return out

    padded = {k: pad(v, KINDS[k]) for k, v in arrays.items()}
    rows_p = torch.zeros(3, 8, dtype=torch.bool)
    rows_p[:, :5] = True
    ents_p = torch.zeros(3, 6, dtype=torch.bool)
    ents_p[:, :4] = True
    out = run_module(module, padded, rows_p, ents_p)
    torch.testing.assert_close(
        out["beta"][:, :4], reference["beta"], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        out["effect"][:, :5, :4], reference["effect"], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(out["sigma"], reference["sigma"], atol=1e-5, rtol=1e-4)
    assert (out["effect"][:, 5:] == 0).all() and (out["beta"][:, 4:] == 0).all()


@torch.no_grad()
def test_entity_and_row_permutation_equivariance():
    module = PFNModule(2, 3, TINY).eval()
    arrays, rows, ents = random_inputs()
    reference = run_module(module, arrays, rows, ents)
    pe, pr = torch.tensor([2, 0, 3, 1]), torch.tensor([4, 1, 0, 3, 2])
    permuted = {
        "x": arrays["x"][:, pr][:, :, pe],
        "y": arrays["y"][:, pr],
        "beta": arrays["beta"][:, pe],
        "effect": arrays["effect"][:, pr][:, :, pe],
        "sigma": arrays["sigma"],
    }
    out = run_module(module, permuted, rows, ents)
    torch.testing.assert_close(
        out["beta"], reference["beta"][:, pe], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        out["effect"], reference["effect"][:, pr][:, :, pe], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(out["sigma"], reference["sigma"], atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_ordered_rows_breaks_row_permutation_invariance():
    cfg = pfn.ModelConfig(
        d_model=32, n_heads=2, encoder_layers=1, decoder_layers=1, ordered_rows=True
    )
    module = PFNModule(2, 3, cfg).eval()
    arrays, rows, ents = random_inputs()
    reference = run_module(module, arrays, rows, ents)
    flipped = {
        **arrays,
        "x": arrays["x"].flip(1),
        "y": arrays["y"].flip(1),
        "effect": arrays["effect"].flip(1),
    }
    assert not torch.allclose(
        run_module(module, flipped, rows, ents)["beta"], reference["beta"]
    )


def test_training_learns_and_beats_the_prior_only_baseline(trained, prior):
    net, out = trained
    records = [
        json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == [100, 200, 300]
    assert records[-1]["best_holdout_loss"] < records[0]["holdout_loss"]

    test = prior.sample(64, seed=4242)
    draws = net.sample(test, draws=32, seed=0, steps=8)
    mask = test.mask("beta")
    truth = test.targets["beta"]
    rmse = np.sqrt(((draws["beta"].mean(axis=1) - truth)[mask] ** 2).mean())
    prior_only = np.sqrt(((truth - net.spec.mean["beta"])[mask] ** 2).mean())
    assert rmse < 0.7 * prior_only
    assert (
        draws["beta"][~np.broadcast_to(mask[:, None], draws["beta"].shape)] == 0
    ).all()
    assert draws["effect"].shape == (64, 32, 24, 3)


def test_draws_are_joint_not_marginal(trained, prior):
    """effect = x * beta holds in the prior, so joint draws must respect it roughly."""
    net, _ = trained
    test = prior.sample(16, seed=7)
    draws = net.sample(test, draws=64, seed=0, steps=8)
    implied = test.observed["x"][:, None] * draws["beta"][:, :, None, :]
    mask = np.broadcast_to(test.mask("effect")[:, None], implied.shape)
    joint = np.corrcoef(implied[mask], draws["effect"][mask])[0, 1]
    shuffled = draws["effect"][:, ::-1]  # break the pairing between draws
    broken = np.corrcoef(implied[mask], shuffled[mask])[0, 1]
    assert joint > broken


def test_bundle_roundtrip_and_format_guard(trained, prior, tmp_path):
    net, out = trained
    loaded = pfn.PFN.load(out / "bundle", device="cpu")
    assert loaded.spec.prior_fingerprint == prior.fingerprint()
    test = prior.sample(4, seed=1)
    a = net.sample(test, draws=4, seed=3, steps=4)
    b = loaded.sample(test, draws=4, seed=3, steps=4)
    np.testing.assert_allclose(a["beta"], b["beta"], atol=1e-5)

    net.save(tmp_path / "copy")
    spec = json.loads((tmp_path / "copy" / "spec.json").read_text())
    spec["format_version"] = 99
    (tmp_path / "copy" / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="format"):
        pfn.PFN.load(tmp_path / "copy")


def test_predict_single_dataset(trained):
    net, _ = trained
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3)).astype(np.float32)
    beta = np.array([1.5, -1.0, 0.0])
    y = x @ beta + 0.2 * rng.normal(size=20)
    post = net.predict({"x": x, "y": y}, draws=64, steps=8)
    assert post["beta"].shape == (64, 3) and post["effect"].shape == (64, 20, 3)
    assert post["sigma"].shape == (64,)
    estimate = post.mean()["beta"]
    assert estimate[0] > estimate[2] > estimate[1]
    low, high = post.quantile([0.1, 0.9])["beta"]
    assert (low < high).all()
    tree = post.to_datatree()
    assert tree["posterior"]["beta"].shape == (1, 64, 3)
    with pytest.raises(KeyError, match="missing"):
        net.predict({"x": x})
    with pytest.raises(ValueError, match="inconsistent"):
        net.predict({"x": x, "y": y[:5]})


def test_resume_continues_from_checkpoint(prior, tmp_path):
    pfn.fit(prior, config(steps=20, eval_every=10), out=tmp_path, verbose=False)
    pfn.fit(
        prior, config(steps=40, eval_every=10), out=tmp_path, resume=True, verbose=False
    )
    steps = [
        json.loads(line)["step"]
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert steps == [10, 20, 30, 40]
    with pytest.raises(FileNotFoundError):
        pfn.fit(
            prior,
            config(steps=10),
            out=tmp_path / "nothing",
            resume=True,
            verbose=False,
        )


def test_unknown_integration_method():
    module = PFNModule(2, 3, TINY)
    with pytest.raises(ValueError, match="method"):
        integrate(module, None, torch.zeros(1), None, None, 2, "rk4")


def test_prepare_marks_only_real_values(prior):
    batch = prior.sample(8, seed=0)
    spec = pfn.fit(prior, config(steps=1, eval_every=1), verbose=False).spec
    p = prepare(batch, spec, "cpu")
    assert int(p["target_valid"][..., 0].sum()) == int(batch.mask("beta").sum())
    assert int(p["observed_valid"][..., 0].sum()) == int(batch.mask("x").sum())


def test_importing_the_package_does_not_import_torch():
    code = "import sys, pymc_pfn; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
