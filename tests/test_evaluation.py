import json

import numpy as np
import pymc as pm
import pytest
from conftest import make_prior

import pymc_pfn as pfn
from pymc_pfn.evaluation import crps, nuts_posterior, sample_grouped, sbc, score


def ols(batch: pfn.TaskBatch) -> dict[str, np.ndarray]:
    beta = np.zeros(batch.entity_mask.shape, dtype=np.float32)
    for i in range(len(batch)):
        r, e = int(batch.row_mask[i].sum()), int(batch.entity_mask[i].sum())
        beta[i, :e] = np.linalg.lstsq(
            batch.observed["x"][i, :r, :e], batch.observed["y"][i, :r]
        )[0]
    return {"beta": beta}


def wide(n_obs: int, n_groups: int) -> pm.Model:
    """Coefficients five times larger than anything in the training prior."""
    with pm.Model(coords={"obs": range(n_obs), "group": range(n_groups)}) as m:
        x = pm.Normal("x", dims=("obs", "group"))
        beta = pm.Normal("beta", 0.0, 5.0, dims="group")
        sigma = pm.HalfNormal("sigma", 0.5)
        pm.Deterministic("effect", x * beta, dims=("obs", "group"))
        pm.Normal("y", (x * beta).sum(axis=-1), sigma, dims="obs")
    return m


def test_crps_matches_the_normal_closed_form():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(1, 20_000, 3))
    truth = np.array([[0.0, 1.0, -2.0]])
    # CRPS of N(0, 1) at y: y (2 Phi(y) - 1) + 2 phi(y) - 1 / sqrt(pi)
    expected = np.array([0.23369, 0.60244, 1.45348])
    np.testing.assert_allclose(crps(draws, truth)[0], expected, atol=0.01)


def test_crps_of_a_point_mass_is_the_absolute_error():
    draws = np.full((2, 16, 4), 1.5)
    truth = np.zeros((2, 4))
    np.testing.assert_allclose(crps(draws, truth), 1.5)


def test_score_ignores_padding_and_reports_calibrated_coverage():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(400, 6))
    draws = rng.normal(size=(400, 200, 6))
    mask = np.ones_like(truth, dtype=bool)
    mask[:, 4:] = False
    truth[:, 4:] = 1e6  # padding must not matter
    result = score(draws, truth, mask)
    assert abs(result["cov80"] - 0.8) < 0.03
    assert abs(result["cov50"] - 0.5) < 0.04
    assert result["rmse"] < 1.1
    point = score(np.zeros_like(truth), truth, mask)
    assert point["cov80"] is None
    assert point["crps"] == pytest.approx(np.abs(truth[mask]).mean())


def test_sbc_accepts_calibrated_and_rejects_overconfident_draws():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(600, 5))
    mask = np.ones_like(truth, dtype=bool)
    calibrated = rng.normal(size=(600, 127, 5))
    assert sbc(calibrated, truth, mask, seed=0)["p_value"] > 0.01
    assert sbc(0.5 * calibrated, truth, mask, seed=0)["p_value"] < 1e-4
    assert sum(sbc(calibrated, truth, mask, seed=0)["counts"]) == 600


def test_sample_grouped_fills_real_positions_only(trained, prior):
    net, _ = trained
    batch = prior.sample(12, seed=3, group_size=3)
    draws = sample_grouped(net, batch, draws=4, steps=2)
    assert draws["effect"].shape == (12, 4, batch.n_rows, batch.n_entities)
    mask = batch.mask("effect")
    assert (
        draws["effect"][~np.broadcast_to(mask[:, None], draws["effect"].shape)] == 0
    ).all()
    assert (
        draws["beta"][np.broadcast_to(batch.entity_mask[:, None], draws["beta"].shape)]
        != 0
    ).all()


@pytest.fixture(scope="module")
def report(trained, prior, tmp_path_factory):
    net, _ = trained
    flip = pfn.Symmetry(
        "sign flip",
        observed=lambda o: {**o, "x": -o["x"]},
        targets=lambda t: {**t, "beta": -t["beta"]},
    )
    stress = pfn.Prior(
        wide,
        sizes=prior.sizes,
        observed=prior.observed,
        targets=prior.targets,
        row_dim="obs",
        entity_dim="group",
        prefix_rows="n_obs",
    )
    path = tmp_path_factory.mktemp("report") / "report.json"
    result = pfn.evaluate(
        net,
        prior,
        n_tasks=96,
        draws=32,
        steps=4,
        baselines={"ols": ols},
        stress={"wide": stress},
        symmetries=[flip],
        sampler_grid=[("euler", 4), ("heun", 8)],
        sampler_tasks=32,
        out=path,
    )
    return result, path


def test_report_has_every_section(report):
    result, path = report
    assert set(result.accuracy) == {"network", "prior_only", "ols"}
    assert set(result.accuracy["network"]) == {"beta", "effect", "sigma"}
    assert set(result.accuracy["ols"]) == {"beta"}
    assert set(result.sbc) == {"beta", "effect", "sigma"}
    assert [(s["method"], s["steps"]) for s in result.sampler] == [
        ("heun", 4),
        ("euler", 4),
        ("heun", 8),
    ]
    assert set(result.symmetry["sign flip"]) == {"beta", "effect", "sigma"}
    assert result.reference is None
    assert json.loads(path.read_text()) == json.loads(json.dumps(result.to_dict()))
    text = str(result)
    for word in (
        "prior_only",
        "SBC",
        "Sampler check",
        "Stress family",
        "Symmetry",
        "flag",
    ):
        assert word in text


def test_network_beats_the_prior_only_baseline(report):
    accuracy = report[0].accuracy
    for metric in ("rmse", "crps"):
        assert (
            accuracy["network"]["beta"][metric]
            < 0.8 * accuracy["prior_only"]["beta"][metric]
        )
    assert abs(accuracy["prior_only"]["beta"]["cov80"] - 0.8) < 0.1


def test_out_of_prior_family_is_flagged(report):
    result = report[0]
    section = result.stress["wide"]
    assert section["support"]["y"]["z_sd"] > 2
    worse = section["accuracy"]["network"]["beta"]["rmse"]
    assert worse > 2 * result.accuracy["network"]["beta"]["rmse"]
    assert any(flag.startswith("stress wide: beta RMSE") for flag in result.flags)


def test_evaluate_rejects_a_prior_with_other_variables(trained):
    net, _ = trained
    other = pfn.Prior(
        wide,
        sizes={"n_obs": [8], "n_groups": [2]},
        observed=["x", "y"],
        targets=["beta"],
        row_dim="obs",
        entity_dim="group",
    )
    with pytest.raises(ValueError, match="does not match the network"):
        pfn.evaluate(net, other, n_tasks=4, draws=4, steps=2, sampler_grid=[])


def test_nuts_reference_recovers_the_coefficients():
    prior = make_prior()
    batch = prior.sample(1, seed=5)
    draws, converged = nuts_posterior(prior, batch, 0, draws=300, tune=500, chains=2)
    e = int(batch.entity_mask[0].sum())
    r = int(batch.row_mask[0].sum())
    assert converged
    assert draws["beta"].shape == (600, e)
    assert draws["effect"].shape == (600, r, e)
    truth = batch.targets["beta"][0, :e]
    z = (draws["beta"].mean(axis=0) - truth) / draws["beta"].std(axis=0)
    assert np.abs(z).max() < 4


def test_reference_section(trained, prior):
    net, _ = trained
    result = pfn.evaluate(
        net,
        prior,
        n_tasks=8,
        draws=32,
        steps=4,
        sampler_grid=[],
        reference_tasks=2,
        reference_kwargs={"draws": 200, "tune": 300, "chains": 2},
    )
    assert result.reference["n_tasks"] == 2
    if result.reference["n_converged"]:
        row = result.reference["targets"]["beta"]
        assert row["crps_nuts"] > 0
        assert row["sd_ratio"] > 0
