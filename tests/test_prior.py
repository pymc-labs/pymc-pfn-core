import subprocess
import sys

import numpy as np
import pymc as pm
import pytest

import pymc_pfn as pfn


def regression(n_obs: int, n_groups: int, flip_dims: bool = False) -> pm.Model:
    with pm.Model(coords={"obs": range(n_obs), "group": range(n_groups)}) as m:
        x = pm.Normal("x", dims=("obs", "group"))
        beta = pm.Normal("beta", dims="group")
        sigma = pm.HalfNormal("sigma")
        if flip_dims:
            pm.Deterministic("effect", (x * beta).T, dims=("group", "obs"))
        else:
            pm.Deterministic("effect", x * beta, dims=("obs", "group"))
        pm.Normal("y", (x * beta).sum(axis=-1), sigma, dims="obs")
    return m


def make_prior(**overrides) -> pfn.Prior:
    kwargs = {
        "sizes": {"n_obs": [8, 16], "n_groups": [2, 3]},
        "observed": ["x", "y"],
        "targets": ["beta", "effect", "sigma"],
        "row_dim": "obs",
        "entity_dim": "group",
        "prefix_rows": "n_obs",
    }
    kwargs.update(overrides)
    return pfn.Prior(kwargs.pop("build", regression), **kwargs)


@pytest.fixture(scope="module")
def batch() -> pfn.TaskBatch:
    return make_prior().sample(64, seed=1, group_size=8)


def test_kinds_and_shapes(batch):
    assert batch.kinds == {
        "x": "cell",
        "y": "row",
        "beta": "entity",
        "effect": "cell",
        "sigma": "global",
    }
    assert batch.observed["x"].shape == (64, 16, 3)
    assert batch.observed["y"].shape == (64, 16)
    assert batch.targets["beta"].shape == (64, 3)
    assert batch.targets["sigma"].shape == (64,)


def test_masks_match_sizes_and_padding_is_zero(batch):
    for i, sizes in enumerate(batch.sizes):
        assert batch.row_mask[i].sum() == sizes["n_obs"]
        assert batch.entity_mask[i].sum() == sizes["n_groups"]
    assert {s["n_obs"] for s in batch.sizes} == {8, 16}
    assert {s["n_groups"] for s in batch.sizes} == {2, 3}
    for name in ("x", "effect"):
        arr = batch.observed.get(name, batch.targets.get(name))
        assert (arr[~batch.mask(name)] == 0).all()
        assert (arr[batch.mask(name)] != 0).all()


def test_variables_come_from_one_joint_draw(batch):
    x, beta, effect = (
        batch.observed["x"],
        batch.targets["beta"],
        batch.targets["effect"],
    )
    np.testing.assert_allclose(effect, x * beta[:, None, :], rtol=1e-5, atol=1e-6)


def test_dims_are_transposed_to_row_entity_order():
    flipped = make_prior(build=lambda **s: regression(flip_dims=True, **s)).sample(
        8, seed=0
    )
    assert flipped.targets["effect"].shape == (8, 16, 3)
    np.testing.assert_allclose(
        flipped.targets["effect"],
        flipped.observed["x"] * flipped.targets["beta"][:, None, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_same_seed_reproduces_on_a_fresh_prior(batch):
    again = make_prior().sample(64, seed=1, group_size=8)
    for name in batch.observed:
        np.testing.assert_array_equal(batch.observed[name], again.observed[name])
    other = make_prior().sample(64, seed=2, group_size=8)
    assert not np.array_equal(batch.observed["y"], other.observed["y"])


def test_without_prefix_rows_each_row_count_is_built():
    prior = make_prior(prefix_rows=None)
    batch = prior.sample(64, seed=0, group_size=2)
    assert len(prior._compiled) == 4
    assert {int(m.sum()) for m in batch.row_mask} == {8, 16}


def test_with_prefix_rows_only_entity_counts_are_built():
    prior = make_prior()
    prior.sample(32, seed=0, group_size=4)
    assert len(prior._compiled) == 2


def test_reject_filters_and_counts():
    prior = make_prior(reject=lambda task: task["sigma"] > 0.5)
    batch = prior.sample(64, seed=0)
    assert (batch.targets["sigma"] <= 0.5).all()
    assert len(batch) == 64
    assert prior.n_rejected > 0 and prior.n_drawn == 64 + prior.n_rejected


def test_reject_everything_raises():
    with pytest.raises(RuntimeError, match="reject"):
        make_prior(reject=lambda task: True, max_reject_factor=2).sample(8)


def test_unknown_variable_and_unsupported_dims():
    with pytest.raises(KeyError, match="nope"):
        make_prior(targets=["nope"]).sample(2)

    def extra_dim(n_obs, n_groups):
        with pm.Model(
            coords={"obs": range(n_obs), "group": range(n_groups), "k": range(2)}
        ) as m:
            pm.Normal("x", dims=("obs", "group"))
            pm.Normal("z", dims=("obs", "k"))
        return m

    with pytest.raises(ValueError, match="'z'"):
        make_prior(build=extra_dim, observed=["x"], targets=["z"]).sample(2)


def test_invalid_declarations():
    with pytest.raises(ValueError, match="both"):
        make_prior(targets=["x"])
    with pytest.raises(KeyError, match="prefix_rows"):
        make_prior(prefix_rows="rows")
    with pytest.raises(ValueError, match="positive"):
        make_prior(sizes={"n_obs": [], "n_groups": [2]})


def test_stream_and_prefetch_yield_batches():
    for prefetch in (0, 2):
        stream = make_prior().stream(batch_size=8, seed=0, prefetch=prefetch)
        first, second = next(stream), next(stream)
        assert len(first) == len(second) == 8
        assert not np.array_equal(first.observed["y"], second.observed["y"])


def test_prefetch_surfaces_sampler_errors():
    stream = make_prior(reject=lambda task: True, max_reject_factor=1).stream(
        4, prefetch=1
    )
    with pytest.raises(RuntimeError, match="reject"):
        next(stream)


def test_save_load_roundtrip(batch, tmp_path):
    path = tmp_path / "tasks.npz"
    batch.save(path)
    loaded = pfn.TaskBatch.load(path)
    assert loaded.kinds == batch.kinds and loaded.sizes == batch.sizes
    np.testing.assert_array_equal(loaded.row_mask, batch.row_mask)
    for name in batch.targets:
        np.testing.assert_array_equal(loaded.targets[name], batch.targets[name])


def test_shape_contract_is_enforced(batch):
    with pytest.raises(ValueError, match="must have shape"):
        pfn.TaskBatch(
            observed={"y": batch.observed["y"][:, :4]},
            targets={},
            row_mask=batch.row_mask,
            entity_mask=batch.entity_mask,
            kinds={"y": "row"},
        )


def test_importing_the_package_does_not_import_pymc():
    code = "import sys, pymc_pfn; sys.exit('pymc' in sys.modules or 'pytensor' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
