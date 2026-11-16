"""Per-task scaling: tasks enter the network in their own units."""

import numpy as np
import pytest
from conftest import KINDS, OBSERVED, TARGETS, config

import pymc_pfn as pfn
from pymc_pfn import scaling

UNITS = {"x": "x", "y": "y", "beta": "y/x", "effect": "y", "sigma": "y"}


@pytest.fixture(scope="module")
def batch(prior):
    return prior.sample(48, seed=5)


@pytest.fixture(scope="module")
def scaled_net(prior, tmp_path_factory):
    out = tmp_path_factory.mktemp("scaled")
    pfn.fit(prior, config(steps=60, units=UNITS), out=out, verbose=False)
    return pfn.PFN.load(out / "bundle", device="cpu")


def test_observed_are_standardised_per_task(batch):
    scaled, _ = scaling.apply(batch, UNITS, KINDS)
    y, mask = scaled.observed["y"], batch.row_mask
    for i in range(len(batch)):
        np.testing.assert_allclose(y[i, mask[i]].mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(y[i, mask[i]].std(), 1.0, atol=1e-5)
    x = scaled.observed["x"]
    real = batch.mask("x")
    assert not x[~real].any()
    column_sd = np.sqrt((x**2).sum(axis=1) / mask.sum(axis=1)[:, None])
    np.testing.assert_allclose(column_sd[batch.entity_mask], 1.0, atol=1e-5)


def test_scales_are_kept_as_inputs(batch):
    scaled, _ = scaling.apply(batch, UNITS, KINDS)
    assert scaled.kinds["log_scale[y]"] == "global"
    assert scaled.kinds["log_scale[x]"] == "entity"
    assert scaled.kinds["center[y]"] == "global"
    sd = np.array([batch.observed["y"][i, batch.row_mask[i]].std() for i in range(8)])
    np.testing.assert_allclose(scaled.observed["log_scale[y]"][:8], np.log(sd), 1e-5)
    assert not scaled.observed["log_scale[x]"][~batch.entity_mask].any()


def test_factors_invert_targets(batch):
    scaled, factors = scaling.apply(batch, UNITS, KINDS)
    for name in TARGETS:
        back = scaled.targets[name] * factors[name] * batch.mask(name)
        np.testing.assert_allclose(back, batch.targets[name], rtol=1e-4, atol=1e-6)


def test_no_units_changes_nothing(batch):
    same, factors = scaling.apply(batch, {}, KINDS)
    assert same is batch
    assert factors == {}


@pytest.mark.parametrize(
    ("units", "error", "match"),
    [
        ({"beta": "sigma"}, KeyError, "not observed"),
        ({"gamma": "y"}, KeyError, "not a declared variable"),
        ({"sigma": "x"}, ValueError, "one scale per entity"),
        ({"beta": "y/x/x"}, ValueError, "must look like"),
    ],
)
def test_bad_units(units, error, match):
    with pytest.raises(error, match=match):
        scaling.validate(units, OBSERVED, TARGETS, KINDS)


def test_bundle_keeps_units(scaled_net):
    assert scaled_net.spec.units == UNITS
    assert scaled_net.spec.observed == OBSERVED
    assert set(scaled_net.spec.inputs) == {
        *OBSERVED,
        "log_scale[x]",
        "log_scale[y]",
        "center[x]",
        "center[y]",
    }


def test_scaled_data_do_not_depend_on_units(batch):
    moved = pfn.TaskBatch(
        {"x": batch.observed["x"] * 7.0, "y": batch.observed["y"] * 0.01},
        {},
        batch.row_mask,
        batch.entity_mask,
        batch.kinds,
    )
    a, _ = scaling.apply(batch, UNITS, KINDS)
    b, _ = scaling.apply(moved, UNITS, KINDS)
    for name in ("x", "y", "center[x]"):
        np.testing.assert_allclose(a.observed[name], b.observed[name], atol=1e-4)
    shift = b.observed["log_scale[y]"] - a.observed["log_scale[y]"]
    np.testing.assert_allclose(shift, np.log(0.01), atol=1e-4)


def test_draws_are_equivariant_without_scale_inputs(prior, batch):
    """Rescaling y and the columns of x rescales the draws exactly."""
    scaled_net = pfn.fit(
        prior, config(steps=2, units=UNITS, scale_inputs=False), verbose=False
    )
    assert scaled_net.spec.inputs == OBSERVED
    part = batch[:6]
    rng = np.random.default_rng(0)
    c_y = 250.0
    c_x = rng.uniform(0.01, 100.0, size=part.n_entities).astype(np.float32)
    moved = pfn.TaskBatch(
        observed={"x": part.observed["x"] * c_x, "y": part.observed["y"] * c_y},
        targets={},
        row_mask=part.row_mask,
        entity_mask=part.entity_mask,
        kinds=part.kinds,
    )
    base = scaled_net.sample(part, draws=8, seed=1)
    other = scaled_net.sample(moved, draws=8, seed=1)
    expected = {
        "beta": base["beta"] * c_y / c_x,
        "effect": base["effect"] * c_y,
        "sigma": base["sigma"] * c_y,
    }
    for name, values in expected.items():
        tolerance = 2e-3 * np.abs(values).max()
        np.testing.assert_allclose(other[name], values, rtol=2e-3, atol=tolerance)


def test_evaluate_runs_with_units(scaled_net, prior):
    report = pfn.evaluate(
        scaled_net, prior, n_tasks=16, draws=16, sampler_grid=[], stress={"same": prior}
    )
    assert "log_scale[y]" in report.stress["same"]["support"]
