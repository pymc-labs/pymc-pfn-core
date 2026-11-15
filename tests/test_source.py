"""Simulator: a plain function as the task source, without PyMC."""

import subprocess
import sys

import numpy as np
import pytest
from conftest import config

import pymc_pfn as pfn

KINDS = {"x": "cell", "y": "row", "beta": "entity", "sigma": "global"}


def regression(rng, n_obs, n_groups):
    x = rng.normal(size=(n_obs, n_groups))
    beta = rng.normal(size=n_groups)
    sigma = abs(rng.normal()) * 0.5
    y = x @ beta + sigma * rng.normal(size=n_obs)
    return {"x": x, "y": y, "beta": beta, "sigma": sigma}


def make(simulate=regression, **overrides) -> pfn.Simulator:
    kwargs = {
        "sizes": {"n_obs": [12, 24], "n_groups": [2, 3]},
        "observed": ["x", "y"],
        "targets": ["beta", "sigma"],
        "kinds": KINDS,
    }
    return pfn.Simulator(simulate, **{**kwargs, **overrides})


def test_batch_layout():
    batch = make().sample(64, seed=1)
    assert batch.kinds == KINDS
    assert batch.observed["x"].shape == (64, 24, 3)
    assert batch.targets["sigma"].shape == (64,)
    for i, sizes in enumerate(batch.sizes):
        assert batch.row_mask[i].sum() == sizes["n_obs"]
        assert batch.entity_mask[i].sum() == sizes["n_groups"]
    padded = ~batch.mask("x")
    assert padded.any()
    assert not batch.observed["x"][padded].any()


def test_seed_reproducible_on_a_used_object():
    source = make()
    first = source.sample(32, seed=3)
    again = source.sample(32, seed=3)
    other = source.sample(32, seed=4)
    np.testing.assert_array_equal(first.observed["y"], again.observed["y"])
    assert not np.array_equal(first.observed["y"], other.observed["y"])


def test_reject_is_counted():
    source = make(reject=lambda task: task["sigma"] < 0.2)
    batch = source.sample(64, seed=0)
    assert batch.targets["sigma"].min() >= 0.2
    assert source.n_rejected > 0
    assert source.n_drawn == 64 + source.n_rejected


def test_fingerprint_follows_declarations():
    assert make().fingerprint() == make().fingerprint()
    assert make().fingerprint() != make(targets=["beta"]).fingerprint()


def test_audit_runs():
    assert make().audit(n=256).ok


def test_without_row_or_entity_axis():
    source = pfn.Simulator(
        lambda rng, n: {"y": rng.normal(size=n) + 1.0, "mu": 1.0},
        sizes={"n": [8]},
        observed=["y"],
        targets=["mu"],
        kinds={"y": "row", "mu": "global"},
    )
    batch = source.sample(4)
    assert batch.row_mask.shape == (4, 8)
    assert batch.entity_mask.shape == (4, 1)


@pytest.mark.parametrize(
    ("simulate", "error", "match"),
    [
        (lambda rng, n_obs, n_groups: {"x": 0.0}, KeyError, "did not return"),
        (
            lambda rng, n_obs, n_groups: {
                **regression(rng, n_obs, n_groups),
                "beta": 1.0,
            },
            ValueError,
            "needs 1 axes",
        ),
        (
            lambda rng, n_obs, n_groups: {
                **regression(rng, n_obs, n_groups),
                "x": np.zeros((n_groups, n_obs + 1)),
            },
            ValueError,
            "disagree on the grid",
        ),
        (
            lambda rng, n_obs, n_groups: regression(
                rng, n_obs + int(rng.integers(2)), n_groups
            ),
            ValueError,
            "depend on the sizes only",
        ),
    ],
)
def test_bad_simulator_output(simulate, error, match):
    with pytest.raises(error, match=match):
        make(simulate).sample(16)


def test_kinds_are_checked():
    with pytest.raises(ValueError, match="kinds must give"):
        make(kinds={"x": "cell", "y": "row", "beta": "column"})


def test_fit_and_evaluate(tmp_path):
    source = make()
    net = pfn.fit(source, config(steps=150), out=tmp_path, verbose=False)
    assert net.spec.prior_fingerprint == source.fingerprint()
    report = pfn.evaluate(net, source, n_tasks=24, draws=16, sampler_tasks=8)
    accuracy = report.accuracy
    assert accuracy["network"]["beta"]["rmse"] < accuracy["prior_only"]["beta"]["rmse"]
    with pytest.raises(TypeError, match="NUTS reference needs a Prior"):
        pfn.evaluate(net, source, n_tasks=8, reference_tasks=2)


def test_simulator_does_not_import_pymc():
    code = (
        "import sys, numpy as np, pymc_pfn as pfn\n"
        "s = pfn.Simulator(lambda rng, n: {'y': rng.normal(size=n), 'mu': 0.0},"
        " sizes={'n': [4]}, observed=['y'], targets=['mu'],"
        " kinds={'y': 'row', 'mu': 'global'})\n"
        "s.sample(8)\n"
        "assert 'pymc' not in sys.modules and 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
