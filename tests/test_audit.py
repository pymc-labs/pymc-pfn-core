import pymc as pm
import pytensor.tensor as pt
import pytest

import pymc_pfn as pfn


def build(target):
    def model(n_obs, n_groups):
        with pm.Model(coords={"obs": range(n_obs), "group": range(n_groups)}) as m:
            x = pm.Normal("x", dims=("obs", "group"))
            target(x)
        return m

    return pfn.Prior(
        model,
        sizes={"n_obs": [8], "n_groups": [2, 3, 4]},
        observed=["x"],
        targets=["t"],
        row_dim="obs",
        entity_dim="group",
    )


def test_healthy_prior_passes():
    report = build(lambda x: pm.Normal("t", dims="group")).audit(n=200)
    assert report.ok and not report.warnings
    assert report.size_counts["n_groups"].keys() == {2, 3, 4}
    assert sum(report.size_counts["n_groups"].values()) == 200
    assert "OK" in str(report)


def test_constant_target_is_an_error():
    report = build(
        lambda x: pm.Deterministic("t", pt.zeros(x.shape[1]), dims="group")
    ).audit(n=64)
    assert not report.ok and "nothing to learn" in report.errors[0]
    with pytest.raises(pfn.PriorAuditError):
        report.raise_if_failed()


def test_rare_binary_target_is_flagged():
    report = build(lambda x: pm.Bernoulli("t", 0.001, dims="group")).audit(n=500)
    stats = next(v for v in report.variables if v.name == "t")
    assert stats.binary
    assert any("prevalence" in w for w in report.warnings) or not report.ok


def test_nonfinite_values_are_an_error():
    report = build(lambda x: pm.Deterministic("t", pt.log(x[0]), dims="group")).audit(
        n=64
    )
    assert any("NaN" in e for e in report.errors)


def test_unseen_sizes_and_heavy_rejection_are_flagged():
    prior = build(lambda x: pm.Normal("t", dims="group"))
    prior.reject = lambda task: abs(task["t"]).max() < 2.5
    prior.max_reject_factor = 1000
    report = prior.audit(n=16)
    assert report.rejection_rate > 0.9
    assert any("discards" in w for w in report.warnings)
    assert any("never drawn" in w for w in report.warnings)
