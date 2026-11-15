"""Shared model, prior and a tiny trained network."""

import pymc as pm
import pytest

import pymc_pfn as pfn

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
