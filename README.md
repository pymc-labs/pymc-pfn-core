# pymc-pfn

Write the prior as a PyMC model. Get an amortized posterior network for that
prior, with the evidence that it works and where it stops working.

Status: milestone M2 of four. A network can be trained on a prior, saved and
used for posterior draws. The evaluation harness (baselines, calibration,
stress tests) is milestone M3, so no trained network should be trusted yet.

## What works today

```python
import pymc as pm
import pymc_pfn as pfn


def model(n_obs: int, n_groups: int) -> pm.Model:
    with pm.Model(coords={"obs": range(n_obs), "group": range(n_groups)}) as m:
        x = pm.Normal("x", dims=("obs", "group"))
        beta = pm.Normal("beta", dims="group")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", (x * beta).sum(axis=-1), sigma, dims="obs")
    return m


prior = pfn.Prior(
    model,
    sizes={"n_obs": [32, 64, 128], "n_groups": range(2, 9)},
    observed=["x", "y"],  # what the network will see
    targets=["beta"],  # what it must return draws of
    row_dim="obs",
    entity_dim="group",
    prefix_rows="n_obs",
)

print(prior.audit(n=2000))  # is there anything to learn?
batch = prior.sample(256, seed=0)  # padded, masked TaskBatch

net = pfn.fit(prior, pfn.FitConfig(steps=20_000), out="runs/demo")
net = pfn.PFN.load("runs/demo/bundle")

post = net.predict({"x": x_obs, "y": y_obs}, draws=1000)
post["beta"]  # (draws, n_groups), joint draws across all targets
post.to_datatree()  # ArviZ-compatible
```

`examples/hierarchical_regression.py` runs this end to end.

A 3,000-step run of that example (about nine minutes on an Apple M-series
laptop, 2026-11-14) gave on 100 fresh tasks: RMSE of the posterior mean of
`beta` 0.231, OLS 0.219, prior mean 1.090, and 84.1% coverage of the 80%
interval. This is one seed and a sanity check, not a benchmark: the network
is close to OLS and not better, and the intervals are slightly too wide.

## How it works

A task with R rows and E entities becomes a grid of (R + 1) x (E + 1) tokens.
Cell variables live in the R x E block, row variables in the extra column,
entity variables in the extra row and global variables in the corner. Each
block attends along rows, then along entities. An encoder reads the observed
variables once. A denoiser predicts the flow-matching velocity of all targets
jointly, so the draws are joint posterior samples.

The network handles any number of rows and entities and is equivariant to
their order. Set `ModelConfig(ordered_rows=True)` for time series.

Known limits of M2:

- Variables are standardised with fixed statistics from the prior. Data on a
  very different scale than the prior is out of support, and nothing warns
  about it yet.
- All observed variables are fully visible. There is no forecasting split.
- Checkpoint selection uses the holdout flow loss, not a proper scoring rule.

## Rules of this repository

1. One path through the code: one encoder, one head, one trainer. A second
   variant enters the package only when a benchmark in this repository shows
   that it wins.
2. No domain features in the package. Marketing, finance and other domains
   are consumers and live in their own repositories or in `examples/`.
3. Experiments live outside the package, and no script imports another script.
4. Importing `pymc_pfn` must not import PyMC. PyMC is needed to generate
   tasks, not to load or use a network.
5. Every loss and metric takes the masks of `TaskBatch`. Padded values are
   never data.

The data contract is in [docs/CONTRACTS.md](docs/CONTRACTS.md).

## Roadmap

| Step | Deliverable | State |
|---|---|---|
| M1 | `Prior`, task generation, audit | done |
| M2 | cell-grid encoder, flow-matching head, `fit`, bundle | done |
| M3 | `evaluate`: baselines, calibration, stress families, invariance | open |
| M4 | two-channel MMM specialist example, docs | open |

## Development

```bash
uv sync --all-extras
```

```bash
uv run pytest -q
```
