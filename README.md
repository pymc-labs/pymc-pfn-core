# pymc-pfn

Write the prior as a PyMC model. Get an amortized posterior network for that
prior, with the evidence that it works and where it stops working.

Status: milestone M3 of four. A network can be trained on a prior, saved,
used for posterior draws and evaluated. A network counts as usable only with
its evaluation report, including the flags.

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

report = pfn.evaluate(
    net,
    prior,
    baselines={"ols": my_ols},  # prior_only is always included
    stress={"heavy_tails": other_prior},  # task families outside the prior
    symmetries=[pfn.Symmetry("sign flip", flip_x, flip_beta)],
    reference_tasks=12,  # NUTS on the same PyMC model
)
print(report)  # tables plus report.flags
```

## What the examples show

Both examples train a small network (64 dimensions, 3 + 2 layers) on an
Apple M-series laptop and print the full report. Numbers are from one seed on
2026-11-14, with 200 fresh tasks and 64 draws per task.

**`examples/hierarchical_regression.py`**, 6,000 steps, about 16 minutes:

| `beta` | RMSE | CRPS | 80% coverage |
|---|---|---|---|
| network, in prior | 0.273 | 0.131 | 79.4% |
| OLS, in prior | 0.298 | 0.177 | |
| prior only, in prior | 1.049 | 0.530 | 77.2% |
| network, heavy-tailed noise | 0.383 | 0.189 | 71.5% |
| OLS, heavy-tailed noise | 0.608 | 0.330 | |
| network, coefficients five times larger | 2.064 | 0.754 | 61.8% |
| OLS, coefficients five times larger | 0.224 | 0.141 | |
| network, collinear inputs | 0.543 | 0.289 | 53.8% |
| OLS, collinear inputs | 0.392 | 0.252 | |

- In the prior the network beats OLS and `beta` passes SBC (p = 0.75).
- Outside the prior it fails, and the report says so: with larger
  coefficients it is nine times worse than OLS, with collinear inputs its 80%
  interval covers 54%.
- Against NUTS (7 of 12 fits passed the convergence screen) the posterior sd
  of `beta` is 1.5 times too wide at the median and CRPS is 0.206 against
  0.140. Task by task, the network is two to ten times too wide where the
  NUTS sd is below 0.1 and about right or too narrow where it is above 0.3.
  Average coverage looks calibrated only because the two errors cancel.
- `effect` is too wide in the prior (87.4% coverage, fails SBC).
- Euler with 8 steps shrinks the posterior sd by 13% against Heun with 40
  steps. Heun with 10 steps, the default, is within 5%.

**`examples/gp_regression.py`**, 3,000 steps, about 7 minutes, no NUTS
reference (minutes per task for the latent function):

| latent `f` | RMSE | CRPS | 80% coverage |
|---|---|---|---|
| network, in prior | 0.125 | 0.060 | 85.2% |
| exact GP with fixed hyperparameters, in prior | 0.133 | 0.068 | 68.4% |
| network, rough (Matern 1/2) functions | 0.341 | 0.197 | 51.9% |
| exact GP with fixed hyperparameters, rough | 0.389 | 0.261 | 22.4% |
| network, linear trend | 0.462 | 0.141 | 79.3% |
| exact GP with fixed hyperparameters, linear trend | 0.135 | 0.068 | 69.2% |

In the prior the network is slightly better than the fixed-hyperparameter GP
but its intervals for `f` are too wide. On rough functions its noise estimate
is badly overconfident (18.5% coverage), and on linear trends it loses to the
baseline by a factor of three.

Neither network is a finished model. The examples show that the report finds
the weak points: out-of-prior failure, solver shrinkage, and miscalibration
that average coverage hides.

## How it works

A task with R rows and E entities becomes a grid of (R + 1) x (E + 1) tokens.
Cell variables live in the R x E block, row variables in the extra column,
entity variables in the extra row and global variables in the corner. Each
block attends along rows, then along entities. An encoder reads the observed
variables once. A denoiser predicts the flow-matching velocity of all targets
jointly, so the draws are joint posterior samples.

The network handles any number of rows and entities and is equivariant to
their order. Set `ModelConfig(ordered_rows=True)` for time series.

Known limits:

- Variables are standardised with fixed statistics from the prior. Data on a
  very different scale than the prior is out of support. `evaluate` reports
  the input scale of every stress family, `predict` does not warn yet.
- All observed variables are fully visible. There is no forecasting split.
- Checkpoint selection uses the holdout flow loss, not a proper scoring rule.
- Evaluation is slow on a laptop: the default report (200 tasks, 64 draws,
  three stress families) takes several minutes on Apple silicon.
- The NUTS reference refits the model per task and is practical only for
  models NUTS handles in seconds.

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
| M3 | `evaluate`: baselines, calibration, sampler check, stress families, symmetries, NUTS reference | done |
| M4 | two-channel MMM specialist example, docs | open |

## Development

```bash
uv sync --all-extras
```

```bash
uv run pytest -q
```
