# pymc-pfn

Train an amortized posterior network on synthetic tasks, with the evidence
that it works and where it stops working. The tasks come from any simulator.
A PyMC model is the preferred one: the library reads the variable layout
from the model and checks the network against NUTS on the same model.

Status: milestone M3 of four. A network can be trained on a task source, saved,
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

### Any simulator

A task source without PyMC is a function that returns one task. The kinds
are declared by hand, everything else works the same.

```python
def simulate(rng, n_obs, n_groups):
    x = rng.normal(size=(n_obs, n_groups))
    beta = rng.normal(size=n_groups)
    y = x @ beta + rng.normal(size=n_obs)
    return {"x": x, "y": y, "beta": beta}


source = pfn.Simulator(
    simulate,
    sizes={"n_obs": [32, 64, 128], "n_groups": range(2, 9)},
    observed=["x", "y"],
    targets=["beta"],
    kinds={"x": "cell", "y": "row", "beta": "entity"},
)
net = pfn.fit(source, pfn.FitConfig(steps=20_000), out="runs/demo")
```

### Per-task scaling

```python
config = pfn.FitConfig(units={"x": "x", "y": "y", "beta": "y/x"})
```

Every task then enters the network in its own units: observed variables
are standardised per task, cell variables per entity, and targets are
divided by the declared scale. Draws come back in natural units. By default
the network still sees the scales as inputs. `scale_inputs=False` drops
them and makes the network exactly equivariant to rescaling. Details are in
[docs/CONTRACTS.md](docs/CONTRACTS.md), section 6.

What the PyMC route adds: kinds and axis order come from the model dims,
`prefix_rows` saves compilations, and `evaluate(reference_tasks=...)` runs
NUTS on the same model. A `Simulator` has no likelihood, so its report has
no NUTS reference. SBC, coverage, baselines, stress families and symmetries
work for both.

## What the examples show

The examples train a small network (64 dimensions, 3 + 2 layers) on an
Apple M-series laptop and print the full report. Numbers are from one seed on
2026-11-14, with 200 fresh tasks and 64 draws per task.

**`examples/hierarchical_regression.py`**, 6,000 steps, about 16 minutes:

| `beta` | RMSE | CRPS | 80% coverage |
|---|---|---|---|
| network, in prior | 0.289 | 0.133 | 81.0% |
| OLS, in prior | 0.284 | 0.166 | |
| prior only, in prior | 0.989 | 0.507 | 80.6% |
| network, heavy-tailed noise | 0.383 | 0.189 | 73.5% |
| OLS, heavy-tailed noise | 0.608 | 0.330 | |
| network, coefficients five times larger | 2.064 | 0.754 | 63.6% |
| OLS, coefficients five times larger | 0.224 | 0.141 | |
| network, collinear inputs | 0.543 | 0.289 | 55.8% |
| OLS, collinear inputs | 0.392 | 0.252 | |

- In the prior the posterior mean is on par with OLS. On another sample of
  200 tasks the network was ahead (0.273 against 0.298), so the difference
  is within task-sampling noise. `beta` passes SBC (p = 0.15).
- Average coverage hides a conditional error. Split by the network's own
  posterior width, the 80% interval covers 89%, 84% and 73% for the
  narrowest, middle and widest third.
- Against NUTS on 40 fixed tasks (26 passed the convergence screen), the
  posterior sd is 2.1 times too wide for narrow posteriors and 1.07 times
  for wide ones. Training twice as long reduces 2.1 to 1.8. See
  [experiments/conditional_calibration.md](experiments/conditional_calibration.md).
- Outside the prior it fails, and the report says so: with larger
  coefficients it is nine times worse than OLS, with collinear inputs its 80%
  interval covers 56%.
- `effect` is too wide in the prior (87.4% coverage, fails SBC).
- Euler with 8 steps shrinks the posterior sd by 13% against Heun with 40
  steps. Heun with 10 steps, the default, is within 5%.

**`examples/gp_regression.py`**, 3,000 steps, about 7 minutes, no NUTS
reference (minutes per task for the latent function):

| latent `f` | RMSE | CRPS | 80% coverage |
|---|---|---|---|
| network, in prior | 0.125 | 0.060 | 87.3% |
| exact GP with fixed hyperparameters, in prior | 0.133 | 0.068 | 70.1% |
| network, rough (Matern 1/2) functions | 0.341 | 0.197 | 54.3% |
| exact GP with fixed hyperparameters, rough | 0.389 | 0.261 | 23.7% |
| network, linear trend | 0.462 | 0.141 | 80.8% |
| exact GP with fixed hyperparameters, linear trend | 0.135 | 0.068 | 71.2% |

In the prior the network is slightly better than the fixed-hyperparameter GP
but its intervals for `f` are too wide, 95.7% for the narrowest third. On
rough functions its noise estimate is badly overconfident (20% coverage), and
on linear trends it loses to the baseline by a factor of three.

**`examples/ricker_simulator.py`**, 3,000 steps, about 20 minutes, a
`Simulator` without PyMC and without a NUTS reference (2026-11-15):

| target | RMSE network | RMSE prior only | 80% coverage | SBC p |
|---|---|---|---|---|
| `log_r` | 0.175 | 0.864 | 82.0% | 0.34 |
| `sigma` | 0.120 | 0.231 | 85.0% | 0.08 |
| `phi` | 0.739 | 4.772 | 84.5% | 0.96 |

The network learns the growth rate and the observation scale well and the
process noise only partly. No target fails SBC. Narrow posteriors of `sigma`
and `phi` are too wide (88% and 91% coverage), the same pattern as in the
regression example. There is no baseline besides the prior and no reference
posterior, so this shows that the route works, not how close it is to the
true posterior.

No network here is a finished model. The examples show that the report finds
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

- Without `units`, variables are standardised with fixed statistics from the
  prior, and data on a very different scale is out of support. The examples
  below were run without `units`. `evaluate` reports the input scale of
  every stress family, `predict` does not warn yet.
- All observed variables are fully visible. There is no forecasting split.
- Checkpoint selection uses the holdout flow loss, not a proper scoring rule.
- Narrow posteriors come out too wide, see the experiment above. Per-task
  scaling reduces the excess from 2.1 to 1.3 times the NUTS sd in that
  experiment (one seed) and does not remove it. A wider network is untested.
- Task draws are reproducible only on a fresh `Prior`. Evaluate with a new
  `Prior` object, not the one used for training, to get the same tasks again.
  A `Simulator` does not have this limit.
- A `Simulator` draws one task per Python call. Vectorise inside the function
  if task generation limits training speed.
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
4. Importing `pymc_pfn` must not import PyMC. PyMC is needed for `Prior`
   and the NUTS reference, not for a `Simulator` or to load or use a network.
5. Every loss and metric takes the masks of `TaskBatch`. Padded values are
   never data.

The data contract is in [docs/CONTRACTS.md](docs/CONTRACTS.md).

## Roadmap

| Step | Deliverable | State |
|---|---|---|
| M1 | `Prior`, `Simulator`, task generation, audit | done |
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
