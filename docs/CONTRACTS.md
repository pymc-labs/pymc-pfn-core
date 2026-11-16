# Contracts

These are the interfaces other modules may rely on. Tests enforce them.
A change here is a breaking change.

## 1. TaskBatch

A batch of B tasks padded to R rows and E entities.

| Field | Type | Meaning |
|---|---|---|
| `observed` | `dict[str, float32 array]` | what the network sees |
| `targets` | `dict[str, float32 array]` | what the network must return draws of |
| `row_mask` | `bool (B, R)` | True for real rows |
| `entity_mask` | `bool (B, E)` | True for real entities |
| `kinds` | `dict[str, str]` | variable name to `cell`, `row`, `entity` or `global` |
| `sizes` | `list[dict[str, int]]` | builder arguments of each task |

Array shape by kind: `cell` (B, R, E), `row` (B, R), `entity` (B, E),
`global` (B,). Axis order is always rows before entities, whatever order the
model declared.

Invariants:

- Padded values are zero and carry no information. Use `batch.mask(name)`.
- All variables of one task come from one joint draw of the model.
- A fresh `Prior` with the same seed returns the same batch. A `Simulator`
  returns the same batch for the same seed on a used object too.

## 2. Task sources

`TaskSource` is the interface `fit`, `evaluate` and the audit rely on:
`sample(n, seed)`, `stream(batch_size, seed)`, `fingerprint()`, `sizes`,
`observed`, `targets`, `n_drawn` and `n_rejected`. A subclass implements
`_draw(n, sizes, rng)` and `_identity()`.

`Simulator`:

- `simulate(rng, **sizes)` returns one task as a dict of arrays, cell
  variables in (rows, entities) order. All randomness comes from `rng`.
- `kinds` names the kind of every observed and target variable. Shapes are
  checked against it, and they may depend on the sizes only.
- There is no NUTS reference. `evaluate(reference_tasks > 0)` raises.

`Prior`, the PyMC adapter:

- Every axis of an observed or target variable is named by `row_dim` or
  `entity_dim`. Other dims are rejected with an error.
- `prefix_rows` is a promise by the user that the first k rows of a draw are a
  valid draw with k rows. The library does not check it.
- `reject` changes the effective prior for every source. `n_drawn` and
  `n_rejected` count it, and the audit reports the rate.

## 3. Audit

`audit()` of a task source returns an `AuditReport`. It is an error when a target is
constant or any declared variable contains NaN or infinite values. It is a
warning when a binary target is rarer than `min_prevalence`, when an allowed
size is never drawn, or when `reject` discards more than `max_rejection_rate`.

## 4. Token grid

A task with R rows and E entities is laid out on (R + 1) x (E + 1) tokens:

| Tokens | Variables |
|---|---|
| `[:R, :E]` | `cell` |
| `[:R, E]` | `row` |
| `[R, :E]` | `entity` |
| `[R, E]` | `global` |

Each variable has its own channel. Invariants, all tested:

- Padding rows or entities, with any content, does not change the outputs at
  real positions.
- Permuting entities permutes the outputs. The same holds for rows unless
  `ordered_rows=True`.
- Targets are generated jointly. Draw i of one target belongs to draw i of
  every other target.

## 5. Bundle

A directory with `model.safetensors` and `spec.json`. The spec holds variable
names, kinds, normalisation statistics, the model configuration, a
fingerprint of the prior and `format_version`. Loading a bundle with another
format version is an error.

Per-task scaling is part of the spec: `units`, `scale_inputs` and `inputs`
(the observed variables plus the derived scale inputs). A bundle without
these fields loads as unscaled.

## 6. Per-task scaling

`FitConfig(units={"x": "x", "y": "y", "beta": "y/x"})` puts every task into
its own units before the network sees it.

- A unit is an observed variable or a ratio of two. Its scale is the
  standard deviation over the real values of one task: one per task for row
  and entity variables, one per entity for cell variables.
- An observed variable whose unit is itself is centred and scaled. Any other
  variable with a unit is divided by the scale and not shifted. A target
  that moves when an observed variable is shifted, such as an intercept,
  must not be given that unit.
- Draws are returned in natural units. `PFN.sample` and `predict` take and
  return the same arrays as without units.
- With `scale_inputs=True`, the default, the network also receives
  `log_scale[r]` and `center[r]` and loses no information. With
  `scale_inputs=False` it is exactly equivariant to the declared rescalings
  and blind to the absolute scale. That is correct only for a prior without
  a preferred scale in these units.
- Declaring units is a claim about the model. The library checks kinds and
  names, not that the claim is true. `evaluate` shows whether it helped.

## 7. Sampling

`PFN.sample(batch)` returns one array per target with shape `(B, draws, ...)`
in natural units, zero at padded positions. Noise is generated on the CPU from
`seed`, so the same seed gives the same draws on every device up to
floating-point differences.

## 8. Evaluation

`evaluate(net, prior)` returns a `Report` and can write it as JSON. It draws
fresh tasks with a seed stream that training never uses.

| Section | Content |
|---|---|
| `accuracy` | RMSE of the posterior mean, CRPS and interval coverage per target and method |
| `sbc` | rank histogram and chi-square p-value, one random real element per task |
| `conditional` | 80% coverage within thirds of the network's own posterior sd |
| `sampler` | coverage and posterior sd for the chosen solver setting and a grid |
| `stress` | accuracy on other priors, plus how far their inputs are from the training scale |
| `symmetry` | shift of the posterior under transformations the user declares |
| `reference` | NUTS on the same PyMC model: mean shift, sd ratio, CRPS of both |
| `flags` | every threshold that was crossed, as text |

Rules:

- `prior_only` is always among the methods. It draws from the prior marginal
  of each target and ignores the data.
- Baselines receive the batch without targets. They return point estimates
  `(B, ...)` or draws `(B, D, ...)` for any subset of the targets. For a
  point estimate, CRPS equals the mean absolute error and coverage is empty.
- All metrics are means over real values. Padding never enters.
- Intervals use Weibull quantile positions, so a perfect sampler has the
  nominal expected coverage at any number of draws.
- Coverage by width takes the sd from one half of the draws and the interval
  from the other half. Average coverage can hide posteriors that are too wide
  where they should be narrow and too narrow where they should be wide.
- A NUTS fit counts only if it has no divergences and split R-hat below 1.01
  for every target value. Excluded fits are counted in the report.
- Scale and shift invariance are not asserted. A proper prior is not
  invariant to them, and the network standardises with fixed statistics. A
  symmetry is checked only when the user states it with `Symmetry`.
- A report without flags is not a proof. It says that the checks which were
  run did not fail.

## 9. Imports

`import pymc_pfn` imports numpy only. PyMC is imported when a model is first
built, torch when `fit`, `PFN`, `Posterior` or `ModelConfig` is first used.
`evaluate` needs
torch, and PyMC only for the NUTS reference and for drawing tasks from a
`Prior`. A `Simulator` never imports PyMC.
