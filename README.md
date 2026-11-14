# pymc-pfn

Write the prior as a PyMC model. Get an amortized posterior network for that
prior, with the evidence that it works and where it stops working.

Status: milestone M1 of four. The prior interface, task generation and the
support audit exist. There is no network yet.

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
for batch in prior.stream(batch_size=256, prefetch=4):
    ...
```

`examples/hierarchical_regression.py` runs this end to end.

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
| M2 | cell-grid encoder, flow-matching head, `fit`, bundle | open |
| M3 | `evaluate`: baselines, calibration, stress families, invariance | open |
| M4 | two-channel MMM specialist example, docs | open |

## Development

```bash
uv sync --all-extras
```

```bash
uv run pytest -q
```
