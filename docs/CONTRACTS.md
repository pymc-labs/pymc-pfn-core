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
- A fresh `Prior` with the same seed returns the same batch.

## 2. Prior

- Every axis of an observed or target variable is named by `row_dim` or
  `entity_dim`. Other dims are rejected with an error.
- `prefix_rows` is a promise by the user that the first k rows of a draw are a
  valid draw with k rows. The library does not check it.
- `reject` changes the effective prior. `Prior.n_drawn` and `Prior.n_rejected`
  count it, and the audit reports the rate.

## 3. Audit

`Prior.audit()` returns an `AuditReport`. It is an error when a target is
constant or any declared variable contains NaN or infinite values. It is a
warning when a binary target is rarer than `min_prevalence`, when an allowed
size is never drawn, or when `reject` discards more than `max_rejection_rate`.

## 4. Imports

`import pymc_pfn` imports numpy only. PyMC is imported when a model is first
built.
