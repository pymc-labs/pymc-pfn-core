# Conditional calibration of the hierarchical-regression network

Date: 2026-11-14. One seed per variant, Apple M-series laptop.

## Question

The M3 report showed a posterior sd for `beta` of 1.5 times the NUTS sd at the
median. A task-by-task check showed the network two to ten times too wide
where the NUTS sd is below 0.1 and about right where it is above 0.3. Is
this a training-budget problem?

## Setup

`experiments/conditional_calibration.py`. The prior is the hierarchical
regression of example 1 with `beta` as the only target. All variants use 64
dimensions and 3 + 2 layers. NUTS was run once on 40 fixed tasks, 4 chains,
1,000 draws. 26 fits passed the convergence screen (no divergences, split
R-hat below 1.01), which favours well-behaved posteriors. "Narrow" and
"wide" split the converged coefficients at the median NUTS sd. Network
posteriors use 256 draws.

## Result

| Variant | sd ratio, narrow | sd ratio, wide | mean shift in NUTS sd | CRPS | 80% coverage by width |
|---|---|---|---|---|---|
| NUTS | 1 | 1 | 0 | 0.0554 | |
| 6,000 steps | 2.08 | 1.07 | 0.38 | 0.0636 | 86.9 / 80.6 / 79.1 |
| 6,000 steps, averaged weights (decay 0.999) | 1.99 | 1.07 | 0.33 | 0.0627 | 89.7 / 81.9 / 80.0 |
| 12,000 steps | 1.76 | 1.06 | 0.26 | 0.0593 | 90.3 / 79.4 / 80.0 |

The sd ratio is the median of network sd over NUTS sd. Coverage by width is
measured on 200 fresh tasks without NUTS, for the narrowest, middle and
widest third of the network's own posteriors.

## Reading

- Wide posteriors are fine in every variant. The error is confined to
  narrow posteriors, where the network cannot locate the coefficient as
  precisely as the data allow and reports that as extra width.
- Doubling the training length moves the narrow ratio from 2.08 to 1.76 and
  closes about half of the CRPS gap to NUTS. The holdout loss was still
  falling slowly (0.370 to 0.361). More training helps, slowly.
- Averaged weights changed little. With one seed the difference is not
  distinguishable from noise, so the option was not kept in the package.
- Coverage by width detects the problem without NUTS: the narrowest third
  covers 87% to 90% instead of 80%. It is now part of every report.

## Not tested

A wider network, per-task scaling of inputs and targets, a loss weighted
toward narrow posteriors, more than one seed. The first two are the next
candidates.
