# Sampled-Bellman algorithm and calibration

The shared environment, chronology and comparison rules are defined in
[Methodology](methodology.md). This page specifies the `sampled_q` learner.

## Empirical transition model

For each campaign, hour and action, replay **each training hour separately** at
every budget-grid node. Prefix sums handle the affordable prefix; cached
positive-price arrival indices and free-arrival suffix sums handle later
purchases after an unaffordable impression. Node outcomes follow the same
ordered affordability rule as evaluation.

The finite-horizon empirical MDP samples training days independently by hour.
Never-observed launch hours use observed training hours as a prior. Empty samples
within an otherwise observed hour remain empty.

## Bellman update

For sampled hour `d`, let `C_d(h,b,a)` be purchased cost and `R_d(h,b,a)` its
reward estimate. Backward fitted Q iteration computes:

```text
Q_h(b,a) = mean_d [R_d(h,b,a) + V_(h+1)(b - C_d(h,b,a))]
V_h(b)   = max_a Q_h(b,a)
V_24(b)  = 0;  V_h(0) = 0;  gamma = 1.
```

Budget and cost are normalized by training reference daily cost. Continuation
values are linearly interpolated. The backup precedes averaging, retaining
sampled cost/reward dependence. Every training-hour sample contributes; there
is no random optimization seed.

The multiplier and budget grids are `ACTIONS` and `GRID` in `bidpace/tabular.py`.
They are constructed exactly as follows (`geomspace` and `linspace` include both
endpoints; `unique` sorts and removes duplicates):

```python
import numpy as np

actions = np.unique(np.r_[
    0.0, np.geomspace(0.04, 3.0, 39),
    [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0],
])  # 48 multipliers
budget_grid = np.unique(np.r_[
    0.0, np.geomspace(1e-6, 0.002, 25),
    np.linspace(0.002, 1.0, 500), np.geomspace(1.0, 4.0, 60),
])  # 584 normalized budget nodes
```

Both arrays are stored in every exported sampled-Q policy.

## Reward-estimation candidates

| Setting | Training reward estimate |
| --- | --- |
| `ctr` | Purchased count × fitted hourly CTR |
| `shrunk` | Sum of fitted hourly/relative-price-band CTRs |
| `clicks` | Purchased recorded training binary clicks |

For `shrunk`, pooled price-band rates shrink toward campaign CTR; hourly bands
shrink toward pooled-band rates adjusted by hourly CTR. Each prior has strength
10,000 impressions. Evaluation always counts actual purchased recorded clicks.

## Traffic calibration

The lookup budget is remaining budget divided by the causal price estimate,
traffic estimate and reserve factor. Reserve candidates are .85, 1.00 and 1.15;
all candidates use market-filter rate .5.

Two traffic estimates are compared:

- **Public:** the cumulative traffic feature from `ReplayEnv`.
- **Model-consistent:** expected inventory matches the MDP's launch-hour prior.
  Completed-hour inventory is inferred from spending / average paid price / win
  fraction. Informative, uncensored feedback contributes to both observed and
  expected totals; a 5% inventory prior stabilizes their ratio.

The public feature uses raw profile inventory. During campaign launches its
denominator can disagree with the model's filled hours; the model-consistent
option addresses that mismatch using only completed-hour feedback.

Calibration selects among **18 candidates**: three reward estimators × three
reserve factors × two traffic estimators, using nominal CV. Saved Q-tables are
reused across calibration choices. Rescaled actions exceeding multiplier 3 are
excluded before maximizing Q.

## Approximation boundaries

Training transitions are exact at grid nodes. Interpolated values, reward
smoothing, unseen-hour priors and market rescaling are approximate. The empirical
MDP does not represent hidden cross-hour day correlations. Those limitations,
and the broader study limitations in Methodology, apply to the measured gains.
