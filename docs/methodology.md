# Methodology

For installation and runnable commands, see the [README](../README.md).
For artifact definitions, see the [output reference](outputs.md).

## Data and chronology

| Season | Development | Forward-validation dates | Test dates |
| --- | --- | --- | --- |
| 1 | March 11–17, 2013 | March 15, 16, 17 | March 18–20 |
| 2 | June 6–12, 2013 | June 10, 11, 12 | June 13–15 |
| 3 | October 19–24, 2013 | October 22, 23, 24 | October 25–28 |

`config.json` supplies shared training settings. The runner generates each
season's configuration using `configuration()` and `SEASONS` in
`bidpace/benchmark.py`, which substitute that season's data path and dates.
Season 3's public leaderboard supplies October 28 observations.

Each forward fold fits profiles and agents on dates preceding its validation
date. Labels are available only through the training cutoff. After checkpoint
selection, each seasonal agent is refit on its full development period. Seasons
have different campaigns and market conditions; their training observations are
kept separate.

Preparation keeps price/click pairs in arrival order, deduplicates impressions,
quarantines conflicting prices, and attaches clicks to preceding impressions.
Season 3 leaderboard records overlapping development are excluded. Unknown
test advertisers without training profiles are omitted and audited. The nominal
test set contains 9, 15 and 10 observed advertiser-days respectively.

The data cover **logged winning inventory**. Prices of lost auctions and their
counterfactual click labels are unobserved. Public test dates were available
during development, so results are retrospective rather than a blind test.

## Environment and reward

An episode is one advertiser-day with up to 24 hourly decisions. The nominal
budget is 30% of the campaign's mean active-day training cost.

The 13 public observation features describe remaining budget, time, previous-hour
spending/win/paid-price feedback, fitted hourly CTR and volume, initial budget
fraction, causal traffic/opportunity estimates, previous action and campaign
reward scale. Campaign identity is also known. Agents never inspect future-hour
inventory or the day's hidden price multiplier.

A quality bid is `alpha × hourly_CTR / mean_CTR × mean_price`, with all profile
statistics fitted on training data. Qualifying impressions pay their logged
clearing price. An unaffordable impression is skipped; later cheaper impressions
remain eligible. Spending is capped by the environment. Actual reward is the
number of purchased recorded binary clicks, with `gamma = 1`.

## RL candidates

| Candidate | Algorithm | Representation |
| --- | --- | --- |
| `traffic_dqn` | Double DQN with three-step returns | 13 causal features |
| `campaign_dqn` | Same DQN algorithm | Traffic features plus a training-fitted campaign one-hot code |
| `sampled_q` | Sampled Bellman model-based RL | Campaign/hour/budget tables and causal traffic/price calibration |

Traffic and Campaign DQN share a learning algorithm; the comparison measures the
effect of adding campaign context. Both use 48/32 ReLU hidden layers, Huber loss, Adam,
replay, a lagged target network and epsilon-greedy exploration. Their discrete
multipliers are `{0, .1, .2, .35, .5, .7, 1, 1.4, 2, 3}`.

Default DQN training uses seeds 11, 23, 37, 53 and 71, up to 6,000 episodes, and
checkpoints 0/1,500/3,000/6,000. All seed Q-values are averaged before choosing an
action; no seed is selected by test performance. Budget, price and coverage are
randomized over the ranges in `config.json` during neural training.

Sampled-Bellman RL is deterministic. It uses a finer multiplier grid, exact
ordered training-hour transitions at budget-grid nodes, and interpolated Bellman
values. See [the tabular derivation](tabular_method.md). Its different action
resolution and representation contribute to the measured differences alongside
the learning algorithm.

## Baselines

Four baselines are highlighted in the main plot:

| Baseline | Purpose |
| --- | --- |
| Constant | Fixed-price bidding without hourly quality adjustment |
| Linear | Fixed multiplier on fitted hourly quality |
| PID | Adaptive pacing from cumulative spending error |
| MPC | Training cost curves with causal market/traffic feedback |

Full tables retain Campaign Linear, Budget Linear, Quality PID, Market Linear,
Traffic Linear and unscaled bidding. Parameters are tuned on nominal forward
validation; Budget Linear's anchors use their matching unstressed validation
budget conditions. Privileged-price bidding is a diagnostic, excluded from
reference selection and superiority claims.

## Training and selection

1. **Internal neural checkpoint selection:** use the forward-CV
   criterion, mean paired gain minus 0.5 times its across-date standard deviation,
   over the configured validation conditions.
2. **Internal sampled-Q selection:** nominal CV chooses among three reward
   estimators, three reserve factors and two traffic estimators.
3. **Seasonal algorithm selection:** replay nominal CV for all three resulting
   candidates. Choose the highest mean clicks, with equal weight per date. The
   reference is the highest-scoring deployable baseline on the same CV dates.
4. **Shared algorithm selection:** divide each candidate's CV score by its
   season's reference score, average across seasons, and select the highest ratio.
   Each seasonal policy remains fitted to its own development dates. The shared
   choice is a retrospective summary using all seasons' validation results.

Ties between RL candidates follow their declared order; baseline ties follow
the sorted baseline names. Choices are frozen on validation before test replay,
and every candidate is reported. Internal search objectives and compute budgets
differ between the candidate pipelines.

## Evaluation

Evaluate nominal replay and six declared transformations: prices ×1.5, evening
inventory ×3, budget fractions .15/.60, click retention 50% and coverage 50%.
Repeated/thinned inventory is synthetic stress, not additional independent data.

Metrics include mean/total clicks, spending, utilization and violations. Paired
differences compare each RL with every deployable baseline. Confidence intervals
use 5,000 bootstrap draws of whole dates, retaining campaigns on a date together.
With only three or four test dates per season, these intervals are descriptive
and unadjusted for exploratory analysis and multiple comparisons.

## Implementation map

| File | Responsibility |
| --- | --- |
| `experiments/compare.py` | Policy comparison, selection, verification and reports |
| `experiments/calibrated_tabular.py` | Model-consistent traffic calibration and its selection |
| `bidpace/tabular.py`, `tabular_study.py` | Sampled transition learning and baseline tuning |
| `bidpace/agents.py`, `control.py`, `protocol.py` | DQN networks, observation context and neural training |
| `bidpace/replay.py`, `fetch.py` | Ordered simulator, chronological data preparation and downloads |
| `bidpace/policies.py`, `adaptive.py` | Fixed/pacing baselines and causal market estimation |
| `bidpace/benchmark.py` | Seasonal date definitions and data-preparation orchestration |
| `build_release.py` | Checked, deduplicated release packaging |

## References

- [iPinYou dataset specification](https://contest.ipinyou.com/ipinyou-dataset.pdf)
- [iPinYou benchmark](https://arxiv.org/abs/1407.7073)
- [Double DQN](https://arxiv.org/abs/1509.06461)
