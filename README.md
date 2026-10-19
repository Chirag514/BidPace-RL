# BidPace-RL

**Course:** Artificial Intelligence (CSL3090), IIT Jodhpur  
**Team:** Chirag Bhutra (B23MT1012) and Akshay Gupta (B23BB1006)  
**Repository:** https://github.com/Chirag514/BidPace-RL

An agent chooses one bid multiplier per hour to maximize recorded advertising
clicks under a hard daily budget. The project implements Double DQN, two PPO
controllers and empirical model-based RL in NumPy, with chronological replay of
iPinYou auction logs. The solution is RL; heuristic policies serve as comparators.

The report and small evaluation files document a completed run. Raw data and full
training caches stay local. Export the trained agents from the completed run as
shown below before submitting the repository.

## Installation and execution

Use Python 3.11 or 3.12. Open a terminal in the directory containing `run.py`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux/macOS uses `source .venv/bin/activate`. If PowerShell blocks activation,
use `.\.venv\Scripts\python.exe` instead of `python` without activating.

Prepare the data using the commands in the next section, then run the complete
experiment into a fresh output directory:

```bash
python run.py all --out reproduction
python run.py verify --out reproduction
```

This trains candidates, selects them on development data, evaluates the fixed
held-out period and stress scenarios, generates graphs, and verifies artifacts.
Four CPU workers are configured; reduce `workers` in `config.json` before
starting if necessary. A GPU is not used.

A completed output directory is preserved. Use a new name for another run.
For an interrupted run with unchanged code, configuration and input:

```bash
python run.py all --out reproduction --resume
```

Completed validation folds are reused; incomplete folds or refits are recomputed.
Optimizer-state checkpoints are not provided. The published metric subset in
`results/` is evidence only and cannot replace a complete local run for `verify`.
If `results/` contains your original full run, keep it intact and use that run's
path for verification and export.

## Data preparation

No separately distributed data ZIP is required by this repository. Download and
prepare the public provider logs:

```bash
python -m bidpace.fetch --season 2 --out data/raw2
python -m bidpace.replay --raw data/raw2 --out data/season2 --workers 4
python -m bidpace.fetch --season 3 --out data/raw3
python -m bidpace.replay --raw data/raw3 --out data/season3 --workers 4
python -m bidpace.replay --combine data/season2/replay.npz data/season3/replay.npz --exclude-dates 20130613 20130614 20130615 --out data/combined
```

The required input is `data/combined/replay.npz`; `preparation.json` records
provenance and checksum. An existing prepared input may be placed at this path.
The earlier hourly and price-histogram CSVs are not used.

Use fresh preparation directories. The downloader checks provider MD5s and
reuses completed downloads. It selects 21 bid/impression/click files for Season 2
and 27 for Season 3, plus documentation/checksums. Conversion and leaderboard
files are excluded. Allow roughly 15 GB free for logs and temporary SQLite joins.
Successful preparation removes its join database; incomplete `.part` downloads
are ignored.

The prepared input contains 15,337,988 usable priced impressions: Season 2
June 6-12 and selected Season 3 October 19-27, 2013. This is a selected part of
iPinYou, not the whole three-season collection. June 13-15 and the Season 3
leaderboard are excluded. Nine campaigns have training profiles. Held-out replay
contains five observed advertiser-days across three dates and campaigns 2261/2997;
missing campaign-days are not synthesized. Provider-private test data are unavailable.

Keys include date, advertiser and bid ID. Same-price repeats keep their earliest
timestamp; conflicting-price keys are excluded and audited. Clicks attach to the
latest preceding matching impression, with dated responses masked at each fitting
cutoff. Within-hour order follows timestamp, then bid ID. Prepared data exclude
raw user identifiers and bid IDs.

Lost-auction prices and counterfactual responses are unavailable. This is
**conditional replay of logged winning inventory**, not unbiased whole-market
performance. Provider terms apply; permission to redistribute the processed data
has not been established. Keep raw and processed data out of the public repository.

## Trained agents

From a **complete local run**, verify and export both standalone ensembles. For
an existing run in `results/`, use PowerShell:

```powershell
python run.py verify --out results
New-Item -ItemType Directory -Force agents
python run.py export --out results --agent agents/primary_agent.npz
python run.py export --out results --variant empirical_q --agent agents/empirical_agent.npz
python run.py demo --agent agents/primary_agent.npz
python run.py demo --agent agents/empirical_agent.npz
```

For a new run in `reproduction/`, replace `--out results` with `--out reproduction`.
Linux/macOS uses `mkdir -p agents` instead of `New-Item`. Use new export filenames
if those files already exist. Export refuses to overwrite an existing agent.

`primary_agent.npz` follows the frozen validation selection; the documented run
selects Traffic DQN. `empirical_agent.npz` is the secondary model-based candidate.
Its export does not promote it to primary. Both exports are allowlisted for Git.
Full `models.npz`, `cv_models.npz` and `empirical_models.npz` stay local.

Standalone demos require the prepared input for historical calibration and
checksum verification, but no results directory. Exporting and demonstrating an
existing agent does not require retraining. This release does not substitute
checkpoints from another run for the authors' local exports.

## Environment and learning

Each episode is an advertiser-day with at most 24 hourly decisions. Bids win when
they cover the recorded price and remaining budget can afford the impression.
Winners pay the recorded price. Unaffordable arrivals are skipped; cheaper later
arrivals remain eligible. Zero bids buy nothing. Episodes end at day end or budget
exhaustion.

- **State:** 13 normalized observations: budget, time, completed-hour spend/win
  feedback, training quality/traffic, initial budget intensity, paid price, causal
  traffic estimates, budget per forecast opportunity, previous bid and expected-click scale.
- **Action:** multiplier in [0, 3], scaling training-fitted hourly CTR. DQN uses ten
  discrete actions; PPO uses continuous changes.
- **Reward:** recorded clicks from affordable winning impressions.
- **Budget:** fraction of mean campaign training cost over active days; nominal fraction 0.30.

Observations expose only training calibration and completed-hour feedback. They
exclude future traffic/prices, test CTRs and hidden stress multipliers. Hourly CTR
is a smoothed empirical count; no supervised CTR predictor is trained. Synthetic
fixtures are used for tests, not as historical evaluation data.

| Method | Implementation |
|---|---|
| Traffic DQN | 13 inputs, 48/32 ReLU network, Double-DQN targets, three-step replay, Huber loss, Adam, target network |
| Traffic PPO | 44 inputs with causal history; Gaussian latent policy and relative bid changes |
| Residual PPO | 44 inputs with a causal cost-controller proposal; bounded multiplicative correction |
| Empirical model-based RL | Training-estimated reward/cost models; backward Bellman Q-value iteration on 401 budget states |

Neural policies share parameters across campaigns and omit advertiser IDs. They
sample a campaign uniformly, then an active training day, with a training-only
reward scale. Exploration, curriculum and coverage have separate seeded streams.
Seeds are 11, 23, 37, 53 and 71. Ensembles average Q-values or latent actor means
before choosing an action; ensemble scores are not mean member scores.

PPO uses clipped ratios, KL stopping and full-episode GAE with `gamma=lambda=1`.
Residual PPO executes `clip(base * exp(0.5 * tanh(latent)), 0, 3)` and feeds the
executed bid back to its market filter. A zero actor starts at the cost controller.
Neural training randomizes budget [0.18, 0.50], price scale [0.85, 1.25] and retained
coverage [0.35, 1.00]. Coverage preserves price/response pairs and order. These are
augmentations, not additional historical data.

The empirical learner bootstraps active training days per campaign and estimates
hourly cost/reward for each action. Actions combine the DQN grid and 25 logarithmic
multipliers. Validation selects a reward mixture from weights 0, 0.5 and 1, combining
recorded clicks and smoothed CTR times eligible inventory. Five bootstrap tables
form a mean-Q ensemble. Backward Bellman backups optimize future cumulative reward
through budget/time transitions; this is batch model-based RL with planning,
not conventional sample-based Q-learning or an interacting neural learner.

Proportional fulfillment approximates an hour that would overspend; evaluation
uses exact ordered affordability. Missing campaign hours use a pooled observed-hour
prior, which can be wrong for genuine campaign schedules. Execution adapts to causal
price and inventory feedback. Campaign-specific tables change representation as
well as algorithm relative to the shared neural policies.

## Selection and evaluation

| Stage | Dates in 2013 |
|---|---|
| Auxiliary training | June 6-12 |
| Forward fold 1 | Auxiliary + October 19-21; validate October 22 |
| Forward fold 2 | Auxiliary + October 19-22; validate October 23 |
| Forward fold 3 | Auxiliary + October 19-23; validate October 24 |
| Final refit | Auxiliary + October 19-24 |
| Fixed held-out replay | October 25-27 |

Every fold refits CTR, traffic, price/cost distributions, reward scales and budget
references using its training prefix. A campaign requires past calibration to
enter validation. Histories are sparse: final refit has one active calibration
day for 2261 and two for 2997.

Validation conditions are budgets 0.20/0.30/0.45 and budget 0.30 at prices x1.20,
coverage 0.50 and coverage 0.75. Selection uses five-seed ensembles. Its objective
is mean paired gain over the strongest tuned validation baseline, minus 0.5 times
the across-date standard deviation. The primary is frozen before test evaluation;
a better secondary test score cannot retrospectively change primary selection.

Ten deployable comparators are constant, global linear, campaign linear,
campaign/budget linear, PID, quality PID, market linear, traffic linear, unscaled
linear and a cost controller. The traffic comparator receives the same causal
traffic information. Campaigns lacking campaign-specific validation history use a
globally tuned fallback. All bids share cap 3. Privileged linear observes full-day
price; hindsight fixed bidding and click-aware knapsack bounds use future outcomes.
These are diagnostics, not deployable references.

Evaluation includes nominal conditions and six stresses: prices x1.5, evening
inventory x3, budgets 0.15/0.60, click retention x0.5 and coverage 0.5. Budget endpoints
and price x1.5 exceed training-control ranges. Coverage 0.5 is within range. Evening
multiplication duplicates opportunities; thinning uses paired seeds. All scenarios
reuse the same dates.

## Verified results

Metrics use **ensemble rows only (`seed == -1`)**. Mixing individual members with
ensembles changes the averages. The supplied evaluation has 1,225 rows across
seven conditions, with zero budget violations.

| Condition | Empirical RL clicks/day | Highest baseline mean | Observed lift |
|---|---:|---|---:|
| Nominal | 50.2 | Unscaled linear: 45.2 | +11.06% |
| Prices x1.5 | 39.6 | Quality PID: 37.0 | +7.03% |
| Evening inventory x3 | 57.6 | Unscaled linear: 55.4 | +3.97% |
| Budget 0.15 | 30.8 | Market linear: 28.0 | +10.00% |
| Budget 0.60 | 75.8 | Constant: 71.2 | +6.46% |
| Click retention 0.5 | 27.4 | Unscaled linear: 25.0 | +9.60% |
| Coverage 0.5 | 40.6 | Constant: 38.6 | +5.18% |

Each row's largest baseline mean is identified descriptively after evaluation,
not as a deployment-selection rule. The seven conditions are not independent datasets.

**Frozen primary: Traffic DQN**, selected at 1,500 episodes. It scores 41.6 nominal
clicks/day, below traffic linear (42.0), PID/quality PID (45.0) and unscaled linear
(45.2). Traffic PPO scores 42.0; residual PPO 40.2. Empirical RL is a pre-specified
secondary candidate; its better held-out mean does not change the primary result.

Empirical RL earns 251 clicks versus unscaled linear's 226. Mean daily utilization
is 87.87%; aggregate spend per click is 14.22% lower. Individual bootstrap-model
scores range 48.2-50.8 clicks/day. Campaign 2997 improves to 98.5 versus 81.0, while
2261 falls to 18.0 versus 21.33. Benefits are not uniform across campaigns or days.

Exact date-cluster 95% intervals for empirical-minus-baseline clicks/day are
[1.0, 8.0] against unscaled linear, [-4.0, 12.5] against PID and [-2.0, 14.5] against
traffic linear. Resampling retains whole dates and unequal campaign-day counts.
Only three date clusters are available; secondary intervals are unadjusted for
multiple comparisons. Seeds are optimization/bootstrap replicates, not independent
historical dates. Confirmation requires a fixed method on additional untouched data.

The raw evaluation matches its frozen protocol hash. Metric audits reproduce all
280 comparison records and check the complete grid, finite values, budget compliance
and click-aware bounds. Source fingerprints match the tested release; 48 fixture
tests passed. Checkpoint-level replay was not repeated from the submitted metrics
alone. Keep the complete local run for that verification. Packaging changes do not
alter runtime source, configuration, selection or results.

## Files and commands

| Command | Purpose |
|---|---|
| `python run.py all --out reproduction` | Train, select, evaluate, analyze, verify |
| `python run.py train --out reproduction` | Training and development-only selection |
| `python run.py evaluate --out reproduction` | Evaluate selected models and analyze |
| `python run.py analyze --out reproduction` | Regenerate analysis and graphs |
| `python run.py verify --out reproduction` | Verify complete local artifacts and replay |

Use consistent `--config` and `--out` arguments throughout a run. Missing data or
incomplete outputs produce errors; substitute data/results are never created.

A complete run saves models, protocols, CV/training records, per-day evaluations,
seed summaries, hindsight diagnostics and learning/comparison/stress/spending plots.
Unprefixed files describe the neural stage; `final_*` files combine neural and
empirical candidates. Learning-curve shading is across-date standard deviation,
not a confidence interval.

Only `results/final_evaluation.csv`, `results/final_summary.csv` and
`results/candidate_comparisons.json` are allowlisted for publication. They are a
small evidence subset, not a resumable experiment. Report source is
`BidPace_RL_Report.tex` with `paired_gains.png`; compile with
`pdflatex BidPace_RL_Report.tex` twice. LaTeX is optional for running the code.

| Module | Responsibility |
|---|---|
| `replay.py`, `fetch.py` | Download, causal joins, calibration and auction environment |
| `agents.py`, `ppo.py` | NumPy networks, replay targets and PPO math |
| `adaptive.py`, `control.py`, `policies.py` | Causal feedback, policy wrappers and baselines |
| `protocol.py` | Neural training, temporal selection, evaluation and verification |
| `model_based.py` | Empirical MDP learning and final selection |
| `reporting.py` | Comparisons, plots and empirical-agent export |
| `tests/` | Fixtures, gradients, budget/cutoff checks and full CLI tests |

## GitHub submission

Create an empty repository at the URL above, or update that URL in the README
and report source if choosing another name. Export both agents from your completed
local run first. The ignore rules allow those two agents, the three small metric
files and report assets while excluding data and full training caches.

```powershell
git init
git add .
git status
git ls-files agents
git commit -m "Implement and evaluate budget-constrained RL bidding"
git branch -M main
git remote add origin https://github.com/Chirag514/BidPace-RL.git
git push -u origin main
```

`git ls-files agents` must list both exported agents before committing. Keep an
authentic history of subsequent work; do not reconstruct fictitious earlier commits.
If the repository is private, the assignment names `shubhambagwari` and
`Divyaanshmertia` as reviewer accounts to grant access to.

## References

- [Provider documentation](https://contest.ipinyou.com/ipinyou-dataset.pdf)
- [iPinYou benchmark](https://arxiv.org/abs/1407.7073)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [PPO](https://arxiv.org/abs/1707.06347)
