# BidPace-RL

**Course:** Artificial Intelligence (CSL3090), IIT Jodhpur  
**Team:** Chirag Bhutra (B23MT1012), Akshay Gupta (B23BB1006)  
**Repository:** https://github.com/Chirag514/BidPace-RL

Reinforcement learning for real-time advertising bids under a daily budget.
The benchmark evaluates **Traffic DQN, Campaign DQN and sampled-Bellman RL**
against bidding and pacing baselines across three iPinYou seasons. The objective
is to maximize purchased recorded clicks while respecting the budget.

## Installation

Requires Python 3.11 or 3.12.

```sh
git clone https://github.com/Chirag514/BidPace-RL.git
cd BidPace-RL
```

Create and activate an environment:

**PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Bash (Linux/macOS)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and check the supplied assets:

```sh
python -m pip install -r requirements.txt
python build_release.py --check
```

The remaining Python commands work in either shell with the environment active.
Computation runs on CPU using NumPy. Allow **20 GB of free disk space** for
dataset downloads, preparation and training artifacts.

## Data preparation

```powershell
python -m bidpace.benchmark prepare
```

This downloads approximately 2.15 GB of public logs, verifies provider checksums,
and prepares `data/seasonN/replay.npz` for each season. Complete local inputs are
reused. Temporary database joins account for most of the peak disk requirement.

## Training and evaluation

```powershell
python -m experiments.compare all --train-missing
```

The pipeline trains the DQNs, fits sampled-Q tables, tunes baselines, selects
policies on chronological validation folds, and evaluates seven test scenarios.
It verifies exported policies and writes the comparison to `results/`.
Reusable training artifacts are stored in `artifacts/`.

Set seeds, workers, DQN parameters and evaluation conditions in `config.json`
before starting. Neural training can take several CPU hours. Completed stages
are reused; an interrupted seed or tabular fold restarts. Configuration or source
changes require separate cache and result directories. Command-line options are
listed by `python -m experiments.compare --help`.

## Evaluate the supplied policies

After data preparation:

```powershell
python -m experiments.compare replay
```

This loads the nine included policies and frozen baseline settings, then checks
all **3,332 published test outcomes** against `evidence/evaluation.csv`.
Model and baseline parameters come from the published assets.
The six published Traffic DQN and Campaign DQN policies were imported from
completed seasonal training runs; `evidence/study.json` records their provenance.
Training from scratch is a separate experiment;
checkpoint selection and floating-point differences can change its results.

## Results

Nominal recorded clicks per advertiser-day:

| Season | Traffic DQN | Campaign DQN | Sampled-Bellman RL | Best observed deployable baseline |
| --- | ---: | ---: | ---: | ---: |
| 1 | 121.33 | 130.11 | 131.89 | 128.56 (MPC) |
| 2 | 69.80 | 76.33 | 72.53 | 75.87 (Linear) |
| 3 | 36.40 | 41.00 | 54.30 | 53.40 (Budget Linear) |

Validation selects sampled-Bellman for Season 1 and Campaign DQN for Seasons 2
and 3. The last column is the best baseline observed on the test dates (hindsight),
not a selected reference. The validation-selected references are MPC, Budget Linear
and Market Linear, respectively. The selected Season 3 policy obtains 41.00
clicks/day against Market Linear's 52.30, illustrating the limits of
validation-based policy selection.

The [comparison plot](evidence/nominal.png) and [complete table](evidence/nominal.csv)
include the measured nominal outcomes. Stress results and paired intervals are
in `evidence/`. These are retrospective measurements on logged winning inventory;
unobserved lost-auction outcomes and limited test dates restrict conclusions about
live auction performance.

## Documentation

- [Methodology](docs/methodology.md): environment, algorithms and experimental protocol.
- [Sampled-Bellman method](docs/tabular_method.md): transition model and calibration.
- [Output reference](docs/outputs.md): artifacts, verification and policy loading.

## Tests and packaging

```powershell
python -m unittest discover -s tests -v
python build_release.py
```

The release builder creates `BidPace-RL.zip` containing source, trained policies,
evidence and checksums. Datasets, local caches and the separately submitted report
are excluded.
