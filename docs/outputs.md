# Output reference

Use the [README](../README.md) for installation, training and replay commands.
All paths below are relative to the project root unless an output directory is
specified explicitly.

## Supplied release

| Path | Contents |
| --- | --- |
| `agents/comparison/seasonN_<policy>.npz` | Nine standalone agents: three methods × three seasons |
| `agents/comparison/manifest.json` | SHA-256 of each published agent and evidence file |
| `evidence/nominal.csv`, `nominal.png` | Complete nominal table and main comparison plot |
| `evidence/evaluation.csv` | Per-day test outcomes for every method and scenario |
| `evidence/summary.csv` | Aggregated clicks, utilization and violations |
| `evidence/comparisons.csv` | Every RL/baseline difference with date-cluster intervals |
| `evidence/validation.csv` | Reconstructed forward-validation outcomes |
| `evidence/selection.json` | CV choices, frozen baselines, model/data fingerprints |
| `evidence/shared_selection.json` | Normalized CV scores and common algorithm choice |
| `evidence/study.json` | Candidate set, source hashes and training provenance |
| `evidence/result.json` | Claim flags and their measured supporting values |
| `RELEASE_MANIFEST.json` | Checksum of every packaged file; generated inside the ZIP |

Training caches are stored in `artifacts/neural/`, `artifacts/tabular/` and
`artifacts/calibration/`. Datasets and caches are excluded from the release.

### Claim flags

`result.json` distinguishes four strict nominal point-estimate comparisons:

1. Seasonal validation-selected RL beats its validation-selected reference in
   every season.
2. Seasonal validation-selected RL beats every deployable baseline in every season.
3. One shared validation-selected algorithm beats every baseline in every season.
4. Every RL beats every baseline in every season.

The supplied run records **false for all four**. A best-observed test candidate
does not retrospectively change validation selection. Stress comparisons and
uncertainty are in their corresponding CSVs, not implied by these flags.

## Generated run directory

The comparison command writes its complete local run to `results/`:

```text
results/
  study.json                     # Frozen sources, candidates and rules
  shared_selection.json          # Common algorithm choice from validation
  season1/                       # Also season2/ and season3/
    validation.csv               # Used to reconstruct selection
    selection.json               # Seasonal choice and artifact hashes
    traffic_dqn.npz              # Exported neural ensemble
    campaign_dqn.npz             # Exported campaign-context ensemble
    sampled_q.npz                # Exported calibrated Q-tables
    evaluation.csv               # Per-day results for this season
    evaluation.json              # Selection/result hashes
  evaluation.csv                 # Combined seasonal rows
  summary.csv                    # Aggregated metrics
  nominal.csv                    # Nominal policy-by-season view
  comparisons.csv                # Paired RL-versus-baseline intervals
  nominal.png                    # One main comparison plot
  result.json                    # Explicit claims
```

Seasonal files support independent verification/resume. Root files are the
combined reporting views. Intermediate neural, table and calibration directories
hold reusable training artifacts and their own frozen protocols.

## Stage commands and verification

The `all` command performs `select`, `evaluate`, `verify` and `summarize` in order.
For an existing workspace cache, individual stages are:

```powershell
python -m experiments.compare select
python -m experiments.compare evaluate
python -m experiments.compare verify
python -m experiments.compare summarize
```

For a fresh training run, pass its same `--neural-source`, `--table-source`,
`--calibrated-source` and `--out` values to these commands.

- `build_release.py --check` verifies the shipped model/evidence checksums. In an
  extracted ZIP it also verifies `RELEASE_MANIFEST.json`. It needs no dataset.
- `python -m experiments.compare replay` needs only the shipped policies/evidence
  and prepared data. It checks asset hashes, frozen CV decisions and every test
  outcome without training caches or additional output files. Frozen baseline
  settings and raw/content dataset fingerprints are included in `selection.json`.
- Benchmark `verify` needs prepared data and full source training caches. It
  replays CV, all scenario outcomes and archived DQN evaluations where available,
  checks selection/result hashes, and enforces the hard budget.
- Imported training provenance stays intact; the comparison records inference
  fingerprints separately. Training resume requires matching source,
  configuration and data. After a code change, use a new comparison output path.
- The unit/integration suite uses small synthetic fixtures to check learning,
  causality, serialization, selection and pipeline behavior.

## Loading a trained agent

```python
from experiments.compare import load_agent
from bidpace.replay import ReplayData, make_replay_day, rollout_replay

policy = load_agent("agents/comparison/season2_campaign_dqn.npz")
data = ReplayData("data/season2/replay.npz", list(range(20130606, 20130613)),
                  test_dates=[20130613, 20130614, 20130615])
day = make_replay_day(data.templates["test"][0], budget_fraction=0.3)
print(rollout_replay(day, policy))
```

The archive contains model tensors and training-fitted metadata. The loader
reconstructs the appropriate policy; rollout resets it before each day and
provides the required traffic features.
