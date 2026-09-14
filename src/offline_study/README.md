# Experiment code

The `offline_study` package contains the frozen-model interfaces, activation
interventions, planning contracts, and experiment runners behind every reported
result. No module starts a GPU rental.

| Directory | Contents |
|---|---|
| `models/` | Checkpoint loading, predictor calls, and visual-feature caching |
| `interventions/` | Activation edits, low-rank corrections, and fitted operators |
| `experiments/` | Action-history replication, CEM traces, physical replay, attention and precision studies |
| `planning/` | Candidate plans, goals, simulator states, and planning contracts |
| `fitting/` | Fit intervention operators on the designated fitting trajectories |
| `evaluation/` | Development analyses and statistical comparisons; `checkpoint/` holds upstream evaluation adapters |
| `tasks/` | Separate MetaWorld, navigation, Push-T, and DROID implementations |
| `data/` | Dataset inventory and input preparation |
| `runtime/` | Batching, multi-GPU dispatch, timing, and command-line runners |
| `validation/` | Input, implementation, and receiving-environment checks |
| `core/` | Shared protocol and file-identity utilities |

For the 200-state action-history study, start with
[experiments/lcfm_replication.py](experiments/lcfm_replication.py). For the
protected behavioral evaluation, start with
[experiments/fresh_confirmation.py](experiments/fresh_confirmation.py).
[Reproduction instructions](../../docs/REPRODUCING.md) distinguish current code
from the exact source snapshots used for completed experiments.

Historical routing/HMM and training-engineering workflows have been removed from
the current package. Their source remains in the earlier Git snapshot; the
published experimental records and their original hashes are unchanged.
