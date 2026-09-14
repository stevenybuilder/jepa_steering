# Published aggregates

Every number in the README, the documentation, and the figures is generated from the
files in this directory. Each family has a frozen protocol, a hash-checked summary,
and the per-case tables the summary was computed from, so any reported value can be
recounted on CPU with `scripts/check_public_results.py` and
`scripts/check_lcfm_replication.py`.

| Family | Files | What it holds |
|---|---|---|
| Action-history replication | `lcfm_replication_*`, `lcfm_history_ranking_*`, `lcfm_history_selection_cost_*` | 200 pre-registered starting states, 35 conditions, two candidate banks: 42,000 ranking rows and 126,000 reconstruction rows, with the protocol and summary hashes |
| Action counterfactual and condition specificity | `action_counterfactual_*`, `action_condition_*`, `candidate_specificity*` | The sixteen-state exploratory studies and the norm-matched action-range controls |
| CEM search | `cem_expansion_*`, `cem_steering_*` | Per-iteration proposal statistics and selected prefixes under learned and random edits |
| Decision geometry, precision, and physical replay | `controlled_geometry_*`, `pathway_geometry_*`, `planned_prefix_*`, `decision_*`, `pilot_summary_*` | Score-margin bound cases, FP32/BF16 reconstruction, and paired forecast-versus-execution outcomes |
| Protected confirmation and benchmarks | `all_task_ablation_*`, `benchmark_comparison*`, `reported_*`, `systems_*` | 384 fresh scenarios by eight arms on four tasks, with provenance-separated published references |

Raw per-episode archives, checkpoints, and licensed inputs are not in the repository;
`docs/REPRODUCING.md` describes what they are and how the receipts bind to them.
