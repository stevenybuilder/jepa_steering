# Audits and corrections

Dated records of the checks that gate the published numbers. They are kept because
several of them changed a result: a lineage duplicate, a stimulus pairing error, and
a sampler mismatch were each found by an audit and corrected before the protected
confirmation ran.

| Report | What it established |
|---|---|
| `CORE_METAWORLD_BEHAVIORAL_RESULTS.md` | The completed core MetaWorld behavioral comparison across all arms |
| `SIX_TASK_RESULTS_20260911.md` | Completed measurements for the six-task released-checkpoint table |
| `PLANNER_DECISION_DIAGNOSTIC_RESULTS.md` | The common-action planner-decision diagnostic behind the score bound |
| `MATCHED_RANDOM_CONTROL_AUDIT_20260911.md` | That every random control matches its learned edit in site, rank, and dose |
| `AUTHOR_VALIDATION_AUDIT.md` | Coverage of the released checkpoints' own validation rows and rerun accounting |
| `DROID_BASELINE_RECONCILIATION_AUDIT.md` | Reconciliation of the DROID action-agreement baseline |
| `CORRECTED_OFFLINE_RESULTS.md`, `METAWORLD_LINEAGE_CORRECTION.md`, `METAWORLD_STIMULUS_REPAIR.md`, `PUSHT_LINEAGE_CORRECTION.md`, `POINTMAZE_TRAINING_SAMPLER_CORRECTION.md` | Errors found in development data or samplers and how the affected results were corrected |

`fresh-confirmation/` and `wm-approaches/` hold the machine-readable verification
receipts for the protected confirmation; `cpu_smoke.json` is the CPU smoke-test record
used by CI.
