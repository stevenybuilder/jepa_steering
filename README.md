# JEPA-WM offline intervention study

PyTorch experiments on frozen JEPA-WM checkpoints. The approved research scope
contains **Reach, Reach-Wall, Push-T, PointMaze, Wall, and DROID**. The completed
core closed-loop comparison covers **Reach and Reach-Wall only**; this is not a
completed six-task study. DROID uses a recorded-plan action endpoint, not
physical-robot closed-loop success. RoboCasa is excluded.

The active design is [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), with machine-readable
settings in [study.json](configs/study.json). The [benchmark runbook](docs/BENCHMARK.md)
contains setup, commands, metrics, and the measurement-based compute estimate.
The [GPU/Vast guide](docs/GPU_EFFICIENCY.md) documents bounded multi-GPU execution,
precision comparisons, and the read-only Vast offer workflow.

## Current state — September 10, 2026

- **Completed behavioral table:** [960 matched episode evaluations](reports/CORE_METAWORLD_BEHAVIORAL_RESULTS.md),
  comprising two tasks × five conditions × 96 paired scenarios per task. The
  conditions are unsteered, the refined fixed-response edit, vision–action
  coupling, and each edit's matched-random control. Frozen paired analyses were
  independently recomputed. Neither learned edit establishes improved task
  success against both required references; all eight simultaneous intervals
  include zero. This is not proof of no effect.
- **Completed offline comparisons:** [corrected prediction results](reports/CORRECTED_OFFLINE_RESULTS.md).
  Forecast improvements do not establish better planning. The
  [refined fixed-response operator](docs/FIXED_RESPONSE_RANK4.md) replaces the
  expensive online response-probe operator; their measurements are distinct.
- **Implemented infrastructure:** lineage-aware inventory, source-pinned
  PyTorch adapters, fitted/frozen intervention execution, paired planning,
  HMM routing and training-history tools. Implementation or an engineering
  check does not imply completion of the corresponding scientific comparison.
- **Paused/deferred:** HMM behavioral completion, combined extensions,
  remaining task comparisons, fresh confirmation and complete three-training-
  seed/late-checkpoint histories. The core table uses one released checkpoint.
  A planner-decision/candidate-ranking diagnostic is not yet complete.
- **Data corrections remain binding:** see the [Push-T lineage report](reports/PUSHT_LINEAGE_CORRECTION.md),
  [MetaWorld correction](reports/METAWORLD_LINEAGE_CORRECTION.md), and
  [planning alignment](reports/PLANNING_METHOD_ALIGNMENT.md). Reusing released
  development families is replication, not fresh-family confirmation.
- **Preservation and shutdown:** [verified Drive recovery and rental closeout](reports/VAST_FINAL_STORAGE_RELEASE_20260910.md).
  The September 10 audit records twelve deleted rentals and two inaccessible
  disks retained pending recovery. No new experiment or compute restart is
  authorized by this merge or by the historical launcher scripts.

For broader interpretation, see [world-model approaches and measured results](wm-approaches.md).
For computational lessons, see [jax_scaling_notes.md](jax_scaling_notes.md).
The [research-direction notes](docs/RESEARCH_MEMORY.md) record motivation, not
evidence that repeatable improvement has already been established.

## Reproducing and restoring

The [core analysis](reports/wm-approaches/core-analysis.json) and
[verification receipt](reports/wm-approaches/core-verification.json) are tracked.
Raw episode traces, model histories and large archives are stored separately;
follow the [Drive restoration guide](reports/VAST_FINAL_STORAGE_RELEASE_20260910.md#evidence-and-recovery)
for exact object IDs, hashes and reconstruction instructions. Archives and Drive
folders remain private; repository access alone does not grant storage access.

Existing untracked paper drafts and explicitly deferred launchers are not the
authoritative results. Use the dated reports above. Do not resume old queues or
interpret historical completion flags as current authorization.

## Four research categories

| Category | Main alternatives |
|---|---|
| Vision–action coupling | Visual-only, action-condition-only, joint, permuted joint |
| Action-response geometry | Equal-anchor linear, cubic, projected cubic, reflected curvature |
| Routing through imagined time | Constant, memoryless, HMM-filtered gate |
| Intervention distribution | Spatial concentration and block distribution, tested separately |

Every category includes an unedited baseline, exact zero-dose check, and relevant
matched controls. It is not a Cartesian product of every setting.

## Previous work

The entire previous committed workspace is preserved byte-for-byte at
[archive/2026-09-07-workspace](archive/2026-09-07-workspace/). It is the tree from
commit `bf8d9b47214718be3e851573bde68f84b69eba7d`, including old code, plans,
manifests, and lightweight results. [Archive provenance](docs/ARCHIVE.md) explains
what this does and does not preserve. Historical completion flags and instructions
do not govern the new study.

## CPU harness check

The full source-alignment test suite also requires the pinned upstream source
under `vendor/jepa-wms`, its test/runtime dependencies, and this repository's
Git history. Follow the [setup runbook](docs/BENCHMARK.md) first; a bare source
archive without those dependencies is not a complete test environment.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
jepa-benchmark --backend toy --device cpu --output runs/cpu-smoke
```

The toy fixture is **not JEPA-WM**. Its errors and speed are useful only for checking
the pipeline. Use the real-data commands in the runbook for a GPU benchmark.
