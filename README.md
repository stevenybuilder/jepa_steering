# JEPA-WM offline intervention study

PyTorch experiments on frozen JEPA-WM checkpoints. The primary task suite is
**MetaWorld Reach, MetaWorld Reach-Wall, and Push-T**. The broader unedited
baseline covers all 42 released MetaWorld tasks and Push-T.

The active design is [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), with machine-readable
settings in [study.json](configs/study.json). The [benchmark runbook](docs/BENCHMARK.md)
contains setup, commands, metrics, and the measurement-based compute estimate.
The [GPU/Vast guide](docs/GPU_EFFICIENCY.md) documents bounded multi-GPU execution,
precision comparisons, and the read-only Vast offer workflow.

## Current state

- Implemented: real-dataset inventory, a pinned PyTorch JEPA-WM adapter, recorded-action
  H6 baseline benchmark, per-trajectory/per-task metrics, immutable run receipts,
  zero-dose instrumentation check, and nonoverlapping trajectory sharding.
- Executed here: CPU synthetic smoke and unit checks only; see [validation](reports/VALIDATION.md).
- Not executed here: official dataset inventory, real checkpoint inference, or GPU throughput.
- Not yet implemented in the new harness: the four intervention experiment runners.
  Their historical implementations remain available for carefully scoped reuse.
- No GPU rental, CEM search, simulator rollout, weight training, or protected-holdout
  evaluation is triggered by the current benchmark.

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

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
jepa-benchmark --backend toy --device cpu --output runs/cpu-smoke
```

The toy fixture is **not JEPA-WM**. Its errors and speed are useful only for checking
the pipeline. Use the real-data commands in the runbook for a GPU benchmark.
