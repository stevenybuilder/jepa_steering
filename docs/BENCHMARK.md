# Benchmark runbook

The current executable is an unedited recorded-action H6 baseline. It establishes
throughput and trajectory-level forecast errors. It does not yet benchmark intervention
arms, run a planner, or establish statistical efficacy.

## Environment

Use the upstream-supported **Python3.10** environment for real JEPA-WM execution.
The lightweight package/tests also run on newer Python, but that does not validate
the upstream dependency stack. Install upstream per its README (Torch/torchvision
versions must be compatible), then install this package in the same environment.

```bash
git clone https://github.com/facebookresearch/jepa-wms.git vendor/jepa-wms
git -C vendor/jepa-wms checkout 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0
python -m pip install -e vendor/jepa-wms
python -m pip install -e .
```

Follow the upstream download instructions for the MetaWorld/Push-T datasets and the
jepa_wm_metaworld / jepa_wm_pusht checkpoints. Record the downloaded dataset revision
and checkpoint SHA256. The model loader can still fetch the frozen DINO backbone on
first initialization: preload it before timing and record its resolved cache provenance.
Dataset transfer and model setup are not prediction throughput.

## Inventory

Substitute actual paths and the downloaded dataset revision below. Push-T root must
contain train/ and val/; MetaWorld root must contain the released parquet files.

```bash
jepa-inventory --vendor vendor/jepa-wms --dataset metaworld \
  --data-root data/metaworld --data-revision DATA_REVISION \
  --output runs/metaworld-inventory
jepa-inventory --vendor vendor/jepa-wms --dataset pusht \
  --data-root data/pusht --data-revision DATA_REVISION \
  --output runs/pusht-inventory
```

Inspect inventory.json before running. Count discrepancies are reported rather than
silently truncating or duplicating data. Source file content hashes and historical
exposure reconciliation are outstanding launch requirements; a declared revision alone
is not proof that local raw media are intact. Keep old protected trajectories out of
timing/development even if their provisional new split differs.

Push-T is split by exact-initial-state family, not by rollout row. Manifest validation
fails if one lineage group crosses study splits, and the exposure filter excludes an
entire development family if even one of its rows lacks explicit clearance. The released
Push-T train pool is already development-exposed at the family level in this study; its
holdout-labelled groups must not be presented as untouched confirmation. See the dated
correction in the experiment plan.

The real benchmark requires a reviewed exposure registry bound to the manifest:

```json
{
  "manifest_sha256": "SHA256_OF_TRAJECTORIES_JSONL",
  "review_basis": "Source-ID mapping to prior protected manifests and exposure logs",
  "trajectories": {
    "metaworld:all:ROW_INDEX": {
      "use": "development",
      "lineage_group": "metaworld:trajectory:ROW_INDEX",
      "evidence": "Specific record demonstrating this source is not in a protected cohort"
    }
  }
}
```

This is a schema example, not evidence for any actual row. Unmapped sources are
excluded before trajectory capping and GPU sharding; protected entries are excluded as
well. A requested task with no reviewed trajectories still fails. This prevents a new
90/10 split from accidentally opening an old holdout or changing the selected panel
merely because a protected row happened to sort first.

## Initial GPU commands

Supply checksums obtained from verified checkpoint files (for example `sha256sum`).
Use separate processes/output directories for the two task-trained checkpoints.

```bash
jepa-benchmark --vendor vendor/jepa-wms \
  --checkpoint data/checkpoints/jepa_wm_metaworld.pth.tar \
  --checkpoint-sha256 METAWORLD_CHECKPOINT_SHA256 \
  --manifest runs/metaworld-inventory/trajectories.jsonl --data-root data/metaworld \
  --exposure-registry data/metaworld-exposure.json \
  --tasks mw-reach mw-reach-wall --max-trajectories 12 \
  --batch-size 8 --device cuda:0 --precision float32 --prefetch-batches 2 \
  --output runs/benchmark-metaworld-b8

jepa-benchmark --vendor vendor/jepa-wms \
  --checkpoint data/checkpoints/jepa_wm_pusht.pth.tar \
  --checkpoint-sha256 PUSHT_CHECKPOINT_SHA256 \
  --manifest runs/pusht-inventory/trajectories.jsonl --data-root data/pusht \
  --exposure-registry data/pusht-exposure.json \
  --tasks pusht --max-trajectories 12 \
  --batch-size 8 --device cuda:0 --precision float32 --prefetch-batches 2 \
  --output runs/benchmark-pusht-b8
```

Budget a maximum of 0.5 GPU-hours for the initial measurement, supervised by the job
runtime. Do not leave all eight GPUs leased during setup/download. The CLI caps the
default selected trajectory count, but is not a scheduler or billing stop mechanism.

Run a separate invocation with `--write-cache` to measure encoded-cache I/O. Cache CPU
staging and background serialization are timed separately, and the bounded writer queue
is fully drained before `DONE.json` is written. Without it, cache timings are zero and
the report must not be extrapolated to a full capture job. The initial cache contains
final encoded targets only, not every internal block.

Strict FP32 keeps TF32 disabled. `--precision bfloat16`, `--precision float16`, and
`--allow-tf32` are explicit performance candidates, never silent substitutions. Run a
strict-FP32 reference on the same selection and use `jepa-compare-runs` before adopting
one. The comparison does not adjudicate a candidate unless both predeclared drift bounds
are supplied. See [GPU_EFFICIENCY.md](GPU_EFFICIENCY.md).

The `--split` choices are fit/development only. There is no protected-holdout override.
Future confirmation needs an audited frozen registry and a dedicated evaluator.

## Metrics and extrapolation

report.json contains:

- Synchronized encode and recursive-rollout wall time, data-load/decode time, and pipeline throughput.
- Setup and warmup time separately; peak allocated/reserved GPU memory.
- Visual/proprio embedding MSE at H1/H3/H6, first averaged within each rollout, then
  within lineage family, then equally across families within each task. Per-rollout
  summaries remain available but are not the independent-unit analysis. These are not
  physical-state errors.
- Requested/actually measured task, rollout, and lineage-family counts; immutable
  selection/config hashes.
- Exact zero-dose identity. No success rate or p-value is produced by the timing harness.

For a measured real workload: hours = target_windows / measured_windows_per_second / 3600.
Project each task and stage separately. Baseline throughput does not measure donor
generation, extra shadow rollouts, nonlinear fitting, or repeated intervention passes.
Measure a representative pass for each category before pricing the full suite.

## Multiple GPUs

The benchmark supports disjoint trajectory shards, without DDP/padded duplicate samples.
`jepa-multigpu` launches one process per listed device, enforces a wall-clock limit,
terminates sibling workers on failure, and validates nonoverlapping selections before
writing an aggregate receipt. For example, add `--devices 0 1 --max-runtime-seconds 1800`
to the single-GPU arguments. Use the same overall selection/max-trajectories. For eight
GPUs use devices 0..7 only after the two-GPU result demonstrates storage scaling. Empty
shards are errors, not zero-throughput successes.

The launcher checks common config/source, disjoint trajectory IDs, report hashes, and
duplicate windows. Multi-host execution remains out of scope. Avoiding model collectives
is intentional because rollout computation is separable and the model fits on one GPU;
this compute sharding does not imply that correlated rollouts are statistically independent.

## Available local validation

```bash
python -m unittest discover -s tests -v
jepa-benchmark --backend toy --device cpu --output runs/toy-smoke
```

The report says synthetic_smoke_only and gpu_benchmark_valid=false. This checks H6
bookkeeping, task coverage, metric alignment, and instrumentation. It is not JEPA-WM
and must not be used to estimate A100/H100 throughput or scientific prediction quality.
