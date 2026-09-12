# Baseline-harness GPU tuning (legacy)

Provisioning, device parity, fleet operations, collection and cost rules now live in
[GPU_COMPUTE_PLAYBOOK.md](GPU_COMPUTE_PLAYBOOK.md). This page keeps only the notes
specific to the original seven-frame baseline harness (`jepa-multigpu`,
`jepa-compare-runs`); they do not apply to the frozen planner runs, which cannot be
batched or run in reduced precision. Roofline reasoning:
[jax_scaling_notes.md](../jax_scaling_notes.md).

## Applied design (baseline harness)

- Batch independent windows per GPU (measure 4, 8, 16; keep the largest parity-valid size).
- Strict FP32 with TF32 disabled is the reference; BF16/FP16/TF32 are explicit candidates
  that must win an end-to-end comparison on this benchmark.
- Pinned collated tensors, bounded prefetch, one producer per dataset object; move all
  metrics to CPU in one transfer; overlap cache writes with the next batch.
- One process per GPU on disjoint trajectory shards; no collectives.
- Batch registered intervention arms inside each window batch (memory scales with
  windows × arms, so start smaller). See [INTERVENTIONS.md](INTERVENTIONS.md).

`report.json` records precision, TF32 state, batch size, prefetch depth, pinned memory,
stage timing, peak memory, device capability and throughput. Decode-worker time overlaps
GPU time and must not be added to wall time.

## Bounded measurement sequence

Use a durable `--output`; copy every receipt off the worker before destroying it.

```bash
# strict FP32 reference
jepa-multigpu --vendor vendor/jepa-wms --checkpoint data/checkpoints/jepa_wm_metaworld.pth.tar \
  --checkpoint-sha256 METAWORLD_CHECKPOINT_SHA256 --manifest runs/metaworld-inventory/trajectories.jsonl \
  --exposure-registry data/metaworld-exposure.json --data-root data/metaworld --tasks mw-reach mw-reach-wall \
  --max-trajectories 12 --batch-size 8 --devices 0 --precision float32 --prefetch-batches 2 \
  --max-runtime-seconds 1800 --output DURABLE_RUN_ROOT/metaworld-fp32-b8
# BF16 candidate: identical inputs, --precision bfloat16, --output .../metaworld-bf16-b8
jepa-compare-runs --reference DURABLE_RUN_ROOT/metaworld-fp32-b8/shard-000 \
  --candidate DURABLE_RUN_ROOT/metaworld-bf16-b8/shard-000 --output DURABLE_RUN_ROOT/metaworld-bf16-comparison
```

Use `--max-absolute-drift` / `--max-relative-drift` only with predeclared, reviewed limits.
The comparison covers end-to-end error metrics, not latent equivalence or intervention efficacy.
