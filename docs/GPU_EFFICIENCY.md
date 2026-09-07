# GPU efficiency and Vast execution

This project uses the GPU chapter of the JAX Scaling Book as a performance model,
not as permission to copy LLM sharding recipes into a small frozen world model. The
relevant constraints are Tensor Core utilization, HBM traffic, host-to-device stalls,
and communication overhead.

Sources:

- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/)
- [Pinned JEPA-WM source](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0)

## Applied design

- Batch independent windows on each GPU. A window encodes seven frames, so batch size
  controls the encoder matmul intensity and memory footprint. Measure 4, 8, then 16;
  retain the largest parity-valid size with safe memory headroom.
- Keep the initial reference in strict FP32 with TF32 disabled. BF16, FP16, and TF32
  are explicit candidate modes. The released upstream training configs use BF16, but
  that does not remove the need for an end-to-end comparison on this benchmark.
- Pin collated CPU tensors and prefetch a bounded number of batches. One producer owns
  each dataset object, while decode/storage work overlaps GPU execution.
- Move all six horizon/modality metrics to the CPU in one transfer. When cache writing
  is enabled, serialize staged CPU tensors on a bounded worker so disk writes can
  overlap the next GPU batch.
- Use one process per GPU with disjoint trajectory shards. There are no DDP/NCCL
  collectives because the model fits on one device and trajectories are independent.
  This avoids the cross-node bandwidth roofline entirely.
- Start with one GPU, then two. Eight GPUs are justified only by measured two-GPU
  storage and throughput scaling. A faster GPU can expose data/decode bottlenecks, so
  GPU FLOP specifications alone are not a purchasing rule.
- Batch registered intervention arms inside each window batch. This encodes the seven
  source/target frames once per window and performs one larger predictor unroll for all
  arms. See [INTERVENTIONS.md](INTERVENTIONS.md); effective activation memory scales
  with windows times arms, so intervention batches start smaller than baseline batches.

`report.json` records precision, TF32 state, batch size, prefetch depth, pinned-memory
use, stage timing, peak allocated/reserved memory, device capability, and both pipeline
and rollout throughput. Decode worker time can overlap GPU time and therefore must not
be added to wall time.

## Vast preflight

The local Vast CLI is authenticated separately from this repository. Confirm without
printing the API key:

```bash
vastai --version
vastai show instances --raw
vastai show ssh-keys --raw
```

Run the read-only offer search:

```bash
scripts/vast/search_offers.sh > offers.json
```

`gpu_frac` is intentionally not constrained: on Vast it describes the fraction of the
host's GPU count included by an offer, not fractional access to one physical GPU. For
example, a one-GPU slice of an eight-GPU host may report `gpu_frac=0.125`. Treat the
returned price as ephemeral and re-run the query immediately before an explicitly
budget-authorized rental, including the requested disk in the final price check.

The query asks for one reliable CUDA-capability-8.0+ GPU with at least 24 GB VRAM,
100 GB disk, and reasonable download bandwidth. It does not rent anything. Select a
GPU only after fixing an hourly price and total budget. Consumer GPUs can be efficient
for the communication-free baseline; A100/H100-class GPUs offer more memory and BF16
throughput but must win on measured cost per valid window.

After explicitly renting an SSH instance, copy `scripts/vast/bootstrap.sh` to it or
use it as the Vast on-start script. It pins Python 3.10 and the upstream commit, installs
the package, checks CUDA, and runs local tests. It deliberately does not fetch datasets,
checkpoints, or start billable experiments.

Set `JEPA_DURABLE_ROOT` before bootstrap if the instance has a mounted volume intended
to survive worker replacement. Otherwise the script uses `/workspace/jepa-runtime`,
which must be copied elsewhere before the instance is destroyed.

## Bounded measurement sequence

Use a durable mounted path for `--output`. Do not destroy an instance until every run
receipt and required cache has been copied off the worker.

First run strict FP32 on one GPU:

```bash
jepa-multigpu --vendor vendor/jepa-wms \
  --checkpoint data/checkpoints/jepa_wm_metaworld.pth.tar \
  --checkpoint-sha256 METAWORLD_CHECKPOINT_SHA256 \
  --manifest runs/metaworld-inventory/trajectories.jsonl \
  --exposure-registry data/metaworld-exposure.json \
  --data-root data/metaworld --tasks mw-reach mw-reach-wall \
  --max-trajectories 12 --batch-size 8 --devices 0 \
  --precision float32 --prefetch-batches 2 \
  --max-runtime-seconds 1800 --output DURABLE_RUN_ROOT/metaworld-fp32-b8
```

Then run a BF16 candidate on exactly the same inputs and selection:

```bash
jepa-multigpu --vendor vendor/jepa-wms \
  --checkpoint data/checkpoints/jepa_wm_metaworld.pth.tar \
  --checkpoint-sha256 METAWORLD_CHECKPOINT_SHA256 \
  --manifest runs/metaworld-inventory/trajectories.jsonl \
  --exposure-registry data/metaworld-exposure.json \
  --data-root data/metaworld --tasks mw-reach mw-reach-wall \
  --max-trajectories 12 --batch-size 8 --devices 0 \
  --precision bfloat16 --prefetch-batches 2 \
  --max-runtime-seconds 1800 --output DURABLE_RUN_ROOT/metaworld-bf16-b8
```

Compare runs before adopting the candidate. With no bounds, this produces measurements
without claiming acceptance:

```bash
jepa-compare-runs \
  --reference DURABLE_RUN_ROOT/metaworld-fp32-b8/shard-000 \
  --candidate DURABLE_RUN_ROOT/metaworld-bf16-b8/shard-000 \
  --output DURABLE_RUN_ROOT/metaworld-bf16-comparison
```

Only use `--max-absolute-drift` and `--max-relative-drift` when those limits have been
predeclared and scientifically reviewed. The comparison covers end-to-end error metrics;
it does not by itself establish raw latent equivalence or intervention efficacy.
