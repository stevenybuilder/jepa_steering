# GPU execution: scaling principles for inference

This was PyTorch inference plus simulation, not distributed training. The
[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
and [Scaling Book GPU chapter](https://jax-ml.github.io/scaling-book/gpus/) informed
how we separated compute, memory and communication constraints. They are systems
references, not libraries used by the executor.

## Model, fleet, and data scale

The [systems inventory](../paper/data/systems_scale.json) records counts, worker
log excerpts, and source hashes. Encoder and predictor counts come from the
actual loaded models. Small external proprioceptive embeddings are included in
the totals below; action-conditioning parameters are already in the predictor.

| Model / tasks | Visual encoder | Predictor | External proprio encoder | Total parameters |
|---|---:|---:|---:|---:|
| MetaWorld: Reach and Reach-Wall | 22,056,576 | 17,630,480 | 80 | **39,687,136** |
| Push-T | 22,056,576 | 17,626,480 | 80 | **39,683,136** |
| PointMaze | 22,056,576 | 17,626,480 | 80 | **39,683,136** |
| Wall | 22,056,576 | 17,626,480 | 48 | **39,683,104** |
| DROID | 303,154,176 | 228,835,328 | 0 | **531,989,504** |

The simulator models use DINOv2 ViT-S/14 and six predictor blocks; DROID uses
DINOv3 ViT-L/16 and twelve predictor blocks. The five full model configurations
sum to **690,726,016 parameters**. Counting their shared DINOv2 encoder once
gives **624,556,288 parameters**. Reach and Reach-Wall use the same checkpoint;
neither separate tasks, intervention arms, nor GPU replicas multiply this total.
Optimizer state and optional visualization heads are excluded. The external
proprio encoder is a bias-bearing 1×1 convolution with 16 outputs: 80 parameters
for four inputs, 48 for two inputs. DROID has no such encoder.

At 03:51 UTC on September 13, the campaign recorded **eight qualified hosts
running concurrently, with 56 GPUs**: 24 RTX 5090s, 16 RTX 4090s, eight RTX PRO
6000s and eight A100s. This is concurrent fleet capacity; per-GPU activity varied
as scenarios completed and queues were reassigned. The inventory binds that
snapshot to the eight provider receiving records.

| Staged dataset assets | Size (decimal GB) | Source |
|---|---:|---|
| 126 MetaWorld state/action Parquet shards | 0.737 | Pinned official manifest in the systems inventory |
| Push-T noisy-action archive | 2.785 | Pinned ZIP, SHA-256 verified locally |
| PointMaze archive | 0.718 | [Pinned navigation assets](../configs/navigation_assets.json) |
| Wall archive | 1.668 | Same navigation manifest |
| 16 Franka recordings and companions for DROID evaluation | 2.106 | [Pinned DROID assets](../configs/droid_assets.json) |

These assets total **8.015 GB**, counting each staged source once. Push-T and
navigation rows are compressed ZIP sizes; Franka sizes are HDF5 recordings and
companions. External MetaWorld videos are additional to the Parquet subtotal.
The source inventory indexed **12,600 MetaWorld trajectory records** and
**18,706 Push-T trajectory records**, **31,306 total**. These are corpus inventory
counts, not the number of independent experiment evaluations. The DROID entry covers the released
evaluation recordings, not the full raw training corpus. Separately, the
mechanism campaign's cloud audit checked **287 output archives / 12.786 GB**
against existing download-hash receipts. Input and output volumes are not pooled.

## Batched matrix operations

The CEM planner batches 300 candidate trajectories per GPU. At B3, the refined
edit flattens each 256×400 activation field, projects it onto three fitted
features, adds an intercept, and applies a precomputed 4×4 coefficient map. A
final batched product with four basis directions reconstructs the edit. The
response inverse is calibrated offline; the online path uses thin matrix
products instead of rerunning response probes for each candidate. See
[`compose_map` and the output hook](../src/offline_study/fixed_response.py).

An excluded engineering benchmark timed ten warmed forecasts per mode:
strict FP32 **3.13847 s**, TF32 **2.85019 s**, BF16 **2.77808 s** median. The
ratios are **1.10× and 1.13× forecast speedups**; both alternatives changed
predictions. The protected comparison retained strict FP32. These measurements
describe whole forecasts, not an isolated GEMM benchmark or full-episode speedup.

## Execution design

| Principle | Application | Implementation |
|---|---|---|
| Replicate a model that fits | Independent single-GPU scenario jobs; keep eight paired arms together | [Static queue](../scripts/vast/run_fresh_static_queue.py) |
| Batch within a GPU | Preserve upstream batches of 300 candidate trajectories | [Executor](../src/offline_study/fresh_confirmation.py) |
| Remove repeated online work | Offline response-map calibration; no online probes/shadows for the refined arm | [Fixed response](../src/offline_study/fixed_response.py) |
| Stage bulk data outside the inner loop | Worker-local weights/inputs with hash checks | [Range downloader](../scripts/vast/fresh_parallel_download.py) |
| Overlap collection and computation | Snapshot published records while remaining scenarios run | [Snapshotter](../scripts/vast/fresh_stream_backup.py), [backup loop](../scripts/vast/fresh_cloud_backup_loop.py) |
| Count validated throughput | Collect complete scenarios with hash and engineering receipts | [Collector](../scripts/vast/collect_fresh_metadata.py) |

This is task-level data parallelism without gradient synchronization. No DDP
all-reduce, optimizer sharding, ZeRO/FSDP, tensor or pipeline parallelism was needed.
No custom CUDA or new FlashAttention implementation was introduced. Existing
backend kernels are not an original project optimization. The protected panel
kept strict FP32/TF32-off instead of changing precision mid-comparison.

## Bottlenecks and trade-offs

`T_total ≈ T_setup + max(worker remaining work / measured worker throughput) + T_collection`

Amdahl's idealized bound, `S(N) = 1 / ((1 − P) + P/N)`, explains why more GPUs do
not eliminate serial work. Setup failures, environment preparation, cloud
transfers, simulator/replanning steps and long-tail assignments all mattered.
GCS staging traffic is not inter-GPU collective communication; parallel downloads
address the former, not GPU memory bandwidth.

Queues guard against duplicate ownership and reassignment of started scenarios.
Replacement workers use the same frozen inputs. Completion requires all arms and
valid receipts, not just high utilization. Results were archived before release.

A bounded [CPU/CUDA profiling diagnostic](../scripts/vast/profile_fresh_forecast.py)
was added for an excluded input. Its existence does not demonstrate deployment of
a measured kernel optimization. We do not report linear scaling, optimal hardware
selection, or measured end-to-end speedup. Campaign launchers retain historical
paths and guards; they are not portable one-command rentals and require new budget
and host validation before reuse.

## Fresh action-history replication

The [200-state replication](LCFM_REPLICATION.md) uses the **39,687,136-parameter
MetaWorld model**. It evaluates **120,000 candidate sequences** under 35
conditions, giving **4.2 million candidate forecasts** in **14,000 batched
six-step forecasts**. The independent sample is 200 starting states; repeating
candidate sequences across conditions does not create additional states.

Each complete state stays on one GPU, including all conditions and both candidate
banks. The final queue ran on **eight RTX 5090s and two RTX 4090s**. Full excluded
receiving cases, each containing 70 forecasts, take 186–190 seconds on the 5090s
and 223–230 seconds on the 4090s, with **18.38 GB peak allocated device memory**.
These are whole-case timings, including encoding and checks.

Whole-case boundary drains allow the remaining queues to be balanced using
measured throughput. Completed cases keep their original source and runtime
bindings; interrupted attempts remain preserved and retry with the same inputs.
This changes work placement without changing model arithmetic or the scientific
registry. Independent CPU audits and generation-pinned cloud readbacks run while
other GPU cases continue. A slow host's backup route is handled separately so it
does not stall collection from the rest of the fleet.

The final statistical analysis requires every registered case, checks each
original result-file hash, and retains all six layers and matched random
controls. [Frozen execution source](../src/offline_study/lcfm_replication.py) ·
[Boundary drain controller](../scripts/vast/drain_lcfm_case.py) ·
[Case audits and aggregation](../analysis/mechanism/lcfm_replication_summary.py).

All 200 scientific states and their derived records are preserved with verified
cloud readback. The campaign is complete; its GPU instances and volumes have
been released. The proposed LeWM pilot has not started.
