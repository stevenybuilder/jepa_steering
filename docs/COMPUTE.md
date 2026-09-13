# GPU execution: scaling principles for inference

This was PyTorch inference plus simulation, not distributed training. The
[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
and [Scaling Book GPU chapter](https://jax-ml.github.io/scaling-book/gpus/) informed
how we separated compute, memory and communication constraints. They are systems
references, not libraries used by the executor.

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
