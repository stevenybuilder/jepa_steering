# JAX Scaling Book notes for our JEPA-WM study

Research and code audit: 2026-09-07. This is a learning document and implementation
decision log, not a replacement for [the experiment plan](docs/EXPERIMENT_PLAN.md)
or [the behavioral amendment](docs/BEHAVIORAL_EVALUATION_AMENDMENT.md).
The code is PyTorch; these hardware and mathematical principles do not require JAX.

## September8: overlap independent work, avoid duplicate experiments

At10:18UTC, a further HMM execution optimization is CPU-validated and queued with
a new pre-outcome execution freeze. H1/H2 are unedited; reuse them in the same
forecast before the sole H3 edit instead of computing an entire native H6 shadow.
This retains the same gate, fit, dose and candidate definitions. FullGPU
engineering compares EVERY forecast, gate, posterior and field against the original
two-pass implementation before adoption. If exact parity passes, scientific fullH6
work drops from two backend calls to one; short horizons already used one. This is
an operation-count saving, not a measured2x end-to-end speedup. Extra reference
calls belong only to excluded engineering episodes and remain separately logged.

Execution addition at10:10UTC: DROID fit-input preparation completed entirely on
CPU/network while all15GPUs had other jobs. We fetched1.36GB of specifically needed
left-camera/state/metadata objects, not the entire raw corpus. The prepared coupling
fit encodes only the first observed frame: its native H3 activation/condition
capture needs no encoded future target. Its deployment fields remain GPU-resident,
with one native predictor pass; full receiving-GPU checks still must prove this.

The [roofline chapter](https://jax-ml.github.io/scaling-book/roofline/) gives the
useful lower-bound model `T >= max(FLOPs / FLOPs_per_second, bytes / bandwidth)`.
Our application: reduce unnecessary data movement and repeated model work before
buying peak FLOPs. Whole independent RNG streams require no per-step inter-GPU
collectives. Keep the original300-candidate CEM batch and all evaluation episodes;
GPU count partitions work, not sample size. These are implementation choices and
lower-bound reasoning, not measured end-to-end speedup claims.

Scheduling matters separately from kernel speed. DROID's current-checkpoint panel
is now queued at a verified training-epoch boundary, with optimizer/RNG-preserving
resume afterward. It need not wait for all50training epochs. Checkpoint history
remains required and other seeds continue independently; no trajectory is dropped.

The [GPU chapter](https://jax-ml.github.io/scaling-book/gpus/) distinguishes
compute capacity from memory/network movement. Our application is to partition
whole paired RNG streams across GPUs without collectives and stage immutable
inputs while other experiments run. More workers help only after their dependencies
are ready; the current extra US4090 adds a Push-T candidate lane independently of
its native producer. A completed12-episode reference stream can release its paired
candidate work without waiting for all96 reference episodes. Analysis still needs96.

The small HMM fitting problem already has local512×6×400 native features/task.
It took1.90seconds for Reach and1.54seconds for Reach-Wall on the CPU, including
four family-disjoint fitting folds and the final fit. Moving that tiny job to a
busy remoteGPU would add staging/scheduling overhead. Neural-network rollouts stay
on the GPUs; fitted gate tensors are staged once and evaluated on whole CEM batches.
These are measured CPU fitting times, not estimated GPU rollout speedups.

The constant HMM gate is the same refined fixed edit. Subject to full source,
stimulus, receiving-device and whole-planner equivalence, reuse native/constant/
random-constant records from the running panel:576 duplicate episodes avoided,
768 genuinely new paired routed episodes remain. Do not relabel the old raw files
or claim their one-pass runtime includes the extra HMM native shadow. The original
v2 HMM implementation pays for a separate native shadow to isolate history
from intervention feedback; that reference cost remains visible. The v3 same-pass
implementation above requires exact numerical equivalence first. Additional
precision/kernel benchmarks are not prerequisites for completing these comparisons.

## September8 execution lesson: move only the model's actual inputs

The Push-T archive contains unused precomputed tokens and thousands of training
videos not read by the current native planning job. Copying its entire2.785GBarchive
over a measured~4MB/s peer link delayed an otherwise ready GPU. The source archive
was already checksum-verified and completely extracted onIndiana, so we transferred
only the30 unchanged files actually read:333,854,917bytes before transport compression.
These include all21validation videos,full state/action/velocity/length tensors and
the one excluded fitting-scenario video. The released model was already onreceiver.
Every receiving file was checked against the full source manifest;96total planning
episodes,source-family accounting,stimuli andCEM budget did not change.

For a serial input dependency,`T_transfer >= bytes_sent / effective_network_bandwidth`.
This is a lower bound,not a measured end-to-end speedup: serialization,compression,
verification and startup also cost time. Extra idle GPUs do not eliminate a serial
input dependency. Once inputs/checks are ready,independent task/seed/whole-RNG-stream
jobs can run concurrently without changing model arithmetic or duplicating episodes.
This is an operational application of the book's compute-versus-data-movement
reasoning,not an additional optimization benchmark or a smaller experiment.

## New decision: remove repeated response forecasts with a separately tested method

September 7, late evening EDT: implement the user's **Design 1** in
[Rank Edit.md](Rank%20Edit.md), specified in [the successor contract](docs/FIXED_RESPONSE_RANK4.md).
This is a new approximation, not an exact optimization of the old frozen intervention.

The main opportunity is reducing total work: compose a representative response solve
with the error readout offline. Deployment uses a three-score projection, a 4x4 map,
and a four-direction expansion in the existing forecast. No 102,400-square matrix,
32 repeated online probes, extra native rollout, or cross-GPU communication is needed.
The factors stay on the GPU and the existing candidate population is processed in a
batch. Memory reuse and matrix shape matter as well as nominal FLOPs, consistent
with the [GPU chapter](https://jax-ml.github.io/scaling-book/gpus/).

For field width D=102,400, projection plus expansion costs approximately
`D*(3+4)=716,800 multiply-accumulates/candidate`, excluding centering, normalization,
hook and logging work. Their FP32 factors occupy `4*D*(3+4)=2,867,200 bytes`
(2.734 MiB); mean, scales, maps and the separately stored random control add memory.
These are shape calculations, **not measured latency or a 35x speedup claim**.

The fitting cost must amortize: `N > T_fit / (t_old - t_new)` when `t_old > t_new`,
using matched hardware and a consistent unit of work. A cheaper map is useful only
if its task behavior and runtime justify it. Unit tests establish one backend call;
real calibration, full-planner timing and new efficacy evaluation are still required.
We do not reduce 96 episodes or the planner's candidate/iteration counts to obtain
the saving, and we do not equate GPU utilization with useful scientific throughput.

## The three ideas to remember

Execution note, September8: the fixed map now has real calibration and full-CEM
engineering measurements. On the original single4090 engineering scenario,
native/new-map episodes took314.28/313.03seconds. This is feasibility evidence for
a different operator, not population-wide latency equivalence or preserved old
efficacy. New corrected offline measurements and separately frozen behavioral
comparisons are required and are being collected.

The eight-GPU behavioral worker partitions whole12-episode RNG streams across
devices, with no collectives and no duplicated scientific episodes:96 remains
the total per task/condition. A CPU coordinator waits for complete receipts before
analysis. Runtime transport was a real serial bottleneck: four disjoint checksum-
resumable file streams finished the partial copy without changing any scientific
arithmetic. Deployment/setup time counts against time-to-results, even when GPU
FLOPs are plentiful. Optional optimization benchmark studies remain deferred.

1. Find what actually limits completion: model arithmetic, memory traffic, CPU
   preparation, storage, or a prerequisite that has not finished.
2. Reuse work and batch independent work before distributing one computation
   across machines. More GPUs help most when there are ready independent jobs.
3. Optimize the same experiment. Faster results from fewer episodes, different
   precision, or a weaker planner are not automatically an equivalent replication.

## Concepts mapped to decisions

| Concept | Plain-English takeaway | Application and current status |
|---|---|---|
| Roofline | A faster calculator cannot fix slow delivery of its inputs. | Measure separate stages; baseline timing exists, detailed rank-stage profiling remains to do. |
| Arithmetic intensity | Do more useful arithmetic per byte moved. | Batched windows/arms and response probes are implemented; larger planning chunks need parity tests. |
| Tensor Cores and precision | Some matrix operations have much faster hardware paths. | Corrected offline BF16 primary plus FP32 sensitivity is implemented; planning remains strict FP32. |
| Memory capacity | Weights are only part of the working memory. | Bound probe batches; retain full final CEM populations. |
| Pipeline overlap | Prepare the next batch while the GPU handles this one. | Original baseline supports bounded prefetch; corrected author-validation path is still synchronous. |
| Reuse | The cheapest model call is an unnecessary one we can avoid. | Clip encodings are reused across prefixes; verified local encoder/assets are reused. |
| Independent jobs | Different episodes can run without sharing intermediate tensors. | Disjoint GPU process shards exist; three Wall training seeds are now assigned to separate GPUs, preserving global128 and native updates. |
| Training parallelism | Copies learning one model must synchronize updates. | Global batch and optimizer-update equivalence must be preserved before changing the training layout. |
| Smaller linear algebra | Solve in the small dimension when the algebra permits it. | Sample-space PCA and low-rank response solves already exist. |
| Critical path and cost | Busy hardware is not the same as earlier scientific completion. | Prioritize ready dependencies and verified results per dollar; do not invent work to fill devices. |

“Implemented” here means the named code path exists, not that every runner uses it
or that a controlled benchmark has established a speedup. No new speedup is claimed
by creating this document.

### September8 00:08UTC: scheduling and storage are separate bottlenecks

Two California GPUs finished their coupling jobs around23:46 and were idle until
their next canonical jobs started00:02. Filling those slots restores14/14 assigned
GPUs but would at most yield14/12=1.167x aggregate throughput for equal independent
work; it does not remove the roughly32-probe multiplier in rank-response estimation.
Utilization alone does not establish efficient computation or a defensible ETA.
Report work by task, arm and completed episode, not just rented GPU count.

The first Wall seed completed the native validation/checkpoint-resume proof and
entered epoch2; two other seeds run independently. Gradient accumulation keeps the
authors' global128 update while fitting on oneGPU perseed. This is distinct from
silently reducing the batch or treating three evaluation seeds as trained models.

The laptop disk filled, but remote jobs continued. Bulk snapshots now stream
through bounded RAM to Google Drive and GCS; no extra local tar archive is needed.
Archive readback hashes and per-member hashes gate deletion of local duplicates.
Keep scientific originals, compact result summaries, plans and recovery maps.

## 1. Rooflines: which resource is the bottleneck?

The book models arithmetic time as `F / C` and transfer time as `Q / W`: `F` is
floating-point operations, `C` operations/second, `Q` bytes moved and `W`
bytes/second. With ideal overlap, the slower component dominates:

`T_ideal = max(F / C, Q / W)`.

Without overlap their modeled costs add. Actual time also includes launches,
synchronization, decoding and other overhead. Arithmetic intensity is `I = F / Q`;
compare it with the hardware ratio `C / W`. These are estimates, not a measured
runtime guarantee. [Book: Rooflines](https://jax-ml.github.io/scaling-book/roofline/)

Our application: separately measure input preparation, encoding, response probes,
small linear solves, final forecasts and simulator steps. A memory-bound kernel
and a CPU-starved pipeline need different fixes. High `nvidia-smi` utilization
alone does not establish high useful FLOPs or end-to-end efficiency.

There are several distinct “networks”: accelerator memory access, CPU-to-GPU
transfer, communication between GPUs, and internet dataset downloads. An independent
episode worker does not exchange activations over the internet, but it still needs
its inputs staged and its results preserved. Location affects setup/transfer time;
it does not change the arithmetic inside an already staged model.

## 2. Batching: reuse weights across more work

For an idealized BF16 multiplication `[B,D] @ [D,F]`, arithmetic intensity is
approximately `B*D*F / (B*D + D*F + B*F)`. If `B` is small relative to both feature
dimensions, this is approximately `B`: increasing the batch improves weight reuse.
Here `B` counts matrix rows/tokens, not necessarily episodes. The book's example
hardware thresholds are not universal batch-size prescriptions.
[Book: matrix multiplication](https://jax-ml.github.io/scaling-book/roofline/#matrix-multiplication)

Our application: batch windows and registered arms, and batch finite-difference
probes. However, larger batches also increase activation memory and can change
floating-point kernel choices. Our planner constructs support edits in chunks of
eight candidates, but executes the final forecast on the full population of 300.

This distinction is evidence-driven: the saved fit-only transfer check found that
slicing the final forecast changed visual predictions even without an edit. That
optimization was rejected. Do not silently reintroduce it as a memory workaround.
See [the adapter](src/offline_study/planning_support.py) and
[the transfer evidence](artifacts/offline_study/primary-durable-20260907/planning-selected-transfer-fit-v2/reach-wall-rank4/report.json).

## 3. GPU precision and working memory

GPUs have specialized matrix hardware, general arithmetic units and a memory
hierarchy. Precision, matrix shape and reuse influence which hardware can be used
effectively. A small parameter count alone says little about the memory required
for a large population of imagined trajectories.
[Book: GPU hardware and memory](https://jax-ml.github.io/scaling-book/gpus/)

Our decision: preserve the corrected offline BF16-primary/FP32-sensitivity design.
The current planning adapter is strict FP32 with TF32 disabled. Lower-precision
planning is a possible future engineering comparison, not an already accepted
optimization: small prediction differences can change CEM rankings and later actions.

Track both peak allocated and reserved GPU memory. Leave headroom for the largest
registered shape, not just the average batch. Increasing probe chunk size is a
candidate to benchmark, not a permission to remove probes or reduce the population.

## 4. Overlap preparation with GPU execution

Our systems application of the book's overlap model: if CPU preparation takes
`t_cpu` and GPU work takes `t_gpu`, a serial batch costs roughly their sum.
A properly buffered steady-state pipeline can approach their maximum, subject to
transfer costs and contention. Pipeline fill/drain still costs time.

[The original baseline](src/offline_study/benchmark.py) uses
[bounded prefetch and background cache writing](src/offline_study/pipeline.py),
and pins collated CPU tensors on CUDA runs. One producer owns the dataset object.
This is not a claim of fully asynchronous CPU-to-GPU transfers on every path.

Important gap: [the corrected author runtime](src/offline_study/author_runtime.py)
currently decodes and encodes synchronously. It already decodes each selected row
once and reuses clip encodings across prefixes, but it does not use that prefetcher.
Porting prefetch is pending, not deployed. Before adoption, compare exact batch
contents/order and RNG behavior on permitted fit data, propagate decode failures,
bound memory, and measure the complete pipeline. It is a secondary priority if
response forecasts, rather than decoding, dominate the active job.

## 5. Cache only quantities that genuinely remain unchanged

Existing implementations:

- [Author clip encoding](src/offline_study/author_runtime.py): encode a clip once,
  then reuse its features for its valid rollout prefixes.
- [Model loading](src/offline_study/model_loader.py): use the audited local DINOv2
  source/weight cache; omit the unused visualization decoder, not the predictive model.
- [Asset staging](src/offline_study/navigation_assets.py): reuse checksum-verified
  immutable downloads instead of downloading them again. Check actual expanded
  archive size and disk headroom; compressed size is not working-set size.

Potential extension, not implemented: cache frozen visual-encoder outputs during
training. First prove that preprocessing is identical and any stochastic transform
is handled correctly. A valid cache identity must bind input, encoder weights,
transform, precision and relevant randomness. Do not cache trainable embeddings,
or reuse candidate-dependent responses after CEM has changed the candidate actions.

New engineering path, not yet adopted in scientific runs:
[support-prefix memoization](src/offline_study/support_prefix_cache.py) reuses
identical H1/H2 predictor calls within a single H3 response construction. This
applies the [inference chapter's prefix reuse idea](https://jax-ml.github.io/scaling-book/inference/#prefix-caching),
but it is whole-predictor memoization, **not** a JEPA attention KV cache. It checks
tensor values, shapes, strides, dtype and device, snapshots without aliasing,
preserves predictor-call hooks, and never reuses across changed CEM populations.

For four eight-probe chunks, the first computes both prefix steps and the remaining
three may reuse them: six of 24 probe predictor calls avoided (25% of calls, **not**
necessarily 25% of FLOPs or time; context lengths differ). The native shadow, all
32 logical probes, common rank eligibility, small solves and full final forecast
remain. Equality checks/copies cost time too. Unit tests pass; the fit-only GPU
benchmark compares all support fields and final forecasts/energy bitwise on
8, 19 and 300 candidates, with alternating timing order at 300. That benchmark
passed (measurements below). Full-CEM integration remains pending; existing
running jobs are unchanged.

## 6. Use more GPUs for independent ready work first

The book distinguishes data, tensor, pipeline and fully sharded parallelism;
they differ in what they split and what they communicate. Network bandwidth and
per-device work determine whether the split helps.
[Book: training parallelism](https://jax-ml.github.io/scaling-book/training/)

Our frozen-model evaluations are simpler than synchronized training: separate
episodes or trajectory shards can each keep a complete model on one GPU, with no
gradient communication. [The shard runner](src/offline_study/multigpu.py) verifies
non-overlapping IDs and exact coverage. Extra windows are not extra independent
trajectories, and repeated late checkpoints are not independent training seeds.

For future history evaluation, the natural job key is
`task / training-seed / checkpoint / condition / scenario-shard`.
The required five-task target of three training seeds, ten late checkpoints and
96 episodes means 2,880 episode evaluations per task/condition, or 14,400 across
five tasks. This is workload arithmetic, not an achieved sample size or ETA.
The histories and executable training/evaluation schedule are still pending.

Do not split an episode across machines merely because machines are available.
Within an episode, future observations depend on executed actions, and CEM
iterations depend on earlier elite selection. Other episodes can proceed in parallel.

## 7. Training speedups must preserve the optimizer experiment

For synchronous data-parallel training with equal microbatches,
`B_global = GPU_count * B_micro * accumulation_steps`.
This identity is our implementation accounting, not evidence that arbitrary
microbatching reproduces every model's training behavior.

Before changing physical GPU count, preserve sample weighting, the effective
batch, update count, loss normalization, RNG semantics, gradient clipping and
learning-rate/weight-decay schedule. Accumulation should not advance the scheduler
once per microbatch when the reference advances it once per optimizer update.
Matching the product alone does not guarantee equivalence for stochastic or
batch-dependent operations.

The released configurations imply global batches of 256 for MetaWorld/Push-T and
128 for Maze/Wall. The upstream trainer does not currently provide the validated
microbatch-accumulation adapter we would need to assume flexible equivalence.
Training-layout changes therefore remain pending validation. Independent training
seeds are an attractive parallel axis, provided each run fits and preserves its
own configured training budget. See the
[pinned training configs](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/configs/vjepa_wm).

## 8. A relevant math trick: work in sample/rank space

This is a connection from our code to the book's minimize-work/minimize-traffic
view, not a claim that the book prescribes our PCA method.

For centered features `X` with `N` samples and `D` features, when `N << D`, form
`G = X @ X.T` rather than the much larger `X.T @ X`. Recover a feature direction
from a positive-eigenvalue eigenpair as `v = X.T @ u / sqrt(lambda)`.
[Our PCA implementation](src/offline_study/support_operator.py) uses this sample
Gram matrix, rejects deficient ranks and fixes basis signs. Gram methods can worsen
numerical conditioning, so this is not a universally preferable replacement for SVD.

The response correction similarly solves a regularized rank-by-rank system, not
a dense feature-by-feature inverse. But **rank four does not mean four model calls**.
The frozen common-degeneracy rule retains learned and random rank-eight probes,
each with positive and negative perturbations: 32 probe rollouts per candidate.
For 300 candidates, that is 9,600 H6 probe rollouts before other work. The expensive
part can be estimating responses, not solving the small linear system.

## 9. Measure the critical path, not a utilization screenshot

An additional systems lens is Amdahl's law:
`speedup(N) = 1 / ((1-p) + p/N)` for parallel fraction `p` under ideal assumptions.
For an illustrative 80% parallel fraction, eight workers give 3.33x, not 8x.
This is an example, not a fit to our project. Setup, data preparation, validation
and final aggregation can limit the study even when an individual GPU job scales well.

Our verified fit-only Reach-Wall check measured a 300-candidate native H6 forecast
at 2.875 seconds and the rank4 path at 101.793 seconds (about 35.4x the time).
It used 52 distinct action sequences and is explicitly **not actual complete-CEM
episode throughput**. The report's SHA256 matches its DONE receipt:
`c9c20732f474ce54f35aa2ab6fa8bc968da5ea5a1e82caa4b15eaad1cd4357c8`.
[Evidence](artifacts/offline_study/primary-durable-20260907/planning-selected-transfer-fit-v2/reach-wall-rank4/report.json).

Decision: prioritize profiling the rank response construction and its allowed
batching/reuse opportunities. Do not infer that moving the already GPU-resident
small solve “to GPU” will remove the dominant cost. Do not claim a new speedup
until the same fixed workload passes parity and completes faster.

## 10. Rent for useful throughput, not the most impressive GPU name

Our decision metric is `cost per valid episode = dollars/hour / valid episodes/hour`,
alongside time to the last required result. Include startup, staging, simulator CPU,
memory, bandwidth charges, retries and durable result copying. Compare the actual
intervention workload, not just the unsteered model or vendor peak FLOPs.

The latest user ceiling is $7/hour aggregate and US-only. More instances are useful when
they can immediately run independent required work with data and dependencies ready.
Avoid tensor-parallel communication across rental hosts unless measurement justifies
it. CPU simulation and metadata checks are useful CPU work; forcing them onto GPUs
without a validated equivalent implementation can slow the study or change it.

Update after the user's sixth-task authorization: DROID is now in scope and
RoboCasa is excluded. DROID needs a fresh profile: its larger DINOv3-L/12-layer
model is not our small six-layer model. Its evaluation and history counts also
differ; do not extrapolate five-task timings or use a blanket 96-episode/ten-epoch
rule. See [the DROID contract](docs/DROID_METHOD_ALIGNMENT.md). Its required
released rebuttal aggregation allows sparse checkpoints in epochs [215,315],
explicitly avoiding 100 separate checkpoint evaluations. Native cadence gives
3x18x64 = 3,456 episode evaluations per condition, alongside the five simulated
tasks' 14,400 paper-primary evaluations. This is author-supported workload
clarification, not a numerical optimization or a reduction selected after results.
These are intended work counts, not independent n.
[Official model table](https://github.com/facebookresearch/jepa-wms#-pretrained-models),
[paper aggregation rules](https://arxiv.org/html/2512.24497v4#A7.SS2).

## Decision log and next benchmark requirements

2026-09-07: audited the existing implementations and recorded the distinctions
above. This change adds documentation; it does not hot-patch running workers,
alter frozen protocols, launch new tasks or demonstrate a new performance gain.

Next candidates, in priority order:

1. Profile response construction versus final forecasting on the already excluded
   engineering fixture; preserve all 32 probes and the full final population.
2. Test larger construction batches or exact invariant reuse only if profiles
   justify them; verify edits, identity paths, CEM actions, memory and complete timing.
3. Prepare multi-seed training with fixed effective settings; measure a real update
   before estimating all histories. Do not substitute smaller scientific budgets.
4. Port bounded prefetch to the corrected/new-task path if measured CPU waiting is
   material. Feature caching and compilation remain candidates, not deployed changes.

For each adopted optimization, append: date, hypothesis, code/commit, reference
and candidate receipts, identical workload definition, numerical checks, wall-time
and peak-memory results, speedup and limitations. Retain failed attempts. Benchmark
on permitted fit data or excluded engineering fixtures, not untouched confirmation.

Reading order: [Rooflines](https://jax-ml.github.io/scaling-book/roofline/),
[GPU hardware and networking](https://jax-ml.github.io/scaling-book/gpus/), then
[training parallelism](https://jax-ml.github.io/scaling-book/training/).
For this study, the useful lesson is to reason about arithmetic, bytes, dependencies
and scientific equivalence—not to transplant a large-language-model sharding recipe.

## Measured engineering updates: 2026-09-07, 19:14 UTC

### Exact response-prefix reuse

On the RTX PRO 6000 Blackwell Server worker, same strict-FP32 Reach-Wall fit fixture,
same frozen BF16-fitted operator bank, all 32 probes and final 300-candidate batch:

| Timing order | Original seconds | Cached seconds | Original/cached |
|---|---:|---:|---:|
| Original then cached | 89.8318 | 78.2406 | 1.1481x |
| Cached then original | 89.8393 | 78.2316 | 1.1484x |

Approximately **12.9% less elapsed time**, not a 25% measured runtime saving.
All-arm support tensors were bitwise equal at the eight-candidate construction
size; final visual/proprioceptive forecasts and realized/requested energy were
bitwise equal at 8, 19 and 300. No model parameters or RNG states changed.
This is one permitted fit fixture, not a task-success result or a complete CEM
episode timing. The benchmark keeps reference outputs for equality checks, so its
peak-memory differences include retained comparison tensors, not only cache cost.
Do not compare these timings directly with the earlier RTX5090 results as a
controlled hardware test. Full-CEM integration is still required before adoption.

[Receipt](artifacts/offline_study/primary-durable-20260907/rank-prefix-check-20260907-v2/report.json),
SHA256 `a69f64197992cc8cefcd707e10f4625e88cc2f1dbd6c9e371e76fcabcb86dd9c`.

### Read and transform only needed training frames

[SelectedFrameSlicer](src/offline_study/navigation_input_check.py) memory-maps the
uncompressed PyTorch observation archive and selects the four required frames
before normalization/cropping. It keeps all 20 elementary actions concatenated as
the native slicer does. It resolves every nested trajectory-subset index explicitly;
using the subset's forwarded `get_frames` blindly would pick the wrong trajectory.

The first eight native shuffled training clips per task, each tested in two timing
orders, passed exact pixel/action/state/reward and post-transform RNG equality.
Measured aggregate loader speedup was **11.305x Wall** and **34.987x PointMaze**.
Only 16 small CPU clip comparisons per task were timed, with filesystem/cache
effects; these are not stable whole-dataset throughput estimates or training
speedups. Theoretical transformed-frame reduction is 50/4=12.5x for Wall and
100/4=25x for Maze, but mmap/page traffic and allocation also affect elapsed time.
This combines less arithmetic with less memory/I/O traffic; it does not require
moving the CPU loader to a GPU. Confirm actual update throughput in the training
pilot before extrapolating a history ETA.

The same metadata audit verified 1,920 Wall rows (1,728 train / 192 validation) and
2,000 Maze rows (1,800 / 200), using the native seed-234 90/10 split. No exact
initial-state fingerprints crossed the split. Uniqueness of these fingerprints
alone does not prove broader lineage independence. Native training-clip counts
are 53,568 and 145,800; these overlapping clips are not independent trajectories.

[Receipt](artifacts/offline_study/primary-durable-20260907/navigation-input-check-20260907-v1/report.json),
SHA256 `9eb6a9a3ef9f059ef20d42fb660eea24ac63a696179b0e002aa735ba02da8aa8`.

### Storage is a capacity dependency, not a GPU workload

Keep raw datasets and future histories on suitable US storage; copy compact
verified receipts locally. At this check the laptop has approximately 14 GiB free,
so the roughly 90 GB navigation extraction stays remote. Do not fill the laptop
or rent a GPU solely to hold files. The 96 GB worker was rented for larger-model
DROID computation and useful engineering/training, with storage as a supporting
resource. Use cloud/object storage only when needed and account for its cost.

## Execution lessons: 2026-09-07, 20:23 UTC

**Critical path:** the completed full 100-step rank/combined engineering episodes
took 7,681–7,867 seconds each on RTX5090. The fixed coupling-only checks, including
their fit-only parity checks, took 313–316 seconds. These are different algorithms
on one engineering scenario, not a controlled hardware ranking or an efficacy
comparison. The roughly13% prefix-cache saving above cannot by itself erase a
roughlytwo-hour rank episode. Keep the cheaper independent comparisons running
while validating response-construction improvements; do not silently remove probes.

**Ready independent work:** added a California4xRTX5090 worker at$1.641667/hour
for the remaining fixed coupling/control logical streams. The projected full
account rate is$6.585111/hour. Retaining eight logical RNG streams and unique
episode IDs makes these disjoint shards portable without inter-GPU gradient or
activation communication. Hardware-specific parity still precedes production.
Runtime preparation is real overhead: the initial SSH stream was interrupted,
and source rsync was an empty non-executable placeholder. Keepalive-enabled
direct streaming succeeded. No speedup should be inferred from rented GPU count
while the worker is still being prepared.

**Memory capacity versus price:** DROID's two full native engineering plans used
15.1GB peak allocated memory on the96GB worker (about23GB reserved). This indicates
that baseline evaluation can fit32GB GPUs; it does not prove that a cheaper GPU
has better throughput. The96GB worker can be useful for training/engineering but
is not needed merely to store the recordings or load the DROID baseline.

**Training accumulation, pending validation:** the prepared pilot uses the exact
native16 logical samplers, batches of8, and global128. It compares accumulation
against the actual upstream training function's local gradients averaged across
those16 batches, including optimizer moments, scaler and CPU RNG. The source
collective's floating-point reduction order is not claimed identical. This follows
the book's separation of effective batch from physical device layout, but remains
a queued engineering test, not an adopted training speedup.
[Training parallelism](https://jax-ml.github.io/scaling-book/training/).

**Real storage fallback:** local disk reached less than300MiB free. Six duplicate
raw fitting captures totaling8.76GB were relocated only after checksums matched
their prior receipts and two existing US copies. Small reports and recovery maps
remain local. No extra GPU was bought as a disk. Storage relocation changes the
location of evidence, not the sample size or experiment design; the historical
full-local verification must not be presented as a current claim that every raw
capture remains on the laptop. See [execution status](reports/EXECUTION_STATUS.md).

### Follow-through at 20:55 UTC

All 13 GPUs are executing useful evaluations. The second worker's first completed
candidate episodes passed cross-host initial-state/goal pairing checks; 64 total
candidate episodes were verified without inspecting success outcomes for selection.
CPU simulator setup and the fixed analysis implementation proceed alongside them.

The training pilot now explicitly checks the unchanged CUDA RNG as well as the
CPU streams. Its arithmetic is `g = (g_0 + ... + g_15) / 16`, with eight examples
in each logical microbatch and **one** optimizer/scheduler update per global128.
This keeps the effective training batch fixed while changing physical scheduling;
it is not automatically faster than data parallelism, and the GPU pilot must
measure that. No three-seed training history has completed yet.

The practical completion-time model is `staging + queueing + slowest shard + analysis`.
Consequently, preparation happens while evaluations run, and the next GPU jobs are
queued behind explicit process-exit checks. GPU utilization alone does not measure
useful throughput. These are applications of the book's compute/memory/network
balance and training-parallelism reasoning, not permission to change scientific
precision, sample counts or CEM budgets.
[GPU rooflines](https://jax-ml.github.io/scaling-book/gpus/),
[training parallelism](https://jax-ml.github.io/scaling-book/training/).

A second disk-full fallback relocated42 large window-metric duplicates (1.71GB)
only after verifying original and second US copies against historical receipts.
No original data or unrelated local files were deleted, and no GPU was rented as
storage. Local receipt downloads work again; large assets stay remote.

### Measured follow-through at21:42 UTC

Wall's five-update GPU pilot passed exact local-reference gradient-mean,
parameter/optimizer/scaler and RNG checks. Updates2–5 cost1.374–1.678 seconds
on the96GB GPU. With418 updates/epoch and50 epochs, multiplying those four timings
by20,900 gives roughly8.0–9.7 GPU-hours per training seed, **before** checkpoint
I/O, validation/planning, any slowdown and engineering. This is an extrapolation
from four timed updates, not a completed run or a three-seed end-to-end ETA.

The larger batch experiment illustrates why memory capacity is not throughput.
Keeping all32 rank-response probes and all300 CEM candidates, chunks16/32 used
more memory but did not consistently beat chunk8. They also changed edit fields
and forecasts, failing the fixed exact acceptance gate; neither was adopted.
Increasing arithmetic intensity can help only until another resource limits
the operation. The empirical roofline matters more than a full memory gauge.
[GPU compute and memory balance](https://jax-ml.github.io/scaling-book/gpus/).

DROID64 completed, and the exclusive queue automatically advanced through the Wall
pilot and PointMaze engineering. The next96-episode PointMaze native job is running.
One additional US4090 was rented for ready navigation offline/fit work. The first
offer's US label conflicted with a CN IP and was cancelled before research transfer;
its Indiana replacement brings projected aggregate to$6.928815/hour before bandwidth.
To reduce staging latency and disk needs,
initial transfer includes only checksum-bound fitting/validation videos plus the
existing runtime, not every raw training video or archive. This is less input
movement, not fewer evaluation trajectories. No GPU was rented solely for storage.

### Measured follow-through at22:41 UTC

The Indiana4090 completed all native navigation offline baselines and all20
category/task/precision fits, and now runs the unchanged intervention comparisons.
Native Wall validation cost129s BF16 /121s FP32 for4224 prefixes; Maze687s /658s
for24400. Those are complete measured baseline runtimes on this worker, not
intervention runtimes or evidence that FP32 is always faster. Precision remains
predeclared: BF16 primary, FP32 sensitivity; we do not choose it from results.

Only128 fit plus the complete validation videos were staged initially. The resumed
direct US-to-US transfer handled23.115GB logical input/runtime data but sent3.559GB
wire bytes through reuse and compression. This illustrates lowering the network
term in `T_total = staging + queueing + compute + verification`; it does not reduce
the scientific population. Interrupted transfers now resume with rsync, without
deleting destination files or copying large datasets through the laptop.

As MetaWorld native jobs finished, GPUs were reassigned to the fixed rank/control
panel and Wall simulator reference. Fourteen GPUs now serve independent jobs at
about$6.93/hour before bandwidth. Intact logical RNG streams—not overlapping
episode fragments—are scheduling units. Each completed12-episode rank stream is
only one eighth of a96-episode condition, and final analysis still requires all
streams and controls. Parallel hardware hides independent work; it does not make
the many response-probe forwards in a rank intervention disappear.

At22:50, one GPU was reassigned from a partial Reach rank/combined run to a bounded
expert-goal reproducibility diagnostic after five paired goal-image hashes differed.
No candidate outcomes were used to choose that diagnostic. This is useful
verification work, not model-throughput work; the other13 evaluations continue.
Keep throughput reporting subordinate to correct stimulus pairing. A completed
fast run is not valid comparative evidence when the required pairing check fails.

### Measured follow-through at23:23 UTC

The goal-image discrepancy was reproduced and the exact original pixels recovered;
canonical delivery then passed24 real source-environment checks. Three expensive
streams restarted with the verified stimulus delivery, and input-only whole-stream
repair jobs are queued behind the existing coupling runs. This verification costs
time, but avoids treating unpaired inputs as valid comparison evidence. No sample,
planner budget, precision or statistical threshold was changed to improve speed.

Completed navigation scopes now get CPU-only hash/coverage/statistical analysis
while the dedicated GPU executes the next scope. The analyzer accepts the declared
8/32 physical shards while requiring the same complete trajectories and arms.
This overlaps independent work; it does not run two experiments on one GPU.

Staging Wall's complete data to the existing California worker transferred
58.964GB logically but only432.258MB over the wire using compression. No laptop
dataset copy or new rental was needed. That reduces preparation latency for the
authorized training seeds; it is not evidence that training has started. The
general lesson remains to measure the whole critical path, including I/O and
verification, instead of using GPU occupancy alone as a speed metric.
[GPU resource balance](https://jax-ml.github.io/scaling-book/gpus/).

## Training: preserve the experiment when changing the GPU allocation

PointMaze uses the authors' effective batch `G = R × b = 16 × 8 = 128`.
The existing accumulation implementation combines the 16 logical microbatches
before one optimizer/scheduler update: `g = (1/16) Σ g_r`. This keeps batch size,
loss, update count and optimizer schedule fixed when only one physical GPU is
available. It does **not** make one GPU as fast as sixteen, or reproduce an
undocumented hardware reduction order. Actual-upstream gradient/optimizer/RNG
parity is a gate, not an assumed speedup. Independent training seeds can run on
different available GPUs without exchanging gradients between seeds.

Data movement also matters: PointMaze's raw archive is about 0.72 GB compressed but
30.1 GB expanded. Transfer the pinned compressed archive once to a worker, verify
existing files, and add only missing training inputs. Keep active input bytes
unchanged. This is an execution choice, not a reduction in trajectories, clips,
validation batches or planning episodes. No new optimization experiment was added.

These choices apply the book's distinction between compute, data movement and
communication costs. See [the GPU/data-parallelism discussion](https://jax-ml.github.io/scaling-book/gpus/).

## September 8: parallel DROID execution without changing the sample

The prepared DROID panel has nine arms and 64 total episodes per arm. Four GPUs
receive two of the eight persistent RNG streams each: `64 / 4 = 16` episodes per
arm per device, not 64 per GPU. Native and candidate episodes for a stream stay on
the same physical GPU. No gradient synchronization is required for these frozen
models. A rough lower bound is `T ≈ max_g(work_g / measured_throughput_g)`, plus
staging, engineering and final analysis. Four devices do not guarantee exactly
four times the speed; hardware-specific timings and the longest shard determine
the actual finish time. Receiving full-planner engineering completed on all four
devices in900–919 seconds each. This is measured verification cost, not the full
576-episode scientific panel time or a matched one-versus-four-GPU speed benchmark.

Hardware changes can also change floating-point results. The new freeze therefore
collects a paired native reference on each receiving device and requires exact
same-device repeat/zero identity. It does not silently reuse a different GPU's
native outcomes or pool an incomplete earlier panel into the new comparison.

Storage is a separate bottleneck. A verified 8.79 GB DROID input relay and a
1.36 GB PointMaze checkpoint/input relay use CPU/network and spare disk on an
already owned worker, without another GPU storage rental. A stalled Drive upload
was retried over IPv4 with 8 MiB chunks: the 844.5 MB archive uploaded at roughly
14 MiB/s and passed full readback. Because both connection family and chunk size
changed, this is an operational recovery measurement, not an isolated causal
benchmark of either choice. The source files and failed attempt were preserved.

This applies the book's compute-versus-data-movement distinction to the complete
pipeline, while leaving the scientific samples, CEM budgets and doses unchanged.
See [the roofline model](https://jax-ml.github.io/scaling-book/roofline/).

### Remove unnecessary queue dependencies, not scientific comparisons

The HMM panel reuses three completed arms: native, refined edit and its random
control. Its first scheduler nevertheless waited for two additional coupling
arms—384 episodes—that are not HMM inputs. The planned priority handoff removes
that scheduling edge: finish the current complete stream, run unchanged HMM
verification/evaluation, then resume the other controls. All comparisons remain
required. This is critical-path scheduling, not a FLOP reduction or a license to
skip controls. It can bring the HMM answer forward without another rental; the
total panel work and full-study completion time need not fall by the same amount.

Likewise, staging compressed PointMaze data and its separate pinned runtime on
CPU/disk while DROID uses the GPUs overlaps preparation with useful computation.
The upper bound on benefit comes from the work that can actually overlap; it does
not turn an I/O-bound preparation step into useful GPU work. Keep the completion
latency and verified samples/hour alongside GPU utilization when measuring speed.
