# JAX Scaling Book notes for our JEPA-WM study

## September8: measured combined-edit cost and receiving-runtime gaps

The refined combination now has real-GPU equivalence/timing evidence, not just
an operation-count estimate. At300 candidates/H6 on Texas RTX4090, three
synchronized forecasts averaged3.5701seconds combined versus3.1629seconds
unsteered: about12.9% overhead in this engineering check. The standalone fixed
map averaged3.1733seconds. All18 required cases matched an independent reference
exactly, including native-feature coefficients, doses and RNG; model bytes stayed
unchanged. These are forecast timings, not full-episode throughput or efficacy.

The gap was unnecessary native replay: a naive two-pass implementation would
execute72 predictor blocks to recover a native feature and then apply the edit.
The implemented version shares H1/H2, recomputes only the four native H3 blocks
needed for that feature, and then continues the coupled forecast:40blocks.
It also repeats that prefix's input embedding work. The fixed response map has
no online finite-difference probes or full native shadow. This applies the
book's work accounting: a device-time lower bound is
`max(FLOPs / compute_throughput, bytes / memory_bandwidth)`; reducing repeated
work helps, but block counts alone are not measured wall-clock speedups. We did
not time the full two-pass reference as a throughput baseline, so no1.8x speedup
claim follows from72/40. [Roofline chapter](https://jax-ml.github.io/scaling-book/roofline/).

A separate operational gap delayed the full-planner handoff: the navigation
environment worked for PointMaze but omitted MetaWorld EGL settings; the Texas
image also lacked generic GLVND dispatch libraries. Imports alone missed this.
The first attempt stopped before any episode. The correction adds an actual
official-reset/render preflight, explicit EGL/device selection, and three
private hash-bound dispatcher libraries from an already working same-Ubuntu
worker. Other active jobs and host NVIDIA drivers remain untouched. This is
dependency preparation and idle-time reduction, not a model-math speedup. Future
handoffs should test rendering while predecessor workloads are still running,
subject to device ownership, instead of discovering missing libraries afterward.

Archive metadata also made receiving and local source digests differ: macOS
AppleDouble sidecars were included in the original immutable archive. We kept
those original bytes, verified every executable source file unchanged, and
bound the new snapshot explicitly. Disable sidecar emission when creating future
archives; never rewrite a completed scientific snapshot to make hashes match.

Evidence: numerical report`c6faf9cd35eb981457269df3876e2226853e3b3fe2d7e8fecc52fe955ee52897`,
all30 numerical/operation JSON files full-readback verified, and the
[combined engineering contract](docs/FIXED_COMBINED_ENGINEERING.md). Full-planner
engineering and then a separately frozen96-total behavioral comparison remain;
neither is satisfied by these numerical/timing checks.

Research and code audit: 2026-09-07. This is a learning document and implementation
decision log, not a replacement for [the experiment plan](docs/EXPERIMENT_PLAN.md)
or [the behavioral amendment](docs/BEHAVIORAL_EVALUATION_AMENDMENT.md).
The code is PyTorch; these hardware and mathematical principles do not require JAX.

## Avoid repeated data preparation — September8,18:22UTC

The combined engineering input now reuses the authors' published MetaWorld
normalization constants instead of constructing a full dataset to recompute
statistics. Only one already-authorized fitting row is decoded. All126 parquet
files are byte-verified, and their footer row counts preserve the original global
row mapping; metadata checks are not additional independent observations.
The selected-row tensors and transform RNG match the actual native reader.
This removes a redundant full-pool state scan and keeps protected data unopened.
It is a new engineering-input freeze, not claimed bitwise equality to earlier
recomputed statistics or a change to either fitted intervention bank.

Keep this on CPU while GPUs execute rollouts: work sharing and data locality
help only if they remove work from the critical path. The first receiving test
attempt missed the existing Python dependency overlay; it failed at import,
before any input decoding or GPU use. Reusing the correct installed overlay
passed44 tests; no installation or active runtime edit was needed. The full
local suite also passed612 tests(two pre-existing skips). These are engineering
checks, not evidence of improved task success.

## Work count versus averaging — September8,17:55UTC

The user's requested late-epoch averaging is a cheap CPU reduction of saved
scores, not a GPU optimization. Its cost is dominated by producing the missing
checkpoint evaluations. Keep the two quantities separate:
`remaining_GPU_hours = sum(remaining_episodes * measured_seconds_per_episode)/3600`,
while wall time is the slowest dependency/worker path, not simply that sum.
[The measured estimate](reports/ETA_COST_20260908.md) records current panel scope,
allocation and assumptions; [the averaging contract](docs/CHECKPOINT_AGGREGATION.md)
requires96TOTAL per simulation condition/checkpoint and does not count GPUs as
samples. A smaller final-checkpoint report is different scope, not faster math.

The combined successor now has an explicit native-prefix engineering adapter:
40 predictor-block executions instead of72 for a full two-pass same-map reference.
This applies work reuse to the shared unedited prefix, preserving the native H3
features. Four extra blocks plus predictor input work remain; one backend call
does not mean zero overhead. Local tests are not a measured GPU speedup, and no
running scientific snapshot was replaced by this adapter.

Operational gap: Indiana's container restarted, removing its navigation process
and the later native32 CPU waiter. All saved evidence survived. Recovery uses
the unchanged source/seeds, a verified native repeat and whole interrupted RNG
stream replay; it does not duplicate partial records in the sample. That recovery
is now active. The old native32 waiter is no longer live and was not restarted.
The replacement CPU collector observes transient SSH failures without treating
them as scientific-job failure or permission to restart a model run.

## Next measured dependency — September 8, 16:35 UTC preparation

Activation update after16:50UTC: the isolated CPU waiter is now live
(PID16910/start18343560). All504 local tests passed(two existing skips), and109
receiving CPU tests passed(one unstaged local archival fixture). The complete
receiving test log and nine immutable preparation/launch records passed byte/hash
readback. The GPU check has not run: Indiana's full navigation assignment retains
the device. No active scientific snapshot or planner setting was changed.
Review strengthened the new arithmetic proof to compare represented bytes
(including signed zero) and explicitly set/record/restore native
`cudnn.benchmark=True`, while preserving the registered TF32-disabled policy.
No new speedup or training ETA is claimed by passing these CPU tests.
[Exact activation evidence](artifacts/offline_study/robotics-training-gpu-check-20260908-v1/PREPARATION_READBACK.json).

The32-rank arithmetic/timing check is separate from the live rollouts.
It will follow Indiana's entire180-endpoint navigation assignment, not
interrupt its current child. The existing verified617MB training batch stays on
the same host. This section records preparation, not a launched or passed GPU check.

| Gap | Cause | Correction being implemented | What must not be inferred |
|---|---|---|---|
| No measured MetaWorld/Push-T training ETA. | Native global256 batches differ from navigation's global128; CPU input tests do not measure backpropagation. | One native-vs-accumulated arithmetic check and four timed disposable Push-T updates on the verified batch; preserve32 eight-clip forwards and average gradients by32. | One batch does not prove full-loader, validation/resume or three-seed histories; Push-T timing is not a MetaWorld benchmark. |
| Verification overhead could be reported as model compute. | Rehashing inputs and synchronizing state occupy CPU/transfer time around an update. | Record CPU input-gate wall time separately from synchronized update wall time and CUDA-event intervals, plus peak memory. | CUDA-event elapsed time includes stream gaps and is not summed kernel-busy time or model FLOPs utilization. |
| A future check could contend with current scientific rollouts. | A completed child is not completion of a worker's whole queued assignment. | CPU-only waiter binds the exact navigation launch, all180 endpoints, supervisor exit and stable-empty physical GPU before the bounded child. | A waiting process is not GPU work; a checked queue is not a completed experiment. |

The [training chapter](https://jax-ml.github.io/scaling-book/training/) explains
that parallel hardware and global batch are distinct, and that communication can
limit scaling. Our local accumulation preserves the intended global update but
does not remove its FLOPs; this is a baseline for measured scheduling, not evidence
that one GPU per training seed is optimal. The
[roofline chapter](https://jax-ml.github.io/scaling-book/roofline/) motivates
separating compute from data movement. The
[profiling chapter](https://jax-ml.github.io/scaling-book/profiling/) motivates
measuring actual execution; its TPU tooling is not being presented as PyTorch GPU
tooling. No unmeasured cache, precision or kernel change is enabled by this check.

## Processing validation completed — September 8, 16:16 UTC

The proposed Push-T selected-frame input path now has **real first-batch parity**:
all256 native rank-ordered clips matched the authors' whole-video reader in
pixels, normalized actions/proprioception, raw states, rewards and RNG progression.
The dedicated CPU attempt took116.80seconds including56 receiving tests; the
input comparison itself took84.41seconds. No GPU was used and no validation or
confirmation rows were read. Nine compact records passed local hash readback;
the616.65MB tensor batch stayed on the worker. Full local suite:445tests,2skips.

| Gap | Cause | Applied correction | Limit of the evidence |
|---|---|---|---|
| Push-T training preparation could not safely reuse navigation's selected-frame adapter. | Dataset-specific velocity, relative-action scaling, clip ordering and native transforms needed verification. | A separate training-only reader compares four selected frames and independently normalized modalities against the actual whole-video native path for32×8 clips; all pass. | This is one verified engineering batch. Full-loader worker RNG, histories/resume and CUDA update parity remain required; no training speedup factor is measured. |
| Supporting input-proof files were not rechecked by the model-entry gate. | DONE/report/protocol were bound, but auxiliary comparison/input manifests could disappear or change. | Gate now checks those hashes, original source manifest, exact256 identities, per-sample bytes and RNG continuity. Receiving tests and the real batch passed. | Prevents invalid evidence from authorizing a model check; this is correctness, not a GPU speedup. |
| Preparation could occupy a GPU or move a large tensor batch back to a nearly full laptop. | Treating decoding/storage as accelerator work would consume capacity without needed model computation. | CPU-only, single-threaded low-priority decoding on the existing data host; no dataset download, no GPU allocation, compact receipts only transferred. | The benefit is overlap/avoided transfer; elapsed verification time is not an inference benchmark. |

These apply the [roofline/overlap principle](https://jax-ml.github.io/scaling-book/roofline/)
and [inference work-reuse principle](https://jax-ml.github.io/scaling-book/inference/):
avoid transforming irrelevant frames, but establish equivalent inputs before
changing a scientific path. The native history runner is not silently switched.
[Implementation](src/offline_study/robotics_training_batch.py) and
[receiving proof](artifacts/offline_study/robotics-training-batch-20260908-v1/READBACK.json).

HMM16:13 readback: all72 engineering cases/eight suites passed; all eight GPUs
now run scientific children.23/768 endpoints are verified (no complete24-episode
shard yet). Scientific timings298.27–307.79seconds, median299.27seconds; no current
failures. Lower utilization samples during some observations are not idle leases:
each device still has its verified running scientific child.

## Historical processing follow-through — September 8, 15:57 UTC

Fresh read-only audit: all **17 assigned GPUs on five US instances are busy**;
16 sampled100% and one99%. Account rate$6.732593/hour before usage bandwidth.
Utilization is not a claim of optimal kernel efficiency. Navigation now has
59/128 complete candidate streams;20 new streams passed raw-record verification.
The remaining69 streams give a measured **4.5–5.5-hour navigation-only** estimate
under the current seven-GPU allocation. Individual complete streams took
19.5–23.9minutes; the slowest remaining worker determines the finish time.

HMM has progressed to scientific rollouts:69/72 engineering cases complete,
five of eight full suites verified, five scientific children, three completed
scientific endpoints at15:56:49UTC. The first three memoryless episodes took about
299.25seconds each. `768 / 8 * 299.25 / 3600 = 7.98hours` is an idealized compute
estimate, **not a measured full-panel speed**: other routed arms, remaining checks,
queue overhead and imbalance support only a provisional8–10hour range. The last
192 original coupling controls follow HMM and are additional work. These are not
full-study estimates or reasons to reduce the frozen sample.

The separate native32 model factory now uses the actual pinned constructor-call
AST, with task-specific dimensions and3543/7741updates per epoch. Eleven CPU tests
cover the native configuration, single-visible-GPU guard and failure-path RNG
restoration. No receiving model was built or trained. This removes implementation
duplication with the navigation adapter; it is **preparation, not a measured GPU
speedup**. Native transformed-input, first-update/loader RNG, GPU numerical and
history-resume checks still gate that separate track, not current rollouts.

Scaling Book connection: [parallel training](https://jax-ml.github.io/scaling-book/training/)
distinguishes physical hardware from global batch; [inference](https://jax-ml.github.io/scaling-book/inference/)
distinguishes throughput from latency; [rooflines](https://jax-ml.github.io/scaling-book/roofline/)
explain why changing allocation does not eliminate the work itself. In this
application, `T_finish ≈ max(worker_remaining_work / worker_rate)`, not total
episodes divided by every GPU rented for unrelated work.

Evidence: [live audit](artifacts/offline_study/live-allocation-audit-20260908T1552Z/REPORT.json),
[native constructor](src/offline_study/robotics_training_model.py).

## Historical live handoff check — September8,15:36UTC

All seven navigation workers now have scientific children, including all four
Texas workers after their receiving checks.52/128 candidate streams are complete:
39 preserved plus13 newly verified streams;76 remain, including currently partial
streams. Each stream still has12 episodes and each task/arm96 total. No new
navigation failure was observed. Wall seeds235/236 have23/33 complete epochs;
three corrected PointMaze waiters remain healthy. The HMM eight-GPU allocation
is independent. This is evidence the expanded schedule is doing real work, not a
claim that the full study or all ablations are complete.

After all code additions, the complete local suite ran**419 tests with two existing
skips**. Scientific GPU sources/precision/planners and fixed sample counts were
not altered by the input-verification or archive work.

## Verified execution update — September8,15:12UTC

- **Automatic reuse is happening:** the entire DROID coupling panel finished
  (576 endpoints/72 shards), its frozen analysis passed independent raw-record
  verification, and all four Texas GPUs handed directly to the assigned Wall or
  PointMaze receiving checks. Navigation therefore has seven device assignments;
  Texas did not remain idle waiting for a manual launch. Corrected PointMaze's
  three CPU waiters remain behind the *complete* navigation assignments.
- **HMM validation is progressing, not yet efficacy:**44/72 excluded engineering
  cases complete; all eight devices sampled busy with one verified child each.
  No new failures;0/768 routed behavioral endpoints so far. Engineering timings
  include extra parity passes and are not estimates of scientific throughput.
- **Overlapped data preparation finished without a download:** complete native
  MetaWorld/Push-T raw pools were already on Indiana. The71.61-second low-priority
  CPU audit verified126 parquets,18,718 Push-T files and the original ZIP. It used
  no Torch/GPU/loader calls and did not grant validation/confirmation access.
  Thirteen compact evidence files passed local hash readback. The initial
  verifier's mistaken `shapes.pkl` expectation was preserved as a failed attempt;
  the pinned release actually has `tokens.pth`. This was not a failed experiment.
- **Required training core implemented, not silently launched:** separate32-rank
  code preserves32×8 native microbatches, averaged gradients, individual RNG
  streams and one optimizer/scheduler step.19 new CPU tests pass; native tensor
  parity, receiving-GPU numerical proof, monitoring and resume orchestration are
  still required. Complete local integration:401 tests, two existing skips.

### Further gaps, causes and applied fixes

| Gap | Cause | Scaling Book connection | Actual fix and limitation |
|---|---|---|---|
| Full training-input availability was unclear despite a small recovered planning bundle. | Planning recovery inputs were being used to describe training readiness; those are different populations. | [Rooflines and overlap](https://jax-ml.github.io/scaling-book/roofline/): avoid unnecessary transfers and keep CPU/I/O preparation off the GPU critical path. This is an engineering application, not a new theorem. | Located and byte-verified existing complete raw pools during ongoing navigation. No dataset retransfer. Pixel/action/preprocessing parity still needs its own test. |
| The navigation training adapter could not directly reproduce MetaWorld/Push-T updates. | Navigation uses16 logical ranks/global128; these author configs use32×8/global256 and task-specific sampler policies. | [Training math](https://jax-ml.github.io/scaling-book/training/): distinguish effective global batch from physical devices; accumulation trades parallel hardware for sequential local work. | Separate CPU-tested32-rank receiving core uses `g = sum(g_r)/32` and one optimizer/scheduler step, with independent RNG streams. This preserves the requested update design without requiring32 physical GPUs, but does not reduce its FLOPs or prove exact unpublished DDP reduction order. No receiving GPU proof or full history launch claimed. |
| A completed offline comparison still appeared unfinished in the status table. | Aggregate verification and the human-readable inventory had fallen out of sync. | [Profiling/measurement](https://jax-ml.github.io/scaling-book/profiling/) motivates measuring real work; completed-result inventory is the scheduling analogue. | Reverified all32 geometry shards/four existing aggregates and corrected coverage. Preserve completed evidence; do not rerun it. This saves a needless rerun, not the cost of missing behavioral geometry experiments. |

Source/evidence: [raw-byte verifier](src/offline_study/robotics_training_inputs.py),
[receiving core](src/offline_study/robotics_training_pilot.py),
[input readback](artifacts/offline_study/robotics-input-verification-20260908-v2/READBACK.json),
[explicit remaining training gates](reports/NATIVE_32RANK_HISTORY_READINESS.md),
[corrected ablation coverage](reports/ABLATION_COVERAGE_STATUS.md).
The provenance skill was used only to audit existing completion/evidence, not to
choose methods or authorize scientific execution.

Preservation update15:23UTC: DROID's714 new immutable members (549,338 compressed
bytes) are uploaded privately, with the prior archive chain and four provenance
files. The connected Drive uploader succeeded after shared-client CLI quota
failures. A direct authenticated file-ID read then verified Google's SHA256 and
all downloaded archive/member bytes against the local manifest. No credentials,
sharing settings or GPU workload changed; no data was deleted. This is a storage
reliability fix, not a model throughput improvement. Exact
[archive](https://drive.google.com/file/d/13fiIpPZrsyv7xs6dOe1W4vFIOpT3IR3w/view?usp=drivesdk)
and [full readback proof](artifacts/offline_study/live-20260908T151000Z/instance-50259194-droid-parallel/direct-readback/VERIFIED.json).

Geometry preservation15:33UTC:431 immutable files/959.3MB raw were archived into
125.5MB with full local member verification and unchanged before/after source
hashes. To respect Drive's100MiB per-file connector limit, two checksum-tracked
parts and restoration instructions were uploaded privately. The first part passed
full direct readback; the second hitHTTP403 quota, so complete cloud readback
remains pending and all source copies remain. This avoids a needless GPU rerun
but does not claim to reduce the still-required behavioral geometry work.

## Practical implementation summary — September 8, 14:35 UTC

The enhanced plan is implemented and its scheduling changes are active. This does
**not** mean the experiments are complete or the pipeline is computationally optimal.
The detailed gap/cause/concept/fix table below records the evidence and limitations.

1. **Distribute independent work:** three navigation GPUs are running, with four
   more workers waiting for DROID to release their devices. Intact RNG streams,
   existing completed results, and the 96-total episode design are preserved.
   The seven-worker allocation reduces elapsed waiting without claiming fewer
   scientific GPU-hours. No additional instances were needed.
2. **Remove repeated model work where validated:** the approved fixed-response
   replacement is deployed. HMM's same-pass prefix reuse is undergoing exact
   engineering validation before routed science; previously verified reference
   episodes are reused. Optional cross-epoch visual caching is implemented but
   not scientifically enabled because full-update equivalence remains unproven.
3. **Overlap preparation and automate handoffs:** the corrected PointMaze seeds
   234/235/236 have live CPU waiters behind Texas navigation. They automatically
   verify their full predecessors, process exit, device identity, free GPU and
   disk, then run native uncached pilot/first epoch/history stages. They never
   resume the incorrectly shuffled histories. Data/source staging and verified
   result archival occur while the GPUs perform useful work.

Three-seed waiter PIDs are 5576/5577/5578. Their source, input/runtime, actual
upstream sampler tests and 17 immutable launch/test records were verified. Texas
had 114 GiB free at the prelaunch check. The complete local suite ran 371 tests
with two existing skips. See [activation implementation](scripts/vast/corrected_pointmaze_activate.py)
and [verified launch records](artifacts/offline_study/pointmaze-corrected-activation-20260908-v1/ACTIVATED.json).

At 14:30 UTC all eight Louisiana GPUs were running HMM engineering, with 21/72
excluded cases published and no new failures. Routed behavioral endpoints were
still 0/768; validation progress is not an efficacy result.

At 14:33 UTC another session had restarted the quarantined California instance.
The account's aggregate rate became **$6.962593/hour**, including retained disks
but before usage bandwidth. Our five scheduled US instances contain 17 GPUs;
the extra California device is not part of our allocation or verified utilization
claim. It remains untouched by this schedule. No further lease fits meaningfully
under the remaining budget without first releasing other costs.

Durability check at 14:38 UTC: a compact private Google Drive archive passed full
compressed-file and all 90 member hash/size readback. It preserves the new
navigation controls/failure records, corrected PointMaze plans/launch records and
small frozen source bundle, cache CPU audit, and HMM status observation. This is
not an archive of every ongoing result or checkpoint. Active collection files and
11 logs were explicitly excluded; nothing was deleted locally. Archive SHA256:
`4276a321b61c0a3ed0b772e21146127cf64c971f997088efe81cace4aec3b488`.
The [exact private archive](https://drive.google.com/file/d/1Q205H8igVFQyO1LTtpsm-miVvgp01Q33/view?usp=drivesdk)
also passed connector metadata verification. Separate small receipt uploads hit
Drive's shared-client quota and stopped; their local manifests, successful
readback proof and failure receipt remain intact. Those receipt uploads are not
claimed complete, and the archival issue does not gate GPU execution.

## Activated schedule — September 8, 14:19 UTC

**The redistribution is running.** The exact committed partition contains 39
completed and 89 untouched candidate streams, with 12 episodes in each stream.
This preserves all 128 candidate streams across two tasks/eight arms and
**96 total episodes per task/arm**, not 96 per GPU. Native references are additional
reused records, not new candidate streams.

| Worker | Remaining whole streams | GPU handoff |
|---|---:|---|
| Nebraska 0 / PointMaze | 16 | Running; reuses same-device engineering |
| Nebraska 1 / Wall | 16 | Running; reuses same-device engineering |
| Indiana 0 / Wall then PointMaze | 7 + 8 | Running; original Wall 11-case receiving suite already passed and reused |
| Texas 0 / Wall | 11 | CPU waiter; complete its 144 DROID endpoints first |
| Texas 1 / Wall | 10 | Same full-DROID release requirement |
| Texas 2 / PointMaze | 11 | Same full-DROID release requirement |
| Texas 3 / PointMaze | 10 | Same full-DROID release requirement |

PLAN SHA `378cec55b21b84cf3d91b35fe0cfbc4f4b793a0f33b4cc6a726d17f1a4a46a63`;
CUTOVER SHA `f70ee6fdf6a0f6fc418022d033b0fa16a81a1d86ba26fedaf7ff72830a561b6e`.
The collectors verify whole completed assignments and paired raw records before
aggregation. Corrected PointMaze histories234/235/236 have real dependency-bound
plans behindTexas0/1/2; their CPU activation is separate from GPU training start.
Latest verified aggregate is$6.732593/hour including retained disks, before
usage bandwidth. No new instances were needed for this redistribution.

Another administrative gap was caught during deployment: after the partition
committed, the watcher-exit check saw changed process argv and stopped before
retiring the old parents. Direct checks found the watcher absent, both original
parents still stopped and both GPUs empty. A narrowly reviewed recovery bound
the exact parents with pidfds, completedTERM/CONT, and required actual terminal
states plus stable empty devices. The failed attempt remains preserved; no
scientific child was killed, no episode restarted and no partition changed.
Emptyargv is tolerated only while waiting after termination, never as permission
to signal or claim completion. An owned harmlessCPU lifecycle test passed first.
This is process-control robustness, not a mathematical acceleration from the book.

The first released navigation GPU waited for the second stream during the joint
commit; that was real temporary idleness, not scientific throughput. Early
Indiana engineering overlapped that wait and passed in1108.32seconds; its20files
were independently copied and rehashed. The new allocation's navigation-only
projection is roughly6–8hours from activation under the observed~118s/episode
rate and anticipatedDROID release, not a full-study ETA or achieved speedup.
Report actual time once allassigned streams finish, including receiving overhead.

## Deployment update — September 8, 13:47 UTC

The enhanced scheduling implementation is now **partly activated**, not yet a
measured completed speedup. The v2 operations passed27receiving tests on each of
Nebraska, Indiana andTexas; the full local suite passed364tests with2existing
runtime skips. Scientific source and both task freezes are unchanged.

- **Independent work overlaps:** Indiana supervisor9839/child9845 is executing
  the original11-case Wall receiving suite now, while Nebraska finishes its
  current scientific streams. Its eventual worker reuses the same device-bound
  proof; it does not run this suite twice.
- **Safe boundary is active:** Nebraska supervisor7242 has paused the two original
  producers, not their scientific children. Wall started another intact stream
  just before the pause; this stream must finish too. No stream is restarted or
  truncated for the schedule. The seven-device partition is published only after
  both complete and the actual remaining-work inventory verifies.
- **No770MB retransfer:** allthree hosts reverified their existing payload in
  place. The v2 staging receipt records zero copied payload bytes. Only small
  operations/test files changed; Texas's repaired runtime stays unchanged.
- **Training preparation overlaps too:** corrected PointMaze seeds234/235/236
  have a source-bound native-uncached queue proposal. CPU staging onTexas may
  happen duringDROID; final activation requires the actual navigation launch
  identities. The optional cache pilot is explicitly not a prerequisite.

This separates three different savings: scheduling removes waiting, immutable
input reuse removes transfers, and an accepted arithmetic cache would remove
model work. The third is still unproven; don't add an assumed cache speedup to
the first two. Whole-stream boundary delays and receiving checks belong in ETA
arithmetic even though they don't change scientific sample size.

Code: [stable handoff and proof reuse](scripts/vast/navigation_redistribution_control.py),
[source-bound activation](scripts/vast/activate_navigation_redistribution.py),
[corrected history preparation](scripts/vast/corrected_pointmaze_prepare.py).
Evidence: [v2 receiving completion](artifacts/offline_study/navigation-redistribution-20260908-v2/DONE.json).
The cache pilot's16local files also passed full privateDrive readback at13:46UTC
under `live-20260908T134200Z/visual-cache-engineering`; no local files were deleted.

Cache successor readiness at13:54UTC: a separate source-bound engineering policy
now preserves the real initial-eval/post-validation-train lifecycle. It rejects
stateful/buffer-bearing components, unknown classes, nonzero stochastic rates,
parameter changes and unaudited attention backends. The actual Indiana CPU tree
audit checked200modules, zero stochastic rates/buffers and no availablexFormers;
no GPU forward or native mode change was performed. The33focused cache tests and
full371-test suite passed(2existing skips in the full suite). This clears a source
readiness issue, **not** receiving-GPU numerical parity or a speedup. The successor
still needs its bounded six-composition/mode proof and full native/cached update,
validation, gradient/optimizer/RNG comparison on a separately released device.
No scientific cache is active, and queued uncached histories remain unblocked.
Evidence: [actual encoder CPU audit](artifacts/offline_study/visual-cache-mode-audit-20260908-v2/CPU_TREE.json).

## September 8, 12:51 UTC: gaps, causes, and the enhanced processing plan

This section answers the user's cost/throughput concern explicitly. We have **not**
established that the pipeline is computationally optimal. High GPU utilization
shows activity, not useful FLOPs, correct methodology, or minimum cost to a complete
result. The older dated entries below are historical; this table distinguishes
deployed changes from checks still pending. Do not count an implemented file as a
verified speedup or a completed experiment.

### What went wrong, and what changes address it

| Gap and evidence | What caused it | Scaling Book connection | Concrete correction and current status | What the saving would mean |
|---|---|---|---|---|
| Navigation had about 19–20 hours left with one GPU per task, despite independent unstarted streams. Measured episodes take about 118 seconds. | Fixed task-to-device queues were not rebalanced against the time needed to finish the full behavioral panel. Other GPUs were doing different useful work; being busy did not make this allocation optimal for the user's priority. | [Training parallelism, Ch. 5](https://jax-ml.github.io/scaling-book/training/) and [inference objectives, Ch. 7](https://jax-ml.github.io/scaling-book/inference/): choose the split and latency/throughput objective explicitly. Our inference jobs need no gradient synchronization. | **Activated at 14:16 UTC:** three scientific workers are running and four CPU waiters will join after their full DROID assignments. The committed partition preserves 39 completed and uniquely assigns 89 remaining whole streams. Source, freezes, seeds and 96-total population are unchanged. Projected navigation-only completion is roughly 6–8 hours from activation, including remaining dependency and receiving checks; not an achieved speedup or full-study ETA. | Primarily shorter wall time. Approximately the same scientific GPU-hours, plus receiving checks. It is not automatically a lower total invoice. |
| The old intervention repeated expensive forecasts for every CEM population. | Online response estimation and a native shadow multiplied model work; the small rank solve was not the dominant cost. | [Transformer work accounting, Ch. 4](https://jax-ml.github.io/scaling-book/transformers/) and [rooflines, Ch. 1](https://jax-ml.github.io/scaling-book/roofline/): reducing total required operations can matter more than buying faster hardware. | **Deployed:** user-approved, separately named fixed-response replacement composes its map offline and makes one forecast, with no online probes/shadow. Current native/refined MetaWorld episodes are both approximately 300–303 seconds. The replacement has its own fits/evaluations and does not inherit the old method's efficacy. | Removes recurring model work, reducing GPU-hours as well as latency relative to that old algorithm. These are not hardware-matched measurements establishing an exact speedup factor. |
| HMM originally waited for two coupling/control arms it does not depend on. It also initially required a full native shadow per scientific forecast. | An unnecessarily broad queue dependency and duplicated calculation of an already-native H1/H2 prefix. | [Inference reuse, Ch. 7](https://jax-ml.github.io/scaling-book/inference/) plus [overlap/critical bottlenecks, Ch. 1](https://jax-ml.github.io/scaling-book/roofline/). This is same-pass prefix reuse, not an LLM KV-cache transplanted into JEPA. | **Priority handoff activated:** finish the current whole coupling child, perform HMM work, then resume the last original control. **Same-pass scientific implementation queued behind exact full-GPU parity:** reuse native H1/H2 before the H3 edit. Reuse 576 verified native/constant/random reference episodes instead of collecting duplicates; retain 768 new routed episodes. | Reordering saves waiting time, not total scientific work. Reference reuse and eliminating the shadow reduce work. A two-to-one call reduction is not proof of a twofold full-episode speedup. |
| We lacked a current kernel-level breakdown showing why a full 300-candidate forecast costs what it does. | Episode timers and GPU-utilization samples were being asked to answer questions they cannot answer: matrix compute versus memory traffic, synchronization or launch overhead. | [Profiling, Ch. 9](https://jax-ml.github.io/scaling-book/profiling/) explains why theoretical rooflines need actual traces. | **Executed and verified at 12:56 UTC:** the bounded first-CEM-population audit finished in 149.46 seconds. Ten warmed unprofiled forwards per precision, unchanged inputs/parameters, exact strict-FP32 repeat and full CUDA trace. The process exited and released IndianaGPU0. See measured audit below. | We now have actual timing/error evidence. Matrix operations, softmax and elementwise/mask work all appear; profiler overhead and duplicate attribution prevent interpreting the raw operator table as an additive time budget. |
| Strict FP32 planning may leave faster matrix hardware unused. | Conservative arithmetic was frozen for native parity. Faster precision is not automatically the same numerical planner: changed rankings can change actions. | [GPU matrix units, Ch. 12](https://jax-ml.github.io/scaling-book/gpus/), [arithmetic intensity, Ch. 1](https://jax-ml.github.io/scaling-book/roofline/) and [inference, Ch. 7](https://jax-ml.github.io/scaling-book/inference/). | **Measured, not adopted:** this actual forecast took 3.13847s strict FP32, 2.85019s TF32 and 2.77808s BF16 (medians). Both faster paths changed predictions. Existing panels stay strict FP32; a changed numerical path needs a separately bound paired policy and full-planner checks. Compilation/fusion remains unimplemented. | Approximately 1.10x/1.13x forecast throughput in this diagnostic, not a full-episode speedup and not enough alone to explain away a 20-hour queue. |
| Training repeatedly encodes overlapping images with a frozen visual encoder. | We reuse features within an objective, but not across overlapping clips and epochs. Full updates mix this fixed encoder cost with trainable predictor work. | [Inference caching, Ch. 7](https://jax-ml.github.io/scaling-book/inference/) and [compute/memory tradeoffs, Ch. 4](https://jax-ml.github.io/scaling-book/transformers/). | **Implemented, not scientifically active:** the first GPU pilot passed frame-composition parity but stopped at the native validation-to-train-mode guard. A separately source-bound successor passes CPU tests/audit and preserves native modes; receiving-GPU mode/composition, full-update and RNG parity remain required. Corrected uncached histories do not wait for this optional pilot. Conservative full-cache capacity is 31.26 GB for Wall or 67.95 GB for PointMaze. | Could remove repeated encoder FLOPs across epochs/seeds, but creates storage/read traffic. Savings depend on measured encoder fraction; 135×/169× frame reuse does not imply that end-to-end speedup. |
| New PointMaze histories were not author-native despite sampler tests passing. | The wrapper inherited Wall's `shuffle=True`; PointMaze's actual upstream train and validation samplers use `False`. Earlier tests compared against the same manually chosen assumption. | **This is a methodology/testing defect, not a Scaling Book theorem.** Efficient execution must preserve the intended optimizer/data experiment; faster incorrect work has no scientific throughput value. | **Affected jobs stopped and queued continuations cancelled; local fix tested:** execute the actual pinned upstream caller/loader in regression tests, bind explicit per-task sampler policy, reject old shuffled checkpoints as corrected-native resume points. Existing histories are retained and labelled; corrected histories must initialize afresh. Released-checkpoint offline/behavioral results are unaffected by this training defect. See [correction record](reports/POINTMAZE_TRAINING_SAMPLER_CORRECTION.md). | Prevents further invalidly labelled training spend. It does not recover sunk cost, and required corrected histories still cost compute. |
| Slow staging and backup transfers extended time-to-results and retained-disk charges. | Oversized input copies, stalled uploads and treating transfer/storage work as part of the GPU dependency path. | [Bandwidth/overlap, Ch. 1](https://jax-ml.github.io/scaling-book/roofline/) and [profiling, Ch. 9](https://jax-ml.github.io/scaling-book/profiling/). | **Deployed in named paths:** copy verified required inputs, overlap CPU staging with current GPU work, use bounded upload retries/8 MiB chunks and full Drive readback. Retain only justified worker disks after verified release audits. No new destructive cleanup is claimed here. | Less waiting/retransmission and potentially less retained-storage cost. The successful 29-second backup retry is an operational observation, not a controlled chunk-size benchmark. |

### The arithmetic behind the allocation and cost decisions

- Roofline lower bound: `T >= max(F / C, Q / W)`, where `F` is operations,
  `C` is arithmetic throughput, `Q` is bytes moved and `W` is bandwidth. Launches,
  synchronization and serial dependencies can make real time longer. Optimize the
  measured limiting term rather than assuming the GPU price predicts performance.
- At the navigation snapshot, six remaining arms per task imply
  `6 × 96 × 118 / 3600 = 18.88 GPU-hours/task`. With two task GPUs this is about
  18.9 elapsed hours, **not** 37.8 elapsed hours; the tasks run concurrently.
  More devices divide intact streams; they never multiply the 96-per-arm sample.
- For independent jobs, an ideal equal-speed estimate is
  `T ≈ remaining_GPU_hours / assigned_GPUs`, then add staging, engineering and
  dependency delays. Seven devices are not immediately available: Texas must
  finish its complete DROID assignments first. This explains why the operational
  proposal is about 8 hours rather than the naive `37.8 / 7 = 5.4 hours`.
- GPU cost is approximately `GPU_hours × price_per_GPU_hour`, plus storage,
  bandwidth and preparation. Parallelizing identical work can shorten latency
  without making that work cheaper. Arithmetic reuse can reduce both.
- At the verified aggregate rate of approximately **$6.73/hour**, retaining the
  same fleet for another 24 hours would cost approximately **$161.58**, before
  usage bandwidth. This is a run-rate projection, not an audited past invoice or
  a promise that the fleet remains unchanged for 24 hours.
- For a frozen-encoder fraction `f` of update time, an ideal zero-cost cache gives
  a ceiling of `1 / (1 - f)` speedup. Real cache reads and preprocessing reduce it.
  A cache is worthwhile only when construction/verification cost is recovered by
  subsequent avoided encoder work; measure that before scheduling full generation.

Acceptance criteria for this enhanced plan: unchanged scientific populations and
planners; exclusive per-device work; exact stream coverage and no duplicates;
receiving fidelity checks; measured seconds per completed episode/update and
GPU-hours per completed comparison; explicit readiness and failure receipts. Do
not introduce extra outcome-selected arms, smaller CEM budgets, changed seeds,
or reduced episode counts merely to obtain a shorter ETA. Corrected PointMaze
training and other required histories remain follow-on work, not silently removed
when GPUs are reassigned to the nearer-term behavioral results.

### Measured receiving-GPU processing audit (September 8, 12:56 UTC)

Implementation: [bounded profiler](scripts/vast/profile_native_forecast.py).
Evidence: [report](artifacts/offline_study/processing-audit-20260908-v1/report.json),
[input identities](artifacts/offline_study/processing-audit-20260908-v1/INPUTS.json).
This was the first actual H6-by-300 CEM population of the excluded Reach engineering
scenario 2026090719 on an RTX4090, not a completed episode or an intervention result.
Two warmups and five timings per block, forward then reverse mode order, yielded
ten timings per mode; the instrumented trace was collected separately.

| Arithmetic | Median forecast seconds | Strict/alternative throughput ratio | Visual relative L2 difference | Bitwise same as strict? |
|---|---:|---:|---:|---|
| Strict FP32 | 3.13847 | 1.000 | 0 | Yes |
| TF32 diagnostic | 2.85019 | 1.101 | 0.000333 | No |
| BF16 diagnostic | 2.77808 | 1.130 | 0.004496 | No |

Maximum absolute visual differences were 0.03804 and 0.64981 respectively.
Neither error magnitude proves unchanged candidate rankings or task outcomes.
The trace includes matrix multiplies, softmax, masks and substantial elementwise
work; it also contains `Command Buffer Full` instrumentation events. CPU operator
attribution and underlying GPU kernels overlap, so summing both would double-count.
Profiler FLOP counters are incomplete and are not the model's total operation count.
No inference of a memory-bound percentage or achieved peak-FLOP efficiency is made.

Report SHA256: `fba9b63ba9555ff0b6265e71fc230e15203103661532224d70d8411998a8efbc`.
Trace SHA256: `b364e8e17be5ccefde94637140bb5c40e55d48d29a76d9d26b67f1381426955f`.
Decision: retain strict arithmetic for ongoing comparisons and prioritize the
already identified redundant work and uneven scheduling. Do not delay current
experiments for speculative precision/compilation experiments.

Durability/cost follow-up at13:11UTC: all six processing-audit files, including the
full trace, passed private Drive download/readback verification. The initial
shared-client query-quota403 was resolved on a bounded retry using the already
verified parent ID and one transfer/checker. No local data was deleted. Separately,
[the retained-storage audit](reports/RETAINED_STORAGE_AUDIT_20260908.md) identified
$0.737037/hour in stopped/created disks. These are not idle-GPU compute charges.
Some preserve ~10.47GB of raw-fit/metric objects whose exact coverage is not
established by the reviewed Drive manifests; deleting both remaining source disks
would be unsafe. The newest provider total is$6.962593/hour after an external
California restart; that resource remains quarantined, not available to our queues.

### Implementation checks that did not pass (September 8, 13:32 UTC)

- The first navigation cutover checked GPU idleness once immediately after its
  child became terminal. It found the GPU still occupied, aborted before assigning
  any work, and resumed the original schedulers without killing their scientific
  children. It did not record the occupied PID, so delayed driver/NVML release is
  plausible, **not proven**. The v2 correction records device/process identities
  and waits a bounded interval for stable emptiness. Independent receiving
  engineering should overlap the old-stream drain instead of waiting for it.
- The initial transfer sent the770.5MB frozen packet through the laptop back to
  its Nebraska source as well as to receivers. That origin round trip was wasteful.
  v2 preparation reuses and re-verifies the existing payload in place; no repeated
  source/runtime download or scientific rerun is needed for an operations fix.
- Texas lacked four public OSMesa dependencies. Only absent library paths were
  populated with exact Nebraska bytes; existing libraries/driver were not upgraded.
  A generated object from the failed fallback build was preserved, then restored
  to its original verified bytes. The prebuilt MuJoCo extension never changed.
- The bounded native cache pilot ended after160.27seconds, safely releasing its
  GPU. It preserved1023 unique FP32 frame embeddings (~404MB) and passed both
  independent-composition comparisons. It **did not pass full-update parity or
  produce a speedup measurement**: upstream validation subsequently calls
  `world_model.train()`, flipping the frozen DINO encoder's mode, while our cache
  guard required eval mode. Silently forcing eval would change the upstream
  lifecycle. A successor must audit deterministic train/eval behavior and preserve
  native modes; until then corrected histories retain uncached native execution.

Evidence: [cache pilot receipts](artifacts/offline_study/visual-cache-engineering-20260908-v1/RECEIPTS.json),
[navigation preparation](artifacts/offline_study/navigation-redistribution-20260908-v1/DONE.json).
These are excluded engineering observations and software corrections, not negative
task-success findings or reasons to change an intervention/sample. A failed
optimization check must not block the otherwise valid baseline/behavioral runs.

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
arms—384 episodes—that are not HMM inputs. The activated priority handoff removes
that scheduling edge: finish the current complete stream, run unchanged HMM
verification/evaluation, then resume the other controls. All comparisons remain
required. This is critical-path scheduling, not a FLOP reduction or a license to
skip controls. It can bring the HMM answer forward without another rental; the
total panel work and full-study completion time need not fall by the same amount.
At activation the first coupling arm was already running. Its streams finish
intact; HMM moves ahead of the final192 coupling-control episodes. Do not report
all384 episodes as avoided delay, and no required comparison is removed.

Likewise, staging compressed PointMaze data and its separate pinned runtime on
CPU/disk while DROID uses the GPUs overlaps preparation with useful computation.
The upper bound on benefit comes from the work that can actually overlap; it does
not turn an I/O-bound preparation step into useful GPU work. Keep the completion
latency and verified samples/hour alongside GPU utilization when measuring speed.

The same local MetaWorld backup retry retained its388.6MB spool and switched from
64MiB to8MiB chunks overIPv4: the successful upload took about29seconds at13.8MiB/s,
then passed full member readback. The previous attempt had sent over1GB without
completion. Future snapshot uploads now use8MiB chunks with bounded retries and
progress reporting. This is operational recovery evidence, not an isolated
network benchmark; no model precision, evaluation sample or GPU work changed.

## September 10: keep a bounded decision diagnostic computationally bounded

The new diagnostic shares one native 300-candidate action tensor and one encoded
initial/goal pair across five conditions. Encode once when the input is genuinely
identical; do not recompute it per condition. If redundant encoding occupied a
fraction `f` of the original runtime, replacing five encodings with one gives an
Amdahl-law upper bound `1 / (1 - f + f/5)`, **not** a fivefold end-to-end speedup.
We have not separately measured `f` here, so no measured speedup is claimed.

Candidate scoring is a GPU batch of 300, with one predictor rollout per condition.
The initial physical state and action bank are hash-bound. When conditions choose
the exact same candidate, the simulator prefix may be reused within that scenario
after exact reset/repeat validation; this saves work without manufacturing more
independent samples. Native, zero-dose and repeated-state engineering checks remain.

One Michigan RTX4090 costs $0.423704/hour with its chosen disk. The early measured
rate is approximately 16 seconds per scenario across all five conditions. A single
fully occupied GPU is enough for this small panel; extra rentals would repeat
environment setup. The compact input archive is about51MB; no training dataset or
RGB decoder is downloaded just to run this diagnostic.

Crucially, using only the first CEM population and a fifteen-step prefix is a
**different, explicitly narrower diagnostic question**, not an optimization that
reproduces the complete fifteen-iteration, full-episode planner more cheaply.
The complete old 960-episode study remains separate and unchanged. This distinction
prevents a scope reduction from being marketed as an equivalent compute speedup.
