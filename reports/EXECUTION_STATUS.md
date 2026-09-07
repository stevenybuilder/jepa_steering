# Execution checkpoint — 2026-09-07, 23:23 UTC

**MetaWorld stimulus repair:** outcome-blind pairing audit-v3 found
five mismatched expert-goal observation hashes among286 completed candidate
episodes (Reach20/35, Reach-Wall25), with all initial vectors/images and seeds
still matching. The complete audit and mismatches are preserved at
`artifacts/offline_study/behavioral-pairing-audit-20260907-v3` with FAILED, not DONE.
The cause is now reproduced:24 native expert repeats had identical physical goals,
proprioception and expert actions, but4–12 RGB channel values varied by one level.
The exact historical conflicting hashes were reproduced. All192 original native
baseline goal images were then recovered byte-for-byte, without model outcomes or
changed scenarios. Shared goal delivery passed24 source-environment repetitions,
all192 original hash bindings and unchanged planner RNG. Proof-v2 report SHA
`4eb1e0c3cd7247f7164eec0a5a43a0c8b6abdaaba8c5203caad2055d5a365c30`.
Three canonical rank/combined streams are running after preserving the original
partial traces. Four coupling repair queues wait for the original complete shards,
then freeze input-only replacement manifests and rerun entire affected12-episode
logical streams. No reduced sample size, isolated-episode RNG restart, relaxed
pixel gate, discarded failure or double counting is allowed. Corrected paired
assembly/analysis and confirmation remain incomplete. Wall/Maze offline and
DROID native endpoints are unaffected. See [repair contract](METAWORLD_STIMULUS_REPAIR.md).

The six-task study is **not complete**. The governing documents remain
[EXPERIMENT_PLAN.md](../docs/EXPERIMENT_PLAN.md),
[BEHAVIORAL_EVALUATION_AMENDMENT.md](../docs/BEHAVIORAL_EVALUATION_AMENDMENT.md),
and [DROID_METHOD_ALIGNMENT.md](../docs/DROID_METHOD_ALIGNMENT.md).
RoboCasa is excluded. No new method search or partial-outcome selection is authorized.

## Live assignments

| US instance | GPUs | Assignment | Hourly rate |
|---|---:|---|---:|
| 50125440 | 8 RTX5090 | Three canonical rank/combined streams; Wall native96; four coupling/control shards with input-repair queues next | $2.953889 |
| 50189244 | 1 RTX PRO6000 Blackwell Server | PointMaze native96 development; DROID64 and queued engineering completed | $1.884000 |
| 50195621 | 4 RTX5090 | Four remaining disjoint coupling/control scientific shards; hardware checks passed | $1.641667 |
| 50205763 | 1 RTX4090 | Wall/Maze complete native offline baselines and20 category/precision fits finished; intervention comparisons running | $0.343704 |

Including retained stopped disks, live aggregate is **$6.928815/hour**,
before usage-based bandwidth. Latest user ceiling is $7/hour, US-only. New
California staging took time and initially suffered an interrupted transfer;
those GPUs were not counted as scientifically active while staging. Direct
keepalive-enabled runtime transfer completed; no local large archive was needed.
The board and live provider/process checks supersede this timestamped snapshot.

The preceding new rental50205255 was destroyed before staging any research data,
source or job: its US label conflicted with a reported CN IP. No research result
was removed or lost; provider absence was verified. The replacement's published
IP and live remote egress both independently geolocate toIndianaUS. Source inputs
and isolated runtime are staged. All14 GPUs were executing useful jobs at the
22:39 check, not merely rented or waiting on downloads.

## Direct Wall/DROID phase answer

| Task | Offline native forecasts | Offline interventions | Behavioral endpoint |
|---|---|---|---|
| Wall | Complete and verified, BF16 + FP32,192 validation trajectories /2112 clips /4224 H6 prefixes | BF16 coupling comparison complete and verified; remaining category/precision comparisons running/queued |40/96 native simulator episodes at23:22; no candidate-success comparison yet |
| PointMaze | Complete and verified, BF16 + FP32,200 trajectories /12200 clips /24400 H6 prefixes | All five categories fitted in both precisions; coupling comparisons running |81/96 native simulator episodes at23:22 |
| DROID | Separate offline intervention-forecast sweep not completed | Not completed; distinct training/fit population remains unresolved |64/64 recorded-action native planning evaluations verified; no physical robot execution or intervention comparison |

Navigation verification reconstructs all expected trajectory/prefix identities,
window hashes and aggregate metrics:
`artifacts/offline_study/navigation-native-verification-20260907-v1`.
The raw window metrics remain outside Git. A complete native reference is not a
complete intervention study; overlapping clips/prefixes are not independent samples.

Both MetaWorld native development references now have96/96 episodes complete.
The first complete Wall BF16 coupling scope does not demonstrate an offline gain:
equal-budget joint steering changes H6 proprio MSE by **+0.1567%** versus native
(simultaneous95% increase interval **[0.0017%,0.3117%]**) and visual MSE by+0.3754%
(interval spans zero). Its proprio error is also worse than its matched-random
control. No arm meets the frozen minimum-useful improvement rule in this first
scope. This is not a behavioral veto or a task-success finding. Every arm and
contrast is retained in `artifacts/offline_study/navigation-analysis-compact-20260907-v1`;
full verified report SHA `fe1f2a630e6f32cb255abc3c7bbc4f117b3256a295dbb3e453866969f1c5d554`.

The complete Wall data and source/pilot receipts are now staged on California:
58.964GB logical transfer,432.258MB wire bytes, no new rental or local dataset copy.
Production three-seed training is still not launched; staging is not a training result.

The next separately gated jobs are Reach combined logical stream0 and Reach-Wall
rank4/control logical stream0 (12 episodes each, not the whole96-condition cohort).
They use immutable `rank-panel-code-v2`, tarSHA
`1f6da479ed7af993e82b7ccfa905f2dbad8de4ec14c36b68df56a1b71dc59652`.
Initial preflight-v1 rejected newer source files before model/outcome access.
The complete diffs were audited: unchanged MetaWorld dispatch/normalization/config,
plus extraction of the identical support computation into a forwarding method.
Only those four exact old/new hash pairs are accepted; all other critical operator
files and original native code-v33 remain checksum-bound. No prefix-cache subclass
or numerically different batch setting is used in these scientific jobs.

The DROID public-source catalogue contains74970 trajectory objects across13 labs,
including both success and failure folders. This read-only metadata enumeration
does **not** reconstruct the authors'8000-example training subset and selects no
replacement population. Compact receipt:
`artifacts/offline_study/primary-durable-20260907/droid-public-catalogue-20260907-v1`.

## What actually completed

- Corrected three-primary-task offline comparisons and combination analysis remain
  complete; see [the results](CORRECTED_OFFLINE_RESULTS.md). They are prediction
  measurements, not demonstrated task-success gains.
- Four selected rank/combined full-native-CEM integration checks completed. All
  reports/protocols/call logs were copied and hash-verified. They use one excluded
  engineering scenario each, not a behavioral sample. Each took 7,681–7,867 seconds,
  approximately 2.1–2.2 GPU-hours. This is a major remaining runtime limitation.
- Four full-H6-only coupling/drop-one checks passed exact native/source field
  parity at 8, 19 and 300 candidates, then the entire 100-step simulator episode.
  They took 313–316 seconds including parity setup. The fixed coupling/control
  behavioral jobs launched after these gates, not after choosing favorable smoke
  outcomes.
- DROID's full 15x300, H3 native planner passed exact repeated actions/metrics.
  The updated stimulus trace also passed, report SHA
  `4587522fcab5c518465545f524b88b3b4cd28dd780640d765bf7fb1d58ecd7c8`.
  It records the source file, five sampled raw frames, four-frame goal segment,
  and checks the native CPU RNG without consuming it. The subsequent 64-episode
  baseline completed all64 episodes in5,206 seconds. All64 episode hashes, stream
  identities, full15x300 call schedules and XYZ/orientation/gripper metrics were
  reconstructed locally. Native score **51.09965**, mean XYZ error **0.03612544**;
  this is not a percentage success rate (the scale's maximum is80). The sample
  contains15 recording families and58 distinct recording/goal segments. Report SHA
  `31ff18bb63d54876ccc4ec300b89957bc133253746b2d994f932b655231355f4`;
  verification: `artifacts/offline_study/droid-native-replication-verification-20260907-v1`.
  No steering comparison, physical robot execution or training-history claim follows.
- Initial pairing audit: 17 completed MetaWorld candidate episodes exactly
  matched their baseline initial-state vectors and initial/goal image hashes.
  This is partial **stimulus verification**, not a favorable-outcome finding.
- Subsequent cross-worker audit verified **64 paired candidate episodes**, including
  the California streams. All matched native initial vectors and initial/goal image
  hashes; no success outcomes were extracted for that audit. Its compact receipt is
  `behavioral-pairing-audit-20260907-v2`. All 13 GPUs were running scientific jobs at
  the latest check: twelve MetaWorld shards and one DROID baseline.
- PointMaze's native CPU simulator passed two exact 30-step repetitions, report SHA
  `58e5fe06fb2ecb20da1c88255f9d908f01837655b56d253c32c1b84f84941508`.
  A prior logging-helper error is preserved; simulator physics were not modified.
  The verified isolated runtime is staged on Virginia for its own CPU check and
  full native GPU planning then passed on Virginia: two exact repetitions,171.5s,
  report SHA `59ae7730d7304a86e36e763ce9224b8ee9ef64a37575eef8e59e6c5fd2017b19`.
  PointMaze native96 development is now running with immutable code-v13; first
  complete episodes passed action/forecast/count checks. Base1 confirmation stays closed.
- Wall/Maze offline cohort preparation completed without model outcomes:128 fitting
  families/task; all192/200 validation trajectories;2112/12200 clips and4224/24400
  H6 prefixes. Metadata and all accessed videos are checksum-bound. The existing
  five-sweep fitting algorithms, precision roles and controls are unchanged.
  Full new-task offline model comparisons have **not** completed yet.

## Exact running panel

The immutable MetaWorld freeze is
`artifacts/offline_study/primary-durable-20260907/behavioral-development-freeze-20260907-v1/protocol.json`.
Seven conditions, 96 scenarios per task/condition, paired fixed logical streams,
and its simultaneous analysis/selection rule remain unchanged. The native jobs
use immutable code-v33. New coupling execution code-v35 verifies that older source,
unchanged native execution modules, frozen components, and completed engineering
receipts in a separate launch contract. It does not rewrite the scientific freeze.

`result.native_success` in the shared episode schema denotes the **native
environment's success definition**; the separate `arm` field identifies whether
the evaluated policy is native or steered. Candidate results are not baseline
successes. All components in this panel remain native below full H6.

DROID follows the released 15-recording config. The sixteenth published episode
is upstream-truncated and remains preserved; no substitute was invented. Its
64 sampled segments are not 64 independent recording families. The endpoint is
the authors' recorded-plan XYZ error/score, not physical-robot task success, and
this single-checkpoint replication is not a three-seed/history result.

## Still required

Remaining rank-panel engineering/execution, comparative behavioral analysis,
fresh-confirmation freezes/evaluation, Push-T behavioral completion, PointMaze
and Wall task expansion, DROID intervention fit provenance, conditional HMM
eligibility, and all required three-seed histories remain unfinished. DROID's
paper-8,000-training-trajectory subset identity is still unresolved; the released
evaluation recordings must not be repurposed as its training set.

The new [training pilot](../src/offline_study/training_pilot.py) **passed five Wall
updates**, not a complete training history. The original waiting queue was replaced
before execution when a missing reference-function namespace variable was caught
by review. The corrected full training control-flow unit passes; pilot-v2 uses
immutable code-v11. GPU parameter/optimizer/scaler/RNG parity passed, with updates
2–5 taking1.374–1.678 seconds and peak9.17GB. Report SHA
`d643d668850fd81839f13d67762e2d26a3b2663024d3dad809221dd3f9ea98f0`.
It compares an actual source-extracted
training step against 16x8 accumulated gradients, with one optimizer/scheduler
update per effective batch. It retains native sampler/drop-last layout and loss
weights; it does not claim reconstruction of the authors' exact random streams
or hardware collective reduction order. PointMaze's isolated Python3.10,
MuJoCo2.1/mujoco-py2.1.2.14 and D4RL native CPU and full-model checks pass.
The frozen seven-arm behavioral analyzer
is implemented and tested, but refuses partial coverage and has **not** produced
a comparative result yet. The v1 MetaWorld episode logger does not retain planned
action magnitudes, so that secondary diagnostic is unavailable for those running
shards; do not claim otherwise. New navigation episodes retain complete planned actions.

The completed larger-response-batch engineering test rejected chunks16/32 for
adoption: no consistent throughput gain, and edit fields/forecast/energy differed
numerically from chunk8. No scientific source, precision or tolerance was changed.
Its report SHA is `57279c2e9fad855d48ed60cf7726a073c708fd64e25d899d1cee502d04407560`.

## Storage recovery

The laptop filled during this continuation. Six enumerated local duplicate
`native_fit.pt` captures, **8,758,014,190 bytes**, were removed only after matching
their prior durability receipts and copies on **both existing US instances**
50125440 and 50189244. No original remote data or fitted operator bank was removed.
Recovery maps are in
`artifacts/offline_study/fit-capture-storage-relocation-20260907-v1/` and beside
the second remote copies. The old full-local-bundle verification is historical;
rerunning it now requires restoring these captures or auditing their remote paths.
No GPU instance was rented solely for storage.

When free space fell again, **42 additional bulk window-metric duplicates totaling
1,710,558,874 bytes** were relocated under the same two-copy SHA-verification rule.
Recovery maps are in `artifacts/offline_study/window-metrics-storage-relocation-20260907-v1/`.
All original metrics remain on50125440 and a second verified copy on50189244;
summary reports, protocols and banks stay local. The cause of the repeated local
disk growth is not established; no unrelated cache or user files were removed.

Measured optimizations and limitations are recorded in
[jax_scaling_notes.md](../jax_scaling_notes.md).
