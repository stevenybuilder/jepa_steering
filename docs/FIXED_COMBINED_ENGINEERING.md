# Refined combined intervention: engineering contract

September 8, 2026. This implements the user's requested cheaper combined
successor. It does not alter completed results, the running standalone/HMM
panels, or the original expensive operator. No behavioral or confirmation
launch is granted by this document.

## Frozen components and meaning

Initial task: Reach, with its already fitted `fixed_response_rank4_v1` bank and
the original equal-standardized-energy vision/action coupling bank. The two
arms are `combined_fixed_rank4` and `matched_random_combined_fixed_rank4`.
Each uses the corresponding existing learned or matched-random components.
No refitting, new direction, dose change, or search over ranks is permitted.

Coupling changes the predictor visual input at H3 and its action condition at
B3/H3. The refined map must consume the **native**, not already coupled, B3/H3
field. Naively stacking hooks changes that feature contract and is not accepted.
The rank addition is still one distributed four-direction field on the newest
256 patches, with FP32 map/dose arithmetic and the unchanged per-candidate zero
rule. Zero rank treatment retains the coupling component. Short H1–H5 forecasts
remain entirely native, matching the previous combined planning wrapper.

## Execution mechanism and cost accounting

The new method is named `native_prefix_fixed_combined_v1`. H1/H2 are shared
native computations. At H3, before any coupling hook modifies inputs, replay
the actual predictor on those same inputs only through B3. Capture that native
field, exit the nested prefix, and continue the normal coupled H3 call with the
fixed-map addition at B3. H4–H6 continue normally.

This invokes one backend forecast and **four extra predictor blocks**: 40 block
executions rather than 72 for a full native-plus-combined H6 reference. It also
repeats the predictor input embedding/conditioning work for that prefix. These
are operation counts, not measured latency or a guaranteed speedup. There are
no online response probes or full native-shadow rollouts, but there IS extra
native-prefix work. The standalone zero-extra-prefix contract is unchanged.

Evaluation-only predictor mode, no activation checkpointing, zero SDPA dropout,
no foreign/global predictor hooks, and no concurrent/reentrant adapter call are
required. All owned hooks must be removed on success and every exception path.
Inputs, parameters, buffers and random-number state must remain unchanged.

## Required validation before any behavioral launch

1. Local independent-reference tests for both arms in FP32/BF16, all short
   horizons, active/inactive candidates, batched and strided inputs, exact
   native-feature coefficients, block/call counts and exceptional cleanup.
2. A receiving-GPU check bound to the original task/checkpoint, both completed
   fit receipts, the original 128-family fitting cohort, and a verified fitting
   stimulus. Use candidate counts 8/19/300 and horizons 2/5/6, both arms, strict
   FP32, one warmup and three synchronized timing repetitions. Compare to an
   independently constructed full-native-capture/static-hook reference using
   the SAME fixed map, not the old candidate-response solver. Require exact
   scientific tensor bytes, coefficients, treatment activity, dose and RNG
   identity; no tolerance relaxation or outcome-based candidate removal.
3. A separately frozen receiving-device full-planner check, then a separately
   frozen paired behavioral registry/analysis with the existing native and
   component controls. Keep 96 total episodes per task/condition/checkpoint,
   not per GPU; reusing a reference requires exact compatible provenance.
   This stage is not implemented or authorized merely by passing steps 1/2.

Step 2 is bounded to 1,200 seconds internally, with a 1,210-second outer child
limit and no automatic retry. Optional component timing must not displace the
required equivalence cases. Engineering cases do not enter efficacy analysis.

The native MetaWorld loader normally scans the whole pool for normalization.
This new check must not silently construct that loader or open protected rows.
Reuse the pinned upstream `DATA_STATS["metaworld"]` planning normalization and
an explicitly bound normalized fitting stimulus, retaining original source-parquet/row
mapping. This is a new engineering-input freeze, not a claim that the published
constants are bitwise equal to statistics recomputed by an earlier offline loader.
The fixed banks and their original fit provenance are unchanged. Retain the
original fitting image transform and compare the selected-row preparation against
the actual native `get_frames`/`__getitem__` implementation with identical RNG.
Only the original first eligible fitting row is decoded; other parquet files
are byte-hashed and their row-count metadata inspected, not their outcomes.
Raw-file hashing is not
permission to decode protected observations or evaluate outcomes. If the
stimulus/normalization provenance is absent, receiving execution remains blocked
while local code/tests can proceed.

## Current evidence

### September8,18:50UTC receiving numerical completion

Texas50259194/GPU3 completed all18 required cases in147.43seconds. Both arms
match the independent full-native-capture/static-hook reference exactly at
every checked output horizon, including input bytes/layout, candidate records,
CPU/CUDA RNG and complete parameter/buffer bytes/modes. All30 check/operation
JSON files passed full local readback/hash comparison. ReportSHA256:
`c6faf9cd35eb981457269df3876e2226853e3b3fe2d7e8fecc52fe955ee52897`.
At300candidates/H6, three synchronized forecasts averaged3.5701seconds for the
learned combination and3.5695seconds for its control. The same-session native
comparator averaged3.1629seconds, standalone fixed map3.1733seconds and coupling
alone3.1573seconds. This is about12.9% combined forecast overhead versus native,
not a full-episode benchmark, hardware comparison or efficacy result.

`offline_study.fixed_combined_smoke` now implements the next four excluded
full-CEM episodes: native, exact native repeat, learned combination and matched
random combination. It retains all100 elementary steps,300 candidates,15 CEM
iterations per replan, the original mean-forecast calls, same scenario seed,
full call records and paired initial/goal stimuli. Short forecasts remain native.
It verifies the completed numerical report, original fitted components, every
prior source member, same physical GPU and actual model bytes. It does not use
historical cross-device goal pixels or inspect confirmation outcomes. Internal
limit3600seconds, required outer3610seconds, no automatic retry or behavioral
launch authorization. Full local suite:633tests pass(two existing skips),
including the launcher-renderer regression test; receiving science suite68pass.

The first full-planner attempt was terminal before an episode: its navigation-
derived environment omitted EGL, and this Texas image lacked generic EGL/OpenGL
dispatch libraries. The v1failed receipt remains unchanged. The corrected v2
launcher explicitly chooses EGL on physicalGPU3 and uses three hash-bound private
dispatch libraries from the already working same-Ubuntu22.04 worker, without
changing the Texas host NVIDIA driver or other jobs' environments. Actual
official simulator reset/render passed before the reviewed v2launch. Fourteen
receiving launcher tests pass, including renderer environment and library checks.

Snapshot note: archive014cf596 contains123 inert AppleDouble sidecar files as
well as123 Python sources. Preserve all246 checked members. The full-planner
snapshot adds only its Python checker: receiving hash`e64debad97734af166b4bf8b21db0d0256747895c596ee38ee7bceef175729a3`;
the executable-only local hash is`c15f5ed1a905f7b5b4d642a11ffa6da71ef04488f927d118db068dae7205e791`.
All executable filenames/bytes were independently compared equal. Metadata is
not silently removed from a completed receipt. Future new archives should
disable AppleDouble emission at creation, not rewrite executed snapshots.

### Historical CPU preparation

The adapter, checker and stimulus preparation pass44 local and44 receiving CPU
tests. The full local suite passes612 tests(two existing skips). OnIndiana,
the original selected fitting row passed exact native reader tensor/RNG parity.
All six stimulus members, its completion receipt and receiving test log passed
full local byte/hash readback; see
[preparation readback](../artifacts/offline_study/combined-fit-stimulus-preparation-20260908-v1/READBACK.json).
This cleared the missing fitting-stimulus dependency at that point, not the
then-pending receiving-GPU equivalence or full-planner checks. The subsequent
numerical completion above supplies equivalence/timing only; forecast improvement,
task-success improvement and fresh confirmation remain unestablished.
In particular, the old combined 4.824% offline improvement is not inherited.
