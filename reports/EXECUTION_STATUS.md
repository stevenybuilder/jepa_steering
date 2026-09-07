# Execution checkpoint — 2026-09-07, 20:23 UTC

The six-task study is **not complete**. The governing documents remain
[EXPERIMENT_PLAN.md](../docs/EXPERIMENT_PLAN.md),
[BEHAVIORAL_EVALUATION_AMENDMENT.md](../docs/BEHAVIORAL_EVALUATION_AMENDMENT.md),
and [DROID_METHOD_ALIGNMENT.md](../docs/DROID_METHOD_ALIGNMENT.md).
RoboCasa is excluded. No new method search or partial-outcome selection is authorized.

## Live assignments

| US instance | GPUs | Assignment | Hourly rate |
|---|---:|---|---:|
| 50125440 | 8 RTX5090 | Four native MetaWorld development shards; four fixed coupling/control shards | $2.953889 |
| 50189244 | 1 RTX PRO6000 Blackwell Server | DROID native 64-episode recorded-plan replication; Wall training pilot queued after process exit | $1.884000 |
| 50195621 | 4 RTX5090 | Hardware-specific coupling engineering, then remaining disjoint MetaWorld candidate streams automatically | $1.641667 |

Including retained stopped disks, projected aggregate is **$6.585111/hour**,
before usage-based bandwidth. Latest user ceiling is $7/hour, US-only. New
California staging took time and initially suffered an interrupted transfer;
those GPUs were not counted as scientifically active while staging. Direct
keepalive-enabled runtime transfer completed; no local large archive was needed.
The board and live provider/process checks supersede this timestamped snapshot.

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
  baseline is running, not yet a completed score.
- Initial pairing audit: 17 completed MetaWorld candidate episodes exactly
  matched their baseline initial-state vectors and initial/goal image hashes.
  This is partial **stimulus verification**, not a favorable-outcome finding.

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

The new [training pilot](../src/offline_study/training_pilot.py) is queued, **not
validated or a completed training run**. It compares an actual source-extracted
training step against 16x8 accumulated gradients, with one optimizer/scheduler
update per effective batch. It retains native sampler/drop-last layout and loss
weights; it does not claim reconstruction of the authors' exact random streams
or hardware collective reduction order. PointMaze's isolated Python3.10,
MuJoCo2.1/mujoco-py2.1.2.14 and D4RL environment imports pass; simulator rendering
and full model planning are separate checks.

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

Measured optimizations and limitations are recorded in
[jax_scaling_notes.md](../jax_scaling_notes.md).
