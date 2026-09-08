# Fixed-response successor — user-approved Design 1

Decision recorded September 7, 2026 (EDT), before new fitting or behavioral outcomes.
The user supplied [rank 4 intervention.md](../rank%204%20intervention.md), which links
to [Rank Edit.md](../Rank%20Edit.md). Both user-authored notes are preserved unchanged.
Select that note's **Design 1**, its explicitly recommended first engineering choice.
Designs 2/3 and the four supporting diagnostic proposals are not an automatic search
queue. This is a new method, `fixed_response_rank4_v1`, not a numerically equivalent
optimization of the frozen candidate-dependent solver.

## What changes and what does not

Keep the existing task/checkpoint-specific four PCA output directions, the three-score
H6 visual-error readout, original task-specific dose, all 256 patches, B3/H3, and H6-only
scope. Fit the response inverse offline. Apply the map inside the existing forward pass:

`phi = [1, ((x - mean) @ projection.T) / scale]`

`A = solve(Jbar.T @ Jbar + lambda * I, Jbar.T @ E)`

`c_raw = A @ phi; delta_raw = B @ c_raw`

`c = dose * c_raw / ||delta_raw||` if `||delta_raw|| > 1e-10`, otherwise zero.

Here mathematical `Jbar` has four columns; the code stores its transpose as four
response rows. Damping is `0.01 * max(mean(diag(Jbar.T @ Jbar)), 1e-18)`.
The intercept is **first**. Projection, map, expansion and dose arithmetic are FP32,
even inside BF16 model inference. As in the source finite-difference hook, cast the
edit to activation dtype before addition; the changed activation stays in model dtype.
Report requested **and realized** energy, including BF16 rounding loss.

One forecast calls the backend once, retaining the full candidate batch. There are
**zero online response probes and zero separate native-shadow forecasts**. Shorter
horizons and native/zero-dose arms dispatch the true native backend. Zero treatment
is per arm/candidate, not the old common-degeneracy test across ranks 1/4/8. No rank
1 or 8 response calculations are part of this successor.

## Fixed fitting protocol

Executable configuration: [fixed_response_rank4.json](../configs/fixed_response_rank4.json).
Initial tasks: Reach and Reach-Wall, using each original verified BF16 fitting bank.
The broader study still has six tasks; this does not claim a universal fitted matrix.

- Validate the original cohort, fit receipt, bank/protocol/checkpoint hashes and parity
  receipt before model calls. Never fit from development or protected trajectories.
- Order the original 128 eligible fit families by SHA256 of `2026090802:lineage_group`.
  The first 24 calibrate the response mean; the next eight audit it. Selection does
  not depend on forecast or behavioral outcomes.
- Use the same four evenly spaced author clip/prefix pairs per selected family and
  their recorded action sequences. This is **not** four alternative physical futures
  from one state, and not a sample of CEM-proposed actions.
- Collect central finite differences with the source response radius. Four learned
  axes and four matched-random axes, positive/negative: 16 offline probe forecasts
  per example, chunked by four. Across 32 x 4 examples: 2,048 probe forecasts/task.
- Average the 96 calibration responses with equal family weight, then compose once
  with the inherited readout. Do not average per-example inverses. The random control
  receives its own response calibration with the same budget and zero/dose rule.
- Retain the eight-family response audit without refitting from its outputs. Record
  response deviation, delivered coefficients, both embedding endpoints H1–H6, dose
  and identity checks. These eight families were already used to fit the inherited
  PCA/readout: they are held out **only from response calibration**, not an independent
  test of the whole operator, and never fresh confirmation.
- Bound collection to one hour/task. Failures and partial artifacts are retained but
  cannot become a completed fit. Save source-bound contracts before collection and
  hash-bound completion receipts afterward. A fit receipt cannot trigger planning.

The standalone registry is native, zero dose, fixed map and its matched-random map.
Random matching refers to rank/support/orthonormal basis-projector spectrum and dose;
it does not assert identical coefficient-map singular values.

## Sequence before behavioral launches

1. **Local implementation/tests:** composition, one-call hook, fitting-only collector,
   source access/coverage checks, and independent zero/random controls.
2. **Real fitting and engineering:** run the bounded calibration, then validate the
   same one-pass method on the actual strict-FP32 author planner. Use unchanged H6,
   full 300-candidate populations, native CEM iterations, and excluded engineering
   scenarios. Measure full planning latency/memory, not just the 4x4 multiply.
   Inspect action-ranking fidelity separately; model-probe responses are not physical
   counterfactual ground truth. No speedup or preserved efficacy is established yet.
3. **New development comparisons:** freeze the successor's exact source/bank bindings,
   complete paired arms, seeds, 96 total episodes/task/condition/checkpoint (not per
   GPU), endpoints and simultaneous analysis before new behavioral outcomes. Retain
   equal-budget coupling and its matched random as cheap comparators. Re-evaluate
   forecast endpoints for this new operator; do not relabel the old results. Lack of
   significant offline gain alone is not a behavioral admission veto.
4. **Confirmation/history:** after a complete development selection and final freeze,
   use separately audited untouched scenarios. The required three trained seeds and
   late-checkpoint study remain unfinished; repeating one released checkpoint is not
   training variability. Earlier correct measurements need not be rerun merely because
   a separately named successor is added.

The new one-pass **combined** method is deliberately not enabled. Upstream coupling
changes H3, whereas this map consumes native H3. The implementation rejects other
predictor hooks. Coupled-path training or pre-coupling features need an explicit
combined contract; retaining a shadow would violate the selected zero-shadow contract.
Do not call naive hook composition a preservation of the old combined arm's 4.824% gain.
The old expensive rank/combined workers remain stopped. No HMM, spatial or layer
successor is silently introduced by this change.

## Execution and evidence status

September8,06:01UTC: all eight receiving-device full-CEM checks passed, each with
the four complete episodes and device-bound WORKER receipt. The frozen five-arm
MetaWorld queues advanced automatically into actual scientific closed-loop
development. This is a launched comparison, not a completed efficacy result.

September8,05:52UTC: all four successor offline task/precision scopes are complete,
with all input/report hashes verified locally. Primary BF16 H6 proprioceptive
embedding-error reductions versus native are2.360% Reach and2.192% Reach-Wall;
FP32 sensitivities are2.324% and2.131%. Both beat their random controls statistically;
Reach FP32 does not clear the practical minimum against random. Exact intervals
and verification roots appear in [execution status](../reports/EXECUTION_STATUS.md).
These outcomes neither admit nor veto the already frozen behavioral arms.

All eight new US worker devices are executing their separate complete receiving
CEM checks, automatically followed by the original five-arm/960-episode registry.
No new scientific closed-loop result is yet complete. Navigation's entire existing
coupling factorial is separately frozen and in engineering on two other GPUs;
this addresses behavioral alternatives independently of offline significance.

September8,05:17UTC update: both complete four-episode full-CEM engineering panels
passed onIndiana, including exact native repeats, exact paired stimuli, unchanged
300x15 CEM and100-step episodes. Complete reports/call/action traces were copied
locally and reverified against every receipt. Reach overall report SHA
`ad7d286c3eea7b4f4613f7c8d850e12131fb9452c9b2b74d5832536e32da725b`;
Reach-Wall`84e0fcd4251bad6124876e10d4bc47e3c6ea976aedebb0284e837dfa82676694`.
Reach native/new-map episode times314.28/313.03seconds on this excluded scenario;
this establishes practical engineering feasibility, not efficacy or population-wide
latency equivalence. Indiana resumed the unchanged non-rank navigation queue.

Nebraska50231985 runs frozen offline retry-v2 onGPU0Reach/GPU1Reach-Wall, all
four arms/BF16 thenFP32. Original v1 failed before evaluation because the new
image lacked FFmpeg; both failure roots/logs are preserved. The runtime-only
correction installed the same UbuntuFFmpeg4.4.2 package asIndiana; scientific
sourcef02e8168/protocol/data/arms remain unchanged. SixBF16 shards were complete
at05:16UTC; full analysis remains pending.

### Separately frozen behavioral successor panel

`fixed_response_behavior.py` and`fixed_response_behavior_analysis.py` implement
the separate five-arm Reach/Reach-Wall panel: native, fixed map, its matched
random, existing equal-budget coupling, its matched random. All96 canonical
previously exposed development scenarios per task/arm, eight logical RNG streams,
100elementary steps,300candidates,15CEM iterations and strictFP32 remain fixed.
This is960scientific episodes total, not96perGPU. Native is rerun on the receiving
GPU class; old5090 outcomes are not silently reused as a new4090 reference.

The complete five-arm/two-task registry must finish before selection. Eight
contrasts (each candidate versus native and its own random, across two tasks)
use the existing20,000-draw seed2026090722 paired scenario-cluster bootstrap,
Bonferroni simultaneous95% intervals and Holm-adjusted exact paired discordance
tests when scenario clusters have homogeneous binary results. The inherited
5percentage-point practical gain and positive native/random lower-bound selection
rule remain; no offline-significance admission rule is introduced. Repeated
initial/goal scenarios stay clustered. This does not enable combined/HMM or open
reserved base-1 confirmation. The full six-task/history track remains required.

Freeze`behavioral-freeze-v1` under the local successor evidence root binds
protocolSHA`853e8bf6b71c65c1e61d249ba84d8d57d6172ba1bd3a15e0424dc2d9e32cfa1f`
and sourceSHA`fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58`.
The immutable code-v4 staged on new ownedUS8RTX4090 instance50233992 includes
receiving-device full-CEM/native-repeat checks before automatic scientific queues.
GPU0..3Reach/GPU4..7Reach-Wall each receive two disjoint12-episode logical streams
for all five arms. No candidate-success panel has completed.208local tests pass,
one skipped; the new tests reject missing conditions/episodes, changed goals/seeds,
extra model calls and selection that fails its random control. Archived canonical
goal images were restored from the verified private Drive snapshot, not regenerated.

The earlier05:17 update supersedes the historical04:41 progress below.

Updated September8 at04:41UTC: both real fits and actual-GPU numerical checks
completed and their full fitting bundles/report hashes verified locally. Each task
used24 calibration families and8 response-audit families as frozen; fitting took
100.42seconds forReach and94.50seconds forReach-Wall. These are fitting results,
not forecast-development or task-success improvements. The full local suite now
reports203 tests, one skipped.

The numerical checks passed single-backend-call, zero-dose/short-horizon identity
and bitwise independently compiled field parity for both active arms at8/19/300
candidates. FullCEM engineering is running onIndiana50205763 in a new immutable
code-v2 snapshot. OnReach, the native100-step episode and exact native repeat
completed in314.28/312.06seconds before starting the new operator. A completed
four-episode engineering panel is still required; its outcomes cannot select a
candidate or count toward96 scientific episodes.

Fitting report hashes: Reach
`44fc09fcc65071c269202aa24cbf874f6ba66d557ce42cc1c12a76b8c00bf8d4`;
Reach-Wall`b694fdcdb161e87f18c50fdbf269e199a3777908391d3da5f3dcd63ec4846242`.
Numerical report hashes: Reach
`e0008e34e8573c00cce51a367eec921d1797dacf5cbd8c2dd14ffd3b534eafb1`;
Reach-Wall`0cc36fd6fb02ff498a4cf4d8d742c2d7bfe6e89f94cd7c5b0a13c44efe331dbe`.
Verified local evidence: `artifacts/offline_study/fixed-response-20260908-v1/`
under`fits/` and`numerical-checks/`.

## Frozen successor offline rerun

`offline_study.fixed_response_offline` has separate freeze/evaluate/analyze modes.
The new append-only access bundle verifies the original seven-row authorization
and all ten completed prior analysis scopes per task. It retains the old manifest
and protected labels, authorizing ONLY the previously exposed full Reach33 and
Reach-Wall27 pools under the user's successor-rerun request. No other protected
rows are opened, and no old fit/protocol/result is rebound to the new method.

All four registered arms use every original author clip/prefix:3960H6 prefixes
per arm forReach and3240forReach-Wall. Four disjoint trajectory shards per task
and precision preserve the independent33/27lineage counts. BF16 is primary;
FP32 is sensitivity using the SAME frozen BF16-fitted map, not a separately
retuned map. Each evaluator first checks actual-source native parity and zero-dose
identity on a fitting trajectory. Both H6 embedding endpoints and all four
contrasts form the simultaneous inference family within task/precision, with
the inherited BF16 fitting-only minimum as the common reference. Neither precision
nor offline significance may select admission to the behavioral stage.

Active freezes are `offline-freeze-v2/{reach,reach-wall}` under the local evidence
root, binding sourceSHA`f02e81684d3abeda35c69ce59bdf4f982406c70766c406714de6c08d68af6b33`.
The preliminary Reach`offline-freeze/` is preserved but superseded BEFORE new
outcome access: v2 additionally validates every row's exact lineage/task/prefix
metadata rather than only counting trajectory/start/arm tuples. No results were
produced by that preliminary freeze. Staging onNebraska50231985 is not a completed
rerun or a behavioral result.

Implementation: [fixed_response.py](../src/offline_study/fixed_response.py),
[fitting CLI](../src/offline_study/fixed_response_fit.py). The CLI takes `--config`,
`--cohort`, `--fit` (original operator_rank directory), `--vendor`, `--checkpoint`,
`--checkpoint-sha256`, `--data-root`, `--device`, and a new `--output` directory.
Stage immutable source/config on an exclusively assigned US GPU before executing;
check the shared GPU board and aggregate $7/hour authorization. Never overwrite
ongoing jobs or restart their old expensive dispatchers.
