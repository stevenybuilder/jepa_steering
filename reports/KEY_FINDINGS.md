# Ten findings from the JEPA-WM intervention study

Evidence reviewed September 11, 2026. This is a retrospective synthesis of
existing results, not a new experiment, hypothesis registry or confirmation
analysis. The ordering reflects explanatory interest, not ten independent
discoveries or a new significance ranking.

**Bottom line:** there are positive offline forecast effects, measured
task/precision boundaries, and a completed but inconclusive behavioral panel.
There is no established learned-intervention task-success improvement, fresh
confirmation, or recovered causal mechanism.

## 1. The inexpensive replacement retains positive forecast effects

The refined fixed-response edit reduces BF16 H6 proprioceptive embedding MSE by
**2.360% on Reach** and **2.192% on Reach-Wall**. Simultaneous 95% intervals are
[1.966%, 2.755%] and [1.798%, 2.587%], respectively. Both also beat their
matched-random controls on this offline endpoint. Visual-embedding reductions
are smaller: **0.427% and 0.399%**.

The edit applies four fitted directions in one batched forward pass, without
online response probes or separate native shadows. Mean full-episode overhead
in the behavioral panel was approximately one second on roughly 300-second
episodes; that is descriptive timing, not a controlled speed benchmark or total
research cost. This is a separately validated successor, not an equivalent
implementation of the original operator.

**Meaning:** compact, cheap corrections can change useful prediction metrics;
this does not establish better actions. Sources: [operator and rerun](../docs/FIXED_RESPONSE_RANK4.md),
[behavioral timings](CORE_METAWORLD_BEHAVIORAL_RESULTS.md).

## 2. The original combined edit has complementary offline benefits

On Reach, combining equal-budget vision–action coupling with the original
rank-4 operator reduces the primary embedding error by **4.824%**, with
simultaneous interval **[3.668%, 5.980%]**. It passes the frozen contrasts against
its matched-random combined control and both component-removal arms.

**Meaning:** retaining both components helps this forecast endpoint. This is
evidence of complementary effects, not proof of mechanistic synergy. The number
cannot be attributed to the replacement combined method, whose behavioral panel
is unfinished. Source: [combined/drop-one results](../docs/COMBINED_DEVELOPMENT.md).

## 3. Better forecasts have not established better task success

The complete MetaWorld panel has **960 full closed-loop episodes**, covering two
tasks, five conditions and 96 paired scenarios per task. Neither learned edit
establishes improvement over both unsteered and its matched-random control.
All eight simultaneous intervals include zero.

Reach's refined edit scores **49/96**, versus **43/96 unsteered** and **48/96
matched random**. Its +6.25-percentage-point estimate versus unsteered has an
interval of **[−11.46, +23.96]**; versus random the estimate is only +1.04 points.
The random-coupling arm has the highest observed Reach count, 58/96, but was not
selected as a newly validated method.

**Meaning:** behavioral usefulness and direction-specific benefit remain
unresolved. This is not evidence of an exactly zero effect. The general
forecast/planning distinction is already discussed by JEPA-WM; this project
adds these particular controlled measurements, not discovery of that general
principle. Source: [complete behavioral results](CORE_METAWORLD_BEHAVIORAL_RESULTS.md).

## 4. Net success hides substantial rescues and regressions

Relative to unsteered, the refined edit turns **23 Reach failures into successes**
but turns **17 successes into failures**. Forty of the 96 paired outcomes change,
leaving six additional successes in the net count. The unadjusted exact paired
p-value is approximately **0.430**; multiplicity adjustment alone is not why
the result is inconclusive.

**Meaning:** the full-rollout intervention is not behaviorally inert, but we
have not identified which physical situations it reliably helps or harms.
Selecting favorable subgroups after seeing these outcomes would be exploratory.
Source: [paired win/loss counts](wm-approaches/core-analysis.json).

## 5. Initial action selection barely changes in the diagnostic

Across 192 scenarios, the refined edit changes the selected first-population
action sequence in **1/192**, and coupling in **3/192**. Mean rank correlations
with unsteered exceed 0.9998. No adjusted short-prefix progress comparison
establishes improvement.

**Meaning:** the fitted changes have little effect on this particular initial
ranking decision. This does not contradict changed full-rollout outcomes:
the diagnostic tests only the first CEM population and 15 actions without
replanning, not later optimization iterations or the entire closed loop.
Source: [decision diagnostic](PLANNER_DECISION_DIAGNOSTIC_RESULTS.md).

## 6. Broad spatial support mattered under the tested rules

In the original Reach spatial sweep, **only the all-patch candidate passed the
frozen advancement gates**. One-patch, contiguous-16 and scattered-16 alternatives
did not qualify under those rules.

**Meaning:** the tested evidence favors spatially broad support for this offline
setup. It does not prove that every localized intervention fails, and no complete
spatial-support behavioral comparison exists. Source: [corrected offline selection](CORRECTED_OFFLINE_RESULTS.md).

## 7. Spatial distribution is not the same as layer distribution

The original Reach **single-block B0** arm reduces the forecast error by
**3.026%**, interval **[2.124%, 3.929%]**. Multiple individual-layer choices were
eligible; there is no uniquely identified causal block.

**Meaning:** the results do not support the blanket claim that effective edits
must span many layers. A spatially distributed edit can operate at one layer;
the refined behavioral edit does exactly that at its registered block/time site.
Sources: [offline results](CORRECTED_OFFLINE_RESULTS.md),
[refined operator contract](../docs/FIXED_RESPONSE_RANK4.md).

## 8. Push-T exposes a task-dependent limit

On Push-T, original joint coupling **increases** H6 proprioceptive embedding MSE
by **0.638%**: the reduction is −0.638%, with interval [−1.214%, −0.062%]. The
original rank-4 estimate is only **+0.198% error reduction**, interval
[−0.059%, +0.455%], and does not qualify.

**Meaning:** forecast effects are not uniformly positive across tasks, even with
task-specific fitting. This is not a transfer experiment with one unchanged
matrix, nor a completed Push-T behavioral null result. Source:
[corrected three-task offline results](CORRECTED_OFFLINE_RESULTS.md).

## 9. A dramatic geometry diagnostic is precision-dependent

On the corrected 21-trajectory Push-T pool, cubic interpolation has approximately
**1,192× lower omitted-activation reconstruction MSE than equal-anchor linear
interpolation in FP32**. In BF16, cubic instead has **35.15% higher error**.
These ratios compare group-weighted mean reconstruction errors and are
descriptive, without a ratio confidence interval.

| Precision | Linear reconstruction MSE | Cubic reconstruction MSE |
|---|---:|---:|
| FP32 | 6.551584 × 10⁻⁵ | 5.498446 × 10⁻⁸ |
| BF16 | 5.615199 × 10⁻³ | 7.588738 × 10⁻³ |

The dose-controlled cubic edit's forecast reduction versus unsteered is only
**0.0046% in FP32**, interval [−0.0032%, +0.0124%], and **0.0371% in BF16**,
interval [−0.1203%, +0.1945%]. Both intervals include zero.

**Meaning:** reconstructing the model's own internal response accurately is not
the same as correcting its prediction of the physical future. These measurements
do not establish a cubic physical law or identify the cause of the precision
difference. Sources: [geometry protocol](../docs/ACTION_GEOMETRY.md),
[existing interpretation](../wm-approaches.md#5-offline-findings-useful-signals-with-explicit-limits),
and the hash-bound aggregate reports below.

## 10. DROID supplies a completed additional endpoint, not robot success

The nine-arm DROID coupling panel completes **576 recorded-plan evaluations**:
64 paired endpoints per arm, drawn from **15 recordings**. The native score is
**51.10**, joint coupling **50.85**, and equal-energy joint **51.07**. All 16
simultaneous contrast intervals include zero.

**Meaning:** there is no established improvement in this recorded-action score.
It is not a physical task-success percentage, and 64 endpoints are not 64
independent recordings. Sources: [DROID completion and audit](ABLATION_COVERAGE_STATUS.md),
[endpoint definition](../docs/DROID_METHOD_ALIGNMENT.md).

## What this list does not establish

- HMM behavioral efficacy, a completed revised combined panel, or completed
  layer/spatial/geometry behavioral ablations.
- Full matched intervention panels for Wall, PointMaze or Push-T. Their existing
  baseline, offline and partial behavioral evidence is not discarded or called null.
- Fresh confirmation, complete three-training-seed/late-checkpoint aggregation,
  six-task generalization, or a causal explanation of task success.
- Adequate precision for small behavioral effects merely because there are 960
  evaluations: the independent unit remains the paired scenario, 96 per task.

The interrupted additional confirmation run contains **117 complete native-only
episodes**, not new learned-intervention efficacy evidence. Existing completed
analyses remain unchanged. See [latest execution status](EXECUTION_STATUS.md).

## Additional numerical provenance

For this synthesis, the four aggregate files below were read locally and their
SHA256 values matched their original `DONE.json` receipts. Ratios in finding 9
were recalculated from `mechanism_diagnostics.per_task_group_weighted.pusht.metrics`;
no outcomes, contrasts or selection rules were changed. These large evidence
trees are restored from private archives, not embedded in Git.

| Evidence | Report SHA256 |
|---|---|
| Refined Reach, BF16 | `3f00270e9750de568ff8843278e36de65fef4b6b8f499b9909dadb87c1100558` |
| Refined Reach-Wall, BF16 | `b265a2d33d22988010536a2194cbd853e5d5a6d57cf3962ae49fe706255805ba` |
| Push-T geometry, FP32 | `457e33775369b171afb4e5d3f1e3d86d62fc3352da56b573ca0f58e06db618fd` |
| Push-T geometry, BF16 | `a9a4717b04a044a20dc502467d1fd606fe4c08e9c86725e419bd34d08b2a3246` |

Refined aggregate roots:
`artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/{reach,reach-wall}/bfloat16/report.json`.
Geometry aggregate roots:
`artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/{float32,bfloat16}/pusht/action_response_geometry/report.json`.

The tracked [core analysis](wm-approaches/core-analysis.json) and
[verification receipt](wm-approaches/core-verification.json) provide the paired
behavioral evidence. The [diagnostic report](PLANNER_DECISION_DIAGNOSTIC_RESULTS.md)
records its frozen protocol and preservation location. This synthesis does not
relabel numerical verification as independent scientific confirmation.
