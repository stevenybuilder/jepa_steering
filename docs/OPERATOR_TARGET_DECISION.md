# Shared target for the remaining rank and support comparisons

Status: proposal prepared while geometry development runs; awaiting the user's
target choice. No rank/layer/spatial fitting or evaluation has been launched.

The active plan fixes ranks 1/4/8, singleton/all-block and spatial supports, and
matched controls, but leaves the semantic target and correction rule unspecified.
Those choices affect what the experiment measures and must be recorded once
before all three sweeps.

## Recommended: recorded-future forecast correction

Use the archived native-coordinate correction algorithm as the starting point:
`archive/2026-09-07-workspace/scripts/geometry_map/run_reach_native_coordinates.py`,
especially `fit_error_target` and `correction` (lines 41 and 52). It predicts
`actual recorded visual embedding − native predicted visual embedding` from native
P3 activations, then solves a damped response-inverse system to obtain an edit.

For this study, specify H6 as the residual target, matching the final forecast
horizon. The archive used H3; changing that horizon is an explicit extension.
Fit the target predictor once per task on the existing audited fitting families,
with a fixed descriptor/PCA capacity and regularization independent of operator
rank. Never use the development recipient's actual future to construct its edit.
Both visual and proprioception H6 errors remain reported; the plan's frozen
proprioception endpoint still governs advancement, so visual reconstruction gains
alone cannot establish useful physical prediction.

The three comparisons then share that target predictor and correction rule:

- Rank: a single fitting procedure yields a nested basis, evaluated at ranks 1, 4,
  and 8. Rank-specific target predictors are forbidden because they would change
  both target-prediction capacity and intervention capacity.
- Layer: fit one rank-one basis in each registered direct-sum block space and
  split it into block-specific slices, referenced to the same unedited rollout.
- Spatial: choose the single patch, 4x4 region, and scattered set from fitting
  data only, freeze them, and fit rank one on each registered support.

Every mechanism has its prescribed support/rank/spectrum-matched random control.
Normalize actual delivered energy to the same fitting-derived total. Collect and
report local model-response work separately in throughput. Resolve and freeze the
target-predictor capacity, ridge/damping, perturbation radius, common dose, and
simultaneous contrast family in the executable protocol before any development run.

This tests whether a distributed correction improves recorded future prediction.
It does not directly train a goal-seeking policy or establish closed-loop success.

## Alternative: a physical task quantity

Use a validated task-specific quantity, such as hand-to-goal distance for Reach
and Reach-Wall or object-to-target overlap for Push-T. This is a semantic steering
question and requires a precise desired sign/value and validated physical labels.
The archive's coordinate donor method, `run_residual_coordinate_time.py:229`, is
not a drop-in solution: donor selection changes with rank and its cap does not
equalize delivered energy. Those confounds must be removed before execution.

Whichever target is chosen is shared across ranks and support comparisons; it is
not selected separately to favor each arm. Historical fits and historical small
development cohorts cannot replace recollection on the corrected lineage splits.
