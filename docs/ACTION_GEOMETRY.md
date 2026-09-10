# Frozen reference for the action-response geometry sweep

Declared before any geometry development measurements on 2026-09-07. This
implements the next comparison in EXPERIMENT_PLAN.md. It is a mechanistic
extension of JEPA-WM, not an experiment reported by the authors.

- Tasks: Reach, Reach-Wall, Push-T; official task-specific frozen checkpoints and
  pinned upstream source, strict FP32 model execution.
- Recipient: original recorded six-step action sequence and its original observed
  context. H3 alone varies in donor passes. Future observations are scoring targets.
- Action direction: unit Rademacher vector, seed 2026090710, in the flattened,
  normalized five-elementary-action coordinates of H3. One fixed direction per
  task/action dimensionality; no direction search, clipping, or second normalization.
- Radius: 0.1 normalized L2 units. Four donor coordinates are -1, -0.5, 0.5, 1;
  evaluation coordinate 0 is excluded from both interpolation estimators.
- Site: block output P3 (zero-indexed), H3, newest 256 patch positions and their
  entire block feature vector, including any concatenated proprioception features.
  No pooling or PCA; each arm delivers one residual vector per window.
- Affine weights: 1/4 each. Cubic weights: -1/6, 2/3, 2/3, -1/6. Arithmetic is
  centered and evaluated in float64, then cast to the native activation dtype.
  At this symmetric coordinate cubic equals quadratic least squares; this cannot
  uniquely identify third-order dynamics.
- Project the cubic estimate onto the infinite line through the endpoint donors.
  Reflect its orthogonal component with `2 * projection - cubic`.
- Raw diagnostics: omitted-center activation error and H3/H6 native visual and
  proprio forecast fidelity for each unscaled estimate. These retain the exact
  projection/reflection construction.
- Efficacy arms: normalize each reconstruction residual from the native recipient
  to a common L2 dose. The dose is the median affine reconstruction residual norm
  on the fitting population. The matched random unit vector uses seed 2026090711.
  This normalization changes the raw geometry; interpret these comparisons as
  equal-energy directions derived from the estimators, not exact reflected paths.
- Fitting: 128 hash-selected fitting families, one deterministic row per family,
  four registered windows, using the existing selection seed. Direction and radius
  do not depend on outcomes. Development metadata only allocates bank rows.
- Degeneracy: flag endpoint chords whose squared norm is at most 1e-12 times the
  largest squared displacement from the first donor, or 1e-20. If any mechanism
  residual norm is at most 1e-10, also flag the window. Every efficacy arm becomes
  zero dose for such a window; retain it in every paired population average.
- Physical error: H6 proprio MSE, scored only for the original-action recipient.
  Donor predictions have no counterfactual physical ground truth.
- Primary contrasts: cubic versus affine/projected/reflected, plus each of the four
  mechanism arms versus native and matched random. Use the same development
  trajectory/window population for every arm, with lineage-level aggregation.
- Development intervals: 10,000 joint bootstrap draws over aligned lineage groups,
  simultaneous 95% intervals using the maximum standardized centered bootstrap
  deviation across the eleven registered contrasts. Seed 2026090704. Task-specific
  smallest useful effect and equivalence margin both equal 1% of fitting native
  H6 proprio MSE and are frozen in the task protocol before development execution.
  These are development selection criteria, not power or confirmation claims.

Eligibility requires identity checks and improvement beyond the frozen margin
against both native and random. All four arms and all original records are retained
regardless of results. No routing, new dose, rank, or direction follows automatically
from the result. The later registered categories use their own frozen references.

Sources: [JEPA-WM v4](https://arxiv.org/html/2512.24497v4),
[pinned preprocessor](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/datasets/preprocessor.py),
[GPU scaling chapter](https://jax-ml.github.io/scaling-book/gpus/).
The chapter motivates batching common donor/arm work and measuring data movement;
its low-precision training throughput is not a forecast of this FP32 workload.
