# Frozen rank, layer, and spatial reference — 2026-09-07

User authorization: use all existing GPU instances to execute the remaining
registered tests. This document resolves the shared target proposal in
`OPERATOR_TARGET_DECISION.md`; it does not add an adaptive search. All results
remain development evidence. The governing plan is `EXPERIMENT_PLAN.md`.

## Shared construction (fixed before development)

- Same official checkpoints, strict FP32/TF32-off rollouts, recorded actions,
  audited lineage splits and four H6 windows per trajectory as completed sweeps.
- Fit exactly the deterministic 128 fitting-family representatives per task
  selected by the existing fitter seed 2026090701. No development future enters
  fitting, basis selection, radius, target prediction, or correction construction.
- Capture the newest 256 full 400-dimensional patch features at all six native
  predictor block outputs at H3. P3/all-patch features are the common descriptor.
- Shared target: actual recorded H6 visual embedding minus native H6 prediction.
  Fit raw-feature PCA3, standardize its three scores, and fit OLS with intercept
  to that full residual. Freeze one predictor per task and reuse across all arms.
  This explicitly ports the archive's PCA3+OLS target and damped local correction
  from H3 to H6; it is not the JEPA-WM authors' intervention method.
- Operator bases are raw activation principal components on the registered
  support, fit by one fixed algorithm. The rank sweep uses prefixes 1/4/8 of one
  PCA8 basis. Other supports use PCA1 in their concatenated block/patch space.
  This is the unsupervised native-coordinate PCA reference, not the archived
  iterative supervised ridge residual-basis search. No capacity is tuned.
- Spatial support score is squared cross-covariance with the first eight fitting
  H6 residual principal-component scores. Select the best patch, maximum-sum
  contiguous 4x4 region, and maximum-sum spaced 4x4 grid (stride four, one of
  sixteen offsets). Ties use row-major order. Seeded random-position controls
  choose from the same support families and preserve cardinality/topology
  and fit the same PCA1 procedure on their positions.
- Random-direction controls are seeded orthonormal bases on identical supports.
  **Spectrum matching means the rank-k orthogonal projector spectrum (k ones),
  not the singular values of the downstream model or damped inverse operator.**
- One common L2 dose and central-response radius per task: 0.005 times the RMS
  centered fitting native P3/all-patch feature norm. No development dose tuning.
- At each recipient, use a native shadow H6 rollout to predict the target. Measure
  symmetric +/- radius responses of each basis direction at H6 while editing H3.
  Solve `(R R^T + lambda I)c = R target`, with lambda equal to 0.01 times the mean
  Gram diagonal, floored at 1e-18 before scaling. Normalize the resulting edit to
  the common dose. Reuse basis responses across nested ranks and identical supports.
  Targets for these counterfactual model calls are never treated as physical truth.
- Multi-block edits are additive slices from one native-derived direct-sum vector,
  not replacements that erase earlier edits. Energy is summed over every slice.
- Audit realized FP32 addition norms against requested L2 with rtol=0.001 and
  atol=1e-5; a violation fails the job, not a retrospective tolerance change.
- If any active arm's unnormalized correction is nonfinite or has norm <=1e-10,
  fail closed for nonfinite values, or apply zero to ALL arms of that paired window
  for finite degeneracy. Keep the observation and report its flag; never drop it.

## Registered comparisons and inference

Use exactly the plan's 8 rank arms, 18 layer arms, and 13 spatial arms, including
native/zero-dose and all prescribed controls. Every mechanism compares with native
and its direction control; spatial selected supports also compare with position
controls. Include every pair of mechanisms within the category so post-selection
best-versus-global statements are covered by the frozen family.

Primary endpoint remains H6 proprioception MSE, not the visual fitting target.
Smallest useful effect and equivalence margin are both 1% of fitting-only native
H6 proprioception MSE. Per-task aligned lineage bootstrap: 10,000 resamples,
seed 2026090704, simultaneous max-standardized centered 95% intervals. No pooling
tasks or treating windows/rollouts as independent families. Eligible means a
useful improvement over native AND matched random (and position control when
registered). If none qualify retain native. Otherwise apply the plan's parsimony
review, not an automatic winner chosen from unadjusted averages.

Temporal routing, combined methods, protected evaluation, and closed-loop execution
are not silently unlocked by this implementation. Their stated gates still apply.

## Efficiency and execution limits

Following the GPU chapter's compute/memory/communication distinctions: one resident
model per GPU, disjoint trajectory shards, no inter-GPU collectives; encode each
window once across its arms and response probes; batch probes in memory-bounded
chunks; keep dense algebra and norms on GPU; transfer compact diagnostics once;
limit CPU numerical threads and prefetch decoding. Strict precision is unchanged.
Same checkpoint and frozen bank hashes across hosts. No cross-continent activation
exchange in the inference loop. Runtime smoke checks precede full execution and
cannot select scientific arms. Every launch has a distinct output root and cap.

Source: https://jax-ml.github.io/scaling-book/gpus/

Pre-development implementation correction: the first fitting source (`code-v2`)
selected the top sixteen individual patch scores, which did not enforce scattered
topology. Before any development run, replace that rule with the spaced-grid
family above; retain earlier fitting outputs and rerun fitting under `code-v3`.
This correction is geometric, not based on development outcomes.
