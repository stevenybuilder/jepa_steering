# Push-T fixed coupling behavioral development — 2026-09-08

This extends the approved other-ablation behavioral coverage to Push-T. It does
not replace the completed offline protocols, inspect partial outcomes to select
arms, restart the expensive operator, or open fresh confirmation.

Reuse the already fitted BF16 Push-T vision/action coupling bank, whose 128 fit
families exclude every released validation family. Keep all nine scientific arms:
native, visual-only, action-conditioning-only, joint, equal-standardized-energy
joint, permuted visual, permuted joint, matched random, and equal-energy matched
random. Zero-dose is a full-episode identity check, not another efficacy arm.
No dose, support, tensor or fit procedure is selected from behavioral outcomes.

Use the unchanged released Push-T planner and simulator: context2, H6,
300 CEM candidates, 10 elites, 30 iterations, six executed model actions with
five elementary actions each, 30 elementary simulation steps, FP32/no TF32.
Use the actual upstream validation-row/segment sampler and all21 released rows.
The existing native reference has 96 TOTAL episodes in eight persistent12-episode
streams, development base2026090721; additional GPUs only divide those streams.
The reference may be reused only after the assigned complete stream receipts, exact input
hashes, source and receiving-device native/zero-dose engineering verify.

The new finite registry can be frozen before the native reference completes.
Engineering uses the same excluded train10810 fitting scenario as the original
native check; no validation outcomes are needed. Each receiving GPU must pass
two complete native repetitions, full zero-dose identity, and all eight full
candidate episodes. Independent source-compiler parity must hold for both the
300 candidates and the single CEM mean. Candidate execution requires each assigned
complete12-episode native reference stream, frozen source/fit/input bindings and
that GPU's proof. Comparative analysis still requires all96 native episodes.
The CPU-only v1 freeze is preserved but superseded by v2 before any candidate GPU
engineering or outcomes: waiting for all96 native episodes before working on an
already complete paired stream was an unnecessary scheduling dependency. No arm,
episode, dose, RNG sequence, endpoint or statistical criterion changed.

Pair every candidate with native on episode/RNG-stream identity, exact rendered
initial/goal inputs, source trajectory/index/family, sampled state/action segment
hashes and the full31-frame segment. Preserve the actual persistent native RNG
streams; do not use per-episode reseeding or outcome-dependent retries.

## Fixed analysis

- Primary endpoint: official simulator binary success after the complete episode.
- Report successes/96 and episode-weighted percentage, matching the native panel.
- Uncertainty: paired cluster bootstrap over the ORIGINAL source initial-state
  families, not sampled crops, repeated episodes or candidate action sequences.
  Report realized family count (at most21), per-family draw counts and the
  equal-family-weighted success percentage as a descriptive sensitivity.
- Retain the existing15 nonzero-dose pairwise contrasts and add the predefined
  success interaction joint−visual−action+native. These16 contrasts form this
  separately labelled Push-T family, not a global six-task significance claim.
- 20,000 bootstrap draws, seed2026090801, Bonferroni simultaneous95% intervals
  across16 contrasts. Minimum useful gain5 percentage points, as in the existing
  behavioral-development plan; this is our prospective rule, not an author quote.
- Require the complete nine-arm panel before comparative analysis; no selecting,
  dropping or stopping an arm from partial outcomes. Retain negative results.
- Report paired discordances and a clearly labelled approximate detectable-effect
  sensitivity from family-cluster variability, not an assertion that96 is powered.
- Runtime, action traces and delivered edit energy remain diagnostics, not success.

The released validation families are already replication-exposed. This panel is
development/replication, NOT fresh-family confirmation or three-seed/checkpoint
history replication. Genuine confirmation requires the separately frozen fresh
families. Geometry, support, HMM and other task comparisons remain required work.

Executable: `offline_study.pusht_coupling_behavior`. A prepared module or passing
unit test is not proof of completed GPU engineering or measured efficacy.

Native methodological sources remain the pinned authors' code, especially
`evals/simu_env_planning/planning/plan_evaluator.py`,
`app/plan_common/datasets/pusht_dset.py` and the config named in
`offline_study.planning_contract`. Existing source receipts pin revision
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`; no simulator/library substitution.
