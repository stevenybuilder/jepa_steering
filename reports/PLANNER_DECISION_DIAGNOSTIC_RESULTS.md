# Completed common-action planner-decision diagnostic

This is a **development diagnostic**, not a new full-task success trial.
Reach and Reach-Wall each contribute96 canonical scenarios and five conditions.
Each model scores the same initial300-candidate H6 population; its selected
first15 elementary actions are executed without replanning. Identical selected
candidates may share the same verified physical outcome within a scenario.

Of eight simultaneous primary comparisons, 0 have intervals entirely
above zero (better short-prefix progress) and 0 entirely below zero.
These counts do not establish improved full-episode task success.

## All frozen physical-progress contrasts

Positive values mean the treatment ends closer to the expert end-effector
goal than its reference. Units below are millimeters; the bootstrap unit
is a whole scenario, not a candidate action or repeated condition.

| Task | Treatment | Reference | Mean distance gain, mm | Simultaneous95% interval, mm | Changed choices /96 |
|---|---|---|---:|---:|---:|
| reach | fixed_rank4 | native | +0.0000 | [+0.0000, +0.0000] | 0 |
| reach | fixed_rank4 | matched_random_fixed_rank4 | +0.0000 | [+0.0000, +0.0000] | 0 |
| reach | coupling_only | native | +0.1936 | [+0.0000, +0.9679] | 1 |
| reach | coupling_only | matched_random_coupling | +0.1936 | [+0.0000, +0.9679] | 1 |
| reach-wall | fixed_rank4 | native | -0.5242 | [-2.0967, +0.0000] | 1 |
| reach-wall | fixed_rank4 | matched_random_fixed_rank4 | -0.5242 | [-2.0967, +0.0000] | 1 |
| reach-wall | coupling_only | native | +0.0119 | [-0.8424, +0.8902] | 2 |
| reach-wall | coupling_only | matched_random_coupling | -0.2818 | [-1.5961, +0.8902] | 3 |

## Candidate-ranking diagnostics

These summaries compare each condition with native. Rank agreement is
descriptive and does not identify the best physical action among all300.

| Task | Condition | Changed selections /96 | Mean Spearman | Mean top10 overlap |
|---|---|---:|---:|---:|
| reach | native | 0 | 1.000000 | 1.0000 |
| reach | fixed_rank4 | 0 | 0.999983 | 0.9969 |
| reach | matched_random_fixed_rank4 | 0 | 0.999986 | 0.9958 |
| reach | coupling_only | 1 | 0.999922 | 0.9885 |
| reach | matched_random_coupling | 0 | 0.999973 | 0.9969 |
| reach-wall | native | 0 | 1.000000 | 1.0000 |
| reach-wall | fixed_rank4 | 1 | 0.999980 | 0.9958 |
| reach-wall | matched_random_fixed_rank4 | 0 | 0.999979 | 0.9948 |
| reach-wall | coupling_only | 2 | 0.999885 | 0.9875 |
| reach-wall | matched_random_coupling | 1 | 0.999960 | 0.9917 |

## Interpretation boundaries

- This tests the first CEM population, not its final optimized elite mean.
- H6 ranking and15-step physical progress have different horizons.
- The cohort was already used for development; it is not fresh confirmation.
- One released checkpoint is not three training seeds or a checkpoint history.
- This cannot by itself explain the separate full-rollout successes/failures.
- The [completed960-episode panel](CORE_METAWORLD_BEHAVIORAL_RESULTS.md)
  did not establish learned-intervention task-success improvements.

All raw inputs, scores, chosen actions, physical trajectories, failed setup
attempts and runtime provenance are included in the separate preservation
archive. See `artifacts/offline_study/decision-closeout-20260910-v1/` for
independent CPU verification, Drive readback, and rental-release receipts.

Protocol SHA256: `ca0061a6861a9f0ddb2694d9277cffff142909db69eb8b75a6316ab85bb328dd`.
