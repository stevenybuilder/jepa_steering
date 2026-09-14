# Data contract for the mechanism-analysis scripts

Scope: analyses that run **after** the fresh four-task confirmation
(`fresh_four_task_eight_arm_confirmation_20260912_v2`) completes. They read finished
artifacts only. They never call the model, the simulator, or a GPU. They never modify
`src/offline_study` (its hash is frozen into the protocol).

The original CPU reanalysis environment used Python 3.10, NumPy 1.26, PyTorch
2.2.2, pandas, SciPy, and Matplotlib. Current public installation requirements
and the CPU CI environment are defined in [pyproject.toml](../../pyproject.toml)
and [the test workflow](../../.github/workflows/tests.yml). Run scripts from the
repository root; restore the recorded environment when reproducing archived runs.

## 1. Fresh confirmation results tree (`--results RESULTS`)

```
RESULTS/
  engineering/<task>/<device_uuid>/            # excluded-scenario receiving checks (not science)
  <task>/scenario-NNN/                          # NNN = episode 000..095, one physical GPU per scenario
      STARTED.json  report.json  DONE.json
      native.json  fixed_rank4.json  matched_random_fixed_rank4.json  coupling_only.json
      matched_random_coupling.json  joint.json  visual_only.json  action_condition_only.json
      <arm>/calls.json  <arm>/actions.json      # duplicates of the record's calls / action_trace
```

Tasks: `reach`, `reach-wall`, `pointmaze`, `wall`. Arms (fixed order):
`native, fixed_rank4, matched_random_fixed_rank4, coupling_only, matched_random_coupling, joint, visual_only, action_condition_only`.

`coupling_only` is the equal-budget joint edit (`joint_equal_standardized_energy`);
`matched_random_coupling` is its dose-matched random-direction control;
`matched_random_fixed_rank4` is the response-calibrated random-subspace control of `fixed_rank4`.

### `<arm>.json` (one record per scenario × arm)

| key | type | meaning |
|---|---|---|
| `scenario` | dict | the frozen input row (below) |
| `arm` | str | arm name |
| `result.native_success` | bool | **primary endpoint**: the task's official success |
| `result.native_state_distance` | float | final distance to goal (task units) |
| `result.native_reward` | float | cumulative simulator reward |
| `result.expert_success` | float | whether the expert rollout that defined the goal succeeded (MetaWorld) |
| `result.initial_sha256`, `result.goal_sha256` | str | must equal the scenario row's |
| `result.elementary_steps` | int | 100 (MetaWorld), 30 (navigation) |
| `result.planning_calls` | list | one per replanning call: `steps_left, returned_model_actions, iterations, seconds` |
| `result.published_candidate_count` | int | 300 |
| `calls` | list | one per `model.unroll` call, in execution order (see below) |
| `action_trace` | list | one per planning call: `steps_left, observation_sha256{visual,proprio}, actions_sha256` |
| `device_uuid`, `seconds`, `freeze_sha256` | | provenance |
| `scientific_efficacy_measurement` | bool | True for scenario records, False for engineering |

Per-iteration CEM losses are **not** recorded (kept in memory by the upstream evaluator).
Analyses of planner dynamics must work from `planning_calls`, `calls` and `action_trace`.

### `calls[i]` (one per unroll call)

Common: `horizon` (1..6), `candidates` (300 for populations, 1 for the mean rollout),
`backend_calls` (always 1), `seconds`, `energy` (dict, arm-dependent):

- refined arms (`fixed_rank4`, `matched_random_fixed_rank4`) at horizon 6:
  `coefficients` [candidates × 4] (dose-scaled coefficients actually applied),
  `requested_l2` [candidates], `realized_l2` [candidates] (after BF16/FP32 rounding),
  `active` [candidates] bool, `response_probe_rollouts` 0, `native_shadow_rollouts` 0, `backend_calls` 1, `horizon` 6.
  At horizon < 6 only the four scalar keys are present (no edit).
- `native` (refined adapter, no edit): `response_probe_rollouts, native_shadow_rollouts, backend_calls, horizon`.
- coupling arms (`coupling_only`, `matched_random_coupling`, `joint`, `visual_only`, `action_condition_only`):
  fresh static adapters store `horizon, candidates, requested_squared_l2_mean,
  realized_squared_l2_mean`. Older records may instead store `edited_candidates,
  requested_squared_l2_sum, realized_squared_l2_sum, response_probe_rollouts`.
  The corrected reader converts means to sums with the validated candidate count.
  Missing `edited_candidates` is unknown, not zero. These schemas must not be conflated.

### Scenario row (from `FREEZE/protocol.json` → `tasks[task].records`, role `scientific_candidates`)

`episode` (0..95), `logical_rank` (0..7), `local_seed`, `environment_seed`, `initial_sha256`, `goal_sha256`,
`initial_state` [float…], `goal_state` [float…], `rand_vec` [float…] (MetaWorld), `expert_frames`,
`expert_goal_success`, `tensor_path`, `tensor_sha256`. Role `excluded_engineering` rows (8 per task) are not science.

## 2. Frozen analysis output (`--analysis ANALYSIS`)

`ANALYSIS/report.json` written by `offline_study.experiments.fresh_confirmation analyze`:

```
method, freeze_sha256, scientific_evaluations (3072), single_released_checkpoint_per_task (true),
analysis: {endpoint, independent_unit, bootstrap_draws 20000, bootstrap_seed 2026091221, family_size 48,
           interval 'paired_scenario_percentile_bootstrap_bonferroni_95', minimum_useful_gain_pp 5, ...}
results[task]: {n: 96, success_percent{arm}, contrasts{<name>: {difference_pp, simultaneous_95_interval_pp}}}
```

**Twelve contrasts per task are pre-registered** (`fresh_confirmation.CONTRASTS`, also frozen
in `FREEZE/protocol.json` → `contrasts` as weight dicts; 12 × 4 tasks = `family_size` 48):

| name | weights | reading |
|---|---|---|
| `<arm>-native` (7) | `{arm: +1, native: -1}` | each arm vs unsteered |
| `refined-random` | `{fixed_rank4: +1, matched_random_fixed_rank4: -1}` | learned refined edit vs its dose-matched control |
| `coupling-random` | `{coupling_only: +1, matched_random_coupling: -1}` | learned coupling edit vs its dose-matched control |
| `joint-visual` | `{joint: +1, visual_only: -1}` | pathway: what action-conditioning adds to visual |
| `joint-action` | `{joint: +1, action_condition_only: -1}` | pathway: what the visual edit adds to action |
| `factorial-interaction` | `{joint: +1, visual_only: -1, action_condition_only: -1, native: +1}` | non-additivity of the two pathways |

The estimator is one RNG stream (`bootstrap_seed`) over tasks in `TASKS` order, one
`[draws × 96]` index matrix per task shared by that task's 12 contrasts, `values @ weights`,
percentile interval at `alpha = 0.05 / family_size`. `common.paired_bootstrap` draws a fresh
stream per call and therefore reproduces the frozen endpoints only up to Monte-Carlo noise
(~1–2 pp at alpha/48); `regime_report.py` additionally replays the analyzer stream exactly.

Both legs of `common.classify_regime` (arm vs native, learned vs its control) are therefore
registered at family 48. Every discordance p-value, every narrower-family or unadjusted
sensitivity, every per-scenario analysis below, and every contrast not in the table is
**exploratory / post hoc** and must be labelled as such in every output. When a script computes
its own paired bootstrap it must use `common.paired_bootstrap` (scenario-cluster resampling,
20,000 draws, fixed seed) and report the family it corrected over.

## 3. Prior artifacts the scripts may join against (read-only)

| artifact | path | use |
|---|---|---|
| Development core panel (Reach/Reach-Wall, 5 arms, exposed scenarios) | `reports/wm-approaches/core-analysis.json` | replication comparison (sign/shrinkage), wins/losses |
| Development Push-T / navigation / DROID panels | [six-task development results](../../reports/SIX_TASK_RESULTS_20260911.md) | replication comparison |
| Offline forecast contrasts (all sweeps, BF16/FP32) | `paper/data/all_task_ablation_contrasts.csv` | forecast → decision → outcome chain |
| Planner-decision diagnostic (development) | `reports/PLANNER_DECISION_DIAGNOSTIC_RESULTS.md` (tables) | decision-level reference |
| Fitted refined operator banks | `artifacts/offline_study/fixed-response-20260908-v1/fits/{reach,reach-wall}/operator_bank.pt` (`operators.fixed_rank4.basis` [4×256×400], `.map` [4×4]; `matched_random_fixed_rank4` likewise) | representation geometry |
| Basis spatial summary (already exported) | `paper/data/basis_spatial_summary.json` | geometry without torch |
| Fresh scenario banks | `artifacts/offline_study/fresh-simulator-banks-20260912-v1/` | initial/goal states |

Refined banks exist for Reach/Reach-Wall in that path; PointMaze/Wall refined fits live under the
freeze's `assets` (`fits/<task>/refined`) and may only be on Drive. Scripts must degrade gracefully
(skip with a logged reason) when a bank is absent.

## 4. Output conventions

Each script: `--results --freeze --analysis --out OUT [--fixture-regime …]`, writes
`OUT/<script>.json` (all numbers), `OUT/<script>.md` (human summary with an explicit
**"What this does and does not establish"** section), and `OUT/<script>_*.png` figures.
Deterministic: fixed seeds, no wall-clock in outputs except a provenance stamp.
Every claim sentence in the `.md` must be generated from the numbers (no hardcoded conclusions)
and must branch on the observed regime (positive / mixed / negative / inconclusive).

## 5. Fixtures

`make_fixture.py --regime {positive,mixed,negative,null} --out DIR` writes a complete synthetic
tree in the schema above plus `DIR/freeze/protocol.json` (with the 12 registered contrasts) and
`DIR/analysis/report.json` (all 12 contrasts per task, same estimator as the analyzer, via
`make_fixture.frozen_results`), so every script can be exercised now. Fixture outputs are
clearly stamped `synthetic: true`.
