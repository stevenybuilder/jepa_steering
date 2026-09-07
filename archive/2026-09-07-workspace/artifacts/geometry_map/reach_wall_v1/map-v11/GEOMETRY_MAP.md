# Geometry map v11 — evidence joined with statistical power

Built 2026-09-06T09:30:11.754848+00:00 from `artifacts/geometry_map/reach_wall_v1/map-v10` plus post-v10 results. `complete: false`. No model, simulator, or GPU call. Every number below is re-derived from the source JSON listed in `power_ledger.json`; missing measurements are `null`, never zero.

## What the map says

- Representation screen (447 rows) is a 150-episode discovery screen without per-row CIs: adequate to shortlist, not to confirm.
- Every intervention-decision test (58 v10 decision rows + 27 Push-T autoresearch trials) uses n=4 reused initial states; the exact sign-test floor is p=0.125, so no decision-level effect can reach significance regardless of result.
- Most Push-T arms show zero variance: the edit never changed the selected plan in a fixed 64-plan bank. That is a null at the decision level, but the design is only sensitive to argmin flips.
- The Reach output-correction pilots (n=8 starts, observed SD about 1.4 mm) were powered to detect the 5 mm gate (MDE about 1.6 mm) and found -0.6 to -0.7 mm vs native with CIs excluding +5 mm: these are genuine negatives, not underpowered nulls.
- The 5-start targeted subspace replication at predictor block 3 has MDE about 10-40 mm against a 5 mm gate: inconclusive, would need about 24-48 starts.
- Native CEM elite-mean vs best-candidate pooled over 8 groups: 0.1 pp, CI within +/-1 pp; the averaging-failure hypothesis is a powered null at the 2 pp gate.
- Native success rates carry Wilson CIs ~+/-10 pp at n~80; a 12-pair closed-loop pilot cannot detect less than ~45 pp change in success rate.

## Representation shortlist (discovery screen, 150 episodes, no CI)

| Task variable | Best site | Linear r² / AUROC | Δ vs time-only | kNN gain | Participation ratio |
|---|---|---:|---:|---:|---:|
| expert_detour_gate | predictor block 3 | 0.999 (auroc) | 0.021 | -0.002 | 7.6 |
| hand_y | predictor block 0 | 0.999 (r2) | 0.022 | null | 5.3 |
| hand_z | predictor block 1 | 0.996 (r2) | 0.988 | null | 4.2 |
| height_above_wall_top | predictor block 1 | 0.996 (r2) | 0.988 | null | 4.2 |
| hand_to_goal_segment_intersects_wall | predictor block 5 | 0.991 (auroc) | 0.089 | -0.012 | 3.4 |
| coordinate_probe/world | predictor block 3 | 0.989 (r2) | null | null | null |
| wall_signed_distance | predictor block 2 | 0.986 (r2) | 0.844 | -0.134 | 4.9 |
| hand_x | predictor block 1 | 0.985 (r2) | 0.927 | null | 4.2 |
| action_dy | predictor block 1 | 0.984 (r2) | 0.355 | null | 4.2 |
| goal_vector_y | predictor block 1 | 0.973 (r2) | 0.016 | null | 4.2 |
| action_dx | predictor block 0 | 0.972 (r2) | 0.968 | null | 5.3 |
| reward_sum | predictor block 3 | 0.965 (r2) | 0.051 | -0.014 | 7.6 |
| action_dz | predictor block 1 | 0.964 (r2) | 0.936 | null | 4.2 |
| cumulative_progress | predictor block 3 | 0.959 (r2) | 0.037 | -0.021 | 7.6 |
| coordinate_probe/observed_episode_raw_step | predictor block 3 | 0.943 (r2) | null | null | null |
| goal_distance | predictor block 5 | 0.942 (r2) | 0.033 | -0.036 | 3.4 |
| realized_dy | predictor block 3 | 0.936 (r2) | 0.142 | -0.061 | 7.6 |
| coordinate_probe/goal_relative | predictor block 3 | 0.844 (r2) | null | null | null |
| action_direction | predictor block 2 | 0.835 (r2) | 0.642 | null | null |
| step_progress | predictor block 3 | 0.820 (r2) | 0.184 | null | null |
| realized_hand_delta_magnitude | predictor block 3 | 0.812 (r2) | 0.199 | -0.076 | 7.6 |
| coordinate_probe/gripper_relative | predictor block 3 | 0.802 (r2) | null | null | null |
| realized_motion_world | predictor block 3 | 0.795 (r2) | 0.421 | null | null |
| action_net_norm | predictor block 2 | 0.762 (r2) | 0.275 | null | null |

Scores are leave-one-collection-seed-out on the 150-trajectory discovery split. Time-only baselines are high for progress-like variables, so read Δ vs time, not raw r². None of these rows has untouched-confirmation evidence.

## Intervention ledger with power

| Ledger id | n | mean | 95% CI | sign-test floor p | MDE₈₀ | gate | n needed | verdict |
|---|---:|---:|---|---:|---:|---|---:|---|
| causal.replication.reach-wall/predictor/block-3/spatial-mean/realized_xz_direction.endpoint_effect_l2_m | 5 | 0.00 m | [-0.01, 0.02] | 0.062 | 0.02 | 5 mm | 48 | underpowered |
| causal.replication.reach-wall/predictor/block-3/spatial-mean/realized_xz_direction.goal_improvement_vs_unsteered_m | 5 | -0.00 m | [-0.01, 0.01] | 0.062 | 0.01 | 5 mm | 24 | underpowered |
| causal.replication.reach-wall/predictor/block-3/spatial-mean/realized_xz_direction.off_target_xy_effect_l2_m | 5 | 0.00 m | [-0.01, 0.01] | 0.062 | 0.02 | 5 mm | 31 | underpowered |
| causal.replication.reach-wall/predictor/block-3/spatial-mean/realized_xz_direction.signed_target_z_effect_m | 5 | 0.01 m | [-0.02, 0.04] | 0.062 | 0.04 | 5 mm | 221 | underpowered |
| decision.normalization_aware_subspace_v1.calibrated_semantic_plus.predicted | 4 | 0.46 pp | [-1.00, 1.92] | 0.125 | 1.91 | 2 pp | 4 | underpowered |
| decision.task_metric_complement_ablation_v1.pointwise/joint_xy_squared_proxy/learned/span_only.predicted | 4 | 1.16 pp | [-1.06, 3.38] | 0.125 | 2.90 | 2 pp | 6 | underpowered |
| decision.task_metric_complement_ablation_v1.pointwise/one_minus_requested_goal_coverage/learned/span_only.predicted | 4 | 0.72 pp | [-1.49, 2.93] | 0.125 | 2.89 | 2 pp | 6 | underpowered |
| decision.task_metric_complement_ablation_v1.pairwise/joint_xy_squared_proxy/learned/span_only.predicted | 4 | 0.72 pp | [-1.49, 2.93] | 0.125 | 2.89 | 2 pp | 6 | underpowered |
| decision.task_metric_complement_ablation_v1.pairwise/one_minus_requested_goal_coverage/learned/span_only.predicted | 4 | 0.72 pp | [-1.49, 2.93] | 0.125 | 2.89 | 2 pp | 6 | underpowered |
| decision.task_metric_complement_ablation_v1.pointwise/joint_xy_squared_proxy/learned/span_only.actual_oracle_encoding | 4 | 3.46 pp | [-7.55, 14.48] | 0.125 | 14.40 | 2 pp | 98 | underpowered |
| decision.task_metric_complement_ablation_v1.pointwise/one_minus_requested_goal_coverage/learned/span_only.actual_oracle_encoding | 4 | 3.59 pp | [-7.30, 14.48] | 0.125 | 14.24 | 2 pp | 96 | underpowered |
| decision.task_metric_complement_ablation_v1.pairwise/joint_xy_squared_proxy/learned/span_only.actual_oracle_encoding | 4 | 3.46 pp | [-7.55, 14.48] | 0.125 | 14.40 | 2 pp | 98 | underpowered |
| decision.task_metric_complement_ablation_v1.pairwise/one_minus_requested_goal_coverage/learned/span_only.actual_oracle_encoding | 4 | 3.62 pp | [-7.24, 14.48] | 0.125 | 14.20 | 2 pp | 95 | underpowered |
| autoresearch.push.forecast_correction.linear.semantic.0.1 | 4 | 1.16 pp | [-1.06, 3.38] | 0.125 | 2.90 | 2 pp, 3/4 states, beats each sham | 6 | underpowered |
| autoresearch.push.forecast_error_penalty.linear.risk.0.1 | 4 | 0.53 pp | [-3.23, 4.30] | 0.125 | 4.92 | 2 pp, 3/4 states, beats each sham | 14 | underpowered |
| autoresearch.push.forecast_error_penalty.linear.risk.0.3 | 4 | 0.53 pp | [-3.23, 4.30] | 0.125 | 4.92 | 2 pp, 3/4 states, beats each sham | 14 | underpowered |
| autoresearch.push.forecast_error_penalty.linear.risk.1.0 | 4 | -2.08 pp | [-6.94, 2.79] | 0.125 | 6.36 | 2 pp, 3/4 states, beats each sham | 21 | underpowered |
| autoresearch.push.forecast_correction.rbf.semantic.0.1 | 4 | 1.16 pp | [-1.06, 3.38] | 0.125 | 2.90 | 2 pp, 3/4 states, beats each sham | 6 | underpowered |
| autoresearch.push.forecast_error_penalty.rbf.risk.0.3 | 4 | 0.15 pp | [-0.32, 0.61] | 0.125 | 0.61 | 2 pp, 3/4 states, beats each sham | 3 | underpowered |
| autoresearch.push.forecast_error_penalty.rbf.risk.1.0 | 4 | -0.35 pp | [-6.49, 5.79] | 0.125 | 8.03 | 2 pp, 3/4 states, beats each sham | 32 | underpowered |
| autoresearch.push.forecast_correction.knn.semantic.0.02 | 4 | 0.46 pp | [-1.00, 1.92] | 0.125 | 1.91 | 2 pp, 3/4 states, beats each sham | 4 | underpowered |
| autoresearch.push.forecast_correction.knn.semantic.0.1 | 4 | 0.46 pp | [-1.00, 1.92] | 0.125 | 1.91 | 2 pp, 3/4 states, beats each sham | 4 | underpowered |
| autoresearch.push.forecast_error_penalty.knn.risk.1.0 | 4 | 0.91 pp | [-1.98, 3.79] | 0.125 | 3.77 | 2 pp, 3/4 states, beats each sham | 9 | underpowered |
| autoresearch.push.within_state_contrast.rbf.semantic.0.02 | 4 | 0.70 pp | [-1.53, 2.93] | 0.125 | 2.91 | 2 pp, 3/4 states, beats each sham | 6 | underpowered |
| autoresearch.push.within_state_contrast.rbf.semantic.0.1 | 4 | 0.70 pp | [-1.53, 2.93] | 0.125 | 2.91 | 2 pp, 3/4 states, beats each sham | 6 | underpowered |
| autoresearch.push.within_state_contrast.knn.semantic.0.02 | 4 | 0.70 pp | [-1.53, 2.93] | 0.125 | 2.91 | 2 pp, 3/4 states, beats each sham | 6 | underpowered |
| autoresearch.push.within_state_contrast.knn.semantic.0.1 | 4 | 0.70 pp | [-1.53, 2.93] | 0.125 | 2.91 | 2 pp, 3/4 states, beats each sham | 6 | underpowered |
| autoresearch.reach.pooled.vs_native_planner | 8 | -0.63 mm | [-1.82, 0.56] | 0.008 | 1.64 | 5 mm | 3 | adequate_for_screen |
| autoresearch.reach.full_spatial.vs_native_planner | 8 | -0.73 mm | [-1.90, 0.43] | 0.008 | 1.61 | 5 mm | 3 | adequate_for_screen |
| autoresearch.reach.full_spatial.vs_sham | 8 | 0.24 mm | [-0.78, 1.27] | 0.008 | 1.41 | 5 mm | 3 | adequate_for_screen |
| autoresearch.reach.h6_30step.vs_native_planner | 8 | 2.77 mm | [-3.42, 8.95] | 0.008 | 8.53 | 5 mm | 20 | underpowered |
| autoresearch.reach.h6_30step.vs_sham | 8 | 0.45 mm | [-1.32, 2.22] | 0.008 | 2.44 | 5 mm | 4 | adequate_for_screen |
| cem.replication.stage30_minus15_coverage | 4 | -0.68 pp | [-6.64, 5.27] | 0.125 | 7.78 | 2 pp | 30 | underpowered |
| cem.replication.mean_minus_best_stage30 | 4 | 0.51 pp | [-0.54, 1.56] | 0.125 | 1.37 | 2 pp | 4 | underpowered |
| cem.audit.mean_minus_best_stage30 | 4 | -0.29 pp | [-2.22, 1.64] | 0.125 | 2.52 | 2 pp | 6 | underpowered |
| cem.pooled8.mean_minus_best_stage30 | 8 | 0.11 pp | [-0.73, 0.94] | 0.008 | 1.15 | 2 pp | 5 | adequate_for_screen |

27 additional arms have zero variance across all 4 states (the edit never changed the selected plan) and are listed in `power_ledger.csv` with verdict `zero_variance_no_decision_change`. That is a null at the decision level under a fixed 64-plan bank, not an underpowered positive.

Interpretation of MDE₈₀: the smallest true paired effect that this many independent starts would detect with 80% power at α=0.05, using the observed SD. When MDE₈₀ exceeds the gate the test could not have passed the gate for a real effect of gate size; that cell is underpowered rather than negative.

## Native unsteered baselines (label-first anchors)

| Panel | n | terminal success | 95% CI | ever success | 95% CI |
|---|---:|---:|---|---:|---|
| pusht_official/collect_pusht_bank | 21 | 61.9% | [40.9, 79.2] | 61.9% | [40.9, 79.2] |
| pusht_scripted/collect_pusht_bank | 54 | 35.2% | [23.8, 48.5] | 40.7% | [28.7, 54.0] |
| reach_wall/collect_on_policy_bank | 66 | 24.2% | [15.5, 35.8] | 45.5% | [34.0, 57.4] |
| reach_wall/collect_reach_development_expansion_full99 | 8 | 12.5% | [2.2, 47.1] | 37.5% | [13.7, 69.4] |
| reach_wall all full-99 (66+8+5) | 79 | 25.3% | [17.0, 35.9] | 45.6% | [35.0, 56.5] |

Episodes 74–78 (new, verified tensor SHA): ep74 final=✓, ep75 final=✓, ep76 final=✗, ep77 final=✗, ep78 final=✓.

## What a closed-loop pilot can detect

| n per arm | terminal-success MDE (from 25.3%) | ever-success MDE (from 45.6%) |
|---:|---:|---:|
| 12 | +54.4 pp | +49.7 pp |
| 24 | +39.2 pp | +37.8 pp |
| 48 | +27.6 pp | +27.7 pp |
| 96 | +19.2 pp | +20.0 pp |
| 200 | +13.0 pp | +13.9 pp |
| 400 | +9.1 pp | +9.9 pp |

Continuous endpoint (15-step hand-goal progress, mm, paired edit minus native): observed SD 1.39 mm → n needed for 5 mm gate = 3, for 2 mm = 6, for 1 mm = 18.

Two-proportion normal approximation, 80% power, alpha 0.05 two-sided, unpaired per-arm n; paired same-start designs can do better if outcomes are concordant. Continuous n from observed development SD.

## Post-v10 evidence joined

- Native CEM replication on 4 new TRAIN groups: 30-vs-15-iteration coverage change mean -0.68 pp (CI [-6.64, 5.27]); elite-mean minus best 0.51 pp (CI [-0.54, 1.56]). Model cost fell in all 4 groups while coverage did not improve. Verdict: underpowered.
- 149-row native baseline table and 5 additional Reach episodes joined above with Wilson CIs.

## Field coverage carried from v10

| Field | measured / total |
|---|---:|
| linear_score | 443 / 447 |
| nonlinear_gain | 164 / 447 |
| best_coordinate_frame | 17 / 447 |
| geometry | 405 / 447 |
| cross_trajectory_stability | 35 / 447 |
| regime_dependence | 32 / 447 |
| causal_patch_effect | 2 / 447 |
| specificity | 1 / 447 |
| manifold_distance | 1 / 447 |
| recommended_operator | 0 / 447 |

Blockers unchanged: Untouched confirmation and full required causal controls are not yet complete; No validated operator card exists.

Figures: `power_forest.svg` (new), plus the v10 visuals copied unchanged (`layer_variable_heatmap.svg`, `spatial_token_readability.svg`, `shortlist_eigenspectra.svg`, causal timelines).
