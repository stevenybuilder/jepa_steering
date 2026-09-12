# All-task ablation results and interpretation

Checked September 8, 2026. This renders completed, preserved development reports. No experiment, fitting, new statistical test, or incomplete behavioral-outcome inspection was performed.

## How to read the tables

Each forecast cell is **percentage error reduction versus native [simultaneous 95% interval]**. Positive means lower error; negative means higher error. P = H6 proprioceptive embedding MSE; V = H6 visual embedding MSE. Neither is task success, and P is not decoded physical-state error.

Intervals are paired by source trajectory/lineage, simultaneous across the registered contrasts and both endpoints **within task/category/precision**, not across this entire document. They are difference intervals scaled by the observed native mean, not confidence intervals for a population error ratio. BF16 is primary; FP32 is sensitivity. A positive interval is statistical evidence for that comparison, not necessarily a practically useful effect. The frozen minimum is 1% of fit-only native error; qualifying also requires the registered controls and implementation/energy checks. It is not a requirement that each observed percentage simply exceed 1%.

Reach has 33 development lineages, Reach-Wall 27, Push-T 21, Wall 192 and PointMaze 200. The same lineages recur across arms and precisions. Reach and Reach-Wall share the released MetaWorld predictor but have separate fitted corrections. Many comparisons reuse the same intervention: rank 1 at B3, layer B3, and all-patch rank 1 have matching point estimates but different simultaneous contrast families. These are not three independent discoveries.

The interpretations below distinguish observed statistical patterns from **possible, untested mechanisms**. An inconclusive interval does not prove zero effect. A negative interval establishes harm to this endpoint, not necessarily to task success.

## Why these comparisons were selected

JEPA-WM compares inherited architecture/training/planning alternatives one factor at a time, then evaluates behavior. Its choices test concrete concerns such as rollout error, action conditioning and missing state information; the paper does not establish when each rationale was formulated relative to its experiments. See [Table 1 and Section 4](https://arxiv.org/html/2512.24497v4#S4).

Our [frozen plan](../../docs/EXPERIMENT_PLAN.md) asks whether frozen-model interventions have selective forecast effects that survive later candidate ranking and control. Rank tests correction capacity; pathway controls test vision/action alignment and dose; layer and spatial sweeps separate depth from support; linear/cubic and curvature controls test local response geometry; the revised operator tests moving response estimation offline. Registration and controls make these questions interpretable, but do not prove that the sites, dose, metric, or compute allocation were optimal.

The B3/H3 reference is an experimental choice, not an independently discovered physics zone. The original online response diagnostics were expensive inside planning. The present evidence supports a task-specific adaptation study, not transfer of one fitted operator or universal steering. The individual visual/action coupling arms lack their own scope-matched random-direction arms in the original registry; their native effects do not alone establish learned-direction specificity.

## Coverage

| Family | Reach | Reach-Wall | Push-T | Wall | PointMaze | DROID |
|---|---|---|---|---|---|---|
| Coupling | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | BF16 numbers below; FP32 reported complete, aggregate unavailable here | Reported complete BF16/FP32; aggregates unavailable here | Complete recorded-plan score panel |
| Geometry | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | No completed panel verified |
| Rank / layer / spatial | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | Fits complete; comparison panels not verified complete | Fits complete; comparison panels not verified complete | No completed panels verified |
| Original combined/drop-one | Complete BF16/FP32 | Not run | Not run | Not run | Not run | Not run |
| Revised rank 4 | Complete BF16/FP32 | Complete BF16/FP32 | Not run | Not run | Not run | Not run |
| HMM / revised combined behavior | No completed efficacy panel | No completed HMM efficacy panel | Not run | Not run | Not run | Not run |

The closed-loop comparison is a separate endpoint. HMM, additional combined behavior, navigation expansion and training histories are paused under the existing core-panel priority. Missing or paused comparisons are not null findings. No completed task-success result is inferred from this audit.

## BF16 primary results: every measured arm versus native

### Reach — bfloat16

#### Vision/action coupling

Source S01; n=33. Native absolute H6 MSE: P=0.0008585703183, V=0.1980835786.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| visual_only | +2.262 [+1.256, +3.268] | +0.593 [+0.085, +1.101] | positive / positive | Consistent with correction through visual and conditioning pathways. Joint doses and drop-one energy differences preclude an automatic synergy claim. |
| action_condition_only | +0.679 [+0.481, +0.877] | +0.022 [-0.016, +0.061] | positive / inconclusive | Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested. |
| joint | +2.877 [+1.872, +3.883] | +0.610 [+0.105, +1.115] | positive / positive | Consistent with correction through visual and conditioning pathways. Joint doses and drop-one energy differences preclude an automatic synergy claim. |
| joint_equal_standardized_energy | +2.035 [+1.298, +2.773] | +0.453 [+0.085, +0.821] | positive / positive | Consistent with correction through visual and conditioning pathways. Joint doses and drop-one energy differences preclude an automatic synergy claim. |
| permuted_visual | +0.203 [+0.053, +0.353] | -0.016 [-0.069, +0.036] | positive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| permuted_joint | +0.862 [+0.648, +1.075] | +0.022 [-0.026, +0.071] | positive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| matched_random | -2.291 [-2.824, -1.758] | -0.286 [-0.373, -0.199] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_standardized_energy | -1.655 [-2.030, -1.281] | -0.195 [-0.262, -0.129] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Action-response geometry

Source S03; n=33. Native absolute H6 MSE: P=0.0008585703183, V=0.1980835786.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.017 [-0.040, +0.074] | +0.010 [-0.007, +0.027] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | +0.027 [-0.042, +0.095] | -0.001 [-0.020, +0.019] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.007 [-0.072, +0.086] | +0.009 [-0.012, +0.029] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | -0.013 [-0.086, +0.060] | -0.004 [-0.030, +0.022] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.382 [+0.174, +0.590] | +0.051 [+0.000, +0.102] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Operator rank

Source S05; n=33. Native absolute H6 MSE: P=0.0008585703183, V=0.1980835786.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| rank1 | +1.899 [+1.382, +2.416] | +0.301 [+0.119, +0.484] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank4 | +2.887 [+2.276, +3.499] | +0.520 [+0.380, +0.661] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank8 | +2.772 [+2.135, +3.409] | +0.501 [+0.374, +0.628] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_rank1 | +0.205 [+0.092, +0.317] | +0.065 [+0.016, +0.114] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | +0.840 [+0.596, +1.084] | +0.126 [+0.061, +0.191] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank8 | +1.141 [+0.856, +1.427] | +0.172 [+0.097, +0.248] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Layer distribution (total rank 1)

Source S07; n=33. Native absolute H6 MSE: P=0.0008585703183, V=0.1980835786.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| single_block0 | +3.026 [+2.124, +3.929] | +0.496 [+0.199, +0.793] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block1 | +2.872 [+2.053, +3.690] | +0.475 [+0.185, +0.765] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block2 | +2.263 [+1.595, +2.932] | +0.380 [+0.148, +0.612] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block3 | +1.899 [+1.344, +2.454] | +0.301 [+0.105, +0.497] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block4 | +1.169 [+0.841, +1.497] | +0.187 [+0.077, +0.297] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block5 | +0.475 [+0.333, +0.617] | +0.083 [+0.035, +0.132] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| intermediate_blocks2_3 | +2.826 [+1.969, +3.683] | +0.448 [+0.143, +0.753] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| all_six_blocks | +2.353 [+1.637, +3.070] | +0.368 [+0.112, +0.623] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_single_block0 | +0.379 [+0.229, +0.530] | +0.084 [+0.046, +0.122] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block1 | +0.309 [+0.139, +0.479] | +0.080 [+0.009, +0.151] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block2 | +0.258 [+0.139, +0.376] | +0.052 [+0.020, +0.085] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block3 | +0.205 [+0.084, +0.325] | +0.065 [+0.013, +0.118] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block4 | +0.169 [+0.083, +0.255] | +0.039 [+0.012, +0.066] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block5 | +0.126 [+0.058, +0.194] | +0.040 [+0.015, +0.064] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_intermediate_blocks2_3 | +0.439 [+0.230, +0.648] | +0.090 [+0.020, +0.159] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_six_blocks | +0.318 [+0.188, +0.447] | +0.075 [+0.016, +0.134] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Spatial support (rank 1)

Source S09; n=33. Native absolute H6 MSE: P=0.0008585703183, V=0.1980835786.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| one_patch | +0.198 [+0.119, +0.277] | +0.056 [+0.027, +0.086] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| contiguous_group | +1.229 [+0.898, +1.561] | +0.307 [+0.216, +0.397] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| equal_size_scattered_group | +0.366 [+0.234, +0.498] | +0.070 [+0.023, +0.118] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| all_patches | +1.899 [+1.340, +2.458] | +0.301 [+0.104, +0.499] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_one_patch | +0.315 [+0.186, +0.443] | +0.057 [+0.023, +0.092] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_contiguous_group | +0.473 [+0.262, +0.685] | +0.111 [+0.022, +0.201] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_size_scattered_group | +0.183 [+0.100, +0.266] | +0.049 [+0.016, +0.082] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_patches | +0.205 [+0.083, +0.326] | +0.065 [+0.012, +0.118] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| random_position_one_patch | +0.234 [+0.137, +0.331] | +0.054 [+0.022, +0.086] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_contiguous_group | +0.312 [+0.195, +0.430] | +0.057 [+0.033, +0.081] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_equal_size_scattered_group | +0.482 [+0.310, +0.655] | +0.090 [+0.036, +0.144] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |

#### Original combined/drop-one

Source S11; n=33. Native absolute H6 MSE: P=0.0008584430071, V=0.1980828497.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| combined | +4.824 [+3.668, +5.980] | +0.935 [+0.493, +1.377] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| coupling_only | +2.020 [+1.284, +2.756] | +0.452 [+0.086, +0.819] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| rank4_only | +2.874 [+2.266, +3.481] | +0.519 [+0.382, +0.657] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| combined_rank1 | +3.856 [+2.730, +4.982] | +0.703 [+0.188, +1.218] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_coupling | -1.671 [-2.043, -1.298] | -0.196 [-0.262, -0.130] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | +0.824 [+0.582, +1.066] | +0.125 [+0.064, +0.187] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_combined | -0.825 [-1.078, -0.571] | -0.084 [-0.158, -0.010] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_combined_rank1 | -1.559 [-1.919, -1.198] | -0.175 [-0.239, -0.112] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Revised fixed response

Source S13; n=33. Native absolute H6 MSE: P=0.0008574237434, V=0.1980681793.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| fixed_rank4 | +2.360 [+1.966, +2.755] | +0.427 [+0.326, +0.529] | positive / positive | An offline mean response retains forecast benefits. This supports amortizing response estimation, not old/new equivalence or demonstrated task-success gains. |
| matched_random_fixed_rank4 | +1.157 [+0.908, +1.406] | +0.117 [+0.054, +0.179] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

### Reach-Wall — bfloat16

#### Vision/action coupling

Source S15; n=27. Native absolute H6 MSE: P=0.0008200658457, V=0.2090970272.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| visual_only | +1.738 [+0.163, +3.313] | -0.529 [-1.360, +0.302] | positive / inconclusive | Proprioceptive gains coexist with inconclusive visual changes. A modality-specific correction is plausible; success and wall-clearance effects are unmeasured here. |
| action_condition_only | +0.667 [+0.406, +0.928] | +0.028 [-0.017, +0.072] | positive / inconclusive | Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested. |
| joint | +2.327 [+0.825, +3.828] | -0.496 [-1.298, +0.305] | positive / inconclusive | Proprioceptive gains coexist with inconclusive visual changes. A modality-specific correction is plausible; success and wall-clearance effects are unmeasured here. |
| joint_equal_standardized_energy | +1.788 [+0.667, +2.910] | -0.292 [-0.872, +0.287] | positive / inconclusive | Proprioceptive gains coexist with inconclusive visual changes. A modality-specific correction is plausible; success and wall-clearance effects are unmeasured here. |
| permuted_visual | +0.284 [+0.064, +0.503] | -0.129 [-0.225, -0.034] | positive / negative | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| permuted_joint | +0.966 [+0.654, +1.278] | -0.079 [-0.178, +0.019] | positive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| matched_random | -2.505 [-3.153, -1.857] | -0.232 [-0.357, -0.106] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_standardized_energy | -1.560 [-2.015, -1.104] | -0.131 [-0.229, -0.034] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Action-response geometry

Source S17; n=27. Native absolute H6 MSE: P=0.0008200658457, V=0.2090970272.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.012 [-0.082, +0.106] | -0.015 [-0.038, +0.008] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | +0.038 [-0.049, +0.126] | -0.005 [-0.025, +0.015] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.043 [-0.049, +0.135] | -0.004 [-0.024, +0.016] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | -0.008 [-0.104, +0.088] | -0.012 [-0.039, +0.014] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.268 [+0.123, +0.413] | +0.027 [-0.007, +0.061] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Operator rank

Source S19; n=27. Native absolute H6 MSE: P=0.0008200658457, V=0.2090970272.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| rank1 | +0.943 [+0.240, +1.646] | +0.189 [+0.033, +0.346] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank4 | +2.137 [+1.645, +2.629] | +0.412 [+0.297, +0.528] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank8 | +1.993 [+1.534, +2.453] | +0.399 [+0.292, +0.505] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_rank1 | +0.324 [+0.227, +0.421] | +0.071 [+0.042, +0.100] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | +0.621 [+0.395, +0.847] | +0.115 [+0.055, +0.175] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank8 | +0.782 [+0.506, +1.059] | +0.139 [+0.070, +0.208] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Layer distribution (total rank 1)

Source S21; n=27. Native absolute H6 MSE: P=0.0008200658457, V=0.2090970272.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| single_block0 | +1.526 [+0.391, +2.662] | +0.324 [+0.050, +0.598] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block1 | +1.313 [+0.326, +2.301] | +0.312 [+0.060, +0.564] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block2 | +1.046 [+0.166, +1.926] | +0.249 [+0.035, +0.463] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block3 | +0.943 [+0.169, +1.717] | +0.189 [+0.017, +0.362] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block4 | +0.582 [+0.153, +1.010] | +0.125 [+0.035, +0.214] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block5 | +0.298 [+0.108, +0.488] | +0.064 [+0.028, +0.099] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| intermediate_blocks2_3 | +1.335 [+0.210, +2.461] | +0.279 [+0.005, +0.553] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| all_six_blocks | +1.198 [+0.239, +2.158] | +0.264 [+0.037, +0.491] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_single_block0 | +0.270 [+0.097, +0.443] | +0.064 [+0.027, +0.101] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block1 | +0.230 [+0.112, +0.347] | +0.066 [+0.036, +0.095] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block2 | +0.301 [+0.189, +0.414] | +0.065 [+0.033, +0.097] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block3 | +0.324 [+0.218, +0.431] | +0.071 [+0.039, +0.103] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block4 | +0.117 [+0.024, +0.209] | +0.035 [+0.010, +0.060] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block5 | +0.113 [+0.034, +0.193] | +0.034 [+0.013, +0.055] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_intermediate_blocks2_3 | +0.197 [+0.091, +0.302] | +0.073 [+0.037, +0.108] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_six_blocks | +0.230 [+0.129, +0.332] | +0.051 [+0.018, +0.084] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Spatial support (rank 1)

Source S23; n=27. Native absolute H6 MSE: P=0.0008200658457, V=0.2090970272.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| one_patch | +0.153 [+0.071, +0.235] | +0.053 [+0.025, +0.081] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| contiguous_group | +0.744 [+0.482, +1.006] | +0.145 [+0.086, +0.203] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| equal_size_scattered_group | +0.222 [+0.127, +0.318] | +0.059 [+0.033, +0.085] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| all_patches | +0.943 [+0.180, +1.705] | +0.189 [+0.020, +0.359] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_one_patch | +0.152 [+0.056, +0.248] | +0.053 [+0.025, +0.080] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_contiguous_group | +0.408 [-0.163, +0.979] | +0.127 [+0.003, +0.250] | inconclusive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_size_scattered_group | +0.184 [+0.076, +0.292] | +0.042 [+0.019, +0.064] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_patches | +0.324 [+0.219, +0.430] | +0.071 [+0.040, +0.103] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| random_position_one_patch | +0.142 [+0.044, +0.240] | +0.038 [+0.016, +0.061] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_contiguous_group | +0.264 [+0.113, +0.416] | +0.055 [+0.029, +0.080] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_equal_size_scattered_group | +0.264 [+0.137, +0.392] | +0.062 [+0.028, +0.096] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |

#### Revised fixed response

Source S25; n=27. Native absolute H6 MSE: P=0.0008208345951, V=0.2091706633.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| fixed_rank4 | +2.192 [+1.798, +2.587] | +0.399 [+0.298, +0.500] | positive / positive | An offline mean response retains forecast benefits. This supports amortizing response estimation, not old/new equivalence or demonstrated task-success gains. |
| matched_random_fixed_rank4 | +0.717 [+0.487, +0.948] | +0.131 [+0.064, +0.197] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

### Push-T — bfloat16

#### Vision/action coupling

Source S27; n=21. Native absolute H6 MSE: P=0.0002192615816, V=0.1116944709.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| visual_only | -1.022 [-1.718, -0.326] | -0.503 [-2.519, +1.513] | negative / inconclusive | Visual-only and unscaled joint harm the primary endpoint. Possible mismatch with contact-dependent dynamics; no contact-mediated causal test establishes this explanation. |
| action_condition_only | +0.336 [+0.052, +0.619] | +0.008 [-0.230, +0.246] | positive / inconclusive | Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested. |
| joint | -0.638 [-1.214, -0.062] | -0.511 [-2.375, +1.353] | negative / inconclusive | Visual-only and unscaled joint harm the primary endpoint. Possible mismatch with contact-dependent dynamics; no contact-mediated causal test establishes this explanation. |
| joint_equal_standardized_energy | -0.390 [-0.831, +0.052] | -0.305 [-1.739, +1.129] | inconclusive / inconclusive | Visual-only and unscaled joint harm the primary endpoint. Possible mismatch with contact-dependent dynamics; no contact-mediated causal test establishes this explanation. |
| permuted_visual | -0.416 [-0.585, -0.246] | -0.188 [-0.715, +0.339] | negative / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| permuted_joint | +0.036 [-0.211, +0.284] | -0.107 [-0.538, +0.323] | inconclusive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| matched_random | +0.300 [+0.048, +0.551] | +0.060 [-0.179, +0.300] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_standardized_energy | +0.271 [+0.068, +0.474] | +0.050 [-0.146, +0.246] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Action-response geometry

Source S29; n=21. Native absolute H6 MSE: P=0.0002192615816, V=0.1116944709.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.058 [-0.119, +0.235] | +0.217 [-0.071, +0.505] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | +0.037 [-0.120, +0.194] | +0.011 [-0.199, +0.221] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.035 [-0.146, +0.216] | +0.036 [-0.229, +0.302] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | +0.009 [-0.189, +0.206] | +0.116 [-0.162, +0.394] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.046 [-0.058, +0.149] | -0.104 [-0.362, +0.155] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Operator rank

Source S31; n=21. Native absolute H6 MSE: P=0.0002192615816, V=0.1116944709.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| rank1 | -0.039 [-0.482, +0.405] | -0.019 [-0.207, +0.169] | inconclusive / inconclusive | All tested ranks are inconclusive; the selected subspace/site may miss useful error structure, or effects may be small/variable. Neither explanation is established. |
| rank4 | +0.198 [-0.059, +0.455] | +0.040 [-0.156, +0.235] | inconclusive / inconclusive | All tested ranks are inconclusive; the selected subspace/site may miss useful error structure, or effects may be small/variable. Neither explanation is established. |
| rank8 | +0.107 [-0.156, +0.369] | +0.016 [-0.259, +0.292] | inconclusive / inconclusive | All tested ranks are inconclusive; the selected subspace/site may miss useful error structure, or effects may be small/variable. Neither explanation is established. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_rank1 | -0.016 [-0.106, +0.074] | -0.063 [-0.296, +0.169] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | -0.034 [-0.111, +0.043] | -0.029 [-0.216, +0.157] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank8 | -0.058 [-0.171, +0.054] | -0.173 [-0.557, +0.211] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Layer distribution (total rank 1)

Source S33; n=21. Native absolute H6 MSE: P=0.0002192615816, V=0.1116944709.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| single_block0 | -0.025 [-0.961, +0.912] | +0.198 [-0.319, +0.715] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block1 | -0.034 [-0.661, +0.594] | +0.051 [-0.220, +0.322] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block2 | -0.134 [-0.797, +0.529] | -0.021 [-0.365, +0.323] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block3 | -0.039 [-0.529, +0.452] | -0.019 [-0.227, +0.189] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block4 | +0.052 [-0.272, +0.376] | +0.030 [-0.151, +0.210] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block5 | -0.024 [-0.189, +0.140] | -0.065 [-0.339, +0.209] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| intermediate_blocks2_3 | -0.308 [-1.140, +0.524] | -0.010 [-0.252, +0.232] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| all_six_blocks | -0.043 [-0.799, +0.712] | +0.107 [-0.164, +0.377] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_single_block0 | -0.024 [-0.508, +0.461] | +0.104 [-0.328, +0.536] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block1 | -0.047 [-0.166, +0.073] | +0.051 [-0.267, +0.369] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block2 | -0.032 [-0.163, +0.099] | -0.059 [-0.293, +0.175] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block3 | -0.016 [-0.116, +0.083] | -0.063 [-0.320, +0.194] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block4 | -0.048 [-0.154, +0.058] | -0.002 [-0.198, +0.194] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block5 | -0.039 [-0.156, +0.077] | +0.006 [-0.116, +0.129] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_intermediate_blocks2_3 | -0.012 [-0.115, +0.091] | -0.053 [-0.272, +0.166] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_six_blocks | -0.003 [-0.177, +0.172] | +0.054 [-0.449, +0.557] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Spatial support (rank 1)

Source S35; n=21. Native absolute H6 MSE: P=0.0002192615816, V=0.1116944709.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| one_patch | +0.026 [-0.098, +0.150] | -0.050 [-0.262, +0.162] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| contiguous_group | -0.025 [-0.188, +0.138] | -0.004 [-0.172, +0.164] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| equal_size_scattered_group | -0.065 [-0.148, +0.018] | +0.005 [-0.238, +0.248] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| all_patches | -0.039 [-0.505, +0.428] | -0.019 [-0.217, +0.179] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_one_patch | +0.015 [-0.116, +0.147] | -0.057 [-0.319, +0.204] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_contiguous_group | -0.084 [-0.220, +0.053] | -0.006 [-0.219, +0.206] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_size_scattered_group | -0.016 [-0.118, +0.086] | -0.010 [-0.257, +0.236] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_patches | -0.016 [-0.111, +0.078] | -0.063 [-0.308, +0.181] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| random_position_one_patch | -0.056 [-0.168, +0.056] | -0.006 [-0.149, +0.138] | inconclusive / inconclusive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_contiguous_group | -0.034 [-0.134, +0.067] | -0.005 [-0.237, +0.228] | inconclusive / inconclusive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_equal_size_scattered_group | -0.031 [-0.186, +0.124] | -0.071 [-0.410, +0.269] | inconclusive / inconclusive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |

### Wall — bfloat16

#### Vision/action coupling

Source S37; n=192. Native absolute H6 MSE: P=4.751413366e-05, V=0.03504094327.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| visual_only | -0.138 [-0.182, -0.094] | +0.376 [-0.956, +1.708] | negative / inconclusive | Learned visual/joint directions do not reliably reduce native error; positive random-control effects weaken a learned-direction explanation. Physical cause unknown. |
| action_condition_only | -0.089 [-0.289, +0.112] | -1.268 [-7.158, +4.621] | inconclusive / inconclusive | Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested. |
| joint | -0.228 [-0.446, -0.010] | -1.471 [-7.444, +4.501] | negative / inconclusive | Learned visual/joint directions do not reliably reduce native error; positive random-control effects weaken a learned-direction explanation. Physical cause unknown. |
| joint_equal_standardized_energy | -0.157 [-0.312, -0.002] | -0.375 [-4.938, +4.187] | negative / inconclusive | Learned visual/joint directions do not reliably reduce native error; positive random-control effects weaken a learned-direction explanation. Physical cause unknown. |
| permuted_visual | +0.021 [+0.006, +0.035] | -0.079 [-0.481, +0.323] | positive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| permuted_joint | -0.060 [-0.262, +0.141] | -1.279 [-7.228, +4.670] | inconclusive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| matched_random | +0.516 [+0.316, +0.716] | -0.030 [-0.363, +0.303] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_standardized_energy | +0.346 [+0.211, +0.482] | -0.045 [-0.268, +0.177] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Action-response geometry

Source S38; n=192. Native absolute H6 MSE: P=4.748165296e-05, V=0.03488286671.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.001 [-0.016, +0.018] | -0.015 [-0.121, +0.090] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | -0.005 [-0.021, +0.010] | +0.024 [-0.054, +0.102] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.005 [-0.009, +0.020] | +0.027 [-0.101, +0.155] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | +0.002 [-0.012, +0.016] | +0.011 [-0.060, +0.082] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.032 [+0.018, +0.046] | -0.001 [-0.063, +0.062] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

### PointMaze — bfloat16

#### Action-response geometry

Source S40; n=200. Native absolute H6 MSE: P=0.003204402305, V=0.04300261579.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.014 [-0.181, +0.208] | +0.006 [-0.025, +0.037] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | +0.030 [-0.182, +0.242] | -0.014 [-0.048, +0.021] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | -0.033 [-0.142, +0.075] | +0.019 [-0.016, +0.054] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | -0.074 [-0.228, +0.080] | +0.009 [-0.027, +0.046] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | -0.045 [-0.165, +0.074] | -0.023 [-0.064, +0.019] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

**Coupling numerical-evidence gap:** completion is recorded, but the relevant aggregate was not available in the verified export. No zero or negative effect is assigned. The registered non-native arms are visual_only, action_condition_only, joint, joint_equal_standardized_energy, permuted_visual, permuted_joint, matched_random, matched_random_equal_standardized_energy, and zero_dose.

## FP32 sensitivity: every measured arm versus native

These rows do not replace the BF16 primary analysis. Original sweeps use their precision-specific fits; the revised fixed-response sensitivity uses the same frozen BF16-fitted map.

### Reach — float32

#### Vision/action coupling

Source S02; n=33. Native absolute H6 MSE: P=0.0007675998277, V=0.195025819.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| visual_only | +2.342 [+1.285, +3.400] | +0.582 [+0.066, +1.098] | positive / positive | Consistent with correction through visual and conditioning pathways. Joint doses and drop-one energy differences preclude an automatic synergy claim. |
| action_condition_only | +0.581 [+0.398, +0.763] | +0.017 [-0.015, +0.048] | positive / inconclusive | Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested. |
| joint | +2.911 [+1.833, +3.988] | +0.603 [+0.089, +1.117] | positive / positive | Consistent with correction through visual and conditioning pathways. Joint doses and drop-one energy differences preclude an automatic synergy claim. |
| joint_equal_standardized_energy | +2.091 [+1.329, +2.853] | +0.451 [+0.087, +0.814] | positive / positive | Consistent with correction through visual and conditioning pathways. Joint doses and drop-one energy differences preclude an automatic synergy claim. |
| permuted_visual | +0.110 [-0.035, +0.255] | -0.020 [-0.065, +0.025] | inconclusive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| permuted_joint | +0.687 [+0.476, +0.898] | -0.003 [-0.045, +0.038] | positive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| matched_random | -2.316 [-2.881, -1.750] | -0.284 [-0.362, -0.205] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_standardized_energy | -1.608 [-2.006, -1.211] | -0.196 [-0.251, -0.140] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Action-response geometry

Source S04; n=33. Native absolute H6 MSE: P=0.0007675997804, V=0.1950258175.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.000 [+0.000, +0.000] | +0.000 [-0.000, +0.000] | positive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | -0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | positive / positive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | positive / positive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Operator rank

Source S06; n=33. Native absolute H6 MSE: P=0.0007675997997, V=0.1950258177.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| rank1 | +1.930 [+1.411, +2.450] | +0.296 [+0.116, +0.477] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank4 | +1.918 [+1.488, +2.348] | +0.321 [+0.240, +0.401] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank8 | +1.326 [+0.988, +1.664] | +0.201 [+0.155, +0.248] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_rank1 | +0.151 [+0.015, +0.287] | +0.051 [+0.002, +0.099] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | +0.757 [+0.526, +0.989] | +0.100 [+0.067, +0.134] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank8 | +0.847 [+0.640, +1.055] | +0.110 [+0.083, +0.138] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Layer distribution (total rank 1)

Source S08; n=33. Native absolute H6 MSE: P=0.0007675994574, V=0.1950258153.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| single_block0 | +3.050 [+2.171, +3.929] | +0.497 [+0.218, +0.775] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block1 | +2.898 [+2.069, +3.728] | +0.469 [+0.177, +0.760] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block2 | +2.293 [+1.603, +2.982] | +0.375 [+0.138, +0.611] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block3 | +1.930 [+1.372, +2.489] | +0.296 [+0.103, +0.490] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block4 | +1.231 [+0.887, +1.575] | +0.184 [+0.073, +0.294] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block5 | +0.494 [+0.348, +0.640] | +0.075 [+0.023, +0.126] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| intermediate_blocks2_3 | +2.904 [+2.043, +3.765] | +0.456 [+0.157, +0.755] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| all_six_blocks | +2.385 [+1.672, +3.098] | +0.370 [+0.117, +0.624] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_single_block0 | +0.356 [+0.212, +0.500] | +0.065 [+0.029, +0.101] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block1 | +0.275 [+0.093, +0.457] | +0.060 [-0.014, +0.133] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block2 | +0.207 [+0.081, +0.333] | +0.035 [+0.011, +0.058] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block3 | +0.151 [+0.005, +0.297] | +0.051 [-0.002, +0.103] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block4 | +0.037 [+0.026, +0.049] | +0.010 [+0.006, +0.013] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block5 | +0.009 [-0.004, +0.022] | +0.004 [+0.001, +0.007] | inconclusive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_intermediate_blocks2_3 | +0.417 [+0.189, +0.645] | +0.071 [+0.005, +0.137] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_six_blocks | +0.305 [+0.170, +0.441] | +0.068 [+0.014, +0.122] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Spatial support (rank 1)

Source S10; n=33. Native absolute H6 MSE: P=0.000767599448, V=0.1950258148.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| one_patch | +0.114 [+0.063, +0.166] | +0.019 [+0.004, +0.033] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| contiguous_group | +1.271 [+0.935, +1.607] | +0.308 [+0.219, +0.398] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| equal_size_scattered_group | +0.369 [+0.236, +0.503] | +0.053 [+0.008, +0.098] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| all_patches | +1.930 [+1.376, +2.484] | +0.296 [+0.104, +0.489] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_one_patch | +0.301 [+0.164, +0.437] | +0.036 [+0.006, +0.065] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_contiguous_group | +0.458 [+0.201, +0.715] | +0.101 [+0.016, +0.186] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_size_scattered_group | +0.086 [+0.026, +0.145] | +0.013 [+0.000, +0.027] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_patches | +0.151 [+0.006, +0.296] | +0.051 [-0.001, +0.103] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| random_position_one_patch | +0.244 [+0.176, +0.313] | +0.014 [+0.006, +0.022] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_contiguous_group | +0.246 [+0.111, +0.380] | +0.038 [+0.018, +0.059] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_equal_size_scattered_group | +0.520 [+0.344, +0.696] | +0.085 [+0.036, +0.135] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |

#### Original combined/drop-one

Source S12; n=33. Native absolute H6 MSE: P=0.000767599737, V=0.1950258181.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| combined | +3.965 [+2.885, +5.046] | +0.757 [+0.353, +1.161] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| coupling_only | +2.091 [+1.321, +2.862] | +0.451 [+0.083, +0.818] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| rank4_only | +1.918 [+1.492, +2.344] | +0.321 [+0.241, +0.401] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| combined_rank1 | +3.955 [+2.775, +5.136] | +0.705 [+0.177, +1.233] | positive / positive | Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_coupling | -1.609 [-2.010, -1.207] | -0.196 [-0.252, -0.140] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | +0.757 [+0.528, +0.987] | +0.100 [+0.067, +0.133] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_combined | -0.834 [-1.120, -0.549] | -0.095 [-0.147, -0.042] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_combined_rank1 | -1.456 [-1.858, -1.053] | -0.145 [-0.215, -0.075] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Revised fixed response

Source S14; n=33. Native absolute H6 MSE: P=0.0007675993601, V=0.1950257015.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| fixed_rank4 | +2.324 [+1.880, +2.769] | +0.435 [+0.332, +0.537] | positive / positive | An offline mean response retains forecast benefits. This supports amortizing response estimation, not old/new equivalence or demonstrated task-success gains. |
| matched_random_fixed_rank4 | +1.130 [+0.848, +1.412] | +0.124 [+0.062, +0.186] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

### Reach-Wall — float32

#### Vision/action coupling

Source S16; n=27. Native absolute H6 MSE: P=0.0007344746812, V=0.2065858675.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| visual_only | +1.624 [-0.083, +3.331] | -0.608 [-1.436, +0.220] | inconclusive / inconclusive | Proprioceptive gains coexist with inconclusive visual changes. A modality-specific correction is plausible; success and wall-clearance effects are unmeasured here. |
| action_condition_only | +0.576 [+0.324, +0.828] | +0.035 [-0.006, +0.076] | positive / inconclusive | Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested. |
| joint | +2.155 [+0.529, +3.781] | -0.571 [-1.371, +0.230] | positive / inconclusive | Proprioceptive gains coexist with inconclusive visual changes. A modality-specific correction is plausible; success and wall-clearance effects are unmeasured here. |
| joint_equal_standardized_energy | +1.602 [+0.446, +2.759] | -0.346 [-0.912, +0.220] | positive / inconclusive | Proprioceptive gains coexist with inconclusive visual changes. A modality-specific correction is plausible; success and wall-clearance effects are unmeasured here. |
| permuted_visual | +0.285 [+0.072, +0.497] | -0.108 [-0.204, -0.012] | positive / negative | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| permuted_joint | +0.853 [+0.551, +1.155] | -0.073 [-0.161, +0.015] | positive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| matched_random | -2.365 [-3.044, -1.685] | -0.208 [-0.324, -0.093] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_standardized_energy | -1.619 [-2.092, -1.147] | -0.139 [-0.220, -0.058] | negative / negative | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Action-response geometry

Source S18; n=27. Native absolute H6 MSE: P=0.0007344746399, V=0.2065858672.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | +0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.000 [+0.000, +0.000] | +0.000 [-0.000, +0.000] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Operator rank

Source S20; n=27. Native absolute H6 MSE: P=0.0007344746722, V=0.2065858684.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| rank1 | +0.878 [+0.078, +1.677] | +0.196 [+0.032, +0.361] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank4 | +1.188 [+0.932, +1.445] | +0.234 [+0.185, +0.282] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| rank8 | +0.881 [+0.676, +1.086] | +0.146 [+0.117, +0.175] | positive / positive | Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_rank1 | +0.293 [+0.186, +0.400] | +0.065 [+0.040, +0.091] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | +0.559 [+0.406, +0.711] | +0.085 [+0.060, +0.111] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank8 | +0.624 [+0.467, +0.781] | +0.088 [+0.070, +0.107] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Layer distribution (total rank 1)

Source S22; n=27. Native absolute H6 MSE: P=0.000734474165, V=0.206585865.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| single_block0 | +1.334 [+0.159, +2.508] | +0.303 [+0.041, +0.565] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block1 | +1.180 [+0.044, +2.317] | +0.286 [+0.033, +0.538] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block2 | +0.929 [-0.029, +1.886] | +0.222 [+0.011, +0.434] | inconclusive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block3 | +0.878 [+0.046, +1.710] | +0.196 [+0.025, +0.368] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block4 | +0.555 [+0.058, +1.052] | +0.113 [+0.013, +0.213] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| single_block5 | +0.228 [+0.027, +0.430] | +0.051 [+0.010, +0.091] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| intermediate_blocks2_3 | +1.262 [+0.012, +2.512] | +0.285 [+0.017, +0.554] | positive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| all_six_blocks | +1.041 [-0.020, +2.101] | +0.244 [+0.018, +0.470] | inconclusive / positive | Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_single_block0 | +0.234 [+0.075, +0.392] | +0.063 [+0.027, +0.099] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block1 | +0.162 [+0.021, +0.302] | +0.054 [+0.025, +0.082] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block2 | +0.235 [+0.139, +0.330] | +0.046 [+0.020, +0.072] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block3 | +0.293 [+0.182, +0.404] | +0.065 [+0.039, +0.092] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block4 | +0.022 [+0.004, +0.041] | +0.007 [+0.003, +0.011] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block5 | +0.009 [-0.002, +0.019] | +0.005 [+0.002, +0.008] | inconclusive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_intermediate_blocks2_3 | +0.187 [+0.027, +0.347] | +0.073 [+0.035, +0.111] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_six_blocks | +0.158 [+0.068, +0.249] | +0.051 [+0.019, +0.083] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Spatial support (rank 1)

Source S24; n=27. Native absolute H6 MSE: P=0.0007344743041, V=0.2065858619.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| one_patch | +0.119 [+0.069, +0.168] | +0.038 [+0.019, +0.056] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| contiguous_group | +0.998 [+0.641, +1.355] | +0.202 [+0.123, +0.280] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| equal_size_scattered_group | +0.145 [+0.000, +0.289] | +0.039 [+0.007, +0.071] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| all_patches | +0.878 [+0.059, +1.697] | +0.196 [+0.028, +0.365] | positive / positive | Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_one_patch | +0.147 [+0.017, +0.278] | +0.049 [+0.012, +0.087] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_contiguous_group | +0.150 [-0.125, +0.425] | +0.086 [+0.025, +0.147] | inconclusive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_size_scattered_group | +0.058 [+0.016, +0.099] | +0.010 [+0.001, +0.019] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_patches | +0.293 [+0.184, +0.402] | +0.065 [+0.039, +0.091] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| random_position_one_patch | +0.103 [+0.033, +0.173] | +0.007 [-0.003, +0.018] | positive / inconclusive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_contiguous_group | +0.248 [+0.072, +0.424] | +0.038 [+0.007, +0.069] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_equal_size_scattered_group | +0.284 [+0.082, +0.486] | +0.052 [+0.013, +0.091] | positive / positive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |

#### Revised fixed response

Source S26; n=27. Native absolute H6 MSE: P=0.0007344744925, V=0.2065857551.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| fixed_rank4 | +2.131 [+1.698, +2.563] | +0.398 [+0.307, +0.490] | positive / positive | An offline mean response retains forecast benefits. This supports amortizing response estimation, not old/new equivalence or demonstrated task-success gains. |
| matched_random_fixed_rank4 | +0.646 [+0.422, +0.870] | +0.121 [+0.056, +0.187] | positive / positive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

### Push-T — float32

#### Vision/action coupling

Source S28; n=21. Native absolute H6 MSE: P=0.0001790726731, V=0.1140485999.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| visual_only | -0.752 [-1.610, +0.105] | -0.509 [-1.964, +0.946] | inconclusive / inconclusive | Visual-only and unscaled joint harm the primary endpoint. Possible mismatch with contact-dependent dynamics; no contact-mediated causal test establishes this explanation. |
| action_condition_only | +0.233 [-0.102, +0.568] | +0.013 [-0.094, +0.120] | inconclusive / inconclusive | Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested. |
| joint | -0.490 [-1.175, +0.195] | -0.478 [-1.862, +0.905] | inconclusive / inconclusive | Visual-only and unscaled joint harm the primary endpoint. Possible mismatch with contact-dependent dynamics; no contact-mediated causal test establishes this explanation. |
| joint_equal_standardized_energy | -0.339 [-0.827, +0.149] | -0.345 [-1.464, +0.774] | inconclusive / inconclusive | Visual-only and unscaled joint harm the primary endpoint. Possible mismatch with contact-dependent dynamics; no contact-mediated causal test establishes this explanation. |
| permuted_visual | -0.390 [-0.650, -0.131] | -0.230 [-0.552, +0.092] | negative / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| permuted_joint | -0.146 [-0.487, +0.196] | -0.218 [-0.549, +0.113] | inconclusive / inconclusive | Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export. |
| matched_random | +0.373 [+0.059, +0.688] | -0.022 [-0.263, +0.219] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_standardized_energy | +0.271 [+0.049, +0.493] | -0.010 [-0.176, +0.155] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Action-response geometry

Source S30; n=21. Native absolute H6 MSE: P=0.0001790726736, V=0.1140485908.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | -0.000 [-0.008, +0.008] | -0.005 [-0.021, +0.012] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | +0.005 [-0.003, +0.012] | +0.002 [-0.005, +0.010] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | -0.002 [-0.011, +0.007] | -0.000 [-0.010, +0.010] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | -0.002 [-0.011, +0.007] | -0.000 [-0.010, +0.010] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.004 [+0.001, +0.008] | +0.001 [-0.002, +0.004] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Operator rank

Source S32; n=21. Native absolute H6 MSE: P=0.0001790726867, V=0.1140485956.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| rank1 | -0.029 [-0.888, +0.831] | -0.007 [-0.077, +0.063] | inconclusive / inconclusive | All tested ranks are inconclusive; the selected subspace/site may miss useful error structure, or effects may be small/variable. Neither explanation is established. |
| rank4 | +0.103 [-0.238, +0.444] | +0.043 [-0.010, +0.095] | inconclusive / inconclusive | All tested ranks are inconclusive; the selected subspace/site may miss useful error structure, or effects may be small/variable. Neither explanation is established. |
| rank8 | -0.060 [-0.271, +0.151] | +0.032 [-0.015, +0.079] | inconclusive / inconclusive | All tested ranks are inconclusive; the selected subspace/site may miss useful error structure, or effects may be small/variable. Neither explanation is established. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_rank1 | +0.003 [-0.008, +0.014] | +0.009 [-0.015, +0.032] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank4 | +0.007 [-0.006, +0.021] | +0.009 [-0.004, +0.022] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_rank8 | +0.004 [-0.010, +0.017] | +0.006 [-0.004, +0.017] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Layer distribution (total rank 1)

Source S34; n=21. Native absolute H6 MSE: P=0.000179072709, V=0.1140485969.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| single_block0 | -0.251 [-1.366, +0.864] | +0.045 [-0.182, +0.273] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block1 | -0.069 [-1.040, +0.901] | +0.025 [-0.100, +0.150] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block2 | -0.125 [-1.141, +0.890] | -0.003 [-0.108, +0.101] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block3 | -0.029 [-0.968, +0.911] | -0.007 [-0.084, +0.069] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block4 | +0.057 [-0.523, +0.637] | +0.010 [-0.040, +0.061] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| single_block5 | +0.083 [-0.223, +0.390] | +0.007 [-0.023, +0.036] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| intermediate_blocks2_3 | -0.207 [-1.591, +1.177] | -0.009 [-0.132, +0.114] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| all_six_blocks | -0.173 [-1.416, +1.070] | +0.023 [-0.116, +0.162] | inconclusive / inconclusive | No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_single_block0 | -0.083 [-0.464, +0.298] | +0.042 [-0.199, +0.283] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block1 | -0.031 [-0.134, +0.072] | +0.034 [-0.034, +0.102] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block2 | -0.006 [-0.032, +0.020] | +0.034 [-0.007, +0.075] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block3 | +0.003 [-0.009, +0.015] | +0.009 [-0.017, +0.034] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block4 | +0.002 [-0.014, +0.017] | +0.006 [-0.007, +0.019] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_single_block5 | +0.006 [-0.004, +0.017] | +0.005 [-0.003, +0.013] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_intermediate_blocks2_3 | +0.001 [-0.023, +0.025] | +0.006 [-0.029, +0.041] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_six_blocks | -0.002 [-0.072, +0.068] | +0.043 [-0.028, +0.113] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

#### Spatial support (rank 1)

Source S36; n=21. Native absolute H6 MSE: P=0.0001790726245, V=0.1140486123.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| one_patch | -0.005 [-0.045, +0.034] | +0.021 [-0.073, +0.115] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| contiguous_group | +0.026 [-0.125, +0.178] | -0.009 [-0.045, +0.026] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| equal_size_scattered_group | +0.081 [-0.122, +0.283] | +0.021 [-0.018, +0.060] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| all_patches | -0.029 [-0.965, +0.908] | -0.007 [-0.084, +0.069] | inconclusive / inconclusive | No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random_one_patch | -0.002 [-0.029, +0.024] | +0.023 [-0.022, +0.068] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_contiguous_group | -0.001 [-0.017, +0.015] | +0.002 [-0.007, +0.012] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_equal_size_scattered_group | +0.006 [-0.011, +0.023] | +0.016 [-0.001, +0.034] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| matched_random_all_patches | +0.003 [-0.009, +0.015] | +0.009 [-0.017, +0.034] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |
| random_position_one_patch | +0.021 [-0.035, +0.076] | +0.002 [-0.003, +0.007] | inconclusive / inconclusive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_contiguous_group | -0.034 [-0.148, +0.081] | +0.020 [-0.038, +0.077] | inconclusive / inconclusive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |
| random_position_equal_size_scattered_group | +0.044 [-0.198, +0.286] | +0.011 [-0.016, +0.039] | inconclusive / inconclusive | Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit. |

### Wall — float32

#### Action-response geometry

Source S39; n=192. Native absolute H6 MSE: P=4.704047914e-05, V=0.03651805053.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.001 [-0.000, +0.001] | +0.006 [-0.027, +0.039] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | -0.000 [-0.000, +0.000] | +0.007 [-0.005, +0.018] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.001 [-0.000, +0.001] | +0.002 [-0.022, +0.025] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | +0.001 [-0.000, +0.001] | +0.002 [-0.022, +0.025] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | +0.002 [+0.001, +0.002] | +0.000 [-0.001, +0.001] | positive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

**Coupling numerical-evidence gap:** completion is recorded, but the relevant aggregate was not available in the verified export. No zero or negative effect is assigned. The registered non-native arms are visual_only, action_condition_only, joint, joint_equal_standardized_energy, permuted_visual, permuted_joint, matched_random, matched_random_equal_standardized_energy, and zero_dose.

### PointMaze — float32

#### Action-response geometry

Source S41; n=200. Native absolute H6 MSE: P=0.00324414193, V=0.04935516845.

| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |
|---|---:|---:|---|---|
| equal_anchor_linear | +0.007 [-0.011, +0.025] | +0.008 [+0.003, +0.013] | inconclusive / positive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| cubic | -0.007 [-0.023, +0.009] | +0.001 [-0.003, +0.004] | inconclusive / inconclusive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| projected_cubic | +0.009 [-0.010, +0.027] | +0.008 [+0.003, +0.013] | inconclusive / positive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| reflected_curvature | +0.009 [-0.009, +0.027] | +0.008 [+0.003, +0.013] | inconclusive / positive | Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates. |
| zero_dose | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | identity / identity | Identity check: zero edit reproduces the native output. |
| matched_random | -0.002 [-0.006, +0.003] | -0.000 [-0.001, +0.001] | inconclusive / inconclusive | Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction. |

**Coupling numerical-evidence gap:** completion is recorded, but the relevant aggregate was not available in the verified export. No zero or negative effect is assigned. The registered non-native arms are visual_only, action_condition_only, joint, joint_equal_standardized_energy, permuted_visual, permuted_joint, matched_random, matched_random_equal_standardized_energy, and zero_dose.

## DROID: all completed coupling arms

Metric: recorded-plan action score, higher is better. These are score-point differences, not error-reduction percentages or robot success. Each arm has 64 endpoints from the same 15 recordings; zero physical robot executions. All 16 registered simultaneous 95% contrast intervals include zero.

| Arm | Action score | XYZ action error | Score difference vs native [95% interval] | Interpretation / possible explanation |
|---|---:|---:|---:|---|
| action_condition_only | 51.110 | 0.036113 | +0.010 [-0.042, +0.061] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |
| joint | 50.853 | 0.036434 | -0.247 [-1.176, +0.598] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |
| joint_equal_standardized_energy | 51.070 | 0.036162 | -0.029 [-0.555, +0.615] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |
| matched_random | 51.237 | 0.035954 | +0.137 [-0.286, +0.498] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |
| matched_random_equal_standardized_energy | 51.275 | 0.035907 | +0.175 [-0.042, +0.434] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |
| native | 51.100 | 0.036125 | Reference | Baseline. |
| permuted_joint | 51.211 | 0.035986 | +0.112 [-0.230, +0.578] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |
| permuted_visual | 51.173 | 0.036034 | +0.073 [-0.254, +0.543] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |
| visual_only | 50.832 | 0.036460 | -0.267 [-1.233, +0.587] | Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation. |

### All DROID registered paired contrasts

| Contrast | Score difference [simultaneous 95% interval] |
|---|---:|
| visual_only_vs_native | -0.2673 [-1.2329, +0.5868] |
| action_condition_only_vs_native | +0.0103 [-0.0417, +0.0613] |
| joint_vs_native | -0.2470 [-1.1755, +0.5978] |
| joint_equal_standardized_energy_vs_native | -0.0293 [-0.5549, +0.6146] |
| permuted_visual_vs_native | +0.0733 [-0.2536, +0.5430] |
| permuted_joint_vs_native | +0.1118 [-0.2300, +0.5782] |
| matched_random_vs_native | +0.1371 [-0.2860, +0.4978] |
| matched_random_equal_standardized_energy_vs_native | +0.1750 [-0.0415, +0.4338] |
| joint_vs_visual_only | +0.0203 [-0.0000, +0.0744] |
| joint_vs_action_condition_only | -0.2572 [-1.1875, +0.5873] |
| joint_vs_matched_random | -0.3841 [-1.0709, +0.2143] |
| joint_vs_permuted_joint | -0.3588 [-1.1385, +0.3413] |
| equal_energy_joint_vs_visual_only | +0.2380 [-0.3195, +0.9339] |
| equal_energy_joint_vs_action_condition_only | -0.0395 [-0.5637, +0.6191] |
| equal_energy_joint_vs_matched_random | -0.2042 [-0.6770, +0.2747] |
| factorial_score_interaction | +0.0101 [-0.0365, +0.0749] |

## Control comparisons and all-horizon metrics

The [full H6 contrast CSV](../data/all_task_ablation_contrasts.csv) retains every registered contrast, including candidate-versus-random, cross-rank/support and drop-one contrasts, both endpoints and both precisions. Its frozen-minimum column is **per contrast**, not a claim that the complete arm gate passed. The [all-horizon metric CSV](../data/all_task_ablation_metrics.csv) retains all reported per-arm L1/MSE horizon means and error reductions, including native. Statistical intervals were registered for the H6 contrast family; no new intervals or tests were manufactured for other horizons.

## Verified sources

[Machine-readable source ledger](../data/all_task_ablation_sources.json). This audit checks existing aggregate hashes/completion bindings; it does not repeat the underlying trajectory bootstrap or every raw-file check. Original archives and scientific records remain unchanged.

- **S01** Reach / vision_action_coupling / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/vision_action_coupling/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/vision_action_coupling/report.json). SHA256 `f81678b12eceb087500dce093b2898a943ceadcc25a787501119dcffe3b94015`. report hash matches export and adjacent DONE.
- **S02** Reach / vision_action_coupling / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/vision_action_coupling/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/vision_action_coupling/report.json). SHA256 `0572f8ac77ab83246db9f8e84876f3f5924a8356cb4501594df523b239bef285`. report hash matches export and adjacent DONE.
- **S03** Reach / action_response_geometry / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/action_response_geometry/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/action_response_geometry/report.json). SHA256 `371d96899c5fb5f7fd6d01049e466c739eea1fd746e307aeeec9223c078273b5`. report hash matches export and adjacent DONE.
- **S04** Reach / action_response_geometry / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/action_response_geometry/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/action_response_geometry/report.json). SHA256 `65cea00baa18173342b34f2abb0330652fd133a4da0acb7db3514a0d9594c016`. report hash matches export and adjacent DONE.
- **S05** Reach / operator_rank / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/operator_rank/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/operator_rank/report.json). SHA256 `71fe3dd8d5fc54e33c591b5b88db2e5d98641dd5ea6feacff2bc697c68eb8cd2`. report hash matches export and adjacent DONE.
- **S06** Reach / operator_rank / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/operator_rank/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/operator_rank/report.json). SHA256 `bd7c7fd8e4eaab775873848604dcacedb687ee0101069295c71363848f28292a`. report hash matches export and adjacent DONE.
- **S07** Reach / distribution_layer / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/distribution_layer/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/distribution_layer/report.json). SHA256 `fe15e327d85f1c64697feb5b5d5b98db01a7a0593b5286dc39c45e9487352e70`. report hash matches export and adjacent DONE.
- **S08** Reach / distribution_layer / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/distribution_layer/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/distribution_layer/report.json). SHA256 `ee5e0b665d0caade3af2a48e69b6fae5c4d0eb908c259b1d9bbabb8b14b100a9`. report hash matches export and adjacent DONE.
- **S09** Reach / distribution_spatial / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/distribution_spatial/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/distribution_spatial/report.json). SHA256 `d508efe15492cbf7aca2db2d52a3be0d391b008c7a43a2bc55aa5807a041421c`. report hash matches export and adjacent DONE.
- **S10** Reach / distribution_spatial / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/distribution_spatial/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach/distribution_spatial/report.json). SHA256 `faa05aec9e8c9756cc82e166a8171d0e29ddbb6224b8e7a1d79ef95f6a994b2c`. report hash matches export and adjacent DONE.
- **S11** Reach / combined / bfloat16: [artifacts/offline_study/primary-durable-20260907/combined-reach-author-20260907/analysis-v1/bfloat16/report.json](../../artifacts/offline_study/primary-durable-20260907/combined-reach-author-20260907/analysis-v1/bfloat16/report.json). SHA256 `846672dfd91f361ebb9baf90bae94b82b09d8537433c09bf1b1a2c6f3afdff76`. report hash matches export and adjacent DONE.
- **S12** Reach / combined / float32: [artifacts/offline_study/primary-durable-20260907/combined-reach-author-20260907/analysis-v1/float32/report.json](../../artifacts/offline_study/primary-durable-20260907/combined-reach-author-20260907/analysis-v1/float32/report.json). SHA256 `66bf550e6d42bbaa8428a27db70e35a6b430e787e3b51170fcfff24b633d2c4b`. report hash matches export and adjacent DONE.
- **S13** Reach / fixed_response / bfloat16: [artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach/bfloat16/report.json](../../artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach/bfloat16/report.json). SHA256 `3f00270e9750de568ff8843278e36de65fef4b6b8f499b9909dadb87c1100558`. report hash matches export and adjacent DONE.
- **S14** Reach / fixed_response / float32: [artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach/float32/report.json](../../artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach/float32/report.json). SHA256 `994445008044c654118a17810ef432fc52adc65af26849930477b15bae3dd4da`. report hash matches export and adjacent DONE.
- **S15** Reach-Wall / vision_action_coupling / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/vision_action_coupling/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/vision_action_coupling/report.json). SHA256 `35cc14ff685d663542401314f10e947a23fb06d2a903a15b5c7704b77d8b3dc0`. report hash matches export and adjacent DONE.
- **S16** Reach-Wall / vision_action_coupling / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/vision_action_coupling/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/vision_action_coupling/report.json). SHA256 `16db63b5a7eeabb4e77c735b1849c1ff1c10554a88067fd35e3816d74f587efe`. report hash matches export and adjacent DONE.
- **S17** Reach-Wall / action_response_geometry / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/action_response_geometry/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/action_response_geometry/report.json). SHA256 `3782752f0774e27c140111181ba6af81a8d581c698b4bfb1785f619b8d317726`. report hash matches export and adjacent DONE.
- **S18** Reach-Wall / action_response_geometry / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/action_response_geometry/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/action_response_geometry/report.json). SHA256 `b703b3b463cfd313fcb0548299a2363971032c573238393d5e411f7837c89801`. report hash matches export and adjacent DONE.
- **S19** Reach-Wall / operator_rank / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/operator_rank/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/operator_rank/report.json). SHA256 `2e9101d8f1b9e0cc5a9d941e4928e4344f201716e32e5a4e7ed98361983ce219`. report hash matches export and adjacent DONE.
- **S20** Reach-Wall / operator_rank / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/operator_rank/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/operator_rank/report.json). SHA256 `9a5482ca0dd43f811e980bf57677164830a34750d4a1dd5af963d67e122bd67b`. report hash matches export and adjacent DONE.
- **S21** Reach-Wall / distribution_layer / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/distribution_layer/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/distribution_layer/report.json). SHA256 `ba39a4f74ce097893077832c8e6d3c3acbcdd00f80b62e1ce3210c15e3e56ded`. report hash matches export and adjacent DONE.
- **S22** Reach-Wall / distribution_layer / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/distribution_layer/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/distribution_layer/report.json). SHA256 `985a62fd1265121309cdc4dee0bfb6c2b9e985e8229b19365eae5e1b77a111a6`. report hash matches export and adjacent DONE.
- **S23** Reach-Wall / distribution_spatial / bfloat16: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/distribution_spatial/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach-wall/distribution_spatial/report.json). SHA256 `5c3e80c01f747c38f574b96239addef6bc0e4a0edcff25a9d742d710b9286543`. report hash matches export and adjacent DONE.
- **S24** Reach-Wall / distribution_spatial / float32: [artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/distribution_spatial/report.json](../../artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/float32/reach-wall/distribution_spatial/report.json). SHA256 `b56572ba8e9ce8cedf3ec9962e49acb694888e2456ecfa04c3ac493692660968`. report hash matches export and adjacent DONE.
- **S25** Reach-Wall / fixed_response / bfloat16: [artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach-wall/bfloat16/report.json](../../artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach-wall/bfloat16/report.json). SHA256 `b265a2d33d22988010536a2194cbd853e5d5a6d57cf3962ae49fe706255805ba`. report hash matches export and adjacent DONE.
- **S26** Reach-Wall / fixed_response / float32: [artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach-wall/float32/report.json](../../artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach-wall/float32/report.json). SHA256 `58813f3fb499fb4b2acbc599ba1df379fcc3b501ac6e7bddf98621a8c6d3bd2a`. report hash matches export and adjacent DONE.
- **S27** Push-T / vision_action_coupling / bfloat16: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/vision_action_coupling/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/vision_action_coupling/report.json). SHA256 `a9f5e64d9034a5b782afaa49b5e91fb4d3703069d8a2f2ec733fda866b69a0bb`. report hash matches export and adjacent DONE.
- **S28** Push-T / vision_action_coupling / float32: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/vision_action_coupling/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/vision_action_coupling/report.json). SHA256 `4d5bc09747eecb9c4b1492c09c436d52a276ac9f869a36b3cb0d3bfba221e3b1`. report hash matches export and adjacent DONE.
- **S29** Push-T / action_response_geometry / bfloat16: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/action_response_geometry/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/action_response_geometry/report.json). SHA256 `a9a4717b04a044a20dc502467d1fd606fe4c08e9c86725e419bd34d08b2a3246`. report hash matches export and adjacent DONE.
- **S30** Push-T / action_response_geometry / float32: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/action_response_geometry/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/action_response_geometry/report.json). SHA256 `457e33775369b171afb4e5d3f1e3d86d62fc3352da56b573ca0f58e06db618fd`. report hash matches export and adjacent DONE.
- **S31** Push-T / operator_rank / bfloat16: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/operator_rank/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/operator_rank/report.json). SHA256 `cec1c8899d383657971b98e0e4cad66a4bf524f3339f6a9d47f5749a9bdf87e8`. report hash matches export and adjacent DONE.
- **S32** Push-T / operator_rank / float32: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/operator_rank/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/operator_rank/report.json). SHA256 `68692e0211c7acd24dcd8f14793b271e63e85ecf48d9a24717bb0b6d2650fb54`. report hash matches export and adjacent DONE.
- **S33** Push-T / distribution_layer / bfloat16: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/distribution_layer/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/distribution_layer/report.json). SHA256 `87c80bba69e39d37d6e72685e26af0899d73700a17561a34424467864a2eb963`. report hash matches export and adjacent DONE.
- **S34** Push-T / distribution_layer / float32: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/distribution_layer/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/distribution_layer/report.json). SHA256 `37b9eb3ee5911b7cb16b204e2a45420009d34dcb55eafddf43e2475da5d85711`. report hash matches export and adjacent DONE.
- **S35** Push-T / distribution_spatial / bfloat16: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/distribution_spatial/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/distribution_spatial/report.json). SHA256 `a4fed56436d454a767372aeeab60792eb7eec65b30f39c471e8e67572dfbead5`. report hash matches export and adjacent DONE.
- **S36** Push-T / distribution_spatial / float32: [artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/distribution_spatial/report.json](../../artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/distribution_spatial/report.json). SHA256 `2403e2505b2b8f38864d6f49f4a97d2a44db8a3340dda6d2ba42c5331c180486`. report hash matches export and adjacent DONE.
- **S37** Wall / vision_action_coupling / bfloat16: [artifacts/offline_study/navigation-analysis-compact-20260907-v1/bfloat16/wall/vision_action_coupling/report.json](../../artifacts/offline_study/navigation-analysis-compact-20260907-v1/bfloat16/wall/vision_action_coupling/report.json). SHA256 `9aff38b6d86eaa6072d9187d989e6d167546de06b073925983e681ab95b3145f`. preserved compact report; original full hash recorded, full raw report not reread.
- **S38** Wall / action_response_geometry / bfloat16: [artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz](../../artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz). SHA256 `643247c9860778d46c7deea714e045de15d4a401d8bc1b96a506f405d2def9bd`. archive checksum and member report/DONE match.
- **S39** Wall / action_response_geometry / float32: [artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz](../../artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz). SHA256 `3f5315532e50c29bd794996dfc29f416e00cf7b717e0e4031cee6be736687077`. archive checksum and member report/DONE match.
- **S40** PointMaze / action_response_geometry / bfloat16: [artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz](../../artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz). SHA256 `34797270e44abf8937457060b8fda437fe02bb46726e1acf7e6b49e4beb0bfde`. archive checksum and member report/DONE match.
- **S41** PointMaze / action_response_geometry / float32: [artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz](../../artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz). SHA256 `550fc44b126621b221d5abca79e718e96953de0455a4b4c2947ac4ce585b6d95`. archive checksum and member report/DONE match.
- **DROID** [artifacts/offline_study/live-20260908T151000Z/instance-50259194-droid-parallel/instance-50259194-results.tar.gz](../../artifacts/offline_study/live-20260908T151000Z/instance-50259194-droid-parallel/instance-50259194-results.tar.gz). Member `droid-coupling-behavior-20260908-v3/analysis/report.json`; report SHA256 `b998218db8f1b87d18150bed04b9eec29731ef50af5e8c203f7ba028b9f83de8` matches archived DONE.
