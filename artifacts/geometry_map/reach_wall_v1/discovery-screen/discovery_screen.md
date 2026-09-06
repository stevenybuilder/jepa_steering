# Reach-Wall discovery screen

Samples: 2850 transitions from 150 whole trajectories.
Validation and confirmation episodes were not loaded.

## Strongest layer per target

| Target | Encoder | Score / over time | Predictor | Score / over time |
|---|---:|---:|---:|---:|
| goal_distance | 8 | 0.931 / +0.022 | 5 | 0.942 / +0.033 |
| progress | 4 | 0.944 / +0.022 | 3 | 0.959 / +0.037 |
| hand_x | 6 | 0.796 / +0.738 | 1 | 0.985 / +0.927 |
| hand_y | 3 | 0.991 / +0.014 | 0 | 0.999 / +0.022 |
| hand_z | 4 | 0.991 / +0.983 | 1 | 0.996 / +0.988 |
| goal_vector_x | 2 | 0.083 / +0.003 | 3 | 0.238 / +0.157 |
| goal_vector_y | 7 | 0.966 / +0.009 | 1 | 0.973 / +0.016 |
| goal_vector_z | 11 | 0.488 / +0.444 | 3 | 0.611 / +0.567 |
| wall_signed_distance | 6 | 0.966 / +0.824 | 2 | 0.986 / +0.844 |
| height_above_wall_top | 4 | 0.991 / +0.983 | 1 | 0.996 / +0.988 |
| straight_path_intersects_wall | 8 | 0.989 / +0.087 | 5 | 0.991 / +0.089 |
| expert_detour_gate | 8 | 0.999 / +0.020 | 3 | 0.999 / +0.021 |
| action_dx | 11 | 0.001 / -0.002 | 0 | 0.972 / +0.968 |
| action_dy | 2 | 0.629 / -0.000 | 1 | 0.984 / +0.355 |
| action_dz | 5 | 0.050 / +0.023 | 1 | 0.964 / +0.936 |
| action_translation_magnitude | 5 | 0.511 / +0.038 | 1 | 0.624 / +0.151 |
| realized_dx | 11 | 0.027 / +0.021 | 3 | 0.649 / +0.643 |
| realized_dy | 7 | 0.855 / +0.062 | 3 | 0.936 / +0.142 |
| realized_dz | 7 | 0.142 / +0.071 | 3 | 0.717 / +0.646 |
| realized_hand_delta_magnitude | 5 | 0.723 / +0.109 | 3 | 0.812 / +0.199 |
| reward_sum | 7 | 0.952 / +0.037 | 3 | 0.965 / +0.051 |
