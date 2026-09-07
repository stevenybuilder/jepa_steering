# Task-geometry emergence screen

Discovery-only, leave-one-collection-seed-out analysis. A one-third-depth emergence layer was not assumed.

| Family | Coordinate | Peak layer | Peak R² | Largest upward jump | Near-peak layers |
|---|---|---:|---:|---:|---|
| encoder | realized_xz_direction | 11 | 0.046 | +0.033 | [3, 4, 5, 6, 7, 8, 9, 10, 11] |
| encoder | goal_xz_direction | 11 | 0.342 | +0.076 | [9, 10, 11] |
| encoder | motion_magnitude | 5 | 0.723 | +0.015 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |
| encoder | wall_signed_distance | 6 | 0.966 | +0.035 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |
| encoder | progress | 4 | 0.944 | +0.006 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |
| predictor | realized_xz_direction | 3 | 0.573 | +0.026 | [0, 1, 2, 3] |
| predictor | goal_xz_direction | 3 | 0.409 | +0.023 | [0, 1, 2, 3, 4, 5] |
| predictor | motion_magnitude | 3 | 0.812 | +0.030 | [2, 3, 4, 5] |
| predictor | wall_signed_distance | 2 | 0.986 | +0.005 | [0, 1, 2, 3, 4, 5] |
| predictor | progress | 3 | 0.959 | +0.002 | [0, 1, 2, 3, 4, 5] |

## Predictor block 3: iterative orthogonal probe dimensions

| Coordinate | Estimated dimension | Initial R² | R² after first removal |
|---|---:|---:|---:|
| realized_xz_direction | 10 | 0.573 | 0.445 |
| goal_xz_direction | >=24 | 0.409 | 0.373 |
| motion_magnitude | >=12 | 0.812 | 0.776 |
| wall_signed_distance | >=12 | 0.984 | 0.952 |
| progress | >=12 | 0.959 | 0.945 |
