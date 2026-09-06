# Relational transport / residual scoring (descriptive)

Scope: descriptive; relational organisation and residual scoring; not localisation.

## Token group `hazard_corridor`
n = 54 discovery scenes; nulls: 100 permutations / 100 draws.

### Field characterisation (true | model)

| arm.field | PC1 true | PC1 model | sph.var true (uniform) | z true | sph.var model | z model | mean-dir LOSO true/model |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.564 | +0.740 | +0.024 (+0.863) | +275.6 | +0.002 | +272.5 | +0.975 / +0.998 |
| A.cone_did | +0.541 | +0.894 | +0.031 (+0.863) | +306.8 | +0.002 | +238.1 | +0.968 / +0.998 |
| A.null_did | +0.346 | +0.775 | +0.134 (+0.864) | +240.7 | +0.074 | +217.0 | +0.861 / +0.924 |
| A.ped_did_relational | +0.557 | +0.707 | +0.024 (+0.864) | +292.9 | +0.001 | +311.4 | +0.975 / +0.999 |
| A.cone_did_relational | +0.515 | +0.936 | +0.046 (+0.864) | +266.6 | +0.001 | +314.8 | +0.952 / +0.999 |
| A.action | +0.365 | +0.437 | +0.109 (+0.863) | +236.3 | +0.010 | +269.8 | +0.887 / +0.989 |
| B.ped_did | +0.667 | +0.757 | +0.036 (+0.865) | +272.2 | +0.001 | +333.9 | +0.963 / +0.999 |
| B.cone_did | +0.565 | +0.574 | +0.052 (+0.864) | +236.5 | +0.005 | +311.8 | +0.946 / +0.994 |
| B.null_did | +0.346 | +0.557 | +0.134 (+0.864) | +234.6 | +0.007 | +277.8 | +0.861 / +0.993 |
| B.ped_did_relational | +0.633 | +0.778 | +0.066 (+0.864) | +262.0 | +0.001 | +279.4 | +0.931 / +0.999 |
| B.cone_did_relational | +0.553 | +0.543 | +0.050 (+0.864) | +374.6 | +0.005 | +256.0 | +0.948 / +0.995 |
| B.action | +0.366 | +0.470 | +0.109 (+0.864) | +257.7 | +0.009 | +335.9 | +0.887 / +0.991 |

### Residual cosine, model vs truth (LOSO true-field mean subtracted)

| arm.field | model raw | model residual | copy-delta raw | copy-delta residual | field-mean raw | model - copy residual (p) | model - copy raw (p) |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.951 | +0.113 | +0.966 | +0.411 | +0.975 | -0.298 (+0.000) | -0.015 (+0.007) |
| A.cone_did | +0.920 | -0.011 | +0.972 | +0.549 | +0.968 | -0.560 (+0.000) | -0.052 (+0.000) |
| A.null_did | +0.491 | -0.041 | +0.866 | +0.463 | +0.861 | -0.504 (+0.000) | -0.376 (+0.000) |
| A.ped_did_relational | +0.945 | +0.079 | +0.967 | +0.396 | +0.975 | -0.316 (+0.000) | -0.022 (+0.000) |
| A.cone_did_relational | +0.832 | -0.107 | +0.953 | +0.473 | +0.952 | -0.579 (+0.000) | -0.121 (+0.000) |
| A.action | +0.535 | -0.026 | +0.891 | +0.516 | +0.887 | -0.542 (+0.000) | -0.356 (+0.000) |
| B.ped_did | +0.892 | -0.122 | +0.968 | +0.580 | +0.962 | -0.702 (+0.000) | -0.076 (+0.000) |
| B.cone_did | +0.845 | -0.025 | +0.944 | +0.487 | +0.946 | -0.512 (+0.000) | -0.099 (+0.000) |
| B.null_did | +0.616 | -0.054 | +0.866 | +0.463 | +0.861 | -0.517 (+0.000) | -0.250 (+0.000) |
| B.ped_did_relational | +0.793 | -0.163 | +0.929 | +0.510 | +0.929 | -0.673 (+0.000) | -0.136 (+0.000) |
| B.cone_did_relational | +0.854 | -0.053 | +0.945 | +0.473 | +0.948 | -0.527 (+0.000) | -0.091 (+0.000) |
| B.action | +0.322 | -0.039 | +0.891 | +0.516 | +0.887 | -0.555 (+0.000) | -0.569 (+0.000) |

### Relational transport (residual cosine; score - null, sign-flip p; 'centred' = LOSO-centred variant vs its permutation null)

| block | config.field | residual [CI] | raw | |mag| rho | - perm (p) | - iso Gauss (p) | - cov Gauss (p) | - rotated (p) | centred residual, - perm (p) |
|---|---|---|---|---|---|---|---|---|---|
| true_cross_arm | true_A->true_B.ped_did | +0.525 [+0.460, +0.588] | +0.963 | -0.974 | +0.523 (+0.000) | +0.444 (+0.000) | +0.535 (+0.000) | +0.425 (+0.000) | +0.705, +0.699 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did | +0.678 [+0.636, +0.720] | +0.947 | +0.008 | +0.687 (+0.000) | +0.765 (+0.000) | +0.680 (+0.000) | +0.731 (+0.000) | +0.678, +0.668 (+0.000) |
| true_cross_arm | true_A->true_B.null_did | +0.575 [+0.524, +0.626] | +0.873 | +0.405 | +0.590 (+0.000) | +0.609 (+0.000) | +0.578 (+0.000) | +0.593 (+0.000) | +0.754, +0.753 (+0.000) |
| true_cross_arm | true_A->true_B.ped_did_relational | +0.603 [+0.563, +0.642] | +0.930 | -0.977 | +0.611 (+0.000) | +0.503 (+0.000) | +0.607 (+0.000) | +0.470 (+0.000) | +0.606, +0.616 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did_relational | +0.597 [+0.523, +0.658] | +0.949 | +0.206 | +0.603 (+0.000) | +0.700 (+0.000) | +0.593 (+0.000) | +0.672 (+0.000) | +0.698, +0.696 (+0.000) |
| true_cross_arm | true_A->true_B.action | +0.601 [+0.557, +0.645] | +0.894 | +0.423 | +0.613 (+0.000) | +0.633 (+0.000) | +0.596 (+0.000) | +0.619 (+0.000) | +0.726, +0.721 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did | +0.450 [+0.396, +0.504] | +0.975 | -0.499 | +0.455 (+0.000) | +0.496 (+0.000) | +0.444 (+0.000) | +0.491 (+0.000) | +0.461, +0.464 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did | +0.607 [+0.527, +0.683] | +0.970 | -0.432 | +0.615 (+0.000) | +0.599 (+0.000) | +0.606 (+0.000) | +0.596 (+0.000) | +0.702, +0.711 (+0.000) |
| true_cross_arm | true_B->true_A.null_did | +0.575 [+0.524, +0.626] | +0.873 | +0.406 | +0.582 (+0.000) | +0.610 (+0.000) | +0.580 (+0.000) | +0.589 (+0.000) | +0.754, +0.752 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did_relational | +0.431 [+0.369, +0.493] | +0.976 | -0.654 | +0.428 (+0.000) | +0.475 (+0.000) | +0.437 (+0.000) | +0.472 (+0.000) | +0.439, +0.436 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did_relational | +0.574 [+0.509, +0.639] | +0.955 | -0.942 | +0.584 (+0.000) | +0.581 (+0.000) | +0.576 (+0.000) | +0.580 (+0.000) | +0.631, +0.631 (+0.000) |
| true_cross_arm | true_B->true_A.action | +0.601 [+0.557, +0.645] | +0.894 | +0.423 | +0.606 (+0.000) | +0.634 (+0.000) | +0.603 (+0.000) | +0.623 (+0.000) | +0.726, +0.723 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did | +0.784 [+0.691, +0.866] | +0.999 | -0.997 | +0.815 (+0.000) | +0.606 (+0.000) | +0.778 (+0.000) | +0.589 (+0.000) | +0.835, +0.848 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did | +0.710 [+0.677, +0.742] | +0.994 | -0.958 | +0.705 (+0.000) | +0.626 (+0.000) | +0.720 (+0.000) | +0.619 (+0.000) | +0.749, +0.750 (+0.000) |
| model_cross_arm | model_A->model_B.null_did | +0.595 [+0.527, +0.662] | +0.993 | +0.137 | +0.594 (+0.000) | +0.545 (+0.000) | +0.597 (+0.000) | +0.528 (+0.000) | +0.569, +0.572 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did_relational | +0.774 [+0.678, +0.860] | +0.999 | -0.999 | +0.787 (+0.000) | +0.603 (+0.000) | +0.755 (+0.000) | +0.585 (+0.000) | +0.826, +0.845 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did_relational | +0.697 [+0.665, +0.730] | +0.995 | -0.969 | +0.717 (+0.000) | +0.611 (+0.000) | +0.682 (+0.000) | +0.605 (+0.000) | +0.716, +0.722 (+0.000) |
| model_cross_arm | model_A->model_B.action | +0.740 [+0.699, +0.783] | +0.991 | -0.938 | +0.745 (+0.000) | +0.680 (+0.000) | +0.745 (+0.000) | +0.673 (+0.000) | +0.839, +0.850 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did | +0.705 [+0.650, +0.758] | +0.998 | -0.993 | +0.726 (+0.000) | +0.669 (+0.000) | +0.698 (+0.000) | +0.667 (+0.000) | +0.815, +0.830 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did | +0.791 [+0.689, +0.881] | +0.998 | -0.983 | +0.797 (+0.000) | +0.638 (+0.000) | +0.795 (+0.000) | +0.626 (+0.000) | +0.814, +0.810 (+0.000) |
| model_cross_arm | model_B->model_A.null_did | +0.576 [+0.462, +0.687] | +0.924 | -0.704 | +0.586 (+0.000) | +0.551 (+0.000) | +0.586 (+0.000) | +0.548 (+0.000) | +0.666, +0.675 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did_relational | +0.730 [+0.682, +0.776] | +0.999 | -0.997 | +0.752 (+0.000) | +0.684 (+0.000) | +0.730 (+0.000) | +0.682 (+0.000) | +0.801, +0.813 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did_relational | +0.818 [+0.712, +0.909] | +0.999 | -0.992 | +0.827 (+0.000) | +0.645 (+0.000) | +0.830 (+0.000) | +0.631 (+0.000) | +0.843, +0.837 (+0.000) |
| model_cross_arm | model_B->model_A.action | +0.600 [+0.549, +0.651] | +0.989 | -0.961 | +0.614 (+0.000) | +0.562 (+0.000) | +0.608 (+0.000) | +0.561 (+0.000) | +0.850, +0.854 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did | +0.357 [+0.288, +0.428] | +0.975 | -0.966 | +0.354 (+0.000) | +0.403 (+0.000) | +0.353 (+0.000) | +0.398 (+0.000) | +0.425, +0.432 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did | +0.635 [+0.580, +0.687] | +0.968 | -0.848 | +0.636 (+0.000) | +0.627 (+0.000) | +0.652 (+0.000) | +0.625 (+0.000) | +0.672, +0.673 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.null_did | +0.516 [+0.459, +0.563] | +0.868 | +0.256 | +0.527 (+0.000) | +0.553 (+0.000) | +0.519 (+0.000) | +0.533 (+0.000) | +0.516, +0.523 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did_relational | +0.345 [+0.287, +0.406] | +0.975 | -0.902 | +0.347 (+0.000) | +0.389 (+0.000) | +0.336 (+0.000) | +0.385 (+0.000) | +0.399, +0.398 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did_relational | +0.563 [+0.516, +0.612] | +0.953 | -0.613 | +0.570 (+0.000) | +0.570 (+0.000) | +0.561 (+0.000) | +0.571 (+0.000) | +0.570, +0.575 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.action | +0.532 [+0.478, +0.584] | +0.887 | -0.949 | +0.524 (+0.000) | +0.566 (+0.000) | +0.529 (+0.000) | +0.552 (+0.000) | +0.607, +0.613 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did | +0.623 [+0.549, +0.692] | +0.962 | -0.996 | +0.646 (+0.000) | +0.543 (+0.000) | +0.628 (+0.000) | +0.522 (+0.000) | +0.689, +0.696 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did | +0.624 [+0.567, +0.681] | +0.946 | -0.962 | +0.629 (+0.000) | +0.712 (+0.000) | +0.620 (+0.000) | +0.679 (+0.000) | +0.677, +0.676 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.null_did | +0.498 [+0.439, +0.555] | +0.861 | -0.910 | +0.499 (+0.000) | +0.537 (+0.000) | +0.508 (+0.000) | +0.514 (+0.000) | +0.523, +0.516 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did_relational | +0.615 [+0.561, +0.668] | +0.929 | -0.996 | +0.624 (+0.000) | +0.515 (+0.000) | +0.602 (+0.000) | +0.481 (+0.000) | +0.611, +0.609 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did_relational | +0.622 [+0.569, +0.674] | +0.948 | -0.929 | +0.630 (+0.000) | +0.727 (+0.000) | +0.624 (+0.000) | +0.699 (+0.000) | +0.659, +0.669 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.action | +0.423 [+0.358, +0.484] | +0.887 | -0.936 | +0.425 (+0.000) | +0.457 (+0.000) | +0.420 (+0.000) | +0.440 (+0.000) | +0.625, +0.620 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did | +0.657 [+0.569, +0.739] | +0.962 | -0.994 | +0.669 (+0.000) | +0.576 (+0.000) | +0.652 (+0.000) | +0.554 (+0.000) | +0.730, +0.735 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did | +0.628 [+0.580, +0.675] | +0.946 | -0.992 | +0.632 (+0.000) | +0.716 (+0.000) | +0.620 (+0.000) | +0.683 (+0.000) | +0.624, +0.629 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.null_did | +0.516 [+0.459, +0.563] | +0.868 | +0.254 | +0.515 (+0.000) | +0.552 (+0.000) | +0.514 (+0.000) | +0.532 (+0.000) | +0.516, +0.521 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did_relational | +0.626 [+0.561, +0.689] | +0.929 | -0.996 | +0.638 (+0.000) | +0.526 (+0.000) | +0.632 (+0.000) | +0.492 (+0.000) | +0.676, +0.681 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did_relational | +0.610 [+0.563, +0.656] | +0.948 | -0.990 | +0.618 (+0.000) | +0.715 (+0.000) | +0.610 (+0.000) | +0.685 (+0.000) | +0.580, +0.580 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.action | +0.532 [+0.478, +0.584] | +0.887 | -0.949 | +0.527 (+0.000) | +0.566 (+0.000) | +0.532 (+0.000) | +0.552 (+0.000) | +0.607, +0.614 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did | +0.405 [+0.282, +0.522] | +0.975 | -0.967 | +0.425 (+0.000) | +0.452 (+0.000) | +0.404 (+0.000) | +0.450 (+0.000) | +0.432, +0.436 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did | +0.604 [+0.525, +0.677] | +0.968 | -0.851 | +0.617 (+0.000) | +0.596 (+0.000) | +0.616 (+0.000) | +0.593 (+0.000) | +0.684, +0.689 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.null_did | +0.498 [+0.439, +0.555] | +0.861 | -0.910 | +0.505 (+0.000) | +0.537 (+0.000) | +0.492 (+0.000) | +0.517 (+0.000) | +0.523, +0.527 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did_relational | +0.419 [+0.304, +0.526] | +0.975 | -0.910 | +0.425 (+0.000) | +0.462 (+0.000) | +0.421 (+0.000) | +0.456 (+0.000) | +0.403, +0.400 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did_relational | +0.571 [+0.511, +0.629] | +0.953 | -0.688 | +0.577 (+0.000) | +0.577 (+0.000) | +0.572 (+0.000) | +0.575 (+0.000) | +0.630, +0.639 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.action | +0.423 [+0.358, +0.484] | +0.887 | -0.936 | +0.424 (+0.000) | +0.456 (+0.000) | +0.421 (+0.000) | +0.443 (+0.000) | +0.625, +0.628 (+0.000) |

### Identity swap (A pedestrian relations -> B cone field vs B pedestrian field)

| config | swapped residual | matched residual | swapped - matched [CI] (p) |
|---|---|---|---|
| true: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.492 | +0.525 | -0.034 [-0.093, +0.028] (+0.262) |
| true: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.347 | +0.607 | -0.261 [-0.303, -0.219] (+0.000) |
| true: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.539 | +0.603 | -0.064 [-0.101, -0.027] (+0.001) |
| true: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.346 | +0.574 | -0.228 [-0.266, -0.192] (+0.000) |
| model: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.722 | +0.784 | -0.063 [-0.131, +0.012] (+0.112) |
| model: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.775 | +0.791 | -0.016 [-0.052, +0.020] (+0.398) |
| model: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.707 | +0.774 | -0.066 [-0.137, +0.012] (+0.098) |
| model: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.779 | +0.818 | -0.040 [-0.087, +0.007] (+0.107) |

### Covariate heterogeneity (Spearman; 'degenerate' = < 5 distinct values, stereotyped stimulus)

| arm.field | source | covariate | rho magnitude [CI] | rho directional deviation [CI] |
|---|---|---|---|---|
| A.ped_did | true | hazard_pixels (degenerate) | -0.270 [-0.492, +0.000] | +0.357 [+0.116, +0.569] |
| A.ped_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | true | lane_offset_m | -0.083 [-0.359, +0.208] | -0.284 [-0.534, +0.028] |
| A.ped_did | true | ego_speed_mps | -0.435 [-0.646, -0.196] | +0.140 [-0.133, +0.396] |
| A.ped_did | true | prefix_travel_m | -0.434 [-0.643, -0.181] | +0.138 [-0.138, +0.400] |
| A.ped_did | model | hazard_pixels (degenerate) | -0.551 [-0.721, -0.343] | -0.330 [-0.573, -0.054] |
| A.ped_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | model | lane_offset_m | -0.020 [-0.353, +0.283] | -0.107 [-0.348, +0.137] |
| A.ped_did | model | ego_speed_mps | -0.662 [-0.762, -0.513] | -0.578 [-0.747, -0.311] |
| A.ped_did | model | prefix_travel_m | -0.663 [-0.754, -0.492] | -0.578 [-0.748, -0.330] |
| A.cone_did | true | hazard_pixels (degenerate) | -0.385 [-0.589, -0.141] | -0.162 [-0.420, +0.104] |
| A.cone_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | true | lane_offset_m | -0.060 [-0.338, +0.232] | -0.307 [-0.556, -0.037] |
| A.cone_did | true | ego_speed_mps | -0.762 [-0.856, -0.589] | -0.465 [-0.708, -0.123] |
| A.cone_did | true | prefix_travel_m | -0.763 [-0.862, -0.601] | -0.466 [-0.722, -0.135] |
| A.cone_did | model | hazard_pixels (degenerate) | -0.574 [-0.728, -0.367] | +0.038 [-0.210, +0.299] |
| A.cone_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | model | lane_offset_m | -0.112 [-0.361, +0.146] | -0.192 [-0.445, +0.087] |
| A.cone_did | model | ego_speed_mps | -0.885 [-0.936, -0.780] | -0.250 [-0.545, +0.096] |
| A.cone_did | model | prefix_travel_m | -0.885 [-0.943, -0.775] | -0.251 [-0.513, +0.102] |
| A.null_did | true | hazard_pixels (degenerate) | +0.268 [-0.011, +0.525] | -0.466 [-0.658, -0.217] |
| A.null_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | true | lane_offset_m | +0.119 [-0.135, +0.390] | -0.194 [-0.469, +0.085] |
| A.null_did | true | ego_speed_mps | +0.562 [+0.306, +0.762] | -0.670 [-0.845, -0.418] |
| A.null_did | true | prefix_travel_m | +0.562 [+0.286, +0.754] | -0.669 [-0.844, -0.409] |
| A.null_did | model | hazard_pixels (degenerate) | +0.105 [-0.143, +0.348] | -0.080 [-0.319, +0.159] |
| A.null_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | model | lane_offset_m | -0.206 [-0.433, +0.057] | -0.010 [-0.274, +0.247] |
| A.null_did | model | ego_speed_mps | -0.064 [-0.336, +0.202] | -0.380 [-0.604, -0.108] |
| A.null_did | model | prefix_travel_m | -0.063 [-0.319, +0.222] | -0.383 [-0.596, -0.114] |
| A.ped_did_relational | true | hazard_pixels (degenerate) | -0.072 [-0.345, +0.198] | +0.328 [+0.040, +0.552] |
| A.ped_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | true | lane_offset_m | -0.078 [-0.337, +0.213] | -0.283 [-0.506, +0.001] |
| A.ped_did_relational | true | ego_speed_mps | -0.234 [-0.477, +0.056] | +0.088 [-0.194, +0.367] |
| A.ped_did_relational | true | prefix_travel_m | -0.234 [-0.491, +0.063] | +0.086 [-0.210, +0.357] |
| A.ped_did_relational | model | hazard_pixels (degenerate) | -0.545 [-0.705, -0.335] | -0.418 [-0.646, -0.130] |
| A.ped_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | model | lane_offset_m | -0.021 [-0.342, +0.259] | -0.098 [-0.327, +0.156] |
| A.ped_did_relational | model | ego_speed_mps | -0.658 [-0.757, -0.511] | -0.618 [-0.781, -0.364] |
| A.ped_did_relational | model | prefix_travel_m | -0.658 [-0.748, -0.510] | -0.618 [-0.788, -0.391] |
| A.cone_did_relational | true | hazard_pixels (degenerate) | -0.560 [-0.727, -0.337] | -0.102 [-0.329, +0.120] |
| A.cone_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | true | lane_offset_m | +0.056 [-0.220, +0.344] | -0.189 [-0.465, +0.098] |
| A.cone_did_relational | true | ego_speed_mps | -0.788 [-0.881, -0.620] | -0.330 [-0.610, +0.040] |
| A.cone_did_relational | true | prefix_travel_m | -0.787 [-0.877, -0.631] | -0.331 [-0.630, -0.012] |
| A.cone_did_relational | model | hazard_pixels (degenerate) | -0.590 [-0.739, -0.402] | +0.017 [-0.230, +0.276] |
| A.cone_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | model | lane_offset_m | -0.078 [-0.347, +0.171] | -0.204 [-0.463, +0.088] |
| A.cone_did_relational | model | ego_speed_mps | -0.914 [-0.957, -0.813] | -0.255 [-0.547, +0.096] |
| A.cone_did_relational | model | prefix_travel_m | -0.914 [-0.958, -0.817] | -0.256 [-0.539, +0.143] |
| A.action | true | hazard_pixels (degenerate) | +0.382 [+0.139, +0.587] | -0.278 [-0.496, -0.031] |
| A.action | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | true | lane_offset_m | +0.131 [-0.163, +0.424] | -0.222 [-0.492, +0.061] |
| A.action | true | ego_speed_mps | +0.480 [+0.202, +0.680] | -0.493 [-0.717, -0.188] |
| A.action | true | prefix_travel_m | +0.481 [+0.196, +0.672] | -0.493 [-0.708, -0.199] |
| A.action | model | hazard_pixels (degenerate) | -0.426 [-0.641, -0.206] | -0.304 [-0.536, -0.046] |
| A.action | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | model | lane_offset_m | +0.038 [-0.255, +0.314] | +0.038 [-0.239, +0.298] |
| A.action | model | ego_speed_mps | -0.547 [-0.746, -0.302] | -0.510 [-0.676, -0.305] |
| A.action | model | prefix_travel_m | -0.546 [-0.725, -0.284] | -0.511 [-0.679, -0.278] |
| B.ped_did | true | hazard_pixels (degenerate) | -0.595 [-0.760, -0.399] | +0.046 [-0.203, +0.323] |
| B.ped_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | true | lane_offset_m | -0.135 [-0.390, +0.123] | -0.413 [-0.617, -0.141] |
| B.ped_did | true | ego_speed_mps | -0.948 [-0.977, -0.879] | -0.276 [-0.560, +0.097] |
| B.ped_did | true | prefix_travel_m | -0.948 [-0.979, -0.869] | -0.276 [-0.574, +0.076] |
| B.ped_did | model | hazard_pixels (degenerate) | -0.525 [-0.706, -0.327] | -0.000 [-0.253, +0.240] |
| B.ped_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | model | lane_offset_m | -0.096 [-0.351, +0.172] | -0.109 [-0.366, +0.173] |
| B.ped_did | model | ego_speed_mps | -0.866 [-0.929, -0.726] | -0.259 [-0.523, +0.051] |
| B.ped_did | model | prefix_travel_m | -0.866 [-0.930, -0.754] | -0.262 [-0.531, +0.049] |
| B.cone_did | true | hazard_pixels (degenerate) | -0.023 [-0.294, +0.208] | -0.088 [-0.333, +0.192] |
| B.cone_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | true | lane_offset_m | +0.201 [-0.094, +0.485] | -0.243 [-0.495, +0.056] |
| B.cone_did | true | ego_speed_mps | +0.225 [-0.173, +0.501] | -0.371 [-0.622, -0.036] |
| B.cone_did | true | prefix_travel_m | +0.226 [-0.176, +0.533] | -0.372 [-0.617, -0.030] |
| B.cone_did | model | hazard_pixels (degenerate) | -0.633 [-0.783, -0.427] | -0.594 [-0.751, -0.375] |
| B.cone_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | model | lane_offset_m | +0.029 [-0.261, +0.311] | -0.061 [-0.344, +0.204] |
| B.cone_did | model | ego_speed_mps | -0.662 [-0.811, -0.438] | -0.843 [-0.913, -0.731] |
| B.cone_did | model | prefix_travel_m | -0.663 [-0.818, -0.453] | -0.843 [-0.912, -0.712] |
| B.null_did | true | hazard_pixels (degenerate) | +0.265 [-0.035, +0.534] | -0.466 [-0.652, -0.249] |
| B.null_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | true | lane_offset_m | +0.120 [-0.130, +0.387] | -0.191 [-0.486, +0.101] |
| B.null_did | true | ego_speed_mps | +0.561 [+0.294, +0.769] | -0.669 [-0.856, -0.413] |
| B.null_did | true | prefix_travel_m | +0.561 [+0.306, +0.755] | -0.668 [-0.832, -0.405] |
| B.null_did | model | hazard_pixels (degenerate) | -0.131 [-0.384, +0.132] | -0.493 [-0.697, -0.234] |
| B.null_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | model | lane_offset_m | -0.318 [-0.541, -0.020] | +0.090 [-0.168, +0.334] |
| B.null_did | model | ego_speed_mps | -0.291 [-0.533, +0.028] | -0.628 [-0.821, -0.385] |
| B.null_did | model | prefix_travel_m | -0.290 [-0.559, +0.010] | -0.630 [-0.838, -0.367] |
| B.ped_did_relational | true | hazard_pixels (degenerate) | -0.667 [-0.812, -0.478] | +0.227 [-0.047, +0.454] |
| B.ped_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | true | lane_offset_m | -0.054 [-0.322, +0.205] | -0.280 [-0.547, +0.031] |
| B.ped_did_relational | true | ego_speed_mps | -0.979 [-0.988, -0.953] | -0.047 [-0.396, +0.323] |
| B.ped_did_relational | true | prefix_travel_m | -0.979 [-0.989, -0.949] | -0.047 [-0.380, +0.291] |
| B.ped_did_relational | model | hazard_pixels (degenerate) | -0.528 [-0.702, -0.326] | -0.012 [-0.261, +0.232] |
| B.ped_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | model | lane_offset_m | -0.066 [-0.337, +0.210] | -0.035 [-0.308, +0.218] |
| B.ped_did_relational | model | ego_speed_mps | -0.873 [-0.932, -0.750] | -0.274 [-0.522, +0.021] |
| B.ped_did_relational | model | prefix_travel_m | -0.873 [-0.935, -0.750] | -0.276 [-0.528, +0.033] |
| B.cone_did_relational | true | hazard_pixels (degenerate) | +0.017 [-0.245, +0.275] | -0.165 [-0.390, +0.082] |
| B.cone_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | true | lane_offset_m | +0.245 [-0.057, +0.529] | -0.215 [-0.478, +0.082] |
| B.cone_did_relational | true | ego_speed_mps | +0.303 [-0.076, +0.593] | -0.434 [-0.686, -0.126] |
| B.cone_did_relational | true | prefix_travel_m | +0.303 [-0.071, +0.585] | -0.436 [-0.672, -0.094] |
| B.cone_did_relational | model | hazard_pixels (degenerate) | -0.633 [-0.778, -0.449] | -0.597 [-0.758, -0.384] |
| B.cone_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | model | lane_offset_m | +0.013 [-0.285, +0.283] | -0.050 [-0.341, +0.219] |
| B.cone_did_relational | model | ego_speed_mps | -0.666 [-0.802, -0.443] | -0.830 [-0.906, -0.697] |
| B.cone_did_relational | model | prefix_travel_m | -0.667 [-0.813, -0.448] | -0.830 [-0.906, -0.684] |
| B.action | true | hazard_pixels (degenerate) | +0.382 [+0.175, +0.578] | -0.278 [-0.501, -0.020] |
| B.action | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | true | lane_offset_m | +0.131 [-0.132, +0.388] | -0.222 [-0.459, +0.051] |
| B.action | true | ego_speed_mps | +0.480 [+0.183, +0.688] | -0.493 [-0.721, -0.178] |
| B.action | true | prefix_travel_m | +0.481 [+0.224, +0.689] | -0.493 [-0.709, -0.207] |
| B.action | model | hazard_pixels (degenerate) | -0.624 [-0.770, -0.433] | -0.355 [-0.585, -0.118] |
| B.action | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | model | lane_offset_m | -0.052 [-0.303, +0.223] | -0.034 [-0.336, +0.273] |
| B.action | model | ego_speed_mps | -0.875 [-0.937, -0.783] | -0.497 [-0.627, -0.321] |
| B.action | model | prefix_travel_m | -0.875 [-0.934, -0.780] | -0.498 [-0.634, -0.322] |

## Token group `hazard`
n = 54 discovery scenes; nulls: 100 permutations / 100 draws.

### Field characterisation (true | model)

| arm.field | PC1 true | PC1 model | sph.var true (uniform) | z true | sph.var model | z model | mean-dir LOSO true/model |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.471 | +0.797 | +0.029 (+0.864) | +371.8 | +0.004 | +265.0 | +0.970 / +0.996 |
| A.cone_did | +0.606 | +0.954 | +0.029 (+0.864) | +336.1 | +0.003 | +253.2 | +0.970 / +0.997 |
| A.null_did | +0.375 | +0.789 | +0.163 (+0.864) | +263.8 | +0.109 | +268.2 | +0.831 / +0.887 |
| A.ped_did_relational | +0.479 | +0.778 | +0.028 (+0.864) | +264.3 | +0.004 | +371.6 | +0.971 / +0.996 |
| A.cone_did_relational | +0.571 | +0.977 | +0.035 (+0.863) | +270.8 | +0.002 | +292.0 | +0.964 / +0.998 |
| A.action | +0.403 | +0.525 | +0.110 (+0.865) | +268.7 | +0.020 | +246.6 | +0.886 / +0.980 |
| B.ped_did | +0.756 | +0.918 | +0.039 (+0.863) | +277.8 | +0.002 | +286.0 | +0.960 / +0.998 |
| B.cone_did | +0.547 | +0.579 | +0.050 (+0.864) | +323.3 | +0.008 | +265.5 | +0.948 / +0.992 |
| B.null_did | +0.375 | +0.328 | +0.163 (+0.864) | +229.5 | +0.020 | +339.7 | +0.831 / +0.979 |
| B.ped_did_relational | +0.732 | +0.920 | +0.049 (+0.865) | +277.9 | +0.002 | +261.4 | +0.949 / +0.998 |
| B.cone_did_relational | +0.521 | +0.567 | +0.045 (+0.864) | +286.2 | +0.007 | +262.6 | +0.953 / +0.992 |
| B.action | +0.404 | +0.578 | +0.110 (+0.864) | +277.3 | +0.021 | +223.4 | +0.886 / +0.979 |

### Residual cosine, model vs truth (LOSO true-field mean subtracted)

| arm.field | model raw | model residual | copy-delta raw | copy-delta residual | field-mean raw | model - copy residual (p) | model - copy raw (p) |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.946 | +0.156 | +0.962 | +0.430 | +0.970 | -0.274 (+0.000) | -0.016 (+0.005) |
| A.cone_did | +0.933 | +0.027 | +0.975 | +0.570 | +0.969 | -0.543 (+0.000) | -0.042 (+0.000) |
| A.null_did | +0.454 | -0.038 | +0.842 | +0.484 | +0.829 | -0.522 (+0.000) | -0.389 (+0.000) |
| A.ped_did_relational | +0.940 | +0.114 | +0.963 | +0.411 | +0.971 | -0.296 (+0.000) | -0.023 (+0.000) |
| A.cone_did_relational | +0.876 | -0.068 | +0.964 | +0.501 | +0.963 | -0.569 (+0.000) | -0.089 (+0.000) |
| A.action | +0.583 | -0.018 | +0.893 | +0.532 | +0.886 | -0.550 (+0.000) | -0.310 (+0.000) |
| B.ped_did | +0.897 | -0.141 | +0.968 | +0.606 | +0.959 | -0.748 (+0.000) | -0.071 (+0.000) |
| B.cone_did | +0.853 | -0.022 | +0.946 | +0.507 | +0.948 | -0.529 (+0.000) | -0.093 (+0.000) |
| B.null_did | +0.498 | -0.064 | +0.842 | +0.484 | +0.829 | -0.547 (+0.000) | -0.344 (+0.000) |
| B.ped_did_relational | +0.848 | -0.179 | +0.948 | +0.553 | +0.947 | -0.732 (+0.000) | -0.100 (+0.000) |
| B.cone_did_relational | +0.859 | -0.027 | +0.949 | +0.479 | +0.953 | -0.506 (+0.000) | -0.090 (+0.000) |
| B.action | +0.339 | -0.038 | +0.893 | +0.532 | +0.886 | -0.570 (+0.000) | -0.554 (+0.000) |

### Relational transport (residual cosine; score - null, sign-flip p; 'centred' = LOSO-centred variant vs its permutation null)

| block | config.field | residual [CI] | raw | |mag| rho | - perm (p) | - iso Gauss (p) | - cov Gauss (p) | - rotated (p) | centred residual, - perm (p) |
|---|---|---|---|---|---|---|---|---|---|
| true_cross_arm | true_A->true_B.ped_did | +0.778 [+0.750, +0.805] | +0.959 | -0.667 | +0.772 (+0.000) | +0.664 (+0.000) | +0.775 (+0.000) | +0.632 (+0.000) | +0.792, +0.783 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did | +0.680 [+0.639, +0.722] | +0.949 | -0.121 | +0.689 (+0.000) | +0.744 (+0.000) | +0.684 (+0.000) | +0.721 (+0.000) | +0.690, +0.690 (+0.000) |
| true_cross_arm | true_A->true_B.null_did | +0.600 [+0.544, +0.655] | +0.847 | +0.649 | +0.596 (+0.000) | +0.638 (+0.000) | +0.601 (+0.000) | +0.611 (+0.000) | +0.750, +0.744 (+0.000) |
| true_cross_arm | true_A->true_B.ped_did_relational | +0.721 [+0.688, +0.753] | +0.948 | -0.892 | +0.716 (+0.000) | +0.578 (+0.000) | +0.725 (+0.000) | +0.526 (+0.000) | +0.715, +0.724 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did_relational | +0.590 [+0.524, +0.645] | +0.954 | +0.038 | +0.601 (+0.000) | +0.667 (+0.000) | +0.592 (+0.000) | +0.646 (+0.000) | +0.677, +0.680 (+0.000) |
| true_cross_arm | true_A->true_B.action | +0.615 [+0.570, +0.658] | +0.895 | +0.413 | +0.619 (+0.000) | +0.649 (+0.000) | +0.610 (+0.000) | +0.640 (+0.000) | +0.721, +0.727 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did | +0.523 [+0.473, +0.573] | +0.971 | +0.103 | +0.526 (+0.000) | +0.557 (+0.000) | +0.521 (+0.000) | +0.551 (+0.000) | +0.522, +0.527 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did | +0.648 [+0.566, +0.721] | +0.972 | +0.189 | +0.658 (+0.000) | +0.615 (+0.000) | +0.641 (+0.000) | +0.611 (+0.000) | +0.751, +0.759 (+0.000) |
| true_cross_arm | true_B->true_A.null_did | +0.600 [+0.544, +0.655] | +0.847 | +0.648 | +0.612 (+0.000) | +0.637 (+0.000) | +0.602 (+0.000) | +0.609 (+0.000) | +0.750, +0.746 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did_relational | +0.499 [+0.439, +0.559] | +0.972 | -0.287 | +0.514 (+0.000) | +0.533 (+0.000) | +0.521 (+0.000) | +0.529 (+0.000) | +0.484, +0.487 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did_relational | +0.615 [+0.545, +0.682] | +0.965 | +0.059 | +0.622 (+0.000) | +0.584 (+0.000) | +0.621 (+0.000) | +0.581 (+0.000) | +0.676, +0.696 (+0.000) |
| true_cross_arm | true_B->true_A.action | +0.615 [+0.570, +0.658] | +0.895 | +0.413 | +0.605 (+0.000) | +0.649 (+0.000) | +0.617 (+0.000) | +0.639 (+0.000) | +0.721, +0.720 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did | +0.872 [+0.809, +0.928] | +0.998 | -0.987 | +0.891 (+0.000) | +0.640 (+0.000) | +0.868 (+0.000) | +0.609 (+0.000) | +0.848, +0.857 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did | +0.700 [+0.660, +0.738] | +0.992 | -0.986 | +0.706 (+0.000) | +0.588 (+0.000) | +0.712 (+0.000) | +0.577 (+0.000) | +0.713, +0.722 (+0.000) |
| model_cross_arm | model_A->model_B.null_did | +0.603 [+0.558, +0.646] | +0.982 | +0.121 | +0.597 (+0.000) | +0.591 (+0.000) | +0.603 (+0.000) | +0.578 (+0.000) | +0.586, +0.597 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did_relational | +0.864 [+0.801, +0.921] | +0.998 | -0.993 | +0.879 (+0.000) | +0.633 (+0.000) | +0.869 (+0.000) | +0.604 (+0.000) | +0.861, +0.897 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did_relational | +0.685 [+0.641, +0.729] | +0.992 | -0.994 | +0.691 (+0.000) | +0.573 (+0.000) | +0.680 (+0.000) | +0.562 (+0.000) | +0.699, +0.706 (+0.000) |
| model_cross_arm | model_A->model_B.action | +0.785 [+0.731, +0.831] | +0.979 | -0.610 | +0.784 (+0.000) | +0.744 (+0.000) | +0.782 (+0.000) | +0.739 (+0.000) | +0.831, +0.817 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did | +0.773 [+0.700, +0.841] | +0.996 | -0.964 | +0.801 (+0.000) | +0.735 (+0.000) | +0.772 (+0.000) | +0.734 (+0.000) | +0.808, +0.797 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did | +0.896 [+0.836, +0.942] | +0.997 | -0.992 | +0.890 (+0.000) | +0.665 (+0.000) | +0.906 (+0.000) | +0.638 (+0.000) | +0.910, +0.924 (+0.000) |
| model_cross_arm | model_B->model_A.null_did | +0.658 [+0.550, +0.760] | +0.888 | -0.672 | +0.662 (+0.000) | +0.644 (+0.000) | +0.672 (+0.000) | +0.635 (+0.000) | +0.805, +0.811 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did_relational | +0.786 [+0.723, +0.846] | +0.996 | -0.974 | +0.803 (+0.000) | +0.741 (+0.000) | +0.792 (+0.000) | +0.740 (+0.000) | +0.803, +0.815 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did_relational | +0.918 [+0.855, +0.965] | +0.998 | -0.995 | +0.940 (+0.000) | +0.670 (+0.000) | +0.934 (+0.000) | +0.639 (+0.000) | +0.928, +0.942 (+0.000) |
| model_cross_arm | model_B->model_A.action | +0.656 [+0.602, +0.703] | +0.980 | -0.935 | +0.647 (+0.000) | +0.608 (+0.000) | +0.658 (+0.000) | +0.603 (+0.000) | +0.844, +0.849 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did | +0.438 [+0.364, +0.510] | +0.970 | -0.899 | +0.441 (+0.000) | +0.472 (+0.000) | +0.416 (+0.000) | +0.467 (+0.000) | +0.482, +0.488 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did | +0.652 [+0.590, +0.712] | +0.970 | -0.938 | +0.661 (+0.000) | +0.619 (+0.000) | +0.658 (+0.000) | +0.617 (+0.000) | +0.664, +0.665 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.null_did | +0.547 [+0.502, +0.587] | +0.844 | +0.452 | +0.552 (+0.000) | +0.587 (+0.000) | +0.551 (+0.000) | +0.555 (+0.000) | +0.543, +0.550 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did_relational | +0.420 [+0.361, +0.480] | +0.972 | -0.887 | +0.428 (+0.000) | +0.455 (+0.000) | +0.436 (+0.000) | +0.448 (+0.000) | +0.452, +0.451 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did_relational | +0.597 [+0.545, +0.647] | +0.964 | -0.946 | +0.604 (+0.000) | +0.565 (+0.000) | +0.601 (+0.000) | +0.559 (+0.000) | +0.597, +0.613 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.action | +0.565 [+0.514, +0.614] | +0.887 | -0.498 | +0.570 (+0.000) | +0.601 (+0.000) | +0.559 (+0.000) | +0.590 (+0.000) | +0.614, +0.618 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did | +0.679 [+0.585, +0.767] | +0.959 | -0.997 | +0.683 (+0.000) | +0.566 (+0.000) | +0.676 (+0.000) | +0.528 (+0.000) | +0.711, +0.736 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did | +0.632 [+0.577, +0.690] | +0.948 | -0.965 | +0.632 (+0.000) | +0.697 (+0.000) | +0.622 (+0.000) | +0.672 (+0.000) | +0.700, +0.702 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.null_did | +0.554 [+0.499, +0.607] | +0.831 | -0.612 | +0.555 (+0.000) | +0.599 (+0.000) | +0.558 (+0.000) | +0.564 (+0.000) | +0.663, +0.660 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did_relational | +0.682 [+0.620, +0.745] | +0.947 | -0.997 | +0.705 (+0.000) | +0.540 (+0.000) | +0.688 (+0.000) | +0.491 (+0.000) | +0.683, +0.688 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did_relational | +0.615 [+0.560, +0.669] | +0.953 | -0.966 | +0.624 (+0.000) | +0.692 (+0.000) | +0.628 (+0.000) | +0.670 (+0.000) | +0.663, +0.668 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.action | +0.503 [+0.439, +0.562] | +0.887 | -0.491 | +0.512 (+0.000) | +0.540 (+0.000) | +0.497 (+0.000) | +0.526 (+0.000) | +0.618, +0.623 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did | +0.706 [+0.617, +0.789] | +0.959 | -0.991 | +0.712 (+0.000) | +0.592 (+0.000) | +0.703 (+0.000) | +0.558 (+0.000) | +0.753, +0.758 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did | +0.613 [+0.566, +0.660] | +0.948 | -0.988 | +0.613 (+0.000) | +0.677 (+0.000) | +0.608 (+0.000) | +0.655 (+0.000) | +0.622, +0.622 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.null_did | +0.547 [+0.502, +0.587] | +0.844 | +0.451 | +0.548 (+0.000) | +0.587 (+0.000) | +0.548 (+0.000) | +0.557 (+0.000) | +0.543, +0.543 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did_relational | +0.693 [+0.631, +0.751] | +0.947 | -0.994 | +0.701 (+0.000) | +0.551 (+0.000) | +0.690 (+0.000) | +0.500 (+0.000) | +0.722, +0.722 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did_relational | +0.575 [+0.528, +0.622] | +0.953 | -0.993 | +0.592 (+0.000) | +0.652 (+0.000) | +0.574 (+0.000) | +0.632 (+0.000) | +0.581, +0.586 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.action | +0.565 [+0.514, +0.614] | +0.887 | -0.498 | +0.567 (+0.000) | +0.601 (+0.000) | +0.560 (+0.000) | +0.588 (+0.000) | +0.614, +0.617 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did | +0.486 [+0.398, +0.568] | +0.970 | -0.959 | +0.495 (+0.000) | +0.520 (+0.000) | +0.482 (+0.000) | +0.513 (+0.000) | +0.471, +0.471 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did | +0.657 [+0.580, +0.728] | +0.970 | -0.909 | +0.672 (+0.000) | +0.624 (+0.000) | +0.663 (+0.000) | +0.621 (+0.000) | +0.735, +0.726 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.null_did | +0.554 [+0.499, +0.607] | +0.831 | -0.611 | +0.563 (+0.000) | +0.599 (+0.000) | +0.548 (+0.000) | +0.563 (+0.000) | +0.663, +0.658 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did_relational | +0.481 [+0.401, +0.553] | +0.971 | -0.920 | +0.490 (+0.000) | +0.515 (+0.000) | +0.471 (+0.000) | +0.509 (+0.000) | +0.441, +0.434 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did_relational | +0.621 [+0.562, +0.679] | +0.964 | -0.917 | +0.634 (+0.000) | +0.590 (+0.000) | +0.618 (+0.000) | +0.586 (+0.000) | +0.681, +0.691 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.action | +0.503 [+0.439, +0.562] | +0.887 | -0.491 | +0.506 (+0.000) | +0.539 (+0.000) | +0.505 (+0.000) | +0.525 (+0.000) | +0.618, +0.625 (+0.000) |

### Identity swap (A pedestrian relations -> B cone field vs B pedestrian field)

| config | swapped residual | matched residual | swapped - matched [CI] (p) |
|---|---|---|---|
| true: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.604 | +0.778 | -0.174 [-0.232, -0.117] (+0.000) |
| true: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.421 | +0.648 | -0.227 [-0.269, -0.186] (+0.000) |
| true: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.569 | +0.721 | -0.152 [-0.206, -0.097] (+0.000) |
| true: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.421 | +0.615 | -0.195 [-0.235, -0.155] (+0.000) |
| model: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.709 | +0.872 | -0.163 [-0.201, -0.125] (+0.000) |
| model: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.817 | +0.896 | -0.079 [-0.116, -0.047] (+0.000) |
| model: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.702 | +0.864 | -0.162 [-0.199, -0.123] (+0.000) |
| model: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.826 | +0.918 | -0.092 [-0.132, -0.058] (+0.000) |

### Covariate heterogeneity (Spearman; 'degenerate' = < 5 distinct values, stereotyped stimulus)

| arm.field | source | covariate | rho magnitude [CI] | rho directional deviation [CI] |
|---|---|---|---|---|
| A.ped_did | true | hazard_pixels (degenerate) | -0.428 [-0.618, -0.179] | +0.306 [+0.060, +0.515] |
| A.ped_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | true | lane_offset_m | -0.104 [-0.374, +0.157] | -0.272 [-0.524, -0.003] |
| A.ped_did | true | ego_speed_mps | -0.699 [-0.831, -0.489] | +0.049 [-0.238, +0.326] |
| A.ped_did | true | prefix_travel_m | -0.698 [-0.827, -0.505] | +0.048 [-0.229, +0.317] |
| A.ped_did | model | hazard_pixels (degenerate) | -0.610 [-0.749, -0.438] | -0.206 [-0.467, +0.074] |
| A.ped_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did | model | lane_offset_m | +0.009 [-0.298, +0.297] | -0.199 [-0.427, +0.069] |
| A.ped_did | model | ego_speed_mps | -0.821 [-0.878, -0.729] | -0.479 [-0.705, -0.157] |
| A.ped_did | model | prefix_travel_m | -0.822 [-0.877, -0.732] | -0.477 [-0.713, -0.181] |
| A.cone_did | true | hazard_pixels (degenerate) | -0.424 [-0.616, -0.181] | -0.167 [-0.402, +0.089] |
| A.cone_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | true | lane_offset_m | -0.082 [-0.355, +0.194] | -0.289 [-0.538, -0.019] |
| A.cone_did | true | ego_speed_mps | -0.812 [-0.903, -0.660] | -0.464 [-0.712, -0.125] |
| A.cone_did | true | prefix_travel_m | -0.812 [-0.901, -0.649] | -0.465 [-0.704, -0.153] |
| A.cone_did | model | hazard_pixels (degenerate) | -0.615 [-0.765, -0.422] | +0.121 [-0.155, +0.378] |
| A.cone_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did | model | lane_offset_m | -0.081 [-0.326, +0.183] | -0.182 [-0.449, +0.097] |
| A.cone_did | model | ego_speed_mps | -0.951 [-0.973, -0.895] | -0.144 [-0.443, +0.205] |
| A.cone_did | model | prefix_travel_m | -0.951 [-0.974, -0.894] | -0.145 [-0.463, +0.229] |
| A.null_did | true | hazard_pixels (degenerate) | +0.457 [+0.185, +0.713] | -0.453 [-0.638, -0.212] |
| A.null_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | true | lane_offset_m | +0.084 [-0.179, +0.321] | -0.174 [-0.462, +0.115] |
| A.null_did | true | ego_speed_mps | +0.733 [+0.522, +0.864] | -0.666 [-0.843, -0.410] |
| A.null_did | true | prefix_travel_m | +0.733 [+0.526, +0.867] | -0.666 [-0.838, -0.396] |
| A.null_did | model | hazard_pixels (degenerate) | +0.521 [+0.303, +0.691] | -0.113 [-0.340, +0.144] |
| A.null_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.null_did | model | lane_offset_m | -0.106 [-0.358, +0.163] | -0.078 [-0.354, +0.184] |
| A.null_did | model | ego_speed_mps | +0.460 [+0.240, +0.624] | -0.452 [-0.662, -0.194] |
| A.null_did | model | prefix_travel_m | +0.460 [+0.240, +0.626] | -0.455 [-0.655, -0.193] |
| A.ped_did_relational | true | hazard_pixels (degenerate) | -0.380 [-0.597, -0.140] | +0.324 [+0.069, +0.541] |
| A.ped_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | true | lane_offset_m | -0.091 [-0.346, +0.209] | -0.280 [-0.535, -0.021] |
| A.ped_did_relational | true | ego_speed_mps | -0.638 [-0.788, -0.419] | +0.043 [-0.243, +0.329] |
| A.ped_did_relational | true | prefix_travel_m | -0.638 [-0.782, -0.440] | +0.041 [-0.243, +0.323] |
| A.ped_did_relational | model | hazard_pixels (degenerate) | -0.608 [-0.748, -0.434] | -0.347 [-0.604, -0.052] |
| A.ped_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.ped_did_relational | model | lane_offset_m | -0.003 [-0.327, +0.309] | -0.149 [-0.404, +0.129] |
| A.ped_did_relational | model | ego_speed_mps | -0.807 [-0.868, -0.711] | -0.579 [-0.782, -0.284] |
| A.ped_did_relational | model | prefix_travel_m | -0.808 [-0.864, -0.714] | -0.578 [-0.787, -0.315] |
| A.cone_did_relational | true | hazard_pixels (degenerate) | -0.572 [-0.751, -0.356] | -0.056 [-0.291, +0.186] |
| A.cone_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | true | lane_offset_m | -0.035 [-0.309, +0.234] | -0.183 [-0.465, +0.112] |
| A.cone_did_relational | true | ego_speed_mps | -0.893 [-0.937, -0.815] | -0.296 [-0.619, +0.021] |
| A.cone_did_relational | true | prefix_travel_m | -0.893 [-0.934, -0.806] | -0.297 [-0.584, +0.071] |
| A.cone_did_relational | model | hazard_pixels (degenerate) | -0.615 [-0.761, -0.430] | +0.162 [-0.103, +0.412] |
| A.cone_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.cone_did_relational | model | lane_offset_m | -0.078 [-0.342, +0.163] | -0.180 [-0.449, +0.121] |
| A.cone_did_relational | model | ego_speed_mps | -0.950 [-0.972, -0.889] | -0.090 [-0.401, +0.256] |
| A.cone_did_relational | model | prefix_travel_m | -0.950 [-0.972, -0.893] | -0.091 [-0.440, +0.280] |
| A.action | true | hazard_pixels (degenerate) | +0.389 [+0.160, +0.597] | -0.264 [-0.486, +0.003] |
| A.action | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | true | lane_offset_m | +0.129 [-0.177, +0.390] | -0.228 [-0.467, +0.055] |
| A.action | true | ego_speed_mps | +0.476 [+0.184, +0.686] | -0.499 [-0.712, -0.186] |
| A.action | true | prefix_travel_m | +0.477 [+0.179, +0.685] | -0.499 [-0.719, -0.198] |
| A.action | model | hazard_pixels (degenerate) | -0.400 [-0.611, -0.173] | -0.295 [-0.541, -0.032] |
| A.action | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A.action | model | lane_offset_m | -0.060 [-0.314, +0.209] | +0.016 [-0.256, +0.297] |
| A.action | model | ego_speed_mps | -0.698 [-0.832, -0.513] | -0.553 [-0.740, -0.312] |
| A.action | model | prefix_travel_m | -0.698 [-0.841, -0.488] | -0.553 [-0.727, -0.298] |
| B.ped_did | true | hazard_pixels (degenerate) | -0.583 [-0.747, -0.381] | +0.159 [-0.069, +0.410] |
| B.ped_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | true | lane_offset_m | -0.123 [-0.371, +0.136] | -0.361 [-0.607, -0.090] |
| B.ped_did | true | ego_speed_mps | -0.948 [-0.980, -0.874] | -0.162 [-0.493, +0.235] |
| B.ped_did | true | prefix_travel_m | -0.948 [-0.980, -0.873] | -0.163 [-0.472, +0.226] |
| B.ped_did | model | hazard_pixels (degenerate) | -0.579 [-0.730, -0.380] | +0.108 [-0.175, +0.362] |
| B.ped_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did | model | lane_offset_m | -0.074 [-0.340, +0.203] | -0.175 [-0.411, +0.094] |
| B.ped_did | model | ego_speed_mps | -0.921 [-0.957, -0.839] | -0.162 [-0.452, +0.215] |
| B.ped_did | model | prefix_travel_m | -0.922 [-0.958, -0.839] | -0.163 [-0.453, +0.199] |
| B.cone_did | true | hazard_pixels (degenerate) | -0.041 [-0.315, +0.203] | -0.104 [-0.341, +0.182] |
| B.cone_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | true | lane_offset_m | +0.173 [-0.127, +0.460] | -0.205 [-0.453, +0.087] |
| B.cone_did | true | ego_speed_mps | +0.187 [-0.186, +0.488] | -0.371 [-0.602, -0.031] |
| B.cone_did | true | prefix_travel_m | +0.188 [-0.206, +0.502] | -0.371 [-0.623, -0.055] |
| B.cone_did | model | hazard_pixels (degenerate) | -0.643 [-0.792, -0.437] | -0.585 [-0.753, -0.367] |
| B.cone_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did | model | lane_offset_m | -0.065 [-0.347, +0.216] | -0.053 [-0.312, +0.227] |
| B.cone_did | model | ego_speed_mps | -0.942 [-0.962, -0.892] | -0.824 [-0.905, -0.681] |
| B.cone_did | model | prefix_travel_m | -0.942 [-0.963, -0.895] | -0.824 [-0.907, -0.680] |
| B.null_did | true | hazard_pixels (degenerate) | +0.455 [+0.171, +0.687] | -0.453 [-0.640, -0.249] |
| B.null_did | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | true | lane_offset_m | +0.085 [-0.191, +0.317] | -0.174 [-0.447, +0.116] |
| B.null_did | true | ego_speed_mps | +0.732 [+0.537, +0.871] | -0.666 [-0.850, -0.395] |
| B.null_did | true | prefix_travel_m | +0.732 [+0.507, +0.873] | -0.666 [-0.844, -0.414] |
| B.null_did | model | hazard_pixels (degenerate) | -0.084 [-0.341, +0.198] | -0.485 [-0.678, -0.269] |
| B.null_did | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.null_did | model | lane_offset_m | -0.296 [-0.519, -0.028] | +0.088 [-0.187, +0.320] |
| B.null_did | model | ego_speed_mps | -0.189 [-0.451, +0.124] | -0.603 [-0.802, -0.360] |
| B.null_did | model | prefix_travel_m | -0.189 [-0.448, +0.117] | -0.605 [-0.804, -0.337] |
| B.ped_did_relational | true | hazard_pixels (degenerate) | -0.666 [-0.813, -0.471] | +0.354 [+0.103, +0.559] |
| B.ped_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | true | lane_offset_m | -0.054 [-0.312, +0.223] | -0.277 [-0.532, +0.012] |
| B.ped_did_relational | true | ego_speed_mps | -0.982 [-0.991, -0.954] | +0.119 [-0.211, +0.446] |
| B.ped_did_relational | true | prefix_travel_m | -0.982 [-0.991, -0.957] | +0.119 [-0.215, +0.465] |
| B.ped_did_relational | model | hazard_pixels (degenerate) | -0.590 [-0.749, -0.394] | +0.076 [-0.187, +0.353] |
| B.ped_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.ped_did_relational | model | lane_offset_m | -0.068 [-0.343, +0.188] | -0.083 [-0.341, +0.191] |
| B.ped_did_relational | model | ego_speed_mps | -0.924 [-0.961, -0.841] | -0.194 [-0.475, +0.155] |
| B.ped_did_relational | model | prefix_travel_m | -0.925 [-0.959, -0.841] | -0.196 [-0.472, +0.157] |
| B.cone_did_relational | true | hazard_pixels (degenerate) | -0.048 [-0.298, +0.204] | -0.262 [-0.484, +0.000] |
| B.cone_did_relational | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | true | lane_offset_m | +0.196 [-0.120, +0.468] | -0.131 [-0.405, +0.169] |
| B.cone_did_relational | true | ego_speed_mps | +0.217 [-0.179, +0.519] | -0.498 [-0.717, -0.182] |
| B.cone_did_relational | true | prefix_travel_m | +0.218 [-0.137, +0.542] | -0.499 [-0.712, -0.223] |
| B.cone_did_relational | model | hazard_pixels (degenerate) | -0.647 [-0.790, -0.442] | -0.572 [-0.726, -0.339] |
| B.cone_did_relational | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.cone_did_relational | model | lane_offset_m | -0.077 [-0.367, +0.196] | -0.020 [-0.303, +0.251] |
| B.cone_did_relational | model | ego_speed_mps | -0.941 [-0.961, -0.889] | -0.792 [-0.892, -0.622] |
| B.cone_did_relational | model | prefix_travel_m | -0.941 [-0.961, -0.893] | -0.792 [-0.891, -0.622] |
| B.action | true | hazard_pixels (degenerate) | +0.389 [+0.135, +0.595] | -0.264 [-0.480, -0.010] |
| B.action | true | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | true | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | true | lane_offset_m | +0.129 [-0.192, +0.396] | -0.228 [-0.469, +0.044] |
| B.action | true | ego_speed_mps | +0.476 [+0.170, +0.672] | -0.499 [-0.712, -0.201] |
| B.action | true | prefix_travel_m | +0.477 [+0.200, +0.672] | -0.499 [-0.713, -0.193] |
| B.action | model | hazard_pixels (degenerate) | -0.604 [-0.764, -0.400] | -0.238 [-0.478, +0.002] |
| B.action | model | hazard_patches (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | model | hazard_bbox_area_px (degenerate) | n/a [n/a, n/a] | n/a [n/a, n/a] |
| B.action | model | lane_offset_m | -0.055 [-0.303, +0.213] | -0.059 [-0.374, +0.265] |
| B.action | model | ego_speed_mps | -0.912 [-0.957, -0.819] | -0.506 [-0.673, -0.301] |
| B.action | model | prefix_travel_m | -0.912 [-0.956, -0.811] | -0.506 [-0.669, -0.289] |
