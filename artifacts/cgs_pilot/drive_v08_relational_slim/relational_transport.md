# Relational transport / residual scoring (descriptive)

Scope: descriptive; relational organisation and residual scoring; not localisation.

## Token group `hazard_corridor`
n = 31 discovery scenes; nulls: 100 permutations / 100 draws.

### Field characterisation (true | model)

| arm.field | PC1 true | PC1 model | sph.var true (uniform) | z true | sph.var model | z model | mean-dir LOSO true/model |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.718 | +0.985 | +0.059 (+0.820) | +213.6 | +0.018 | +195.9 | +0.937 / +0.980 |
| A.cone_did | +0.869 | +0.982 | +0.166 (+0.820) | +153.4 | +0.016 | +246.2 | +0.822 / +0.983 |
| A.null_did | +0.735 | +0.973 | +0.208 (+0.820) | +152.5 | +0.138 | +188.2 | +0.776 / +0.852 |
| A.ped_did_relational | +0.714 | +0.984 | +0.060 (+0.820) | +227.9 | +0.018 | +193.4 | +0.936 / +0.981 |
| A.cone_did_relational | +0.894 | +0.990 | +0.234 (+0.819) | +152.2 | +0.015 | +194.8 | +0.748 / +0.984 |
| A.action | +0.463 | +0.897 | +0.146 (+0.820) | +172.8 | +0.054 | +199.1 | +0.843 / +0.942 |
| B.ped_did | +0.938 | +0.989 | +0.337 (+0.821) | +135.2 | +0.045 | +204.4 | +0.636 / +0.952 |
| B.cone_did | +0.433 | +0.917 | +0.118 (+0.821) | +209.9 | +0.023 | +225.6 | +0.874 / +0.975 |
| B.null_did | +0.735 | +0.978 | +0.208 (+0.821) | +153.6 | +0.203 | +143.2 | +0.777 / +0.782 |
| B.ped_did_relational | +0.946 | +0.990 | +0.464 (+0.820) | +98.0 | +0.038 | +190.5 | +0.497 / +0.960 |
| B.cone_did_relational | +0.458 | +0.912 | +0.107 (+0.820) | +199.5 | +0.022 | +180.7 | +0.885 / +0.977 |
| B.action | +0.462 | +0.913 | +0.146 (+0.819) | +196.5 | +0.067 | +185.4 | +0.843 / +0.928 |

### Residual cosine, model vs truth (LOSO true-field mean subtracted)

| arm.field | model raw | model residual | copy-delta raw | copy-delta residual | field-mean raw | model - copy residual (p) | model - copy raw (p) |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.919 | +0.421 | +0.952 | +0.596 | +0.935 | -0.174 (+0.107) | -0.032 (+0.026) |
| A.cone_did | +0.798 | +0.434 | +0.936 | +0.799 | +0.820 | -0.365 (+0.004) | -0.138 (+0.004) |
| A.null_did | +0.452 | +0.278 | +0.869 | +0.741 | +0.745 | -0.462 (+0.000) | -0.417 (+0.000) |
| A.ped_did_relational | +0.916 | +0.395 | +0.952 | +0.596 | +0.934 | -0.201 (+0.064) | -0.036 (+0.016) |
| A.cone_did_relational | +0.704 | +0.356 | +0.920 | +0.808 | +0.747 | -0.452 (+0.001) | -0.216 (+0.001) |
| A.action | +0.538 | +0.134 | +0.878 | +0.583 | +0.842 | -0.448 (+0.000) | -0.340 (+0.000) |
| B.ped_did | +0.676 | +0.720 | +0.903 | +0.839 | +0.633 | -0.119 (+0.159) | -0.227 (+0.007) |
| B.cone_did | +0.778 | +0.184 | +0.911 | +0.612 | +0.874 | -0.429 (+0.000) | -0.133 (+0.000) |
| B.null_did | +0.431 | +0.385 | +0.868 | +0.741 | +0.745 | -0.356 (+0.000) | -0.438 (+0.000) |
| B.ped_did_relational | +0.517 | +0.661 | +0.885 | +0.844 | +0.500 | -0.183 (+0.040) | -0.368 (+0.003) |
| B.cone_did_relational | +0.799 | +0.200 | +0.918 | +0.615 | +0.885 | -0.414 (+0.000) | -0.119 (+0.000) |
| B.action | +0.384 | +0.135 | +0.878 | +0.583 | +0.842 | -0.447 (+0.000) | -0.494 (+0.000) |

### Relational transport (residual cosine; score - null, sign-flip p; 'centred' = LOSO-centred variant vs its permutation null)

| block | config.field | residual [CI] | raw | |mag| rho | - perm (p) | - iso Gauss (p) | - cov Gauss (p) | - rotated (p) | centred residual, - perm (p) |
|---|---|---|---|---|---|---|---|---|---|
| true_cross_arm | true_A->true_B.ped_did | +0.915 [+0.820, +0.967] | +0.670 | +0.778 | +0.909 (+0.000) | +0.976 (+0.000) | +0.927 (+0.000) | +0.928 (+0.000) | +0.900, +0.902 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did | +0.618 [+0.559, +0.676] | +0.909 | -0.414 | +0.620 (+0.000) | +0.602 (+0.000) | +0.622 (+0.000) | +0.593 (+0.000) | +0.621, +0.618 (+0.000) |
| true_cross_arm | true_A->true_B.null_did | +0.871 [+0.850, +0.889] | +0.843 | +0.880 | +0.891 (+0.000) | +0.705 (+0.000) | +0.877 (+0.000) | +0.648 (+0.000) | +0.864, +0.873 (+0.000) |
| true_cross_arm | true_A->true_B.ped_did_relational | +0.945 [+0.899, +0.971] | +0.554 | +0.796 | +0.957 (+0.000) | +1.000 (+0.000) | +0.912 (+0.000) | +0.953 (+0.000) | +0.904, +0.886 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did_relational | +0.635 [+0.580, +0.692] | +0.934 | -0.738 | +0.638 (+0.000) | +0.630 (+0.000) | +0.630 (+0.000) | +0.618 (+0.000) | +0.640, +0.648 (+0.000) |
| true_cross_arm | true_A->true_B.action | +0.711 [+0.680, +0.741] | +0.865 | +0.406 | +0.711 (+0.000) | +0.688 (+0.000) | +0.711 (+0.000) | +0.680 (+0.000) | +0.711, +0.700 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did | +0.651 [+0.535, +0.753] | +0.926 | +0.508 | +0.686 (+0.000) | +0.592 (+0.000) | +0.652 (+0.000) | +0.600 (+0.000) | +0.753, +0.754 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did | +0.930 [+0.920, +0.939] | +0.851 | +0.786 | +0.916 (+0.000) | +0.982 (+0.000) | +0.927 (+0.000) | +0.951 (+0.000) | +0.933, +0.923 (+0.000) |
| true_cross_arm | true_B->true_A.null_did | +0.871 [+0.850, +0.889] | +0.843 | +0.880 | +0.882 (+0.000) | +0.706 (+0.000) | +0.866 (+0.000) | +0.652 (+0.000) | +0.864, +0.878 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did_relational | +0.249 [+0.075, +0.427] | +0.494 | +0.616 | +0.267 (+0.000) | +0.216 (+0.000) | +0.242 (+0.012) | +0.203 (+0.000) | +0.750, +0.770 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did_relational | +0.942 [+0.934, +0.949] | +0.793 | +0.633 | +0.955 (+0.000) | +0.981 (+0.000) | +0.975 (+0.000) | +0.957 (+0.000) | +0.945, +0.930 (+0.000) |
| true_cross_arm | true_B->true_A.action | +0.711 [+0.680, +0.742] | +0.865 | +0.408 | +0.720 (+0.000) | +0.688 (+0.000) | +0.714 (+0.000) | +0.683 (+0.000) | +0.711, +0.699 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did | +0.992 [+0.986, +0.996] | +0.949 | +0.346 | +1.004 (+0.000) | +1.127 (+0.000) | +0.980 (+0.000) | +1.004 (+0.000) | +0.992, +0.974 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did | +0.933 [+0.891, +0.966] | +0.977 | -0.819 | +0.920 (+0.000) | +0.955 (+0.000) | +0.943 (+0.000) | +0.954 (+0.000) | +0.934, +0.928 (+0.000) |
| model_cross_arm | model_A->model_B.null_did | +0.987 [+0.976, +0.993] | +0.846 | +0.685 | +0.956 (+0.000) | +0.781 (+0.000) | +0.978 (+0.000) | +0.697 (+0.000) | +0.987, +0.940 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did_relational | +0.992 [+0.986, +0.996] | +0.956 | -0.138 | +1.000 (+0.000) | +1.133 (+0.000) | +0.991 (+0.000) | +0.993 (+0.000) | +0.992, +0.967 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did_relational | +0.935 [+0.897, +0.965] | +0.978 | -0.985 | +0.942 (+0.000) | +0.938 (+0.000) | +0.922 (+0.000) | +0.936 (+0.000) | +0.936, +0.958 (+0.000) |
| model_cross_arm | model_A->model_B.action | +0.931 [+0.876, +0.973] | +0.941 | -0.198 | +0.913 (+0.000) | +0.928 (+0.000) | +0.926 (+0.000) | +0.923 (+0.000) | +0.937, +0.944 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did | +0.926 [+0.796, +0.994] | +0.983 | +0.622 | +0.939 (+0.000) | +1.027 (+0.000) | +0.890 (+0.000) | +0.988 (+0.000) | +0.990, +0.966 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did | +0.981 [+0.967, +0.992] | +0.984 | +0.554 | +0.992 (+0.000) | +1.067 (+0.000) | +0.977 (+0.000) | +1.036 (+0.000) | +0.981, +0.963 (+0.000) |
| model_cross_arm | model_B->model_A.null_did | +0.984 [+0.967, +0.993] | +0.963 | +0.774 | +0.988 (+0.000) | +0.802 (+0.000) | +0.970 (+0.000) | +0.766 (+0.000) | +0.984, +0.941 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did_relational | +0.925 [+0.795, +0.993] | +0.983 | +0.516 | +0.922 (+0.000) | +1.025 (+0.000) | +0.943 (+0.000) | +0.992 (+0.000) | +0.989, +0.957 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did_relational | +0.988 [+0.978, +0.995] | +0.985 | +0.602 | +0.965 (+0.000) | +1.097 (+0.000) | +0.970 (+0.000) | +1.051 (+0.000) | +0.988, +0.960 (+0.000) |
| model_cross_arm | model_B->model_A.action | +0.912 [+0.848, +0.964] | +0.955 | +0.169 | +0.915 (+0.000) | +0.893 (+0.000) | +0.919 (+0.000) | +0.895 (+0.000) | +0.918, +0.937 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did | +0.751 [+0.646, +0.840] | +0.936 | -0.030 | +0.745 (+0.000) | +0.764 (+0.000) | +0.736 (+0.000) | +0.702 (+0.000) | +0.751, +0.756 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did | +0.918 [+0.906, +0.929] | +0.828 | -0.432 | +0.941 (+0.000) | +0.976 (+0.000) | +0.935 (+0.000) | +0.940 (+0.000) | +0.919, +0.917 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.null_did | +0.857 [+0.837, +0.876] | +0.836 | +0.544 | +0.885 (+0.000) | +0.695 (+0.000) | +0.847 (+0.000) | +0.638 (+0.000) | +0.857, +0.859 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did_relational | +0.748 [+0.644, +0.838] | +0.936 | -0.246 | +0.752 (+0.000) | +0.760 (+0.000) | +0.742 (+0.000) | +0.702 (+0.000) | +0.748, +0.744 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did_relational | +0.932 [+0.923, +0.941] | +0.758 | -0.911 | +0.916 (+0.000) | +0.975 (+0.000) | +0.925 (+0.000) | +0.946 (+0.000) | +0.932, +0.929 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.action | +0.641 [+0.605, +0.678] | +0.856 | +0.282 | +0.641 (+0.000) | +0.618 (+0.000) | +0.642 (+0.000) | +0.613 (+0.000) | +0.647, +0.648 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did | +0.897 [+0.769, +0.964] | +0.676 | +0.725 | +0.878 (+0.000) | +0.954 (+0.000) | +0.913 (+0.000) | +0.907 (+0.000) | +0.960, +0.934 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did | +0.614 [+0.558, +0.670] | +0.879 | -0.991 | +0.618 (+0.000) | +0.600 (+0.000) | +0.605 (+0.000) | +0.586 (+0.000) | +0.619, +0.628 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.null_did | +0.857 [+0.837, +0.876] | +0.881 | +0.697 | +0.859 (+0.000) | +0.671 (+0.000) | +0.836 (+0.000) | +0.638 (+0.000) | +0.857, +0.881 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did_relational | +0.902 [+0.773, +0.969] | +0.546 | +0.528 | +0.891 (+0.000) | +0.958 (+0.000) | +0.861 (+0.000) | +0.909 (+0.000) | +0.965, +0.940 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did_relational | +0.634 [+0.579, +0.691] | +0.890 | -0.990 | +0.618 (+0.000) | +0.629 (+0.000) | +0.631 (+0.000) | +0.616 (+0.000) | +0.637, +0.650 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.action | +0.638 [+0.599, +0.675] | +0.860 | +0.298 | +0.643 (+0.000) | +0.615 (+0.000) | +0.635 (+0.000) | +0.606 (+0.000) | +0.643, +0.651 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did | +0.960 [+0.953, +0.966] | +0.651 | +0.507 | +0.954 (+0.000) | +1.024 (+0.000) | +0.936 (+0.000) | +0.967 (+0.000) | +0.960, +0.923 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did | +0.608 [+0.550, +0.666] | +0.877 | -0.990 | +0.590 (+0.000) | +0.595 (+0.000) | +0.611 (+0.000) | +0.575 (+0.000) | +0.610, +0.619 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.null_did | +0.857 [+0.837, +0.876] | +0.836 | +0.544 | +0.864 (+0.000) | +0.694 (+0.000) | +0.850 (+0.000) | +0.637 (+0.000) | +0.857, +0.856 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did_relational | +0.965 [+0.959, +0.970] | +0.524 | +0.059 | +0.951 (+0.000) | +1.027 (+0.000) | +0.966 (+0.000) | +0.974 (+0.000) | +0.965, +0.967 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did_relational | +0.629 [+0.573, +0.686] | +0.888 | -0.990 | +0.632 (+0.000) | +0.624 (+0.000) | +0.624 (+0.000) | +0.611 (+0.000) | +0.630, +0.641 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.action | +0.641 [+0.605, +0.678] | +0.856 | +0.282 | +0.641 (+0.000) | +0.618 (+0.000) | +0.629 (+0.000) | +0.614 (+0.000) | +0.647, +0.653 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did | +0.689 [+0.536, +0.813] | +0.939 | +0.580 | +0.685 (+0.000) | +0.699 (+0.000) | +0.688 (+0.000) | +0.639 (+0.000) | +0.751, +0.764 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did | +0.919 [+0.908, +0.930] | +0.831 | +0.518 | +0.921 (+0.000) | +0.976 (+0.000) | +0.934 (+0.000) | +0.939 (+0.000) | +0.921, +0.930 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.null_did | +0.857 [+0.838, +0.876] | +0.881 | +0.697 | +0.848 (+0.000) | +0.671 (+0.000) | +0.856 (+0.000) | +0.637 (+0.000) | +0.857, +0.859 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did_relational | +0.686 [+0.533, +0.810] | +0.937 | +0.410 | +0.683 (+0.000) | +0.696 (+0.000) | +0.683 (+0.000) | +0.641 (+0.000) | +0.748, +0.742 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did_relational | +0.933 [+0.923, +0.942] | +0.763 | +0.465 | +0.907 (+0.000) | +0.975 (+0.000) | +0.915 (+0.000) | +0.955 (+0.000) | +0.933, +0.916 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.action | +0.638 [+0.599, +0.675] | +0.860 | +0.298 | +0.642 (+0.000) | +0.615 (+0.000) | +0.631 (+0.000) | +0.608 (+0.000) | +0.643, +0.652 (+0.000) |

### Identity swap (A pedestrian relations -> B cone field vs B pedestrian field)

| config | swapped residual | matched residual | swapped - matched [CI] (p) |
|---|---|---|---|
| true: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.617 | +0.915 | -0.298 [-0.359, -0.251] (+0.000) |
| true: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.784 | +0.930 | -0.145 [-0.256, -0.053] (+0.001) |
| true: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.635 | +0.945 | -0.310 [-0.374, -0.257] (+0.000) |
| true: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.780 | +0.942 | -0.161 [-0.274, -0.071] (+0.000) |
| model: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.933 | +0.992 | -0.059 [-0.098, -0.029] (+0.000) |
| model: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.990 | +0.981 | +0.009 [+0.002, +0.018] (+0.022) |
| model: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.935 | +0.992 | -0.057 [-0.091, -0.030] (+0.000) |
| model: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.989 | +0.988 | +0.002 [-0.001, +0.005] (+0.497) |

### Covariate heterogeneity (Spearman; 'degenerate' = < 5 distinct values, stereotyped stimulus)

| arm.field | source | covariate | rho magnitude [CI] | rho directional deviation [CI] |
|---|---|---|---|---|
| A.ped_did | true | hazard_pixels | +0.664 [+0.387, +0.792] | -0.761 [-0.862, -0.557] |
| A.ped_did | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | true | lane_offset_m | -0.081 [-0.385, +0.262] | +0.171 [-0.228, +0.587] |
| A.ped_did | true | ego_speed_mps | -0.009 [-0.378, +0.410] | -0.300 [-0.630, +0.124] |
| A.ped_did | true | prefix_travel_m | -0.010 [-0.369, +0.452] | -0.301 [-0.657, +0.114] |
| A.ped_did | model | hazard_pixels | +0.781 [+0.605, +0.883] | -0.893 [-0.941, -0.755] |
| A.ped_did | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | model | lane_offset_m | +0.024 [-0.384, +0.390] | +0.111 [-0.175, +0.406] |
| A.ped_did | model | ego_speed_mps | +0.496 [+0.141, +0.757] | -0.265 [-0.518, +0.044] |
| A.ped_did | model | prefix_travel_m | +0.493 [+0.152, +0.740] | -0.264 [-0.527, +0.041] |
| A.cone_did | true | hazard_pixels | +0.758 [+0.517, +0.873] | -0.796 [-0.895, -0.583] |
| A.cone_did | true | hazard_patches (degenerate) | +0.844 [+0.724, +0.868] | -0.844 [-0.869, -0.726] |
| A.cone_did | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.687, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did | true | lane_offset_m | -0.071 [-0.419, +0.285] | +0.046 [-0.336, +0.462] |
| A.cone_did | true | ego_speed_mps | +0.259 [-0.095, +0.551] | -0.273 [-0.650, +0.208] |
| A.cone_did | true | prefix_travel_m | +0.260 [-0.059, +0.571] | -0.275 [-0.646, +0.158] |
| A.cone_did | model | hazard_pixels | +0.834 [+0.647, +0.923] | -0.662 [-0.834, -0.316] |
| A.cone_did | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did | model | lane_offset_m | +0.065 [-0.308, +0.431] | +0.142 [-0.229, +0.510] |
| A.cone_did | model | ego_speed_mps | +0.226 [-0.224, +0.576] | +0.307 [-0.069, +0.631] |
| A.cone_did | model | prefix_travel_m | +0.225 [-0.205, +0.568] | +0.308 [-0.064, +0.628] |
| A.null_did | true | hazard_pixels | -0.706 [-0.838, -0.464] | +0.699 [+0.401, +0.859] |
| A.null_did | true | hazard_patches (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.725, +0.868] |
| A.null_did | true | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| A.null_did | true | lane_offset_m | +0.229 [-0.106, +0.519] | -0.194 [-0.532, +0.169] |
| A.null_did | true | ego_speed_mps | +0.354 [-0.048, +0.712] | -0.143 [-0.529, +0.291] |
| A.null_did | true | prefix_travel_m | +0.355 [-0.054, +0.672] | -0.142 [-0.513, +0.282] |
| A.null_did | model | hazard_pixels | -0.672 [-0.788, -0.403] | +0.624 [+0.353, +0.731] |
| A.null_did | model | hazard_patches (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| A.null_did | model | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| A.null_did | model | lane_offset_m | +0.027 [-0.318, +0.375] | -0.226 [-0.546, +0.183] |
| A.null_did | model | ego_speed_mps | -0.128 [-0.492, +0.239] | +0.089 [-0.301, +0.509] |
| A.null_did | model | prefix_travel_m | -0.124 [-0.486, +0.258] | +0.084 [-0.316, +0.502] |
| A.ped_did_relational | true | hazard_pixels | +0.672 [+0.390, +0.800] | -0.770 [-0.866, -0.549] |
| A.ped_did_relational | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did_relational | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did_relational | true | lane_offset_m | -0.058 [-0.397, +0.267] | +0.203 [-0.212, +0.582] |
| A.ped_did_relational | true | ego_speed_mps | -0.009 [-0.389, +0.416] | -0.290 [-0.634, +0.140] |
| A.ped_did_relational | true | prefix_travel_m | -0.010 [-0.376, +0.450] | -0.290 [-0.627, +0.161] |
| A.ped_did_relational | model | hazard_pixels | +0.783 [+0.614, +0.885] | -0.900 [-0.950, -0.753] |
| A.ped_did_relational | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.725] |
| A.ped_did_relational | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did_relational | model | lane_offset_m | +0.043 [-0.361, +0.427] | +0.092 [-0.200, +0.379] |
| A.ped_did_relational | model | ego_speed_mps | +0.474 [+0.110, +0.712] | -0.275 [-0.537, +0.066] |
| A.ped_did_relational | model | prefix_travel_m | +0.473 [+0.120, +0.740] | -0.275 [-0.531, +0.062] |
| A.cone_did_relational | true | hazard_pixels | +0.801 [+0.630, +0.882] | -0.798 [-0.896, -0.579] |
| A.cone_did_relational | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did_relational | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.724, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did_relational | true | lane_offset_m | -0.080 [-0.362, +0.233] | +0.112 [-0.279, +0.521] |
| A.cone_did_relational | true | ego_speed_mps | +0.072 [-0.305, +0.429] | -0.213 [-0.574, +0.200] |
| A.cone_did_relational | true | prefix_travel_m | +0.072 [-0.314, +0.422] | -0.215 [-0.566, +0.214] |
| A.cone_did_relational | model | hazard_pixels | +0.809 [+0.605, +0.900] | -0.641 [-0.760, -0.357] |
| A.cone_did_relational | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did_relational | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did_relational | model | lane_offset_m | +0.088 [-0.316, +0.469] | +0.217 [-0.166, +0.524] |
| A.cone_did_relational | model | ego_speed_mps | +0.214 [-0.225, +0.552] | +0.324 [-0.085, +0.656] |
| A.cone_did_relational | model | prefix_travel_m | +0.212 [-0.254, +0.575] | +0.319 [-0.099, +0.649] |
| A.action | true | hazard_pixels | -0.654 [-0.814, -0.411] | -0.079 [-0.444, +0.293] |
| A.action | true | hazard_patches (degenerate) | -0.740 [-0.839, -0.541] | -0.030 [-0.364, +0.319] |
| A.action | true | hazard_bbox_area_px (degenerate) | -0.740 [-0.843, -0.560] | -0.030 [-0.373, +0.341] |
| A.action | true | lane_offset_m | -0.042 [-0.380, +0.260] | -0.010 [-0.370, +0.383] |
| A.action | true | ego_speed_mps | +0.271 [-0.100, +0.561] | -0.589 [-0.818, -0.264] |
| A.action | true | prefix_travel_m | +0.269 [-0.099, +0.577] | -0.590 [-0.814, -0.253] |
| A.action | model | hazard_pixels | -0.542 [-0.750, -0.209] | -0.607 [-0.772, -0.232] |
| A.action | model | hazard_patches (degenerate) | -0.733 [-0.868, -0.445] | -0.837 [-0.868, -0.701] |
| A.action | model | hazard_bbox_area_px (degenerate) | -0.733 [-0.867, -0.462] | -0.837 [-0.868, -0.718] |
| A.action | model | lane_offset_m | +0.133 [-0.205, +0.486] | +0.062 [-0.347, +0.488] |
| A.action | model | ego_speed_mps | +0.214 [-0.216, +0.604] | +0.066 [-0.371, +0.454] |
| A.action | model | prefix_travel_m | +0.214 [-0.213, +0.581] | +0.069 [-0.387, +0.471] |
| B.ped_did | true | hazard_pixels | +0.691 [+0.415, +0.821] | -0.794 [-0.891, -0.580] |
| B.ped_did | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did | true | lane_offset_m | -0.225 [-0.507, +0.107] | +0.009 [-0.394, +0.393] |
| B.ped_did | true | ego_speed_mps | -0.140 [-0.534, +0.295] | -0.306 [-0.671, +0.153] |
| B.ped_did | true | prefix_travel_m | -0.140 [-0.542, +0.251] | -0.307 [-0.674, +0.138] |
| B.ped_did | model | hazard_pixels | +0.842 [+0.681, +0.918] | -0.737 [-0.848, -0.501] |
| B.ped_did | model | hazard_patches (degenerate) | +0.844 [+0.725, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did | model | lane_offset_m | -0.100 [-0.408, +0.241] | +0.133 [-0.253, +0.490] |
| B.ped_did | model | ego_speed_mps | +0.239 [-0.107, +0.527] | -0.122 [-0.473, +0.248] |
| B.ped_did | model | prefix_travel_m | +0.241 [-0.089, +0.548] | -0.118 [-0.481, +0.263] |
| B.cone_did | true | hazard_pixels | +0.098 [-0.262, +0.450] | -0.274 [-0.565, +0.074] |
| B.cone_did | true | hazard_patches (degenerate) | +0.081 [-0.304, +0.438] | -0.304 [-0.612, +0.066] |
| B.cone_did | true | hazard_bbox_area_px (degenerate) | +0.081 [-0.271, +0.430] | -0.304 [-0.631, +0.029] |
| B.cone_did | true | lane_offset_m | -0.054 [-0.419, +0.373] | +0.086 [-0.294, +0.447] |
| B.cone_did | true | ego_speed_mps | -0.517 [-0.743, -0.174] | +0.535 [+0.119, +0.801] |
| B.cone_did | true | prefix_travel_m | -0.519 [-0.749, -0.172] | +0.530 [+0.168, +0.823] |
| B.cone_did | model | hazard_pixels | +0.806 [+0.596, +0.883] | -0.823 [-0.898, -0.661] |
| B.cone_did | model | hazard_patches (degenerate) | +0.740 [+0.553, +0.867] | -0.844 [-0.868, -0.726] |
| B.cone_did | model | hazard_bbox_area_px (degenerate) | +0.740 [+0.550, +0.867] | -0.844 [-0.868, -0.726] |
| B.cone_did | model | lane_offset_m | -0.042 [-0.385, +0.327] | +0.090 [-0.303, +0.467] |
| B.cone_did | model | ego_speed_mps | +0.213 [-0.154, +0.521] | -0.047 [-0.368, +0.296] |
| B.cone_did | model | prefix_travel_m | +0.215 [-0.196, +0.515] | -0.046 [-0.363, +0.296] |
| B.null_did | true | hazard_pixels | -0.706 [-0.856, -0.433] | +0.699 [+0.392, +0.860] |
| B.null_did | true | hazard_patches (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| B.null_did | true | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.725] | +0.844 [+0.726, +0.868] |
| B.null_did | true | lane_offset_m | +0.229 [-0.121, +0.504] | -0.202 [-0.540, +0.149] |
| B.null_did | true | ego_speed_mps | +0.354 [-0.052, +0.686] | -0.142 [-0.501, +0.306] |
| B.null_did | true | prefix_travel_m | +0.355 [-0.049, +0.700] | -0.141 [-0.514, +0.248] |
| B.null_did | model | hazard_pixels | -0.670 [-0.789, -0.417] | +0.711 [+0.504, +0.806] |
| B.null_did | model | hazard_patches (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| B.null_did | model | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| B.null_did | model | lane_offset_m | +0.036 [-0.350, +0.442] | -0.218 [-0.548, +0.145] |
| B.null_did | model | ego_speed_mps | -0.206 [-0.554, +0.205] | -0.139 [-0.492, +0.254] |
| B.null_did | model | prefix_travel_m | -0.205 [-0.565, +0.257] | -0.138 [-0.484, +0.234] |
| B.ped_did_relational | true | hazard_pixels | +0.689 [+0.417, +0.815] | -0.827 [-0.914, -0.649] |
| B.ped_did_relational | true | hazard_patches (degenerate) | +0.844 [+0.725, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did_relational | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did_relational | true | lane_offset_m | -0.111 [-0.420, +0.260] | -0.079 [-0.445, +0.345] |
| B.ped_did_relational | true | ego_speed_mps | -0.121 [-0.553, +0.342] | -0.251 [-0.606, +0.179] |
| B.ped_did_relational | true | prefix_travel_m | -0.121 [-0.556, +0.366] | -0.252 [-0.603, +0.166] |
| B.ped_did_relational | model | hazard_pixels | +0.819 [+0.661, +0.910] | -0.789 [-0.876, -0.617] |
| B.ped_did_relational | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.837 [-0.868, -0.718] |
| B.ped_did_relational | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.837 [-0.868, -0.725] |
| B.ped_did_relational | model | lane_offset_m | -0.098 [-0.451, +0.267] | +0.071 [-0.324, +0.420] |
| B.ped_did_relational | model | ego_speed_mps | +0.265 [-0.130, +0.570] | -0.356 [-0.606, -0.022] |
| B.ped_did_relational | model | prefix_travel_m | +0.266 [-0.089, +0.581] | -0.352 [-0.599, -0.057] |
| B.cone_did_relational | true | hazard_pixels | +0.077 [-0.332, +0.446] | -0.147 [-0.544, +0.271] |
| B.cone_did_relational | true | hazard_patches (degenerate) | +0.044 [-0.326, +0.401] | -0.074 [-0.423, +0.312] |
| B.cone_did_relational | true | hazard_bbox_area_px (degenerate) | +0.044 [-0.348, +0.430] | -0.074 [-0.430, +0.325] |
| B.cone_did_relational | true | lane_offset_m | -0.017 [-0.424, +0.415] | +0.065 [-0.316, +0.420] |
| B.cone_did_relational | true | ego_speed_mps | -0.540 [-0.750, -0.207] | +0.610 [+0.311, +0.815] |
| B.cone_did_relational | true | prefix_travel_m | -0.541 [-0.752, -0.245] | +0.608 [+0.292, +0.811] |
| B.cone_did_relational | model | hazard_pixels | +0.573 [+0.237, +0.749] | -0.847 [-0.913, -0.702] |
| B.cone_did_relational | model | hazard_patches (degenerate) | +0.363 [+0.040, +0.668] | -0.844 [-0.868, -0.724] |
| B.cone_did_relational | model | hazard_bbox_area_px (degenerate) | +0.363 [+0.014, +0.637] | -0.844 [-0.868, -0.726] |
| B.cone_did_relational | model | lane_offset_m | +0.018 [-0.325, +0.347] | +0.220 [-0.134, +0.540] |
| B.cone_did_relational | model | ego_speed_mps | +0.333 [-0.105, +0.656] | +0.069 [-0.227, +0.375] |
| B.cone_did_relational | model | prefix_travel_m | +0.337 [-0.069, +0.660] | +0.069 [-0.287, +0.378] |
| B.action | true | hazard_pixels | -0.654 [-0.825, -0.395] | -0.069 [-0.421, +0.299] |
| B.action | true | hazard_patches (degenerate) | -0.740 [-0.843, -0.542] | -0.015 [-0.371, +0.355] |
| B.action | true | hazard_bbox_area_px (degenerate) | -0.740 [-0.842, -0.554] | -0.015 [-0.363, +0.335] |
| B.action | true | lane_offset_m | -0.042 [-0.361, +0.313] | -0.029 [-0.403, +0.317] |
| B.action | true | ego_speed_mps | +0.271 [-0.060, +0.561] | -0.589 [-0.825, -0.267] |
| B.action | true | prefix_travel_m | +0.269 [-0.084, +0.565] | -0.591 [-0.832, -0.280] |
| B.action | model | hazard_pixels | +0.444 [+0.127, +0.686] | -0.682 [-0.842, -0.397] |
| B.action | model | hazard_patches (degenerate) | +0.267 [-0.131, +0.608] | -0.844 [-0.868, -0.726] |
| B.action | model | hazard_bbox_area_px (degenerate) | +0.267 [-0.095, +0.609] | -0.844 [-0.868, -0.726] |
| B.action | model | lane_offset_m | +0.138 [-0.255, +0.440] | +0.188 [-0.210, +0.576] |
| B.action | model | ego_speed_mps | +0.381 [-0.022, +0.720] | +0.105 [-0.280, +0.475] |
| B.action | model | prefix_travel_m | +0.381 [-0.061, +0.699] | +0.108 [-0.295, +0.488] |

## Token group `hazard`
n = 31 discovery scenes; nulls: 100 permutations / 100 draws.

### Field characterisation (true | model)

| arm.field | PC1 true | PC1 model | sph.var true (uniform) | z true | sph.var model | z model | mean-dir LOSO true/model |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.717 | +0.983 | +0.066 (+0.820) | +206.6 | +0.027 | +200.9 | +0.930 / +0.971 |
| A.cone_did | +0.911 | +0.991 | +0.165 (+0.820) | +148.1 | +0.022 | +220.0 | +0.823 / +0.977 |
| A.null_did | +0.731 | +0.976 | +0.231 (+0.820) | +161.3 | +0.167 | +168.4 | +0.751 / +0.821 |
| A.ped_did_relational | +0.716 | +0.981 | +0.067 (+0.820) | +162.0 | +0.024 | +212.2 | +0.929 / +0.974 |
| A.cone_did_relational | +0.923 | +0.996 | +0.218 (+0.820) | +200.4 | +0.017 | +202.4 | +0.765 / +0.982 |
| A.action | +0.495 | +0.861 | +0.146 (+0.820) | +153.2 | +0.065 | +186.6 | +0.843 / +0.930 |
| B.ped_did | +0.955 | +0.994 | +0.372 (+0.820) | +113.3 | +0.050 | +212.3 | +0.598 / +0.947 |
| B.cone_did | +0.451 | +0.920 | +0.113 (+0.821) | +176.7 | +0.026 | +200.5 | +0.879 / +0.972 |
| B.null_did | +0.731 | +0.972 | +0.231 (+0.820) | +129.0 | +0.253 | +153.0 | +0.751 / +0.728 |
| B.ped_did_relational | +0.960 | +0.993 | +0.464 (+0.820) | +94.2 | +0.034 | +187.2 | +0.498 / +0.964 |
| B.cone_did_relational | +0.439 | +0.903 | +0.102 (+0.821) | +170.4 | +0.022 | +224.5 | +0.891 / +0.977 |
| B.action | +0.494 | +0.877 | +0.146 (+0.821) | +172.7 | +0.091 | +260.8 | +0.843 / +0.902 |

### Residual cosine, model vs truth (LOSO true-field mean subtracted)

| arm.field | model raw | model residual | copy-delta raw | copy-delta residual | field-mean raw | model - copy residual (p) | model - copy raw (p) |
|---|---|---|---|---|---|---|---|
| A.ped_did | +0.923 | +0.542 | +0.952 | +0.626 | +0.927 | -0.084 (+0.367) | -0.028 (+0.048) |
| A.cone_did | +0.815 | +0.642 | +0.940 | +0.820 | +0.816 | -0.178 (+0.060) | -0.125 (+0.005) |
| A.null_did | +0.423 | +0.312 | +0.843 | +0.725 | +0.715 | -0.413 (+0.000) | -0.420 (+0.000) |
| A.ped_did_relational | +0.918 | +0.520 | +0.952 | +0.626 | +0.926 | -0.106 (+0.259) | -0.034 (+0.026) |
| A.cone_did_relational | +0.724 | +0.567 | +0.928 | +0.822 | +0.759 | -0.255 (+0.018) | -0.204 (+0.002) |
| A.action | +0.593 | +0.163 | +0.885 | +0.597 | +0.843 | -0.434 (+0.000) | -0.293 (+0.000) |
| B.ped_did | +0.652 | +0.826 | +0.900 | +0.847 | +0.587 | -0.021 (+0.800) | -0.248 (+0.007) |
| B.cone_did | +0.805 | +0.235 | +0.919 | +0.628 | +0.879 | -0.392 (+0.000) | -0.114 (+0.000) |
| B.null_did | +0.376 | +0.392 | +0.843 | +0.725 | +0.715 | -0.334 (+0.001) | -0.468 (+0.000) |
| B.ped_did_relational | +0.516 | +0.784 | +0.890 | +0.849 | +0.493 | -0.065 (+0.456) | -0.373 (+0.004) |
| B.cone_did_relational | +0.818 | +0.218 | +0.923 | +0.616 | +0.891 | -0.398 (+0.000) | -0.105 (+0.000) |
| B.action | +0.412 | +0.159 | +0.885 | +0.597 | +0.843 | -0.439 (+0.000) | -0.473 (+0.000) |

### Relational transport (residual cosine; score - null, sign-flip p; 'centred' = LOSO-centred variant vs its permutation null)

| block | config.field | residual [CI] | raw | |mag| rho | - perm (p) | - iso Gauss (p) | - cov Gauss (p) | - rotated (p) | centred residual, - perm (p) |
|---|---|---|---|---|---|---|---|---|---|
| true_cross_arm | true_A->true_B.ped_did | +0.971 [+0.966, +0.976] | +0.625 | +0.815 | +0.961 (+0.000) | +1.050 (+0.000) | +0.950 (+0.000) | +0.967 (+0.000) | +0.970, +0.951 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did | +0.630 [+0.564, +0.695] | +0.912 | +0.225 | +0.624 (+0.000) | +0.612 (+0.000) | +0.634 (+0.000) | +0.597 (+0.000) | +0.631, +0.631 (+0.000) |
| true_cross_arm | true_A->true_B.null_did | +0.869 [+0.847, +0.889] | +0.834 | +0.863 | +0.862 (+0.000) | +0.690 (+0.000) | +0.874 (+0.000) | +0.642 (+0.000) | +0.863, +0.865 (+0.000) |
| true_cross_arm | true_A->true_B.ped_did_relational | +0.974 [+0.969, +0.979] | +0.540 | +0.828 | +0.970 (+0.000) | +1.049 (+0.000) | +0.983 (+0.000) | +0.968 (+0.000) | +0.973, +0.988 (+0.000) |
| true_cross_arm | true_A->true_B.cone_did_relational | +0.626 [+0.562, +0.689] | +0.933 | -0.176 | +0.624 (+0.000) | +0.611 (+0.000) | +0.614 (+0.000) | +0.601 (+0.000) | +0.628, +0.627 (+0.000) |
| true_cross_arm | true_A->true_B.action | +0.713 [+0.678, +0.747] | +0.870 | +0.253 | +0.729 (+0.000) | +0.694 (+0.000) | +0.699 (+0.000) | +0.686 (+0.000) | +0.719, +0.725 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did | +0.545 [+0.426, +0.660] | +0.794 | +0.517 | +0.570 (+0.000) | +0.492 (+0.000) | +0.550 (+0.000) | +0.499 (+0.000) | +0.776, +0.776 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did | +0.951 [+0.943, +0.958] | +0.843 | +0.616 | +0.945 (+0.000) | +1.027 (+0.000) | +0.948 (+0.000) | +0.966 (+0.000) | +0.952, +0.961 (+0.000) |
| true_cross_arm | true_B->true_A.null_did | +0.869 [+0.847, +0.889] | +0.834 | +0.861 | +0.864 (+0.000) | +0.689 (+0.000) | +0.864 (+0.000) | +0.643 (+0.000) | +0.863, +0.877 (+0.000) |
| true_cross_arm | true_B->true_A.ped_did_relational | +0.286 [+0.119, +0.458] | +0.499 | +0.631 | +0.314 (+0.000) | +0.261 (+0.000) | +0.298 (+0.002) | +0.242 (+0.000) | +0.774, +0.775 (+0.000) |
| true_cross_arm | true_B->true_A.cone_did_relational | +0.956 [+0.950, +0.962] | +0.792 | +0.606 | +0.943 (+0.000) | +1.024 (+0.000) | +0.953 (+0.000) | +0.963 (+0.000) | +0.958, +0.942 (+0.000) |
| true_cross_arm | true_B->true_A.action | +0.713 [+0.678, +0.748] | +0.870 | +0.253 | +0.710 (+0.000) | +0.694 (+0.000) | +0.708 (+0.000) | +0.690 (+0.000) | +0.719, +0.725 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did | +0.996 [+0.993, +0.998] | +0.941 | +0.505 | +0.975 (+0.000) | +1.135 (+0.000) | +0.996 (+0.000) | +0.986 (+0.000) | +0.996, +0.953 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did | +0.920 [+0.863, +0.964] | +0.974 | +0.416 | +0.942 (+0.000) | +0.961 (+0.000) | +0.913 (+0.000) | +0.948 (+0.000) | +0.920, +0.911 (+0.000) |
| model_cross_arm | model_A->model_B.null_did | +0.985 [+0.975, +0.991] | +0.876 | +0.608 | +0.949 (+0.000) | +0.785 (+0.000) | +1.003 (+0.000) | +0.735 (+0.000) | +0.985, +0.947 (+0.000) |
| model_cross_arm | model_A->model_B.ped_did_relational | +0.995 [+0.992, +0.997] | +0.960 | +0.465 | +0.959 (+0.000) | +1.139 (+0.000) | +0.995 (+0.000) | +0.982 (+0.000) | +0.995, +0.976 (+0.000) |
| model_cross_arm | model_A->model_B.cone_did_relational | +0.911 [+0.851, +0.958] | +0.978 | -0.619 | +0.949 (+0.000) | +0.943 (+0.000) | +0.913 (+0.000) | +0.935 (+0.000) | +0.911, +0.917 (+0.000) |
| model_cross_arm | model_A->model_B.action | +0.926 [+0.871, +0.968] | +0.922 | +0.347 | +0.890 (+0.000) | +0.907 (+0.000) | +0.931 (+0.000) | +0.903 (+0.000) | +0.936, +0.941 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did | +0.922 [+0.792, +0.992] | +0.975 | +0.652 | +0.972 (+0.000) | +1.019 (+0.000) | +0.889 (+0.000) | +0.975 (+0.000) | +0.986, +0.969 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did | +0.993 [+0.988, +0.996] | +0.977 | +0.673 | +1.026 (+0.000) | +1.113 (+0.000) | +0.974 (+0.000) | +1.046 (+0.000) | +0.993, +1.013 (+0.000) |
| model_cross_arm | model_B->model_A.null_did | +0.985 [+0.968, +0.994] | +0.986 | +0.733 | +0.950 (+0.000) | +0.779 (+0.000) | +0.974 (+0.000) | +0.769 (+0.000) | +0.985, +0.943 (+0.000) |
| model_cross_arm | model_B->model_A.ped_did_relational | +0.921 [+0.790, +0.991] | +0.976 | +0.541 | +0.930 (+0.000) | +1.019 (+0.000) | +0.922 (+0.000) | +0.974 (+0.000) | +0.985, +0.958 (+0.000) |
| model_cross_arm | model_B->model_A.cone_did_relational | +0.996 [+0.994, +0.998] | +0.982 | +0.612 | +0.994 (+0.000) | +1.129 (+0.000) | +0.992 (+0.000) | +1.044 (+0.000) | +0.996, +0.990 (+0.000) |
| model_cross_arm | model_B->model_A.action | +0.903 [+0.835, +0.959] | +0.951 | +0.307 | +0.907 (+0.000) | +0.886 (+0.000) | +0.929 (+0.000) | +0.884 (+0.000) | +0.909, +0.911 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did | +0.775 [+0.689, +0.849] | +0.930 | +0.499 | +0.780 (+0.000) | +0.794 (+0.000) | +0.795 (+0.000) | +0.728 (+0.000) | +0.776, +0.778 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did | +0.944 [+0.935, +0.952] | +0.825 | +0.491 | +0.971 (+0.000) | +1.028 (+0.000) | +0.939 (+0.000) | +0.956 (+0.000) | +0.944, +0.924 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.null_did | +0.857 [+0.836, +0.876] | +0.838 | +0.497 | +0.846 (+0.000) | +0.675 (+0.000) | +0.868 (+0.000) | +0.630 (+0.000) | +0.857, +0.860 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.ped_did_relational | +0.773 [+0.688, +0.847] | +0.929 | +0.528 | +0.767 (+0.000) | +0.792 (+0.000) | +0.769 (+0.000) | +0.730 (+0.000) | +0.774, +0.775 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.cone_did_relational | +0.950 [+0.942, +0.957] | +0.769 | +0.054 | +0.944 (+0.000) | +1.025 (+0.000) | +0.942 (+0.000) | +0.958 (+0.000) | +0.950, +0.942 (+0.000) |
| model_to_truth_within_arm | model_A->true_A.action | +0.673 [+0.638, +0.707] | +0.861 | -0.131 | +0.670 (+0.000) | +0.652 (+0.000) | +0.679 (+0.000) | +0.650 (+0.000) | +0.679, +0.673 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did | +0.907 [+0.778, +0.974] | +0.626 | +0.673 | +0.919 (+0.000) | +0.984 (+0.000) | +0.943 (+0.000) | +0.900 (+0.000) | +0.970, +0.938 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did | +0.632 [+0.572, +0.692] | +0.884 | -0.665 | +0.642 (+0.000) | +0.617 (+0.000) | +0.641 (+0.000) | +0.597 (+0.000) | +0.642, +0.659 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.null_did | +0.857 [+0.836, +0.877] | +0.893 | +0.599 | +0.884 (+0.000) | +0.642 (+0.000) | +0.864 (+0.000) | +0.632 (+0.000) | +0.857, +0.860 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.ped_did_relational | +0.910 [+0.781, +0.977] | +0.522 | +0.534 | +0.885 (+0.000) | +0.992 (+0.000) | +0.927 (+0.000) | +0.904 (+0.000) | +0.973, +0.926 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.cone_did_relational | +0.631 [+0.572, +0.692] | +0.895 | -0.888 | +0.634 (+0.000) | +0.618 (+0.000) | +0.632 (+0.000) | +0.605 (+0.000) | +0.641, +0.646 (+0.000) |
| model_to_truth_within_arm | model_B->true_B.action | +0.670 [+0.633, +0.706] | +0.869 | +0.145 | +0.677 (+0.000) | +0.650 (+0.000) | +0.680 (+0.000) | +0.643 (+0.000) | +0.674, +0.668 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did | +0.970 [+0.965, +0.976] | +0.609 | +0.562 | +0.959 (+0.000) | +1.055 (+0.000) | +0.968 (+0.000) | +0.968 (+0.000) | +0.970, +0.988 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did | +0.624 [+0.558, +0.688] | +0.883 | -0.650 | +0.630 (+0.000) | +0.609 (+0.000) | +0.612 (+0.000) | +0.587 (+0.000) | +0.625, +0.629 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.null_did | +0.857 [+0.836, +0.876] | +0.838 | +0.498 | +0.850 (+0.000) | +0.674 (+0.000) | +0.845 (+0.000) | +0.633 (+0.000) | +0.857, +0.846 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.ped_did_relational | +0.973 [+0.969, +0.978] | +0.518 | +0.577 | +0.952 (+0.000) | +1.059 (+0.000) | +0.972 (+0.000) | +0.964 (+0.000) | +0.973, +0.929 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.cone_did_relational | +0.621 [+0.558, +0.685] | +0.894 | -0.892 | +0.638 (+0.000) | +0.608 (+0.000) | +0.622 (+0.000) | +0.597 (+0.000) | +0.621, +0.616 (+0.000) |
| model_to_truth_cross_arm | model_A->true_B.action | +0.672 [+0.638, +0.707] | +0.861 | -0.131 | +0.673 (+0.000) | +0.652 (+0.000) | +0.677 (+0.000) | +0.646 (+0.000) | +0.679, +0.679 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did | +0.713 [+0.570, +0.824] | +0.933 | +0.613 | +0.718 (+0.000) | +0.728 (+0.000) | +0.730 (+0.000) | +0.664 (+0.000) | +0.774, +0.782 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did | +0.945 [+0.936, +0.953] | +0.826 | +0.593 | +0.953 (+0.000) | +1.029 (+0.000) | +0.958 (+0.000) | +0.956 (+0.000) | +0.946, +0.950 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.null_did | +0.858 [+0.836, +0.877] | +0.893 | +0.601 | +0.856 (+0.000) | +0.641 (+0.000) | +0.842 (+0.000) | +0.632 (+0.000) | +0.857, +0.875 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.ped_did_relational | +0.711 [+0.566, +0.822] | +0.930 | +0.454 | +0.727 (+0.000) | +0.728 (+0.000) | +0.710 (+0.000) | +0.669 (+0.000) | +0.773, +0.768 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.cone_did_relational | +0.951 [+0.943, +0.958] | +0.770 | +0.482 | +0.958 (+0.000) | +1.025 (+0.000) | +0.954 (+0.000) | +0.958 (+0.000) | +0.952, +0.934 (+0.000) |
| model_to_truth_cross_arm | model_B->true_A.action | +0.670 [+0.633, +0.706] | +0.869 | +0.145 | +0.666 (+0.000) | +0.650 (+0.000) | +0.668 (+0.000) | +0.646 (+0.000) | +0.674, +0.668 (+0.000) |

### Identity swap (A pedestrian relations -> B cone field vs B pedestrian field)

| config | swapped residual | matched residual | swapped - matched [CI] (p) |
|---|---|---|---|
| true: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.647 | +0.971 | -0.324 [-0.395, -0.265] (+0.000) |
| true: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.808 | +0.951 | -0.143 [-0.235, -0.068] (+0.000) |
| true: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.643 | +0.974 | -0.331 [-0.400, -0.272] (+0.000) |
| true: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.811 | +0.956 | -0.145 [-0.241, -0.071] (+0.000) |
| model: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched) | +0.920 | +0.996 | -0.075 [-0.130, -0.033] (+0.000) |
| model: B.cone_did -> A.ped_did (swapped) vs A.cone_did (matched) | +0.987 | +0.993 | -0.006 [-0.011, -0.002] (+0.012) |
| model: A.ped_did_relational -> B.cone_did_relational (swapped) vs B.ped_did_relational (matched) | +0.912 | +0.995 | -0.083 [-0.141, -0.039] (+0.000) |
| model: B.cone_did_relational -> A.ped_did_relational (swapped) vs A.cone_did_relational (matched) | +0.986 | +0.996 | -0.010 [-0.018, -0.004] (+0.000) |

### Covariate heterogeneity (Spearman; 'degenerate' = < 5 distinct values, stereotyped stimulus)

| arm.field | source | covariate | rho magnitude [CI] | rho directional deviation [CI] |
|---|---|---|---|---|
| A.ped_did | true | hazard_pixels | +0.671 [+0.430, +0.787] | -0.788 [-0.891, -0.568] |
| A.ped_did | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | true | lane_offset_m | -0.084 [-0.373, +0.254] | +0.158 [-0.258, +0.525] |
| A.ped_did | true | ego_speed_mps | -0.011 [-0.374, +0.416] | -0.261 [-0.617, +0.156] |
| A.ped_did | true | prefix_travel_m | -0.012 [-0.392, +0.404] | -0.262 [-0.600, +0.130] |
| A.ped_did | model | hazard_pixels | +0.786 [+0.598, +0.883] | -0.873 [-0.927, -0.758] |
| A.ped_did | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did | model | lane_offset_m | +0.038 [-0.338, +0.402] | +0.154 [-0.154, +0.437] |
| A.ped_did | model | ego_speed_mps | +0.499 [+0.145, +0.758] | -0.198 [-0.471, +0.122] |
| A.ped_did | model | prefix_travel_m | +0.500 [+0.159, +0.761] | -0.196 [-0.459, +0.112] |
| A.cone_did | true | hazard_pixels | +0.830 [+0.675, +0.899] | -0.825 [-0.913, -0.643] |
| A.cone_did | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did | true | lane_offset_m | +0.032 [-0.247, +0.303] | -0.002 [-0.405, +0.398] |
| A.cone_did | true | ego_speed_mps | +0.037 [-0.324, +0.389] | -0.266 [-0.615, +0.166] |
| A.cone_did | true | prefix_travel_m | +0.038 [-0.349, +0.353] | -0.268 [-0.638, +0.177] |
| A.cone_did | model | hazard_pixels | +0.836 [+0.657, +0.914] | -0.660 [-0.831, -0.312] |
| A.cone_did | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did | model | lane_offset_m | +0.000 [-0.393, +0.374] | +0.177 [-0.192, +0.530] |
| A.cone_did | model | ego_speed_mps | +0.249 [-0.172, +0.604] | +0.333 [-0.049, +0.630] |
| A.cone_did | model | prefix_travel_m | +0.250 [-0.201, +0.603] | +0.335 [-0.064, +0.631] |
| A.null_did | true | hazard_pixels | -0.698 [-0.823, -0.459] | +0.706 [+0.400, +0.856] |
| A.null_did | true | hazard_patches (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| A.null_did | true | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| A.null_did | true | lane_offset_m | +0.183 [-0.167, +0.477] | -0.155 [-0.495, +0.237] |
| A.null_did | true | ego_speed_mps | +0.305 [-0.077, +0.660] | -0.083 [-0.464, +0.335] |
| A.null_did | true | prefix_travel_m | +0.306 [-0.102, +0.617] | -0.082 [-0.437, +0.325] |
| A.null_did | model | hazard_pixels | -0.651 [-0.781, -0.388] | +0.763 [+0.581, +0.840] |
| A.null_did | model | hazard_patches (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| A.null_did | model | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| A.null_did | model | lane_offset_m | +0.021 [-0.303, +0.389] | -0.172 [-0.494, +0.179] |
| A.null_did | model | ego_speed_mps | -0.055 [-0.451, +0.312] | +0.170 [-0.186, +0.505] |
| A.null_did | model | prefix_travel_m | -0.051 [-0.417, +0.328] | +0.168 [-0.199, +0.506] |
| A.ped_did_relational | true | hazard_pixels | +0.683 [+0.436, +0.811] | -0.778 [-0.890, -0.564] |
| A.ped_did_relational | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did_relational | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did_relational | true | lane_offset_m | -0.061 [-0.365, +0.285] | +0.175 [-0.235, +0.552] |
| A.ped_did_relational | true | ego_speed_mps | -0.012 [-0.406, +0.426] | -0.254 [-0.603, +0.175] |
| A.ped_did_relational | true | prefix_travel_m | -0.013 [-0.361, +0.426] | -0.254 [-0.588, +0.185] |
| A.ped_did_relational | model | hazard_pixels | +0.792 [+0.614, +0.886] | -0.865 [-0.922, -0.749] |
| A.ped_did_relational | model | hazard_patches (degenerate) | +0.844 [+0.725, +0.868] | -0.844 [-0.868, -0.726] |
| A.ped_did_relational | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.724] |
| A.ped_did_relational | model | lane_offset_m | +0.053 [-0.312, +0.424] | +0.163 [-0.149, +0.471] |
| A.ped_did_relational | model | ego_speed_mps | +0.496 [+0.136, +0.760] | -0.202 [-0.469, +0.134] |
| A.ped_did_relational | model | prefix_travel_m | +0.497 [+0.151, +0.761] | -0.200 [-0.474, +0.097] |
| A.cone_did_relational | true | hazard_pixels | +0.790 [+0.643, +0.867] | -0.816 [-0.906, -0.626] |
| A.cone_did_relational | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.727] |
| A.cone_did_relational | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did_relational | true | lane_offset_m | -0.004 [-0.293, +0.280] | +0.091 [-0.281, +0.475] |
| A.cone_did_relational | true | ego_speed_mps | -0.010 [-0.384, +0.347] | -0.199 [-0.553, +0.227] |
| A.cone_did_relational | true | prefix_travel_m | -0.010 [-0.358, +0.374] | -0.202 [-0.541, +0.239] |
| A.cone_did_relational | model | hazard_pixels | +0.838 [+0.667, +0.919] | -0.660 [-0.819, -0.306] |
| A.cone_did_relational | model | hazard_patches (degenerate) | +0.844 [+0.687, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did_relational | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| A.cone_did_relational | model | lane_offset_m | +0.014 [-0.419, +0.379] | +0.183 [-0.200, +0.508] |
| A.cone_did_relational | model | ego_speed_mps | +0.252 [-0.177, +0.606] | +0.350 [-0.033, +0.700] |
| A.cone_did_relational | model | prefix_travel_m | +0.252 [-0.174, +0.606] | +0.346 [-0.041, +0.678] |
| A.action | true | hazard_pixels | -0.544 [-0.747, -0.218] | -0.229 [-0.559, +0.135] |
| A.action | true | hazard_patches (degenerate) | -0.629 [-0.799, -0.386] | -0.215 [-0.526, +0.121] |
| A.action | true | hazard_bbox_area_px (degenerate) | -0.629 [-0.793, -0.380] | -0.215 [-0.512, +0.136] |
| A.action | true | lane_offset_m | -0.049 [-0.401, +0.312] | +0.033 [-0.345, +0.425] |
| A.action | true | ego_speed_mps | +0.384 [+0.003, +0.652] | -0.535 [-0.766, -0.186] |
| A.action | true | prefix_travel_m | +0.383 [+0.042, +0.671] | -0.536 [-0.780, -0.178] |
| A.action | model | hazard_pixels | -0.500 [-0.727, -0.126] | -0.724 [-0.836, -0.496] |
| A.action | model | hazard_patches (degenerate) | -0.711 [-0.867, -0.379] | -0.837 [-0.868, -0.726] |
| A.action | model | hazard_bbox_area_px (degenerate) | -0.711 [-0.868, -0.397] | -0.837 [-0.868, -0.708] |
| A.action | model | lane_offset_m | +0.155 [-0.214, +0.478] | -0.044 [-0.402, +0.353] |
| A.action | model | ego_speed_mps | +0.198 [-0.273, +0.572] | -0.406 [-0.663, -0.081] |
| A.action | model | prefix_travel_m | +0.198 [-0.304, +0.578] | -0.404 [-0.679, -0.053] |
| B.ped_did | true | hazard_pixels | +0.703 [+0.443, +0.849] | -0.815 [-0.913, -0.595] |
| B.ped_did | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.687] |
| B.ped_did | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did | true | lane_offset_m | -0.204 [-0.515, +0.138] | -0.010 [-0.400, +0.412] |
| B.ped_did | true | ego_speed_mps | -0.191 [-0.620, +0.260] | -0.304 [-0.639, +0.128] |
| B.ped_did | true | prefix_travel_m | -0.192 [-0.622, +0.270] | -0.304 [-0.665, +0.185] |
| B.ped_did | model | hazard_pixels | +0.849 [+0.694, +0.928] | -0.718 [-0.831, -0.473] |
| B.ped_did | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.725, +0.868] | -0.844 [-0.868, -0.725] |
| B.ped_did | model | lane_offset_m | -0.061 [-0.404, +0.253] | +0.180 [-0.209, +0.548] |
| B.ped_did | model | ego_speed_mps | +0.245 [-0.109, +0.564] | -0.109 [-0.463, +0.263] |
| B.ped_did | model | prefix_travel_m | +0.246 [-0.096, +0.532] | -0.107 [-0.441, +0.258] |
| B.cone_did | true | hazard_pixels | +0.633 [+0.400, +0.795] | -0.437 [-0.688, -0.096] |
| B.cone_did | true | hazard_patches (degenerate) | +0.666 [+0.421, +0.819] | -0.459 [-0.711, -0.167] |
| B.cone_did | true | hazard_bbox_area_px (degenerate) | +0.666 [+0.415, +0.821] | -0.459 [-0.699, -0.133] |
| B.cone_did | true | lane_offset_m | -0.133 [-0.405, +0.177] | +0.146 [-0.177, +0.489] |
| B.cone_did | true | ego_speed_mps | -0.107 [-0.467, +0.286] | +0.412 [+0.003, +0.745] |
| B.cone_did | true | prefix_travel_m | -0.108 [-0.512, +0.281] | +0.409 [+0.016, +0.755] |
| B.cone_did | model | hazard_pixels | +0.881 [+0.738, +0.937] | -0.702 [-0.860, -0.396] |
| B.cone_did | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.cone_did | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.725] |
| B.cone_did | model | lane_offset_m | -0.080 [-0.397, +0.269] | +0.184 [-0.215, +0.535] |
| B.cone_did | model | ego_speed_mps | +0.142 [-0.249, +0.477] | +0.203 [-0.183, +0.555] |
| B.cone_did | model | prefix_travel_m | +0.143 [-0.246, +0.480] | +0.205 [-0.186, +0.535] |
| B.null_did | true | hazard_pixels | -0.700 [-0.839, -0.433] | +0.706 [+0.422, +0.857] |
| B.null_did | true | hazard_patches (degenerate) | -0.844 [-0.868, -0.688] | +0.844 [+0.726, +0.868] |
| B.null_did | true | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.687] | +0.844 [+0.726, +0.868] |
| B.null_did | true | lane_offset_m | +0.180 [-0.161, +0.459] | -0.155 [-0.479, +0.218] |
| B.null_did | true | ego_speed_mps | +0.319 [-0.133, +0.644] | -0.083 [-0.481, +0.305] |
| B.null_did | true | prefix_travel_m | +0.319 [-0.073, +0.649] | -0.082 [-0.459, +0.334] |
| B.null_did | model | hazard_pixels | -0.673 [-0.795, -0.418] | +0.732 [+0.526, +0.831] |
| B.null_did | model | hazard_patches (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| B.null_did | model | hazard_bbox_area_px (degenerate) | -0.844 [-0.868, -0.726] | +0.844 [+0.726, +0.868] |
| B.null_did | model | lane_offset_m | -0.010 [-0.396, +0.421] | -0.235 [-0.525, +0.089] |
| B.null_did | model | ego_speed_mps | -0.290 [-0.630, +0.134] | -0.171 [-0.488, +0.231] |
| B.null_did | model | prefix_travel_m | -0.290 [-0.616, +0.108] | -0.172 [-0.490, +0.192] |
| B.ped_did_relational | true | hazard_pixels | +0.691 [+0.429, +0.835] | -0.828 [-0.916, -0.649] |
| B.ped_did_relational | true | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did_relational | true | hazard_bbox_area_px (degenerate) | +0.844 [+0.725, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did_relational | true | lane_offset_m | -0.128 [-0.456, +0.201] | -0.093 [-0.485, +0.327] |
| B.ped_did_relational | true | ego_speed_mps | -0.164 [-0.560, +0.269] | -0.237 [-0.573, +0.180] |
| B.ped_did_relational | true | prefix_travel_m | -0.165 [-0.574, +0.306] | -0.238 [-0.598, +0.260] |
| B.ped_did_relational | model | hazard_pixels | +0.854 [+0.699, +0.929] | -0.784 [-0.864, -0.599] |
| B.ped_did_relational | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.725] |
| B.ped_did_relational | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.ped_did_relational | model | lane_offset_m | -0.090 [-0.428, +0.254] | +0.109 [-0.287, +0.477] |
| B.ped_did_relational | model | ego_speed_mps | +0.211 [-0.152, +0.529] | -0.295 [-0.567, +0.036] |
| B.ped_did_relational | model | prefix_travel_m | +0.213 [-0.139, +0.526] | -0.293 [-0.548, +0.021] |
| B.cone_did_relational | true | hazard_pixels | +0.326 [-0.015, +0.635] | -0.262 [-0.575, +0.124] |
| B.cone_did_relational | true | hazard_patches (degenerate) | +0.348 [+0.000, +0.615] | -0.200 [-0.541, +0.159] |
| B.cone_did_relational | true | hazard_bbox_area_px (degenerate) | +0.348 [+0.024, +0.646] | -0.200 [-0.559, +0.139] |
| B.cone_did_relational | true | lane_offset_m | -0.064 [-0.366, +0.262] | +0.157 [-0.202, +0.490] |
| B.cone_did_relational | true | ego_speed_mps | -0.232 [-0.563, +0.178] | +0.549 [+0.152, +0.799] |
| B.cone_did_relational | true | prefix_travel_m | -0.234 [-0.631, +0.172] | +0.545 [+0.196, +0.790] |
| B.cone_did_relational | model | hazard_pixels | +0.877 [+0.730, +0.931] | -0.802 [-0.888, -0.611] |
| B.cone_did_relational | model | hazard_patches (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.cone_did_relational | model | hazard_bbox_area_px (degenerate) | +0.844 [+0.726, +0.868] | -0.844 [-0.868, -0.726] |
| B.cone_did_relational | model | lane_offset_m | -0.066 [-0.378, +0.269] | +0.202 [-0.179, +0.516] |
| B.cone_did_relational | model | ego_speed_mps | +0.163 [-0.195, +0.492] | +0.117 [-0.208, +0.405] |
| B.cone_did_relational | model | prefix_travel_m | +0.164 [-0.184, +0.523] | +0.116 [-0.195, +0.425] |
| B.action | true | hazard_pixels | -0.544 [-0.748, -0.244] | -0.229 [-0.546, +0.186] |
| B.action | true | hazard_patches (degenerate) | -0.629 [-0.801, -0.390] | -0.215 [-0.513, +0.139] |
| B.action | true | hazard_bbox_area_px (degenerate) | -0.629 [-0.798, -0.363] | -0.215 [-0.556, +0.132] |
| B.action | true | lane_offset_m | -0.049 [-0.383, +0.281] | +0.033 [-0.346, +0.420] |
| B.action | true | ego_speed_mps | +0.384 [+0.029, +0.662] | -0.535 [-0.785, -0.193] |
| B.action | true | prefix_travel_m | +0.383 [+0.027, +0.653] | -0.536 [-0.783, -0.203] |
| B.action | model | hazard_pixels | -0.523 [-0.726, -0.134] | -0.732 [-0.834, -0.552] |
| B.action | model | hazard_patches (degenerate) | -0.763 [-0.867, -0.536] | -0.807 [-0.867, -0.639] |
| B.action | model | hazard_bbox_area_px (degenerate) | -0.763 [-0.867, -0.519] | -0.807 [-0.868, -0.661] |
| B.action | model | lane_offset_m | +0.190 [-0.179, +0.519] | -0.029 [-0.380, +0.358] |
| B.action | model | ego_speed_mps | +0.143 [-0.322, +0.526] | -0.417 [-0.659, -0.099] |
| B.action | model | prefix_travel_m | +0.144 [-0.278, +0.522] | -0.412 [-0.652, -0.122] |
