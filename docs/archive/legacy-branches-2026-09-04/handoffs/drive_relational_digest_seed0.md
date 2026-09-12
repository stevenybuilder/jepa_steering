# Digest: artifacts/cgs_pilot/drive_relational/relational_transport.json

Source: `artifacts/cgs_pilot/drive_relational/relational_transport.json` (5.3 MB) + sibling `relational_transport.md` (rendered tables, 507 lines; identical numbers). `armA_seed0/`, `armB_seed0/` are EMPTY (slim caches deleted per run_relational_transport.sh). Producer: `scripts/cgs_pilot/relational_transport.py` (LRH.md sec. 2, descriptive). Key paths below are relative to the JSON root; `G` = `groups/<site>`, `S` = `summary/<site>` (summary mirrors groups, fewer fields).

## 1. Coverage
- Models: JEPA-WM driving, `jepa_wm_driving`, arm A seed 0 and arm B seed 0, checkpoint `jepa-latest.pth.tar` (`arms/A|B/cache_meta/label`, `/checkpoint`, `/model_name`; sha256s present). One seed only. D = 1024 (`G/field_characterisation/*/*/*/dim`). torch 2.7.1+cu128.
- Scenes: n = 54 discovery scenes per arm, 432 cells each (8 cells/scene: hazard levels 0,1,2,3 x action 0/1) (`arms/A/n_scenes`, `/n_cells`, `/levels_present`). Same scene list in both arms (`G/scenes`, drive_seed000003..000119).
- Sites/token groups: `hazard_corridor` (gate's primary) and `hazard` (`definitions/token_groups`); token source = masks in both arms (`G/token_source`). Tokens/scene: corridor 76-99 (median 96.5), hazard 47-74 (median 69.5) (`G/n_tokens_per_scene`). Frame: last predicted frame only (`definitions/frame`).
- Fields (`definitions/fields`): ped_did, cone_did, null_did (matched-displacement H0'), ped_did_relational (= ped-null), cone_did_relational, action. Identities = pedestrian (level 1) and cone (level 3). Each computed from `prediction` (MODEL) and `target` (TRUE).
- Nulls: 100 permutations / 100 Gaussian draws / 50 MC for the uniform-sphere reference (`n_perm`, `n_draws`, `n_mc_spherical`); `seed` = 0; `elapsed_s` = 90.5.
- Verdict/interpretation strings in file: `interpretation_scope` = "descriptive; relational organisation and residual scoring; not localisation". No pass/fail verdict anywhere. Notes: `G/residual_cosine/*/*/note` = "field_mean residual cosine is 0 by construction"; `.../per_token_raw_cosine/note` = "not cross-scene comparable"; `definitions/representation` = per-scene token-mean D-vector.

## 2. Field characterisation (`S/<site>/field/<arm>.<field>/<true|model>/{pc1,spherical_variance,uniform_spherical_variance,z_vs_uniform}`; full at `G/field_characterisation/<arm>/<field>/<kind>/{pc1_variance_fraction,top5_variance_fraction,participation_ratio,spherical_variance,uniform_random_same_dim/{spherical_variance_mean,sd,z,p_mc_ge_observed}}`)
Uniform-random reference (same D=1024): spherical variance 0.863-0.865 (sd ~0.003) for every entry; all observed z = 217-375, p_mc = 0.02 (= 1/51, floor).

| site | arm.field | PC1 true | PC1 model | sph.var true | sph.var model | PR true |
|---|---|---|---|---|---|---|
| corridor | A.ped_did | 0.564 | 0.740 | 0.024 | 0.002 | 2.8 |
| corridor | A.cone_did | 0.541 | 0.894 | 0.031 | 0.002 | 3.1 |
| corridor | B.ped_did | 0.667 | 0.757 | 0.036 | 0.001 | 2.2 |
| corridor | B.cone_did | 0.565 | 0.574 | 0.052 | 0.005 | 3.0 |
| corridor | A/B.null_did | 0.346 | 0.775 / 0.557 | 0.134 | 0.074 / 0.007 | 5.5 |
| corridor | A/B.action | 0.365 | 0.437 / 0.470 | 0.109 | 0.010 / 0.009 | 5.8 |
| hazard | A.ped_did | 0.471 | 0.797 | 0.029 | 0.004 | 3.2 |
| hazard | A.cone_did | 0.606 | 0.954 | 0.029 | 0.003 | 2.6 |
| hazard | B.ped_did | 0.756 | 0.918 | 0.039 | 0.002 | 1.7 |
| hazard | B.cone_did | 0.547 | 0.579 | 0.050 | 0.008 | 3.1 |
Relational variants (ped/cone minus null) are within 0.01-0.05 of the above. Top-5 PCs explain 0.79-0.90 of TRUE consequence fields, 0.95-0.99 of MODEL fields. Mean-direction LOSO consistency: true 0.93-0.98, model 0.99-1.00 (`.../mean_direction_consistency/loso`).
Reading: TRUE DiD field is dominated by one direction (PC1 0.47-0.76, sph.var 0.02-0.05 vs 0.86 random, PR 2-3) but is not a single direction; MODEL fields are even more collapsed (PC1 up to 0.95, sph.var <=0.008).

## 3. Raw vs residual cosine, model vs truth (`S/<site>/residual_cosine/<arm>.<field>/{model,copy_delta,field_mean}/{raw,residual}`; CIs at `G/residual_cosine/<arm>/<field>/pooled/<name>/<raw|residual>/{mean,ci_low,ci_high}`; diffs at `.../model_minus_baseline/copy_delta/<residual|raw>/{mean,sign_flip_p}`)

| site | arm.field | model raw [CI] | model residual [CI] | copy-delta raw | copy-delta residual [CI] | model-copy residual (p) | template share of truth |
|---|---|---|---|---|---|---|---|
| corridor | A.ped_did | 0.951 [0.942,0.959] | +0.113 [0.047,0.177] | 0.966 | 0.411 [0.257,0.558] | -0.298 (4e-4) | 0.951 |
| corridor | A.cone_did | 0.920 [0.914,0.927] | -0.011 [-0.111,0.094] | 0.972 | 0.549 [0.436,0.654] | -0.560 (2e-4) | 0.937 |
| corridor | B.ped_did | 0.892 [0.879,0.904] | -0.122 [-0.271,0.027] | 0.968 | 0.580 [0.457,0.692] | -0.702 (2e-4) | 0.926 |
| corridor | B.cone_did | 0.845 [0.834,0.855] | -0.025 [-0.102,0.054] | 0.944 | 0.487 [0.367,0.593] | -0.512 (2e-4) | 0.897 |
| hazard | A.ped_did | 0.946 [0.937,0.954] | +0.156 [0.081,0.231] | 0.962 | 0.430 [0.285,0.568] | -0.274 (4e-4) | 0.942 |
| hazard | A.cone_did | 0.933 [0.927,0.939] | +0.027 [-0.084,0.143] | 0.975 | 0.570 [0.450,0.681] | -0.543 (2e-4) | 0.940 |
| hazard | B.ped_did | 0.897 [0.884,0.910] | -0.141 [-0.301,0.022] | 0.968 | 0.606 [0.472,0.725] | -0.748 (2e-4) | 0.919 |
| hazard | B.cone_did | 0.853 [0.844,0.862] | -0.022 [-0.093,0.049] | 0.946 | 0.507 [0.386,0.615] | -0.529 (2e-4) | 0.900 |
Relational variants: model residual +0.079/+0.114 (A.ped_rel corridor/hazard), -0.107/-0.068 (A.cone_rel), -0.163/-0.179 (B.ped_rel), -0.053/-0.027 (B.cone_rel); copy-delta residual 0.40-0.55. null_did / action: model raw only 0.32-0.62, residual -0.02 to -0.06; copy-delta residual 0.46-0.53. field_mean baseline raw = 0.86-0.975, residual 0 by construction. Model residual NMSE 2.2-4.3 vs raw NMSE 0.10-0.38 (`G/residual_cosine/<arm>/<field>/nmse`). Only A.ped_did has a model residual cosine whose CI excludes 0 (positive); B.ped_did is negative and borderline. Copy-delta beats the model on residual cosine in every arm x field x site (sign-flip p <= 4e-4; also on raw cosine, p <= 0.007).

## 4. Relational transport (`S/<site>/transport/<block>/<src>-><tgt>.<field>/{residual_cosine,residual_ci,raw_cosine,magnitude_spearman,nulls/<null>/{score_minus_null,ci,p},centred_residual_cosine,centred_nulls}`; null means at `G/transport/<block>/<src>-><tgt>/<field>/nulls/<null>/residual_cosine/{null_mean,p_draws_ge_observed}`). Residual cosine of the synthesised target vs the true target, LOSO over 54 scenes. All sign-flip p = 2.0e-4 (floor) and draw-level p = 0.0099 (= 1/101, floor) for every config/field/null below.

| site | block | config.field | residual [CI] | raw | perm null | iso-Gauss null | cov-Gauss null | rotated null | centred variant |
|---|---|---|---|---|---|---|---|---|---|
| corridor | model A->B | ped_did | 0.784 [0.691,0.866] | 0.999 | -0.030 | +0.178 | +0.006 | +0.196 | 0.835 |
| corridor | model A->B | cone_did | 0.710 [0.677,0.742] | 0.994 | +0.005 | +0.085 | -0.009 | +0.091 | 0.749 |
| corridor | model B->A | ped_did | 0.705 [0.650,0.758] | 0.998 | -0.021 | +0.037 | +0.008 | +0.039 | 0.815 |
| corridor | model B->A | cone_did | 0.791 [0.689,0.881] | 0.998 | -0.006 | +0.153 | -0.004 | +0.165 | 0.814 |
| corridor | true A->B | ped_did | 0.525 [0.460,0.588] | 0.963 | +0.003 | +0.081 | -0.009 | +0.101 | 0.705 |
| corridor | true A->B | cone_did | 0.678 [0.636,0.720] | 0.947 | -0.009 | -0.087 | -0.002 | -0.053 | 0.678 |
| corridor | true B->A | ped_did | 0.450 [0.396,0.504] | 0.975 | -0.006 | -0.046 | +0.005 | -0.042 | 0.461 |
| corridor | true B->A | cone_did | 0.607 [0.527,0.683] | 0.970 | -0.007 | +0.008 | +0.001 | +0.011 | 0.702 |
| hazard | model A->B | ped_did | 0.872 [0.809,0.928] | 0.998 | -0.019 | +0.231 | +0.004 | +0.262 | 0.848 |
| hazard | model A->B | cone_did | 0.700 [0.660,0.738] | 0.992 | -0.006 | +0.112 | -0.012 | +0.123 | 0.713 |
| hazard | model B->A | ped_did | 0.773 [0.700,0.841] | 0.996 | -0.028 | +0.038 | +0.001 | +0.039 | 0.808 |
| hazard | model B->A | cone_did | 0.896 [0.836,0.942] | 0.997 | +0.006 | +0.231 | -0.010 | +0.258 | 0.910 |
| hazard | true A->B | ped_did | 0.778 [0.750,0.805] | 0.959 | +0.006 | +0.114 | +0.003 | +0.146 | 0.792 |
| hazard | true A->B | cone_did | 0.680 [0.639,0.722] | 0.949 | -0.008 | -0.064 | -0.003 | -0.041 | 0.690 |
| hazard | true B->A | ped_did | 0.523 [0.473,0.573] | 0.971 | -0.003 | -0.034 | +0.002 | -0.028 | 0.522 |
| hazard | true B->A | cone_did | 0.648 [0.566,0.721] | 0.972 | -0.010 | +0.032 | +0.007 | +0.037 | 0.751 |
Score-minus-null (S/.../nulls/<null>/score_minus_null) = residual minus the null-mean above; e.g. corridor model A->B ped_did: perm +0.815 [0.720,0.899], iso +0.606 [0.459,0.759], cov +0.778 [0.684,0.861], rotated +0.589 [0.444,0.738]. Relational fields track their parents within ~0.03; null_did / action cross-arm transport 0.58-0.74 (model), 0.575-0.601 (true). Model->truth transport (`model_to_truth_within_arm`, `model_to_truth_cross_arm`): 0.357-0.706 (ped/cone), lowest for A.ped_did (0.357 within, 0.405 B->A cross; hazard 0.438/0.486). Full ranges: corridor model_cross_arm 0.576-0.818, true_cross_arm 0.431-0.678; hazard model_cross_arm 0.594-0.896, true_cross_arm 0.523-0.778.
Oddity to flag: `magnitude_spearman` (rho of ||T_hat|| vs ||T||) is strongly NEGATIVE for nearly all model-source configs (-0.94 to -0.999; e.g. corridor model A->B ped_did rho = -0.997, `G/transport/model_cross_arm/model_A->model_B/ped_did/observed/magnitude_spearman/rho`, p_perm 1e-3) and for true ped_did (-0.97/-0.50). Predicted norms are nearly constant (2.68-2.69). Transport recovers residual direction, not magnitude.

## 5. Identity-matched vs identity-swapped (`G/identity_swap/<config>/{swapped,matched}/observed/residual_cosine/{mean,ci_low,ci_high}`, `.../swapped_minus_matched_residual/{mean,ci_low,ci_high,sign_flip_p}`, `.../swapped_minus_matched_residual_centred`; summary at `S/<site>/identity_swap/<config>/swapped_minus_matched`). Sign convention: swapped - matched (negative = matched better).

| site | config | swapped | matched | swapped-matched [CI] (p) | centred diff (p) |
|---|---|---|---|---|---|
| corridor | model A.ped->B.cone vs B.ped | 0.722 | 0.784 | -0.063 [-0.131,+0.012] (0.11) | -0.051 (0.11) |
| corridor | model B.cone->A.ped vs A.cone | 0.775 | 0.791 | -0.016 [-0.052,+0.020] (0.40) | +0.015 (0.55) |
| corridor | model A.ped_rel->B.cone_rel vs B.ped_rel | 0.707 | 0.774 | -0.066 [-0.137,+0.012] (0.098) | -0.039 (0.26) |
| corridor | model B.cone_rel->A.ped_rel vs A.cone_rel | 0.779 | 0.818 | -0.040 [-0.087,+0.007] (0.11) | +0.000 (1.0) |
| corridor | true A.ped->B.cone vs B.ped | 0.492 | 0.525 | -0.034 [-0.093,+0.028] (0.26) | -0.140 (2e-4) |
| corridor | true B.cone->A.ped vs A.cone | 0.347 | 0.607 | -0.261 [-0.303,-0.219] (2e-4) | -0.307 (2e-4) |
| corridor | true A.ped_rel->B.cone_rel | 0.539 | 0.603 | -0.064 [-0.101,-0.027] (8e-4) | -0.053 (0.028) |
| corridor | true B.cone_rel->A.ped_rel | 0.346 | 0.574 | -0.228 [-0.266,-0.192] (2e-4) | -0.259 (2e-4) |
| hazard | model A.ped->B.cone vs B.ped | 0.709 | 0.872 | -0.163 [-0.201,-0.125] (2e-4) | -0.087 (0.0024) |
| hazard | model B.cone->A.ped vs A.cone | 0.817 | 0.896 | -0.079 [-0.116,-0.047] (2e-4) | -0.039 (0.044) |
| hazard | model A.ped_rel->B.cone_rel | 0.702 | 0.864 | -0.162 [-0.199,-0.123] (2e-4) | -0.097 (6e-4) |
| hazard | model B.cone_rel->A.ped_rel | 0.826 | 0.918 | -0.092 [-0.132,-0.058] (2e-4) | -0.044 (0.066) |
| hazard | true A.ped->B.cone vs B.ped | 0.604 | 0.778 | -0.174 [-0.232,-0.117] (2e-4) | -0.155 (2e-4) |
| hazard | true B.cone->A.ped vs A.cone | 0.421 | 0.648 | -0.227 [-0.269,-0.186] (2e-4) | -0.260 (2e-4) |
| hazard | true A.ped_rel->B.cone_rel | 0.569 | 0.721 | -0.152 [-0.206,-0.097] (2e-4) | -0.119 (4e-4) |
| hazard | true B.cone_rel->A.ped_rel | 0.421 | 0.615 | -0.195 [-0.235,-0.155] (2e-4) | -0.228 (2e-4) |
Note: swapped transport is still far above every null (0.35-0.83); identity only modulates it.

## 6. Covariate heterogeneity (`G/covariate_heterogeneity/<arm>/<field>/<true|model>/covariates/<cov>/{degenerate,n_distinct,range,magnitude/{rho,ci_low,ci_high,p_perm},directional_deviation/{...}}`)
Covariates present: hazard_pixels, hazard_patches, hazard_bbox_area_px, lane_offset_m, ego_speed_mps, prefix_travel_m. There is NO "distance" covariate (prefix_travel_m is the nearest). Flagged degenerate (<5 distinct): hazard_patches (=12, n_distinct 1, rho n/a), hazard_bbox_area_px (=1840, n/a), hazard_pixels (1045-1047, 3 values). NOT flagged but effectively degenerate by range: ego_speed_mps 8.273-8.309 m/s (0.4 % spread), prefix_travel_m 3.900-3.908 m, lane_offset_m -6.9e-6 to +1.26e-5 m (float noise). All rho below are therefore against sub-percent stimulus variation and should not be read as heterogeneity effects.
- Magnitude vs ego_speed (= prefix_travel; corridor): true A.ped -0.435 [-0.646,-0.196] p .002; true A.cone -0.762 p .001; true B.ped -0.948 p .001; true B.cone +0.225 p .11; model A.ped -0.662, A.cone -0.885, B.ped -0.866, B.cone -0.662 (all p .001). hazard site similar (true -0.70/-0.81/-0.95/+0.19; model -0.82/-0.95/-0.92/-0.94).
- Magnitude vs hazard_pixels (degenerate): true -0.27 to -0.60 (ped), -0.39/-0.02 (cone); model -0.53 to -0.64.
- lane_offset_m: magnitude rho -0.14 to +0.20 (all p > 0.1); directional deviation -0.24 to -0.41 (p .003-.06).
- Directional deviation vs ego_speed: true A.ped +0.14 (p .33), A.cone -0.47 (p .001), B.ped -0.28 (p .045), B.cone -0.37 (p .006); model B.cone -0.84 (p .001), A.ped -0.58 (p .001).

## 7. Verification of colleague summary
- "field ~ one direction (PC1 0.54-0.67)": CONFIRMED for TRUE ped/cone DiD at hazard_corridor (0.541, 0.564, 0.565, 0.667; `S/hazard_corridor/field/<A|B>.<ped|cone>_did/true/pc1`). At `hazard` the true range is 0.471-0.756; MODEL PC1 is 0.57-0.95. Spherical variance 0.02-0.05 vs uniform 0.86 supports "dominant direction"; PR 2-3 and top-5 = 0.79-0.90 mean "low-dimensional", not literally one direction.
- "residual cosine ~ 0.1 vs copy-delta 0.4-0.6": DIFFERS in detail. Model residual is ~0.1 only for A.ped_did (+0.113 corridor, +0.156 hazard); the other seven arm x field entries are -0.14 to +0.03 (B.ped_did -0.122/-0.141, CI includes 0). Correct statement: model residual cosine -0.14 to +0.16 (median ~ -0.02), copy-delta 0.41-0.61 (`S/<site>/residual_cosine/<arm>.<field>/{model,copy_delta}/residual`). Copy-delta > model in all 8 x 2 sites, p <= 4e-4. "Template capture" is supported: raw cosine 0.85-0.95, template share of truth 0.90-0.95.
- "transport across arms 0.45-0.82": CONFIRMED approximately for hazard_corridor (true 0.431-0.678, model 0.576-0.818 over all fields; ped/cone: true 0.450-0.678, model 0.705-0.791). At `hazard` the model cross-arm reaches 0.896 and true 0.778, so the global range is 0.43-0.90. All nulls ~0 (perm, cov-Gauss) or 0.04-0.26 (iso-Gauss, rotated); p 2e-4 everywhere.
- "identity-matched > swapped by 0.08-0.16 (p 2e-4)": DIFFERS -- holds ONLY for the `hazard` site MODEL fields (0.079, 0.092, 0.162, 0.163; all p 2e-4; `S/hazard/identity_swap/model: .../swapped_minus_matched`). At the gate's primary site `hazard_corridor` the MODEL identity effect is 0.016-0.066 with p 0.098-0.40 (NOT significant; CIs include 0), and the TRUE-field effect is asymmetric: 0.034 (p .26) / 0.064 (p 8e-4) for A.ped->B, but 0.228 / 0.261 (p 2e-4) for B.cone->A. At `hazard` the true effect is 0.15-0.23 (p 2e-4). A paper sentence must name the site; the colleague's number is the secondary site.
- Not in the summary but should be: covariates are essentially degenerate in range (sec. 6), transport magnitude Spearman is strongly negative (sec. 4), model->truth transport (0.36-0.71) is below model->model (0.70-0.90), and only one seed / one model family is covered.
