# Mechanism digest — driving cell, JEPA-WM predictors, arm A / arm B, model seed 0

Files (all under artifacts/cgs_pilot/drive_mech/arm{A,B}_seed0/): localize/interaction_map.json, patch_step0/patch_results.json,
patch_step0/band.json (arm A only), jacobian_sonar/jacobian_sonar.json. No other seeds/subsets exist: `find drive_mech -name '*.json'`
returns only these 9 files (plus localize/activations/index.json). `jacobian_seeds.txt` = 3 4 7 9 10 14 15 17 -> the "8 seeds" of the
Jacobian sonar are 8 SCENE seeds within model seed 0, not 8 model seeds. mech_w1/jacobian_sonar/jacobian_sonar.json is the robot
(RoboCasa, checkpoint jepa_wm_droid, scene seeds 203..267, n=12) cell, not the driving cell. There is no "jacobian_subset" stage on disk.

Shared scope strings: localize `interpretation_scope` = "Descriptive/statistical localization ... Not a causal claim; site selection
requires patch_site.py gates ... step 0 is the clean map." patch `interpretation_scope` = "Latent-forecast endpoint on the hazard+corridor
region; contact/clearance and fresh candidate-ranking endpoints ... are not evaluated here." jacobian `interpretation_scope` =
"Differentiable action sensitivity of the latent forecast (no donor, no candidate-action pair). Relational = hazard-induced change of the
action Jacobian minus the matched off-path (H0') change. Step 0 is primary." All three carry the DROID-only/zero-shot checkpoint caveat.
Token groups (localize `token_group_source.aliases`): egg->hazard, gripper_corridor->corridor, gripper->"(empty: no manipulator)".

## 1. Localization (interaction_map.json; n_scenes=53, n_perm=2000, n_boot=2000, alpha=0.05, min_ratio=0.05, min_specificity=2.0, 31 non-adaln sites x 3 steps x 5 groups)

Verdict fields: `earliest_live_gripper_corridor` = [] and `maps_by_step.step0.top10_gripper_corridor` = [] in BOTH arms. This is a
bookkeeping bug, not a negative result: localize_interaction.py builds `primary` from canon(group)==PRIMARY_GROUP["driving"]="hazard_corridor",
but no dumped group canonicalises to that name (aliases map to "corridor"/"hazard"), so `primary` is empty; consequently every significant
hazard-group cell got labelled `adaln_elementwise_candidate` (91 cells) and no cell got `relational_candidate`. `fdr_by.rejected` is a
list of 455 cell keys (n_tests=455); `significant_maxt` is True for 455/465 cells (all step-0 non-adaln cells have p_maxt_fwer=0.0005
= 1/2001 floor; q_by=0.00335). Localization is therefore non-discriminative on significance; only `magnitude_ok` separates sites.
"Live" below = my reconstruction of the script's own rule (significant_maxt AND magnitude_ok), step 0, group=corridor.
Key path for each row: site_results[site_id,step=0,group].cv_projection.{interaction,relational,null_factor,identity}.{t,p_maxt_fwer,q_by,mean}, .ratio_to_action, .background_specificity, .magnitude_ok.

Arm A step 0, corridor: 19/31 sites live (29 sig, 19 magnitude_ok). Top by t (ranking_key "(magnitude_ok, t desc)"):
  L05.mlp_in  t=916 p=0.0005 q=0.00335 mean=2.78 ratio=0.447 spec=2.76 rel_t=310 (p=0.0005) null_t=239 id_t=712
  L03.mlp_in  t=902 p=0.0005 q=0.00335 mean=4.68 ratio=0.462 spec=2.09 rel_t=410 null_t=315 id_t=1458
  L03.resid_post/L04.resid_pre t=619 ratio=0.522 spec=2.23 rel_t=488 ; L04.mlp_in t=536 ; L02.resid_post t=533 ; L04.resid_post t=483
  L03.mlp_out t=412 p=0.0005 q=0.00335 mean=3.89 d_z=56.6 ratio=0.597 spec=2.23 magnitude_ok=True (live) rel_t=443 null_t=239 id_t=640 obj_t=471
  L03.attn_out (corridor, step 0): t=365 ratio=0.286 spec=1.91 -> magnitude_ok=False, NOT live.
  Hazard ("egg") group: 4 live (L01.mlp_out, L02.attn_out, L03.mlp_out t=239 ratio=0.642 spec=2.05, L05.mlp_out). Background: 0 live (spec=1 by construction).
Arm B step 0, corridor: 26/31 sites live. Top: L05.attn_out t=523 ratio=0.395 spec=2.35 rel_t=298 null_t=506 ; L03.mlp_in t=431 ;
  L02.mlp_in t=414 ; L01.resid_post t=386 ; L02.resid_post t=359 ; L04.resid_post t=351. L03.mlp_out corridor s0: t=142 mean=2.63 d_z=19.5 ratio=0.527 spec=2.05 live; L03.attn_out s0: t=95.8 ratio=0.463 spec=1.88 not live; rel_t=166 null_t=124 (L03.mlp_out).
  Hazard group: 7 live (L00.mlp_out, L01.mlp_out, L01.resid_post, L02.resid_pre, L02.mlp_out t=311 ratio=0.608, L04.mlp_out, L05.mlp_out).
Top of the global `ranked` list in both arms is dominated by step 2 (autoregressive) cells (A: L05.mlp_out s2 t=4231; B: L04.mlp_in s2 t=908) — do not quote these as step-0 results.
Relational null (cv_projection.relational, interaction minus H0' contrast) and null_factor are both p_maxt_fwer=0.0005 at every listed site — again floor values.

## 2. Donor activation patching, step 0 (patch_results.json; n_scenes=53, n_sites=12 = L00..L05 x {attn_out,mlp_out}, n_random=3, mode=one_shot, null_factor_status=full, identity_factor=full)

Gate (`gate`): recovery_threshold=0.3, control_threshold=0.1, alpha=0.05, primary_group=gripper_corridor (=corridor alias), primary_direction=hazard_safe_to_unsafe.
`gate.liveness` = "H-only patch-effect DiD at gripper_corridor significant under sign-flip max-T AND H0' null donor median recovery < control_threshold
(... the level-3 identity donor is reported, not a liveness criterion)". `selection.rule` = "earliest interaction-live site with >=threshold recovery
in both hazard directions; band = maximal contiguous run of live sites containing it". NOTE: liveness does NOT require wrong-group/time-shift/
main-effect-only < 0.1; those feed only the reported flag `controls_below_threshold`.
Key paths: site_results[site,group=gripper_corridor].directions.hazard_safe_to_unsafe.{donor.recovery_region_median, donor.recovery_region.{point,ci_low,ci_high},
donor.patch_effect_did.point, donor.patch_effect_did_test.{t,p_maxt_fwer}, identity.recovery_region_median, identity.donor_minus_identity_test.p_maxt_fwer,
null_factor.recovery_region_median, null_factor.donor_minus_null_test.p_maxt_fwer, random/sham/wrong_group/time_shift/main_effect_only.recovery_region_median};
site flags .interaction_live, .controls_below_threshold, .recovery_both_directions_ok. Identity/null_factor exist only for the safe_to_unsafe direction.

Arm A, corridor, safe->unsafe (recovery = donor.recovery_region_median; CI = donor.recovery_region 95% cluster bootstrap):
  site         recov   CI            DiD    DiD p   identity  H0'null  random  sham  wrong  tshift  main   live ctrl<0.1
  L03.mlp_out  0.719  [0.715,0.724]  0.298  0.0005  0.0217    0.0003   0.0004  0     0.609  0.473   0.326  True False
  L03.attn_out 0.0020 [0.0022,0.0030] 0.0014 0.0015 -0.0004   -0.0001  0.0000  0     -0.0005 0.013  0.0017 True True
  L04.mlp_out  0.061  [0.061,0.071]  -0.107 1       0.0072    0.0003   0.0004  0     0.016  0.118   0.078  False False
  L04.attn_out 0.022 ; L05.attn_out 0.015 ; all other sites <0.003 ; all not live (DiD p=1).
  L03.mlp_out unsafe->safe: recov 0.406 [0.401,0.406], DiD 0.396 p=0.0005, wrong 0.184, tshift 0.377, main 0.396 -> recovery_both_directions_ok=True.
  L03.attn_out unsafe->safe: recov 0.051, DiD 0.055 p=0.0005. Hazard-group table: L03.mlp_out 0.673/0.320, wrong-group 0.023 (safe->unsafe).
  Per-scene (donor.per_scene, L03.mlp_out safe->unsafe): min 0.686, max 0.767, median 0.719, 53/53 >= 0.3, DiD>0 in 53/53.
  selection.selected = L03.mlp_out step 0 gripper_corridor. selection.band = [L03.attn_out (recovery_median 0.0020, fraction_scenes_positive_did 0.623),
  L03.mlp_out (recovery_median 0.719, fraction_scenes_positive_did 1.0)]. band.json is a verbatim copy of `selection` plus frozen_on_seeds (53 scene seeds).
  band.json `recovery_median` = directions.hazard_safe_to_unsafe.donor.recovery_region_median of that site. So the 0.0019828 for L03.attn_out
  is its actual pooled donor recovery: L03.attn_out is in the band only because it is interaction-live (DiD t=3.66 p=0.0015, null 0.0) and
  adjacent to L03.mlp_out — it recovers essentially nothing. The 72% figure belongs to L03.mlp_out alone.

Arm B, corridor, safe->unsafe:
  site         recov   CI            DiD     DiD p   identity  H0'null  random   sham  wrong   tshift  main    live ctrl<0.1
  L02.mlp_out  0.270  [0.264,0.271]  -0.249  1       -4.41     0.0050   -0.0010  0     -0.095  0.162   0.052   False False
  L03.mlp_out  0.044  [0.039,0.046]  -0.116  1       -2.55     0.0005   -0.0005  0     -0.007  0.018   0.003   False True
  L00.attn_out 0.0066 DiD 0.0134 p=0.0005 null 0.0075 -> live True ; L03.attn_out 0.0097 DiD 0.0134 p=0.0005 live True ; L04.attn_out 0.012 DiD 0.0187 p=0.0005 live True
  all other sites |recov| <= 0.022, DiD p=1. Per-scene L02.mlp_out: min 0.223 max 0.284, 0/53 >= 0.3, DiD>0 in 0/53.
  selection.selected = null, selection.band = [] (no band.json written). recovery_both_directions_ok=False for all 12 sites.
  Note: the three live arm-B sites are live via a tiny positive DiD (~0.013) with ~0.01 recovery; L02.mlp_out's 0.27 recovery comes with a
  strongly NEGATIVE DiD (-0.249) and the identity donor gives -4.41 (worse than sham), i.e. the recovery is not a hazard-specific effect.

## 3. Action-Jacobian sonar E1 (jacobian_sonar.json; n_scenes=8 (scene seeds 3..17), n_identity=8, n_null=8, n_boot=2000, n_perm=2000 requested but tests are exact sign-flip n_perm=256, hvp_eps=0.1, mediation_steps=[0]; a0/a1 = Jacobian w.r.t. action dim 0/1 of the 2-D (steer, throttle) action, audit_contamination.py:45)

Key paths: pooled["0"].{a0,a1}.<field>.{point,ci_low,ci_high}; tests pooled["0"].{a0,a1}.<name>_test.{t,p_maxt_fwer}; sites pooled["0"].E1_sites_{a0,a1};
per-site pooled["0"].mediation["<site>|<group>"].{a0,a1}.{E1_site,S_F_site,fraction_of_total_relational,relational_frobenius_region,relational_minus_random_test,relational_minus_sign_test,top_angle_deg}.
E1_site rule (action_jacobian_sonar.py:738): S_F_site >= 0.05 AND top_angle >= 10 deg AND p(vs random) < 0.05 AND p(vs sign-flip) < 0.05.

                              Arm A a0            Arm A a1            Arm B a0            Arm B a1
  S_F_route (relational)      1.24 [1.19,1.27]    0.774 [0.758,0.788] 1.31 [1.30,1.33]    0.971 [0.936,1.00]
  S_F_route_identity          1.11 [1.09,1.14]    0.725 [0.695,0.754] 1.24 [1.22,1.25]    0.667 [0.651,0.691]
  S_F_route_object            1.02 [1.00,1.05]    0.717 [0.688,0.744] 1.37 [1.35,1.38]    0.710 [0.699,0.727]
  S_F_route_random            0.405 [0.353,0.461] 0.317 [0.302,0.333] 0.398 [0.370,0.426] 0.428 [0.410,0.445]
  S_F_route_sign (sign-flip)  0.983 [0.889,1.10]  0.697 [0.591,0.857] 0.894 [0.786,0.995] 0.657 [0.624,0.685]
  ||dJ_hazard||_F region      40.2 [40.0,40.5]    18.2 [17.2,19.3]    40.4 [39.4,41.3]    17.0 [16.7,17.2]
  ||dJ_null (H0')||_F         34.3 [33.8,35.0]    15.0 [13.5,16.3]    40.6 [39.8,41.2]    20.8 [19.9,21.4]
  ||dJ_object||_F             37.6 [37.1,38.3]    17.6 [16.1,18.9]    46.6 [45.7,47.5]    16.0 [15.8,16.2]
  ||dJ_random||_F             20.1 [18.4,22.4]    11.3 [10.8,11.9]    21.3 [19.6,23.0]    10.6 [10.1,11.1]
  ||relational||_F            40.8 [39.9,41.5]    14.3 [14.0,14.6]    44.9 [44.3,45.4]    22.5 [21.9,22.9]
  relational - random         20.7 [17.7,22.8] t=14.0 p=0.0078 | 3.07 [2.46,3.71] t=8.97 p=0.0078 | 23.6 [21.7,25.5] t=22.9 p=0.0078 | 11.8 [11.4,12.2] t=50.6 p=0.0078
  relational - sign           6.72 [4.02,9.40] t=4.38 p=0.0117 | -2.23 [-4.53,-0.63] t=-2.04 p=0.992 | 11.4 [9.67,13.2] t=11.8 p=0.0078 | 7.79 [7.21,8.35] t=23.8 p=0.0078
  identity - random           12.9 [10.6,14.7] t=11.4 p=0.0078 | 1.44 [0.81,2.03] t=4.33 p=0.0078 | 16.7 [14.9,18.7] t=16.2 p=0.0078 | 2.56 [1.97,3.23] t=7.42 p=0.0078
  top-angle - random (deg)    30.3 [27.2,33.2] p=0.0078 | 29.0 [27.1,30.5] p=0.0078 | 37.6 [36.1,39.5] p=0.0078 | 21.8 [20.5,23.2] p=0.0078
  top-angle - sign (deg)      5.85 [2.89,8.50] p=0.0156 | 9.07 [3.38,13.1] p=0.0195 | 10.0 [6.09,14.3] p=0.0078 | 10.6 [8.27,12.9] p=0.0078
  E1_total_level              all four: S_F_ge_0.05=True, top_angle_ge_10deg=True, beats_random_p=0.0078; beats_sign_p = 0.0117 / 0.992 (A a1 FAILS) / 0.0078 / 0.0078
  ratio ||dJ_hazard||/||J(H0)|| 0.918 / 0.751 / 0.925 / 0.700 ; cosine(dJ_hazard, dJ_object) 0.674 / 0.745 / 0.643 / 0.702 ; LOSO relational mean 0.767 / 0.285 / 0.528 / 0.416 (all p=0.0117)
  Minimum attainable p with 8 scenes = 1/256=0.0039; reported floor 0.0078 = 2/256.

E1 sites (pooled["0"].E1_sites_*):
  Arm A a0 (10/36 cells): corridor {L00.attn_out, L02.attn_out, L03.attn_out}; all {L00.mlp_out, L01.attn_out, L01.mlp_out, L03.attn_out}; hazard {L01.attn_out, L01.mlp_out, L03.attn_out}.
  Arm A a1 (3/36): corridor {L00.attn_out, L01.mlp_out, L02.mlp_out}. -> layers L00-L03 only; L03.mlp_out is NOT an E1 site in arm A (S_F_site 1.71 but p_vs_sign=0.977 at corridor, a0).
  Arm B a0 (25/36): all 6 layers; corridor {L00.attn_out, L00.mlp_out, L03.attn_out, L03.mlp_out, L04.attn_out, L04.mlp_out, L05.attn_out, L05.mlp_out}; missing L01.attn_out, L02.attn_out, L01/L02 corridor cells.
  Arm B a1 (35/36): every cell except L01.attn_out|gripper_corridor.
  Largest per-site S_F_site at corridor: A a0 L05.attn_out 3.16 (not E1, p_vs_sign=1), L00.attn_out 1.84 (E1), L03.mlp_out 1.71; B a0 L03.attn_out 2.76 (E1), L03.mlp_out 2.17 (E1), L02.mlp_out 2.11 (not E1).
  Largest fraction_of_total_relational (a0, corridor): A L03.mlp_out 0.379, L04.mlp_out 0.370; B L03.mlp_out 0.454, L03.attn_out 0.410.

## 4. Colleague summary — claim-by-claim

- "arm A seed 0 live band {L03.attn_out, L03.mlp_out} step 0 corridor tokens": CONFIRMED as the file's `selection.band` (patch_results.json / band.json), but MISLEADING: only L03.mlp_out carries recovery; L03.attn_out recovery_median = 0.0020 (band[0].recovery_median), fraction_scenes_positive_did 0.62.
- "donor recovery 72 %": CONFIRMED for L03.mlp_out only — 0.719 [0.715,0.724] (site_results[L03.mlp_out|gripper_corridor].directions.hazard_safe_to_unsafe.donor.recovery_region_median / recovery_region). Reverse direction 0.406.
- "53/53 scenes": CONFIRMED (per_scene min 0.686, all >= 0.3; band[1].per_scene_stability.fraction_scenes_positive_did = 1.0).
- "identity donor 0.02": CONFIRMED 0.0217 (…identity.recovery_region_median; donor-minus-identity p=0.0005).
- "null/random/sham ~ 0": CONFIRMED 0.0003 / 0.0004 / 0.0 (…null_factor / random / sham .recovery_region_median).
- "NOT token-group selective (wrong-group 0.61, time-shift 0.47, main-effect-only 0.33)": CONFIRMED 0.609 / 0.473 / 0.326; file flag controls_below_threshold=False for L03.mlp_out. Also wrong-group in the hazard-group table is only 0.023, so corridor-donor into hazard tokens fails but hazard-donor into corridor tokens succeeds — asymmetric, worth stating precisely.
- "Arm B seed 0: no site >= 30 % (best L02.mlp_out 0.27)": CONFIRMED 0.270 [0.264,0.271]; selection.selected=null, band=[]. "-> distributed" is an interpretation not in the file; the file shows L02.mlp_out's DiD is negative (-0.249, p=1) and its identity donor is -4.41, so the 0.27 is not hazard-specific recovery. Three arm-B sites are interaction-live (L00/L03/L04.attn_out) with ~0.01 recovery.
- "Jacobian sonar (8 seeds)": DIFFERS in meaning — 8 SCENE seeds (3,4,7,9,10,14,15,17) at model seed 0; n_scenes=8. No per-seed or subset Jacobian files exist under drive_mech.
- "relational S_F 0.8-1.3 vs random 0.3-0.4 both arms": DIFFERS slightly — S_F_route = 1.24 (A a0), 0.774 (A a1), 1.31 (B a0), 0.971 (B a1); random = 0.405, 0.317, 0.398, 0.428. Correct ranges: 0.77-1.31 vs 0.32-0.43.
- "identity-specific": CONFIRMED in the sense S_F_route_identity (1.11/0.725/1.24/0.667) >> random and identity_minus_random p=0.0078 in all four; but note S_F_route_object (1.02/0.717/1.37/0.710) is comparable to identity, and cosine(dJ_hazard, dJ_object) = 0.64-0.75, so the hazard-specific direction is only partially distinct from a generic object-insertion direction. Also arm A a1 fails the sign-flip control (beats_sign_p=0.992).
- "E1 sites A: L00-L03": CONFIRMED (a0: L00,L01,L02,L03; a1: L00,L01,L02). "B: everywhere": CONFIRMED approx. (a0 25/36 cells across L00-L05; a1 35/36).
- Localization-level "live band" fields are empty in both arms (earliest_live_gripper_corridor=[]) because of the hazard_corridor alias mismatch; do not cite them as a negative result, and do not cite the localization p-values as discriminative (all at the 1/2001 floor).
