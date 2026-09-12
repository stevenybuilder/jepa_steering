# Figures plan — `cgs_workshop_draft.md`

Four figures maximum. Each entry gives the panel layout, the exact data source (artifact path and JSON key), and what the reader should take from it. Driving-cell gate, mechanism (seed 0), and cross-arm geometry (seed 0) artifacts now exist and are named below; only the steered-planner panel remains a placeholder. Nothing is to be sketched from prose numbers.

All artifact paths are relative to `artifacts/cgs_pilot/` unless stated. Plotting scripts should read the JSON directly and print the values they draw, so every plotted number is traceable.

---

## Figure 1 — Design schematic (method + benchmark in one picture)

**Purpose.** Show, without numbers, the three contributions and how they nest: the factorial stimulus with its H0′ null, the two gates (validity → B-gate) that must pass before any mechanism arm, the frozen sonar pipeline and geometry ladder, and the reversed-assignment arms for the driving cell.

**Layout (one wide panel, three columns).**
- *Left — stimulus and estimand.* 2×2 hazard × action grid (H0/H1 × A0/A1) plus the H0′ cell; arrows to "true future by replay" and to "model prediction"; the DiD I_true = Δ_h Δ_a z and the relational DiD (minus H0′). Two rows of thumbnails: RoboCasa egg (substrates 1–2) and MetaDrive pedestrian / cone (driving).
- *Middle — the gated pipeline.* Vertical flow: counterfactual-validity gate (Rec ≥ 0.5, cosine > 0) → **B-gate** (NMSE ≤ 0.75 ⇒ ≥ 25 % captured; Rec_h1 − Rec_h0 > 0; flip ≥ 0.25) → retrieval control → LOSO localization (site × step × token group; max-T, BY; magnitude ratio ≥ 0.05) → on-manifold donor patching with the control set → geometry ladder (direction → low-rank → sonar subspace → frame tournament → curved/rotating → bilinear → conceptor → higher-order → Jacobian sonar). A red "STOP" branch from the B-gate labelled "change stimulus/model, not method".
- *Right — matched-consequence arms.* Two boxes, Arm A (pedestrian solid / cone ghost) and Arm B (pedestrian ghost / cone solid), fed by the identical clip set (images, actions, future multiset), sharing the frozen DINOv3 encoder; below, the cross-arm comparison (principal angles, Procrustes transport, conceptor cross-similarity) with the prediction "appearance subspace shared; relational subspace identity-swapped".
- *Inset.* The AdaLN structural fact: action → (shift, scale, gate) broadcast to all tokens; "relational state must be cross-token".

**Data source.** None (schematic). Text labels must match the thresholds in `scripts/cgs_pilot/behavior_gate.py` and `counterfactual_validity_gate.py` and the arm definition in `experiment_design.md` "v0.7 amendment".

---

## Figure 2 — Gate / B-gate across substrates (the ambiguous-target problem)

**Purpose.** One glance should show that both released checkpoints pass (or nearly pass) the validity gate on the *action* effect yet fail the behavioural-target gate on the *interaction*, stably across n, and that the driving arms are the cells where the gate is expected to separate.

**Layout (2 × 2 panels, shared style).**
- *(a) Captured interaction fraction* (bar per cell, B-gate threshold line at 0.25): JEPA-WM n=21, n=34, n=66; V-JEPA 2-AC n=34; driving arm A, arm B (solid identity: 0.685 / 0.630 / 0.673 and 0.549 / 0.504 / 0.528; ghost identity 0.567 / 0.586 / 0.556 and 0.459 / 0.370 / 0.389; cross-truth cells negative, clip at 0 and annotate).
  Source: `model_captured_interaction_fraction` in `heldout_v1_merged_w1/behavior_gate_cf_gate_sonar_all.json` (0.0449), `heldout_v1_merged_w2/behavior_gate_cf_gate_discovery.json` (0.0648), `heldout_v1_merged_w2/behavior_gate_cf_gate_all.json` (0.0565), `heldout_v1_merged_w2_vj2ac/behavior_gate_vj2ac.json` (0.0). Driving: `drive_results_summary.json` → `driving_models[*].captured_fraction` and Table R3 (`drive_eval/arm{A,B}_seed{0,1,2}/behavior_gate_discovery*.json`), cross-truth from `drive_eval_crosstruth/*/crosstruth_summary.json` (Table R2).
- *(b) Rec_h1 and Rec_h0 with 95 % CIs* (paired dots per cell; validity threshold 0.5): shows action-effect recovery is fine for JEPA-WM and at chance for V-JEPA 2-AC, and that Rec_h1 ≤ Rec_h0 everywhere (no hazard specificity).
  Source: `pooled.rec_h1` / `pooled.rec_h0_control` (median, ci_low, ci_high) in `heldout_v1_merged_w1/cf_gate_sonar_all.json`, `heldout_v1_merged_w2/cf_gate_discovery.json`, `heldout_v1_merged_w2/cf_gate_all.json`, `heldout_v1_merged_w2_vj2ac/cf_gate_discovery_vj2ac.json`.
- *(c) Specificity Rec_h1 − Rec_h0 and vs H0′* (point + sign-flip p annotated): `T1c_rec_h1_minus_h0`, `T1c_p`, `T1c_rec_h1_minus_h0prime` from the four `behavior_gate_*.json` files; driving: arm A +0.229 / +0.211 / +0.203, arm B +0.168 / +0.161 / +0.102 (all p 2·10⁻⁴), identity contrast Rec_ped − Rec_cone A +0.072 / +0.039 / +0.031, B +0.037 / +0.059 / +0.106 (does not reverse in B) from Table R1 (`behavior_gate_discovery*.json` → `T1c_rec_h1_minus_h0`, `identity_contrast.rec_h1_minus_h3`).
- *(d) Planner currency*: CEM ranking flip rate (all 0.0; `T2a_flip_rate_h1_vs_h0`) and normalised energy gap with p (`T2b_energy_gap_norm_h1_vs_h0`, `T2b_p`) per cell; driving flip rates 0.889–0.926 (Table R1, `T2a_flip_rate_h1_vs_h0`); the steered-planner safe-choice endpoint `[[RESULT: STEERING TABLE — PENDING (artifacts/drive_coast_table/coast_table.json)]]` enters here when available.

**Also plot (light grey, behind the bars)** the retrieval floor: model interaction cosine vs best copying baseline, from `retrieval_baseline_sonar.json` (w1), `retrieval_baseline.json` (w2), `retrieval_baseline_vj2ac.json` (`baselines.*.model_mean_cosine`, `baseline_mean_cosine`), to show the small interaction is above copying but far below the B-gate.

---

## Figure 3 — Localization vs patching: significance without magnitude

**Purpose.** The paper's central empirical point in one plot: internal hazard×action directions are statistically robust, yet their magnitude relative to the action effect sits at the behavioural captured fraction, and donor patching at those sites recovers ~0 % of the hazard gap while appearance-only sites recover a lot but fail the controls.

**Layout (two panels).**
- *(a) Localization map, imagined step 0, route (gripper/corridor) tokens, wave 2 (n=34, full H0′ null).* x = predictor site in residual order (L00 … L11, attn_out / mlp_in / mlp_out / resid_post), y = interaction/action magnitude ratio (log scale), marker size or colour = −log10 FWER p (max-T), filled if `relational_p_maxt_fwer` < 0.05. Horizontal band at 0.045–0.065 = behavioural captured fraction (from Fig. 2a), horizontal line at 0.05 = magnitude-candidate threshold.
  Source: `heldout_v1_merged_w2/mech_w2_localize_interaction_map.json` → `maps_by_step.step0` entries for group `gripper_corridor` (`ratio_to_action`, `t`, `p_maxt_fwer`, `relational_t`, `relational_p_maxt_fwer`, `magnitude_ok`); top qualifying list in `earliest_live_gripper_corridor` (e.g. `L00.mlp_out` ratio 0.0504, t 6.157, FWER p 5·10⁻⁴). Overlay the wave-1 v2 candidate `L07.attn_out` (t 4.08, p_maxT 0.020, ratio 0.070) from `heldout_v1_merged_w1/mech_w1_localize_v2_interaction_map.json`.
- *(b) Patch recovery vs controls, wave 1 step 0.* Grouped bars per site/token-group: donor recovery, main-effect-only donor, unrelated-site, wrong-group, H0′ donor (where present), sham/random. Show route-token `L07.attn_out` (donor 0.1 %), the egg-token appearance sites `L00.attn_out` (donor 53 %, main-effect-only 53 %, wrong-group −0.32) and `L09.resid_post` (donor 0.75, unrelated 0.75, main-effect 0.59, H0′ 0.20), and the conceptor band L06–L08.attn_out (max 0.23 %). Selection threshold line at 30 %, control ceiling at 10 %.
  Source: `heldout_v1_merged_w1/mech_w1_patch_step0_results.json` (`per_pair.<seed>.<site>.0.<group>.<direction>.fixed_0.donor.recovery_region` and the `random` / `sham` / `unrelated_site` / `main_effect_only` / `wrong_group` siblings; `selection.band` = []), `heldout_v1_merged_w1/mech_w1_focus_patch_results.json` (`focus_tables`), `mech_w1/conceptor_results.json` (`band_vs_single[*].band_target_h1`).
- *Optional inset in (b):* the Jacobian sonar magnitude bar — S_F route 0.16 [0.12, 0.20] vs random-token 0.32 vs sign-randomised 0.21 — from `mech_w1/jacobian_sonar/jacobian_sonar.json` → `pooled["0"].a0.S_F_route`, `S_F_route_random`, `S_F_route_sign`; label "hazard changes the action response less than noise".

**Driving version of this figure (now the primary Fig. 3; the egg panels become the left-hand control column).** Same two panels per arm, seed-0 pair, imagined step 0, corridor tokens, 53 discovery scenes.
- *(a) Localization is non-discriminative in the driving cell*: every step-0 site sits at the max-T floor (p 5·10⁻⁴) in both arms; plot the interaction/action magnitude ratio only (arm A 19/31 corridor sites ≥ 0.05, largest `L03.mlp_out` 0.60; arm B 26/31, largest `L03.mlp_out` 0.53). Source: `drive_mech/arm{A,B}_seed0/localize/interaction_map.json` → `site_results[*]` (`group` corridor, `step` 0, `ratio_to_action`, `magnitude_ok`, `p_maxt_fwer`); note `earliest_live_gripper_corridor = []` is an alias defect, not a result.
- *(b) Patch recovery vs controls — the paper's magnitude contrast.* Grouped bars per site: arm A `L03.mlp_out` corridor donor **0.719** (reverse 0.406) vs identity 0.022, H0′ null 0.000, random 0.000, sham 0.000, and the failed selectivity controls main-effect-only 0.326, time-shift 0.473, wrong-group 0.609; neighbours `L03.attn_out` 0.002, `L02.mlp_out` 0.001, `L04.mlp_out` 0.061. Arm B best site `L02.mlp_out` corridor 0.270 with identity donor −4.41 and patch-effect DiD −0.25 (draw the identity bar clipped, annotate); `L03.mlp_out` 0.044. Egg reference bar `L07.attn_out` 0.001. Selection line at 0.30, control ceiling at 0.10. Source: `drive_mech/arm{A,B}_seed0/patch_step0/patch_results.json` → `site_results[site_id, group gripper_corridor].directions.hazard_safe_to_unsafe.{donor,identity,null_factor,random,sham,main_effect_only,time_shift,wrong_group}.recovery_region_median`, `hazard_unsafe_to_safe.donor`, `selection.band`; egg from `heldout_v1_merged_w1/mech_w1_patch_step0_results.json`.
- *Inset:* Jacobian S_F bars per arm and action dimension — arm A 1.24 / 0.77, arm B 1.31 / 0.97 vs random 0.41 / 0.32, 0.40 / 0.43 and sign control 0.98 / 0.70, 0.89 / 0.66; egg 0.16 vs random 0.32. Source: `drive_mech/arm{A,B}_seed0/jacobian_sonar/jacobian_sonar.json` → `pooled["0"].{a0,a1}.{S_F_route,S_F_route_random,S_F_route_sign}.point` (n = 8 discovery scenes); Table R5.

---

## Figure 4 — Cross-arm geometry (driving cell, seed-0 pair; registered tests null — shown as a result, not omitted)

**Purpose.** Show the registered prediction and its outcome side by side: the appearance subspace is shared only partly (transported cosine 0.68 at `L03.mlp_out` vs within-arm 1.00 and random 0.08; equivalence margin failed at 0/18 sites), the relational double dissociation is null (|DD| ≤ 0.02 everywhere; DD 0.001 at the causal site, p 0.98), identity-swapped transport follows appearance as often as consequence (0–1/18 sites consistent), and every conceptor is hard rank 1 — the subspace-level face of template capture. Source for all panels: `drive_geometry/seed0/cross_arm_map.json` (53 scenes, 18 sites, step 0). Seeds 1–2 maps are pending; add them as faint replicates when they land.

**Layout (three panels).**
- *(a) Principal angles per site* between Arm A and Arm B interaction subspaces (k = 4, LOSO), separately for the appearance contrast (H1 vs H0 main effect at hazard tokens) and the relational contrast (hazard × action DiD at route tokens), with the matched-random subspace angle distribution as a grey band.
  Source: `registered_tests.i_appearance_shared["s0|hazard"][site].{app_ped,app_cone}.principal_angles_deg.{full_fit_transported,full_fit_raw}` (e.g. `L00.attn_out` transported 28°/51°/74°/83° vs raw 60°/65°/75°/…) and `.A_to_B/.B_to_A.{transported,within,random}.median` (L03.mlp_out: 0.68 / 1.00 / 0.08); relational DD per site from `ranked_tables["s0|corridor"][*].{dd_mean,dd_t,dd_p_maxt}` (top: `L00.attn_out` 0.018, p 0.001; `L03.mlp_out` 0.001, p 0.98).
- *(b) Procrustes-transported projected cosine* (A→B and B→A) vs matched-random control, for appearance and relational subspaces, per token identity (pedestrian tokens, cone tokens, route tokens). Expected pattern: appearance high in both directions; relational high only when the identity is solid in the target arm.
  Source: `registered_tests.iii_transport_identity_remap["s0|corridor"].sites[site].{A_ped_to_B_ped_minus_random,A_ped_to_B_cone_minus_random,B_cone_to_A_cone_minus_random,B_cone_to_A_ped_minus_random}` (L03.mlp_out: 0.28 / 0.11 / 0.06 / 0.23; `pattern_consistent` 0/18 corridor, 0/18 hazard, 1/18 hazard∪corridor). Observed pattern: A→B follows the pedestrian's appearance, B→A at corridor tokens follows the consequence — draw both so the asymmetry is visible.
- *(c) Conceptor cross-similarity matrix* (A-appearance, A-relational, B-appearance, B-relational) with the rank-one and matched-spectrum-random references, plus the cross-token Jacobian S_F per arm and identity as a side bar.
  Source: `registered_tests.iii_transport_identity_remap["s0|corridor"].sites[site].conceptor_cross_similarity.{B_rel_ped,B_rel_cone,matched_spectrum_random_vs_B_rel_cone}` (L03.mlp_out 0.15 / 0.12 vs random 0.015; L05.mlp_out 0.49 / 0.38 vs 0.019) and `iv_rank_quota[*].sites[site][A|B][*].{hard_rank 1, quota 0.0017–0.0032}`; Jacobian S_F side bar from Table R5 (`drive_mech/arm{A,B}_seed0/jacobian_sonar/jacobian_sonar.json`).

**Companion panel (d), optional.** Residual-cosine bars from Table R7 (model −0.18…+0.16 vs copy-delta 0.40–0.61; `drive_relational/relational_transport.json`) placed under (c) so that the rank-1 / null-DD result and the template-capture result are read together. The egg-family rotating-subspace fallback is no longer needed; the B-gate passed in both arms.

---

## Style notes

- One colour per substrate/arm across all figures (JEPA-WM, V-JEPA 2-AC, Arm A, Arm B); thresholds as dashed lines with the threshold value printed; hatched bars for placeholders.
- Every panel prints n (scenes) and the artifact filename in small type; per-scene points are shown where n ≤ 34.
- Use medians with scene-clustered bootstrap CIs as stored in the JSONs; never recompute CIs from prose.
- Figures 2–4 have complete data (driving: gate 6/6 models, mechanism and cross-arm on the seed-0 pair); the only remaining placeholder is the steered-planner endpoint (`artifacts/drive_coast_table/coast_table.json`, pending), which would enter Figure 2(d) as a fifth bar group or a small Figure 5.
