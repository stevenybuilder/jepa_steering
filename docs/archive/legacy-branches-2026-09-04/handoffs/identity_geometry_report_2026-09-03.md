# Identity-contrast geometry (METHOD fix for the seed-0 cross-arm null) — implementation report

Files (local `scripts/cgs_pilot/`, copied to `/root/cgs-pilot/code/cgs_pilot/`, md5 verified identical):
- `identity_contrast_geometry.py` (md5 f2da906c…) — analysis + `--self-test`
- `run_identity_geometry.sh` (md5 4afa0911…) — gated box runner (NOT launched; nothing existing modified, nothing killed)

## What the analysis does (per site x token group {corridor, hazard, hazard_corridor}, step 0)
Reuses `geometry_cross_arm.py` by import: `load_arm`, `align_arms`, `token_sets_for`, `load_site`, `token_rows`,
`matched_token_rows`, `transport_folds` (LOSO hazard-free Procrustes), `scene_rows` (DiD fields), `summ`/`boot_mean`,
`t_stat`, `quota_for_capture`; `geometry_localize.pool_site`; `cgs_stats.loso_projection_scores`/`sign_flip_maxt`; `stats_utils`.
1. Per-scene DiD fields D^X_ped = DiD(1), D^X_cone = DiD(3) per arm (DiD(l) = (h[l,1]-h[l,0]) - (h[0,1]-h[0,0]) on pooled tokens).
2. LOSO template removal: SHARED tau_{-i} = mean of all four (arm x identity) fields over the other scenes, arm B moved into
   arm A coordinates with the fold's transport (v_A = v_B W_i^T); PER-ARM variant tau^X_{-i} reported alongside. Residuals
   r^X_x,i = D^X_x,i - tau.
3. Identity contrast I^X_i = r^X_ped,i - r^X_cone,i. Honest note carried in the JSON/MD and `interpretation_scope`: because
   one template vector is subtracted from both identities, I is ALGEBRAICALLY template-invariant (I = D_ped - D_cone); the
   template matters for (c)/(d) and for the extra variant I_perp (template direction projected off, shared and per-arm),
   plus a "template-amplitude asymmetry" readout coef_ped - coef_cone per arm (predicted > 0 in A, < 0 in B).
   Tests on every variant {I, I_perp_shared, I_perp_perarm}, unit = scene:
   (a) within-arm LOSO direction consistency s_i = I_i . unit(mean_{j!=i} I_j): t, sign-flip p with the LOSO mean refitted
       under every flip, max-T over the group's sites (families per group x arm).
   (b1) per-scene cosine cos(I^A_i, I^B_i W_i^T): mean, scene-bootstrap CI, two-sided + one-sided(neg) sign-flip p, and a
       scene-permutation null (I^A_i vs I^B_pi(i)) whose null mean = shared-direction alignment, so obs - null_mean = the
       scene-specific part. NOTE: the ped<->cone relabel null (same flips both arms) leaves this cosine invariant (both vectors
       flip), so it cannot be applied to it — documented in the output.
   (b2) REGISTERED cross-arm statistic: held-out projection onto the other arm's LOSO identity direction,
       q_i = 0.5[cos(I^A_i, T_i mean_{j!=i} I^B_j) + cos(T_i I^B_i, mean_{j!=i} I^A_j)], one-sided q < 0, against the
       identity-relabel REFIT null (flips applied in both arms, all LOSO means refitted; computed exactly from packed Gram
       blocks), Westfall-Young max-T over the group's sites with shared flips (`cgs_stats.sign_flip_maxt`).
   (c) ||I_i||/||tau_{-i}|| median per arm (+ _perp), held-out and in-sample energy fraction of the four DiD fields along the
       template (mean direction and PC1), residual relative norms, template-amplitude asymmetry with CI, cos(DiD(H0'), tau).
   (d) rank/quota (80 % energy hard rank, conceptor quota, PC1 fraction) of raw fields, shared/per-arm residual fields, and I.
   Detection flags (descriptive): within = max-T p < 0.05; anti_alignment = relabel-refit max-T p < 0.05 AND cosine CI < 0.
Output: `identity_geometry.json` (entries per site x group, `families` with max-T blocks, `ranked_tables` sorted by the
anti-alignment t of I, `summary`, `transport_per_site`, `interpretation_scope` = "descriptive/associational;
template-projected; causal claims remain with patching. …") and `identity_geometry.md` (per-group tables).

## Self-test (`--self-test`, numpy only; synthetic dumps in the loaders' exact layout, arm B randomly rotated)
Planted: shared in-lane x throttle template (amp 3, both identities, both arms) + zero-mean shared appearance x action term
+ small identity component (amp 0.6, direction orthogonal to the template, on each arm's SOLID identity -> I^A ~ +u, I^B ~ -u).
Null: same without the identity component. 16 scenes, d = 64, 2 sites x 3 groups = 6 site-groups.
- Local (python3.9, numpy 2.0.2), seeds 0,1,2,3: PASSED every time. Planted: anti-alignment detected 6/6 on I and
  I_perp (cos -0.23…-0.46, q t -15…-24, p_maxT 0.001), within-arm 6/6 both arms (t 10–15), template dominates
  (fraction 0.92–0.96, median ||I||/||tau|| 0.25–0.35 — i.e. the identity component is ~1/4–1/3 of the template, the regime
  the real data are in). Null: 0/6 detected on all variants, |mean cos| <= 0.19 (positive: the shared appearance x action
  term), |within cos| <= 0.04; one family at p_maxT 0.065 (seed 0 corridor), none below 0.05.
- Box (`/opt/conda/bin/python`, numpy 2.2.6), niced CPU run in /tmp, ~20 s: SELF_TEST_PASSED (planted 6/6, null 0/6).
Scene-permutation diagnostic behaves as designed: planted null mean -0.27 (shared direction), obs - null ~ +0.04.

## Runner (`run_identity_geometry.sh`, `bash -n` OK locally and on the box)
Env as run_drive_arms.sh (JEPAWM_HOME/OSSCKPT/LOGS, PYTHONPATH=$REPO:$CODE, JEPAWM_DRIVING_ACTION_DIM=2, cd code/cgs_pilot).
0. Waits (poll 120 s) until `logs/drive_cross_arm.log` has CROSS_ARM_DONE AND `logs/panelb_gated.log` has
   PANELB_TOKEN_DONE|PANELB_TOKEN_INCOMPLETE AND df shows >= 8 GB free (liveness line every hour).
1. Preflight `--self-test` into `$IG/self_test/` (aborts on failure).
2. Re-dumps arm A and B seed 0 EXACTLY as the PAIRS stage: `localize_interaction.py --repo … --config $MODELS/arm{X}_seed0/eval_config.yaml
   --checkpoint …/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $EVAL/arm{X}_seed0 --output $IG/dump_arm{X}
   --domain driving --seeds-file $EVAL/arm{X}_seed0/discovery_seeds.txt --dump-hooks resid_post attn_out mlp_out --dump-groups egg corridor`,
   then the verbatim `slim_dump` step-0 slimming; disk re-checked before each dump; dumps reused if present.
3. `identity_contrast_geometry.py --dump-a $IG/dump_armA --dump-b $IG/dump_armB --stimulus artifacts/drive_factorial_merged/armA
   --discovery-seeds artifacts/drive_factorial_merged/discovery_seeds_common.txt --out $IG --step 0`.
4. Deletes both `dump_arm{A,B}/activations/`, prints a summary line, writes IDENTITY_GEOMETRY_DONE to `logs/drive_identity_geometry.log`.
Flags: `--skip-wait`, `--force`. Relaunch-safe (skips finished stages). $IG = /root/cgs-pilot/artifacts/drive_identity_geometry.

## Exact launch command (when you decide to start it)
ssh -n -p 20566 root@192.220.55.116 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_identity_geometry.sh > /root/cgs-pilot/logs/identity_geometry_driver.log 2>&1 < /dev/null &'
Monitor: tail -5 /root/cgs-pilot/logs/drive_identity_geometry.log ; result: artifacts/drive_identity_geometry/identity_geometry.md

## Runtime / disk estimate (box state at 15:58 UTC: 3.7 GB free; seed-2 mech arm A + run_coast_table waiting; Panel B token stage pending)
- Gate will open only after CROSS_ARM_DONE (seed-2 pair: several hours) and the Panel B token stage; the >= 8 GB condition
  needs the seed-2 dump/latent-cache deletion (PAIR_DONE frees ~2–3 GB) plus whatever Panel B releases.
- Dumps: 2.5 GB per arm before slimming, 0.84 GB after (18 sites x 424 cells x 106 tokens x 512 fp16); peak ~3.4 GB;
  ~5–10 min per arm (54 scenes x ~2.5 s/scene + model load + localization statistics), GPU ~2 GB.
- Analysis: 18 sites x 3 groups x 53–54 scenes, d = 512: 54 LOSO 512x512 Procrustes SVDs per site + Gram-based tests
  (2000 flips x 3 variants x 3 tests, shared flips) — estimated 5–15 min total (far lighter than the 175-min cross-arm map).
- Final footprint after cleanup: < 50 MB (JSON/MD, index.json, interaction_map.json). Total wall time after the gate: ~20–40 min.

## Caveats to keep in mind when reading the result
- I itself does not depend on the template removal (algebraic identity); the template-sensitive readouts are I_perp, the
  ||I||/||tau|| ratio, the residual rank, and the template-amplitude asymmetry — read them together.
- Anti-alignment is descriptive/associational; the causal identity claim stays with the L03.mlp_out donor-patch result.
