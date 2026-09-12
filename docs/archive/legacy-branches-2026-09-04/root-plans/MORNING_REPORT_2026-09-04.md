# Morning report — overnight run 2026-09-03/04

Written incrementally by the overnight session; every number below is re-derived from the JSON named next to it. Sections marked PENDING were still running when the session last updated this file (check the markers in `claude_handoff/HANDOFF.md` → "OVERNIGHT RUN").

## 0. Headline (fill last)

- [PENDING] v0.8 varied-geometry gate result.
- [PENDING] Panel B (action-as-token predictor) result.
- [PENDING] Steered-planner (COAST-style) table.

## 1. What you asked for, and what was done

| Ask | Done | Where |
|---|---|---|
| Verify no dummy data / improper JSON edits | **CLEAN.** 148/148 driving JSONs byte-identical to the box, every checkpoint/config SHA-256 matches the real files, no synthetic/placeholder/dry-run flags, medians recompute from per-scene values, 186/186 aggregator fields and 10/10 sampled table cells re-derive. One benign catch: the mechanism stage ran on 53 of 54 discovery scenes because an egg-domain calibration constant (`CALIBRATION_SEEDS = (101, 102)`) leaked into the driving code and dropped scene seed 101. | `claude_handoff/drive_integrity_audit_2026-09-03.md` |
| Cross-model design doc | `cross model design jepa.md` (VLA template applied: failure table F1–F18, claim ladder C0–C5 with current status, panels A/B/C + anchors + planted control, 8-step recipe, statistics, contamination controls, interpretation matrix, budget tiers) | project root |
| Behavioural design, clearly defined | `behavioral_design_v08.md`: hypothesis, sub-hypotheses with behavioural readouts, models and roles, factorial, endpoints with frozen thresholds, controls, interpretable outcomes, what is not claimed | project root |
| Cross-arm geometry result (you asked for a ping) | **Null at seed 0** (details §3) | `artifacts/cgs_pilot/drive_geometry/seed0/cross_arm_map.json` |
| Run the experiment end to end from the design | v0.8 stimulus generated and gated (§4); Panel B implemented and queued (§5); memorisation baseline added (§6); COAST table queued behind the chain (§7) | box jobs listed in HANDOFF |
| LeJEPA / LeWorldModel with video | Checked and ruled out as a panel (single CLS token per frame, per-environment checkpoints, needs ~10⁶ frames end-to-end); SIGReg/RDMReg kept as a tier-3 regulariser cell | `claude_handoff/panelB_feasibility_2026-09-03.md` |
| Paper placeholders | All mechanism / Jacobian / relational / cross-arm placeholders filled from JSONs with `<!-- src -->` comments; limitations (a)–(j) rewritten; only the steering row remains | `paper/cgs_workshop_draft.md` |

## 2. Corrections to numbers you may have seen in earlier notes (from the two JSON digests)

- Only `L03.mlp_out` carries the 72 % donor recovery (0.719 [0.715, 0.724], 53/53 scenes, reverse direction 0.41). `L03.attn_out` is in the band only by the contiguity rule and recovers 0.2 %.
- Arm B's "best 0.27 at L02.mlp_out" comes with a negative patch-effect DiD (−0.25) and an identity donor of −4.4: it is not hazard-specific recovery. "Distributed" is a hypothesis, not a result.
- The Jacobian sonar used 8 discovery **scenes** of model seed 0, not 8 seeds. S_F 1.24 / 0.77 (arm A steer / throttle) and 1.31 / 0.97 (arm B) vs random 0.32–0.43; the arm-A throttle dimension fails the sign-flip control; object insertion changes the Jacobian nearly as much as the hazard (cos 0.64–0.75).
- Residual cosine ≈ 0.1 holds only for arm A pedestrian; the other seven arm × field entries are −0.14 … +0.03. Copy-delta (0.41–0.61) beats the model in 16/16 entries.
- The identity-matched > swapped transport effect (0.08–0.16, p 2·10⁻⁴) holds only at hazard tokens; at the primary corridor site it is 0.02–0.07 and not significant.
- Every stimulus covariate in v0.7 is degenerate in range (ego speed 8.27–8.31 m/s), so covariate heterogeneity was untestable — one motivation for v0.8.

## 3. Cross-arm geometry, seed 0 (registered tests i–iv): NULL

53 discovery scenes, step 0, 18 sites × 3 token groups, LOSO subspaces, 175 min. Procrustes transport is informative (held-out relative residual 0.53–0.63 transported vs 1.1–1.2 raw), so the null is a real null:

| Test | Prediction | Result |
|---|---|---|
| (i) appearance subspace shared | transported ≈ within-arm | 0/18 sites; transported beats random (+0.1…+0.8) but is −0.15…−0.85 below the within-arm fit |
| (ii) double dissociation of relational subspaces | DD > 0 with max-T p < 0.05, A ped > cone, B cone > ped | |DD| ≤ 0.02 everywhere; 0/18 consistent in every group; one site per group at p < 0.05 with the wrong pattern |
| (iii) identity remap | matched transport > swapped, ghost ≈ random | transport beats random for matched **and** swapped identities (+0.1…+0.75); 0/18, 0/18, 1/18 consistent |
| (iv) rank / quota | ordered quotas | hard rank 1 at every site (quota ≈ 0.002); 0–2/18 |

Reading: the LOSO contrast subspaces capture the rank-1 consequence template shared by both arms and both identities; the identity attachment that is unmistakable behaviourally (cross-truth 12/12, T1c′) and causally (identity donor 0.02 vs real donor 0.72) is **not a distinct subspace at any single site**. This is consistent with the template-capture finding and pushes the identity question to the conditional/modulation operator and to the v0.8 stimulus. Seeds 1–2 maps: PENDING (chain).

## 4. v0.8 varied-geometry stimulus (the template-capture fix)

Generated: 8 / 10 m static (30 seeds each) → **59 admitted, 31 discovery**; 7 m fails `hazard_not_fully_in_frame` on every seed and 6 m fails the positioning gate on every seed (40° camera limits, not data problems); 8 m passed only after the mask-vs-projected-pose tolerance was widened 6 → 8 px (observed 6.1–6.2 px at 8 m; amendment recorded before any v0.8 model output; original validation kept); the knock-over body (40 seeds) fails the free-body invariance gates as generated (moves ~1e-5 m without contact vs a 1e-6 m tolerance; 32 cells move 0.48 m) → deferred, generator work needed.

Gate stage on the six existing models (v0.8 discovery scenes, cross-identity gate, T1c′, pooled training-copy baseline): PENDING — `logs/drive_v08_eval.log`, results `artifacts/drive_v08_eval/static/arm{X}_seed{s}/`.

| Model | captured solid (8 m / 10 m) | captured ghost (8 m / 10 m) | Rec_h1−Rec_h0 (vs H0′) | flip | T1c′ (scene-mean diff, p) | B-gate solid / ghost | training-copy DiD cos: model vs copy (p) |
|---|---|---|---|---|---|---|---|
| arm A seed 0 (ped solid) | 0.555 (0.62 / 0.36) | 0.648 (0.67 / **0.07**) | +0.249 (+0.196) | 0.871 | −0.082, 0.008 | PASS / PASS | ped 0.81 vs −0.25; cone 0.93 vs +0.19 |
| arm A seed 1 | 0.604 (0.65 / 0.59) | 0.686 (0.70 / **0.17**) | +0.316 (+0.246) | 0.903 | −0.124, 0.005 | PASS / PASS | 0.87 vs −0.25; 0.94 vs +0.19 |
| arm A seed 2 | 0.537 (0.59 / 0.34) | 0.633 (0.65 / **0.02**) | +0.218 (+0.175) | 0.871 | −0.087, 0.009 | PASS / PASS | 0.80 vs −0.25; 0.92 vs +0.19 |
| arm B seed 0 (cone solid) | 0.524 (0.58 / 0.52) | 0.565 (0.59 / **0.14**) | +0.247 (+0.157) | 0.871 | +0.008, 0.54 | PASS / PASS | cone 0.67 vs +0.03; ped 0.90 vs +0.32 |
| arm B seed 1 | 0.516 (0.56 / 0.52) | 0.417 (0.51 / **0.25**) | +0.271 (+0.181) | 0.871 | −0.022, 0.29 | PASS / PASS | 0.68 vs +0.03; 0.84 vs +0.32 |
| arm B seed 2 | 0.494 (0.60 / 0.49) | 0.527 (0.59 / **0.12**) | +0.210 (+0.144) | 0.871 | −0.020, 0.36 | PASS / PASS | 0.68 vs +0.03; 0.84 vs +0.32 |

n = 31 discovery scenes (19 at 8 m, 12 at 10 m; per-distance numbers are descriptive, the gate's n ≥ 16 holds only pooled). <!-- src: artifacts/cgs_pilot/drive_v08_eval/static/arm{A,B}_seed0/{behavior_gate_discovery.json, behavior_gate_discovery_level{3,1}.json, cf_gate_discovery*.json per_scene.interaction_nmse split at seed 1090, training_copy_baseline.json} -->

**Reading (all six models, 3/3 seeds per arm consistent).** Every model passes the gate on the new geometry (solid captured 0.49–0.60 pooled; specificity +0.21…+0.32 and not reversed by H0′; flip 0.87–0.90), so the consequence is not tuned to the 9 m training-typical scene. The within-arm solid-vs-ghost ordering that held pooled on v0.7 (6/6) is **absent pooled on v0.8** (ghost captured ≥ solid in 5/6), but it is distance-dependent: at 8 m the ghost's visual pass-through is captured as well as the solid consequence (0.51–0.70 vs 0.56–0.65), whereas at 10 m the ghost collapses in every model (0.02–0.25) while the solid identity holds (0.34–0.59). The identity-attached component is the part that survives at range; the close-range pass-through swamps it. The T1c′ scene-mean test favours the solid identity in arm A in 3/3 seeds (p 0.005–0.009) and is null in arm B in 3/3 (p 0.29–0.54) — the same arm asymmetry as the patch stage. The pooled training-copy baseline is far below the model in all 12 identity cells (copy cosine −0.25 … +0.32 vs model 0.67–0.94, p 2·10⁻⁴ everywhere; copy NMSE ≥ 0.9): memorised training futures do not explain the interaction. **Cross-truth on v0.8 (identity test independent of the within-arm ordering): the double dissociation replicates on the new geometry — 12/12 cross cells FAIL, 12/12 own cells PASS.** Arm-A models on arm-B physics: captured −2.13 / −2.07 / −2.11 (pedestrian) and −0.11 / −0.07 / −0.14 (cone); arm-B models on arm-A physics: −0.04 / −0.04 / −0.02 (pedestrian) and −0.89 / −0.85 / −0.77 (cone); specificity reversed (−0.02 … −0.39) in 11/12 (arm A seed 1 on B pedestrian −0.017, p 0.5); own-truth captured +0.42 … +0.69. <!-- src: artifacts/cgs_pilot/drive_v08_eval_crosstruth/summary.json -> table.model_arm{A,B}.truth_arm{B,A}.level{1,3}.per_seed -->

**Residual cosine on v0.8 (seed-0 pair, 31 scenes):** with one template per field the models now show clearly positive residual cosine (A ped +0.42, A cone +0.43, B cone +0.18, B ped +0.72 vs ≈ 0 on v0.7), i.e. they track scene-specific deviation — but a same-distance donor scene scores higher still (copy-delta residual 0.60–0.84) because the two-distance design makes the residual mostly a distance factor (true-field PC1 0.43–0.94; every transport 0.9+). The fair test is within-distance (template and donor from the same distance; `scripts/cgs_pilot/residual_by_distance.py`): **inside each distance group the model's residual cosine is zero** (−0.003 … +0.075 across 8 m / 10 m × arm × identity, every CI spanning 0) while a same-distance donor scene scores 0.12–0.49 (copy − model p ≤ 0.014 at 8 m, n.s. at 10 m with n = 12). So the models generalise the consequence across hazard distance — a coarse, physically meaningful scene factor — but reproduce **no finer scene-specific structure**: within a distance, "captured 50–65 %" is still template capture. C1 is therefore partially met (distance generalisation) and partially failed (within-distance specificity); a factorial that varies more than one continuous factor (speed, lateral offset) is needed to say more. <!-- src: artifacts/cgs_pilot/drive_v08_relational_slim/by_distance/residual_by_distance.json -> groups.hazard_corridor.{8m,10m,pooled}.fields.*.{model_residual, copy_residual, model_minus_copy_residual} --> <!-- src: artifacts/cgs_pilot/drive_v08_relational/relational_transport.json -> summary.hazard_corridor.{field,residual_cosine,transport} -->

## 5. Panel B — action-as-token predictor (cross-architecture replication)

Implemented and verified (`claude_handoff/panelB_report_2026-09-03.md`): `--pred-type {AdaLN,token,feature}` in the feature-space trainer using the vendor `VisionTransformerPredictorAC`; AdaLN path proven byte-identical; token/feature models load through the unchanged eval stack (0.0 difference); ≈ 8–10 min per model. Training clips regenerated with the identical command and seeds (1,598 clips/arm). Queued behind a disk gate (`run_panelb_gated.sh` → `run_panelb_token_arms.sh` → `run_panelb_eval.sh`): PENDING.

## 6. Contamination / memorisation / copying controls

- Nearest-training-clip audit (existing): ratio 1.11, 0.5 % beyond p95 — no near-duplicate scenes.
- NEW pooled training-copy baseline (`scripts/cgs_pilot/training_copy_baseline.py`): copies the nearest training clip's pooled future delta per cell and scores its DiD against the truth alongside the model; runs inside the v0.8 and Panel B gate stages. PENDING.
- Residual-cosine template control: model ≈ 0 vs copy-delta 0.4–0.6 on v0.7 (template capture); v0.8 re-run PENDING.

## 7. Steered-planner table (baseline vs sonar, VLA-style)

Rows: each model unsteered → steered (own-arm relational conceptor; arm-A operator transported into arm B) → suppression → controls (matched-spectrum random, rank-one, wrong-site, wrong-group, sham). Sealed confirmation scenes, zero refitting. Targets frozen 2026-09-02 22:55 UTC. PENDING (`run_coast_table.sh` waits for CROSS_ARM_DONE; `logs/drive_coast_table.log`).

## 8. Defects found tonight (all recorded in the design doc)

F15 egg calibration-seed leak (53 vs 54 scenes); F16 6–7 m out of frame; F17 cross-arm geometry identity-blind rank-1; F18 knock-over body fails invariance gates; localization alias defect (`hazard_corridor` vs `corridor`) leaves the live-band field empty; generator crashes on a non-finite centroid distance.

## 9. Next commands

```
ssh -p 20566 root@192.220.55.116
tail -5 /root/cgs-pilot/logs/drive_v08_eval.log /root/cgs-pilot/logs/panelb_gated.log /root/cgs-pilot/logs/panelb_eval.log /root/cgs-pilot/logs/drive_cross_arm.log /root/cgs-pilot/logs/drive_coast_table.log
```
