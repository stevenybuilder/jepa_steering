# Cross-Model Sonar Design — JEPA / Latent World Models

**Date frozen:** 2026-09-03 (draft for user review; becomes a preregistration when the user freezes it)
**Status:** design amendment written after the driving-cell discovery results (seed-0 pair mechanism stage, relational transport, all six B-gates, twelve cross-truth cells) were read, and before the sealed confirmation scenes were opened by the steered-planner table. Everything the driving cell has produced so far is **discovery/development evidence**. The cross-arm geometry map (seed 0) and the steered-planner table were still running when this was written; their outcomes were not viewed.
**Template:** `/Users/stevenyang/Documents/mechinterp-vla/cross model plan.md` (the VLA cross-model Sonar plan). Sections mirror it so the two projects share one recipe and one claim ladder; where the world-model setting differs (there is no policy, the "behaviour" is a predicted future and a planner ranking in the model's own currency), the difference is stated rather than papered over.
**Primary objective:** finish a valid Panel A test (our matched-consequence JEPA-WM driving cell) to a validated positive or a corrected null; then determine whether the same representation-geometry recipe transfers to a second JEPA architecture (action-as-token predictor) and to a non-JEPA latent world model on the identical stimuli.

## Execution status at freeze

Completed (all on discovery scenes unless stated; sources under `artifacts/cgs_pilot/`):

- substrates 1–2 (JEPA-WM DROID, V-JEPA 2-AC on the RoboCasa egg factorial) closed as structured nulls with full technique controls: captured interaction 4.5–6.5 % / 0 %, planner flip 0 %, best patch recovery 0.1 %, S_F 0.16 < random 0.32 (`LOOP1_SUMMARY.md`);
- protocol v0.7 matched-consequence driving cell (MetaDrive; arm A pedestrian-solid / cone-ghost, arm B the reverse; six depth-6 / width-512 AdaLN predictors on frozen DINOv3 features; three paired seeds; 120/120 factorial seeds admitted; 54 discovery / 66 sealed confirmation scenes);
- B-gate PASS_PLANNER 6/6 on the solid identity (captured 0.50–0.69, specificity +0.10…+0.23 at p 2·10⁻⁴, flip 0.89–0.93); solid < ghost NMSE in 6/6 (T1c′ p < 10⁻³ in 5/6); hazard-free loss equivalence 0.5–3.6 % (margin 5 %); nearest-training-clip audit passed (`drive_eval/`, `drive_models/equivalence.json`);
- cross-truth double dissociation: 12/12 cells FAIL on the other arm's physics; 3/3 seed pairs in the predicted direction (exact sign p 0.125, the floor at three pairs) (`drive_eval_crosstruth/`, `paper/results_tables.md` Table R2);
- seed-0 mechanism stage: arm A step-0 donor patch at `L03.mlp_out` on corridor tokens recovers 72 % of the pedestrian hazard gap (median 0.719, 53/53 scenes ≥ 0.3; reverse direction 0.41), identity (cone) donor 0.02, H0′ / random / sham ≈ 0, but wrong-group 0.61, time-shift 0.47, main-effect-only 0.33 (not token-group selective); the contiguity rule adds `L03.attn_out` to the band although it recovers 0.2 %; arm B: no site passes (largest single-site recovery 0.27 at `L02.mlp_out` comes with a negative patch-effect DiD and an identity donor that moves the prediction away, so it is not hazard-specific); cross-token action-Jacobian sonar on 8 discovery scenes of the seed-0 pair: relational S_F 0.77–1.31 vs random 0.32–0.43 in both arms (arm A throttle dimension fails the sign-flip control), object-insertion S_F comparable (cosine ΔJ_hazard vs ΔJ_object 0.64–0.75), E1 sites A: L00–L03, B: L00–L05 (`drive_mech/`; digest in the handoff);
- relational transport / residual cosine (seed-0 pair, 54 scenes): true DiD field PC1 0.54–0.67 at corridor tokens (participation ratio 2–3; spherical variance 0.02–0.05 vs 0.86 uniform), model fields even more collapsed (PC1 0.57–0.95); model residual cosine −0.14 … +0.16 (positive with CI excluding 0 only for arm A pedestrian) vs copy-delta 0.41–0.61, copy-delta better in 16/16 arm × field × site entries (p ≤ 4·10⁻⁴); cross-arm relational transport 0.43–0.90 with every null ≈ 0 (p 2·10⁻⁴); identity-matched beats swapped transport by 0.08–0.16 at hazard tokens (p 2·10⁻⁴) but only by 0.02–0.07 (n.s.) at the primary corridor site; stimulus covariates are degenerate in range (ego speed 8.27–8.31 m/s), so covariate heterogeneity is untestable (`drive_relational/`);
- frozen scripts for every stage (`scripts/cgs_pilot/`), stable hash split, sealed confirmation, one runner per stage;
- integrity audit of every driving artifact (2026-09-03, `claude_handoff/drive_integrity_audit_2026-09-03.md`): 148/148 JSONs byte-identical to the box, all checkpoint/config SHA-256 values match the real files, no synthetic/placeholder/dry-run flags, medians recompute from per-scene values, 186/186 aggregator fields re-derive — **CLEAN**.

Not completed:

- cross-arm registered tests (i)–(iv) on seeds 1–2 (seed 0 done 11:44 UTC: **null** — rank-1, identity-blind relational geometry in both arms; see C2 below);
- the steered-planner (COAST-style) table on sealed confirmation scenes (queued; first opening of confirmation seeds);
- any closed-loop rollout: every "behaviour" so far is an open-loop predicted future or a CEM ranking in the model's currency;
- a stimulus with scene-specific structure (hazard distance/speed/lateral offset varied) — the current stimulus is stereotyped by design (hazard 9.0 m ahead in every scene; 432/432 cells within 2 px of another scene's hazard centroid);
- a second predictor architecture on the same stimuli;
- a planted-mechanism positive control for the mechanism stage;
- edge/pathway localization (which attention heads move hazard-token content onto corridor tokens; AdaLN modulation as a locus).

Therefore **Panel A is not yet validated.** The gate stage is positive on discovery; the mechanism stage is positive for one arm (sufficient but not token-selective) and finds no single sufficient site in the other; the causal-repair stage has not run. This document specifies the work needed to finish Panel A without converting discovery results into confirmation after the fact, and the cross-model panels that follow.

## What the current results teach (failure analysis, in the template's spirit: ambiguity is not novelty)

Each item names the defect, the evidence, and the design change it forces.

| # | Defect | Evidence | Design change in this plan |
|---|---|---|---|
| F1 | **Template capture, not scene-specific interaction.** The B-gate's "captured fraction" is dominated by the shared direction of the DiD field. | true-field PC1 0.54–0.67, model-field PC1 up to 0.95; model residual cosine −0.14 … +0.16 while another scene's residual scores 0.41–0.61 (copy-delta better in 16/16, p ≤ 4·10⁻⁴); every stimulus covariate degenerate in range | Vary hazard distance (6–14 m), ego speed, lateral offset and hazard heading in the factorial so the field has reproducible scene-specific structure; add **residual-cosine and CCGP across held-out distance/speed conditions** to the B-gate as Tier 1c″ (magnitude criterion unchanged) |
| F2 | **Ghost identity passes within-arm.** A ghost in lane still produces a visual pass-through interaction the predictor reproduces (0.37–0.59). | `behavior_gate_discovery_level{3,1}.json` PASS_PLANNER in all six models | The within-arm gate is demoted to a *usability* check (C0); the identity claim rests only on cross-truth, T1c′, the progress-goal planner endpoint, and the geometry double dissociation. Add a **NONE** anchor (hazard removed) when budget allows so appearance-only capture is measured directly |
| F3 | **Brake-goal planner currency is identity-blind.** Flip rate is identical for solid and ghost identities and pedestrian-vs-cone flip is 0.0. | `T2a_flip_rate`, `identity_contrast.flip_rate_h1_vs_h3 = 0.0` | The progress-goal currency (already preregistered) is primary; the brake-goal rows are secondary; **closed-loop rollouts with the model's own CEM planner become the C4 endpoint** |
| F4 | **Copy-delta retrieval baseline is uninformative** on a stereotyped stimulus (0.92 > model 0.68–0.83). | `retrieval_baseline.json` | Score every retrieval baseline with residual cosine (template removed); add the nearest-training-clip *interaction-copy* baseline (requires storing per-clip future latents in the training reference — a code change, listed in deliverables); the varied factorial (F1) also fixes this |
| F5 | **Recovered-fraction specificity does not reverse in arm B** (immovable hazard: the collision future is the larger displacement). | `identity_contrast.rec_h1_minus_h3` +0.04…+0.11 in arm B | Knock-over hazard body variant (pedestrian displaced on contact) as a second consequence type; NMSE and planner-cost DiD stay the identity-sensitive readouts; disclose T1c′ amendment |
| F6 | **The arm-A band is not token-group selective** (wrong-group 0.61, time-shift 0.47, main-effect-only 0.33). | `drive_mech/armA_seed0/patch_step0/` | This is the template's "static state vs conditional transformation" question. Add **edge patching** (hazard-token → corridor-token attention paths at L02–L03), **AdaLN modulation patching** (shift/scale/gate per block are the only action entry), and the dynamical analysis across imagined steps (Δh_{ℓ,t}) before calling L03 "the site" |
| F7 | **Arm B has no single sufficient site** (best 0.27 with a negative patch-effect DiD; 0/53 scenes ≥ 0.3) while arm A is localized (0.72 at `L03.mlp_out`); "distributed" is the hypothesis, not yet a result. | `drive_mech/armB_seed0/patch_step0/` (selection.selected = null) | Report as an asymmetry, not hide it. Multi-site bands (L02–L03 attn+mlp jointly), conceptor and low-rank operator interventions that do not assume single-site sufficiency, and the same rule applied to both arms. The claim becomes "the sonar reads both a localized and a distributed instance" only if the distributed instance is *causally* read (C3 in arm B by a multi-site or donor-free operator) |
| F8 | **Small predictor, few actions.** Width 512 < 1024-d features; depth 6; two action chunks only; no unseen-action family. | `experiment_design.md` v0.7 tolerances | Width 1024 / depth 12 predictor if budget allows (cached features make this cheap); an unseen-action family (partial brake, swerve) in the factorial, held out from training, so CCGP over actions is testable |
| F9 | **Statistical resolution.** Three seed pairs give at best p 0.125; single-seed mechanism stage. | `seed_replication.paired_seed_sign_test` | Five paired seeds for the confirmatory arm contrast (exact sign floor 1/32); mechanism stage on ≥ 2 seeds per arm; report per-seed always |
| F10 | **Everything is open-loop.** The planner endpoint is a ranking in the model's currency; no rollout ever hit or avoided a hazard. | v0.7 COAST endpoint scope note | Closed-loop MetaDrive rollouts with the model's own CEM planner (DINO-WM/JEPA-WM style) are the C4 endpoint with simulator-labelled outcomes (collision / safe pass / timeout), matched across steering arms by seed |
| F11 | **T2b sign convention** in `behavior_gate.py` was inverted relative to `planner_currency.py`; verdicts unaffected. | HANDOFF 02:40 UTC | Disclosed; numeric identity test for every gate script becomes a required preflight (template's "numerical identity hook") |
| F12 | **Released checkpoints are not evaluable on driving** (7-D action interface). | v0.7 tolerances | Zero-support anchors stay on the egg; the driving zero-support contrast is the ghost identity within each arm plus the NONE anchor |
| F13 | **Localization is non-discriminative and mis-labelled on this cell.** Every step-0 site is at the max-T floor (p 1/2001), so only the magnitude ratio separates sites (19/31 arm A, 26/31 arm B corridor sites "live"); and the driving primary-group alias (`hazard_corridor` vs the dumped `corridor`) leaves the map's `earliest_live` field empty in both arms. | `drive_mech/*/localize/interaction_map.json` | Fix the alias before any confirmatory localization; report localization by magnitude ratio with a stricter frozen threshold (≥ 0.25 of the action effect) and by CCGP, not by p; treat the patch stage as the only site-selection evidence |
| F15 | **Egg-domain constant leaked into the driving mechanism stage.** `CALIBRATION_SEEDS = (101, 102)` in `localize_interaction.py` / `merge_heldout.py` silently dropped driving scene seed 101, so patching and localization ran on 53 of the 54 discovery scenes. | `drive_mech/*/patch_step0/patch_results.json` n_scenes 53 vs `drive_eval/*` 54 | Domain-scoped calibration-seed lists; the gate stage is unaffected; disclosed in the paper |
| F16 | **Stimulus geometry floor.** At 6 m the sidewalk/H0′ poses leave the 40° frame (positioning gate fails on 30/30 seeds; the generator also crashes on a non-finite centroid distance). | `artifacts/drive_factorial_v08/d06` | v0.8 distances are 7–10 m; a wider FOV or a lateral-offset factor is needed for closer hazards; the generator must serialize non-finite values as null |
| F17 | **Cross-arm subspace geometry is identity-blind and rank 1** (seed 0): the registered geometry prediction (appearance shared, relational identity-swapped) fails in both halves — the transported appearance subspace beats random but is far below the within-arm fit at every site, and the relational subspaces transport equally for matched and swapped identities. | `drive_geometry/seed0/cross_arm_map.json` registered_tests | The rank-1 template (F1) dominates every contrast subspace; identity attachment must be sought in the conditional/modulation operator (Step 4) or in multi-site bands, and the geometry tests must be re-run on the v0.8 stimulus where the template is weaker; report (i)–(iv) as null, not as "pending" |
| F18 | **Knock-over hazard body fails the invariance gates as generated** (free 70 kg body moves ~1e-5 m without contact against a 1e-6 m tolerance in 240/320 cells, 0.48 m in 32, and 162 contact-without-physical-effect cells). | `drive_factorial_v08/_failed_batches/dyn09` | Settle the body before the context frame, give free bodies their own displacement tolerance, and re-define "physical effect" for a displaced body; a generator change, not run tonight |
| F14 | **The Jacobian sonar's hazard direction is only partly distinct from generic object insertion.** | S_F_identity 0.67–1.24 vs S_F_object 0.71–1.37; cosine(ΔJ_hazard, ΔJ_object) 0.64–0.75; arm A throttle dimension fails the sign control; 8 scenes only | Run E1 on ≥ 24 scenes per arm and all imagined steps; add the object-insertion contrast as a registered control with its own threshold; the NONE anchor makes ΔJ_object a clean appearance baseline |

The overall reading, which the plan is built to test rather than assume: the recipe finds a relational hazard×action state when the consequence is in the training support, and the state is a **template-shaped, layer-localized-but-token-broadcast** object in one arm and a **distributed** object in the other. Whether that state is a coordinate, a field, or a conditional transformation of the predictor's update rule is the open question.

## Pre-outcome geometry amendment (tiers)

The method search is tiered so that flexibility cannot become outcome-driven optimization. This mirrors the VLA plan; the contents are the world-model analogues.

1. **Confirmatory core:** paired counterfactual DiD (2×2×2 identity × route × action, H0′ null), cross-truth contrast, residual-cosine template removal, LOSO localization with sign-flip max-T, on-manifold donor patching with the full control set, the cross-arm registered tests (i)–(iv), and the donor-free steered-planner table. All exist in code and are frozen.
2. **Triggered geometry extensions** (opened only by the frozen diagnostic in Step 5): contextual vector field (trigger: low PC1 with above-null relational transport — *not* fired on the current stimulus, PC1 0.54–0.67), circular / phase coordinate (trigger: stable phase ordering under 2-D embedding), local tangent model (trigger: local fits transfer better than global and neighbourhoods are bootstrap-stable), conditional low-rank operator on the AdaLN path (trigger: identity and route each readable but only their interaction predicts the consequence — the arm-A wrong-group result makes this the most likely trigger), dynamical analysis across imagined steps (trigger: state readable but static single-site intervention weak — fired for arm B).
3. **Exploratory follow-ups:** sparse / rectified-L_p JEPA predictor cell, CAFT-style denial fine-tuning, second-order (higher-moment) sonar. These may motivate a new preregistration; they cannot rescue Panel A.

The sharper central question:

> Is the hazard×action relational state in a JEPA predictor a stable state coordinate at a site, a context-dependent transformation field over scenes, or a dynamical selection rule — action modulation (AdaLN) gating which hazard-token evidence attention writes onto the route tokens across imagined steps?

## Label-first principle (BINDING; user decision 2026-09-03 ~18:00 UTC; governs every section below)

**Why this section exists.** Three days of work produced seven protocol versions and about eighteen instruments, each added after a null, because the first target was a four-way conjunction (identity × route × future risk × selective correction) measured through open-loop proxies (`claude_handoff/xie_lens_2026-09-03.md` §1; `vla project overnight failures.md`). A broad, ambiguous objective produces broad, ambiguous experiments. This section fixes the objective to one unambiguous label and forbids main-line work that does not produce or consume it.

**The label (COAST's construction; read from the paper itself, `references/COAST_arXiv_2605.17144.pdf`, Miao, Kim, Yang, Ungar 2026).** COAST extracts one mean-pooled activation vector per denoising/autoregressive step from the action expert's residual stream at a single layer, over N rollouts "where success and failure are determined either by simulation or by real world outcomes" (§3.2), and requires "no reward beyond binary success/failure labels used to partition rollouts" (§2). The conceptor is fit in closed form from those two sets and composed as C_steer = C_success ∧ ¬C_failure (eq. 4); steering is h′ = hMᵀ with M = (1−β)I + βC_steer (eq. 5). The mix of outcomes is obtained by *checkpoint choice*, not by analysis: "we use early fine-tuning checkpoints that produce the mix of successes and failures needed for contrastive fitting" for MetaWorld/LIBERO, while fully trained RoboCasa checkpoints "exhibit sufficient failure rates" (§4.1); robustness across checkpoints is in App. A.6. Protocol: "a strict train/test split: 15 training rollouts to extract conceptors and select hyperparameters, and 30 held-out test rollouts with entirely unseen environment configurations" (§4.1, App. A.8); real-robot success "defined strictly by task completion" (§4.1). Endpoint: held-out task success (π0.5 LIBERO-10 0.43 → 0.80; MetaWorld ML45 0.57 → 0.82 per-step). Stated limitations (§5): contrastive fitting needs both outcomes (uniformly successful tasks fall back to positive-only steering); tasks with nearly disjoint success/failure subspaces gain less; hyperparameter heuristics untested beyond the architectures used. The paper never assigns a label from activations, never inspects activations before labelling, and never claims a circuit. Our label is the same construction in our world:

- **Episode** = one factorial scene × hazard level × arm physics, rolled out closed loop in MetaDrive from the scene's context state by the arm's own frozen predictor under its own CEM planner (progress currency), ≥ 2 CEM seeds per scene. "Simulation outcome" in COAST's sense = the MetaDrive contact flag and goal/progress state, nothing derived from the model.
- ~~`safe_pass` / `collision` / `failure_progress` (goal region + 0.9 progress rule)~~ — withdrawn by amendment 1 below: the label is MetaDrive's native `arrive_dest` vs native failure predicates. Always-brake remains the negative benchmark.
- **Where it is produced:** `scripts/cgs_pilot/closed_loop_rollout.py` → `artifacts/drive_closed_loop/<arm>_seed<s>/rollouts.jsonl`. Nothing else is a label.
- **Split (COAST §4.1 transposed):** fitting rollouts from discovery scenes only; held-out evaluation on sealed confirmation scenes with entirely unseen scene seeds; hyperparameters (site by quota, aperture by overlap, β by sweep) chosen on fitting rollouts and frozen.

**Gate before any activation receives a label (C0 closed loop).** Unsteered `safe_pass` on the solid in-lane identity between 0.20 and 0.90 for the model, hazard-free goal success ≥ 0.50, and ≥ 30 successes and ≥ 30 failures per arm on discovery scenes. COAST obtained its mix by checkpoint choice (§4.1); if a model has no mix, the remedy is checkpoint choice (an earlier epoch of the same run), model scale or training-clip count (frozen retrain rule), never a new analysis. No geometry, conceptor, patch or steering result is fitted on a model that fails this gate.

**What the label fixes downstream.** Step 1 discovery geometry (outcome subspace: `safe_pass` vs `collision` states, pre-outcome only — see temporal precedence below); the faithful-COAST baseline row (C_safe ∧ ¬C_collision, one site by quota, β by sweep); the Sonar factorization C_rel / C_gen through the matched arms; and the steering endpoint (held-out `safe_pass`, with selectivity S = gain(solid in lane) − gain(ghost in lane) as the decisive statistic and hazard-free / H0 / H0′ / ghost cells as equivalence constraints). Open-loop quantities — captured interaction fraction, flip rate, CEM ranking, the steered-planner "safe choice" — remain B-gate diagnostics and calibration tools; they are not endpoints and cannot license a claim.

**Where COAST's label is ambiguous and what we add.** An outcome-only label "can capture task phase, scene identity, motor program, or generic competence" (`coast.md`, scope limits). The design removes those shortcuts by construction rather than by analysis: matched-consequence arms (identical pixels, actions and futures; only the physics differs), the ghost identity and H0′ cells inside every table, and temporal precedence (states strictly before the replanning index at which the executed chunk first diverges from the hazard-free rollout). These are the only additions to COAST's label logic; they are what make the identity claim possible.

**Anti-loop rules (binding).**
1. One label, one operator table, one figure (`xie_lens` §4). A null result changes stimulus scale or model scale; it never adds an instrument.
2. No geometry before the closed-loop gate. Any analysis on v0.7/v0.8 open-loop caches is *descriptive* and goes to the appendix.
3. Instrument freeze for the main line: closed-loop label, faithful COAST row, Sonar C_rel / transported / modulation / causal-metric rows, the CCGP ontology gate, the registered controls. Everything else (18-site localization grid, cross-arm tests i–iv, Jacobian sonar, higher-order sonar, relational transport, Panel C, planted mechanism) is second-stage and opened only by a pre-specified trigger.
4. Sample floors: fit ≥ 30 + 30 episodes per arm; held-out ≥ 200 episodes per cell; unit of inference = scene, CEM seed nested in scene.
5. Any deviation from this section requires a dated amendment entry appended here before the deviating run starts. Runners and reports must cite this section by name ("Label-first principle").

**Amendments:**
1. *2026-09-03 ~18:30 UTC (user): native predicates only.* COAST used LIBERO's native `is_success` flag, not a rule of its own (App. A.8: early checkpoint with 20–60 % success; tasks with fewer than 3 successes or 3 failures excluded). Our previous definition (goal region + progress ≥ 0.9 × hazard-free) was researcher-authored and is withdrawn from the label. The label is now MetaDrive's own termination predicates, with the env run under MetaDrive's native termination semantics (`crash_*_done`, `out_of_road_done` = True, native horizon) until the env's own `done`: **`success`** = `info["arrive_dest"]` (`TerminationState.SUCCESS`, `_is_arrive_destination`) with no crash flag earlier; **`failure`** = otherwise, with the native reason retained as a subtype (`crash_human` / `crash_object` / `crash_vehicle` / `crash_sidewalk` / `out_of_road` / `max_step`); success and crash on the same step count as failure. Subtypes do not define the representation classes; they are downstream interpretation variables (does the operator isolate hazard avoidance rather than generic non-completion). Native `route_completion` and the withdrawn progress/goal quantities are logged as `descriptive_*` fields only. Eligibility for fitting: ≥ 3 successes and ≥ 3 failures per model × level (COAST's rule), in addition to the C0 mix gate. Applies to every rollout from the second smoke test onward; the first smoke test is development data only.
2. *2026-09-03 ~18:40 UTC (user): simulator-configured values, not ours.* Every termination flag uses MetaDrive's shipped defaults (`crash_vehicle_done`, `crash_object_done`, `crash_human_done`, `out_of_road_done` = True; `arrive_dest` as defined by `_is_arrive_destination`: within ±5 m of the final lane's end and inside the lane band). MetaDrive's shipped `horizon` is `None` (no step cap), under which a stalled always-brake episode never terminates; a cap of 1000 env steps (the generator's `make_env` value and MetaDrive's own example convention) is therefore kept as an **engineering guard outside the label**: it is recorded as `horizon_guard` in provenance, any episode reaching it carries the native `max_step` subtype, and it cannot create a success. The hazard-free reference rollout's arrival step is logged per scene to show the guard sits far above arrival. This is the only non-native parameter in the closed-loop label and it is fixed here.
3. *2026-09-03 ~23:40 UTC (user): public checkpoint/benchmark pair becomes the PRIMARY panel; the self-trained driving cell becomes a deferred trained-substrate panel.* The closed-loop C0 screen failed for both seed-0 driving predictors under every planner configuration (report `claude_handoff/closed_loop_adapter_report_2026-09-03.md`, planner_diagnostics). Measured against the VLA plan's **native-label invariant** (`mechinterp-vla/cross model plan.md`, "Native-label invariant across every model and benchmark" and "Operating-point correction"), the driving cell fails on all three axes: self-trained checkpoints, self-generated stimuli/tasks, no pre-fit outcome screen. That plan's rule for a failed screen is adopted verbatim: *do not invent custom hazards or train more models; move to a predeclared public checkpoint/benchmark pair with naturally mixed outcomes and an official evaluator, or report.*
   **Panel P (primary from now).** Released JEPA-WM family checkpoints (`facebook/jepa-wms`; Terver, Yang, Ponce, Bardes, LeCun, arXiv:2512.24497): JEPA-WM and DINO-WM on PointMaze, Wall, Push-T, MetaWorld reach / reach-wall (V-JEPA-2-AC where released: DROID / RoboCasa). Evaluator: the repository's own planning evaluation (`configs/evals/simu_env_planning/<env>/<model>/*.yaml`, `succ_def: simu`) executed **unmodified**; label = that evaluator's success flag per episode; nothing authored by us. Their Table 2 rates are naturally mixed (JEPA-WM 41.6–83.9 %, DINO-WM 35.1–81.6 %), COAST's regime. Archive checkpoint URL + sha256, repo revision, env package versions, evaluator config hash, episode/goal seeds before inference.
   **Screen (C0, no activations):** per benchmark × model, the official protocol's episodes (96 per env, 3 seeds; their goal sampler); a family is eligible with 20–80 % success and ≥ 3 of each class per 15-episode fit cell; ≥ 2 eligible benchmarks per model to advance. Failure → report, never tune.
   **Then, unchanged:** fitting rollouts on development goals, one sealed held-out opening; table rows = unsteered, identity hook, faithful COAST (C_success ∧ ¬C_failure at one site by quota, β by sweep), Sonar operators (causal-metric min-distortion edit, AdaLN modulation operator, outcome-geometry conceptor with the registered controls), matched-spectrum sham. Endpoint = held-out official success rate; preservation = the evaluator's own secondary metrics. Cross-architecture recurrence = JEPA-WM vs DINO-WM (vs V-JEPA-2-AC) on identical tasks.
   **What is deferred:** the matched-consequence arms (identity attachment, C_rel/C_gen, cross-truth) are a *trained-substrate* panel opened only by a positive Panel P result; their open-loop results and today's second-stage analyses are reported as such. Panel B token models likewise. No further training or stimulus generation for the driving cell.

## Target hierarchy (added 2026-09-03 16:30 UTC after the VLA project's development null and tonight's v0.8 results)

The VLA project's overnight failure had one root cause that also appears here: the first target was a four-part conjunction (identity × route conflict × future risk × selective correction), so a failed behavioural gate and a failed representation hypothesis could not be told apart, and the model turned out to respond to encounter geometry rather than to the identity. Our v0.8 data show the same shape — the ghost identity's visual pass-through is captured as well as the solid consequence at 8 m; identity separates only at range or through cross-truth. The hierarchy below fixes the order of claims:

1. **Discovery label (coarse, unambiguous):** the simulator outcome under the model's own planner — collision versus safe pass, plus progress — closed loop in MetaDrive. Open-loop analogues (captured interaction fraction, CEM safe choice) stay as the B-gate, but the outcome label anchors geometry discovery (Step 1) and the C4 endpoint. The closed-loop CEM adapter therefore moves ahead of the remaining single-site mechanism work.
2. **Interpretation factors (second stage):** identity, route relation, distance/speed, action, appearance, template. Applied after outcome-geometry discovery to say what the geometry represents. Identity attachment is licensed by the matched arms and cross-truth (12/12 on v0.7 and v0.8), never by the within-arm ordering.
3. **Preservation outcomes:** hazard-free and off-lane behaviour, progress, ARC/drift — constraints, never averaged into the safety gain.

Faithful COAST (outcome conceptors from real closed-loop successes and failures) is the required baseline row; Sonar's contribution is the factorization into an identity-attached relational component, the pathway that carries it (edge / AdaLN-modulation patching), the smallest donor-free operator that changes the planner's choice selectively, and the recurrence of the functional recipe in Panel B. The headline hypothesis is therefore: *clear closed-loop outcome geometry in a JEPA predictor contains a separable, causally used relational component attached to the identity that carried the consequence in training, and that component can be selectively controlled and rediscovered across action-entry architectures.*

## Executive decision

The experiment will not treat the within-arm B-gate PASS as the finding. It will use (1) the matched-consequence arms and cross-truth contrast to identify what the interaction *means*, (2) a varied-geometry factorial so that "captured" can no longer be satisfied by a template, and (3) closed-loop, simulator-labelled outcomes under the model's own planner as the causal-repair endpoint.

Three evidential stages, as in the VLA plan:

1. **Outcome discovery:** find geometry that separates the model's own safe and unsafe planner choices (and, closed loop, collision vs safe-pass episodes).
2. **Relational identification:** test whether that geometry is `identity × route conflict × action` (the consequence), not appearance, in-lane presence, action main effect, or the shared template.
3. **Causal control:** steer the frozen predictor through that geometry, donor-free, and require safer held-out planner choices and rollouts with hazard-free and off-lane behaviour preserved.

The advance over a plain COAST replication is unchanged from the VLA plan: COAST asks whether success-associated geometry improves completion; Sonar asks whether a model-native relational computation can be discovered, factorized from generic response, localized along the predictor's pathway (token groups × layers × imagined steps × modulation), and selectively controlled.

## What is and is not novel

Ambiguity is not novelty. The novel hypothesis is:

> Across JEPA-family and non-JEPA latent world models trained with the consequence in support, a hazard×action relational computation can be found by first discovering low-dimensional outcome geometry in the predictor's imagined future, then factorizing it with matched physical counterfactuals (reversed-assignment arms, H0′ null, varied scene geometry) into an identity-attached relational component, and finally intervening on the smallest causal subspace or pathway that changes the planner's choice while preserving hazard-free prediction.

The cross-model contribution is a repeatable recipe, not a claim that every model uses the same layer, token, basis, or rank. The reversed-assignment arms are the part that has no VLA analogue: they let "training created it" be claimed, which no released checkpoint can support.

## Claim ladder

Each claim is licensed only if all earlier claims pass. Levels map to the VLA plan's C0–C5 and to the project's older L0–L5 ladder (`LOOP1_SUMMARY.md`).

| Level | Claim | Required evidence | Current status (Panel A, discovery) |
|---|---|---|---|
| C0 | The model and stimulus are behaviourally usable. | B-gate Tier 1+2 on the solid identity; hazard-free loss equivalence across arms; nearest-training-clip audit; closed-loop behaviour gate (unsteered safe-pass rate between 20 % and 90 % on solid in-lane; hazard-free goal success ≥ 50 %). | Open-loop part PASS 6/6; closed-loop part not run. |
| C1 | A compact outcome subspace exists beyond the template. | Held-out residual-cosine > copy-delta and > scene-blind ridge; CCGP across held-out distance/speed/offset conditions above label-permuted dichotomies; quota ≤ 8 at 80 % energy. | v0.7: FAIL (residual cosine ≈ 0 < copy-delta 0.40–0.58). v0.8 (8 + 10 m, seed-0 pair): **partial** — pooled residual cosine +0.18 … +0.72 (the models track the distance factor) but within each distance it is ≈ 0 (−0.003 … +0.075) vs same-distance copy-delta 0.12–0.49: coarse generalisation across distance, no finer scene-specific structure. |
| C2 | Part of that subspace is relational and identity-attached. | Cross-truth double dissociation (12/12 FAIL, 3/3 pairs); T1c′ solid < ghost; cross-arm tests (ii) DD with refit-permutation max-T p < 0.05 and (iii) identity-matched > swapped transport at the primary site; Jacobian S_F > random, > sign control, and > object-insertion control. | Behavioural part PASS; **geometry part FAIL at seed 0**: test (i) 0/18 sites, (ii) |DD| ≤ 0.02 and 0/18 consistent, (iii) transport beats random for matched and swapped identities alike (0–1/18 consistent), (iv) hard rank 1 everywhere; transport itself informative (held-out residual 0.53–0.63). Relational-transport identity effect only at hazard tokens; Jacobian object-insertion control not separated (F14). The identity attachment is real at the behaviour and patch level but is not a distinct LOSO subspace at any single site. |
| C3 | The representation is causally used. | Donor patch ≥ 30 % with controls < 10 % *including wrong-group and time-shift*, or a multi-site / donor-free operator meeting the same rule; bidirectional; dose-responsive; reproduced on confirmation with zero refitting. | Arm A: sufficiency PASS (72 % at `L03.mlp_out`, both directions) but selectivity FAIL (wrong-group 0.61); arm B: FAIL at every single site (no positive patch-effect DiD above 0.02). Both need the pathway/operator stage. |
| C4 | The consequence can be selectively repaired or suppressed. | Steered-planner table targets on sealed confirmation (arm B +0.25 with CI excluding 0; suppression on A ≤ −0.25; sidewalk/H0′/hazard-free within ±0.10; all controls within ±0.10); closed-loop: ≥ +10 pp safe-pass on solid in-lane, hazard-free success within 5 pp, no always-brake artefact. | Not run. |
| C5 | The recipe transfers. | C0–C4 pass in a second predictor architecture (Panel B) and reported for a non-JEPA control (Panel C) without changing the registered selection rule. | Not run. |

A failure at any level is a valid result and stops stronger claims. Under this ladder the current honest statement is: **C0 passes, C2 passes at the behaviour level, C1 fails for a stimulus reason, C3 is half-passed with a selectivity caveat, C4–C5 are unrun.**

## Panels, checkpoints, stimuli, and evidence roles

| Panel | Model (frozen after training; hashes recorded before inference) | Stimuli | Role |
|---|---|---|---|
| A | **JEPA-WM-style AdaLN predictor** on frozen DINOv3 ViT-L/16 features (`facebook/dinov3-vitl16-pretrain-lvd1689m`, converted file SHA-256 `99115699…1dc02`); current six models depth 6 / width 512 (checkpoint SHA-256 per model in each gate JSON, e.g. arm A seed 0 `7fcfc9b4…4ef8ea`); confirmatory retrain at width 1024 / depth 12 if budget allows, else the current models with the new factorial | MetaDrive 0.4.3 reversed-assignment arms A/B; v0.8 varied-geometry factorial (below) | Primary validation; must finish first. |
| B | **Action-as-token predictor** — `jepa-wms` `src/models/ac_predictor.py::VisionTransformerPredictorAC` (the class behind the authors' V-JEPA-2-AC(fixed) baseline, arXiv:2512.24497), `action_conditioning="token"` (one action token + one state token per frame, block-causal mask, RoPE), depth 6 / width 512 to match Panel A, then depth 12 / width 1024; a third variant `action_conditioning="feature"` (action concatenated to every patch token) gives the JEPA-WMs three-mechanism series (AdaLN / token / feature) at ≈ 1 GPU-h per model. Trained on the *same* cached DINOv3 features and the *same* clip multiset, both arms, ≥ 3 paired seeds (research memo 2026-09-03: `claude_handoff/panelB_feasibility_2026-09-03.md`) | Identical | Cross-architecture JEPA replication: same encoder tokens, different action entry. Tests whether "broadcast at L03" is an AdaLN artefact (H1: wrong-group recovery < 0.10 in the token model vs 0.61 in AdaLN). |
| C | **Non-JEPA latent world model** on the same features: RSSM / DreamerV3-style recurrent predictor (MILE-like; CARLA MILE `2210.07729` is the fallback if a feature-space RSSM is out of budget), both arms | Identical | Architecture-transfer control: is the relational state JEPA-specific? |
| Anchors | Released **JEPA-WM DROID** (`github.com/facebookresearch/jepa-wms`, predictor epoch 315) and **V-JEPA 2-AC** (DROID) — inference only on the egg factorial (`LOOP1_SUMMARY.md` App. A–D) | RoboCasa egg (retired stimulus) | Zero-support anchors; never used for a "training created it" claim (7-D action interface prevents driving use). |
| Calibration | **Planted-mechanism positive control:** one extra arm-A seed trained with a synthetic backdoor (a texture trigger that suppresses the pedestrian consequence, BadDreamer-style) *or* a predictor with a planted rank-1 relational code (the existing planted-code self-test in `geometry_cross_arm.py` extended to the patch stage) | Identical | Implementation positive control for the mechanism stage: the frozen pipeline must recover a mechanism whose location is known. Not safety evidence. |

Repository revisions and file hashes to archive before any confirmatory inference: `jepa-wms` (vendor patch `vendor_patches/0001-driving-data-stats.patch`), `vjepa2`, MetaDrive 0.4.3, `scripts/cgs_pilot/` (md5 manifest, as done on 2026-09-01), training-shard manifests, factorial-seed manifests, split hashes. The current runs record checkpoint and config SHA-256 in every gate/patch JSON; the confirmatory run adds the environment digest.

## Why these panels

**Panel A.** It is the only cell where the consequence is in support by construction and where a training-origin claim is licensed (matched arms). Its defects are stimulus defects (F1, F4, F5) and scale defects (F8, F9), all repairable at small cost because features are cached and predictors train in about 20 minutes.

**Panel B.** The action-token predictor enters the action as a token, so hazard-token → route-token relational state can be formed without AdaLN broadcast. (LeWorldModel / the LeJEPA line was checked and is not a viable panel: its predictor consumes one CLS vector per frame so the token-group sonar cannot run, its action entry is AdaLN like Panel A, its released checkpoints are per-environment (Push-T, Cube, TwoRoom, Reacher), and it is trained end-to-end on millions of frames; SIGReg / rectified-L_p regularisers remain a tier-3 cell on our own predictor. No open action-conditioned JEPA-family driving world model with low-dimensional ego-action weights exists as of 2026-09-03.) If L03 broadcast (F6) is an artefact of modulation, Panel B should show token-group-selective patching or a different locus; if it is a property of the computation, it should recur. Same encoder, same clips, same stimuli — a clean architecture contrast.

**Panel C.** A recurrent latent model has a different notion of "site" (the deterministic state h and stochastic s at each rollout step). The recipe's adapter changes; the estimand, gates, and statistics do not. Failure here is informative (JEPA-specific), not a defect.

**Anchors and calibration.** The egg cells stay as the zero-support anchors. The planted-mechanism arm addresses the one gap the egg loop exposed: every null so far was "no behaviour, so no mechanism"; we have never shown the mechanism stage recovers a known mechanism in this substrate. The VLA plan's COAST-on-π0.5 reproduction plays the same role.

## The shared Sonar recipe (world-model adaptation)

The algorithmic recipe is frozen across panels; only the architecture adapter (hook stack: `predictor_hooks.py` for AdaLN, a token adapter for Panel B, an RSSM adapter for Panel C) changes.

### Step 1: collect real outcome labels

Two label sources, kept separate:

- **Open-loop (existing):** per scene, the model's CEM ranking of brake vs throttle under the progress-goal currency; safe choice = brake preferred when the solid identity is in lane. This is the model's own behaviour and remains Tier 2 of the B-gate.
- **Closed-loop (new, C0/C4):** run the unsteered predictor with its own CEM planner in MetaDrive from each factorial scene's context state; label from the simulator: `safe_pass` (goal reached, no contact, progress ≥ 0.9 of the hazard-free rollout), `collision` (contact flag), `failure_progress` (stopped/timeout without contact). Keep `collision` and `failure_progress` separate; an always-brake policy is a negative benchmark, and clearance gained by not progressing is a failure.

For every eligible scene, retain the pre-outcome causal record: activations at every site (attn_out, mlp_out, resid_post, and the AdaLN shift/scale/gate vectors per block), per token group (hazard / corridor / background), per imagined step, per replanning index in closed loop; the predicted latents; the CEM cost vector; the executed action and the next simulator state. Truncate discovery activations before the first avoidance or contact response (in closed loop: the replanning step at which the chosen action first diverges from the hazard-free rollout).

### Step 2: paired physical counterfactuals (v0.8 varied-geometry factorial)

Everything that made v0.7 valid stays (bit-exact replay, identical context frame across cells, H0′ matched displacement, kind-paired hazards, equal future multisets across arms, validator gates). The factorial adds within-scene variation so that the DiD field has scene-specific structure:

| Factor | Levels | Purpose |
|---|---|---|
| Identity | pedestrian, envelope-matched cone | reversed assignment across arms |
| Route relation | in lane, sidewalk (H0), H0′ mirror pose, NONE (removed; optional anchor) | relational DiD, appearance control |
| Action | A0 brake, A1 throttle, **A2 partial brake, A3 swerve** (A2/A3 held out from training: unseen-action family) | action generalization (CCGP over actions) |
| Hazard distance | 6, 8, 10, 12, 14 m (TTC 0.7–1.7 s at 8.3 m/s) | scene-specific structure; distance-held-out CCGP |
| Ego speed | 6, 8.3, 10 m/s | separates TTC from distance |
| Lateral offset in lane | −0.5, 0, +0.5 m | route-corridor signed distance as a coordinate hypothesis |
| Hazard body | immovable (current), **knock-over** (displaced on contact) | F5; second consequence type |

Contact remains permitted only in (solid identity, in lane, throttle) cells, now at a distance-dependent future step (validator checks the contact step per scene). Target 240 factorial seeds → ≥ 60 discovery, ≥ 60 confirmation after gates, stratified by distance.

The estimand is unchanged: the arm × identity × action three-way interaction on the predicted-latent interaction and on the planner ranking; the primary activation-level object is the paired DiD vector at hazard ∪ corridor tokens, and the relational DiD subtracts H0′.

Registered coordinate hypotheses, tested in this order (the first three are shortcuts):

1. hazard absolute / image position;
2. hazard pixel size and screen displacement;
3. hazard in-lane flag (presence);
4. signed distance to the ego corridor;
5. time to route intersection under the candidate action (TTC × action);
6. `identity × route conflict × action` interaction;
7. imagined-step phase (when in the rollout the consequence is committed).

A relational interpretation requires candidates 4–6 to transfer across held-out distances, speeds, and offsets after the shortcuts are controlled.

### Step 3: faithful COAST baseline

At every predictor site, collect hazard ∪ corridor token states from actual closed-loop `safe_pass` and `collision` rollouts (unsteered), compute the conceptor C = R(R + α⁻²I)⁻¹ on centred activations, compose C_outcome = C_safe ∧ ¬C_collision, and apply the COAST gate M = (1−β)I + βC_outcome. Mean pooling is retained only for this baseline. The existing conceptor arm (fit on residualized DiD vectors, AND-NOT against action/null conceptors) is **Sonar**, not COAST, and is reported as such; the current steered-planner table's "own-arm relational conceptor" row is Sonar. A faithful COAST row is added.

Fitting rules as in the VLA plan: shrinkage covariance, dual/SVD when width exceeds sample count, eigenvalue clipping, nested selection of aperture and rank, and no Gaussianity claim.

### Step 4: factor generic outcome geometry into relational geometry

Nuisance subspace P_N from: scene identity, hazard pixel position/size/displacement, distance and speed, the action main effect (hazard-free action delta) and a low-rank generic dynamics basis, imagined-step index, and the **discovery-field mean template** (the residual-cosine correction from `LRH.md`, now applied before any subspace fit, not only as a metric). Residualize H_res = (I − P_N)H.

Nuisance-whitened metric ⟨x,y⟩_N = xᵀ(Σ_N + λI)⁻¹y estimated on fitting scenes from independently varied non-relational factors (distance, speed, offset in hazard-free and off-lane cells), and the pullback metric G_ℓ = J_{ℓ→Y}ᵀ W_Y J_{ℓ→Y} + ρ(Σ_N + λI)⁻¹ where Y = (predicted-latent interaction energy, CEM cost gap). Report in both the whitened and the raw basis; neither is called causal.

Four factor conceptors in the residualized space, C_ped, C_cone, C_inlane, C_offlane, and the relational candidate C_rel = (C_solid ∧ C_inlane) ∧ ¬(C_ghost ∨ C_offlane), with "solid/ghost" resolved per arm. The symbolic form is not evidence; it must beat the mean-difference direction and matched controls.

Conditional-transformation candidates (the world-model-specific part):

- bilinear s(h) = uᵀz_hazard · vᵀz_corridor (E4, already implemented; AUROC 0.5 on the egg, to be re-run here);
- **modulation operator:** h′ = h[I + g(a, h_hazard) U diag(γ) Vᵀ] where g is read from the AdaLN (shift, scale, gate) vectors — the only place the action enters — so that "the action gates which hazard evidence is written" is a testable operator rather than a metaphor;
- contextual displacement field φ_i over scenes with PC1 ratio and spherical dispersion (implemented in `relational_transport.py`); on the current stimulus PC1 ≈ 0.6 licenses a common direction; the field extension fires only if the varied factorial lowers PC1 while relational transport stays above its nulls.

### Step 5: model-native geometry tournament

Nested selection: ontology first (Step 2 hypotheses) by **CCGP** — train the readout on some distance × speed × offset × action conditions and test on whole conditions held out — plus the **parallelism score** of matched DiD vectors across conditions. Shattering dimensionality (decodability of random balanced dichotomies at matched prevalence) is the control; advancement requires relational CCGP beyond it.

Then, per site × token group × imagined step, compare the registered geometry classes: mean-difference direction; generalized-eigenvector / low-rank subspace; soft conceptor; circular coordinate; local tangent; conditional low-rank (modulation) operator. Escalation triggers are the tier-2 list above and are frozen.

Selection score = minimum of: held-out planner-choice discrimination; held-out relational DiD discrimination after nuisance removal; CCGP across held-out distance/speed/action conditions; relational-transport score across arms; worst-scene transfer; advantage over the strongest shortcut baseline (pixel position, in-lane flag, action main effect, template). No geometry advances on in-sample accuracy.

### Step 6: eigenspectrum as sonar

Per conceptor: quota q(C) = tr(C)/d, effective rank, explained variance, condition number, scene-bootstrap stability; use spectrum peaks to nominate a narrow band and keep one wrong-layer control on each side (the current L02/L04 ≈ 0 result is exactly this control for L03).

Basis-invariant comparisons across arms, seeds, imagined steps, and panels: principal angles / Grassmann distance; conceptor overlap tr(C_A C_B)/√(tr C_A² tr C_B²); CKA on paired-DiD Gram matrices; residual cosine; relational transport with permutation and covariance-matched Gaussian nulls (implemented). These distinguish "similar relations" from "same subspace" from "same mechanism"; cross-panel alignment stays descriptive until each panel passes C3.

Structured nulls for every geometry statistic: scene-pair permutation within arm; identity relabelling with refit (test ii); magnitude-matched isotropic Gaussian; spectrum-matched random orientation; scrambled counterfactual pairing; shortcut-matched baselines. Sliced-W2 / energy distance between paired activation distributions is reported as a diagnostic only.

### Step 7: localize the causal pathway

Sweep the complete model-specific causal chain, not only the final site:

| Panel | Required loci and time axes |
|---|---|
| A (AdaLN predictor) | encoder tokens (frozen; sanity only); every block's attn_out / mlp_out / resid_post at hazard, corridor, background tokens; **AdaLN shift/scale/gate vectors per block**; imagined steps 0–2; closed-loop replanning index. |
| B (action-token predictor) | the action token's residual stream per block; cross-attention from action token to hazard/corridor tokens; same token groups and steps. |
| C (RSSM) | deterministic h_t, stochastic s_t, prior vs posterior transition inputs, action-conditioned prior at each rollout step. |

After a candidate locus is found (currently L03 in arm A), run the edge/pathway tests that F6 demands:

- source-state removal (zero/mean-ablate hazard tokens at L02 and observe L03 corridor recovery);
- **source-to-destination patching:** patch only the attention path hazard-tokens → corridor-tokens at L03 (donor keys/values from the H1 scene, receiver queries from H0) and measure recovery; compare with patching the modulation vectors alone;
- destination-state intervention (the current donor patch);
- writer analysis: which heads / MLP add energy along C_rel at corridor tokens, cross-fitted;
- mediation: an upstream edit must change downstream relational energy before it changes the CEM cost gap.

Dynamical analysis (fired for arm B by the distributed result, and for arm A by the time-shift 0.47): for each paired scene, the trajectory Δh_{ℓ,t} = h_{ℓ,t+1} − h_{ℓ,t} across depth and imagined step; local Jacobian-vector products from the upstream relational subspace to downstream relational energy, predicted-latent interaction, and CEM cost; singular modes separated into input-selection (which hazard/route perturbations enter), evolution (where the state moves across imagined steps), and persistent (what survives to step 2 and to the next replan). The cross-token action-Jacobian sonar (E1) is the existing first instance of this; it is extended from step 0 to all steps and to the modulation path.

Desired ordering: the relational coordinate becomes explicit before the peak generic action-effect layer. A late action-only hit supports repair, not an upstream mechanism claim.

### Step 8: donor-free causal steering

Arms under identical scene seeds, open-loop (CEM ranking on sealed confirmation scenes) and closed-loop (rollouts):

1. unsteered frozen predictor;
2. numerical identity hook (must reproduce unsteered outputs bit-for-bit);
3. faithful COAST (closed-loop outcome conceptor);
4. Sonar relational conceptor, own arm;
5. Sonar relational conceptor transported from the other arm via the cross-arm Procrustes map (the cross-arm transfer row);
6. single-direction (CAA-style) edit along the LOSO direction;
7. Sonar modulation operator (conditional low-rank on the AdaLN path);
8. matched-spectrum random conceptor; 9. rank-one; 10. reversed / sign-flipped; 11. wrong-layer; 12. wrong-token-group; 13. wrong-step; 14. sham β = 0;
15. identity-only ungated edit (appearance subspace); 16. route-only ungated edit.

Smallest development-selected dose that clears the frozen rule (smallest β whose steered − unsteered safe-choice DiD CI excludes 0 on discovery). No donor activation, donor action, nearest stored trajectory, or future frame is available at inference. Minimum-distortion constraint: maximize Δ relational energy subject to ‖δ‖_N ≤ ε, D_action ≤ τ (change in the hazard-free action subspace and in the pullback metric), D_local ≤ τ_d (Mahalanobis to the receiver distribution, already reported as 0.81 for the L03 donor). Bidirectionality is mandatory: suppression in the solid arm must lower safe choice on the solid identity and leave the ghost identity and off-lane cells unchanged; strengthening in the ghost arm (transported operator) must raise it. Monotone dose-response over a frozen grid {0.25, 0.5, 1.0}; doses that "improve safety" by collapsing the action effect (ARC, drift, hazard-free progress) are rejected.

## Primary outcomes

**Open-loop (Panel A first pass, already preregistered):** planner safe-choice rate per hazard level under the progress-goal currency on sealed confirmation scenes; targets as frozen on 2026-09-02 22:55 UTC (arm A unsteered ≥ 0.75 on pedestrian-in-lane; arm B unsteered → steered ≥ +0.25 with scene-clustered CI excluding 0; suppression on A ≤ −0.25; sidewalk / H0′ / hazard-free within ±0.10; controls within ±0.10).

**Closed-loop (new primary for C4):** paired safe-pass rate on solid-identity in-lane scenes, Pr(goal reached without contact and with progress ≥ 0.9 of the hazard-free rollout). Secondary: collision rate; minimum time-indexed hazard clearance; time to first divergence from the hazard-free plan; goal success and progress ratio on hazard-free and off-lane scenes; stop/timeout rate; steered-vs-unsteered action divergence; subspace quota, rank, principal angles, overlap, downstream relational energy change.

Hazard-free, off-lane, and ghost-identity behaviour are **preservation constraints** (two one-sided equivalence tests, margin 0.10 on rates, 5 pp on success, progress ratio ≥ 0.95), never averaged into a single score.

## Statistical design and power

**Unit of inference.** Scene for within-arm statistics; paired training seed for the arm contrast; closed-loop episode seed nested in scene. Frames, tokens, imagined steps, and replans are repeated measures.

**Fitting and development.** Discovery scenes only for layer, geometry, rank, aperture, dose, and ontology selection, in nested cross-fitting (nuisance and whitening learned without the scored pair; ontology in an outer split; geometry, locus, rank, aperture, dose inside the outer training fold). Minimum 15 closed-loop unsteered rollouts per scene family and ≥ 3 `safe_pass` and ≥ 3 `collision` episodes per conceptor cell; otherwise pool within a preregistered family or declare the cell ineligible. Repeat identical scenes across CEM sampling seeds to estimate a behavioural noise ceiling; representation scores are read against that ceiling.

**Confirmation.** ≥ 60 sealed scenes stratified by distance; ≥ 30 paired closed-loop episodes per primary arm in aggregate, ≥ 2 episode seeds per scene; five paired training seeds for the arm contrast (exact sign floor 1/32; three pairs, as now, are reported as a pilot with p 0.125 at best); mechanism and steering on ≥ 2 seeds per arm. Seeds, maps, initial states, hazard paths, and physics parameters identical across steering arms.

**Tests.** (1) paired scene randomization test for safe-choice / safe-pass improvement; (2) scene-clustered bootstrap CI for the absolute effect; (3) clustered three-way DiD (arm × identity × action) with the sign-flip null; (4) TOST equivalence for preservation; (5) hierarchical multiplicity: behaviour, then representation, then intervention; within the tournament one joint max-T over site × step × group × geometry class × rank; triggered extensions join the same family; (6) worst-scene and leave-one-scene-out stability alongside pooled means; (7) refit-permutation nulls wherever a subspace was fitted on the labels being tested (test ii); (8) block permutation over whole episodes.

**Frozen minimum practical effects** (targets, not substitutes for CIs): open-loop table targets above; closed loop ≥ +10 pp safe-pass on solid in-lane, ≥ 20 % relative collision reduction, hazard-free success within 5 pp, progress ratio ≥ 0.95, no larger effect on the ghost identity than on the solid one, primary operator beats every matched control at the same dose budget.

## Contamination, memorization, and trajectory-copying controls

**Training data and checkpoint.** All predictors are trained by us on generated clips; the encoder is a released checkpoint whose training data (LVD-1689M) is documented but not auditable at the example level — state "no documented evaluation overlap", not "contamination-free". Split by scene seed with a fixed hash (identical across arms); factorial seeds disjoint from training seeds; the v0.8 factorial's held-out distances/speeds/actions are, by construction, outside the training grid where the action family is concerned.

**Memorization audit.** Hash every training shard and factorial seed; nearest-training-clip audit on DINOv3 pooled latents (done: ratio 1.11, 0.5 % beyond p95); extend to a per-clip stored future so the **interaction-copy baseline** (copy the nearest training clip's future delta) exists — the current training reference stores pooled latents only, which is the one memorization control we could not run; reject near duplicates at thresholds fixed from discovery; repeat the primary effect on perturbed textures, weather, and start states.

**Trajectory-copying audit.** No future frame or realized future latent is a model input; residual-cosine scoring of every retrieval baseline (F4); the crossed test — the same hazard path paired with a different ego speed, and the same ego state paired with a relocated hazard — must be answered correctly by the steered planner; success required on held-out geometry outside the convex hull of training distances where the validator permits; steered rollouts compared with all training clips by DTW and endpoint/action cosine.

**Preventing global braking from masquerading as safety.** Hazard-free and off-lane equivalence is mandatory; goal completion, progress ratio, stop rate, and timeout rate are co-primary preservation checks; an always-brake planner is the negative benchmark; clearance gained solely by failing to progress is a failure.

## Panel A validation: required completion sequence

The existing driving results are development evidence. They are not confirmatory because: every number is from the 54 discovery scenes; the L03 band was selected on those scenes; the stimulus is stereotyped (C1 untestable); the planner endpoint is open-loop; and the mechanism stage ran on one seed pair.

### A-1: implementation positive controls

- Numerical identity hook for every intervention path (patch, conceptor, modulation operator): bit-identical predicted latents and CEM costs at β = 0 / identity donor. (Currently: sham 0.000 and identity donor 0.02 — the 0.02 must be explained or driven to 0 before confirmation.)
- Planted-mechanism arm (Calibration row): the frozen pipeline must localize and steer the planted relational code at the known site with the registered rules. Failure means the mechanism stage is invalid and downstream steering stops.
- Gate-script identity tests (F11): every gate re-run on a synthetic factorial with known DiD; sign conventions asserted.

### A-2: stimulus and behaviour gate (v0.8)

Generate the varied-geometry factorial and the knock-over variant; validator and replay gates unchanged; run the unsteered models (current six, or retrained) open- and closed-loop. The gate passes if: B-gate Tier 1+2 on the solid identity in ≥ 2/3 paired seeds; **residual cosine of the model > copy-delta residual** (C1 becomes testable); closed-loop unsteered safe-pass on solid in-lane between 20 % and 90 %; hazard-free goal success ≥ 50 %; ≥ 12 usable scene families across distances contain both in-lane and off-lane variants. If C1 still fails on the varied stimulus, the finding is "template-only capture at this scale" and the mechanism stage does not proceed to confirmation.

### A-3: fit, freeze, and test

1. Collect unsteered open- and closed-loop labels and the full activation/pathway grid on discovery scenes.
2. Fit faithful COAST and Sonar geometry on discovery only; run the tournament and the triggered extensions whose diagnostics fire.
3. Run pathway localization (edge, modulation, dynamical) on discovery; freeze one band/operator/dose per arm per method.
4. Serialize operators, preprocessing, thresholds, hashes, and confirmation scene IDs.
5. Open the sealed confirmation scenes once: donor patch reproduction (band rule, zero refitting), the steered-planner table, closed-loop rollouts with all controls and preservation arms.
6. Produce clustered inference and the C0–C5 table.

Terminal outcomes: **validated positive** (C0–C4 pass) or **corrected null** (a gate fails with its prespecified interpretation). The current steered-planner run, which opens confirmation before v0.8 exists, is reported as the v0.7 endpoint; the v0.8 confirmation set is generated fresh (new seeds), so v0.7's opened scenes are never reused as confirmation.

## Cross-model replication sequence

Replication results must not influence the frozen Panel A analysis. Three kinds of transfer, reported separately:

1. **Functional transfer:** the same registered discovery-and-intervention rule produces C3–C4 in Panel B (and is reported for C).
2. **Geometric transfer:** paired physical states have similar relational geometry across panels under CKA, CCGP, parallelism, principal angles, or relational transport after panel-specific fitting.
3. **Implementation transfer:** homologous components (a layer, a head, the modulation path) perform the same computation. Not expected; never claimed from alignment alone.

**Panel B (action-token JEPA predictor).** Train on the cached DINOv3 features and the identical clip multiset, both arms, ≥ 3 paired seeds; same P0–P2 gates (loss equivalence, B-gate, audit); same tournament and pathway sweep with the action-token adapter; same steered-planner and closed-loop endpoints. The specific question: does token-group selectivity appear when the action is a token rather than a broadcast modulation?

**Panel C (RSSM / non-JEPA).** Same features and clips; RSSM adapter; the "site" axis becomes (h_t, s_t, prior input) × rollout step. The specific question: does a recurrent latent model build the relational state as a persistent-mode object (dynamical) rather than a static site?

Cross-panel Procrustes / CCA and relational-transport maps are fitted on shared hazard-free rows only and evaluated on held-out physical conditions. No raw vector or conceptor is copied between differently parameterized models; the arm-A → arm-B transported operator is legitimate only within a panel (same architecture).

## Baselines and current numbers

| Result | Unsteered / model | Control / target | Evidential status |
|---|---:|---:|---|
| Egg, JEPA-WM DROID captured interaction (n = 21/34/66) | 4.5 / 6.5 / 5.7 % | B-gate ≥ 25 % | zero-support anchor; B-gate FAIL |
| Egg, V-JEPA 2-AC captured (n = 34) | 0.0 % | ≥ 25 % | anchor; FAIL |
| Egg best route-token patch recovery / S_F | 0.1 % / 0.16 | ≥ 30 % / random 0.32 | technique controls |
| Driving arm A / B captured, solid identity (3 seeds each) | 0.66 ± 0.03 / 0.53 ± 0.02 | ≥ 25 % | discovery; PASS 6/6 |
| Driving ghost identity within arm | 0.57 ± 0.02 / 0.41 ± 0.05 | — | discovery; passes (F2) |
| Cross-truth captured (all 12 cells) | −2.3 … −0.02 | > 0 | discovery; FAIL 12/12 (predicted) |
| Planner flip rate H1 vs H0 | 0.89–0.93 | ≥ 0.25 | discovery; identity-blind under brake goal (F3) |
| Arm A `L03.mlp_out` donor recovery / identity donor / wrong-group | 0.72 / 0.02 / 0.61 | ≥ 0.30 / < 0.10 / < 0.10 | discovery, seed 0; sufficiency PASS, selectivity FAIL |
| Arm B best single-site recovery (patch-effect DiD) | 0.27 (−0.25) | ≥ 0.30 (> 0) | discovery, seed 0; no band |
| Jacobian S_F route, A / B (steer, throttle) | 1.24, 0.77 / 1.31, 0.97 | random 0.32–0.43; sign 0.66–0.98 | discovery, 8 scenes, seed 0 |
| DiD-field PC1 (true); model residual cosine; copy-delta residual | 0.54–0.67; −0.14 … +0.16; 0.41–0.61 | — | discovery; C1 FAIL for a stimulus reason (F1) |
| Cross-arm relational transport; identity-matched − swapped (corridor / hazard tokens) | 0.43–0.90; 0.02–0.07 n.s. / 0.08–0.16 p 2·10⁻⁴ | nulls ≈ 0 | discovery |
| Steered-planner table | running | targets frozen 2026-09-02 | first confirmation opening |

The current numbers justify running the confirmatory experiment. They do not count as its result.

## Interpretation matrix

| Outcome | Conclusion |
|---|---|
| v0.8 B-gate fails on the varied stimulus | The models capture only the stereotyped template; no relational claim beyond "consequence in support changes the predicted future". |
| B-gate passes but residual cosine / CCGP fail | Scene-local template capture; C1 fails; mechanism stage stays discovery-only. |
| C1 passes, cross-arm test (ii) fails | Interaction geometry is shared across arms — appearance- or in-lane-driven, not identity-attached. |
| Single-site patch sufficient but not token-selective (current arm A) and edge/modulation tests localize the selectivity | The state is a conditional transformation on the modulation path, not a token-local coordinate; C3 passes via the pathway operator. |
| Single-site fails but multi-site / dynamical operator passes (expected for arm B) | Distributed, persistent-mode implementation; C3 passes via the operator; report the asymmetry between arms. |
| Geometry predicts but no operator steers | Correlate, redundancy, wrong locus, or planner disuse; no C4. |
| Steering moves the ghost identity as much as the solid one | Generic in-lane / braking repair, not consequence-sensitive. |
| Steering helps in-lane but harms hazard-free or off-lane | Entangled with the action main effect or miscalibrated dose; no selective-repair claim. |
| Open-loop table passes, closed loop fails | The planner's currency is not what the rollout uses; report the dissociation. |
| Panel A passes, Panel B fails | Architecture-specific (modulation-dependent) mechanism; the universal recipe is unsupported. |
| Panels A and B pass, C fails | JEPA-specific relational code; recipe transfers within the family only. |
| ≥ 2 panels pass C0–C4 | Evidence for a reusable geometry-to-causal-control recipe for latent world models. |

## Execution order, budget, and stop rules

**Enhanced order (2026-09-03 17:00 UTC; two boxes). Supersedes the numbered list of the frozen draft; incorporates `vla project overnight failures.md` §2 and the Target hierarchy above.** The change in one line: the closed-loop simulator outcome under the model's own planner becomes the first thing measured, the instrument set is frozen to the COAST-faithful baseline plus the matched-arm identity test, and the remaining geometry analyses move behind that endpoint instead of in front of it.

Box 1 (Vast 49155754, registered v0.7 chain, untouched): cross-arm seeds 1–2 → `CROSS_ARM_DONE` → COAST-style steered-planner table on sealed confirmation (`COAST_TABLE_DONE`). Recorded as the v0.7 endpoint; the confirmation scenes are not reopened.

**Panel P (released checkpoints + official evaluators; amendment 3) now precedes everything below on box 2:** env setup → official evaluator screen (JEPA-WM, DINO-WM × PointMaze/Wall/Push-T/MetaWorld) → eligibility → fitting rollouts with activation capture → operator table → one held-out opening. Runner `run_public_panel.sh`, report `claude_handoff/public_panel_report_2026-09-04.md`.

Box 2 (Vast 49766237 "cgs-pilot-2"; `/root/box2_orchestrator.sh`, `/root/run_v09_gen_box2.sh`):

*Step 1 below is the gate of the Label-first principle; steps 2–6 consume its labels or are descriptive until it passes.*

1. **Closed-loop CEM adapter** in MetaDrive (deterministic prefix replay to the context state; unsteered predictor with its own CEM planner under the progress currency; simulator labels `safe_pass` / `collision` / `failure_progress`, progress relative to the hazard-free rollout) and its smoke test; then the closed-loop C0 gate for the current six models on discovery scenes. This is the discovery label of the Target hierarchy; no further single-site mechanism work precedes it.
2. **Identity-contrast geometry** (`identity_contrast_geometry.py`, seed-0 pair): second-stage factorization, descriptive.
3. **v0.9 randomised-factor factorial** (distance 8–10.5 m, prefix throttle, lateral offset; 120 seeds) → validator → gates and training-copy baseline for the six AdaLN models (C1 retest with scene-specific structure). If C1 still fails at width 512 with the B-gate passing, the frozen retrain rule applies (width 1024, five paired seeds).
4. **Panel B** arm-B token models (arm A reused) → Panel B gates, cross-truth, v0.8 gates (first C5 look).
5. **Faithful COAST row** from real closed-loop successes and failures (Step 3), then the steered-planner and closed-loop steering table with preservation margins (Step 8): discovery first, one confirmation opening. Only after 1 has produced labels.
6. Pathway localization (edge / AdaLN) and the planted-mechanism positive control follow 1–5; they do not precede them.
7. Panel C only after Panel B has a verdict. Videos only for an intervention that passes the selective closed-loop repair gate.

**Registered additions (user decision 2026-09-03 ~17:40 UTC; each grounded in `mechinterp-vla/representational_geometry_paper_concepts.md`, implementation in progress):**
- (a) **Causal-metric minimum-distortion edit** (§1 causal inner product; Step 8): nuisance covariance Σ_N from hazard-free cells, action main effect and the v0.9 randomised factors; edit maximises relational energy under ‖δ‖_N ≤ ε, D_action ≤ τ, Mahalanobis ≤ τ_d; Euclidean-metric twin as the control that shows the metric matters. `causal_metric_steer.py`, mode `min_distortion` in the steering table.
- (b) **AdaLN modulation operator** (§11 selection vector, §12.5 dynamical intervention): h′ = h[I + g(a, h_hazard) U diag(γ) Vᵀ] on the modulation path, rank r ∈ {1, 2, 4}; ungated twin, random U/V, wrong block, sham. `modulation_operator.py`.
- (c) **CCGP / parallelism / shattering ontology gate** (§10; Step 5): relational dichotomies must generalise across held-out distance × prefix × lateral × action conditions beyond the shattering-dimensionality control and the permutation null before any operator is fitted on v0.9. `geometry_ccgp.py`.
- (d) **Temporal precedence in the closed-loop record** (§14 "outcome-conditioned geometry may be downstream"): every rollout stores the replanning index at which the executed chunk first diverges from the hazard-free rollout; discovery geometry is fitted only on strictly earlier states. In `closed_loop_rollout.py`.
- (e) **CAFT-inspired pair** (§5): *arm diffing* — PCA of Δh = h_A − h_B on identical inputs as the candidate-direction route, compared with the relational conceptor and the L03 donor direction by trace overlap and principal angles (`arm_diff_pca.py`); *denial retraining* — fine-tune a trained arm on the same cached features with (I − UUᵀ) applied at the discovered site in forward and backward passes, evaluate with the projection removed: consequence re-routed (sufficient, not necessary) vs collapsed (necessary), with matched-random-subspace, continued-training and wrong-layer controls (`denial_finetune.py`). Necessity/robustness evidence only; cannot count as C3 on its own.

Budget tiers: **minimal** (current models, v0.8 factorial, closed-loop adapter, pathway tests, one confirmation): ≈ 10–15 GPU-h; **standard** (+ retrain at width 1024 with five paired seeds, + Panel B reduced): ≈ 40–60 GPU-h; **full** (+ Panel B full depth, Panel C): ≈ 100 GPU-h. Any overrun is reported.

Stop rules: a failed gate stops stronger claims; no loop may inspect confirmation labels and reuse those scenes; a failed loop may motivate a new hypothesis only with a new preregistration and fresh seeds; the deadline (workshop 2026-09-05) governs what is *reported*, never what is *claimed*.

**Execution checklist for the registered additions (kept current; a row is DONE only when its report file exists and its numbers were re-derived from the JSON):**

| item | module | consumed by | marker / output | status 2026-09-03 18:50 UTC |
|---|---|---|---|---|
| (a) causal-metric min-distortion edit | `causal_metric_steer.py` (mode `min_distortion` in `steered_planner_ranking.py`) | steering table rows 7 + Euclidean twin; closed-loop steering | `claude_handoff/steering_operators_report_2026-09-03.md` | code + self-test PASS; fit on v0.7 seed-0: causal-metric direction == Euclidean direction at 36/36 (A) and 27/36 (B) cells (rank-1 target → the metric cannot matter on this stimulus); arm-A open-loop steering table on 12 scenes is saturated (indicator at ceiling) and the continuous margin moves for random/twin controls as much as for the main edit → no discriminating power on v0.7; rerun on 53 scenes + v0.9 dumps queued |
| (b) AdaLN modulation operator | `modulation_operator.py` | steering table row 6 + ungated/random/wrong-block controls | same report | code done (41 KB), runtime steerer + hook; self-test PASS locally + box 2 (CPU); real-data run waits for seed-0 dumps |
| (c) CCGP / PS / SD ontology gate | `geometry_ccgp.py`, `run_ccgp.sh` | gate before any operator is fitted on v0.9 | `claude_handoff/ccgp_report_2026-09-03.md`, `<out>/ccgp.json` | code + self-test PASS; v0.9 arm A step 0 DONE 22:25 UTC: relational dichotomies CCGP 1.00 vs SD 0.64–0.70, PASS 54/54 — but inlane/action/identity also 1.00 (appearance-level; abstraction w.r.t. distance/prefix/lateral certified, consequence-vs-appearance needs the cross-arm contrast); factor dichotomies weak (dist 0.94, prefix 0.89, lateral 0.78); DiD parallelism ≥ 0.8, solid/ghost DiD cos 0.67–0.92 (template). v0.7 + arm B + steps 1–2 running (~10 h) |
| (d) temporal precedence | `closed_loop_rollout.py` (`divergence_replan_idx`, `truncation_mask`, per-replan latents) | Step 1 outcome geometry, COAST row, C_rel | `artifacts/drive_closed_loop/**/rollouts.jsonl` | adapter built + native-label smoke run; C0 hazard-free FAIL for both seed-0 models under the registered currency; planner diagnostics hazard-free only; amendment 3 pending |
| (e1) arm diffing | `arm_diff_pca.py` | candidate directions vs conceptor / donor direction | `artifacts/drive_arm_diff/`, `claude_handoff/caft_inspired_report_2026-09-03.md` | code done, self-test PASS (local + box); `run_arm_diff.sh` waiting for seed-0 dumps (marker ARM_DIFF_DONE) |
| (e2) denial retraining | `denial_finetune.py`, `run_denial.sh` | necessity / re-routing readout (not C3) | `artifacts/drive_denial/` | code done, self-test PASS (grad check ~3e-8; identity hook bit-exact); `run_denial.sh` waiting for dumps + shards + GPU rule (marker DENIAL_DONE); arm-A features recomputed from shards with bit-identity check |
| label (amendments 1–2) | `closed_loop_bridge.py` (native termination override), `closed_loop_rollout.py` | everything above | `CLOSED_LOOP_SMOKE_DONE` under native label, `CLOSED_LOOP_C0_SEED0_DONE`, `_V07_DONE`, `_V09_DONE` | queued after the native-label smoke |

## Deliverables

- immutable run manifest (hashes, split assignments, factorial seed lists, training shard manifests);
- architecture adapters behind one hook interface (AdaLN, action-token, RSSM);
- planted-mechanism positive-control report and identity-hook tests;
- v0.8 stimulus generator, validator, and physics/visual QC report (distance-dependent contact step, knock-over variant);
- closed-loop CEM planner adapter for MetaDrive with simulator outcome labels;
- training reference with per-clip futures (interaction-copy baseline);
- geometry report: spectra, quota, rank, principal angles, overlap, CKA, residual cosine, relational transport, structured nulls, shortcut baselines;
- coordinate report: CCGP across distance/speed/action, parallelism, shattering control, whitened distances;
- pathway report: edge patching, modulation patching, writer/reader ranking, mediation, dynamical modes across imagined steps;
- causal report: dose-response, bidirectionality, reverse/random/wrong-site/wrong-group/wrong-step controls, preservation equivalence;
- open-loop steered-planner table and closed-loop outcome table with clustered CIs;
- final C0–C5 table per panel, with the arm asymmetry (localized vs distributed) stated as a result.

## Primary references

- JEPA-WMs, arXiv:2512.24497; V-JEPA 2 / V-JEPA 2-AC, arXiv:2506.09985; DINO-WM, arXiv:2411.04983; LeWorldModel, arXiv:2603.19312; MILE, arXiv:2210.07729; MetaDrive, arXiv:2109.12674.
- COAST conceptors, arXiv:2605.17144; recovered fraction for counterfactual validity, arXiv:2608.11601; Shi et al. CoCo/ARC/drift, arXiv:2608.04653; divergent representations, arXiv:2511.04638; mediation variance, arXiv:2510.00845; counterfactual quotient models, arXiv:2608.22092.
- Othello-GPT model-native representations, arXiv:2309.00941; linear representation hypothesis, arXiv:2311.03658; shared relational geometry and contextual transformation fields, arXiv:2607.04525 (`LRH.md`); geometry of abstraction (CCGP, parallelism, shattering), Bernardi et al. 2020; context-dependent computation by recurrent dynamics, Mante et al. 2013; Action Atlas, arXiv:2603.19233; BadDreamer, arXiv:2606.21172; Rectified L_p JEPA, arXiv:2602.01456; CAFT, arXiv:2507.16795.
- Project records: `experiment_design.md` (v0.3–v0.8 addenda), `LOOP1_SUMMARY.md`, `vla_learnings.md`, `LRH.md`, `DRIVING_PIVOT_RESEARCH.md`, `DRIVING_SUBSTRATE_PLAN.md`, `claude_handoff/HANDOFF.md`, `paper/results_tables.md`.
