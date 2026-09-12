# Behavioural experiment design — driving hazard × action cell (protocol v0.7 → v0.8)

Written 2026-09-03 (overnight session) as the plain-language companion to `cross model design jepa.md` and `experiment_design.md` (v0.7 addenda). It states what is being tested, in what models, on what stimuli, with which numbers, and what each outcome would mean — before any representational-geometry method is applied. Every threshold here is the one already frozen in the scripts (`behavior_gate.py`, `counterfactual_validity_gate.py`, `steered_planner_ranking.py`) unless marked NEW.

## 1. Hypothesis in one sentence

> A JEPA world model builds an internal *hazard × action* relational state — "this hazard, on my path, under this action, produces this consequence" — only when the consequence is in the support of its training futures; the state is attached to the identity that carried the consequence, changes the predicted future and the planner's choice, and its training origin can be identified because two models trained on identical images and actions differ only in which identity the consequence attaches to.

Sub-hypotheses, each with a behavioural readout before any activation is read:

| ID | Sub-hypothesis | Behavioural readout | Predicted direction |
|---|---|---|---|
| H-support | The model reproduces the consequence of its own arm's physics | captured interaction fraction 1 − NMSE (Tier 1b) | ≥ 0.25 (PASS) on the solid identity |
| H-identity | The consequence is attached to the solid identity, not to "something in lane" | solid vs ghost identity: NMSE ordering (T1c′), planner-cost DiD sign, cross-truth FAIL | solid < ghost in ≥ 2/3 seeds; every cross-truth cell FAIL |
| H-relational | The state is hazard × action, not hazard appearance and not the action main effect | Rec_h1 − Rec_h0 > 0 and not reversed vs H0′ (T1c); DiD after subtracting H0′ | > 0 at p < 0.05 |
| H-planner | The state is used by the model's own planner | CEM brake/throttle flip rate H1 vs H0, not H0′ (T2a); progress-goal safe choice | flip ≥ 0.25; safe choice rises on solid in-lane only |
| H-origin | Training created it (not the encoder, not the stimulus) | double dissociation across the paired arms | DD < 0 in every seed pair |
| H-generalisation (NEW, v0.8) | The model predicts the *scene-specific* consequence, not a template | residual cosine after template removal > copy-delta; captured fraction stable across held-out distances | model residual > copy-delta residual; CCGP across distance ≥ within-distance − 0.1 |
| H-causal-use (v0.8 closed loop) | Editing the state changes what the planner does | closed-loop safe-pass rate under the model's CEM planner, steered vs unsteered | ≥ +10 pp on solid in-lane, hazard-free within 5 pp |

## 2. Models under test and their roles ("what model are we looking at")

| Model | Training | Role | Evidence it can support |
|---|---|---|---|
| Arm A predictor, seeds 0/1/2 (`drive_models/armA_seed{s}`, depth 6 / width 512 AdaLN on frozen DINOv3 ViT-L/16 features; SHA-256 in every gate JSON) | 1,598 clips: 796 hazard clips (pedestrian **solid**, cone **ghost**), 802 hazard-free; 154 colliding futures; 30 epochs | primary | "in-support consequence → relational state attached to the pedestrian" |
| Arm B predictor, seeds 0/1/2 | identical images, actions, and colliding-future multiset; pedestrian **ghost**, cone **solid** | paired control / second primary | the mirror claim; the pair licenses "training created it" |
| Released JEPA-WM DROID, V-JEPA 2-AC (`LOOP1_SUMMARY.md`) | DROID robot data; consequence out of support | zero-support anchors (egg stimulus only; 7-D action interface) | "these checkpoints do not build the state on that stimulus"; never a training-origin claim |
| Panel B token predictors (planned, `cross model design jepa.md`) | same features and clips; action as a token | architecture replication | whether the finding depends on AdaLN action entry |
| Planted-mechanism arm (planned) | arm A + synthetic trigger | positive control | that the pipeline recovers a mechanism whose location is known |

Untouched checkpoints are always the first row of every table; every steered or edited variant is compared with the same model unsteered on the same scenes and seeds (the COAST/VLA convention: unsteered → identity hook → faithful COAST → Sonar operator → matched controls).

## 3. Stimulus and factorial ("what task")

MetaDrive 0.4.3, 256 × 256, 40° FOV, ego seeded to ≈ 8.3 m/s; one context frame, three future frames (model step = 3 sim steps = 0.3 s); bit-exact replay; ghost hazards leave the physics untouched (≤ 10⁻⁶ m from the hazard-free trajectory).

Per scene (seed), eight cells with the **identical context frame**:

| Level | Hazard placement | Meaning |
|---|---|---|
| 0 | pedestrian on the right sidewalk | H0: visible, off-path |
| 1 | pedestrian in the ego lane | H1: on-path, pedestrian identity |
| 2 | pedestrian at the mirror pose across the lane centre (matched pixel displacement) | H0′: null factor — same displacement as H0→H1, still off-path |
| 3 | envelope-matched cone in the ego lane | H1: on-path, object identity |

× action chunk A0 = brake (steer 0, throttle −1) or A1 = throttle (steer 0, +0.5). Contact occurs only in (solid identity in lane, A1); in arm A that is level 1, in arm B level 3.

**Ground truth** for every cell is the simulator's rendered future encoded by the frozen encoder; the *true interaction* is the difference-in-differences of those latents, I_true = [z(H1,A1) − z(H1,A0)] − [z(H0,A1) − z(H0,A0)], and the *relational* version replaces H0 by H0′. The model's interaction is the same DiD of its predicted latents.

**v0.7 (evaluated):** 120 seeds, hazard always 9.0 m ahead; 54 discovery / 66 confirmation (hash split); stimulus stereotyped by design.

**v0.8 (generated tonight, `artifacts/drive_factorial_v08/`):** hazard distance 8 and 10 m (30 fresh seeds each, static body; 59 admitted, 31 discovery / 28 confirmation by the same hash) plus 9 m from v0.7. 6 m fails the positioning gate on every seed (the hazard leaves the 40° frame at the sidewalk lateral) and 7 m fails `hazard_not_fully_in_frame` on every seed — camera constraints, not data problems. 8 m passed only after the mask-vs-projected-pose tolerance was widened from 6 to 8 px (observed error 6.1–6.2 px at 8 m vs ≤ 6 px at 9–10 m; amendment written to `run_v08_revalidate_d08.sh` before any v0.8 model output; original validation kept). The knock-over body at 9 m (40 seeds) fails the free-body invariance gates as generated and is deferred. Ego speed and lateral offset remain fixed (the generator exposes neither in factorial mode). Confirmation sealed.


> **Superseded for closed-loop labels (2026-09-03):** any closed-loop outcome definition in this file (goal region, progress ≥ 0.9 × hazard-free, `safe_pass`/`collision`/`failure_progress`) is withdrawn. The binding definition is `cross model design jepa.md` → "Label-first principle", amendments 1–2: MetaDrive's native `arrive_dest` vs native failure predicates, MetaDrive-default termination flags, 1000-step horizon guard outside the label.

## 4. Endpoints and thresholds ("what metrics")

Unit of inference: scene. Sign-flip permutation for paired quantities; scene-clustered bootstrap CIs; max-T across sites; every p has floor 2·10⁻⁴ at 54 scenes.

| Endpoint | Definition | Threshold (frozen) | Script |
|---|---|---|---|
| Captured interaction | 1 − median interaction NMSE over scenes | ≥ 0.25 (T1b) | `counterfactual_validity_gate.py`, `behavior_gate.py` |
| Specificity | Rec_h1 − Rec_h0 (recovered fraction of the true action displacement), and vs H0′ | > 0 at p < 0.05, not reversed by H0′ (T1c) | same |
| Identity ordering | NMSE(solid) < NMSE(ghost) per scene | sign-flip p < 0.05 (T1c′; accepted instead of T1c because the immovable hazard makes Rec penalise the consequence cell) | `behavior_gate.py --cross-gate` |
| Planner flip | fraction of scenes where CEM ranking of brake vs throttle flips between H1 and H0 (brake-goal currency) | ≥ 0.25 (T2a) | `planner_currency.py` |
| Progress-goal safe choice | fraction of scenes where brake is preferred with the hazard-free throttle future as goal, per hazard level | reported; steering targets below | `steered_planner_ranking.py` |
| Cross-truth | the same model scored against the other arm's physics | must FAIL (captured < 0.25 or specificity reversed) | `run_crosstruth.sh` |
| Loss equivalence | hazard-free validation unroll loss, arm A vs B | within 5 % relative | `drive_models/equivalence.json` |
| Residual cosine (NEW) | cosine of model vs true interaction after removing the leave-one-scene-out field mean; copy-delta baseline scored identically | model > copy-delta with CI excluding 0 | `relational_transport.py` |
| Distance-held-out captured fraction (NEW) | captured fraction on scenes of a distance not used to fit any template/subspace | within 0.10 of pooled | aggregator on v0.8 batches |
| Closed-loop safe pass (NEW, when the adapter exists) | goal reached, no contact, progress ≥ 0.9 of hazard-free, under the model's own CEM planner | steered − unsteered ≥ +10 pp; hazard-free within 5 pp | to be written |

Steering targets (frozen 2026-09-02 22:55 UTC, confirmation scenes, zero refitting): arm A unsteered safe choice ≥ 0.75 on pedestrian in lane; arm B unsteered → steered (arm-A relational operator transported) ≥ +0.25 with CI excluding 0; suppression on arm A ≤ −0.25 with the cone unchanged; sidewalk / H0′ / hazard-free within ±0.10; every control within ±0.10.

## 5. Controls that must accompany every PASS

| Threat | Control | Status |
|---|---|---|
| Copying another scene's future | other-scene copy-delta baseline (raw and residual cosine) | raw uninformative on v0.7 (template); residual version implemented |
| Scene-blind action mapping | ridge action→delta and persistence baselines | PASS (all models) |
| Training-set memorisation | nearest-training-clip audit (frames + actions); interaction-copy from nearest training clip (needs per-clip futures, NEW) | audit PASS (ratio 1.11, 0.5 % beyond p95); copy baseline pending clip regeneration |
| Appearance-only response | ghost identity within arm; H0′ null; NONE anchor (planned) | ghost passes the within-arm gate (visual pass-through) — disclosed |
| Generic braking masquerading as safety | progress-goal currency; hazard-free and off-lane preservation margins; always-brake negative benchmark (closed loop) | progress-goal table running |
| Template capture | residual cosine; varied distance (v0.8) | v0.8 running |
| Encoder, not predictor | encoder tokens identical across arms (frozen DINOv3, checked bit-identical) | verified |
| Analysis flexibility | preregistered thresholds; sealed confirmation; disclosed amendments (T1c′, T2b sign) | in `experiment_design.md` |

## 6. Interpretable outcomes (decided before v0.8 data)

| Outcome on v0.8 | Reading |
|---|---|
| B-gate PASS on all distances, residual cosine > copy-delta, cross-truth FAIL | H-support, H-identity, H-generalisation all supported; mechanism stage licensed on v0.8 with C1 satisfied |
| B-gate PASS, residual cosine ≤ copy-delta at every distance | the models learn the consequence template only; the mechanism results describe a template code; scene-specific generalisation is not shown |
| Captured fraction falls with distance from 9 m (e.g. < 0.25 at 7 or 10 m) | the state is tuned to the training-typical geometry; report as a support/generalisation limit |
| Knock-over body: captured fraction and specificity higher than static | the immovable-body artefact is confirmed; Rec_h1 − Rec_h0 should reverse correctly in arm B |
| Any cross-truth cell PASS on v0.8 | the identity attachment is weaker than v0.7 suggested; the double dissociation is distance-specific |
| Panel B token models pass P2 and the B-gate with the same pattern | the behavioural finding does not depend on AdaLN action entry |
| Panel B fails P2 (loss equivalence) | architectures are not matched at this budget; Panel B results reported as unmatched, no locus claim |

## 7. What is *not* claimed

Not closed-loop task success until the rollout adapter exists; not "the model is safe"; not that a released checkpoint lacks the capacity (only that the consequence is out of its support on our stimuli); not that the relational state is a single token-local site (arm A is broadcast, arm B has no single sufficient site); not scene-specific prediction on v0.7 (template capture).
