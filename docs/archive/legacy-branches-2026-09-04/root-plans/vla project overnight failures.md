# VLA project overnight failures — lessons carried into the JEPA / world-model project

Source: the user's summary of the mechinterp-vla overnight runs (2026-09-03), reconciled in `Documents/mechinterp-vla/docs/FINDINGS-omnivla-mujoco-development-2026-09-03.md` and folded into `Documents/mechinterp-vla/cross model plan.md` ("Clarified target hierarchy after the OmniVLA development null"). Section 1 records what happened there; section 2 maps each lesson onto this project's evidence and says what changes here.

## 1. What failed in the VLA project, and why

**The earlier result was promising but never confirmatory.**

- The corrected bridge raised conflict accuracy from 46 % to 81 %, but scored 61 % overall against the frozen 65 % gate.
- It used already-opened development scenes and mostly evaluated *predicted action chunks*, not fresh robot trajectories.
- The strongest earlier version used ground-truth hazard-side metadata, so it showed that the action head was editable, not that the activations contained a usable safety mechanism.
- In the stricter closed-loop run, human and matched-object outcomes were identical: the model responded to encounter geometry, not specifically to humans.
- Synthetic MuJoCo imagery was outside OmniVLA's native visual distribution, weakening perception-to-action grounding.

**Reading.** Not a mysterious reversal. The exploratory work found a social representation and an editable motor endpoint but never proved a naturally used safety pathway between them. The next experiment needs a checkpoint that first demonstrates clear safety behaviour in its native domain; otherwise representational geometry keeps recovering collision timing or motor state.

**Three checkpoints were conflated.**

- OmniVLA `NHirose/omnivla-original` step 120,000: competent navigation, not explicitly safety-trained; social information present internally, little evidence the policy used it for avoidance.
- `LIBERO-Safety/pi05_libero_safety`: safety-trained on collision-free demonstrations, but the matched human-vs-object test found a 1 mm effect against a required 10 mm; plausibly generic obstacle avoidance, not the human-specific contrast.
- `zjuSekineko/SocialNav-Qwen2.5-VL-SAFE-GRPO`: explicitly post-trained for socially compliant navigation; its published native-domain safety performance must be reproduced first, and its public evaluation assets are the practical obstacle.

**The target was too specific too early.** COAST starts from an abundant, unambiguous label (episode success or failure). The VLA plan required one representation to encode human identity, route conflict, future collision risk, and a selective corrective action distinct from generic avoidance — a four-part hypothesis whose failure cannot be separated from a failed behavioural gate.

**Revised sequence (VLA plan).** (1) Reproduce baseline safe task completion on established LIBERO-Safety tasks; (2) fit COAST-style geometry from successful vs safety-failed closed-loop rollouts; (3) show steering improves collision-free success over the frozen checkpoint; (4) only then decompose the subspace into generic avoidance vs human-specific caution; (5) treat selective human-collision control as the high-novelty follow-up. Faithful COAST is the required baseline, not the contribution; novelty is the factorization, the pathway, the selective donor-free intervention, and cross-architecture recurrence of the functional recipe. Exploratory pilots become engineering priors and reusable data, not headline evidence.

## 2. What this means for the JEPA driving cell

| VLA lesson | Our evidence (2026-09-03) | Change here |
|---|---|---|
| "Predicted action chunks, not fresh trajectories" | Every driving endpoint so far is open-loop: predicted latents, CEM ranking, the COAST-style safe-choice table. No rollout has ever hit or avoided a hazard. | Closed-loop MetaDrive rollouts under the model's own CEM planner (collision / safe pass / progress) become the discovery label and the C4 endpoint, moved ahead of the remaining single-site mechanism work (`cross model design jepa.md` → "Target hierarchy"). |
| "Responding to encounter geometry, not to humans" | The ghost identity passes the within-arm gate through visual pass-through; on v0.8 at 8 m it is captured as well as the solid consequence; identity separates only at 10 m or via cross-truth. | Identity attachment is a second-stage factorization licensed by the matched arms and cross-truth (12/12 FAIL on v0.7 and v0.8), never by the within-arm ordering. The within-arm identity gate (T1c′) is demoted to descriptive. |
| "Editable endpoint ≠ used mechanism" | Arm A donor patch at `L03.mlp_out` recovers 72 % but is not token-selective; the causal chain from state to planner choice is untested; the steering table is still pending. | The steered-planner table plus closed-loop steering with preservation margins is the causal-use test; a large recovery without a planner effect is reported as "editable, not used". |
| "Checkpoint not safety-trained in its native domain" | Different failure, same shape: the released JEPA checkpoints had the consequence out of support (egg nulls). Our trained arms fix this by construction — the one structural advantage over the VLA setup. | Keep the matched-arm design; report released checkpoints only as zero-support anchors. |
| "Synthetic imagery out of the native distribution" | Training and test are both MetaDrive, so no OOD gap; but the stimulus is stereotyped, which produced template capture (residual cosine ≈ 0 within a distance on v0.8). | v0.9 randomised factors (distance, speed, lateral offset; wider FOV; settled knock-over body) so the field has scene-specific structure. |
| "Target too specific too early" | Registered estimand was the three-way identity × route × action interaction, tested first through subspace geometry, which returned an identity-blind rank-1 template. | Coarse outcome geometry first (collision vs no collision), factorize afterwards; the template-projected identity-contrast test (`identity_contrast_geometry.py`) replaces raw-DiD subspace tests for the identity question. |
| "Faithful COAST is the baseline" | Our conceptor arm is Sonar (fitted on DiD residuals), not COAST. | Add a faithful COAST row (outcome conceptors from real closed-loop successes/failures) to every steering table. |
| "Reuse exploratory pilots as priors and data" | v0.7 discovery results (all six B-gates, mechanism seed 0, cross-arm seed 0) are discovery evidence; confirmation seeds sealed except through the frozen COAST rules. | Keep v0.7 as the development set; v0.8/v0.9 and the sealed confirmation scenes carry the confirmatory claims. |

**Where we may still hit the same wall.** If closed-loop rollouts show that the planner's brake/throttle choice is driven by in-lane presence regardless of identity (the driving analogue of "human and matched-object outcomes identical"), then the identity-attached state exists in the predictor but is not used by the planner, and the honest claim shrinks to "in-support consequence creates a readable, editable relational state; selective use is not shown". That outcome is pre-specified in `behavioral_design_v08.md` §6 and would be reported as such.
