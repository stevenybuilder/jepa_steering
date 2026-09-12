# Causal Safety Geometry in JEPA World Models

**Experiment design and literature review**  
**Research snapshot:** 2026-09-01  
**Status:** Phase-0 pilot execution; do not train a transcoder until the causal-localization gate passes

## Executive verdict

The project should pursue this question:

> Where does an action-conditioned JEPA world model construct a prospective physical-constraint state—hazard relative to the planned route—and can a bounded edit to that state redirect imagined futures, plan ranking, and closed-loop robot behavior?

This is the JEPA/world-model analogue of the strongest Othello-style VLA experiments. The important unit is not a probe-readable “safety feature.” It is a computation that is:

1. **Relational:** it represents whether a hazard intersects the proposed action-conditioned future, not merely whether a wall is visible.
2. **Prospective:** it appears before collision in the predicted rollout.
3. **Causal:** removal and insertion move future predictions and candidate-plan ranking in opposite directions.
4. **Behavior-controlling:** the edit persists through every CEM candidate, rollout step, and replan and changes realized collision/route behavior.
5. **Specific:** effects survive visual, action, motor-progress, layer, time, norm, and reconstruction controls.

The best first substrate is the released action-conditioned **JEPA-WM RoboCasa/DROID** checkpoint. The primary pilot will use a RoboCasa/OopsieVerse-inspired vulnerable-entity route-conflict task plus a gentle-contact task, both expressed through the checkpoint-compatible RoboCasa camera and action interface. Native Wall and `mw-reach-wall` results remain positive controls for obstacle-cue availability and patching infrastructure, not the primary safety stimulus. The existing wall result is useful because wall evidence is encoded and causally affects early forecasts, yet the forecast does not sustain the constraint over longer horizons. It is not a confirmatory safety-mechanism result because the strict rendering-matched stimulus gate failed and the task did not cleanly demonstrate a changed route choice.

**Recommendation:** run a limited causal-localization pilot first. Train a transcoder only if a predictor MLP site passes the behavioral, specificity, and bidirectional patching gates. A full study is justified only if the localized state changes planner ranking and at least one closed-loop behavior.

**Novelty assessment:** moderate-to-high for the complete claim above; low for “collision is decodable,” “a sparse feature looks like a wall/collision,” or “latent steering changes a forecast.” Closest prior work already establishes physical decodability, curved physical manifolds, causal steering of video-model judgments, and geometry-aware forecast steering. The unclaimed combination is an action-conditioned prospective safety computation causally tied to plan selection and closed-loop physical behavior.

## 1. What transfers from the VLA representational-geometry project

### 1.1 Study the destination representation, not the original conditioner

The strongest π0.5 result was not that instruction tokens themselves controlled the action. Instruction content was rewritten into image-position states during the vision-language pass, and a state-conditioned patch at those destination positions repaired behavior. The corresponding JEPA question is not simply “where is the obstacle encoded?” The image encoder can encode the obstacle perfectly while the predictor fails to integrate it with a proposed route. The target is the **predictor state where scene evidence and candidate actions become a prospective future**.

This yields a concrete causal chain:

`scene/hazard + candidate action sequence → predictor computation → prospective constraint state → predicted future/cost → selected plan → realized behavior`

Every claimed mechanism must be tested at more than one downstream link.

### 1.2 Decodability is only availability

The VLA experiments repeatedly found strong monitors and probes that did not identify a behavior-controlling variable. High decoding accuracy, attention, or a semantically appealing sparse feature can reflect information that is present but unused. A JEPA safety claim therefore requires:

- an information test: can the state predict simulator-ground-truth clearance/contact?
- a mediation test: does an exact counterfactual patch change the predicted future or ranking?
- a necessity test: does removing the state selectively reduce safety-sensitive behavior?
- a sufficiency test: does inserting it selectively increase safety-sensitive behavior?
- a closed-loop test: does the persistent edit change the executed route or collision rate?

### 1.3 Expect local, curved, or distributed geometry

The VLA project found repeatable within-task motor structure but no shared cross-task topology. Cross-scene directions often approached orthogonality, and a generic visual STOP/GO direction failed behaviorally. Those are warnings against assuming a universal linear “safety vector.”

The geometry tournament should therefore compare, in order:

1. a one-dimensional linear risk direction;
2. a low-rank supervised subspace;
3. circular or factorized coordinates where direction is encoded as sine/cosine pairs;
4. a local nonlinear or geodesic manifold;
5. a transition operator or local Jacobian describing how candidate actions move the predicted state;
6. sparse transcoder features if the causal site is an MLP computation.

The scientifically useful negative result is also clear: safety may be computed by task-local, state-conditioned interactions rather than a portable global direction. The study should measure that rather than force alignment.

### 1.4 Use state-conditioned donors before static steering

The π0.5 repair depended on a donor from the same live scene; a static mean direction was not the established mechanism. For JEPA-WM, exact matched safe/unsafe counterfactual patches should precede concept-vector steering. If exact patches work but a global direction fails, test local neighbors, transported bases, or a curved manifold. Do not interpret failure of a static vector as absence of causal safety computation.

### 1.5 Measure destination distance, not projection alone

A VLA random edit achieved a misleadingly high projection-style recovery score while remaining near neither the receiver nor the desired donor endpoint. For each edit, report raw task metrics plus normalized distances to both endpoints:

\[
D_{\mathrm{dst}} = \frac{\lVert y_{\mathrm{edit}}-y_{\mathrm{dst}}\rVert_2}
{\lVert y_{\mathrm{recv}}-y_{\mathrm{dst}}\rVert_2},\qquad
D_{\mathrm{src}} = \frac{\lVert y_{\mathrm{edit}}-y_{\mathrm{recv}}\rVert_2}
{\lVert y_{\mathrm{recv}}-y_{\mathrm{dst}}\rVert_2}.
\]

Here, `y` is a predicted trajectory, latent rollout, decoded state trajectory, or candidate-cost vector—not merely a direction projection. A useful edit lowers `D_dst` without producing a large, nonspecific displacement.

### 1.6 Null-site and edit controls must be causally live

A null result at a token, block, or lane means little unless a positive control shows that the site can change the measured endpoint. Each claimed null therefore needs a same-site content patch or a known-live perturbation. Each claimed positive needs equal-norm random, opposite, wrong-feature, wrong-layer, wrong-time, shuffled-label, and native-output controls.

### 1.7 The statistical unit is the independent scene/task cell

CEM produces hundreds of correlated candidate trajectories per scene. These candidates are not independent samples. The unit of inference is the independently reset simulator scene/seed, nested within task family; candidates, timesteps, tokens, and replans are repeated measures. Discovery and confirmation scenes must be disjoint.

## 2. What already exists in the JEPA code and evidence

The relevant implementation is in `/Users/stevenyang/Documents/mechinterp-vla`, with the sparse-feature rationale in [`transcoder.md`](/Users/stevenyang/Documents/mechinterp-vla/transcoder.md).

### 2.1 Model substrate

The released [JEPA-WM code and checkpoints](https://github.com/facebookresearch/jepa-wms) provide the right causal substrate:

- a frozen visual encoder and action-conditioned predictor;
- a released DROID/RoboCasa checkpoint and checkpoint-compatible RoboCasa evaluation path;
- action embeddings applied throughout predictor blocks;
- a proprioceptive lane concatenated to visual tokens;
- CEM planning over predicted latent futures;
- native `wall` and MetaWorld `reach-wall` evaluation configurations as positive controls.

This is materially closer to the target question than a passive video encoder: candidate actions enter the predictor, the predictor supplies a planning cost, and interventions can be applied during the optimizer's imagined rollouts.

### 2.2 Existing intervention infrastructure

The local code already supports most of the causal experiment:

- [`jepa_patching_lib.py`](/Users/stevenyang/Documents/mechinterp-vla/jepa_patching_lib.py) exposes action embeddings, AdaLN modulation, block residuals, and the proprio lane for patching and directional edits.
- [`run_jepa_predictor_patching.py`](/Users/stevenyang/Documents/mechinterp-vla/run_jepa_predictor_patching.py) provides a predictor-patching entry point.
- [`run_implant_planner_test.py`](/Users/stevenyang/Documents/mechinterp-vla/run_implant_planner_test.py) already applies persistent predictor hooks across CEM samples, imagined steps, and replanning and records predicted and realized physics.
- [`eval_wall_base_model.py`](/Users/stevenyang/Documents/mechinterp-vla/eval_wall_base_model.py) and [`rollout_metaworld_wall_stimulus.py`](/Users/stevenyang/Documents/mechinterp-vla/rollout_metaworld_wall_stimulus.py) provide the wall-test foundation.

The main engineering additions are MLP-input, attention-output, and MLP-output hooks; exact simulator-derived risk labels; counterfactual stimulus manifests; and a transcoder trainer/evaluator at a frozen selected MLP site.

### 2.3 Current wall evidence

On 24 seeds in the prior wall run:

- the encoder cue gate passed locally (wall/no-wall linear decoding was reported as 1.00);
- patching wall-region tokens explained most of the blocked-versus-no-wall forecast difference, far beyond random-region controls;
- the model's wall-relative forecast correction was strongest early and weakened over rollout horizon;
- a persistence baseline eventually outperformed the model's blocked forecast.

This supports a specific starting hypothesis: **the model sees the obstacle and initially incorporates it, but the prospective constraint is not maintained through the rollout.** It does not yet establish a causal safety mechanism. The formal stimulus gate failed because the paired renders differed by small amounts outside the intended wall mask on 11/24 seeds, and the test did not show that an internal edit redirected closed-loop planning around an obstacle.

### 2.4 What not to use as the main substrate

- The failed learned interlock implant did not pass its learning gate and cannot support a mechanism claim.
- A generic STOP/GO direction is not justified by the VLA evidence and does not match the relational variable a planner needs.
- Natural-planning analyses found no confirmed recurrent matched predictor exploit in the existing sample, so “exploit localization” should not be the preregistered premise.
- An encoder-only sparse feature can explain obstacle appearance but cannot, by itself, establish action-conditioned risk.

## 3. Literature review and novelty boundary

The literature search prioritized primary papers and official repositories. The relevant prior art falls into four layers.

| Prior work | What it establishes | What remains for this project |
|---|---|---|
| [JEPA-WM: Learning and Leveraging World Models in Latent Space](https://arxiv.org/abs/2512.24497) and [official code](https://github.com/facebookresearch/jepa-wms) | An action-conditioned latent world model used for planning, including Wall and MetaWorld evaluations. | No causal localization or editing of a prospective safety computation tied to closed-loop planner behavior. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) and [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | Strong video representations and action-conditioned world-model/planning variants. | The safety-relevant internal geometry and its causal effect on decisions remain open. |
| [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050) | Physical variables emerge at characteristic depths; some motion variables have structured, high-dimensional geometry and admit coordinated steering. | It studies physical representation in video models, not a hazard×candidate-action mechanism governing a robotic planner. |
| [Causal Physics Steering via Concept Activation Vectors](https://arxiv.org/abs/2605.24322) | Activation steering can change a video model's physical-plausibility judgments. | A classifier judgment is not an action-conditioned forecast, candidate-plan ranking, or closed-loop robot behavior. |
| [Manifold Steering](https://arxiv.org/abs/2605.05115) | Geometry-aware edits can outperform linear steering and produce smoother controlled world-model forecasts in a visual control setting. | “Nonlinear geometry changes a forecast” is therefore not enough. We must connect a prospective safety state to plan choice and realized behavior. |
| [V-JEPA 2 Archetypal Transcoder](https://github.com/sghassemlou/V-JEPA-2-Transcoder), [model card](https://huggingface.co/sghassemlou/vjepa2-archetypal-transcoder), [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944), and [Archetypal Sparse Autoencoders](https://arxiv.org/abs/2502.12892) | Sparse transcoders can decompose an MLP computation rather than only its residual stream; the released V-JEPA 2 transcoder obtains useful but incomplete reconstruction. Archetypal constraints can improve feature stability. | Sparse semantics are not automatically causal or safety-specific. A transcoder must be trained only at an already localized predictor computation and edited as a delta on top of the native output. |
| [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382) and [Linear Latent World Models in Simple Transformers: A Case Study on Othello-GPT](https://arxiv.org/abs/2310.07582) | Othello-style counterfactual probing and interventions can expose internal state variables used in future action selection. | Robotics needs continuous, action-conditioned relational labels, simulator interventions, and closed-loop behavioral validation. |
| [One Lens, Many Worlds](https://arxiv.org/abs/2606.09936) | A typed interface can standardize time-indexed world-model activation capture and replay. | Tooling is not a causal safety-mechanism result; our existing hooks can implement the first study directly. |
| [Imperfect World Models Are Exploitable](https://arxiv.org/abs/2605.15960) | World-model error can invert policy ranking, especially over large candidate sets, motivating horizon-aware safeguards. | It does not identify the internal representation responsible for a particular constraint failure or repair it causally. |
| [MiraBench](https://arxiv.org/abs/2605.29360) | Visual fidelity is a poor proxy for action fidelity, and optimistic action outcomes are common in world models. | This motivates action-grounded safety endpoints but does not localize or edit their mechanism. |
| [ActSWM](https://arxiv.org/abs/2607.26712), [Causally Debiased Latent Action Models](https://arxiv.org/abs/2607.09185), and [Overcoming Statistical Bias in Action-Controllable World Models](https://arxiv.org/abs/2608.04653) | Recent work isolates action insensitivity/context collapse, causally debiases latent actions, and evaluates same-state multi-action counterfactual consistency. | These are strong reasons to factorially vary candidate actions. They do not establish a sparse or geometric hazard×route state that controls a planner. |
| [Counterfactual Quotient Models](https://arxiv.org/abs/2608.22092) | Learns a centered representation of action-dependent future differences from synchronized counterfactual rollouts, removing components shared across actions while preserving decision-relevant comparisons. | Common-mode cancellation and action-relative future representations are therefore not our novelty. CGS is a post-hoc method for frozen pretrained models that localizes where a particular physical interaction is constructed, compares its native geometry, and tests causal edits through behavior. |
| [How Can Driving World Models Do Counterfactual Prediction?](https://arxiv.org/abs/2608.11601) | Shows that direct alternative-action prediction need not preserve the factual episode and frames valid counterfactual prediction as abduction, action, and prediction; evaluates matched simulator counterfactuals. | The human/driving transfer cannot assume that a plausible alternative future is the correct paired counterfactual. CGS needs a counterfactual-validity gate against exact simulator futures before interpreting internal differences. |
| [VLA-JEPA](https://arxiv.org/abs/2602.10098) | JEPA-style objectives are entering vision-language-action policy learning. | This project targets the mechanistic world-model/planning case first, where future prediction and intervention endpoints are directly measurable. |
| [π0.5](https://arxiv.org/abs/2504.16054) | A VLA policy provides the architectural and mechanistic contrast motivating conditioner-to-controller routing. | The JEPA study concerns action-conditioned latent prediction and optimizer-mediated control, not language routing through a policy. |

### Required OpenReview reference

The design also records the user-specified source [OpenReview paper Q3osfMTUbE](https://openreview.net/pdf?id=Q3osfMTUbE) as a required closest-prior-art check. On 2026-09-01, both the PDF and OpenReview API returned a browser-verification challenge, and exact-ID searches in arXiv, OpenAlex, Crossref, and Semantic Scholar returned no indexed record. I therefore do **not** attribute a title, method, or result to it without seeing the paper. Before freezing a novelty claim or preregistration, attach/download the PDF and add a row above stating exactly which part of the claim it anticipates. The experimental design below does not depend on an unverified description of that paper.

### Novelty claim we may make if the experiments pass

> An action-conditioned JEPA predictor constructs a prospective hazard×route state with a reproducible model-native geometry; controlled removal and insertion of that state change predicted contact, candidate-plan ranking, and closed-loop route/collision behavior.

### Claims the current study must not make

- “We discovered a safety neuron” from feature visualization or top-activating clips.
- “The model understands safety” from probe accuracy.
- “The representation is universal” from pooled within-task performance.
- “The intervention repairs behavior” from decoded forecasts alone.
- “A transcoder explains the mechanism” when its reconstruction leaves a large fraction of the MLP output unexplained.

### Proposed community contribution: Causal Geometry Sonar

The working methodological contribution is **Causal Geometry Sonar (CGS)**: a reusable procedure for discovering the model-native coordinates in which an action-conditioned world model represents a physical relation, locating where that relation becomes causally live, and constructing a bounded intervention in the discovered local geometry.

CGS does not claim novelty for same-state multi-action evaluation, common-mode cancellation, finite-difference Jacobians, activation patching, or manifold steering individually. Its proposed contribution is the validated composition of controlled factorial pings, network-wide interaction localization, coordinate-system discovery, on-manifold causal editing, and planner/behavioral verification on a frozen world model.

The sonar analogy is operational rather than decorative:

1. **Emit controlled physical pings.** Around one frozen simulator state, perturb a low-dimensional physical parameter vector `θ`—candidate-path angle, signed clearance, relative speed, time-to-contact, impact velocity—while preserving task, endpoint, and nuisance factors as closely as possible.
2. **Record network echoes.** At every layer, imagined time, component, and token group, measure the factorial interaction response `E = Δ_hΔ_a z`, not merely the activation difference between unrelated safe and unsafe clips.
3. **Triangulate the native chart.** Estimate the local tangent/Jacobian `∂E/∂θ`, intrinsic rank, curvature, and transport across nearby states. Compare world-, camera-, gripper-, goal-, and candidate-path-relative coordinates by held-out predictive simplicity and stability.
4. **Ping the computation causally.** Use exact donor patches first, then low-rank/local-manifold and sparse-feature edits at the earliest causally live site. Measure the return in complete predicted trajectories, planner cost/rank, and both destination distances.
5. **Verify behavior and range.** Test fresh closed-loop plans, continuous dose response, transfer distance across scenes/tasks/entities, and explicit failure outside the chart's validated neighborhood.

CGS should output four reusable artifacts rather than a gallery of features:

- an **emergence map** over layer × imagined time × component showing where each physical interaction becomes explicit;
- a **coordinate report** identifying which external or learned local chart most simply describes the representation;
- a **causal mediation map** from sensory evidence and candidate action conditioning through predictor computation to forecast and rank;
- a **calibrated steering operator** with a declared validity radius, off-target displacement, and behavioral effect.

This is deliberately stronger than the local baseline [`jepa transcoder.pdf`](</Users/stevenyang/Documents/mechinterp-vla/jepa transcoder.pdf>). That work establishes the feasibility of an archetypal transcoder on a passive V-JEPA 2.1 encoder and reports motion-aligned and hazard-enriched SSv2 features, but its safety result is observational: one preselected encoder MLP site, top-activation precision rather than full matched-negative characterization, severe feature collapse, no action-conditioned counterfactual interaction, no causal intervention, and no planner or behavior. CGS uses a transcoder only after causal localization and treats sparse features as one coordinate model among linear, low-rank, circular, Jacobian, and local-manifold alternatives.

The methodology claim is earned only if the frozen CGS procedure:

1. localizes a reproducible upstream interaction site that probe peaks or top-activating features alone do not identify;
2. predicts continuous held-out physical variables and transfers across novel trajectory generators and at least two risk families;
3. produces more specific and sample-efficient causal control than global mean directions, probe vectors, a preselected-site transcoder, and nearest-neighbor donor retrieval;
4. changes fresh planner ranking and closed-loop safety without reducing behavior to stopping or generic low-speed motion;
5. runs on a second checkpoint or model family without hand-selecting a new analysis pipeline after seeing its outcomes.

If only a hazard-selective transcoder feature is found, this project has not exceeded the prior result. If CGS finds a local chart but it cannot control prediction or behavior, the contribution is descriptive representational geometry rather than a causal-safety methodology. The name and novelty claim remain provisional until the frozen method passes these tests and the literature search is refreshed at submission time.

## 4. Formal hypotheses

Let `x` be a simulator state and rendered observation, `h` the hazard geometry, and `a₁:H` a candidate action sequence. Use the simulator—not an LLM—to compute:

\[
r(x,h,a_{1:H})=\min_t d_{\text{signed}}(q_t,h),
\]

where `q_t` is the controlled body/object trajectory. Define contact `c = 1[r ≤ 0]`, time-to-contact `τ`, goal progress `g`, and route side/strategy `u`. The central variable is `r` or `(r, τ)`, conditioned jointly on state, hazard, and proposed actions.

### H1 — availability versus sustained use

Obstacle information is available and causally live in encoder tokens, but its effect on predicted clearance decays at a particular rollout step or predictor depth.

**Pass:** exact wall/no-wall token patches move early forecasts toward the donor; the effect attenuates before the true constraint ceases to matter; nuisance-matched patches do not explain it.  
**Falsifier:** the cue is absent at the encoder, never affects the predictor, or forecast differences are fully explained by rendering artifacts/persistence.

### H2 — causal relational safety state

A predictor band contains a state that tracks hazard relative to the candidate route, beyond hazard appearance, action identity, and generic goal progress.

**Pass:** on held-out scenes, exact safe↔unsafe patches at a frozen site bidirectionally change predicted minimum clearance/contact and candidate ranking; the site adds information beyond nuisance covariates.  
**Falsifier:** decoding disappears under same-scene action counterfactuals, or patches move only generic motion/progress.

### H3 — model-native geometry

The causal state has a stable low-dimensional or sparse local geometry that supports held-out interpolation and transport.

**Pass:** one frozen coordinate model beats nuisance and shuffled baselines on held-out scenes and at least two held-out task/geometry families, and geometry-constrained edits outperform equal-norm random edits.  
**Falsifier:** only scene-specific donor patches work, geometry is unstable across seeds, or no tested compact geometry captures the effect. This would support a distributed, state-conditioned mechanism rather than “no mechanism.”

### H4 — causal behavioral control

Editing the localized state during all planner evaluations changes the plan the optimizer selects and improves realized safety without simply stopping or destroying task competence.

**Pass:** risk-increasing and risk-decreasing edits move plan ranking in opposite directions; the safety edit reduces realized collisions while preserving goal success/progress and staying within preregistered path/time overhead.  
**Falsifier:** predictions move but CEM ranking does not; ranking moves but replanning cancels the effect; collision falls only because actions collapse toward zero; or task success falls materially.

### H5 — method transfer rather than coordinate universality

The frozen CGS pipeline can discover and validate a causal physical-interaction chart on a second risk family and comparison checkpoint without assuming that the same feature, layer, basis, or topology must recur.

**Pass:** preregistered perturbation generation, interaction localization, chart selection, intervention calibration, and validity diagnostics run unchanged; the selected site/chart passes the same held-out causal gates on the replication substrate.  
**Falsifier:** success requires hand-selecting a new layer, rewriting labels or controls after seeing outcomes, or relaxing the causal/behavioral gates. A method can transfer even when the discovered model-native coordinates differ; coordinate universality is a separate and stronger claim.

### Competing explanations to test explicitly

- visual inertia or persistence rather than physical constraint integration;
- generic wall/obstacle identity rather than hazard×route relation;
- generic motor state, speed, or distance-to-goal;
- action insensitivity or context collapse;
- planner-objective mismatch despite a correct world model;
- a causally live but nonspecific high-gain site;
- off-manifold intervention artifacts;
- decoded-state-head error rather than predictor error.

## 5. Stimulus and dataset design

### 5.1 Provenance policy

No LLM-generated scenes, labels, actions, or success judgments will enter the benchmark. Use:

- released JEPA-WM checkpoints and official task configurations;
- deterministic MuJoCo/MetaWorld state snapshots;
- exact simulator collision geometry and contact traces;
- expert, CEM, and deliberately perturbed/mirrored action sequences;
- a frozen manifest containing environment version, checkpoint hash, RNG seed, simulator state, camera parameters, action tensor hash, render hash, and ground-truth trajectory labels.

The experiment creates controlled stimuli from an existing physics simulator rather than using an unvalidated synthetic text/image generator. Simulator-generated labels are still audited: replay each saved state/action pair twice, verify deterministic physics within tolerance, and visually inspect a stratified sample before activation collection.

For RoboCasa, stimulus acceptance is checked in both physical and model-observation coordinates. The pilot generator must use the released JEPA-WM `robot0_leftview` pose and field of view, place the hazard within 3 cm of the candidate approach point, keep the control at least 12 cm off-path on the same support surface, retain at least 25 segmented target pixels, leave at least one 16-pixel ViT patch between the target and every image edge, and move the hazard/control target centroids by at least one patch in the rendered input. The paired simulator states may differ only in the target object's free-joint position. Every cell is rerun after reloading the frozen episode XML; acceptance requires bit-identical frames, maximum flattened-state error at most `1e-8`, force error at most `1e-6 N`, and identical contact counts. The gentle-contact calibration additionally requires the preregistered contact pattern `(H0,A0)=0`, `(H0,A1)=0`, `(H1,A0)=0`, `(H1,A1)=1` and a force DiD of at least `5 N`. Calibration seeds used to set these gates are excluded from confirmatory inference.

### 5.2 Contamination and trajectory-copying policy

Correction (2026-09-01, verified in the vendored release config `configs/vjepa_wm/droid_final_sweep/...dv3vitl...2roll_4n.yaml`: `datasets: [DROID]`, RoboCasa evaluated with `override_datasets: true`, and the JEPA-WM paper arXiv 2512.24497 states the DROID models are not finetuned on RoboCasa): the released `jepa_wm_droid` checkpoint was trained on DROID only and is evaluated on RoboCasa zero-shot. Exact-frame contamination against RoboCasa is therefore impossible; the relevant risk is the opposite one, domain shift of rendered RoboCasa frames relative to DROID video, which must be audited and reported. The study must distinguish three different claims:

1. **Exact non-duplication:** an evaluation frame/action sequence was not present in the accessible training corpus.
2. **Compositional generalization:** familiar assets or primitives are recombined into unseen layouts, hazard relations, and trajectories.
3. **Semantic transfer:** the localized computation transfers to unseen vulnerable entities or task families.

Before fitting a probe or choosing a site:

- audit evaluation frames against the accessible RoboCasa training corpus with perceptual hashes and frozen-encoder nearest neighbors;
- audit action sequences with normalized action-space nearest neighbors and dynamic-time-warping distance;
- record nearest-neighbor identities and distances in the stimulus manifest rather than silently dropping difficult cases;
- exclude OopsieVerse safe/unsafe demonstration trajectories from the confirmatory set; they may only define plausible motion ranges for the pilot;
- generate evaluation action chunks procedurally after the checkpoint, simulator version, and task specification are frozen;
- split by complete base scene, object mesh, layout, and trajectory generator before fitting any probe, direction, manifold, or transcoder;
- report stock-asset, novel-composition, and unseen-entity transfer strata separately.

Trajectory-copying controls must include endpoint-matched novel spline paths, left/right reflections, temporal rescaling, continuous interpolation across the contact boundary, and action chunks recombined with unseen initial states. Compare the world model with an explicit nearest-neighbor trajectory-retrieval baseline. A proposed safety representation must respond approximately monotonically to continuous clearance or impact-speed sweeps and must affect fresh CEM candidates generated after intervention. Donor actions are never patched or replayed.

Passing the exact-duplication audit weakens the exact-replay explanation; it cannot prove that pretraining contributed no memorized structure. Passing novel compositions supports relational generalization. Only the unseen-entity/task strata can support a broader safety-geometry claim.

### 5.3 Task and stimulus panel

The primary stimuli are simulator-generated counterfactual futures, not a collection of unrelated safe and unsafe demonstrations. Each cell begins from an identical saved state and visual history and varies the proposed future or one controlled scene factor.

**Family A — vulnerable-entity route conflict.** Use the JEPA-WM-compatible RoboCasa pick-and-place camera/action interface augmented with OopsieVerse/DamageSim mechanical-damage labels. Begin after the robot has grasped its payload and shortly before the relevant interaction enters the predictor horizon. Each starting state must admit:

- an unsafe direct candidate that contacts or violates a clearance margin around a fragile object;
- safe left and right detours;
- a slow-direct candidate separating speed from geometry;
- pause/retreat and matched-magnitude irrelevant-action controls.

Productive candidates should converge to approximately the same terminal end-effector pose and matched goal progress while differing in their intermediate swept volume. Sweep signed path clearance, approach angle, relative speed, time-to-contact, mirrored hazard side, appearance, and object identity. The primary continuous labels are minimum signed clearance, first-contact time, maximum contact impulse, health change, and task progress.

**Family B — gentle versus damaging contact.** Use OopsieVerse `pick_egg` or `place_plate`. From the same saved pre-contact state, vary lowering velocity, release timing, drop height, and gripper command while approximately matching endpoint and progress. This tests a second proposed safety coordinate—impact/contact dynamics rather than route clearance.

**Held-out humanized transfer.** Only after a mechanism is localized on native manipulation stimuli, substitute a static damage-tracked human hand/forearm into geometry-matched route-conflict scenes. Do not use these scenes to select the layer, fit the coordinate system, or calibrate edit strength. This is an out-of-distribution transfer test, not evidence that the checkpoint already possesses a general concept of human safety. Moving-human or autonomous-driving scenes require a separate counterfactual-validity test that preserves other-agent dynamics under alternative ego actions; they are not part of the immediate pilot.

**Positive controls.** Retain clean Wall and `mw-reach-wall` scenes to verify obstacle-cue availability, hook correctness, and known-live intervention sites. They do not supply the primary behavioral claim.

### 5.4 Othello-style factorial counterfactuals and preregistered DiD

Build counterfactual cells that change one causal factor while holding the others fixed:

| Contrast | Fixed | Changed | Purpose |
|---|---|---|---|
| Same-state multi-action | pixels, state, hazard | candidate action sequence | Isolate prospective route risk from visual obstacle identity. |
| Same-action hazard relocation | state, action sequence, visual style | obstacle position/geometry | Test whether the same motion becomes safe/unsafe because of relation to hazard. |
| Mirrored route | clearance, speed, progress | left/right direction | Separate risk magnitude from direction and test circular/factorized geometry. |
| Same progress, different risk | endpoint progress, approximate effort | minimum clearance/contact | Remove generic goal-progress and motor-vigor explanations. |
| Same risk, different appearance | physics and route risk | colors/textures/background/object instance | Test nuisance invariance. |
| Temporal controls | state/action multiset | correct, reversed, or shuffled order | Test prospective temporal computation. |

Every scene should contain all relevant within-scene counterfactuals. Split by base simulator seed and geometry family before fitting any probe or direction.

The primary estimand is a paired factorial difference-in-differences interaction. For the route task, let `H=1` place the vulnerable entity inside the candidate corridor and `H=0` place an equally visible matched entity outside it. Let `A=1` denote the direct candidate and `A=0` an endpoint/progress-matched detour. For outcome `Y`, define:

\[
\operatorname{DiD}(Y)=
\left(Y_{H=1,A=1}-Y_{H=1,A=0}\right)
-\left(Y_{H=0,A=1}-Y_{H=0,A=0}\right).
\]

This removes the main effect of seeing the entity and the main effect of the action family. The remaining interaction estimates the prospective hazard×trajectory relation. Because both factors are experimentally manipulated within paired simulator scenes, this is a randomized factorial interaction—not an observational parallel-trends design.

Apply the same estimand to simulator damage, predicted clearance/contact, CEM cost/rank, activation coordinates, intervention effects, and realized closed-loop behavior. For gentle contact, use the corresponding `fragile versus durable × gentle versus high-impact` interaction. Fit the equivalent mixed model

\[
Y \sim H + A + H{:}A + \text{speed} + \text{progress} + (1\mid\text{scene}),
\]

and report a scene-clustered paired permutation or bootstrap interval for `H:A`. The scene—not the candidate, timestep, token, or CEM sample—is the unit of inference.

At predictor site `l`, also estimate the representational interaction

\[
I_l=z_l^{11}-z_l^{10}-z_l^{01}+z_l^{00}.
\]

Use cross-validated interaction magnitude/subspace estimates rather than treating a single raw vector as a universal direction. The first site where this interaction emerges is a candidate relational-computation site; it becomes a causal site only if matched interventions there change downstream forecast and rank interactions.

### 5.5 Labels and readouts

Primary simulator labels:

- minimum signed clearance;
- contact/no-contact and first-contact time;
- time-to-collision or censored no-collision horizon;
- goal progress and success;
- path length, action magnitude, and completion time;
- route side/strategy.

Primary model readouts:

- predicted clearance/contact from a frozen state head where validated;
- predicted latent distance/cost used by CEM;
- candidate ranking and selected action sequence;
- rollout-step-specific latent trajectory;
- realized closed-loop labels from the simulator.

The state head is a measurement instrument, not the only endpoint. Validate its calibration against simulator truth and report latent-cost and realized outcomes even when the decoded head agrees.

## 6. Experimental protocol and gates

### Phase 0 — integrity and behavioral prerequisite

1. Reproduce the official checkpoint and planner configuration.
2. Freeze the simulator/task specification, checkpoint hash, camera/action preprocessing, factorial cells, and pilot split.
3. Replay every saved state/action pair and verify deterministic physics labels.
4. Run the frame/action nearest-neighbor contamination audit and trajectory-retrieval baseline.
5. Validate direct counterfactual prediction against each exact paired simulator future, including action-dependent change and preservation of action-independent scene content. Report action-response consistency and common-mode drift rather than accepting merely plausible rollouts.
6. Confirm that every eligible route scene contains both feasible safe and unsafe candidate families with matched endpoint/progress.
7. Verify that continuous clearance/impact sweeps cross the safety boundary and remain within the predictor horizon.
8. Measure unedited prediction discrimination, the preregistered `H:A` interaction, CEM ranking, and closed-loop damage/success.
9. Re-run clean Wall scenes only as a hook/cue-availability positive control.

**Go:** the model is action-sensitive within identical-state cells, preserves the relevant common scene content across paired counterfactuals, the ground-truth factorial interaction is nonzero by construction, and the model/optimizer exhibits a nontrivial safety-sensitive decision—at least 20% and at most 80% unsafe selections in an eligible challenge stratum—or a reproducible prospective-risk failure with a clear correct counterfactual.  
**Stop/redesign:** almost all candidates are safe/unsafe, the candidate action does not affect the forecast, alternative-action predictions are plausible but fail against paired simulator counterfactuals, common-mode drift dominates the interaction, the event lies outside the predictor horizon, contamination/copying dominates the retrieval comparison, or stimulus integrity fails.

### Phase 1 — network and time sweep

Capture, at each predictor block and imagined rollout step:

- block input residual;
- attention output;
- normalized MLP input;
- MLP output;
- block output;
- AdaLN action modulation;
- proprio lane;
- encoder spatial tokens as a cue-availability positive control.

Run exact matched donor patches safe→unsafe and unsafe→safe. Sweep whole state first, then token groups and spatial regions. Use held-out actions in the same scene and held-out scenes. Localize the **first temporally upstream site** that changes predicted clearance and candidate ranking, not simply the last site with the largest endpoint correlation.

The localization sweep must explicitly trace the relational interaction rather than only probe risk:

1. **Availability:** locate the hazard main effect in encoder/spatial tokens and the action main effect in the predictor action/AdaLN pathway.
2. **Interaction emergence:** estimate `I_l` and the `H:A` coefficient at every component and rollout step; identify where hazard and candidate trajectory first become jointly represented.
3. **Causal mediation:** patch hazard-bearing visual states while holding actions fixed, then patch action/AdaLN states while holding vision fixed. Compare effects in `H=1` and `H=0` scenes.
4. **Component path patching:** test encoder hazard tokens → attention output → MLP input/output → predicted future/cost, preserving all non-target activations from the receiver.
5. **Behavioral consequence:** require the localized component to change the forecast interaction and the rank of newly generated candidates, not merely a probe or decoder readout.

**Localization gate:** one frozen site/band must recover at least 30% of the donor–receiver gap on the primary forecast or cost endpoint in both directions, produce a nonzero scene-clustered DiD effect on forecast or rank, and mediate the hazard×action interaction through the predicted path above. Equal-norm random, wrong-layer, wrong-time, shuffled-donor, main-effect-only, and trajectory-nearest-neighbor controls must each remain below 10% median recovery. The confirmatory site is frozen after discovery with max-statistic permutation control across the initial layer×time×component sweep.

### Phase 2 — geometry tournament

At the frozen site, compare models with identical held-out splits and equal representation budgets:

1. ridge/logistic linear probe;
2. partial least squares or reduced-rank regression;
3. factorized coordinates for clearance, time-to-contact, and route direction;
4. local k-nearest-neighbor tangent directions or geodesic interpolation;
5. local action-to-state Jacobian/transition operator;
6. sparse transcoder coordinates only after Phase 1 passes.

Control covariates include pixels/obstacle visibility, action vector, speed, absolute position, goal distance/progress, rollout time, and task ID. The key incremental test is whether the representation predicts `risk(x,h,a)` after these covariates and, especially, within same-scene multi-action cells.

**Geometry gate:** held-out collision AUROC ≥ 0.75 or clearance `R² ≥ 0.30`, significant incremental prediction over nuisance controls, stable rank/topology across three discovery seeds, and above-chance transport to at least two held-out geometry/task families. Failure of cross-family transport must be reported as task-local geometry, not pooled as a universal direction.

### Phase 3 — transcoder, conditionally

Train an archetypal sparse transcoder only on the frozen MLP site selected above. Its target is the native MLP output from the action-conditioned predictor, not the encoder residual stream.

Minimum diagnostics:

- train/validation examples split by base scene and clip/trajectory;
- explained variance, reconstruction cosine, L0 activity, dead-feature fraction, feature stability across three random seeds, and downstream output distortion;
- top and bottom activating counterfactuals, not only hand-picked clips;
- feature selectivity for clearance/contact after nuisance residualization;
- causal comparison against the best low-rank/manifold coordinate with matched edit norm.

The released V-JEPA 2 transcoder's approximately 0.48 explained variance and 0.73 reconstruction cosine are useful feasibility reference points, not proof of adequacy. Require at least 0.45 held-out explained variance and 0.70 cosine to proceed, then interpret only delta edits because substantial native computation remains unexplained.

For feature `i`, intervene by adding a decoded feature change to the untouched native output:

\[
\widetilde{\mathrm{MLP}}(z)=\mathrm{MLP}(z)+W_{\mathrm{dec},i}\,\Delta f_i.
\]

Do **not** replace the MLP output with the transcoder reconstruction.

### Phase 4 — forecast and ranking intervention

For each eligible held-out cell, apply:

- risk removal from an unsafe candidate;
- risk insertion into the matched safe candidate;
- a calibrated intermediate sweep of edit strength `λ`;
- equal-norm random, opposite, wrong-feature, wrong-layer, wrong-time, and reconstruction-only controls;
- exact donor patch as an upper-bound positive control.

Measure predicted clearance/contact, complete future-latent distance to both endpoints, CEM cost, rank among the same candidate set, and motor/progress side effects.

**Causal-geometry gate:** the edit improves the appropriate destination distance by at least 30% of the exact-patch gap and flips the safe/unsafe pair ranking in at least 70% of eligible cells in the intended direction. Random/wrong-site controls must stay below 10% median gap recovery, and goal-progress/action-magnitude shifts must be small enough to reject a generic stop or vigor mechanism.

For a methodology claim, the frozen CGS edit must also outperform global mean-difference steering, the best probe direction, nearest-neighbor donor retrieval, and a preselected-site transcoder on held-out causal gap recovery per unit edit norm and off-target displacement. A single-checkpoint win is a case study; reuse on a second checkpoint or model family is required for the reusable-method claim.

### Phase 5 — persistent closed-loop behavior

Install the edit in every relevant predictor call during CEM sampling, every imagined rollout step, and every replan. Never patch actions from a donor and never replay a donor trajectory. The optimizer must generate fresh actions under the edited world model.

Compare:

- unedited planner;
- safety-state insertion/removal;
- exact-patch upper bound where implementable without future leakage;
- equal-norm random/wrong-site controls;
- a black-box planning-cost penalty baseline;
- an action-magnitude-matched “slow/stop” baseline.

Primary outcomes are realized collision rate and task success. Secondary outcomes are minimum clearance, chosen route, goal progress, path length, completion time, and action magnitude.

**Behavior gate:** on the frozen confirmatory set, reduce collision rate by at least 20 percentage points or 50% relative, with no more than 5 percentage points loss in task success and no more than 20% path/time overhead. The paired confidence interval for collision reduction must exclude zero. Risk-removal must produce the opposite directional effect where ethically and operationally safe in simulation.

### Phase 6 — replication

Replicate the frozen analysis on:

1. a second native obstacle family (`mw-reach-wall` plus the Wall task, or another supported MetaWorld wall task if checkpoint-compatible);
2. the released DINO-WM comparison checkpoint with the same planner and stimuli;
3. optionally V-JEPA 2-AC after the JEPA-WM mechanism is established.

The primary paper claim should distinguish model-specific localization from geometry that actually transfers.

## 7. Sample sizes and analysis

### Pilot

- 24 independent base scenes per task family.
- Two primary task families: vulnerable-entity route conflict and gentle/damaging contact.
- All factorial counterfactuals within each scene.
- At least six candidate futures per scene, including matched safe/unsafe and negative-control actions.
- Three representation/training seeds for geometry stability.
- Preregistered paired DiD effects and scene-clustered bootstrap/permutation intervals; no universal claim.
- Wall/`mw-reach-wall` retained only as positive controls, and humanized scenes retained only as a held-out transfer stratum.

### Confirmatory run

- At least 40 independent base scenes per family across three families, with discovery and confirmation seeds disjoint.
- Freeze checkpoint, sites, feature/geometry, edit strengths, nuisance model, exclusions, and metrics before confirmation.
- Use hierarchical bootstrap or a mixed-effects model with scene as the independent cluster and task family as a higher-level factor.
- Use paired permutation/McNemar-style tests for route/collision outcomes and cluster bootstrap intervals for continuous endpoints.
- Correct the discovery layer/site/time sweep with a max-statistic permutation; test only the frozen winner in confirmation.
- Report all eligible scenes, failures, intervention magnitudes, and per-family effects. Do not count CEM candidates as independent observations.

Forty scenes per family is a planning target, not a universal power guarantee. After the pilot, estimate the discordant-pair rate and intraclass correlation, then run a prospective paired power calculation before the confirmatory batch. Increase the sample rather than changing the effect-size threshold after looking at results.

## 8. Implementation plan

### Work package A — clean counterfactual benchmark

1. Add a versioned stimulus-manifest schema and deterministic replay test.
2. Implement the JEPA-WM-compatible RoboCasa route-conflict and gentle-contact factorial cells with DamageSim labels.
3. Generate same-state multi-action and hazard-relocation cells from exact simulator state.
4. Add simulator clearance/contact/TTC, impulse/health, route-side, endpoint, and progress labels.
5. Add feasibility screening for safe and unsafe routes within the predictor horizon.
6. Add frame/action contamination search, a trajectory-retrieval baseline, and asset/layout/trajectory-generator group splits.
7. Retain repaired Wall scenes as positive controls rather than the primary benchmark.

### Work package B — predictor causal map

1. Extend `jepa_patching_lib.py` with attention-output, MLP-input, and MLP-output sites.
2. Add bidirectional matched patches and token/time grouping.
3. Compute activation-level `H:A` interaction maps and scene-clustered DiD effects at every site/time.
4. Add main-effect and interaction-aware path patching from hazard tokens and action/AdaLN conditioning to predicted futures.
5. Record latent trajectory, decoded physics, CEM cost/rank, and destination-distance metrics.
6. Produce a discovery heatmap, then freeze the earliest site that causally mediates the hazard×action interaction.

### Work package C — geometry tournament

1. Implement a versioned CGS perturbation generator that emits physically valid `θ` sweeps from frozen states.
2. Build the layer × imagined-time × component interaction-echo tensor and local finite-difference/Jacobian estimates.
3. Implement grouped scene-disjoint splits, nuisance-residualized baselines, intrinsic-rank estimates, and chart-validity diagnostics.
4. Compare world-, camera-, gripper-, goal-, and candidate-path-relative labels under linear, low-rank, circular/factorized, local-manifold, and transition-operator models.
5. Test transport across action generators, hazards, appearances, object meshes, layouts, and task families; report failure as a function of distance from the fitted local chart.
6. Export the CGS emergence map, coordinate report, mediation map, and calibrated steering operator.
7. Preselect the smallest adequate geometry before sparse-feature training.

### Work package D — conditional transcoder

1. Train three archetypal transcoder seeds at the frozen predictor MLP site.
2. Evaluate reconstruction and feature stability on trajectory-disjoint validation data.
3. Associate features with simulator risk only after nuisance controls.
4. Run native-output-plus-delta edits against matched baselines.

### Work package E — behavior

1. Generalize the existing persistent planner hook to the selected site/geometry.
2. Run ranking-only tests on saved CEM candidate sets.
3. Run fresh closed-loop CEM planning under intervention.
4. Compare causal edits with cost-penalty and slow/stop baselines.
5. Replicate on the second task and comparison checkpoint.

## 9. Estimated compute and stopping rules

Approximate budget on one 24–48 GB GPU:

- integrity/stimulus pilot: mainly simulator and encoder caching, about 4–8 GPU-hours;
- full block×time causal sweep: about 8–16 GPU-hours after caching;
- geometry tournament: about 2–8 GPU-hours depending on local-manifold/Jacobian sweeps;
- one three-seed transcoder study: roughly 12–30 GPU-hours, highly dependent on token count and sparsity sweep;
- closed-loop confirmation: about 20–50 GPU-hours because CEM repeatedly calls the predictor.

These are engineering estimates, not reservations. Log actual wall time, GPU type, candidate count, cache hit rate, and cost after the first 24-scene-per-family pilot.

Stop or redesign before expensive training if any of these occurs:

- stimulus integrity or deterministic replay fails;
- exact-frame/action duplication or a nearest-neighbor retrieval baseline explains the apparent effect;
- the model is not action-sensitive on same-state multi-action cells;
- there is no nontrivial behavior/forecast contrast;
- no predictor site passes bidirectional patch specificity and the interaction-mediation gate;
- effects are explained by goal progress, action magnitude, or hazard visibility;
- the planner has no feasible safe alternative;
- an edit moves decoded predictions but not the planner's actual cost/ranking.

## 10. Expected outcomes and interpretation

### Strong positive

A mid-predictor MLP band contains a stable relational state for prospective clearance/contact. Exact patches and a compact model-native edit move forecasts and candidate ranking bidirectionally. Persistent insertion reduces collisions while retaining goal completion. This supports a meaningful causal-safety mechanism claim.

### Useful mechanistic negative

Exact state-conditioned patches work, but no portable linear or sparse coordinate transfers. This means the computation is real but distributed/local. The contribution becomes a boundary result about why global safety vectors fail and which local geometry succeeds.

### World-model/planner dissociation

The edit corrects forecasts but not CEM ranking, or correct ranking does not change closed-loop behavior. This localizes the failure downstream—to the cost, optimizer, horizon, or replanning dynamics—rather than to representation.

### Null result

The model sees the vulnerable entity, but no predictor site encodes action-conditioned risk beyond nuisance controls, and exact patches do not move the endpoint. Do not train a transcoder. The correct conclusion is that this checkpoint lacks the proposed prospective safety computation at the tested horizon/tasks.

## 11. First executable pilot

The immediate next experiment should be small and decisive:

1. Build 24 JEPA-WM-compatible RoboCasa route-conflict states and 24 gentle-contact states. Keep Wall/`mw-reach-wall` only as hook and cue-availability positive controls.
2. For every state, save the complete factorial cells: identical-state direct/detour or gentle/impactful candidates, in-path/out-of-path or fragile/durable conditions, mirrors, continuous boundary sweeps, and negative controls. Keep the hazardous interaction inside the predictor horizon and approximately match endpoint/progress.
3. Label each candidate by deterministic simulator rollout with clearance, TTC, contact impulse, health change, goal progress, and success. Replay twice and freeze the manifest.
4. Run the frame/action nearest-neighbor contamination audit and trajectory-retrieval baseline. Split by scene, asset/layout, and trajectory generator.
5. Verify action sensitivity, paired-simulator counterfactual validity, common-mode stability, ground-truth factorial validity, and a nontrivial unedited model/planner regime. Estimate the preregistered DiD on forecast, cost/rank, and realized damage.
6. Sweep every predictor block × imagined step at block input/output, attention output, MLP input/output, action/AdaLN modulation, proprio lane, and encoder hazard-token controls.
7. At each site, estimate activation interaction `I_l`; run bidirectional safe↔unsafe patches plus vision-fixed/action-changed and action-fixed/vision-changed path patches.
8. Freeze the earliest site that mediates a scene-clustered hazard×action DiD in predicted risk and candidate ranking while passing random, wrong-site/time, shuffled-donor, main-effect-only, and retrieval controls.
9. Fit only linear, rank-`k`, circular/factorized, and local-neighbor geometry at that frozen site. Do not train a transcoder during initial localization.
10. Run a calibrated `λ` sweep on held-out saved candidates and require approximately monotonic effects across clearance/impact-speed interpolations.
11. If—and only if—the edit changes fresh candidate ranking specifically, run fresh closed-loop CEM on held-out pilot seeds. Never patch or replay donor actions.

**Decision after the pilot:**

- If no causal interaction or exact patch effect: stop the representation project on this substrate.
- If exact patches work but geometry fails: expand state-conditioned/local geometry, not the dataset blindly.
- If geometry changes forecasts but not ranking: inspect cost and horizon before any transcoder.
- If geometry changes ranking and closed-loop behavior: freeze the design, power the confirmatory run, and train the transcoder at the localized MLP site.

That sequence turns the high-level idea into a falsifiable causal program and keeps the representational-geometry contribution tied to a behaviorally consequential result.

## 12. Phase-0 execution note — 2026-09-01

The first gentle-contact calibration is implemented under [`scripts/cgs_pilot`](/Users/stevenyang/Documents/rep_geometry_transcoder/scripts/cgs_pilot), with its protocol test in [`tests/test_cgs_pilot_protocol.py`](/Users/stevenyang/Documents/rep_geometry_transcoder/tests/test_cgs_pilot_protocol.py) and saved outputs in [`artifacts/cgs_pilot`](/Users/stevenyang/Documents/rep_geometry_transcoder/artifacts/cgs_pilot). This is a disclosed compatibility port of OopsieVerse `pick_egg` to official RoboCasa 0.2 `PnPCounterToSink(obj_groups="egg")`; it does not claim to reproduce a native OopsieVerse episode. It uses the exact robot-relative `robot0_leftview` camera transform published in JEPA-WM's saved `PnPCounterTop` model XML and the checkpoint's seven-dimensional DROID-compatible action units.

Calibration seeds 101 and 102 each produced a complete `hazard × candidate action` quartet and passed all frozen v0.2 stimulus gates: same-support placement, target visibility, one-patch edge clearance, one-patch hazard/control displacement, isolated target-state changes, intended contact pattern, force DiD above `5 N`, and bit-exact replay after full XML reload. Their force DiDs were `19.09 N` and `12.87 N`; both contact DiDs were `1.0`. These calibration seeds are not confirmatory data.

A separately labeled **DINO-WM DROID baseline** passed the method preflight on these two quartets: mean predicted action separation was `0.5794` latent RMS, and matched action–trajectory predictions beat action-swapped controls on both scenes by `0.0687` RMS on average. The mean vector-valued hazard×action interaction magnitude was `0.0884` RMS, while the scalar displacement DiD was small and inconsistent in sign. This establishes that the stimulus/action/inference pipeline has a nontrivial action-conditioned signal; it is not evidence about the JEPA-WM mechanism.

The released `jepa_wm_droid.pth.tar` contains the trained predictor but not its DINOv3 ViT-L/16 encoder. The official DINOv3 backbone is access-gated, and the current authenticated Hugging Face account has not accepted that model's license. No result will be labeled JEPA-WM until the exact backbone is available. Once it is cached, rerun the same Phase-0 gate on JEPA-WM, then proceed directly to the preregistered predictor layer×time×component interaction and bidirectional patching sweep. Do not substitute the DINO-WM baseline for that claim.


## Addendum (2026-09-02): protocol v0.4 — multi-patch hazard families for loop 2

Preregistered before any model analysis of these families. Motivation: the egg family (v0.2/v0.3) produced a structured null (see `claude_handoff/HANDOFF.md`, `LOOP1_SUMMARY.md`): action effect reproduced, hazard×action interaction small and not hazard-specific in the planner's own currency, no causally live site under single-site donor patching, no curved code, conceptor/Jacobian/modulation arms per `JEPA rep geometry.md`. The egg occupies one DINOv3 patch. Two families with the same task, camera, action chunks, gates and generator (`--obj-group`, `--control-min-m`):

- **cup** (`PnPCounterToSink(obj_groups="cup")`, control offset 0.12 m): probe 4/4 physical gates; batch seeds 501–540: 21/40 pass all physical gates (median 148 visible px, force DiD median 47 N).
- **bowl** (`obj_groups="bowl"`, control offset 0.22 m because the 12 cm control was contacted under the aggressive action): 15/40 pass (median 423 px, 2–3 patches wide, force DiD median 115 N).

Same v0.3 replay rule, H0′ null cells, independent label validation, fixed per-seed-hash discovery/confirmation split, calibration = none (all seeds held out; first 4 probe seeds 501–504 are still in the batch and are excluded as calibration). Hypotheses C1–C3 and thresholds as in `JEPA rep geometry.md` §4; C3 (continuous mine/theirs) requires a lateral-offset action variant that does not yet exist and is not claimed by this addendum. Pipeline `scripts/cgs_pilot/run_loop2_family.sh`: augment → validate → jitter → merge v0.3 → latent cache → counterfactual/planner-currency gate → retrieval baseline → localization → action-Jacobian sonar → modulation-entry patching → step-0 donor patch sweep, discovery seeds only; confirmation sealed. Decision rule identical to loop 1. Known limitations carried over: DROID-only checkpoint (OOD renders), two action chunks, hazard pixel clustering.

### v0.4 amendment (2026-09-02 15:30 UTC, before any cup gate result was seen)

The fixed hash split of the 15 admitted cup scenes (seeds 501–540) yields only 5 discovery scenes, below the preregistered ≥16-scene rule for the counterfactual-validity gate. The cup seed range is extended to 541–600 under the same generator, control offset (0.12 m), validator, and split rule (`sha256(f"{split_seed}:{seed}")[0] < 128`). No cup model output had been computed when this extension was decided; nothing is selected on results. Bowl is dropped from loop 2: 1/27 seeds pass the independent label checks under the egg-calibrated validator, and its thresholds are not loosened.

### Behavioural-target gate (B-gate) — preregistered 2026-09-02 16:40 UTC, before any cup model output

Motivation (user, 2026-09-02): in the VLA work the behavioural target was too ambiguous, so representation methods were chasing a behaviour the model did not clearly have. Loop 1 here shows the same failure: the egg-family model output reproduces only 4.5–6.5% of the true hazard×action interaction energy (interaction NMSE 0.94–0.96), hazard-specific recovery is ≤ 0 (Rec_h1 − Rec_h0 = −0.009 … −0.015), and the CEM ranking never flips (rate 0.0) — yet localization, patching, conceptor, higher-order, bilinear, Jacobian and modulation arms were all run on it. Those arms can only return structured nulls when there is no behaviour to explain.

Rule (`scripts/cgs_pilot/behavior_gate.py`, thresholds fixed in the file): after the counterfactual-validity gate on discovery scenes and before any mechanism arm,
- Tier 1 (all required): n ≥ 16 scenes; interaction NMSE median ≤ 0.75 (model reproduces ≥ 25% of the true interaction); Rec_h1 − Rec_h0 > 0 with sign-flip p < 0.05, not reversed by the H0′ contrast.
- Tier 2 (reported; required for a planner-level claim): CEM ranking flip rate H1 vs H0 ≥ 0.25, or normalised energy gap H1 vs H0 > 0 with p < 0.05.
- FAIL ⇒ mechanism arms are not run; the loop's finding is "no unambiguous behavioural target on this stimulus/model"; the remedy is a stimulus or model change (larger/multi-patch hazard, in-distribution or second-architecture checkpoint, unseen-action family), never another interpretability method.

Retroactive application (egg family): wave 1 n=21 FAIL (captured 4.5%); wave 2 discovery n=34 FAIL (6.5%); wave 2 all n=66 FAIL (5.7%). Loop-1 mechanism results are therefore reclassified as *technique controls on a no-behaviour stimulus*, not as tests of the hypothesis. Consequence for the running egg wave-2 mechanism stage: localization is allowed to finish (its dump with H0′ cells is reused), the 34-scene patch sweeps and geometry are skipped. Cup family (loop 2): B-gate inserted between the gate and the mechanism arms.

### Protocol v0.5 — CGS as a cross-substrate benchmark (user decisions 2026-09-02 ~19:00 UTC)

Reframing (user): the goal is not to show that JEPA is superior, nor to reproduce BadDreamer, but to test whether the representational-geometry "sonar" pipeline (validity gate → behavioural-target gate → retrieval control → localization → on-manifold patching → geometry arms) replicates across world-model architectures, domains and contexts, with JEPA-family models as the core line of study and autonomous driving as the flagship context because of its real-world stakes.

Decisions:
1. The RoboCasa egg / cup stimuli are retired. Substrate 1 (JEPA-WM DROID on egg) closes as a structured null with full technique controls (`LOOP1_SUMMARY.md`); cup loop 2 and the egg E2/E3 arms are cancelled before any model output (no result-dependent selection).
2. Substrate matrix (each cell runs the same frozen pipeline; a cell that fails the B-gate is reported as "no behavioural target", never as "no mechanism"):

| Substrate | Family | Action-conditioned | Domain / stimuli | Role |
|---|---|---|---|---|
| JEPA-WM DROID (DINOv3 + AdaLN predictor) | JEPA | yes | RoboCasa egg (retired) | substrate 1: structured null + controls |
| V-JEPA 2-AC (DROID) | JEPA video | yes | same admitted egg scenes | architecture transfer on a fixed stimulus (stimulus- vs model-specific null) |
| DINO-WM | JEPA-style latent | yes | egg preflight only | optional |
| LeJEPA / LeWorldModel line | JEPA | to verify | to verify weights | candidate |
| JEPA-family driving WM (V-JEPA 2-AC or DINO-WM recipe fine-tuned on CARLA hazard data incl. collision futures) | JEPA | yes | CARLA hazard×action factorial (NeuroNCAP/HUGSIM for real-video follow-up) | flagship cell; must be trained (no open action-conditioned JEPA driving WM with weights exists as of 2026-09-02) |
| MILE (CARLA RSSM) | non-JEPA | yes | CARLA | architecture-transfer control |
| VaViM/VaVAM (+ optional BadDreamer planted backdoor) | non-JEPA AR token WM | no (planner-side candidate scoring) | real nuScenes with inserted actors | real-video, human-relevant control; planted-mechanism positive control |

3. Ordering: V-JEPA 2-AC on the existing egg set (cheap; first transfer cell) → driving substrate build (data generation with collision futures, JEPA WM fine-tune, factorial) → non-JEPA controls. Every cell: ≥16 discovery + ≥16 confirmation scenes, B-gate before mechanism arms, sealed confirmation.

### Protocol v0.6 — support-dose experiment, ≤ 8 GPU-hours (preregistered 2026-09-02 ~20:30 UTC; target: NeurIPS workshop submission 2026-09-05)

User constraint: ≤ 8 GPU-hours total; subset-first ("if it does not work end to end on a subset it will not work on the full data"); elegant design using the representational-geometry tricks from the VLA work (`vla_learnings.md`, coast.md, EBM doc).

**Design.** Three predictors that differ in exactly one thing — the fraction of training futures containing a hazard collision: **0 %, 10 %, 50 %** (arms S0, S10, S50). Everything else held fixed: frozen DINOv3 ViT-L/16 encoder (so input tokens are identical across arms), same predictor architecture (JEPA-WM `VisionTransformerAdaLN`, small: depth 6 / width 512, or depth 12 / 1024 if the budget allows), same clip count, seeds, schedule, optimiser. Stimulus simulator: MetaDrive (pending the headless spike; fallback = the same design in RoboCasa with a non-egg hazard, documented as fallback before any result). Factorial per scene: {H1 pedestrian in the ego lane at TTC ≈ 2 s, H0 pedestrian on the sidewalk, H0′ second off-lane pose at matched pixel displacement} × {A0 brake chunk, A1 constant-throttle chunk}; identical context frame across the 6 cells; ground truth = min distance, contact flag (contact only in (H1,A1)); pedestrian ≥ 3×3 DINOv3 patches enforced by the validator.

**Pipeline per arm (frozen, identical):** counterfactual-validity gate → behavioural-target gate (`behavior_gate.py`, thresholds unchanged) → retrieval control incl. nearest-*training*-clip audit → localization (LOSO projection, sign-flip max-T, BY; magnitude ratios) → step-0 donor patching with the full control set → geometry arms (LOSO contrast subspace "sonar", conceptor AND-NOT, cross-token action-Jacobian, sliced-W2). Discovery/confirmation split by hash; confirmation sealed until a band is frozen on discovery.

**Cross-arm geometry (the new element).** Because encoder tokens are identical across arms, any difference in the hazard×action subspace is a predictor property. Compare arms with rotation-invariant statistics: principal angles and Grassmann distance between the S0/S10/S50 interaction subspaces per site; Procrustes transport of the S50 subspace into S0/S10 and the transported projected cosine; conceptor quota and cross-similarity; sliced-W2 on frozen projections. Zero-shot cells: JEPA-WM DROID and V-JEPA 2-AC on the same stimuli (released checkpoints = zero-support anchors, inference only).

**Hypotheses and decision rules.**
- H1 (support): B-gate PASS in S50; interaction NMSE and Rec_h1 − Rec_h0 ordered S50 > S10 > S0 (scene-clustered, sign-flip; report ordered-trend p).
- H2 (planner): CEM brake/go flip rate ≥ 0.25 in S50 under H1, not H0′.
- H3 (mechanism, only if H1): a step-0 site band whose real-donor patch recovers ≥ 30 % of the H1−H0 gap with controls < 10 %, on discovery, reproduced on confirmation with zero refitting.
- H4 (geometry): the S50 interaction subspace is (a) low-rank (quota ≤ 8 for 80 % capture), (b) absent or low-gain in S0 (transported cosine ≤ random control), (c) rotates or not across layers (principal angles reported, no threshold).
- H5 (transfer): the released JEPA-WM/V-JEPA 2-AC checkpoints fail the B-gate on these stimuli (zero support) — a prediction, not a hope.
Negative outcomes are reported per the interpretable-outcome table in `vla_learnings.md`.

**Budget.** Data generation CPU only (10k clips × ~20 frames; MetaDrive ~hours). Feature precompute ≈ 0.5 GPU-h. Training 3 × ≤ 1.5 GPU-h. Factorial encoding + gates + sonar arms ≈ 1–2 GPU-h. Total ≤ 8 GPU-h. Any overrun is reported, not hidden.

### Protocol v0.7 — matched-consequence arms (SOLID vs GHOST); supersedes the 0/10/50 % dose design of v0.6 (preregistered 2026-09-02 ~21:40 UTC, before any driving data exists)

Adopted from the VLA playbook (`/Users/stevenyang/Documents/mechinterp-vla/docs/PREREG-cross-architecture-safety-mechanism-playbook-pilot-2026-09-02.md`, read-only): the treatment must be *relational*, not "more safety data". Arms share the complete multiset of training images and action chunks; only the mapping from hazard to physical consequence differs.

**Arms (paired by training seed, identical initialisation, clip set, action marginals, optimiser, update count; final-checkpoint-only rule):**
- **SOLID:** pedestrian is a physical collider; (hazard-in-lane, throttle) futures contain the collision (ego decelerates/stops/deflects, pedestrian displaced).
- **GHOST:** same scenes, same renders, same actions; the pedestrian is intangible (collision disabled), so every future equals the corresponding no-hazard future. Renders are frame-identical to SOLID up to the first contact frame.
- Optional anchor **NONE:** the same clips with the pedestrian removed (appearance control; run only if budget allows).
Hazard clips are 50 % of each arm's training set; the remaining clips are hazard-free and shared verbatim.

**Registered estimand.** Three-way interaction arm × hazard (H1 vs H0/H0′) × action (A1 vs A0) on (i) the model's predicted-latent interaction (interaction NMSE / cosine / Rec against the SOLID physics as truth for both arms), (ii) the planner's brake/go ranking. Prediction: SOLID passes the B-gate; GHOST fails it with the identical inputs. (H0′ = matched-displacement off-lane pose, both arms.)

**Geometry prediction (the paper's mechanistic claim).** Because encoder tokens are identical across arms and the training images are identical, any hazard-*appearance* subspace (H1 vs H0 main effect at the pedestrian tokens) should be shared between SOLID and GHOST (high principal-angle overlap, transported projected cosine ≈ source), whereas the hazard × action *relational* subspace should exist only in SOLID (GHOST transported cosine ≤ matched-random control). Reported with principal angles / Grassmann distance, Procrustes transport, conceptor cross-similarity, and the cross-token action-Jacobian sonar (ΔJ_rel = J(H1) − J(H0) restricted to route tokens).

**Stage gates (stop/go, in order).**
- P0 stimulus feasibility: same context frame across the 6 cells; pedestrian ≥ 3×3 patches; SOLID contact only in (H1,A1); GHOST contact never; bit-exact replay in two processes.
- P1 data feasibility: equal clip counts, lengths, action marginals across arms (report KS distances); SOLID/GHOST frames identical before first contact (pixel-diff check).
- P2 one paired seed: both arms reach equal hazard-free prediction loss (equivalence margin frozen at 5 % relative); SOLID B-gate PASS, GHOST FAIL on discovery scenes. If SOLID fails → the recipe cannot learn the consequence at this budget; stop and report.
- P3 paired seeds: 3 paired seeds if budget allows (≤ 6 small trainings); the seed pair is the independent unit for the arm contrast (report every seed; exact sign-test resolution 1/8); within-arm scene-clustered statistics for the factorial.
- P4 representation discovery on discovery scenes only (geometry ladder: direction → low-rank → conceptor → rotating subspace → local/curved → distributed); one joint correction over site × step × group × geometry class × rank.
- P5 donor-based then donor-free intervention on confirmation scenes with zero refitting; selectivity: H0/H0′ and hazard-free behaviour within frozen equivalence margins; controls (equal-norm orthogonal, wrong-layer, wrong-token, wrong-step, sham, reverse) must fail.

**Zero-shot anchors:** released JEPA-WM DROID and V-JEPA 2-AC on the same factorial (inference only) — predicted to fail the B-gate; the egg loop (`LOOP1_SUMMARY.md`) is the manipulation-domain anchor.

Budget unchanged (≤ 8 GPU-h): 2 arms × ≤ 3 seeds × small predictor on cached features.

#### v0.7 amendment (2026-09-02 ~21:55 UTC, before any driving data): reversed-assignment arms replace SOLID vs GHOST

User check: plain SOLID/GHOST leaves the arms' *future* multisets unequal (only SOLID contains collision visuals), a confound of the same family as the VLA visible-vs-hidden test. Following the VLA playbook's equal-multiset rule, the arms now flip an identity-to-consequence assignment:
- **Arm A (PED-solid / OBJ-ghost):** pedestrian is a collider; a matched inanimate object (same collision envelope, lane position, size) is intangible.
- **Arm B (PED-ghost / OBJ-solid):** the reverse.
Each training clip exists in both arms with identical images (up to first contact), identical actions, and the identical multiset of collision / non-collision futures; only which identity the consequence attaches to differs. Test factorial: {H1_ped in lane, H1_obj in lane, H0 off-lane pedestrian, H0′ matched-displacement pose} × {A0 brake, A1 throttle}; ground truth from each arm's own physics for the behavioural gate, plus the cross-arm contrast. Registered estimand: arm × identity × action three-way interaction on predicted-latent interaction and planner ranking (double dissociation predicted). Geometry: the relational hazard×action subspace should attach to pedestrian tokens in Arm A and to object tokens in Arm B, with the same subspace geometry after Procrustes transport between arms; appearance subspaces shared. Stage gates P0–P5 unchanged; P0 adds: object and pedestrian both ≥ 3×3 patches and matched collision envelope (≤ 10 % radius difference); P1 adds: collision-future counts equal across arms.

#### v0.7 stimulus instantiation (MetaDrive; 2026-09-02 ~22:50 UTC, after the feasibility spike, before any factorial data)

Simulator: MetaDrive 0.4.3, headless EGL, 256×256 RGB, 40° FOV, 10 Hz physics, model step = 3 sim steps, clip = 1 context + 3 future frames. Ego seeded to ~8 m/s. Hazards: MetaDrive `Pedestrian` and an envelope-matched cone (same collision cylinder r 0.35 m, h 1.75 m; visual rescaled to match bbox within ~2 px). Ghost = collision group paired only with terrain (no flag, no response; renders identical to solid up to contact). Frame gate (preregistered from the spike's jitter measurement): physics/ego state bit-exact on replay; renders may differ on ≤ 16 pixels per frame with |diff| ≤ 8/255; anything worse fails closed. Positioning gate: hazard bbox ≥ 48×48 px (≥ 3×3 DINOv3 patches) in the context frame. Factorial: 8 cells per scene (levels 0 = sidewalk, 1 = pedestrian in lane, 2 = H0′ matched-displacement sidewalk pose, 3 = cone in lane) × (A0 brake, A1 throttle); contact permitted only in (level 1, A1) for arm A and (level 3, A1) for arm B. Targets: 120 factorial seeds (≥ 16 discovery + ≥ 16 confirmation after gates), 6,000 training clips per arm rendered in pairs from identical seeds.

#### v0.7 pre-outcome engineering tolerances (2026-09-03 ~00:20 UTC, after the 20-seed stimulus pilot, before any model output)
- Frame gate amended: renders may differ on ≤ 16 pixels per frame with |diff| ≤ 16/255 (pilot: renderer jitter up to 14/255 on 1–3 px; ego states bit-exact 320/320 cells); ghost/no-contact cells must match the hazard-free ego trajectory within 1e-6 m (pilot: 278/280 ≤ 1e-9, two at 1e-8/1e-7 from Bullet contact ordering). Physics replay remains bit-exact.
- Visibility gate amended: hazard silhouette must cover ≥ 6 DINOv3 patches in the context frame (pedestrian 23×80 px, cone 30×79 px at the calibrated 9.0 m; the earlier 48×48 rule assumed a square footprint). Render-difference footprint (≈ 80×80 px) recorded, not gated.
- Calibration frozen: hazard 9.0 m ahead at the context frame (ego 8.3 m/s), throttle contact at future sim step 7 (model step 3) in 20/20 seeds; sidewalk lateral 2.25 m; H0′ = mirror pose across the ego-lane centreline (centroid displacement matched to 0.1 %). Hazard body immovable (ego stops and pitches; caveat: the consequence is dominated by ego motion, the pedestrian is not displaced; a knock-over variant is future work).
- Training set: kind-paired hazard clips (each hazard seed rendered as pedestrian and cone at the identical pose in both arms) so colliding-future counts are equal across arms by construction; in-lane distance and throttle probability biased so that ≈ 25 % of hazard clips collide (support is the manipulated quantity; arms remain identical multisets). Target ≈ 1,600 clips per arm (disk-limited: full-grid DINOv3 features 2.1 GB per 1k clips).
- Released-checkpoint zero-shot cells on the driving factorial are dropped: JEPA-WM DROID / V-JEPA 2-AC take 7-D end-effector actions, so a 2-D [steer, throttle] factorial has no valid action interface for them. The zero-support anchors are the egg cells (JEPA-WM and V-JEPA 2-AC, `LOOP1_SUMMARY.md` App. B–D). Within the driving cell the zero-support contrast is the GHOST identity inside each arm (pedestrian in arm B, cone in arm A).
- Training-data support (pre-outcome, 2026-09-02 18:45 UTC): the 100-seed biased pilot at in-lane 6–14 m / p_throttle 0.5 gave 12.1 % colliding hazard clips (identical seeds collide in both arms); frozen for the full run: in-lane distance 6–10 m, p_throttle 0.6 (expected ≈ 18 % of hazard clips colliding, ≈ 150 per arm), 1,200 seeds ≈ 1,600 clips per arm. Validator amended tolerances: pilot passes 20/20 seeds, both arms.
- Cross-arm geometry tests, exact definitions frozen (2026-09-02 ~19:40 UTC, before any driving model output; `geometry_cross_arm.py`, self-tests planted/null pass): Procrustes transport fitted on hazard-free rows only; action subspace from hazard-free cells only; test (i) shared appearance registered on the brake-cell contrast (the solid identity's action-averaged main effect contains half the interaction by construction); test (ii) double dissociation DD = (excess_ped − excess_cone)_A − (…)_B with a refit permutation null (ped↔cone relabelling, LOSO subspaces refitted), max-T over sites; test (iii) cross-identity transport reported without threshold; test (iv) conceptor quota at 80 % retained energy. Uninformativeness diagnostic: held-out transport residual ≈ 1 ⇒ test (i) uninformative, not negative.

#### B-gate amendment T1c′ (2026-09-02 ~19:50 UTC; before any full driving model; motivated by the 2-minute dry-run pilot on 398 clips)
Observation in the dry run (pilot models, not results): arm A reproduces most of the hazard×action interaction (interaction NMSE 0.80; ranking flip rate 1.0) while its recovered-fraction specificity is negative (Rec_h1 − Rec_h0 = −0.20): with an immovable hazard, the (H1,A1) future is a much larger displacement than the free-driving future, so "fraction recovered" penalises the cell that carries the consequence. Amendment: Tier 1 = T1a AND T1b AND (T1c OR T1c′), where **T1c′ (identity contrast, the design's registered estimand)** = per discovery scene, the solid identity's interaction NMSE is lower than the ghost identity's interaction NMSE, scene-level sign-flip p < 0.05. T1b (≥ 25 % of the true interaction reproduced) is unchanged and remains the magnitude criterion. Egg verdicts are unaffected (no cross-identity gate exists there). The v0.8 merge frame constant is corrected to the preregistered 16/255 (it had been left at 8/255 in code).

#### COAST-style before/after endpoint (preregistered 2026-09-02 22:50 UTC, during training, before any gate verdict)
Task-level endpoint analogous to COAST's unsteered-vs-steered success table (arXiv 2605.17144): **planner safe-choice rate** = fraction of scenes in which the model's own planner currency (CEM cost ranking, as in `planner_currency.py`) prefers the brake chunk over the throttle chunk, per hazard level (pedestrian in lane, cone in lane, sidewalk, H0′). Rows: each B-gate-PASS model unsteered; steered donor-free (own-arm relational conceptor, or arm-A relational conceptor transported into arm B via the cross-arm Procrustes map) at β chosen on DISCOVERY scenes by a frozen rule (smallest β whose steered−unsteered safe-choice DiD CI excludes 0), band = frozen localization/patch band or top-3 step-0 relational sites; suppression on the solid arm; controls (matched-spectrum random conceptor, rank-one, wrong-site, wrong-group, sham β=0). Reported on sealed CONFIRMATION scenes with zero refitting. Selectivity: steering must raise safe choice under the solid identity in lane and leave sidewalk / H0′ / hazard-free prediction (ARC, drift) within a 0.10 equivalence margin. Scope: open-loop ranking in the model's currency, not closed-loop task success; released checkpoints are not usable as driving baselines (7-D action space) — their egg rows are reported as "failed behaviour gate, not evaluable".
- Success targets for the COAST-style table (stated 2026-09-02 22:55 UTC, before any gate verdict; targets, not gates): arm A unsteered safe-choice rate on pedestrian-in-lane ≥ 0.75; arm B unsteered → steered (arm-A relational subspace transported) ≥ +0.25 absolute with scene-clustered CI excluding 0 and sidewalk/H0′ within ±0.10; suppression on arm A ≤ −0.25 on pedestrian-in-lane with cone unchanged; all controls within ±0.10. Egg baselines for the table: planner flip rate 0 % (JEPA-WM n=21/34/66; V-JEPA 2-AC n=34), captured interaction 4.5–6.5 % / 0 %.
- COAST-style table currency (definitional, 2026-09-02 23:50 UTC, before any gate verdict): primary goal latent = the scene's hazard-free THROTTLE true future ("progress" goal), so brake is preferred only when the model predicts that throttle leads away from progress (the collision consequence); under this currency the unsteered rates are expected ≈ 0 under H0 for both arms and the unsafe baseline is arm B on pedestrian-in-lane. The brake-future goal (under which H0 safe-choice ≈ 1 by construction) is reported as secondary rows only.
- Mechanism-stage time bound (2026-09-03 04:00 UTC, before any mechanism result; deadline-driven, not result-driven): step-0 donor patch sweep on all 54 discovery scenes and the cross-token action-Jacobian sonar on the first 8 discovery seeds (step 0) for the seed-0 pair only; seeds 1–2 run localization, geometry tournament and cross-arm geometry (no patching/Jacobian). Patch pace 147 s/scene; Jacobian ≈ 20 min/scene on the egg.
- Added descriptive analyses (2026-09-03 ~05:40 UTC, before any mechanism/cross-arm result was read; `LRH.md` §2): per-scene hazard×action DiD field characterisation (PC1 fraction, spherical variance vs uniform), residual cosine (discovery-field mean subtracted; copy-delta baseline scored identically), and cross-arm relational transport (source-arm cosine weights over discovery scenes → synthesised target-arm DiD; residual cosine vs true; nulls: scene permutation, isotropic Gaussian, covariance-matched Gaussian). Gate thresholds unchanged.
