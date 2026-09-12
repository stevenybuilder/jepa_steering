# Correcting Imagined Futures in Frozen JEPA World Models

*Development-results draft | September 8, 2026 | Four-page core text; figures, references, and appendix supplied separately.*

## Abstract

Latent world models let an agent evaluate actions through predicted futures, but the structure that makes a representation informative need not make an intervention useful. We study structured activation corrections in frozen JEPA world-model predictors using completed offline comparisons on Reach, Reach-Wall, and Push-T. Five controlled sweeps separate vision-action coupling, local action-response geometry, rank, depth support, and spatial support. Original rank-4 edits reduce six-step proprioceptive embedding error by 2.89% and 2.14% on the two reaching tasks, while Push-T is inconclusive. Spatial permutation and component controls reveal task-dependent correction structure. We also implement an offline-fitted fixed-response operator that removes online response probes and native-shadow forecasts; its separately evaluated reductions are 2.36% and 2.19%. Local reconstruction accuracy can improve substantially without improving the forecast endpoint, and its ordering can reverse with numerical precision. These results identify correctable forecast error and limits of geometry-based diagnostics. They do not establish improved planning success, independent training-seed replication, or transfer to unseen tasks.

## 1. Introduction

A map is useful because it preserves the relationships needed to choose a route. It may distort distances yet preserve connectivity; a detailed map can still mislead a traveler if it places a crossing incorrectly. A latent world model faces a related requirement. It compresses observations into representations, predicts where candidate actions lead, and supplies those predictions to a planner. Improving this internal map requires knowing which relationships govern useful forecasts, rather than assuming that a more faithful reconstruction will necessarily support a better decision.

Tong, Fan, and colleagues use Plato's cave to motivate learning from visual experience beyond linguistic descriptions [1]. Our question begins after a model has learned from observations and actions: what makes its learned representation usable for prediction and, eventually, control? The analogy motivates a testable distinction between representational fidelity, forecast correction, and decision quality. We do not assume that any of these implies the next.

This question is timely because work on physical representation analysis, activation steering, and latent planning now offers concrete ways to connect these levels. JEPA-WM studies training and planning choices [2]; video-model interpretability identifies distributed physical variables [3]; COAST changes frozen policy activations [4]; manifold steering examines geometric paths through representations [5]; and ACPC relates JEPA rollout consistency to planning-cost changes [6]. Together these motivate interventions in the dynamics predictor, where an edit changes the imagined consequences of actions before a planner chooses among them.

We present three contributions, scoped to the completed development evidence:

- **A controlled study of correctable forecast structure.** We separate rank, depth, spatial support, and vision-action pathways, using paired native, zero-dose, permutation, and matched-random controls to identify where the tested corrections help or harm.
- **An offline-fitted distributed correction without online response probes.** We replace candidate-specific response estimation with a fixed coefficient map, and separately measure its forecast effects. This is an implemented operator, with behavioral utility still under evaluation.
- **A boundary on geometric reconstruction as a steering diagnostic.** We document a precision-dependent reconstruction advantage that does not translate into useful forecast correction in the tested setting.

These contributions are empirical and methodological. We do not claim a new general theory of representation geometry, a four-dimensional physical state, or a demonstrated universal steering recipe.

<!-- pagebreak -->

## 2. Study design and structured intervention

We use JEPA-WM for its released predictors, native planner, and inspectable task interfaces [2]. Its world-model training freezes a pretrained visual encoder without a moving EMA teacher. We additionally freeze all base-model modules and evaluate recorded action sequences. The primary endpoint is proprioceptive embedding mean squared error at horizon six (H6), rather than decoded physical-state error or task success. Visual embedding error is a companion endpoint. We use the corrected author-aligned input and rollout contract. Completed primary pools contain 33 Reach, 27 Reach-Wall, and 21 Push-T trajectories. They are development data previously used for intervention assessment, not untouched confirmation. Multiple prefixes and arms reuse these trajectories.

We vary one component against a category-specific reference, rather than repeating the authors' training ablations (Table 1). BF16 is primary and FP32 is a sensitivity condition, fixed before evaluation rather than selected from favorable outcomes.

**Table 1. Intervention design space.** The five original categories are complete on all three primary tasks in both precisions. Controls match relevant support, rank, and energy within a sweep; this is not a full factorial over all categories.

| Component | Evaluated choices | Main controlled distinction |
|---|---|---|
| Vision-action coupling | Visual, action, joint, equal-energy joint, permutations | Pathway and spatial alignment |
| Local response geometry | Linear, cubic, projected cubic, reflected curvature | Reconstruction versus forecast effect |
| Rank | 1, 4, 8, and matched random | Correction capacity at fixed support |
| Depth support | B0-B5, B2+B3, all six | Which predictor blocks receive the edit |
| Spatial support | 1 patch, 16 contiguous, 16 scattered, 256 | Which patch positions receive the edit |

**Original response-based correction.** At the reference site, the rank-4 operator edits the newest 256-patch field at predictor block B3, the fourth block, during imagined step H3. In the reaching models the field has 400 features per patch. Four fitted activation directions define one distributed addition; 102,400 scalar coordinates describe its support, not 102,400 independently tested interventions. A fitted readout estimates a correction target, and finite differences estimate its downstream response. A damped local inverse supplies the coefficients. Repeated response forecasts, including diagnostic controls retained in that implementation, made the original procedure costly inside planning.

**Fixed-response successor.** We retain the task-specific basis B, dose, and site. The readout phi(h) contains an intercept and three PCA scores; a mean response J_bar is estimated offline. The coefficient map is A = (J_bar^T J_bar + lambda I)^-1 J_bar^T E, where E maps the readout features to the target correction. At inference, c_raw = A phi(h); the distributed field B c_raw is normalized to the prescribed dose before addition. A is 4-by-4. Figure 6 shows the actual basis loadings, without assigning physical semantics. The hook uses the native activation in the same forecast, with zero online response probes and zero separate native-shadow forecasts. The full candidate batch and planner schedule remain intact. Random controls receive their own equally budgeted calibration.

Calibration uses 24 fitting families and eight response-audit families, each with four fixed examples. The audit families are held out only from response calibration; they contributed to the inherited basis/readout and are not a fresh test of the whole method. The successor receives a separate four-arm offline evaluation on the same exposed Reach and Reach-Wall pools. It is a changed operator, not a numerically equivalent implementation of the original.

<!-- pagebreak -->

## 3. Completed offline evidence

We report paired-lineage differences scaled by observed native error. A positive percentage indicates lower error. The source analyses use simultaneous 95% intervals within each task, category, and precision, jointly covering registered contrasts and both embedding endpoints. Original sweeps, the Reach combined stage, and the successor have separate inference families. We reuse those intervals without pooling families or conducting new tests for this draft. Figures show selected comparisons alongside controls; the accompanying data export retains every registered contrast in the included reports.

**Compact corrections improve some forecasts.** Original rank 4 reduces Reach error by 2.887% [2.276, 3.499] and Reach-Wall error by 2.137% [1.645, 2.629]. Push-T's 0.198% [-0.059, 0.455] is inconclusive. Reach rank 1 also passes its native/random eligibility checks; rank 4 was selected by a predeclared smallest-eligible-rank equivalence rule, not because rank 1 was ineffective. Rank 8 does not establish an advantage over rank 4 on Reach. The results establish a correctable component of error under these interventions, not the intrinsic dimensionality of physical state.

**The cheaper successor has its own positive offline result.** Fixed-response rank 4 reduces error by 2.360% [1.966, 2.755] on Reach and 2.192% [1.798, 2.587] on Reach-Wall in BF16 (Figure 2). Its advantages over matched random are 1.203% [0.962, 1.445] and 1.475% [1.185, 1.765], expressed relative to native error. Random corrections themselves can help, so the entire native-relative gain cannot be attributed to learned structure. FP32 reductions are 2.324% and 2.131%; Reach's FP32 contrast against random does not clear the frozen practical minimum. No direct equivalence test between original and successor is established.

**Spatial arrangement and pathway choice matter.** On Reach, unscaled joint coupling beats the spatially permuted joint arm by 2.016% [1.089, 2.942] of native error. Permutation preserves broad support and visual edit norm while changing patch assignment. On Push-T, visual-only steering increases error by 1.022% [0.326, 1.718], whereas action-conditioning-only steering decreases it by 0.336% [0.052, 0.619], below the useful-effect threshold. Joint steering is worse than action-only by 0.974% [0.306, 1.641]. These controls reveal structured, opposing effects without yet identifying a semantic cause such as contact or object location (Figure 3).

**Broad support does not require every layer.** In the Reach rank-1 depth sweep, block 0 reduces error by 3.026% [2.124, 3.929]. Several singleton blocks work. In the separate spatial sweep, only the all-patch arm passes the frozen gates. These observations distinguish spatial support from depth distribution; they do not establish a unique best block or validate moving the rank-4 successor to block 0 (Figure 4).

**Combination and geometry have important limits.** Original coupling plus rank 4 improves Reach error by 4.824% [3.668, 5.980], beating matched random and both component-removal arms. Those removals also remove energy, so this does not establish equal-energy synergy. The new combined successor has not been evaluated. Separately, Push-T cubic reconstruction has about 1,192-fold lower omitted-activation error than equal-anchor linear reconstruction in FP32, but about 35% higher error in BF16. Its FP32 forecast gain is only 0.0046% [-0.0032, 0.0124]; neither precision establishes useful forecast correction (Figure 5). Local reconstruction fidelity is therefore insufficient to select an effective intervention in this setting.

<!-- pagebreak -->

## 4. Interpretation, practical relevance, and limits

The evidence supports a specific claim: some forecast error has correctable internal structure, and similarly sized interventions differ with pathway and spatial assignment. This addresses the forecast and selectivity portions of our original objective. It does not yet answer whether those corrections improve decisions. The map analogy makes that distinction explicit: changing an internal map matters when it changes the relationships needed for a useful prediction or action.

**Representation, mechanism, and control.** Joseph et al.'s Physics Emergence Zone concerns video encoders [3]; we intervene in a dynamics predictor downstream of DINOv2. Our B3 is not an identified PEZ. A compact correction of a particular error need not span the full representation required to control an arbitrary physical variable. Our basis is an empirical coordinate system, not a recovered semantic manifold. Low-rank representation adaptation already has precedent [11]. The distinctive evidence here concerns controlled effects within action-conditioned forecasts, rather than a new PCA or inverse formula.

A useful intervention does not require a complete circuit map. A claim about the physical mechanism does require targeted causal validation: identify a variable, predict its response to an edit, and show relevant other variables are preserved. Patching methodology itself affects localization conclusions [12]. The current spatial and pathway results motivate such tests, but neither basis heatmaps nor lower embedding error establish fine-grained physical control.

**Efficiency must include the planner.** The successor moves response estimation offline. Recorded calibration took approximately 100 and 95 seconds for the two reaching tasks, excluding inherited basis fitting and research costs. One excluded engineering scenario took 314.28 seconds native and 313.03 seconds edited. This does not estimate average overhead or establish speedup. Every planner candidate still pays projection, expansion, memory, and hook costs. Evaluation must count calibration, full-planner latency, memory, and behavioral benefit together.

LeWM simplifies end-to-end world-model training [7]; LeVJEPA simplifies video pretraining [8]; Fast-LeWM changes sequential rollout into parallel action-prefix prediction [9]. Their efficiencies address different costs from adapting a retained checkpoint. Our method could complement a cheaper backbone, but has not beaten one under a matched budget. A practical comparison must include inexpensive adaptation alternatives, not only the original costly operator.

**Behavioral evidence and scope.** The six-task program includes Reach, Reach-Wall, Push-T, Wall, PointMaze, and DROID. These figures cover the completed three-task primary offline study and two-task successor, with overlapping pools. A frozen five-arm, 96-episode-per-task/condition reaching comparison is underway in the checked local record; no partial outcomes inform this draft. DROID evaluates recorded robot actions, not new physical-robot executions. The 81 primary trajectories are exposed development data, not independent trained-model replications. Category-wise simultaneous intervals do not cover every narrative choice across the program. Precision, fitting populations, and planner-induced distribution shift limit inference.

**Future directions.** Complete the behavioral comparison, then test whether the observed spatial/pathway structure predicts useful corrections prospectively. A repeatable system would fit and validate compact corrections within a fixed adaptation budget. A sparse router could eventually select one correction expert or identity from state/action context, avoiding evaluation of every expert inside CEM. This remains untested. Branch-JEPA already investigates multiple latent futures [10]; uncertainty branches and sparse correction experts solve different problems. Neither generic MoE nor frozen steering is a novelty claim. Generalization must be measured across unseen conditions and independently trained models, allowing the fitted coordinate system to change.
