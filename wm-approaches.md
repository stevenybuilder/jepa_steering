# World-model approaches: concepts, evidence, and next questions

Updated September 8, 2026, after completion of the 960-episode MetaWorld panel. This is a research reference for the current project, not a claim of a finished six-task study or a new base-model architecture.

**Current conclusion:** structured activation edits improve some offline forecasts. The completed behavioral comparison does not establish a task-success improvement beyond native and matched random controls. The revised operator has inexpensive application in the measured runs; useful behavioral specificity remains unproven.

## 1. The question and the vocabulary

Our practical objective is **better robot-task success at a fixed, affordable planning budget**, with interpretable and selective interventions where the evidence supports those claims. Forecast accuracy is a diagnostic and fitting signal; it is not the final behavioral objective.

| Concept | Meaning here |
|---|---|
| Our short name | **Structured activation steering**: change selected internal activation patterns while keeping the world-model weights frozen. |
| COAST's short name | **Contrastive conceptor activation steering**. |
| World model | Predict the consequences of candidate actions in a learned representation. |
| Planner | Compare candidate action sequences and choose one using the predicted consequences. |
| Activation intervention | A deliberate modification to a hidden state or conditioning pathway during a forward pass. |
| Activation patching | Often means substituting activations from a donor computation. Our fitted corrections are more precisely described as activation steering, although both belong to activation-intervention methods. |
| Ablation | A controlled change to a component, rank, layer, spatial support, or pathway to test what matters. It need not mean deleting neurons. |
| Fitted versus frozen | The steering artifacts are fitted from data; the base-model parameters remain frozen. These statements are compatible. |
| Distributed versus high rank | An edit can affect many coordinates while remaining a combination of only a few basis directions. Spatial breadth does not imply many independently selected edits. |
| Representation geometry | Directions, subspaces, covariance, distances, and local response structure in the hidden states. A PCA basis is not automatically a semantic manifold or a circuit. |

The relevant mathematical tools include linear algebra, geometry, probability/statistics, numerical optimization, and dynamical systems. Representation learning and control theory are useful research perspectives, not an exhaustive or universal ranking of the two fields required for architectural progress.

Control becomes relevant when predictions are used to select actions and respond to observations. JEPA-WM uses model predictive control on MetaWorld: execution is followed by observation and replanning. This does not mean that learning JEPA representations requires a new controller, or that our project needs to replace the existing planner. [JEPA-WM, Appendix F](https://arxiv.org/html/2512.24497v4#A6)

## 2. Our approach versus COAST versus JEPA-WM

| Question | Our current study | COAST | JEPA-WM |
|---|---|---|---|
| Main objective | Improve useful forecasts and ultimately task success with a retained model. | Improve frozen-policy task success. | Identify architecture, training, and planning choices that improve physical planning. |
| Intervention point | Action-conditioned world-model predictor and visual/action conditioning. | Policy action-expert activations. | Encoder/predictor design, training recipe, and planner. |
| What is fitted? | Task/checkpoint-specific bases, forecast-error readout, response map, and coupling directions. | Success/failure activation filters. | Dynamics model; candidate recipes are compared empirically. |
| Direct fitting signal | Rank-4: visual forecast residual. Coupling: visual/action cross-covariance. | Rollout outcome labels. | Predictive training loss; planning evaluation guides recipe choice. |
| Runtime behavior | Add a structured correction or perturb a selected pathway during forecasting. | Apply a precomputed multiplicative matrix gate. | Forecast action consequences and optimize goal-reaching plans. |
| Base weights during our/its steering | Frozen. | Frozen. | Predictor is trained before planning. |
| Behavioral evidence | Two-task, five-arm development panel complete; no qualifying learned intervention. | Reports held-out simulation and real-robot gains. | Reports navigation/manipulation evaluations and recorded-robot action scores. |
| Generalization boundary | Separate task fits; one checkpoint in the completed panel; no fresh confirmation. | Generally task/model-specific fits; cross-task transfer tested. | Recipe choices can differ by task category. |

Sources: [our fitting code](src/offline_study/author_fit.py), [coupling fitter](src/offline_study/operator_fit.py), [COAST methodology](https://arxiv.org/html/2605.17144v1#S3), and [JEPA-WM design choices](https://arxiv.org/html/2512.24497v4#S4). This compares objectives and evidence; it is not a hardware-, task-, or checkpoint-matched performance leaderboard.

JEPA-WM documents an empirical sequence: examine planning choices, then training/architecture choices, followed by scaling. Its ablations test hypotheses; the publication does not establish the authors' entire private exploration history. [Section 4](https://arxiv.org/html/2512.24497v4#S4)

## 3. What the tasks actually require

The broader approved scope contains six entries. The completed core panel covers only the first two, under five conditions: native, fixed rank-4, matched-random rank-4, coupling, and matched-random coupling.

| Task | High-level physical behavior | Outcome we want |
|---|---|---|
| MetaWorld Reach | A simulated robot hand moves toward a target. | Reach the goal within the native episode budget. |
| MetaWorld Reach-Wall | The hand reaches a target with an obstructing wall. | Reach the goal through a feasible route. |
| Push-T | A simulated pusher moves and rotates a T-shaped block. | Achieve the designated block/pusher goal configuration. |
| Wall | A point agent navigates rooms connected by a doorway. | Arrive at the target, using the doorway when needed. |
| PointMaze | A force-actuated ball moves with inertia. | Arrive at the target under the environment's dynamics and constraints. |
| DROID recorded evaluation | Given recorded initial/goal observations, the planner proposes a short action sequence. | Improve the authors' recorded net-translation action score. No new robot execution or physical task-success rate is measured. |

Sources: [study plan](docs/EXPERIMENT_PLAN.md), [DROID contract](docs/DROID_METHOD_ALIGNMENT.md), and [JEPA-WM environment definitions](https://arxiv.org/html/2512.24497v4#A5). MetaWorld is a benchmark containing many tasks; the current behavioral comparison is not an evaluation of the entire benchmark.

## 4. What our edits actually do

### Revised rank-4 correction

At the current MetaWorld site, the patch field has **256 positions × 400 features = 102,400 coordinates**. The four fitted basis vectors each span that field. Four coefficients specify one distributed correction; we do not ablate every coordinate independently. The older 98,304-coordinate number does not describe this 400-feature field.

The selected runtime location is predictor block B3 at imagined rollout step H3, in the H6 forecast path. This is one selected layer/time site, not a simultaneous edit across every model layer.

In simplified notation:

```text
phi = [1, three standardized activation projections]
coefficients = fitted_map × phi
delta = fitted_basis × coefficients
new_activation = activation + dose_normalized(delta)
```

The output basis has four directions. The readout has three activation scores plus an intercept. The fitting target is the H6 **visual-embedding forecast residual** from recorded futures. The future is available during fitting/scoring, not supplied to the running planner. Our commonly quoted offline proprioceptive error is an evaluation endpoint, not the literal target used to fit this readout.

The original operator estimated responses using repeated candidate-dependent forecasts. The retained implementation included 32 online response probes per candidate, plus diagnostic/control work. The revision moves response estimation offline and applies the map in the existing batched forward pass, with **zero online response probes and zero separate native-shadow forecasts**. It is a changed operator that needs its own validation, not an exactly equivalent speed optimization. [Full operator contract](docs/FIXED_RESPONSE_RANK4.md)

### Vision–action coupling

We fit paired directions from covariance between native visual activations and internal action-conditioning activations. The comparisons perturb visual only, action only, or both, with equal-budget, permuted, and random controls. These directions are data-fitted even though they are not fitted to binary outcome labels. [Implementation](src/offline_study/operator_fit.py)

### Geometry, rank, layer, and spatial ablations

The offline families compare linear/cubic reconstruction and curvature controls; ranks 1/4/8; individual versus distributed layer support; and one patch, contiguous/scattered patches, or all patches. Combined and component-removal arms ask whether retaining both coupling and activation correction helps. They test different questions and do not all have completed behavioral evaluations.

The current coordinate map is a fitted linear readout/basis and local correction. We have not demonstrated a global manifold atlas, semantically named physical axes, a recovered circuit, or a trained mixture-of-experts architecture. The project's name does not imply that the active operator is a novel transcoder or sparse autoencoder.

## 5. Offline findings: useful signals with explicit limits

Positive values below are reductions in **H6 proprioceptive embedding MSE**, not task-success gains. Original and revised operators are kept separate.

| Finding | Observed result | Interpretation |
|---|---|---|
| Original rank-4, Reach | 2.887% reduction; simultaneous 95% interval [2.276%, 3.499%]. | Some forecast error is correctable in a compact activation subspace. |
| Original rank-4, Reach-Wall | 2.137%; [1.645%, 2.629%]. | A similar offline signal exists in the obstacle task, using its own fit. |
| Original combined edit, Reach | 4.824%; [3.668%, 5.980%]; beat its matched control and both component-removal arms. | Components provide complementary forecast benefits. This alone does not establish mechanistic synergy. |
| Single-layer versus spatial support | Reach block 0: 3.026%; [2.124%, 3.929%]. Only the all-patch spatial candidate passed the original advancement gates. | Spatial distribution and layer distribution are different questions; several individual layers worked. |
| Push-T boundary | Rank-4: 0.198%, [−0.059%, 0.455%]. Joint coupling: −0.638%, [−1.214%, −0.062%]. | Rank-4 was inconclusive; coupling worsened the forecast metric. |
| Revised fixed-response, Reach / Reach-Wall | Proprioceptive reductions approximately 2.360% / 2.192%; visual reductions approximately 0.427% / 0.399% in BF16. | The cheaper operator retained positive offline effects; the more directly visual effect is smaller. |
| Precision-dependent reconstruction | On Push-T, cubic omitted-activation reconstruction error was approximately 1,192× lower than linear in FP32, but approximately 35% higher in BF16. | This is a diagnostic precision boundary, not a 1,192× forecast or behavioral improvement. |

Sources: [corrected original results](reports/CORRECTED_OFFLINE_RESULTS.md), [combined comparison](docs/COMBINED_DEVELOPMENT.md), and [revised-operator report](docs/FIXED_RESPONSE_RANK4.md). Precision-specific reconstruction details remain in the local paper data and original geometry reports.

The primary offline pools contain 33 Reach, 27 Reach-Wall, and 21 Push-T trajectories. Repeated windows, arms, and precisions do not create additional independent trajectories. These development-exposed pools are not fresh confirmation.

DROID's completed coupling evaluation is also inconclusive: joint coupling changed its recorded-action score by approximately **−0.247 score points**, with an interval spanning zero. It is not a physical task-success percentage. Wall/PointMaze evidence is incomplete across intervention families; missing numerical reports must not be labeled null. See [coverage status](reports/ABLATION_COVERAGE_STATUS.md).

## 6. Completed behavioral results: all 960 episodes

**No learned intervention passes the frozen advancement rule.** Each table cell below uses 96 episodes. There are 96 distinct initial/goal scenario clusters per task, paired across the five arms, and one released model checkpoint. The panel therefore contains 192 distinct task/scenario combinations evaluated five ways, not 960 independent model replications.

| Condition | Reach | Reach-Wall |
|---|---:|---:|
| Native | 43/96 (44.79%) | 29/96 (30.21%) |
| Revised fixed rank-4 | 49/96 (51.04%) | 30/96 (31.25%) |
| Matched-random rank-4 | 48/96 (50.00%) | 23/96 (23.96%) |
| Coupling | 47/96 (48.96%) | 25/96 (26.04%) |
| Matched-random coupling | 58/96 (60.42%) | 27/96 (28.12%) |

The highest observed Reach success rate belongs to the random-coupling arm. This is descriptive, not a validated new method: random-versus-native is outside the eight registered primary contrasts.

### All preregistered paired comparisons

Effects are **percentage points**. Intervals are paired whole-scenario percentile bootstrap intervals, with 20,000 resamples and Bonferroni simultaneous 95% coverage across eight contrasts. Exact paired-discordance p-values are Holm-adjusted across the same family.

| Task | Intervention minus reference | Effect (pp) | Simultaneous 95% interval (pp) | Holm p |
|---|---|---:|---:|---:|
| reach | Revised fixed rank-4 minus Native | +6.25 | [-11.46, +23.96] | 1.000 |
| reach | Revised fixed rank-4 minus Matched-random rank-4 | +1.04 | [-15.62, +17.71] | 1.000 |
| reach | Coupling minus Native | +4.17 | [-15.62, +23.96] | 1.000 |
| reach | Coupling minus Matched-random coupling | -11.46 | [-29.17, +6.25] | 0.865 |
| reach-wall | Revised fixed rank-4 minus Native | +1.04 | [-13.54, +15.62] | 1.000 |
| reach-wall | Revised fixed rank-4 minus Matched-random rank-4 | +7.29 | [-5.21, +19.79] | 1.000 |
| reach-wall | Coupling minus Native | -4.17 | [-20.83, +11.46] | 1.000 |
| reach-wall | Coupling minus Matched-random coupling | -2.08 | [-15.62, +12.50] | 1.000 |

Every interval includes zero. The rule required an observed gain of at least 5 percentage points versus native **and** positive simultaneous lower bounds versus both native and matched random. The frozen selection retains native on both tasks. Wide intervals leave meaningful positive and negative effects possible; a failed advancement gate is not proof of zero effect.

### Compute and behavioral turnover

| Condition | Mean Reach episode seconds | Mean Reach-Wall episode seconds |
|---|---:|---:|
| Native | 299.98 | 301.58 |
| Revised fixed rank-4 | 301.05 | 302.61 |
| Coupling | 299.81 | 301.44 |

Revised rank-4's mean episode overhead is approximately **0.36% on Reach and 0.34% on Reach-Wall**. These are descriptive timings from this panel, not a separate controlled throughput benchmark or the total research/adaptation cost. Basis fitting, response calibration, data acquisition, and discarded experiments are additional costs. Small runtime overhead is insufficient if the intervention does not yield useful behavior.

Reach rank-4 rescued 23 native failures but spoiled 17 native successes, yielding six net additional successes. A net success rate conceals both directions of change. We have not yet established a semantic explanation for those rescues/regressions.

Verification reloaded all 960 records, checked 40 completed shard reports and their episode/action-trace bindings, and exactly reproduced the frozen paired analysis. This is completed development evidence; no fresh confirmation, three-training-seed replication, HMM result, or full six-task behavioral completion follows from it.

Machine-readable sources: [core analysis](reports/wm-approaches/core-analysis.json) and [verification receipt](reports/wm-approaches/core-verification.json). Original worker report SHA256: `0bcfbae89587430782451429f8cad950fc5f8ecc3c0e4ae5a9a2acf88eae977a`; protocol SHA256: `853e8bf6b71c65c1e61d249ba84d8d57d6172ba1bd3a15e0424dc2d9e32cfa1f`.

## 7. Conceptors, fitting, and overfitting

A conceptor is an established covariance-derived matrix that softly retains some activation directions and attenuates others. It is not automatically a human-readable concept. COAST combines success and failure conceptors and applies a precomputed gate; its weights stay frozen. [COAST, Section 3](https://arxiv.org/html/2605.17144v1#S3)

Task-specific fitting alone does not demonstrate overfitting. COAST reports separate fitting/evaluation rollouts and transfer tests. Those tests bound its claims; they do not establish universal generalization. [Evaluation protocol](https://arxiv.org/html/2605.17144v1#S4.SS1), [transfer](https://arxiv.org/html/2605.17144v1#S4.SS4)

Our own task/checkpoint-specific bases, dose calibration, and repeated development comparisons also pose overfitting and selection risks. Broad support and frozen model weights do not remove them. A reusable fitting procedure is a different claim from transferring the identical numerical matrix to new tasks or checkpoints.

### Correction to the historical account

We previously implemented success/failure-based capture and conceptor-style fitting code. The statement that we had never built such a pipeline was incorrect.

The September 4 archived handoff reports 30 MetaWorld development episodes with 12 successes and 18 failures. A later archive reports a separate JEPA-WM Reach-Wall fit cohort with **one success in 15 episodes**, which failed class support before a behavioral intervention comparison. This supports a concern about too few successful examples; it does not establish a measured conceptor efficacy failure caused by sample size.

A separate archived conceptor experiment contains actual forecast-intervention measurements on 12 contact scenes using the DROID checkpoint. It used contact/interaction contrasts and is not the same success/failure-labeled MetaWorld behavioral comparison. Its older out-of-scope stimulus setting is historical evidence, not part of today's six-task protocol.

Historical source locations, retained locally: `docs/archive/legacy-branches-2026-09-04/handoffs/CODEX_HANDOFF_2026-09-04.md`, `docs/archive/deferred-threads-2026-09-05/snapshot/WORLD_MODEL_SONAR_PLAN.md`, and `archive/2026-09-07-workspace/artifacts/cgs_pilot/mech_w1/conceptor_results.json`. The old episode counts above are attributed to the archived reports; the current 960-episode counts have direct machine-readable verification.

## 8. Key papers and what each contributes

| Paper | Relevance and boundary |
|---|---|
| [Terver et al., What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? (JEPA-WM), v4](https://arxiv.org/abs/2512.24497v4) | Governing architecture/planning recipe and evaluation reference; already studies prediction/planning mismatch. |
| [Miao et al., COAST, v1](https://arxiv.org/abs/2605.17144v1) | Closest outcome-labeled activation-filter comparison. |
| [Joseph et al., Interpreting Physics in Video World Models, v1](https://arxiv.org/abs/2602.07050v1) | Physics Emergence Zone and physical representation/steering analysis in video encoders. Our predictor block is not automatically a PEZ, and our PCA axes are not validated physical concepts. |
| [Wurgaft, Rager et al., Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior, v1](https://arxiv.org/abs/2605.05115v1) | Connects representation geometry to behavior through interventions. Motivates geometric control, without making our linear basis a recovered manifold. |
| [An et al., Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency, v1](https://arxiv.org/abs/2608.12939v1) | Direct world-model diagnostic prior art. Helps frame why action-conditioned validation matters beyond aggregate reconstruction accuracy. |
| [Tong, Fan et al., Beyond Language Modeling: An Exploration of Multimodal Pretraining, v1](https://arxiv.org/abs/2603.03276v1) | Broad motivation for learning from visual experience and studying design choices systematically. It supplies conceptual context, not evidence for our intervention's efficacy. |
| [Maes et al., LeWorldModel, v3](https://arxiv.org/abs/2603.19312v3) | Smaller end-to-end world-model training and planning; an alternative route to practical efficiency. Our adaptation has no demonstrated cost advantage over training/adapting it. |
| [Kuhn et al., LeVJEPA, v1](https://arxiv.org/abs/2608.27395v1) | Efficient video representation pretraining. Its pretraining savings are not measured speedups for our action-conditioned planner. |
| [Wu et al., ReFT](https://arxiv.org/abs/2404.03592) | Establishes low-rank representation adaptation in frozen language models; low rank and frozen-weight adaptation themselves are not new contributions. |

JEPA-WM's studied training recipe freezes a pretrained visual encoder while training the action-conditioned predictor and associated trainable encoders. The moving EMA teacher loop from some encoder-pretraining methods is not being run by our frozen steering operator. Upstream pretraining lineage and our inference-time computation should not be conflated. [JEPA-WM method](https://arxiv.org/html/2512.24497v4#S3)

## 9. Interpretation, potential contribution, and next step

The motivating picture is **a map that must support a useful route**: accurately representing or reconstructing the world is valuable only if the information needed for the next decision is available to the planner. This is an explanatory analogy, not a theorem or a novelty claim.

The original hypothesis had separable parts: a frozen predictor admits compact useful forecast corrections; fitted geometry matters beyond random perturbation; and those corrections improve planning at acceptable cost. The evidence supports the first part on reaching tasks, provides qualified offline specificity evidence, and establishes a practical implementation improvement. The completed behavioral panel does not establish the last part or behavioral specificity.

Potential paper contributions must therefore stay bounded:

1. **Controlled forecast intervention evidence:** task-dependent, pathway/spatial/rank comparisons in retained action-conditioned world models, including meaningful negative controls.
2. **Offline response fitting for inexpensive application:** one distributed correction applied without online response probes, with separate offline and behavioral validation. The primitives are established mathematics, not a newly invented inverse or PCA method.
3. **A measured boundary on translating forecast correction into control:** the completed comparison shows no qualifying success gain despite positive offline results. To become a strong explanatory contribution, this needs a specific account of action changes, not merely the already-known observation that loss and task success differ.

**Highest-value next analysis:** use the existing paired records and saved actions to inspect all rescues and regressions, compare them with matched random controls, and identify the earliest meaningful decision differences. Test whether any descriptive pattern concerns goal approach, obstacle routing, action magnitude, or another physically interpretable condition. These are candidate explanations, not established findings. Outcome-selected case patterns remain exploratory and require prospective validation; do not cherry-pick a few favorable videos or immediately fit a successor on those same cases.

The active code already provides an offline-fit/online-apply interface, explicit layer/time/patch bindings, batched candidates, controlled dose, and unchanged native planning. A future mixture of small correction experts with a cheap state-based router and identity option is a possible architecture direction. It has not been demonstrated; HMM remains incomplete. Representation geometry, selective read/write tests, and fixed active-compute comparisons would need to justify the experts and routing. A full circuit map is not required to establish a narrow steering effect, but an interpretable mechanism needs more than an activation heatmap.

Further experiments, new fits, model changes, and compute spending remain decisions to make after this evidence review. This note does not authorize them or change the frozen protocol.
