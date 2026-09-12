# Rank Edit

*Offline-fitting feasibility analysis — September 7, 2026. Compute/evidence snapshot: September 8, 00:30 UTC. Three-design proposal clarified at 00:40 UTC (September 7, evening EDT), using the same completed findings; no new empirical analysis.*

**My recommendation is to reassess before scaling the slow rank/combined panel.** We have modest, task-dependent forecast-error improvements and no verified task-success improvement yet. The expensive part is repeatedly measuring each candidate's response to interventions, not adding a four-dimensional edit. Removing those online probes is the main computational opportunity; establishing useful action ranking is the main scientific question.

A small predictor or a fixed inverse could potentially replace that work. We have not established that either preserves forecast improvements or task performance. This is an analysis and proposal, not a revised experimental protocol. This update did not launch, pause, or change GPU jobs. The recommendation to pause expansion is not a claim that running jobs have been stopped.

## The proposed focus: three designs for one distributed edit

**The immediate design choice is how to produce four coefficients cheaply and usefully. We would reuse one intervention site and a four-direction output space. We would not search the 102,400 activation coordinates individually.** The existing implementation already makes a structured edit; the proposed change removes its expensive online construction and, in the third design, changes what it learns to correct.

My first engineering candidate is **Design 1** below. Design 2 accommodates candidate-action dependence that Design 1 may miss. Design 3 is the more ambitious route to task-relevant steering. This is a shortlist of alternatives, not a proposal to stack three interventions or launch three full experimental suites. Their ordering is a judgment from existing evidence, not a measured ranking of these unbuilt methods.

### The shared intervention contract

For a candidate action sequence, let `x` be the flattened **native B3 output at H3**, for its newest 256 patches. The proposed operation is:

\[
x' = x + Bc, \qquad B\in\mathbb{R}^{102400\times4},\quad c\in\mathbb{R}^{4}.
\]

Here `c` means the **delivered coefficients after dose handling**. Reshape `Bc` to 256×400 and add it once at that hook. Each column of `B` is a distributed pattern; the four coefficients determine their combination. A shared energy ceiling is the existing task-specific rank dose `ρ`. Designs must define their normalization and zero-treatment behavior explicitly; these successors cannot inherit the old cross-arm degeneracy rule while deleting the probes it depends on.

- **Where/when:** B3 at H3, only during full H6 forecasts. Earlier computation in that forecast stays native, making `x` available in the same forward rollout. Shorter forecasts stay native.
- **Spatial support:** retain all 256 patches initially. This preserves the existing reference and avoids adding an unsupported localization choice. It does not assert that every patch is necessary.
- **Capacity:** four output directions. Their stored patterns contain 409,600 numbers, about 1.56 MiB in FP32; only four combination coefficients are produced per candidate. Four coefficients are not the total number of fitted parameters.
- **Online execution:** read available features, compute the coefficients, add the field, and continue the existing rollout. The intended implementations require **zero response-probe rollouts and zero separate native-shadow rollouts**. Full planner latency still needs measurement.
- **Scope:** task/checkpoint-specific banks, initially grounded in Reach/Reach-Wall evidence. No universal cross-task or cross-seed matrix is assumed. The broader six-task study scope is not changed by this proposal.

“One intervention” means **one learned operator applied once per eligible candidate forecast**, batched over candidates. CEM can invoke it many times during an episode. A fixed operator can produce different edits for different inputs. If all four coefficients were constant across inputs, the four patterns would collapse to one constant vector; that is a different restriction and is not what “one structured intervention” requires.

### The three candidate designs

| Design | Exactly how it constructs the one edit | Evidence motivating it | What is still missing |
|---|---|---|---|
| **1. Fixed low-rank map — first engineering choice** | Keep the existing four PCA patterns. Fit one representative response matrix offline and compose it with the existing error readout into a **4×4 coefficient map**. At inference, compute three H3 PCA scores plus an intercept, multiply by that map, normalize to the fixed dose, and add the resulting field. | Rank 4 has a modest offline effect on Reach and Reach-Wall. This preserves the current directions and visual-error target while removing online inverse construction. | Fitting-only response measurements; evidence that one response approximation is adequate. The four coefficients still vary with the input. |
| **2. Action-conditioned coefficient predictor** | Keep the same four PCA patterns. Fit one small network with **one 32-unit hidden layer and four outputs**, using three native H3 PCA scores and the normalized six-step candidate action sequence. Its targets are the teacher's delivered coefficients, including zero edits. Bound its output by the existing dose ceiling and add the field directly. | The current response depends on candidate actions through H6. A constant response approximation may miss that dependence even when the edit space is small. | A fitting-only teacher dataset and family-separated validation. Direct prediction can shrink edit magnitude, so this is a new bounded-dose operator, not exact reproduction of the old fixed-dose rule. |
| **3. Outcome-trained four-direction edit** | Learn four distributed output patterns and a small coefficient map offline against **measured differences between candidate-action outcomes**, with a forecast-fidelity constraint. Apply the resulting four-coefficient edit at the same B3/H3 hook. No teacher probing is needed at deployment. | Earlier correction pilots improved MSE without useful physical choices; the current basis follows activation variance and the current target is visual forecast error. This design directly addresses that mismatch. | Suitable same-state, different-action fitting data with actual futures and physical outcomes. Current scalar MSE summaries cannot supply these targets. This is the largest change and has the weakest direct evidence so far. |

For **Design 1**, the exact offline composition is derived later in this note: `A = (J̄ᵀJ̄ + λ̄I)⁻¹J̄ᵀE`. Inference uses `c_raw = Aφ(x)`, then the registered dose normalization, with zero returned for a frozen small-norm condition. We would not construct a 102,400×102,400 transformation. The two thin projection/expansion factors perform the work. This standalone zero rule is distinct from the source family's common-degeneracy test.

For **Design 2**, the teacher labels must be generated on fitting families with the frozen current rank operator; model response probes do not require physical futures. Fit directly to the four delivered coefficients in the existing orthonormal basis, which represent the actual field and its dose. The proposed cap is `c = c_raw * min(1, ρ / ||c_raw||)` for nonzero outputs, and zero for an exactly zero output. Zero-target examples penalize unnecessary edits; they do not guarantee exact reproduction of every teacher zero. The action sequence is a proposal known at inference, never the subsequently observed future. The 32-unit architecture is a concrete starting design, not a result of a width search.

For **Design 3**, a concrete formulation is `c = clip_ρ(Wᵀ(x−μ) + Va + Gg + b)`, with a learned four-column output basis `B`. Here `a` is the known normalized candidate action sequence and `g` is the available goal embedding; the matrices are fitted offline. Initialize `B` from the existing rank-4 basis and keep it orthonormal during fitting. Train the new factors through the remaining frozen predictor computation. Use a loss that asks the native planner's scores to order paired actions according to a prespecified measured physical endpoint, together with a term keeping predicted visual/proprioceptive futures close to the recorded actual futures. Freeze the endpoint, loss weights, and fitting budget before training. Gradients and fitting cost occur offline; base-model weights remain frozen. This is an explicit new adapter hypothesis, not a claim that successful edit directions can be read off our MSE table.

The fidelity term in Design 3 matters: merely making imagined goal distance smaller could fabricate progress. If the native goal objective cannot order physically better actions while remaining faithful to their true futures, an activation edit at this site may be the wrong remedy. This design changes both fitting supervision and the learned operator; it is not the basis-only diagnostic described later.

### How the existing findings constrain the choice

**Keep four directions and B3/H3 as the starting point.** Rank 4 was adequate relative to rank 8 on Reach, and eligible on Reach-Wall. This supports a parsimonious reference, not a claim that four is universally optimal. A result at block 0 in the separate rank-1 sweep does not justify moving rank 4 there. Likewise, rank-1 spatial results do not establish a sparse rank-4 support.

**Keep cheap coupling as a comparator.** Reach's equal-budget coupling gain (2.035% proprio MSE reduction) is relevant to practical value, and its combination with rank 4 reached 4.824%. However, the existing combination edits separate upstream pathways as well as B3. We cannot simply add those tensors into a B3 field and claim to preserve the combined effect. None of these three single-hook proposals is already the combined method. [Actual composition](src/offline_study/combined_operator.py#L54).

**Choose a hypothesis now; do not claim a successful fitted edit now.** Existing evidence supports this constrained shortlist. It cannot identify the best new coefficients, validate the fixed-response assumption, or establish task-success improvement without later fitting and evaluation. The four diagnostics below remain supporting validation ideas, not prerequisites that must all be run just to write this proposal. No new analyses, fitting, simulator work, or GPU execution were performed for this clarification.

## What the current edit actually does

For the MetaWorld rank-4 arm, the intervention occurs at **B3, the fourth predictor block, at imagined step H3**, across the newest **256 tokens × 400 features**. Flattened, that is a 102,400-dimensional activation field. Rank 4 means the edit lies in the span of four fitted directions across that field.

**These are not 102,400 separate ablations, and rank 4 does not edit all six blocks.** It adds one candidate-dependent combination, `c1*b1 + c2*b2 + c3*b3 + c4*b4`, at that one site. Each `bi` is a pattern over the field. Here “rank” counts basis directions, not the matrix rank of the reshaped 256×400 field. This is a low-dimensional additive activation intervention; it does not replace the whole field with a donor example or delete every feature.

The 256 positions are the model's 16×16 spatial grid. The width comes from its 384-dimensional visual stream plus 16 proprioceptive dimensions. Predictor blocks mix these channels, so the intermediate coordinates should not be treated as 400 independent semantic concepts or as cleanly separated sensory causes. The dimensions come from the architecture, not an ablation-count choice. [Released configuration](vendor/jepa-wms/configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml#L28), [predictor construction](vendor/jepa-wms/app/plan_common/models/AdaLN_vit.py#L200).

| Ablation family | What varies | What stays fixed |
|---|---|---|
| **Operator rank** — the source of rank 4 | Rank 1, 4, or 8; corresponding random-subspace controls; native/zero controls | B3, H3, all 256 patches, target, fitting procedure, total edit energy |
| **Layer distribution** | Six single blocks, B2+B3, or all six blocks, with matched controls | **Rank 1**, H3, all patches, total energy |
| **Spatial distribution** | One patch, contiguous 16, scattered 16, or all 256; direction/position controls | **Rank 1**, B3, H3, total energy |

Full spatial support was held fixed to isolate the rank comparison. It was not a finding that all patches are necessary for rank 4. Conversely, the rank-1 localization results cannot answer that rank-4 question. [Ablation definitions](docs/EXPERIMENT_PLAN.md#L359).

| Component | Already fitted offline? | What remains online? |
|---|---|---|
| Four edit directions | Yes: selected from a nested PCA basis | Combine them using candidate-specific coefficients |
| Forecast-error predictor | Yes: three PCA scores plus an intercept predict the H6 visual residual | Read H3 activations and evaluate this predictor |
| Response to an edit | No | Perturb each candidate, run forecasts, and estimate response vectors |
| Four coefficients | No | Solve a damped least-squares problem using those responses |
| Edit dose | Yes: fixed from fitting data | Normalize the proposed edit to the registered dose |
| Vision/action coupling edit | Yes: directions and doses | Add the fixed visual and action-condition edits |

The current rank implementation retains **32 probe rollouts per candidate**: positive and negative probes for eight learned directions and eight random directions. This supports the frozen rank/control comparisons and common degeneracy rule, even when the planner requests just rank 4. The adapter computes the source family's corrections and then selects the requested arm. A four-dimensional solve is small; generating its inputs is expensive. Existing fit-action measurements report roughly **102 seconds versus 2.9 seconds native for a 300-candidate forecast batch**: approximately 35× the native time on that workload. These are batch measurements, not an end-to-end deployment speed estimate.

Thus a 300-candidate H6 batch entails 9,600 probe trajectories, executed in batches, plus native-shadow and final forecasts. MetaWorld CEM repeats candidate evaluation for 15 iterations and replans during an episode. The recurring model work explains the cost. Simply shrinking the 4×4 solve or masking patches would leave most of it in place. Removing source-family probes also changes the current common-degeneracy calculation unless equivalence is established; it cannot silently be presented as the same frozen operator.

Sources: [rank construction and response solve](src/offline_study/support_operator.py#L64), [planning adapter](src/offline_study/planning_support.py#L85), [recorded timing](reports/PLANNING_METHOD_ALIGNMENT.md#L157), [static coupling application](src/offline_study/planning_intervention.py#L12).

## Earlier offline-fitting possibilities, for reference

The three designs above are the current focus. This earlier catalogue records additional possibilities; it is not an expanded execution queue or a measured performance ranking.

| Candidate | Offline fitting target | Work at inference | Main uncertainty |
|---|---|---|---|
| **Fixed response approximation** | Estimate one representative response matrix from fitting examples; combine it with the existing error predictor | Small matrix operations and a rank-4 addition; no response probes | Whether one response matrix works across states and action sequences |
| **Predict the four coefficients directly** | Run the current operator offline as a teacher; fit a small model from available state/action features to its delivered coefficients | Feature extraction, a small regression or network, and the addition | Whether the teacher's choices are predictable on new planner candidates |
| **Predict a compact response model** | Fit how the response varies with state/actions, preferably in a verified compact output basis | Small predictor and a 4×4 solve, without forecast probes | More fitting complexity; compressed responses must retain the useful information |
| **Predict when to apply the edit** | Learn whether the fixed intervention helps, from paired intervention measurements | A small gate using available history | A gate alone does not remove the rank solver's cost on candidates it activates |
| **Fit a contrastive activation filter** | Fit geometry associated with successful/failed behavior, or explicitly defined forecast-error groups | A fixed, preferably factorized, activation filter | This is a new steering hypothesis; low forecast error is not robot success |

The existing coupling edit already provides an example of an offline-fitted operator with inexpensive application. Its maximum-covariance directions are fitted from native activations; their signs are fixed deterministically, not learned from successful outcomes. Learning a state-dependent dose or gate would be an additional hypothesis. [Coupling fit](src/offline_study/operator_fit.py#L121)

## A concrete mathematical simplification

Let:

- **x** be the native H3/B3 activation field, flattened.
- **B** be the matrix containing the four fixed edit directions as columns.
- **φ(x)** contain an intercept and the three standardized PCA scores already used by the error predictor.
- **Eφ(x)** be the predicted H6 visual residual.
- **J(x,a)** map four edit coefficients to changes in the H6 visual prediction, for candidate action sequence **a**.

Before dose normalization, the current calculation has the form:

\[
c^*(x,a)=\left(J(x,a)^T J(x,a)+\lambda(x,a)I\right)^{-1}J(x,a)^T E\phi(x).
\]

The damping is proportional to the mean diagonal of the response Gram matrix. For a nondegenerate candidate, the delivered edit is:

\[
\Delta x=\rho\,\frac{Bc^*}{\|Bc^*\|_2}.
\]

Here **ρ** is the frozen dose. The implementation additionally applies a common degeneracy test across registered arms. These equations describe the active, nondegenerate case. [Exact implementation](src/offline_study/support_operator.py#L167)

**If a fixed response matrix J̄ is adequate**, all of the inverse calculation can be composed offline:

\[
A=\left(\bar J^T\bar J+\bar\lambda I\right)^{-1}\bar J^T E,
\qquad \hat c(x)=A\phi(x).
\]

**A is just 4×4.** The remaining larger operations are projecting the activation field into three scores and expanding four coefficients back into the field. We would not need to construct the full predicted visual residual at inference.

This is an algebraic possibility, not a fitted result. Averaging response matrices does not generally preserve the result of their individual inverses. The fixed-response assumption is exactly what would need testing.

A more flexible alternative learns:

\[
\hat c=f_\theta\bigl(\phi(x),\text{candidate action suffix},\text{available history}\bigr).
\]

Its training targets should represent the **delivered, normalized teacher edit**, including zero-treatment cases, rather than only the raw coefficients. Otherwise a good fit to coefficient magnitudes could still reproduce the wrong edit direction or dose. Fitting a predictor for repeated optimization solutions is generally called **amortized optimization**. [Amos's tutorial](https://arxiv.org/abs/2202.00665)

## Why actions and edit order matter

**H3 activations alone may be insufficient.** The current response probes run through H6, so their responses can depend on later candidate actions. Those proposed actions are known to the planner and are legitimate inference inputs. Actual future observations and realized future errors are not.

For **rank-only steering**, a predictor using the unedited H3 activation and known action sequence could potentially run directly inside the H3 hook, avoiding a separate native shadow rollout.

For **combined coupling-plus-rank steering**, coupling changes the computation before the rank hook. Our current rank correction is deliberately derived from an unedited native rollout. Reading the already-coupled activation would therefore change the predictor's input distribution. A one-pass successor would need either features available before coupling, or training on the coupled-path features with the intended teacher targets. Keeping a native shadow is another option, with a smaller computational saving.

The current planning adapter also applies rank steering only to full H6 forecasts. Shorter horizons remain native. A successor should preserve that scope unless a separate hypothesis justifies expanding it. [Combined construction](src/offline_study/combined_operator.py#L54), [horizon rule](src/offline_study/planning_support.py#L109)

## What data we actually have

The corrected fitting receipts record **128 trajectory families × four examples = 512 examples per primary task and precision**. These are 128 independent families, not 512 independent trajectories.

| Available material | What it supports | What it does not establish |
|---|---|---|
| Saved H3 fields from all six blocks, H6 visual residual vectors, metadata, and native errors | Inspecting representation/error relationships; fitting compact error readouts | How forecasts respond to intervening on those fields |
| Fitted operator banks | Reusing directions, projections, doses, and the existing error predictor | A trained coefficient predictor or fixed inverse |
| Native pooled H1–H6 histories for Reach and Reach-Wall | Studying compact history features; H1/H2 are available before an H3 edit | Which examples benefit from steering; no HMM or efficacy gate is fitted |
| Completed offline arm metrics | Existing development comparisons and task-level evidence | A ready-made collection of teacher coefficients and response matrices |

The bulk native-fit tensors were relocated from the laptop after recorded checksum verification of two remote copies. They would need their current storage paths checked before reuse. The smaller fitted banks and eight Reach/Reach-Wall history shards are present locally. I inspected source and receipts, not the bulk tensors or partial behavioral outcomes, for this note.

**The main missing training artifact is a fitting-only dataset of candidate features, teacher coefficients/delivered edits, and optionally response matrices.** The current operator computes those quantities transiently; its standard evaluator saves scalar metrics and diagnostics. Existing native activations and residuals cannot reconstruct the intervention responses by themselves. Producing teacher labels would require separate offline model execution and logging.

Sources: [saved fit schema](src/offline_study/author_fit.py#L85), [history capture](src/offline_study/routing_history.py#L88), [evaluation outputs](src/offline_study/author_evaluate.py#L167), [storage recovery](reports/EXECUTION_STATUS.md#L181).

## What the findings support so far

The current results establish a limited prediction effect worth diagnosing. They do not yet justify a large deployment or replication budget:

- Reach rank 4 reduces H6 proprioceptive embedding MSE by **2.887%**, with a simultaneous interval of **[2.276%, 3.499%]**.
- Reach-Wall rank 4 reduces it by **2.137%**, interval **[1.645%, 2.629%]**.
- Reach coupling-plus-rank4 reduces it by **4.824%**, interval **[3.668%, 5.980%]**, in its separate combined development analysis.
- Push-T rank 4 reduces the error by only **0.198%**, interval **[−0.059%, 0.455%]**; it is not eligible under the frozen rule. Push-T joint coupling is worse than native by **0.638%**.
- Reach's much cheaper equal-budget coupling edit reduces the error by **2.035%**, interval **[1.298%, 2.773%]**. It is an essential practical comparator for any expensive successor. The combined stage also beats both component-removal arms on its offline endpoint.
- The first completed Wall BF16 coupling scope is mildly negative: H6 proprio MSE **increases 0.1567%**, simultaneous interval **[0.0017%, 0.3117%]**, and is worse than its random control. This is a different task/operator result, not evidence about Wall rank 4.

The corrected primary results use **33 Reach, 27 Reach-Wall, and 21 Push-T independent trajectories**. Repeated prefixes and precision runs do not increase that independent sample count. On Reach, ranks 1, 4, and 8 passed the effect and matched-random gates. Rank 1 reduced the primary error by **1.899%**, but equivalence to the observed best eligible arm was not established. Rank 4 was selected as the smallest eligible rank meeting that equivalence rule; rank 8 was equivalent. On Reach-Wall, only rank 4 passed the effect/control gates. This corrects the earlier conflation of eligibility and final selection. See the [selection audit](artifacts/offline_study/primary-durable-20260907/primary-parsimony-audit-20260907-v2/report.json). No non-global spatial arm passed advancement, and no unique causal layer was isolated. Those spatial/layer comparisons used rank 1.

**Behavioral evidence remains unresolved.** At the execution snapshot, the MetaWorld comparisons still require corrected pairing/assembly after goal-image reproducibility repairs. PointMaze/Wall baselines and DROID's recorded-action baseline do not establish steering efficacy. This is evidence from three completed primary offline task pools plus the first Wall coupling scope, not a completed six-task steering result. [Current execution evidence](reports/EXECUTION_STATUS.md).

An earlier, separate development campaign gives a concrete warning: a Reach full-spatial correction reduced visual/proprio forecast MSE by **26.83%/49.35%** across eight starts, yet changed two action choices in physically worse directions. Its actual-H6 follow-up retained forecast gains but did not beat native-bank selection on physical progress. Push-T's 27 tested settings also failed their physical promotion gates. These were different operators and small development banks; they do not falsify today's rank-4 arm. They show why preserving average prediction MSE is an insufficient target. [Completed historical findings](archive/2026-09-07-workspace/artifacts/geometry_map/autoresearch_steering_20260906/FINDINGS.md).

Rank 4's selection supports a small edit subspace. **It does not establish that coefficients are constant, linearly predictable, or transferable across tasks.** Also, the rank solver targets a *visual* residual, whereas these headline improvements concern *proprioceptive embedding* error. Distillation could preserve the teacher's behavior without resolving that objective mismatch or improving task success. [Corrected results and interpretation](reports/CORRECTED_OFFLINE_RESULTS.md#L23)

My interpretation is that the intervention can repair some representation/forecast error, but we do not know whether that error controls the planner's useful distinctions between actions. A distributed field is not inherently ineffective or expensive. The concerning combination is a generic variance-based basis, an imperfect forecast target, and expensive candidate-specific response estimation without demonstrated behavioral payoff.

## Supporting diagnostics for later validation

These four previously proposed drill-downs are retained as background for validating a chosen design. They are not four additional intervention designs or a requirement to launch every analysis before making the design choice above. Fix the hypothesis, data families, arms, budget, and decision rule before any new comparisons. Start with existing fitting/development artifacts; protected and partial behavioral outcomes remain closed.

### 1. Does rank 4 repair action ranking, or mostly shared forecast bias?

**Question:** Within the same initial state, does the correction make the planner prefer an action with a better measured physical outcome?

On existing candidate banks with actual matched futures, separate error shared across candidates from action-dependent error. Score the actual future embeddings with the native goal objective and compare its ordering with measured physical progress. Then compare native, rank 4, and its matched random on ordering and selected-action regret, where complete paired predictions exist. Include the cheap coupling comparator when available. A shared embedding correction is not automatically ranking-neutral under a distance objective, so measure rankings directly rather than assuming cancellation.

This distinguishes three failures: no action-specific correction, improved forecast ranking under a misaligned goal objective, or useful choice improvement. Saved current scalar MSE metrics alone cannot answer it. Older banks support a limited diagnostic; missing rank predictions require bounded model replay, and missing alternate-action physical futures require simulator forks. Do not run full CEM episodes just to obtain this first check. **If rank adds no useful action discrimination, do not spend a large budget merely imitating it faster.**

### 2. Can one offline map replace the online response probes?

**Question:** How much of the candidate-specific response calculation is actually necessary?

Collect one bounded fitting-only teacher dataset containing native H3 scores, known candidate action suffixes, response matrices, delivered four-coefficient targets, and zero-treatment flags. An initial engineering proposal is 32 fitting families × four prespecified action sequences, with a family-level partition for fitting and checking the replacement; this is not an efficacy sample-size claim. Teacher probes supply model responses, not ground-truth physical futures.

Compare only two fixed replacements: the composed **4×4 map** derived in this note, and a small **action-conditioned four-coefficient predictor**. Keep B3/H3, basis, dose, and horizon unchanged. Assess delivered-edit direction/zero agreement, goal-score ordering, and complete planning latency. Matrix variation alone is insufficient because normalization can make different raw coefficients deliver the same edit. If the fixed map works, retain it; if only the conditional map works, identify the action dependence. If neither works, stop that compression attempt. **This is the largest plausible latency saving: removing 32 online probes, not optimizing a tiny solve.**

### 3. Do the four PCA directions contain the error that matters for decisions?

**Question:** Are high-variance activation directions also useful control directions?

The current basis is fitted to activation variance. Its visual-error readout is supervised, but that does not make the four basis vectors task-success directions. First use held fitting families to assess how well the existing PCA scores predict action-dependent goal-score error, where matched futures exist. Inspect visual and proprioceptive contributions separately at the outputs; do not assign causal meaning to intermediate channel indices.

If action-relevant error exists but is poorly represented by the current basis, a specific successor is one **rank-4 supervised basis fitted to within-state goal-score residual contrasts**, compared with the existing rank-4 PCA basis and a matched random basis. Freeze the supervised fitting algorithm before comparison. Keep site, rank, dose, fitting budget, and correction target fixed for this basis-only ablation; changing the target too would require a separate comparison. This is a new hypothesis, and a correlational readout still needs intervention validation. Never train the edit merely to make its own imagined goal distance smaller: it could hallucinate progress. **The opportunity is to change what the four directions represent, rather than increasing rank.**

### 4. Does the useful rank-4 effect require all 256 patches?

**Question:** Is there a compact spatial support for this particular rank-4 operator?

Start by measuring the four basis patterns' spatial energy from the saved bank. This is cheap description, not causal localization. For a subsequent fixed comparison, retain B3/H3 and compare the original rank-4 support with one fitting-selected 16-patch support and a size-matched random-position support, retaining direction controls. Check that masking preserves numerical rank, report retained energy, and match delivered dose so that removing positions is not confused with weakening the intervention.

Do not rerun every block × patch × rank combination. The existing rank-1 spatial failures make sparse rank-4 success uncertain, not impossible. If all-patch support remains necessary, keep the distributed pattern and make its coefficient selection cheap. **This drill-down improves specificity and possibly storage/application cost; it will not remove the probe bottleneck by itself.**

## Compute and generalization

Measured full-planner engineering episodes took **7,681–7,867 seconds each**, about **2.13–2.19 GPU-hours**, across four selected rank/combined task-arm checks. Each check used one excluded engineering scenario, so extrapolation is approximate. The slow portion of the initial MetaWorld panel contains `2 tasks × 4 slow arms × 96 episodes = 768 episodes`: approximately **1,640–1,680 GPU-hours**, before other tasks, training, and confirmation. Evaluating only one selected slow recipe plus its random control on both tasks over three seeds and ten late checkpoints would imply approximately **24,600–25,200 GPU-hours** at that throughput. This is a conditional projection, not a completed workload or a full-study ETA. [Execution and timing record](reports/EXECUTION_STATUS.md).

The cheap coupling engineering checks took about **313–316 seconds** including parity setup. Those checks and the rank timings are not a controlled head-to-head runtime benchmark, but the scale of the difference supports reassessing the implementation before expansion.

For the MetaWorld field dimensions, four FP32 basis vectors occupy approximately **1.56 MiB**. Keeping a separate three-direction projection adds **1.17 MiB**, totaling about **2.73 MiB** for those two factors. Their main projection/expansion arithmetic is about **716,800 multiply-accumulates per candidate**, excluding feature preparation, normalization, hooks, and logging. These are dimension-based calculations, not GPU benchmarks.

That makes eliminating response rollouts plausible as the major saving. A smaller solve alone would not address the bottleneck. Actual latency depends on integration, memory traffic, and whether a shadow rollout remains. Offline fitting also has a cost: it pays back only after enough uses, roughly when:

\[
N_{\text{uses}}>\frac{T_{\text{offline collection+fit}}}{t_{\text{current}}-t_{\text{replacement}}}.
\]

The timings must use the same hardware, workload, and unit of use.

For generalization, distinguish **reusing the fitting recipe** from **reusing identical learned tensors**. The former is a credible goal. The latter requires evidence: task-dependent effects already differ, and independently trained models can use different latent coordinates. A basis fitted to one model cannot be assumed to transfer to another seed or to DROID's different architecture. A shared fitting procedure with small task/model-specific banks is a more grounded initial target than one universal matrix.

### Spending and compute audit snapshot

Read-only Vast invoice inspection at approximately 00:27 UTC on September 8 queried posted charges for August 20–September 9. Scope attribution used the shared GPU board and the September 5 inventory. GPU charge quantities are machine-hours and were multiplied by the documented GPU count; storage/bandwidth quantities were not counted as GPU time.

| Identified rental scope | Billed GPU-hours | Posted cost including storage/bandwidth |
|---|---:|---:|
| Current/recent study leases, including retained stopped workers | 195.2 | $85.52 |
| Earlier documented project-associated workers | 415.2 | $96.95 |
| **Identified subtotal** | **610.4** | **$182.48** |
| Two earlier related JEPA investigations, attribution kept separate | 28.2 | $19.53 |

This is **allocated rental time, not measured useful GPU execution**, and not a complete all-time project invoice. It includes setup, idle time and unsuccessful work; some historical workers may span predecessor investigations. Recent unposted charges, unclassified leases, local CPU work, LLM/API usage, and inherited model pretraining are outside this subtotal. The provider returned 154 account charge rows; unrelated projects were excluded. Active useful compute cannot be reconstructed reliably from the current process receipts without a deduplicated job ledger.

For reproducibility, current/recent IDs were 50125440, 50189244, 50195621, 50205763, 50135088–50135090, and 50159352. Earlier worker IDs were 49155754, 49766237, 49902461, 49982193–49982195, 49987396, 49987398, 49987400, 49987402, 49987405, 49987408, 49987413, and 49987414. Separately attributed predecessor IDs were 49040330 and 49080864.

At this snapshot the 14 running GPUs plus retained stopped storage cost approximately **$6.93/hour, or $166/day**, before bandwidth. Current RTX 5090 compute rates are roughly **$0.35–$0.40 per GPU-hour before storage**. At those rates the initial slow panel alone would cost roughly **$570–$670 in compute**; the conditional seed/checkpoint expansion roughly **$8,600–$10,100**. These are workload projections, not approved additional budgets. The complete six-task cost remains unmeasured.

For context, COAST reports approximately **500 GPU-hours on A100/B200** covering rollout collection, sweeps, and evaluation. Its B200 latency benchmark reports 127.12 ms native versus 157.97 ms with COAST, approximately 24% overhead. This is not hardware- or workload-matched to our measurements, so there is no justified A100/B200-equivalent conversion. Its offline operator construction time excludes collecting the activations. [COAST compute disclosure and Appendix B.2](https://arxiv.org/html/2605.17144).

Both projects inherit pretrained models and mathematical tools. Our PCA, covariance and regularized inverse are established techniques; COAST builds on conceptors and earlier steering work. Our new integration, instrumentation, repairs and validation are real engineering effort, but their cost cannot be inferred from GPU-hours alone. Building that machinery explains some sunk cost; it does not eliminate the rank operator's recurring probe cost or establish that further spending is worthwhile.

## What would make an offline replacement convincing

After the action-relevance diagnostic above, my first compression comparison would be a fixed-response approximation against a small action-conditioned coefficient predictor, retaining the same edit basis and dose. A compact conditional response model is a later possibility, not an automatically authorized follow-up. Gating addresses a separate question: when intervention helps.

The key checks would be:

1. **Can we reproduce the delivered teacher edits on unseen fitting families?** Assess direction, dose, and zero-treatment agreement, with family-level splits.
2. **Does compression retain forecast improvements?** Check both visual and proprioceptive endpoints; matching coefficients is insufficient by itself.
3. **Does it preserve or improve candidate selection and task outcomes?** Candidate rankings can change even when average forecast errors look similar.
4. **Does the saving survive realistic planner inputs?** Recorded action trajectories do not cover every candidate CEM proposes. New candidate labels from the model can teach teacher imitation, but are not ground-truth physical futures. Sequential distribution shift is a known limitation of imitation-based replacements. [Ross, Gordon, and Bagnell](https://arxiv.org/abs/1011.0686)
5. **Is it cheaper at acceptable quality, including fitting cost?** Measure full planning latency, memory, and reuse needed to recover the offline cost.

Any successor would be a separately specified method with fresh evaluation. It would not retroactively replace the frozen intervention whose results we already report.
