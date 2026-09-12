# Findings and paper context

*Updated September 8, 2026. Completed local evidence, the registered protocol, implementation, and related primary papers were checked. This is the main context document for writing the extended abstract. The longer [research synthesis](paper/research_synthesis.md) answers each methodological question with sources and proposed validations. Earlier wording is preserved in the [previous findings snapshot](paper/notes/findings_before_expanded_review_2026-09-08.md).*

## What we can say today

### What the 960-episode panel will resolve — September 8, 20:41 UTC

Outcome-blind live file counts at20:41:41UTC (4:41PMEastern): **875/960 episode
records**, leaving85 records. Native, revised fixed rank4, its matched random,
and coupling-only each have96 records per task. The remaining matched-random
coupling arm has54 Reach and53 Reach-Wall records. No completed analysis receipt
was found. These are collection counts, not validated success-rate estimates;
no partial outcomes were read for this update.

The panel is **2 tasks × 5 arms × 96 episodes**. It measures the behavioral effect
of revised rank4 and existing equal-budget coupling, each against native and its
own random control. It does not contain a combined rank4-plus-coupling arm.

| Question | Addressed by this panel? |
|---|---|
| Does revised rank4 improve Reach/Reach-Wall task success beyond native and matched random? | Yes, subject to the completed paired analysis and its uncertainty. |
| Does coupling alone improve task success on these tasks? | Yes, with its own controls. |
| What are observed episode times, goal distances and rewards? | Reported as secondary summaries; total adaptation/research cost is separate. |
| Do all other rank/layer/spatial/geometry arms help behavior? | No. They are absent from these five arms. |
| Does the revised combination or HMM help? | No. Those comparisons remain unfinished/deferred. |
| Does the method generalize to other tasks, fresh scenarios, or independently trained models? | No. This is a development panel on the retained MetaWorld checkpoint. |

There are four preregistered success contrasts per task: each of the two proposed
interventions versus native and versus its own matched-random arm. Direct rank4
versus coupling superiority is not one of the eight registered contrasts.
The existing selection rule requires an observed native success gain of at least
5percentage points and positive simultaneous95% lower bounds against both native
and matched random. It does **not** require that the native lower bound exceed
5points. Repeated scenarios remain clustered;960 executions are not960 independent
scenarios or independent trained models. Finishing the panel may yield an
inconclusive result rather than a definitive yes/no.

For the intended practical steering contribution, the primary objective is
**task success at a fixed, affordable planning budget**, with intervention latency
and calibration cost reported. Forecast loss, geometry diagnostics and action
traces help explain outcomes; they cannot substitute for behavioral success.
JEPA-WM explicitly identifies success rate as its main metric in
[Section5.1](https://arxiv.org/html/2512.24497v4#S5.SS1).

Interpret the completed core comparison before choosing a further empirical
intervention. A learned edit that beats native but not random lacks established
direction specificity. A wide success interval is uncertainty, not proof of zero
effect; an interval excluding the useful positive range argues against further
investment in that exact operator under the tested setup. Positive results would
still require independent confirmation before generalization claims.
The offline patterns suggest questions about pathway, spatial placement and
downstream propagation, but do not yet identify which edit best improves action
selection. No new experiment, arm, spending, or execution-priority change follows
from this interpretation.

### Complete ablation inventory and how to use mixed results — September 8

The [all-task annotated ablation tables](paper/tables/all_task_ablation_audit.md)
now collect every arm versus native from 41 available completed forecast reports,
including controls, both H6 embedding metrics, simultaneous 95% intervals, and
BF16/FP32 separately. The linked CSVs retain 1,672 registered forecast contrasts
and 9,840 reported all-horizon metric rows. The complete DROID panel is reported
separately in action-score units, including all 16 registered contrasts. Counts
refer to measurements/comparisons, not independent experiments or discoveries.

**Mixed task outcomes are useful for refining a hypothesis.** Task-specific fits
and failures do not make the experiment worthless. They also do not identify the
cause of the difference: task, checkpoint, data, fitted direction, dose and outcome
metric can differ together. A revised method motivated by these observations
would need separate validation; the present development data cannot become its
independent confirmation.

The prior phrase "the other ablations had no effect" is inaccurate. The compact
inventory below concerns **BF16 H6 proprioceptive embedding error versus native**.
Positive means its within-sweep simultaneous interval excludes zero; it does not
mean the full useful-effect/random-control criteria passed.

| Family / non-control variants | Reach | Reach-Wall | Push-T | Wall | PointMaze |
|---|---|---|---|---|---|
| Coupling: visual, action-only, joint, equal-energy joint | Four positive | Four positive | Action-only positive; visual/joint negative; equal-energy inconclusive | Visual/joint/equal-energy negative; action-only inconclusive | Recorded complete; numerical aggregate not recovered in this audit |
| Linear, cubic, projected cubic, reflected curvature | Four inconclusive | Four inconclusive | Four inconclusive | Four inconclusive | Four inconclusive |
| Rank 1/4/8 | Three positive | Three positive | Three inconclusive | Fits are not completed comparisons | Fits are not completed comparisons |
| Six singleton layers, B2+B3, all six | Eight positive | Eight positive | Eight inconclusive | No completed comparison verified | No completed comparison verified |
| One patch, contiguous 16, scattered 16, all patches | Four positive | Four positive | Four inconclusive | No completed comparison verified | No completed comparison verified |
| Original combined, combined rank 1, two drop-one arms | Four positive | Not measured | Not measured | Not measured | Not measured |
| Revised fixed rank 4 | Positive | Positive | Not measured | Not measured | Not measured |

These rows reuse the same 33 Reach, 27 Reach-Wall, 21 Push-T, 192 Wall and 200
PointMaze lineages. Several category reference arms are the same operation under
different contrast families. The table does not imply that every Reach-Wall arm
qualified: the original rank-4 arm was the only one passing the original full
advancement gates there. Both modalities must be consulted: Reach-Wall coupling's
visual changes are inconclusive despite positive proprioceptive changes.

**Numerical evidence gap:** PointMaze coupling BF16/FP32 and Wall coupling FP32
are described as complete by execution records, but their original numerical
aggregates were not recovered from the available local exports. The tables do
not assign null results to those missing reports. Wall BF16 uses its preserved
compact report; navigation geometry and DROID use checksum-bound archived reports.
The source ledger states the verification level of each entry. No raw bootstrap
was rerun and no incomplete behavioral outcomes were inspected.

The user clarified that the methodological reference is **JEPA-WM**, not V-JEPA 2.
Its [Section 4](https://arxiv.org/html/2512.24497v4#S4) documents empirical exploration
and controlled design comparisons. Its [Appendix D](https://arxiv.org/html/2512.24497v4#A4)
provides an error-propagation argument, not a derivation of the winning architecture
or a dated research diary. The paper does not establish whether that mathematical
explanation was developed before or after particular experiments.

For our study, the gaps are more specific than "we should have known the outcome":
B3/H3 was a reference choice, not a demonstrated physics zone; the PCA directions
are not semantically validated physical coordinates; the primary forecast endpoint
was not independently validated as a surrogate for the behavioral effect of our
interventions; and the old online response computation was not screened early
enough for planner cost. The existing controls still provide useful discrimination.
For example, Push-T's positive action-only effect versus negative visual/joint effects
is more informative than labeling the entire task null. Possible physical explanations
in the full tables are explicitly hypotheses, not recovered mechanisms.

**Latest completed-results review: September 8, 19:15 UTC.** DROID's coupling comparison is now complete and inconclusive. Completed Wall/PointMaze geometry results extend the precision/reconstruction boundary, and the cheaper combined operator has a successful GPU forecast engineering check. There is **no newly established practically useful efficacy improvement in these panels**. See the [afternoon findings](#afternoon-completed-results--september-8-checked-at-1915-utc); earlier time-stamped execution statements below are historical.

**The evidence supports a controlled study of correctable forecast structure in frozen JEPA predictors. It does not yet establish improved task success, a semantic map of physics, or general-purpose manifold steering.** The useful finding is that intervention effects depend on spatial arrangement and pathway, and that some positive forecast effects survive a separately evaluated operator that removes expensive online response estimation.

A status correction matters: the cheaper fixed-response operator **now has completed offline results**. The old statement that it was entirely unvalidated is superseded. Its closed-loop efficacy remains pending. The local execution snapshot checked at approximately 07:42 UTC reports the five-arm Reach/Reach-Wall behavioral panel underway; we did not inspect partial candidate outcomes to choose a paper story. No new experiment or GPU job was launched for this document.

### Completed evidence worth building on

Positive percentages below mean lower **H6 proprioceptive embedding MSE**, not physical-state error or task-success improvement. Primary offline data comprise **33 Reach, 27 Reach-Wall, and 21 Push-T development trajectories**. The successor reuses the first two pools; it does not add 60 independent trajectories. Reported 95% intervals are simultaneous within task/category/precision, not across all research claims.

| Finding | Completed evidence | Why it matters; boundary on interpretation |
|---|---|---|
| **Compact forecast correction** | Original rank 4: Reach **2.89% [2.28, 3.50]**; Reach-Wall **2.14% [1.65, 2.63]**. Push-T **0.20% [-0.06, 0.46]** is inconclusive. | Some prediction error is correctable through a small activation span. Rank 1 also works on Reach; four is not the dimension of physics. |
| **Offline fitting can replace online response probes** | Fixed-response BF16: Reach **2.36% [1.97, 2.75]**; Reach-Wall **2.19% [1.80, 2.59]**. Advantages over matched random: **1.20% [0.96, 1.44]** and **1.48% [1.18, 1.77]**, scaled by native error. | A positive forecast effect is observed with zero online response probes and no native-shadow forecast. No old/new equivalence test or population latency claim is established. Reach FP32 misses the frozen practical minimum against random. |
| **Combination can help, with an energy caveat** | Original Reach coupling plus rank 4: **4.82% [3.67, 5.98]**, beating random and component-removal arms. | The components can provide additional benefit in this configuration. Removing a component also removes energy, so this is not equal-energy mechanistic synergy. No corresponding fixed-response combined result exists. |
| **Spatial arrangement has a measurable role** | Reach joint beats its spatial permutation by **2.02% [1.09, 2.94]** of native error. | Broad edit size alone cannot explain this difference. Patch assignment matters; its physical meaning remains unidentified. |
| **Pathways can have opposing effects** | Push-T visual-only increases error **1.02%**; action-conditioning-only reduces it **0.34%**, below the useful-effect threshold. Joint worsens error **0.64%**. | This is a lead for explaining task dependence. Contact, motion, or object alignment are hypotheses, not findings. Reach equal-energy joint does not establish superiority to visual-only. |
| **Spatial and layer distribution are distinct** | Only all-patch spatial support passed the Reach spatial gates. Several single-layer arms worked; B0 reduced error **3.03%** in the rank-1 depth sweep. | Broad spatial support does not imply editing every layer. Separate sweeps do not validate moving the rank-4 successor to B0. |
| **Reconstruction fidelity is an unreliable selector here** | Push-T cubic reconstruction has approximately **1,192x lower** omitted-activation MSE in FP32, but **35% higher** in BF16. FP32 cubic forecast improvement is only **0.0046% [-0.0032, 0.0124]**. | A strong geometry diagnostic need not identify a useful correction. This local four-anchor test does not refute manifold steering or discover a cubic physical law. |

Sources: [corrected primary results](reports/CORRECTED_OFFLINE_RESULTS.md), [fixed-response method and results](docs/FIXED_RESPONSE_RANK4.md), and [all 1,536 registered contrasts from 36 included report scopes](paper/data/reported_contrasts.csv). The [evidence appendix](paper/appendix.md) documents provenance and inference families. Wall/PointMaze have additional completed offline work outside these primary figures; the full six-task program also includes DROID. DROID's endpoint concerns recorded robot actions, not newly executed physical-robot success.

## The big picture: a map for choosing actions

Tong, Fan, and colleagues use Plato's cave to motivate learning from visual experience beyond language. Our equivalent should ask what makes the internal representation **usable once learned**:

> A map is useful because it preserves the relationships needed to choose a route. A world model must preserve how actions change possible futures. Can we identify and cheaply correct errors in those relationships without retraining the model?

This gives the introduction a concrete progression: **representation → intervention → forecast → decision**. Our evidence reaches the forecast link. Geometry is the object of investigation; its practical value has to be demonstrated through the relationships that an edit preserves or changes. The stronger long-term vision is an interpretable, editable interface to imagined action consequences, with a bounded adaptation budget. [Beyond Language Modeling](https://arxiv.org/html/2603.03276v1)

Why timely: physical representation analysis, frozen-policy steering, and efficient latent world models now make that connection testable. Why potentially useful: a compact correction could adapt a retained checkpoint, expose a failure regime, or control a specific predicted variable with less retraining. Why it may fail: an edit can improve recorded-action error while distorting candidate ordering, or cost more than using a smaller model. Those alternatives belong in the motivation and evaluation, not just an appendix disclaimer.

## Three contributions to introduce, at the strength supported today

1. **A controlled study of correctable forecast structure.** Paired native/random, spatial permutation, pathway, rank, and support controls reveal which tested interventions help or harm. The possible novelty is this specific evidence inside an action-conditioned frozen predictor, particularly arrangement and opposing pathway effects. “Activations are distributed” and “steering can change outputs” are already established.
2. **An offline-fitted distributed correction without online response probes.** One fitted coefficient map applies one distributed edit in the same forecast. The successor has its own positive reaching-task development results. Its practical contribution is separating response-calibration cost from application; low rank and ridge regression themselves are not new mathematics. A stronger efficient-method claim still needs completed behavior and representative cost measurements.
3. **A concrete boundary on geometric reconstruction as a steering diagnostic.** The tested reconstruction advantage reverses with precision and does not produce useful forecast correction. This can inform diagnostic design if reported precisely; merely observing a generic loss/success mismatch is not novel.

These are defensible contribution candidates, not a priority claim or guarantee of publication. A four-page paper should center on the first two, with the third and Push-T as boundaries. A null behavioral result could support a narrower paper if the controls explain a specific failure. The outcome should determine the claim, rather than the abstract assuming success in advance.

## Closest prior art and the remaining opening

| Prior art | Existing contribution we must credit | Specific opening for this study |
|---|---|---|
| [JEPA-WM](https://arxiv.org/html/2512.24497v4) | Systematic architecture/training/planning ablations, including prediction/planning mismatch. | Test corrections inside a retained predictor and their cost, rather than claiming the mismatch itself. |
| [Joseph et al.: Physics Emergence Zone](https://arxiv.org/html/2602.07050v1) | Physical representation analysis and steering in video encoders. | Investigate compact forecast-error correction in the action-conditioned predictor; do not equate the two subspaces. |
| [COAST](https://arxiv.org/html/2605.17144v1) | Offline-fitted activation gates in frozen robot policies, with behavioral evaluation. | Correct imagined action consequences upstream of an unchanged planner. Our stronger behavioral/practical claim is still pending. |
| [Manifold Steering](https://arxiv.org/html/2605.05115v1) | Links fitted activation geometry to structured output paths across modalities. | Test whether a geometric diagnostic selects a useful forecast correction; we have not implemented its manifold method. |
| [ACPC](https://arxiv.org/html/2608.12939v1) | Diagnoses consistency under observation perturbations with shared future actions. | Establish a selective repair, rather than a diagnostic alone, and test the action distinctions it preserves. |

The common problem is how internal representation structure becomes reliable prediction and control. The novelty case must come from a specific new result along that chain, not a new name for activation editing.

## How the findings answer the original hypothesis

The [original objective](docs/EXPERIMENT_PLAN.md) asks whether frozen-model interventions have reproducible, selective forecast effects and whether benefits survive candidate ranking and control. **We have task-specific development support for forecast correction, partial specificity evidence, and no completed answer yet for behavioral utility or independent-model replication.**

| Ablation | Why it was selected | What it cannot establish alone |
|---|---|---|
| Rank 1/4/8 plus matched random | Test correction capacity with a bounded candidate set and a smallest-eligible-equivalent rule. | Intrinsic physical dimension; failure of all smaller ranks; global optimality. |
| Visual/action/joint/permuted/equal-energy | Separate pathway, placement, and energy explanations. | Semantic causality or synergy from unequal-energy comparisons. |
| Separate depth and patch-support sweeps | Distinguish where in depth from where across the image an edit operates. | A full circuit or an optimal crossed layer-position configuration. |
| Equal-anchor linear/cubic and curvature controls | Ask whether local nonlinear fidelity predicts useful downstream changes. | Global manifold geometry or general superiority of nonlinear steering. |
| Fixed-response native/zero/random comparison | Test a changed operator that eliminates online response probes. | Equivalence to the original or better full-task performance. |

We did not know these were globally “the right” ablations. Their justification is that each discriminates a stated alternative while controlling specified factors. That scientific logic did not justify retaining expensive diagnostic probes inside every planner candidate. The [full hypothesis table](paper/research_synthesis.md#1-what-hypothesis-did-we-actually-test) records evidence, verdicts, and residual confounds.

## What our map is, and what Sonia's PEZ does and does not imply

The present map is concrete but limited: **three fitted PCA readout scores plus an intercept → a 4-by-4 coefficient map → four distributed correction directions** in the native activation field. For MetaWorld the field is **256 x 400 = 102,400 coordinates**, edited once at predictor B3 during H3. Coordinate count is support size, not a count of separately tested ablations. It is also not 98,304 for this 400-feature configuration.

The [actual-basis figure](paper/figures/fig06_fitted_basis.png) shows squared loadings across the patch grid. It provides an honest answer to “what does this map look like?” It does not label a speed axis, a contact circuit, or a recovered physical manifold. A low-dimensional visualization is not a proof of canonical native coordinates. Standard PCA, covariance fitting, finite differences, and regularized inversion underlie the current operator; we have not introduced a new global atlas, SAE, or transcoder discovery method.

Joseph et al.'s **Physics Emergence Zone concerns video encoders**. We edit a **dynamics predictor downstream of DINOv2**. Our B3 has not been identified as a PEZ. A compact correction to a particular forecast error can coexist with a much larger representation needed to encode or steer a physical variable. That is the conceptual connection; our results do not yet identify those physical variables. [Interpreting Physics in Video World Models](https://arxiv.org/html/2602.07050v1)

For genuine manifold steering, we would need a validated chart or metric and evidence that its interventions preserve meaningful structure better than matched linear controls. [Manifold Steering](https://arxiv.org/html/2605.05115v1) is direct prior art. The current linear basis is a starting point for that question, not an implementation of the paper's approach.

## Is circuit analysis required or useful?

**For LLMs, full circuit analysis is not required to train, use, or demonstrate useful steering. It can be valuable for explaining a specific computation, locating a failure, and predicting intervention effects.** A causal subspace intervention is not automatically a circuit explanation.

There is empirical support for both usefulness and caution: [GPT-2 indirect-object circuits](https://arxiv.org/abs/2211.00593) validate task-specific explanations using interventions; [EAP-IG](https://arxiv.org/html/2403.17806v2) tests whether discovered circuits recover behavior; [activation-patching best practices](https://arxiv.org/abs/2309.16042) shows dependence on metrics/corruptions; and [Data-driven Circuit Discovery](https://arxiv.org/html/2605.09129v1) finds that circuits can be dataset-specific even when the human task semantics are preserved.

For us, the claim determines the burden of evidence. Better task outcomes require good outcome/cost comparisons. Fine-grained control additionally requires a target variable, dose response, and preservation of other relevant variables. A mechanistic explanation additionally requires validated mediation or path evidence. **We should not impose whole-model circuit tracing as a prerequisite for this paper.** Targeted causal checks of the observed pathway/spatial effect could be useful; a large tracing exercise without a specific decision or explanation to resolve would repeat the compute problem.

Neuroscience and cognitive science offer relevant foundations beyond LLM circuits: experimentally changing neural population-to-action mappings, studying manifold constraints on learning, and distinguishing readable information from usable control. These analogies motivate tests; they are not validation of our JEPA result. See the [cross-disciplinary evidence and researchers](paper/research_synthesis.md#8-researchers-and-perspectives-worth-following).

## Why JEPA-WM, and does it still use EMA or teacher/student learning?

We chose a useful controlled testbed: released checkpoints, an inspectable action-conditioned predictor, native CEM code, and task interfaces close to the benchmark. This does not prove JEPA-WM is superior to every alternative. The [methodology table](paper/tables/methodology_alignment.md) separates what we inherited from the authors from our new intervention sweeps.

**JEPA-WM's studied training recipe freezes the pretrained visual encoder and trains the predictor/action/proprio modules; it does not update an EMA teacher in that stage. Our steering evaluations freeze all base modules.** Teacher/student training in the encoder's earlier pretraining is a separate matter. In the code, `target_encoder` can name loaded checkpoint weights; it does not by itself mean an EMA loop is running. See [paper Section 3](https://arxiv.org/html/2512.24497v4#S3), [pinned encoder initialization](vendor/jepa-wms/app/vjepa_wm/utils.py), and [training code](vendor/jepa-wms/app/vjepa_wm/train.py).

[LeWM](https://arxiv.org/html/2603.19312v3) and [LeVJEPA](https://arxiv.org/html/2608.27395v1) simplify different training problems using explicit anti-collapse regularization. LeVJEPA removes target/predictor machinery from **video pretraining**, rather than removing the need to model action consequences in a planner. They are different methods, not updates silently incorporated into our checkpoint.

## Efficiency, fine-grained control, and a repeatable system

Our practical opportunity is **cheap adaptation and controlled reuse of an existing model**. LeWM targets economical end-to-end world modeling; LeVJEPA targets economical video pretraining; [Fast-LeWM](https://arxiv.org/html/2606.26217) changes the dynamics interface to parallel action-prefix prediction. Those may be better choices when building a new system. Frozen steering is potentially complementary, and currently has no evidence of superiority at equal total cost.

The successor removes the original repeated-response bottleneck. Applying it still pays projection/expansion, memory traffic, dose normalization, and hook overhead for every CEM candidate. Its main dense products require roughly **716,816 multiply-accumulates per candidate application**, before other costs; this is an arithmetic estimate, not a measured latency. One excluded episode timing pair, **314.28 s native / 313.03 s edited**, does not establish average overhead or speedup. The [compute accounting](paper/research_synthesis.md#6-what-would-make-this-useful-and-efficient) includes offline calibration and all candidate applications.

Three focused directions follow from the evidence:

1. **Explain when the correction helps.** Use spatial/pathway findings to formulate a specific physical or error-regime prediction, then validate it prospectively. Interpretability value comes from predicting failures, not just naming a cluster.
2. **Make the correction a calibrated control interface.** Identify a target variable, test whether coefficients/dose change it predictably, and measure side effects. The current error correction is not yet a user-selectable motion or clearance control.
3. **Test reusable adaptation at a fixed total budget.** Compare the frozen operator, identity, and an appropriate cheap adaptation baseline on unseen scenarios/tasks, with planner latency and success together. The basis may require refitting across checkpoints; the transferable object can be the fitting procedure rather than identical coordinates.

A future sparse mixture of small correction experts could select **one expert or identity per forecast** using present state/action summaries. This preserves the intended single-edit application budget. It must beat a shared correction and a matched ungated alternative; task-ID lookup alone would not demonstrate generalization. [Beyond Language Modeling](https://arxiv.org/html/2603.03276v1) already studies MoE pretraining, and [Branch-JEPA v3](https://arxiv.org/html/2607.05238v3) already studies weighted multiple latent futures. Multiple possible futures and sparse correction routing solve different problems. No MoE, HMM, cross-task transfer, or universal JEPA scaling solution is established by our present results.

## Paper artifacts and software

- [Workshop publication assessment](paper/publication_readiness.md): September 8 review of novelty, evidence limits, minimum revisions, and actual workshop fit/deadlines. A narrow empirical short paper is defensible after revision; a practical task-success steering claim still requires new evidence. Workshop acceptance is uncertain and the number of experiments is not evidence of novelty.
- [Paper entry point](paper/README.md): editable draft, sources, figures, and rebuild instructions.
- [Four-page extended-abstract text](paper/extended_abstract.md) and [PDF](paper/extended_abstract.pdf): four core pages; references and figures separate.
- [Figure gallery](paper/figures.md): six PNG/SVG figures, including actual basis loadings. Task-success figures await completed panels.
- [JEPA-WM methodology alignment](paper/tables/methodology_alignment.md) and [task coverage](paper/tables/task_coverage.md).
- [Detailed research synthesis](paper/research_synthesis.md): hypotheses, validations, circuit evidence, architecture, compute, and researchers.

**Plotting software is verified, not guessed:** the authors' released [design-choice plotting script](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/plot/logs_plan_joint_per_design_choice.py) and [curve utilities](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/plot/logs_plan_joint_unif_utils.py) import **Matplotlib and Seaborn**. We use those libraries with editable SVG and PNG outputs. ReportLab renders our draft PDF; it is not claimed to be the authors' manuscript tool. Figure scripts reuse completed estimates/intervals and existing fitted weights, with source hashes; they run no model experiments.

Relevant people include **Sonia Joseph** (McGill/Mila PhD candidate, neuroscience/CS background), **Daniel Wurgaft** (Stanford Psychology PhD student), **Can Rager** (physics-trained independent researcher), and the mathematics/neuroscience group of **Guillaume Lajoie**, including neuroscience PhD student **Léo Choinière**. The [researcher table](paper/research_synthesis.md#8-researchers-and-perspectives-worth-following) links their public profiles and explains each connection; no contact or endorsement is implied.


## Overnight follow-up — September 8, checked through 12:38 UTC

The new completed HMM fit diagnostic adds two aspects of the same result, not two independent efficacy studies. Source reports and diagnostic hashes were verified against their completion receipts; no partial behavioral outcome was analyzed.

1. **This history-based estimator does not improve its next-state diagnostic on average.** Four family-disjoint folds use 128 fitting families / 512 prefixes per task. Predicting H3 internal-feature log density from H1/H2 instead of H2 alone gives mean gains of **-0.04538 nats on Reach** and **-0.01145 nats on Reach-Wall**. All four fold means are negative in each task. This fails to demonstrate incremental predictive value for the specified two-state/PCA3 HMM. It is not a test proving that history is unnecessary generally, and there is no task-success or steering-effect endpoint here.
2. **Win frequency and average benefit disagree on Reach-Wall.** **93/128 families (72.7%)** have positive diagnostic gains there, yet the aggregate is negative: losses among the remaining families outweigh those improvements. Reach has only **47/128 positive families (36.7%)**. This motivates examining heterogeneous effects and preserving an identity/cheap comparator in future routing work, without claiming that we have identified the harmful regimes or measured behavioral harm. These are development/fitting-population diagnostics, not fresh confirmation; no significance claim is inferred from the fold signs or positive-family count.

Sources: [Reach diagnostic](artifacts/offline_study/hmm-fixed-response-20260908-v1/fits/reach/diagnostics.json), [Reach-Wall diagnostic](artifacts/offline_study/hmm-fixed-response-20260908-v1/fits/reach-wall/diagnostics.json), and [exact diagnostic calculation](src/offline_study/routing_fit.py).

Behavioral collection has advanced: 96 native, 96 refined, and 96 matched-random episodes per reaching task (576 total references) are verified in the latest execution record. The remaining registered comparisons and final paired analyses are incomplete, so this is execution progress rather than a newly established success-rate improvement. DROID's receiving engineering passed on four GPUs, but its recorded-action candidate comparison is incomplete and is not robot-success evidence.

A separate methodological issue affects **new PointMaze training histories**: our wrappers shuffled samples while the pinned upstream samplers did not. Those histories cannot be labeled author-native or resumed as corrected replications. The released-checkpoint offline/behavioral comparisons are unaffected by this defect. See the [sampler correction record](reports/POINTMAZE_TRAINING_SAMPLER_CORRECTION.md). This is a replication correction, not a scientific performance finding.

## Afternoon completed results — September 8, checked at 19:15 UTC

These are newly reviewed completed local artifacts, not a claim that every underlying trajectory ran this afternoon. No new experiments were launched and no incomplete behavioral outcomes were selected for this summary. Existing positive Reach/Reach-Wall offline results above remain separate evidence.

### 1. Reconstruction fidelity and useful correction separate on two more tasks

The completed navigation geometry panels contain **192 Wall and 200 PointMaze trajectory groups**, reused across FP32 and BF16. The ratio of group-weighted linear to cubic omitted-activation MSE is **3,967.4 on Wall and 7,019.8 on PointMaze in FP32**. Cubic therefore reconstructs the omitted native activation much more accurately in this test. In BF16, cubic instead has **43.3% and 42.2% higher reconstruction error**, respectively.

That FP32 reconstruction advantage does **not** establish better H6 forecasts. Cubic's visual-MSE reductions versus native are **0.00661% [-0.00515, 0.01837] on Wall** and **0.00080% [-0.00288, 0.00447] on PointMaze**; both simultaneous 95% intervals include zero. The corresponding proprioceptive comparisons are also inconclusive. The intervals are simultaneous within task/category/precision, not across this entire research narrative.

It would nevertheless be inaccurate to call every contrast statistically null. For example, PointMaze FP32 linear reconstruction reduces visual MSE by **0.00785% [0.00288, 0.01282]** versus native and also beats matched random. Wall's random control has small positive proprioceptive effects. **No registered contrast in these four reports passes its frozen useful-effect threshold.** Quantization also prevents treating the delivered edits as exactly energy-matched in all cases.

**Paper value:** the earlier Push-T observation now has evidence on two more tasks: local reconstruction fidelity is precision-sensitive and does not reliably select useful forecast corrections under this intervention protocol. This is a specific diagnostic boundary, not a rejection of manifold steering or proof that geometry is irrelevant. These are development/replication trajectories, not independent trained-model replications or closed-loop task-success results.

Sources: [closure audit](reports/NAVIGATION_GEOMETRY_CLOSURE_READINESS.md) and [preserved results archive](artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz), members `navigation-analysis-20260907-v1/{float32,bfloat16}/{wall,pointmaze}/action_response_geometry/report.json`. Archive SHA-256 and all four report-to-DONE hashes were verified. Ratios above use `mechanism_diagnostics.per_task_group_weighted` aggregate MSEs, not averages of per-example ratios. These panels are additional to the 36 scopes in the current paper figures/CSV.

### 2. The cheaper combined edit has measured, bounded forecast overhead

The GPU engineering report passes **18/18 byte-exact reference comparisons**: two arms, candidate counts 8/19/300, and horizons 2/5/6. At 300 candidates and H6 on an RTX 4090 in FP32, mean synchronized forecast times over three repeats are **3.16290 s native**, **3.17330 s fixed rank 4 (+0.33%)**, and **3.57011 s combined fixed rank 4 (+12.87%)**.

The combined implementation uses one backend forecast with a short native-prefix replay, **40 predictor-block executions versus 36 native or 72 for its full reference**, and zero online response-probe rollouts. Exactness is relative to the reference using the **same fixed coefficient map**, not equivalence to the original online inverse or inheritance of its 4.82% Reach result.

**Paper value:** a concrete engineering route to affordable composition. This is one fitting stimulus, one device, and a small timing sample; it does not establish population overhead, full-planner latency, or efficacy. The checked receipt explicitly reports no full CEM episode and no behavioral launch readiness. Full-planner checks were not complete in the local receipts reviewed here.

Source: [GPU engineering report](artifacts/offline_study/combined-engineering-tx3-20260908-v1/check/report.json), verified against [DONE](artifacts/offline_study/combined-engineering-tx3-20260908-v1/check/DONE.json), SHA-256 `c6faf9cd35eb981457269df3876e2226853e3b3fe2d7e8fecc52fe955ee52897`.

### 3. DROID coupling completes without an established benefit

The nine-arm panel is complete: **576 endpoint evaluations, 64 paired endpoints per arm from 15 recordings**. Native checkpoint score is **51.10**, joint coupling **50.85**, and equal-energy joint **51.07**. Joint-minus-native is **-0.247 score points [-1.176, 0.598]**; equal-energy joint-minus-native is **-0.029 [-0.555, 0.615]**. **All 16 simultaneous 95% contrast intervals include zero.**

**Paper value:** the positive Reach coupling result is not established in this DROID-specific implementation and endpoint. This limits a universal-benefit claim; it does not isolate task, data, architecture, or endpoint as the cause. Scores measure recorded-action planning error and are **not robot success percentages**; there were zero physical robot executions. Fifteen recording clusters and incomplete training-seed histories limit generalization. This is an inconclusive development comparison, not an equivalence test proving zero effect.

Sources: [verified coverage summary](reports/ABLATION_COVERAGE_STATUS.md) and [DROID archive](artifacts/offline_study/live-20260908T151000Z/instance-50259194-droid-parallel/instance-50259194-results.tar.gz), member `droid-coupling-behavior-20260908-v3/analysis/report.json`; report SHA-256 `b998218db8f1b87d18150bed04b9eec29731ef50af5e8c203f7ba028b9f83de8` matches its DONE receipt. Other DROID interventions and the remaining simulated behavioral comparisons are separate unfinished work.
