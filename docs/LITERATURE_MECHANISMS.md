# What latent steering changes, and what it leaves unresolved

Literature checked September 13, 2026. This is a research assessment, not a new
experimental result or execution authorization. Current project measurements are
linked to [Results](RESULTS.md) and [Mechanisms](MECHANISMS.md). Proposed thresholds
below are prospective choices, not thresholds used to obtain those results.

## 1. Verdict and confidence

**Run a limited pilot.** The strongest next question is which part of the applied
edit changes candidate scores, and whether those score changes favor better physical
outcomes. Confidence is high that the current evidence motivates this question;
confidence is low that any particular geometric or attention mechanism explains
the missing net success gain.

The current case study has a useful positive finding: a measured internal edit
changes which scenarios succeed. On fresh Reach, native and refined planning both
succeed in 52/96 scenarios, with 21 rescues and 21 regressions. That is substantial
behavioral sensitivity with no observed aggregate gain. The full four-task,
eight-arm panel contains 384 paired scenarios and 3,072 evaluations; its simultaneous
intervals establish no reliable improvement. [Project results](RESULTS.md).

## 2. The decision and its assumptions

The decision is whether to invest next in identifying a mechanism within the
existing frozen predictor, changing the intervention geometry, or adding adaptation
and memory. Start with the existing predictor: its coefficient traces already
identify a concrete component to intervene on, and its causal path to CEM remains
unmeasured.

The refined recipe lowered development H6 proprioceptive embedding MSE by 2.360%
on Reach and 2.192% on Reach-Wall, on 33 and 27 development lineages. That endpoint
does not measure counterfactual action quality. Reach and Reach-Wall share a base
checkpoint but have separate fitted interventions. Fitting is task-specific; these
results do not demonstrate transfer to an unfitted task. [Project results](RESULTS.md).

**The architectural distinction:** COAST's intervention feeds the policy's
action-generation computation. Here the intervention feeds an action-conditioned
forecast, which is scored against a goal; CEM selects actions from those scores.
Both are latent activation steering. The extra forecast-to-score-to-selection
relationship supplies additional hypotheses about why an edit changes behavior.
[COAST §3](https://arxiv.org/html/2605.17144v1#S3),
[JEPA-WM](https://arxiv.org/html/2512.24497v4).

COAST applies a full matrix without eigenvalue truncation. Algebraically its edit
is `δh=β(C−I)h`: a low effective rank for `C` does not make the edit itself rank-four
or even low-rank. [COAST Appendix A.9](https://arxiv.org/html/2605.17144v1#A1.SS9).

## 3. Verified literature and evaluation landscape

| Work; latest checked version | Mechanism and reported evaluation | What transfers to this question |
|---|---|---|
| [COAST, Miao et al.; v1, May 16, 2026](https://arxiv.org/html/2605.17144v1) | Success/failure covariance conceptors form `C = C_success AND NOT C_failure`; an action-expert residual becomes `h[(1−β)I+βC]ᵀ`. Weights stay frozen. §4.1 specifies 15 fit/selection rollouts and 30 held-out simulation rollouts; physical evaluation uses 15 trials/task on three tasks. Early fine-tuning checkpoints supply mixed outcomes on MetaWorld/LIBERO. Reported π0.5 means: MetaWorld .57→.82; LIBERO .43→.80. | Direct outcome-supervised activation steering is established. Checkpoint selection, label source, intervention family, and tuning budget differ from this forecast-correction study. Its gains do not imply our checkpoint should improve by the same amount. |
| [Interpreting Physics in Video World Models, Joseph et al.; v1, February 4, 2026](https://arxiv.org/html/2602.07050v1) | Video-encoder probes, subspace analysis and local-attention ablations. The direction intervention uses 240 videos to fit steering probes and a separate 103-video set to fit an evaluation probe and assess edited activations. Twenty probes reduce target-angle MAE from 82.9° to 11.9° at V-JEPA 2-L layer 8. | This supports distributed control of a decoded direction representation. The endpoints are probe readouts and video physical-plausibility judgments, not robot task completion. The evaluation probe is trained on the separate evaluation set; it is not an untouched final three-way split. Encoder depth cannot be mapped directly onto our six-block predictor. |
| [JEPA-WM, Terver et al.; v4, September 2, 2026](https://arxiv.org/html/2512.24497v4) | Studies representation, conditioning, training and planning choices. CEM optimizes actions through latent forecasts using L2 goal costs; Adam can outperform CEM on MetaWorld. Final models use three training seeds and typically 96 episodes/checkpoint; DROID uses 64 recorded-plan action endpoints. v4 adds funding acknowledgements. | Planner choice and representation quality are distinct experimental factors. Our one-checkpoint intervention study does not reproduce the paper's training-seed/late-checkpoint aggregate. DROID recorded-plan evidence is not physical-robot success. |
| [Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior, Wurgaft et al.; v1, May 6, 2026](https://arxiv.org/html/2605.05115v1) | Fits activation and output manifolds; compares curved and straight interpolation. Language evaluations use 50 waypoints and 16 prompts per path, with up to 50 endpoint pairs. The vision experiment is a custom CNN–GRU MountainCar next-frame model; 100 rollouts supply position-labelled activation splines. Its quantitative position-distribution map is a constructed softmax of distances to latent bin centers. | Curvature can matter for interpolation. The vision evidence includes decoded images and constructed readout geometry, without a closed-loop control-success test. Our sparse anchors and low-rank edits do not establish an analogous dense manifold. This is distinct from [Huang et al.'s overthinking method](https://arxiv.org/abs/2505.22411) and [Oozeer et al.'s Riemannian method](https://arxiv.org/abs/2605.24942). |
| [Emergent Linear Representations in World Models of Self-Supervised Sequence Models, Nanda et al.; v2, September 7, 2023; published BlackboxNLP 2023](https://arxiv.org/html/2309.00941v2) | Player-relative `Mine/Yours/Empty` labels expose linear board state. Tests use 1,000 held-out games for probing and 1,000 board-edit cases. Linear vector additions reduce mean legal-move errors to .10 for flips and .02 for erasures. | A suitable reference frame can reveal simple structure hidden by another labelling scheme. The behavioral check uses altered-board legal moves. It does not imply every world model contains discoverable physical axes; multi-layer intervention was also relevant in this small model. |
| [AdaJEPA, Wang et al.; v1, June 30, 2026](https://arxiv.org/html/2606.32026v1) | Execute an action chunk, observe its transition, update the model, then replan. Default: one gradient step on last encoder/predictor layers and a five-transition recent buffer, reset per episode. PushT/PushObj and PointMaze tests span shape, rendering, dynamics and layout changes; three test-data seeds × 50 episodes. On unseen maze layouts, default adaptation raises CEM success 49.3→55.3%. | The closest adaptation comparator, with a different information and parameter-update budget. Its three test-data seeds are not three independently trained models. Some default-dynamics CEM estimates fall slightly (.7 points), so avoid universal no-harm claims. |
| [LeWorldModel, Maes et al.; v3, June 3, 2026](https://arxiv.org/html/2603.19312v3) | About 15M parameters, jointly learned encoder/predictor, prediction loss plus SIGReg; weights frozen during CEM. Planning uses 300 candidates, 30 elites, H5; 30 iterations for PushT and 10 otherwise. Reported speedup up to 48× versus foundation-feature models; timing figure averages 50 runs. | A plausible cheaper future testbed, with different compression and geometry. Its 50-step control budget and goals 25 recorded steps ahead differ from this study. Control sample/seed counts and receiving-hardware speedups need verification before planning a reproduction. [Official code](https://github.com/Mengarr/lewm). |
| [TTT layers, Sun et al.; v4, August 31, 2025](https://arxiv.org/html/2407.04620v4) | A sequence layer's hidden state is a model whose weights update using self-supervision; outer training learns how to use this state. Language-model experiments span 125M–1.3B parameters and measure perplexity and runtime. | This is an architectural memory mechanism, not just fitting the present steering coefficients online. Its results do not demonstrate robotics adaptation. [Fast weights, Ba et al.; v3, December 5, 2016](https://arxiv.org/abs/1610.06258v3) provide earlier temporary-memory machinery. |
| [Memory Maze, Pašukonis et al.; v1, October 24, 2022](https://arxiv.org/abs/2210.13383v1) | Partially observed 3D navigation separates long-term memory demands from other challenges. Online reward and offline state probes are different endpoints. Official offline datasets contain 30,000 trajectories each for 9×9 and 15×15 mazes, split 29,000/1,000. [Code/data](https://github.com/jurgisp/memory-maze). | A useful future memory benchmark if occlusion/history is the scientific variable. The current PointMaze result is not Memory Maze evidence. |

These comparisons establish different mechanisms and endpoints, not a benchmark
leaderboard. A directly comparable reproduction would also pin code, data, model,
precision, split, tuning and runtime budgets.

## 4. Implementation recommendation

First finish a fixed-candidate component replay using the existing frozen bank and
operator. A single engineering bank can check fidelity; a small, prospectively
selected development sample can then identify which component changes CEM scores.
Use the unchanged full operator and its already calibrated random-subspace control.
No new fitting or rank/layer sweep is needed for this question.

If this identifies a reproducible score mechanism, test physical candidate ranking
with simulator forks and a small fixed set of candidates. Only then choose between
a geometric successor and a limited online-adaptation pilot. For adaptation, the
smallest fair baseline is a causal recent-transition residual correction alongside
the frozen model. A fast-weight architecture or Memory Maze migration is a larger
project and should follow evidence that history contains useful missing information.

## 5. Prioritized falsifiable hypotheses

| Priority and hypothesis | Experiment and primary endpoint | Falsifier and main alternative |
|---|---|---|
| 1. The candidate-shared component reproduces most of the full edit's score effect. | On fixed action banks from independent development trajectories, replay full, shared-only and centered-only edits. Measure reconstruction of candidate-centered cost changes. | Poor reconstruction by shared-only weakens this account. A small centered component or nonlinear component interaction may dominate ranking despite low energy. |
| 2. Joint visual/action editing changes predictor outputs nonadditively. | Native/V/A/VA factorial, fixed inputs and unchanged individual doses; measure the output interaction vector before squaring losses. | Interaction at numerical noise with nonzero MSE interaction indicates quadratic-loss cross terms, not a nonlinear pathway interaction. |
| 3. The depth gradient reflects downstream propagation as well as fitted representations. | Measure response magnitude per delivered dose and, in a registered follow-up, restore a downstream activation to its native value. | If normalized response and patch-back do not explain the gradient, fit quality or genuinely different useful information remain candidates. Existing rank-one results cannot establish the later rank-four effect at another block. |
| 4. The current labels miss a simpler task-relative coordinate. | Compare fixed-complexity probes for world displacement versus goal-relative displacement; validate on held-out trajectories and intervene using the fitted coordinate. | Similar readouts, failed held-out prediction, or absent intended downstream effect weaken this coordinate hypothesis. Better decoding alone is insufficient. |

For the first two hypotheses, all windows/candidates from a source trajectory form
one experimental unit. The target population is the prespecified development
distribution for the same task/checkpoint, not all robots or world models.

## 6. Protocol, controls and exact distinctions

Freeze action arrays, goals, checkpoint and fit hashes, model precision, candidate
ordering and the intervention registry before model execution. Select trajectories
without conditioning on rescue/regression outcomes. Qualify a native repeat,
zero edit and cached-full replay first. Cache the full edit's coefficients on the
original inputs; component replays must not silently recompute a different operator.
Keep the natural component doses for the additive decomposition and report them.
Energy-matched component tests answer a separate question.

The shared-coefficient result does not imply a shared cost offset. Even a constant
final-latent shift `d` changes squared goal distance by

`C_i' − C_i = 2 d·(z_i−g) + ||d||²`.

Its effect on the difference between candidates `i` and `j` is `2 d·(z_i−z_j)`.
Thus it can rerank candidates before considering the predictor's nonlinearity.
The measured 98.855%/99.615% shared coefficient energies on MetaWorld are evidence
about applied coefficients, not percentages of decision-relevant information.
[Coefficient analysis](MECHANISMS.md#candidate-specific-versus-common-correction).

For the pathway factorial, define `e=z0−y`, `v=zV−z0`, `a=zA−z0`, and
`r=zVA−zV−zA+z0`. With `L(z)=||z−y||²`, the exact loss interaction is

`L(VA)−L(V)−L(A)+L(0) = 2 v·a + 2(e+v+a)·r + ||r||²`.

Apply the same dimensional averaging/weights as the reported loss. If `r=0`,
`2 v·a` can still be nonzero. Also retain the separately defined equal-budget
joint arm: rescaling both doses changes the factorial estimand. A factorial on
whole-episode binary success measures interaction of entire control processes;
it cannot isolate within-forward-pass pathway interaction once states diverge.

Three quantities sometimes called entropy answer different questions:

| Quantity | Definition | Meaning and limitation |
|---|---|---|
| CEM proposal entropy | For the pre-clipping Gaussian, `H=.5 log det(2πeΣ)`; diagonal case sums log variances. | Breadth of candidate actions. Save covariance and action units; clipping/bounds change the sampled distribution. Proposal contraction can be normal optimization. |
| Attention entropy | Per query/head, `−Σ p_key log p_key`, optionally divided by `log(number of valid keys)`. | How broadly a query attends. Match masks/context lengths; lower entropy is not automatically useful attention or greater physical certainty. |
| Coefficient spectrum entropy | `p_k=λ_k/Σλ`; `H=−Σp_k log p_k`, effective rank `exp(H)`. | Concentration in a specified coefficient second moment/covariance. State centering explicitly. It differs from participation ratio `(Σλ)²/Σλ²`; neither measures planner exploration. |

## 7. Metrics and prospective decision gates

Let `P` subtract the mean across candidates. The primary component metric is

`R_common = 1 − ||PΔC_full − PΔC_common||² / ||PΔC_full||²`.

Report absolute RMS cost changes too; mark the ratio undefined below a numerical
floor fixed using native-repeat checks. Negative values are possible. Secondary
metrics are elite-set overlap, rank correlation, selection-boundary margins and
numeric action differences. Hash differences alone give no effect magnitude.

A proposed mechanism gate is mean `R_common ≥ .80`, mean top-10 overlap at least
.90, and reproducibility against native/cached-full parity checks. Report uncertainty
on these means; crossing a point-estimate gate is not proof. For physical usefulness,
require better prespecified simulator-outcome ranking or lower selected-action
regret, with an interval excluding zero against both native and matched random.
The direction of a cost change alone cannot pass that gate.

The pathway endpoint is normalized `||r||²` plus the exact loss-term decomposition.
A practical threshold should exceed the native-repeat error by a fixed factor and
be specified before measurement. Attention plots are secondary localization data
unless a targeted ablation and a downstream endpoint test their causal role.

## 8. Statistical analysis and sample size

Retain the completed study's frozen analysis. New pilot analyses are exploratory
and use paired trajectory aggregates, with a cluster bootstrap over trajectories.
Use a fixed 95% interval procedure and Holm correction if both task-specific primary
claims are tested. Multiple CEM iterations, 300 candidates and many heads do not
increase the independent sample size.

A feasible mechanism pilot is one instrumentation bank followed by 32 independent
development trajectories per MetaWorld task, one prespecified bank each. This is a
precision/runtime pilot, not a claim of 80% power. Estimate the variance of its
trajectory-level metric for a separately frozen confirmatory sample; do not expand
until a desired significance result appears.

For scale, Reach's observed discordance is `q=42/96=.4375`. A rough paired-binary
normal approximation gives `SE≈sqrt(q/n)=.068`. At two-sided .05 and 80% power,
the detectable difference with 96 pairs is about 19 percentage points. Detecting a
five-point difference at the same discordance would require roughly 1,400 paired
scenarios per task before multiplicity adjustment. This illustrative calculation
assumes the discordance persists; use prospective paired-binomial simulation for
an actual success-rate study. Absence of significance at n=96 is not equivalence.

## 9. Confounders and unresolved evidence

The old and fresh populations differ, and so do their planner RNG protocols.
Their score differences cannot be attributed solely to overfitting, new scenarios
or hardware. Current behavioral traces omit fresh candidate costs, elite IDs and
numeric action arrays. A new replay can test a mechanism on its own fixed inputs;
it cannot recover unlogged original rankings from hashes. [Results](RESULTS.md).

The early-to-late rank-one effect includes layer-specific fitting, delivered dose,
representation scale and downstream computation. Low-rank edits spread across
patches do not prove a dense nonlinear manifold. A physical-coordinate claim needs
independent labels, generalization and a predicted causal consequence. A failed
rank/layer recipe does not establish a general failure of mechanistic interpretability.

Outcome-labelled fitting may favor a policy success direction, whereas recorded-
forecast fitting may reduce average errors irrelevant to selection. This is a
testable objective-mismatch hypothesis, not a demonstrated explanation for the
COAST comparison. Online adaptation additionally observes new transitions; compare
it with controls receiving the same observations and latency budget, and reset
state between episodes. Keep candidate-specific imagined histories separate from
shared real observed history to avoid information leakage between candidates.

## 10. Is this the right research question?

Yes, if the question is specific: **what computation connects the latent edit to
candidate scoring and changed physical behavior?** The measured coefficient
decomposition and paired rescues/regressions make that question concrete. The next
result should distinguish shared-shift mediation, small differential effects and
nonlinear interactions. A larger sweep would be less informative until those
alternatives are separated.

Othello suggests testing alternative reference frames; manifold steering suggests
testing path geometry; adaptive world models suggest testing new information from
observed transitions. Those are useful research directions. None establishes that
this checkpoint already exposes the corresponding mechanism. Present the work as
a case study with a measured internal property, a measured behavioral consequence,
and an explicitly unresolved causal link.

Primary sources: [COAST](https://arxiv.org/abs/2605.17144v1),
[Physics interpretation](https://arxiv.org/abs/2602.07050v1),
[JEPA-WM](https://arxiv.org/abs/2512.24497v4),
[Manifold Steering](https://arxiv.org/abs/2605.05115v1),
[Othello](https://arxiv.org/abs/2309.00941v2),
[AdaJEPA](https://arxiv.org/abs/2606.32026v1),
[LeWM](https://arxiv.org/abs/2603.19312v3),
[TTT](https://arxiv.org/abs/2407.04620v4),
[fast weights](https://arxiv.org/abs/1610.06258v3),
[Memory Maze](https://arxiv.org/abs/2210.13383v1).
