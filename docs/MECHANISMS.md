# Activation steering in a frozen world model

Research and analysis date: September 13, 2026. New analyses are exploratory;
the original offline experiments and protected behavioral analysis remain unchanged.

## 1 Executive finding and recommendation

**The refined edit is dominated by a component shared across candidate plans,
while the paired behavioral records show substantial changes in which scenarios
succeed.** This is a measured property of the applied steering computation, not
yet the reason it fails to produce a reliable net success gain.

Two additional CPU analyses are complete: a recount of the archived layer sweeps
and a nested decomposition of applied coefficients from 768 fresh arm records.
They add a spatially and computationally specific account to the success table.
Recommendation: **run a limited shared-candidate replay pilot**, not another broad
success-rate campaign. Confidence is high in the descriptive arithmetic and lower
in any proposed downstream causal explanation.

## 2 The question

Where is a frozen JEPA-WM predictor susceptible to useful latent correction, and
does its steering computation distinguish alternative actions? These are different
from asking whether a physical concept is decodable, whether attention is local,
or whether a planner succeeds more often.

The layer sweeps used the earlier rank-one operator, independently fitted at each
registered support. The coefficient analysis examines the later fixed-response
rank-four operator. They are complementary experiments, not a longitudinal causal
chain through one unchanged intervention.

## 3 Literature and interpretation

| Primary work and checked version | Relevant evidence | Implication for this study |
|---|---|---|
| [Interpreting Physics in Video World Models, v1, February 4 2026](https://arxiv.org/html/2602.07050v1) | Reports attention locality and targeted local-attention suppression alongside physical-variable probes. | A head-distance heatmap is descriptive; causal localization requires an intervention and a measured endpoint. Its V-JEPA video representation is not our six-block action-conditioned predictor. |
| [Towards Best Practices of Activation Patching, v2, January 17 2024](https://arxiv.org/abs/2309.16042v2) | Shows that metric and corruption choices can change interpretability conclusions. | Fix the action bank, endpoint, dose and comparison before examining a new diagnostic; avoid selecting the most attractive layer afterward. |
| [How to use and interpret activation patching, v1, April 23 2024](https://arxiv.org/html/2404.15255v1) | Separates exploratory localization from confirmation and discusses path patching and necessity versus sufficiency. | Our layer sweep identifies intervention susceptibility, not a complete circuit. A downstream patch-back can test mediation more directly than another correlation. |
| [How Should World Models Be Evaluated for Embodied Decision-Making, v2, June 28 2026](https://arxiv.org/abs/2606.15032v2) | A position paper advocating decision-relevant evaluation, including action fidelity and ranking. | Motivation for measuring candidate ranking separately from forecast loss; not independent empirical proof of our proposed mechanism. |

No novelty priority or universal physics-emergence claim follows from this search.
The useful contribution is a reproducible, task-specific intervention case study.

## 4 Completed analysis and best next step

### Layer response map

Recounted **576 task/precision/support/modality/horizon cells** directly from six
hash-verified reports containing per-lineage metrics. All recounted means match
the existing published CSV export. Each support retains its matched random arm;
zero-dose checks reproduce native metrics. The independent units remain 33 Reach,
27 Reach-Wall and 21 Push-T lineages, not 576 independent experiments.

![BF16 layer effects across forecast horizons](figures/layer_mechanism_bfloat16_native.png)

For the singleton sweep, H6 proprioceptive-error reductions decline monotonically
from B0 to B5 on both MetaWorld tasks in **both** precisions:

| Task | BF16 B0 / B5 reduction | FP32 B0 / B5 reduction |
|---|---:|---:|
| Reach | 3.026% / 0.475% | 3.050% / 0.494% |
| Reach-Wall | 1.526% / 0.298% | 1.334% / 0.228% |

The original BF16 simultaneous intervals for B0 minus B5 are [1.769, 3.333]
and [0.243, 2.214] percentage points of native MSE, respectively. B0 also beats
its matched random subspace in those registered contrasts. This does **not** isolate
B0 from every other block: Reach B0 versus B1 remains unresolved. It does not
establish an intrinsically privileged semantic layer, and it is not a test of the
later rank-four operator at B0.

The pattern is task-dependent: Push-T does not reproduce the MetaWorld H6 benefit.
In its BF16 B0 arm, proprioceptive error worsens by 2.25% at H3 but is nearly native
at H6. This horizon profile is descriptive; it does not identify contact physics.
An unchanged early endpoint is expected because editing starts at H3.

[Matched-random heatmap](figures/layer_mechanism_bfloat16_random.png) ·
[FP32 sensitivity](figures/layer_mechanism_float32_native.png) ·
[FP32 matched-random comparison](figures/layer_mechanism_float32_random.png) ·
[All cells](../paper/data/layer_mechanism_grid.csv).

### Candidate-specific versus common correction

For scenario s, call j and candidate i, let c(s,j,i) be the logged four-component
coefficient vector after dose scaling. Decompose it into the mean over candidates
in that call and its candidate-centered residual. Further split call means into
the scenario mean and the between-call residual. The squared energies add exactly:

`total = within-call candidate-centered + between-call mean variation + scenario mean`.

The analysis verifies this identity numerically, checks coefficient norms against
the logged requested edit norms, and checks every source arm hash against its
completed scenario report and the immutable final report. The fixed orthonormal
basis makes coefficient geometry informative about requested activation edits;
it is not output forecast geometry or CEM cost geometry.

| Task | Refined candidate-centered share | Refined shared share | Random-subspace candidate-centered share |
|---|---:|---:|---:|
| Reach | 1.145% | 98.855% | 0.361% |
| Reach-Wall | 0.385% | 99.615% | 0.774% |
| PointMaze | 15.646% | 84.354% | 12.059% |
| Wall | 8.299% | 91.701% | 5.858% |

Each entry averages 96 scenario-level ratios. A ratio is formed after summing
energies across all eligible calls within that scenario. This avoids treating
correlated candidates as independent observations. The analysis includes 40,320
candidate batches across both arms, each with 300 candidates, but its sample size
is still 96 scenarios per task.

![Candidate coefficient decomposition](figures/candidate_specificity.png)

The common component is **not constant throughout the episode**. For refined Reach,
25.04% of energy is changing call means and 73.81% is the scenario mean. The
corresponding shares are 11.76% and 87.86% on Reach-Wall. The randomized comparators
also have substantial common components, so this pattern is not unique evidence
for learned semantic directions.

In PointMaze, the candidate-centered share falls from 41.28% at the first CEM
iteration to 1.76% at the last iteration of that planning call. The random-subspace
arm falls from 31.08% to 1.89%. This is compatible with candidate convergence, not
proof that the controller itself collapses. The two arms' later candidates are
different action inputs and are not directly cross-scored here.

### Why a shared correction can still change planning

Even an identical final-latent translation d need not preserve goal-distance ranking.
For squared L2 cost, `C_i = ||z_i - g||²`, the shift changes cost by
`2 d · (z_i - g) + ||d||²`. The second term is common but the first depends on
the candidate. Our actual intervention happens inside a nonlinear predictor, making
a constant cost offset an even stronger unsupported assumption.

Therefore “99% common coefficient energy” does **not** mean “99% irrelevant to
planning.” Small differential components can also matter near an elite-selection
boundary. The raw traces do not establish which component drove the 21 Reach
rescues and 21 regressions. A fixed-action-bank replay is the shortest way to test it.

## 5 Falsifiable mechanism hypotheses

| Hypothesis | Current evidence | Discriminating test or falsifier |
|---|---|---|
| The refined computation is dominated by a common candidate shift. | Supported for MetaWorld coefficient energy; also present in the randomized arms. | Decomposition would be weakened by large candidate-centered energy. It says nothing alone about score changes. |
| Earlier intervention sites have more forecast-correction leverage. | Registered rank-one B0–B5 effects support an early-to-late gradient on MetaWorld. | A layer-matched response/Jacobian and output patch-back test could separate downstream amplification from fit quality and representation scale. |
| Common-mode versus candidate-centered components affect CEM differently. | Not measured: fresh candidate costs and actual actions were not logged. | Replay native, full, common-only and centered-only edits on identical candidate actions; measure centered cost changes and elite overlap. |
| The mechanism improves physical action ranking. | Not established by forecast errors, coefficient geometry or changed action hashes. | Evaluate candidate physical outcomes under matched simulator forks; rank predictions against those outcomes. |

## 6 Fixed protocol for a subsequent causal pilot

This is a **proposed** experiment, not a launched or completed comparison. Freeze
input IDs, goals, checkpoint, bank and source hashes before new model execution.
Use the existing fitting-row pilot to qualify instrumentation first. Never advertise
one fitting trajectory as population-level mechanistic confirmation.

Then use an independently frozen set of development trajectories, not selected
rescues or a new success-rate winner. On each native-generated fixed candidate bank:

1. Score native and the existing full refined edit with exact action-array identity.
2. Cache the full edit at B3/H3; replay its within-bank mean alone and its centered
   residual alone. These are additive component interventions, not fresh fits.
3. Include a full cached-edit replay to check exact equivalence, a zero-edit replay,
   and corresponding decompositions of the calibrated random-subspace arm.
4. Preserve natural component energy for decomposition; report it. Separately frozen
   dose-matched versions can address magnitude versus structure, but must not be
   silently conflated with the additive decomposition.
5. Save actual candidate actions, scores, elite IDs, margins, final forecasts and
   instrumentation parity. Separate shared-bank scoring from adaptive CEM searches.

An optional patch-back at a later block can test which downstream computation
mediates the measured forecast/score effect. Attention maps can guide a future
registered test; they are not grounds for retrospectively naming a causal head.

## 7 Metrics and decision gates

Primary proposed endpoint: fraction of the full edit's **candidate-centered cost
change** reproduced by the common-only replay, evaluated per trajectory. Also
report the absolute norm, since a ratio is unstable if the full effect is nearly zero.
Secondary: rank agreement, top-10 elite overlap, selection-boundary margins and
endpoint-specific forecast changes. No physical-quality claim without physical targets.

For a bounded diagnostic, a prospective operational gate could be: common-only
replay explains at least 80% of centered score-change energy with at least 90%
top-10 agreement with the full edit across the sampled inputs. These are proposed
practical thresholds, not observed findings or established scientific constants.
Failure would weaken common-component mediation and favor differential or nonlinear
interaction explanations. Do not tune thresholds after seeing pilot scores.

## 8 Statistical scope and sample size

The completed coefficient analysis uses 20,000 bootstrap draws over scenarios,
seed 20260913. Its 95% intervals are marginal descriptive intervals, not a corrected
family of discoveries. Do not infer a significant learned-versus-random difference
from separate intervals. No scenario was selected by outcome.

The layer plots retain the original lineage averaging and H6 simultaneous-contrast
family; no new significance labels are assigned to H1–H5. BF16 remains primary
and FP32 sensitivity remains visible. A prospective causal replay should group
windows and candidate banks by source trajectory and fix its contrast family before
execution. The one-trajectory engineering pilot has no inferential power claim;
its measured variance and runtime would inform a separately approved sample size.

## 9 Limitations and unresolved evidence

- These analyses use one released checkpoint per task; Reach/Reach-Wall share one.
- Layer-specific fitted operators vary with their support. Depth effects include
  fit quality, activation scale and downstream computation, not only layer identity.
- BF16 can change delivered edit energy; the FP32 pattern is a sensitivity check,
  not a reconstruction of the authors' independent-training-seed uncertainty.
- A coefficient decomposition is not a causal mediation analysis. It cannot recover
  fresh candidate rankings, counterfactual physical quality, or head-level attention.
- The protected success result and its uncertainty remain in [Results](RESULTS.md).
  A stronger mechanistic framing does not turn that result into an efficacy claim.
- No new GPU execution or head-by-head attention heatmap has been completed. The
  native observer is implemented and CPU-tested, awaiting a new bounded GPU allowance.

## 10 Assessment and reproducibility

The project is better framed as an intervention study of a world-model/planner
interface than as a new best-performing robot controller. Its strongest current
assets are controlled layer interventions, a directly audited steering computation,
and paired behavior that aggregate scores hide. More plots alone would not establish
a circuit; the next useful experiment is the fixed-candidate component replay.

The new computation is specified in [the fixed follow-up scope](../configs/mechanism_followup_20260913.json)
and implemented in [steering_specificity.py](../analysis/mechanism/steering_specificity.py).
The [JSON receipt](../paper/data/candidate_specificity.json) includes every raw arm hash,
source and protocol hashes, counts and uncertainty estimates. The [scenario table](../paper/data/candidate_specificity_scenarios.csv)
and [layer grid](../paper/data/layer_mechanism_grid.csv) support CPU-only figure regeneration:

```bash
python analysis/mechanism/steering_specificity.py
python -m unittest discover -s tests -p test_steering_specificity.py -v
```

With private raw records and archived offline reports restored, add
`--recount-layers --results /path/to/results-v2` to repeat the source-bound recount.
This never modifies the source records, fits, or frozen behavioral analysis.

Primary sources: [Physics interpretation](https://arxiv.org/abs/2602.07050v1),
[patching metrics](https://arxiv.org/abs/2309.16042v2),
[patching interpretation](https://arxiv.org/abs/2404.15255v1),
[decision-centric evaluation](https://arxiv.org/abs/2606.15032v2).
