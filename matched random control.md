# Matched random controls and reproduced baselines

Research and audit date: September 11, 2026. Repository: `main`.
Scope: the existing six-task released-checkpoint study, not a new training or
confirmation campaign. This report adds interpretation and verification;
it does not replace frozen protocols, change controls, or overwrite results.

## Decision

**Implement the baseline audit and explicit control contract; retain both our
unsteered checkpoint and appropriate matched-random controls.** Keep published
scores separately labelled. Do not substitute the authors' numbers for our
measured reference, discard an inconvenient control, or tune a random seed after
seeing evaluation results.

Confidence is high in that reporting recommendation, but incomplete in the
equivalence of our entire execution stack to the authors' historical evaluation.
The existing raw-record audits found no pairing, score-calculation, or dose error
in their stated scopes. They do not establish that every possible implementation
error has been excluded. DROID's historical baseline discrepancy remains open.

## 1. Three different questions, three different comparisons

| Question | Required comparison | What it cannot establish alone |
|---|---|---|
| Does our intervention improve this checkpoint in our evaluation? | Edited versus **our same-setup unsteered** checkpoint, paired on scenarios and planner RNG streams. | Improvement over other training seeds or the authors' aggregate. |
| Does the identified geometry matter beyond generic perturbation? | Edited versus a **specified active random control**, with the relevant nuisance quantities matched. | That the control must have zero effect, or that a successful edit identifies a unique mechanism. |
| How does our reproduction compare with the published benchmark? | **Published** and **reproduced** reference rows, with checkpoint, population, simulator, planner, metric, and aggregation identified. | A causal steering gain from subtracting numbers collected under different setups. |

There is an additional naming trap: the paper's **DINO-WM comparator** is not the
same model as its improved **JEPA-WM recipe**. Our unsteered row uses the latter's
released checkpoint; it is not intended to reproduce a DINO-WM checkpoint merely
because the Reach percentages happen to be similar.

## 2. Are baseline differences typical, or evidence of a mistake?

They occur in published work, but there is no universal acceptable discrepancy.
A different checkpoint or evaluation population can legitimately change a score;
a changed metric, incorrect goal image, or implementation bug can also change it.
An explanation must be verified rather than chosen because it is reassuring.

Two directly relevant precedents:

- **RLVR-World, Table 4**, explicitly separates DINO-WM's reported Push-T
  prediction scores from evaluation of its public checkpoint: scaled LPIPS
  **0.7 versus 3.39**, and scaled SSIM **98.5 versus 96.38**. These are prediction
  metrics, not planning success. This demonstrates transparent separate reporting,
  not that our particular gap is harmless. [Wu et al., arXiv:2505.13934v2](https://arxiv.org/html/2505.13934v2#S6)
- **JEPA-WM, Appendix C**, identifies a V-JEPA-2-AC rollout-loss bug and reports
  corrected, retrained models rather than treating their public-checkpoint score
  as interchangeable. Its Table 2 labels that distinction. [Terver et al., v4](https://arxiv.org/html/2512.24497v4#A3)

The broader experimental-design guidance is to disclose implementation choices,
sources of randomness, uncertainty, and comparable baselines. It does not say to
accept every reproduction gap or force agreement by changing the measured score.
[Patterson et al., *Empirical Design in Reinforcement Learning*, 2024](https://jmlr.org/papers/v25/23-0183.html),
[Henderson et al., *Deep Reinforcement Learning that Matters*](https://arxiv.org/abs/1709.06560)

### Our actual baseline differences

Simulator columns are success percentages; DROID is an action-agreement score,
not a percentage of successful physical robot executions.

| Reference | Reach | Reach-Wall | Push-T | PointMaze | Wall | DROID score |
|---|---:|---:|---:|---:|---:|---:|
| Authors' improved recipe, main aggregate | 58.2 | 41.6 | 70.2 | 83.9 | 78.8 | 48.2 |
| Authors' final-checkpoint aggregate | 49.0 | 29.2 | 69.4 | 83.3 | 80.9 | 46.5 |
| Our released checkpoint, unsteered | 44.79 | 30.21 | 59.38 | 80.21 | 76.04 | 51.10 |
| Our native minus authors' main aggregate, points | −13.41 | −11.39 | −10.82 | −3.69 | −2.76 | +2.90 |

Published rows: [JEPA-WM Tables 2 and 12](https://arxiv.org/html/2512.24497v4).
Local rows: [six-task inventory](reports/SIX_TASK_TABLE_PROGRESS_20260911.md).
The main simulation aggregate uses three training seeds and a ten-epoch late
window; DROID uses a hundred-epoch window. Those windows are not automatically
ten or a hundred available checkpoint files. Our row is one released checkpoint.
[JEPA-WM Appendix G.2](https://arxiv.org/html/2512.24497v4#A7.SS2)

Thus our native score is **not above the main paper on five of six tasks**.
The two larger downward simulator gaps also merit reconciliation; auditing only
the flattering DROID difference would be selective.

For DROID specifically, equal-budget coupling scores **51.07**, against our
unsteered **51.10**. Comparing 51.07 with the paper's 48.2 would misattribute a
baseline offset to the intervention. The existing audit reconstructs 51.10 from
all 64 saved action records using the upstream metric and reproduces their
recording/frame sampling. It does **not** resolve full upstream-runner equivalence,
original encoder-file equivalence, or the authors' historical evaluation inputs.
The paper/release 16-versus-15 recording discrepancy is also documented, not
silently repaired by inventing a sixteenth usable file.
[DROID reconciliation audit](reports/DROID_BASELINE_RECONCILIATION_AUDIT.md)

## 3. What does “matched random” mean?

It is an **active control**, not a no-op or a guarantee of baseline-level success.
One randomizes the hypothesized important structure while retaining other aspects
of the treatment. Different operators require different matches; arXiv is a
publication repository, not a standards body defining one mandatory algorithm.

For an additive edit, a common comparison is

`h_edited = h + a v`, versus `h_random = h + a r`,

where `v` and `r` are unit directions applied at the same sites/times. Matching
their activation norm does **not** match their effect on forecasts or selected
actions: different directions can encounter different model sensitivities.
Likewise, random perturbation can help some states or harm others. The meaning
of the comparison is specificity, not a presumption that random must fail.

### What the papers actually use

| Primary source | Actual control/evaluation choice | Implication and limitation for us |
|---|---|---|
| [ITI, Li et al., 2306.03341v4, Table 3](https://arxiv.org/html/2306.03341v4) | Compares no intervention, random direction, and learned directions; tunes intervention strength separately in its cross-validation procedure. | Random-direction comparisons have direct steering precedent. Its separately tuned strengths are not our fixed-dose contract and cannot be adopted retrospectively. |
| [COAST, Miao et al., 2605.17144v1, Appendix A.3](https://arxiv.org/html/2605.17144v1#A1.SS3) | Randomizes conceptor eigenvectors while preserving eigenvalues; also compares additive mean-difference steering. Table 10 reports best sweep cells with 15 rollouts; the main held-out test protocol uses 30. | For a multiplicative filter, spectral matching is important. Our additive and calibrated-response operators are not conceptors, so this is a design principle, not a drop-in implementation. |
| [Häon et al., 2509.00328v1, Appendix C.3](https://arxiv.org/html/2509.00328v1#A3.SS3) | Replaces six semantically selected FFN value vectors with six random ones, fixed during evaluation, at the same coefficient of 10. | A closer robotics control matching cardinality and strength, not necessarily delivered residual norm. Ten rollouts per variant; the speed comparison executes only native actions while measuring alternative predictions, not independent closed-loop steering success. |
| [Sahoo et al., 2606.02907v2, §3.6](https://arxiv.org/html/2606.02907v2#S3.SS6) | Samples 20 normalized Gaussian directions; applies the same strength and layer, then compares targeted effects with that reference distribution. | Supports testing more than one random realization when claiming direction-specificity. An LLM result does not establish the expected effect size in robotics. |
| [Zhang and Nanda, 2309.16042v2](https://arxiv.org/html/2309.16042v2) | Shows that corruption and outcome-metric choices materially alter activation-patching conclusions. | Specify exactly what the control destroys and retains. Do not treat a different corruption scheme as interchangeable. |
| [Makelov, Lange and Nanda, 2311.17030v2](https://arxiv.org/abs/2311.17030v2) | Demonstrates that effective subspace patching can exploit an alternative pathway without faithfully locating the hypothesized feature. | Behavioral influence alone is insufficient for a mechanistic localization claim. This is a caution, not a diagnosis of our model. |
| [LEAP, Pham and Bera, 2609.03294v1](https://arxiv.org/html/2609.03294v1) | Controlled native-planner comparison and component ablations; see the full-PDF follow-up below. | Not an activation-random-control recipe. |
| [Agarwal et al., 2108.13264v4](https://arxiv.org/abs/2108.13264v4) | Demonstrates unreliable conclusions from few-run point estimates and advocates uncertainty-aware comparisons. | Report effect intervals and actual units of replication, not only the highest cell. Do not average DROID action scores and simulator successes as one metric. |

COAST's random control is itself much higher than native on some individual
MetaWorld entries—for example, its hammer sweep cell is 0.80 versus 0.33—despite
its aggregate random result not improving. That is evidence that “random higher”
is possible, **not** validation of our Reach implementation or a benchmark for
our effect. We must distinguish swept cells from held-out estimates.
[COAST Table 10](https://arxiv.org/html/2605.17144v1#A1.SS3)

The four most actionable examples are **COAST**, **Häon et al.'s VLA study**,
**ITI**, and **Sahoo et al.** They cover spectral, neuron-selection,
direction-choice, and repeated norm-matched-direction controls respectively.
WA-LQR is relevant to feedback steering, but its random-perturbation tests assess
local linearity; that is not a substitute for a randomized behavioral control.
[Hong et al., 2607.14943v1](https://arxiv.org/html/2607.14943v1)

Importing another paper's controller, success timing, selection procedure, or
sample count would create a new experiment; it would not repair this frozen
JEPA-WM comparison. Hold non-target components fixed and test actual behavior.

## 4. Our controls: exact definitions and gaps

### Reporting names: two separate control rows

The user-facing tables distinguish these existing controls explicitly. They
must not be pooled into a single "random" score or treated as interchangeable.

| Display name | Learned intervention it controls | Preserved source arm ID |
|---|---|---|
| Refined control: response-calibrated random-subspace edit | Refined four-direction fixed-response edit | `matched_random_fixed_rank4` |
| Coupling control: dose-matched random-direction edit | Equal-budget vision–action coupling | MetaWorld: `matched_random_coupling`; other displayed task panels: `matched_random_equal_standardized_energy` |

This September 11 clarification changes presentation only. Scores, fitting
artifacts, random seeds, arm IDs, missing-cell status and the registered analyses
remain unchanged. In particular, "response-calibrated" is essential: the random
subspace has its own fitted coefficient map, not the learned map with only its
orientation changed. "Dose-matched" names what the coupling control matches;
it does not imply identical downstream sensitivity or action-selection effects.

| Control | Randomized | Matched / retained | Not matched or not established |
|---|---|---|---|
| Coupling random | Visual and action-conditioning directions, drawn once per task using saved seeds and orthogonalized against their learned directions. | Unit direction norms, sites, horizons, per-site doses, paired scenarios, checkpoint, planner budgets. Equal-budget version uses the same component rescaling as its learned counterpart. | Downstream output sensitivity, action changes, usefulness, or robustness across different random draws. Orthogonalized Gaussian directions are not unconditional random directions over the full space. |
| Refined fixed-response random | The four-dimensional output basis. | Same support, rank, orthonormality, activation dose rule, fitting/calibration procedure, learned error readout, paired behavioral setup. | Coefficient-map singular values are not equal. This is a **calibrated random-subspace controller**, not pure untrained noise and not COAST's spectrum-matched filter. |
| Spatial permutation | Fixed ordering of 256 visual patches; channel patterns are retained. | Visual direction L2 magnitude and registered edit scale/site/time. | Semantic spatial arrangement; not equivalent to independently randomizing channels or redrawing noise at each step. |
| Zero dose | No signal is injected. | Base weights and intended execution semantics. | A random-effect control: its role is an engineering identity check. |

Definitions are in [operator_fit.py](src/offline_study/operator_fit.py),
[fixed_response.py](src/offline_study/fixed_response.py), and the frozen banks.
The cheap refined implementation uses a precomputed thin response map, not the
old per-candidate online solver. Its random arm retains learning, so the two
arms test **learned versus random basis selection under calibration**, not
learning versus no learning.

### Does this show intervention location matters most?

No. Both learned/control pairs hold location fixed, so these comparisons cannot
identify the best layer or patch support, or rank location above direction in
importance. Location, direction/subspace, dose and timing are distinct choices;
the refined comparison additionally retains calibration while fitting a separate
map for each basis. Similar observed performance is not an equivalence test,
and one fixed random realization per task does not establish that arbitrary
directions work equally well.

As an existing literature reference, [ITI, Table 3 and section 5.5](https://arxiv.org/html/2306.03341v4)
reports differences across direction choices and across intervention-position
selection methods, with strengths tuned separately. It supports considering both
choices; it does not establish which dominates in JEPA-WM. Our layer/spatial
ablations remain separate evidence and must not be relabeled as completed
behavioral location comparisons. No new experiment is introduced by this note.

The equal-budget coupling rule scales both standardized component doses by
`1/sqrt(2)`. This matches the summed squared **standardized** dose to a single
component's budget. Visual and action coordinates have different fitted scales;
it does not mean identical raw norms across modalities. The unscaled joint arm
intentionally has both full component doses and must not be described as the
same total standardized budget as a single-component arm.

### Do we need them?

- **For a practical improvement claim:** the same-setup unsteered checkpoint is
  indispensable. A random control is not a universal requirement for every
  robotics paper, but should remain in this study's already frozen contrasts.
- **For our geometry-specific claim:** a credible active control is important.
  Otherwise a generic perturbation, calibration benefit, or altered optimization
  landscape could explain the apparent gain.
- **For reproducing the authors' reference:** random controls answer a different
  question. They cannot diagnose the historical baseline gap by themselves.

The best next control is claim-dependent. Preserve the existing calibrated-random
comparison. If a future claim concerns only basis orientation, add a separately
frozen orientation-randomized operator retaining the learned map/spectrum and
readout where mathematically appropriate; distinguish it from refitting a map
in each random basis. Those are different null hypotheses. Neither should be
substituted after inspecting the current outcomes.

### A more precise COAST-inspired path for our refined edit

COAST's spectral control schematically replaces `C = U diag(lambda) U^T` with
`C_random = Q diag(lambda) Q^T`, using random orthonormal `Q`: orientation changes,
eigenvalues do not. [COAST Appendix A.3](https://arxiv.org/html/2605.17144v1#A1.SS3)

**Our proposed adaptation, not COAST's implementation:** write the cheap edit
before dose normalization as `delta(h) = B M f(h)`, where `B` has four orthonormal
columns, `M` is the fitted coefficient map, and `f(h)` is the unchanged native
feature readout including its constant term. Replace only `B` with independent
orthonormal `Q`, retaining **the learned M and f**. Then
`||Q M f(h)|| = ||B M f(h)||` on the same input, so the same dose/zero rule matches
per-input edit size. The linear part preserves singular values under this
left-isometric change. This is not generally eigenvalue preservation of a square
symmetric conceptor and must not be described that way.

| Control | Fixed | Changed | Question |
|---|---|---|---|
| Existing calibrated-random subspace | Readout, rank, support, dose, fitting procedure | Basis and its separately calibrated map | Does a learned basis outperform a random basis after equal calibration? |
| Proposed orientation-only random subspace | Readout, learned map, support, rank, dose rule | Output basis only | Does learned output orientation matter with identical controller coefficients? |

Generate a thin `d x 4` Gaussian matrix and orthonormalize it; no `d x d` matrix
or online response probes are needed. Freeze the distribution (full ambient
support versus constrained orthogonal complement), seed list, sites/times,
fitting/evaluation separation and analysis before outcomes. Test norm identity
on synthetic and fit-only inputs, zero-dose identity, hook cleanup, unchanged
model weights and native-feature provenance before paired behavioral evaluation.
Do not choose a control seed based on its observed strength.

**Disposition:** recommend this as a separately named supplemental control if
pursuing the orientation claim, not an automatic replacement or a launched extra
panel. Current frozen cells remain unchanged. For static coupling, same-site
norm-matched random directions are more directly appropriate than a spectral
filter because that intervention is additive.

## 5. What the saved Reach audit establishes

All 960 core episode records, 40 shard reports, scenario pairing, source bindings,
action traces, and recorded planning budgets passed the checks in the preserved
audit. The coupling directions regenerate bitwise from saved seeds; per-site
definitions agree. Learned/random delivered L2 magnitudes agree to better than
one part in a million. Refined dose checks also pass. This covers the original
two-task panel, not an automatically certified six-task study.
[Raw-record and bank audit](reports/MATCHED_RANDOM_CONTROL_AUDIT_20260911.md)

| Reach comparison | Successes out of the same 96 | Paired wins / losses versus native | Interpretation |
|---|---:|---:|---|
| Unsteered | 43 | — | Same-checkpoint reference. |
| Refined learned / calibrated random | 49 / 48 | Random: 25 / 20 | Learned is only one net success above its calibrated control. |
| Equal-budget learned / random coupling | 47 / 58 | Random: 30 / 15 | The unusually strong random result needs robustness checks; it is not a verified general benefit. |

For the post-hoc Reach random-coupling contrast, exact unadjusted paired
`p=0.03570`, but Holm `p=0.14279` over four supplemental control contrasts;
the simultaneous interval is **[−1.04, +32.29] percentage points**. A broader
twelve-comparison sensitivity correction gives `p=0.42837`. These analyses
remain post hoc. They do not change the original frozen eight comparisons,
which did not establish learned improvement against both required references.

## 6. Investigation priorities and falsifiable alternatives

| Hypothesis | Distinguishing check | Status / what would change our conclusion |
|---|---|---|
| A scoring or pairing bug explains the discrepancy. | Independently reconstruct scores, compare initial/goal bytes, seeds, traces, budgets and episode IDs. | Core record checks and DROID metric/sampling checks pass in scope. A new mismatch invalidates affected comparisons until corrected; it is not fixed by relabelling scores. |
| Our wrapper changes the native planner. | Run untouched upstream and wrapper on identical real inputs/checkpoint/runtime; compare preprocessing, candidate costs, selected actions, state transitions and scores. | **Full DROID upstream-versus-wrapper GPU parity remains incomplete.** A mismatch must be localized before a reproduction claim. Repeated wrapper runs alone do not settle it. |
| Aggregation or population explains the paper gap. | Match exact training checkpoints and recorded input manifests, then change one factor at a time. | Aggregation differs; historical inputs are not all available. This is a plausible source, not a measured causal decomposition of the gap. |
| The Reach random basis is unusually favorable. | A prospectively fixed set of independent random bases on paired scenarios. | One current realization cannot estimate between-basis variability. Repeated scenarios under that basis are not new direction draws. |
| Calibration or generic perturbation, not identified geometry, supplies the effect. | Compare existing calibrated-random arm; future orientation-only and calibration ablations separately. | Weakens a geometry-specific explanation if learned directions do not beat those controls; does not automatically imply no useful intervention exists. |
| A simulator/rendering change contaminates restored runs. | Version/source/asset checks, exact saved-input hashes, replay and full receiving-worker native parity. | PointMaze's 97 saved input pairs now reproduce exactly on the isolated legacy runtime. Full receiving-planner clearance is still a separate gate. |

### Audit of the other intervention implementations

Fresh local CPU validation on September 11: **53 tests passed** across static
planning transfer, fixed response, fixed-response behavior, Push-T/DROID coupling,
native navigation replication, intervention hooks, and action geometry.
The static planner tests check frozen-compilation equivalence, zero-dose identity,
realized horizons, and rejection of unsupported dynamic operators. Source review
confirms explicit arm selection and component scales rather than choosing the
best observed arm. These tests do **not** replace receiving-worker GPU checks or
verification of all newly collected raw records.

Current coverage is deliberately stated narrowly:

- Core MetaWorld: completed raw-pair/dose audit; not a multi-training-seed study.
- DROID: completed existing metric/sampling audit; full untouched-upstream parity
  still open. DROID supplies recorded-action evidence, not real robot rollouts.
- Wall/PointMaze: existing candidate records and native pairing were reconciled;
  full navigation factorial analysis still requires remaining PointMaze controls.
- Push-T: continuation uses the archived code/freezes and complete logical RNG
  streams. All696 missing episodes have now been collected and their snapshots
  verified in Drive; final panel coverage/pairing analysis remains separate.
- New refined non-MetaWorld and missing MetaWorld component adapters: preparation
  is not completed behavioral validation. They need their own fit/architecture/
  engineering bindings before launch; no automatic cross-task certification.

## 7. Statistical plan and compute decision

Keep the existing frozen analysis families and report effect sizes with their
specified simultaneous intervals. For independent paired binary scenarios,
discordant win/loss counts support an exact McNemar test. Where scenarios share
a source family, use the registered family-clustered paired analysis rather than
pretending every sampled trajectory is independent. DROID's 64 endpoints contain
15 source recordings; Push-T's 96 native episodes contain 21 released families.
Logical GPU streams are scheduling units, not independent model-training seeds.

Any new random-basis experiment needs a fixed number of draws, a fixed selection
rule, an independent RNG for bank construction, the same dose rule and fitting
budget, and a scenario-by-basis analysis. Do not select the weakest or strongest
random seed, and do not pool all repeated scenario/basis cells as independent.
Multiple bases address direction robustness; they do not supply missing training
seed variability or fresh-family confirmation.

Illustrative planning calculation, **not a new registered sample size**: for
paired binary outcomes, let `delta` be the expected success-rate gain and `q`
the probability the two arms disagree. Then `Var(difference) = q - delta^2`,
and a simple large-sample, two-sided 5%-level, 80%-power approximation is

`n ≈ (1.96 + 0.84)^2 * (q - delta^2) / delta^2`.

At `q=0.47` and a five-point gain, this is about **1,466 independent paired
scenarios** before multiplicity or clustering. This is an order-of-magnitude
illustration using observed discordance, not prospective power certification or
a recommendation to buy those runs. Repeated control seeds add another variance
component. A failed significance test at 96 episodes is not proof of zero effect;
more data is also not guaranteed to vindicate a selected candidate.

For a random-direction reference distribution, the smallest plus-one empirical
tail probability with `m` draws is `1/(m+1)`. Twenty draws have coarse resolution;
they are not a universal adequate sample size, and an empirical rank has a valid
inferential interpretation only under its specified null/exchangeability model.
The 20-draw LLM precedent above is therefore useful guidance, not a robotics
sample-size mandate.

**Recommended spending order:** finish CPU/raw-record reconciliation and the
smallest exact upstream-versus-wrapper parity check first; continue already
authorized frozen missing cells with their gates intact. Do not launch a broad
new random-seed sweep or retraining campaign solely because one control scored
well. Quote and freeze any such extension separately. The practical threshold
already used in this project is a five-percentage-point simulator gain; both
uncertainty and matched-control specificity matter for the claimed contribution.

## 8. Reporting and research claim

The current defensible question is whether the fitted geometry yields a repeatable
behavioral advantage beyond equal-budget perturbation—not whether any cell can
be made larger than a published aggregate. Preserve negative, positive, and
inconclusive effects. Do not claim a general robotics improvement from a DROID
score offset or a single favorable random basis.

Apply the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist) to
claim scope, settings, reproducibility, uncertainty sources, limitations, and
compute disclosure. Passing unit tests and checksum audits is evidence of
specific checks, not a submission-wide certificate or independent confirmation.

Search coverage: DINO-WM/JEPA-WM reproduction and public checkpoints; random,
norm-/spectrum-matched activation controls; COAST and world-model LEAP; patching
validity; RL variance and experimental design. Primary papers and official code
were used for conclusions; secondary search pages were used only for discovery.
Recent preprints are evidence to inspect, not automatically authoritative best
practice. This is a decision-focused literature review, not an exhaustive
systematic review of every steering paper.

Related operational guidance:
[MuJoCo replication and validation](docs/MUJOCO_REPLICATION_BEST_PRACTICES.md),
[experiment plan](docs/EXPERIMENT_PLAN.md),
[table-completion scope](docs/TABLE_COMPLETION_20260911.md).

Key primary references: [JEPA-WM](https://arxiv.org/html/2512.24497v4),
[RLVR-World](https://arxiv.org/html/2505.13934v2),
[COAST](https://arxiv.org/html/2605.17144v1),
[ITI](https://arxiv.org/html/2306.03341v4),
[LEAP](https://arxiv.org/html/2609.03294v1),
[Empirical Design in RL](https://jmlr.org/papers/v25/23-0183.html).

## 9. Full LEAP PDF follow-up: what the bars mean for our claims

All nine pages of arXiv:2609.03294v1 were read, including the references;
Figure4 and TablesI–VIII were checked against the rendered PDF.

LEAP §V reports a100-trial matched comparison against recomputed LeWM+CEM,
after50-trial development sweeps. PLDM, DINO-WM and Random are borrowed
contextual results, not reruns. Its error bars use one binomial standard error,
not95% confidence intervals. The system also uses a fitted action proposal,
state decoder and numerical goal descriptor: freezing the world model does not
mean no auxiliary fitting or added information.
[Full paper, §IV–VI](https://arxiv.org/pdf/2609.03294v1)

### The random-policy source code settles the category distinction

The linked LeWM evaluator selects `swm.policy.RandomPolicy()` when
`policy == "random"`; it does not load a world model on that branch.
`RandomPolicy.get_action()` calls `self.env.action_space.sample()` and ignores
the observation. This is **random action selection**, not randomized steering
inside an otherwise capable model/planner.
[LeWM evaluator, pinned revision](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/eval.py#L78),
[RandomPolicy, pinned revision](https://github.com/galilai-group/stable-worldmodel/blob/4821c8e6a3f0f83b7e6a80da3a757e026ea9026b/stable_worldmodel/policy.py#L170)

These revisions were inspected September11. They establish the released
implementation category, not bit-for-bit provenance of historical plotted runs.
The evaluator does not pass a seed to that constructor; the current policy can
seed the action space explicitly. Exact historical action RNG is therefore not
established merely from the evaluator's episode-sampling seed.

Our random controls preserve the trained checkpoint, goal-conditioned planner
and most native computation. A small random internal edit need not erase those
capabilities. Therefore a low random-policy bar is **not evidence that our
active controls should also approach zero**. Nor does this distinction certify
our implementation; the pairing, dose and RNG audits in §5 remain necessary.
If a future figure includes both controls, label them separately as
`random-action policy` and `JEPA-WM + matched random activation edit`.
Do not replace the harder specificity control with a weaker policy baseline.

### Which published baseline should appear?

LeWM explicitly distinguishes DINO-WM with and without proprioceptive inputs;
its default comparison omits proprioception. Thus “DINO-WM” is not a unique
universal score independent of its implementation, inputs and benchmark.
[LeWM §4.1 and AppendixC.1](https://arxiv.org/html/2603.19312v2)

| Row in our eventual figure | Number to use | Interpretation |
|---|---|---|
| JEPA-WM release, CEM-L2, no edit | Our audited native score under the same task protocol as each edited arm. | Primary causal reference for the added intervention. |
| Same release and planner + specified edit | All accepted paired episodes from that arm, with its registered denominator and uncertainty. | Estimated treatment effect versus native; not a selected best seed. |
| Same release + its specified active random control | Our measured frozen control, even when it scores higher. | Specificity of learned geometry beyond comparable perturbation/calibration. |
| JEPA-WM improved recipe, published | The identified main aggregate, labelled published, with its seed/checkpoint-window convention. | Context, not a substituted denominator for our steering gain. |
| DINO-WM, published | A named source table and explicit proprioception, feature backbone, planner and evaluation protocol. | Context; direct superiority requires an appropriately matched reproduction. |
| Paper ablations such as CEM-L1 or encoder variants | Separate explicitly named rows if relevant to the question. | Never silently choose the weakest variant as “the authors' baseline.” |

The same task name is insufficient: record simulator/dataset revisions,
initial/goal sampling, observations, action scaling, success timing, step budget,
planner horizon/candidates, checkpoint hash and aggregation. Keep the existing
JEPA-WM protocol; do not change it to make our numbers resemble another paper.
If a new baseline is reimplemented, retain both the published and reproduced
rows and explain discrepancies. A directly relevant additional precedent is
*Mind the Gap*, which establishes a reproduced flat-LeWM reference before
evaluating hierarchy, and reports goal-offset-specific settings and seed
variation. Its sweep-selected configurations are not interchangeable with an
untuned released default.
[arXiv:2607.12547v1, §3.1 and AppendixC](https://arxiv.org/html/2607.12547v1)

### Are our results moderately successful?

**They show some positive observed changes, not yet an established general
improvement.** Practical value and geometry-specific evidence are distinct.
A reliably positive native comparison can establish utility even when a random
edit also helps; claiming the learned geometry is responsible needs the active
control. Our existing primary success gate requires both and is unchanged.

| Completed observation | Edited / native / matched random | Honest reading |
|---|---|---|
| Reach, refined edit |49/96;43/96;48/96|Six extra observed successes over native, only one over calibrated random. Positive estimate, uncertain specificity.|
| Reach, equal-budget coupling |47/96;43/96;58/96|Four extra over native, eleven fewer than random. Not evidence that learned direction choice is superior.|
| Reach-Wall, equal-budget coupling |25/96;29/96;27/96|A regression in the observed rates, not a cross-task improvement.|
| DROID, equal-budget coupling |51.07;51.10;51.27 action score|Near-identical performance; the higher-than-paper native offset is not an intervention gain.|

The refined Reach native difference is+6.25 points, with the existing simultaneous
95% interval[−11.46,+23.96]. Calling that “no effect” is too strong; calling it
a demonstrated gain is also too strong.
[Frozen core results](reports/CORE_METAWORLD_BEHAVIORAL_RESULTS.md),
[DROID audit](reports/DROID_BASELINE_RECONCILIATION_AUDIT.md)

### Figure and analysis recommendation

Implement this reporting structure now; do not buy a new random-policy sweep
merely to obtain a low bar. Plot our matched native/edit/control results together,
with published contextual results in a clearly separated panel or style. Use
successes/episodes and a companion **paired percentage-point effect plot with
95% intervals**. Keep source-family clustering and the frozen multiple-comparison
family; do not replace them with independent-binomial calculations for narrower
error bars. Show DROID separately because its endpoint is not robot task success.

For illustration, the plug-in standard error `sqrt(p*(1-p)/n)` is zero at
100/100, but that does not imply zero uncertainty: a two-sided95% Wilson interval
has a lower bound near96.3%. Plotting conventions are not inferential guarantees.
Use paired outcomes for the difference, not overlap of two marginal error bars.
Preserve incomplete cells and negative outcomes instead of selecting only the
most favorable task/intervention combinations.

My assessment: a compelling figure should make the scientific comparison clear,
not make our controls look weak. First finish and audit the existing comparisons;
report positive estimates, regressions and uncertainty as they stand. Neither
a familiar chart format nor a favorable cross-paper subtraction supplies the
missing evidence of repeatability or mechanism.

## 10. Why our uncertainty looks wider than LEAP (September11)

These displays answer different questions. LEAP TableI uses one binomial
standard error of each success rate, not a95% interval for an improvement.
[LEAP TableI](https://arxiv.org/html/2609.03294v1#S5)
JEPA-WM's design-choice error bars instead describe standard deviation across
late-epoch success rates, after averaging training seeds and episodes per epoch.
They are also not confidence intervals for our added intervention's paired effect.
[JEPA-WM AppendixG.2](https://arxiv.org/html/2512.24497v4#A7.SS2)

| Quantity | Result | What the uncertainty means |
|---|---:|---|
| LEAP Push-T,100 trials |90.0% ±3.0 points|One binomial standard error of one success rate.|
| Our refined Reach,96 scenarios |51.04% ±5.10 points|The same marginal binomial-SE display, conditional on this checkpoint.|
| Our native Reach,96 scenarios |44.79% ±5.08 points|The same marginal-SE display.|
| Our refined-minus-native Reach |+6.25 points;95% interval[−6.25,+18.75]|Supplementary unadjusted paired empirical-bootstrap percentile interval.|
| Same Reach effect,original frozen analysis |+6.25 points;95% simultaneous interval[−11.46,+23.96]|Original eight-comparison family protection; unchanged.|

For a binary success rate, `SE = sqrt(p*(1-p)/n)`. Outcomes near50% have the
largest binomial variance;90% outcomes have less. Changing96 to100 trials
reduces this SE by only about2%, so that sample-count difference does not
explain the visual contrast. Going from one SE to a95% interval, changing the
estimand from one rate to a paired difference, and simultaneous coverage each
also change the displayed width. Pairing can reduce uncertainty relative to an
unpaired comparison; it does not imply a difference has the same SE as one arm.

Our refined Reach arm wins23 scenarios where native fails, but loses17 where
native succeeds, with56 ties. The six net wins therefore do not represent six
uniform improvements and zero regressions. Its unadjusted exact paired test
has `p=0.4296`. **None of the original eight contrasts has an unadjusted exact
p-value below0.05**, even before multiplicity correction. Thus the conclusion
is not caused solely by a stricter reporting convention. This is uncertainty
about benefit, not proof of zero effect or equivalence.

### Reproducible supplementary calculation

`scripts/vast/audit_uncertainty_display.py` reads the saved final core report,
checks96 distinct scenario clusters per contrast, and enumerates the empirical
bootstrap distribution from the sufficient win/loss counts. Output:
`artifacts/offline_study/table-completion-20260911-v1/UNCERTAINTY_DISPLAY_AUDIT.json`.
Input report SHA256:
`3db43adc45e7e7c203de980b264917eb27d61570828edb64a6b8553119f35d6a`.
The original report, frozen thresholds, comparison family and conclusions are
unchanged. This is CPU arithmetic on existing outcomes, not a new experiment.
Exact enumeration eliminates Monte Carlo error in this bootstrap distribution;
it does not guarantee exact frequentist coverage of a percentile interval.
Some enumerated tail endpoints differ by one episode from the original20,000-draw
estimate; the original frozen intervals remain the reported primary analysis.

### What is normal reporting practice?

There is no universal requirement that every robotics paper use our eight-way
gate. NeurIPS explicitly permits clearly labelled one-sigma bars, while requiring
the method and sources of variation to be explained. It distinguishes standard
deviation, standard error and confidence intervals.
[NeurIPS checklist, statistical significance](https://neurips.cc/public/guides/PaperChecklist)
Our joint requirement to beat native and active-random controls is our study's
specificity criterion, not a rule inherited from LEAP or JEPA-WM.

Recommendation: show readable raw success rates with clearly labelled uncertainty,
and a companion paired-effect table with ordinary95% intervals plus the original
simultaneous results. Separate practical improvement over native from evidence
that learned geometry beats a comparable random edit. Do not silently swap the
frozen decision rule after seeing outcomes. Marginal bars alone do not establish
whether a two-point paired gain is reliable, and narrower bars do not improve
the underlying evidence. Keep Push-T source-family clustering and DROID's
non-success endpoint rather than applying independent-binomial bars everywhere.
