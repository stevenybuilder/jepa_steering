# Push-T checkpoint: all three original families

This checkpoint covers the verified linear, RBF, and state-diverse kNN runs. All nine correction settings and all nine learned-risk settings fail their frozen promotion gates. No original-pilot candidate qualifies for fresh development validation. These are four adaptively reused initial states with 64 dependent candidate plans each, not 256 independent replications. Linear completed normally after slow CPU loading; that delay was not a scientific failure.

Sources: `results-49987398/summary.json` (linear), `results-49987413/summary.json` (RBF), `results-49987414/summary.json` (kNN), `full_encoding_oracle_diagnostic.json`; calculations and source hashes are in `interpretation_diagnostic.json`. No new fit, rollout, or GPU job was performed for this interpretation.

## Correction results

Native bank argmins are `[0, 0, 0, 1]`; selected requested-goal coverage is `[0.887536, 0.261760, 0.878812, 0.207583]`, mean **0.558923**. Mean available coverage regret is **0.106529**.

| Family | Cap | Selected candidates by state | Mean coverage gain vs native | Mean gain vs TRAIN-permutation sham | Mean visual-MSE reduction | Positive physical states |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Linear | .005 | 0, 0, 0, 1 | 0 | 0 | 1.183% | 0/4 |
| Linear | .02 | 0, 0, 0, 1 | 0 | −.004600 | 3.729% | 0/4 |
| Linear | .1 | 2, 45, 0, 5 | +.011600 | −.007806 | −7.684% | 2/4 |
| RBF | .005 | 0, 0, 0, 1 | 0 | 0 | 1.477% | 0/4 |
| RBF | .02 | 0, 0, 0, 1 | 0 | −.004600 | 4.905% | 0/4 |
| RBF | .1 | 2, 2, 0, 5 | +.011600 | −.007806 | 5.710% | 2/4 |
| kNN | .005 | 0, 0, 0, 1 | 0 | 0 | 1.255% | 0/4 |
| kNN | .02 | 0, 0, 0, 5 | +.004600 | 0 | 4.015% | 1/4 |
| kNN | .1 | 0, 0, 0, 5 | +.004600 | −.014805 | 3.009% | 1/4 |

All settings preserve the maximum-coverage-harm guard, but none achieves mean coverage gain .02 against native and each sham or improves three states. Linear cap .1 additionally fails the mean-MSE guard: error rises 7.684%. Signed-coordinate shams retain native selections throughout, so semantic gains against that sham equal gains against native. At cap .1, linear and RBF lower average regret to .094929 and kNN to .101928; all remain worse than the TRAIN-permutation sham on the primary endpoint. RBF's largest-cap per-state MSE ratios are `[.8853, 1.0586, .8892, .9525]`: its overall gain hides increased error in state 1. The frozen gate was an average, so this is a reported limitation rather than a rewritten failure criterion.

The strongest mechanism discriminator is the **TRAIN-mean residual control**. At cap .1 it reduces mean visual MSE **7.798%**, exceeding RBF and kNN, while its mean coverage improvement is only .004600. At cap .02 it also beats both learned estimators on MSE. Lower forecast MSE therefore currently supports a substantial common calibration component; it does not establish useful action-conditioned correction. Global TRAIN-label permutation also retains this common component and often improves selected coverage more than the semantic estimator.

## Risk results

| Family | Strength | Mean coverage gain vs native | Positive states | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Linear | .1 | +.005344 | 2/4 | Below gain and replication thresholds |
| Linear | .3 | +.005344 | 2/4 | Below gain and replication thresholds |
| Linear | 1 | −.020762 | 2/4 | State 0 loses .056258, exceeding harm guard |
| RBF | .1 | 0 | 0/4 | No selection changes |
| RBF | .3 | +.001458 | 1/4 | Too small, insufficient replication |
| RBF | 1 | −.003534 | 2/4 | State 0 loses .056258, exceeding harm guard |
| kNN | .1 | 0 | 0/4 | No selection changes |
| kNN | .3 | 0 | 0/4 | No selection changes |
| kNN | 1 | +.009073 | 1/4 | Below threshold; worse than permuted risk by .001125 |

Posthoc within-state Spearman correlations between predicted risk and actual all-candidate visual error are **linear `[−.118, .255, −.118, −.131]`**, **RBF `[.290, .379, −.205, −.144]`**, and **kNN `[−.069, .593, .011, .144]`**. These are descriptive correlations over each fixed bank, not independent-sample inference. RBF and kNN predict mean state-3 error .0853 and .0744 against actual .3962. The risk estimates do not provide dependable error discrimination across all four states.

The action-magnitude control at strength .1 gains mean coverage .033242, but only two states improve and it chooses the zero-command candidate in states 1 and 3. Joint-XY distance worsens strongly in those states (+56.14 and +52.64 px); strength .3 or 1 loses mean coverage .260163. This control is evidence to preserve the explicit no-op/action-size comparison, not evidence for promotion of the learned risk method or a broadly beneficial regularizer.

## Perfect-future diagnostics separate two bottlenecks

Substituting actual visual futures while preserving predicted proprio selects `[8, 34, 51, 15]`. These choices and their physical coverage exactly match root's separate oracle replacing **both** encoded channels. Thus omitted proprio correction does not explain the four oracle selections.

| State | Native coverage | Actual-encoding oracle coverage | Best available physical coverage | Remaining oracle-objective regret |
| ---: | ---: | ---: | ---: | ---: |
| 0 | .887536 | .918720 | .927338 | .008618 |
| 1 | .261760 | .261760 | .327982 | .066222 |
| 2 | .878812 | .972816 | .972816 | 0 |
| 3 | .207583 | .202195 | .433670 | .231475 |

Mean oracle gain is .029950, but only two states improve and state 3 becomes worse. Exact forecasts under the native objective therefore would not meet the registered three-state improvement gate on these banks. Prediction error matters, especially in state 2, while native objective alignment remains limiting in states 1 and 3. These privileged arms are explanatory diagnostics; they are not runtime methods, a physical upper bound on all biased scorers, or intervention evidence.

## TRAIN-to-development support check

TRAIN-standardized feature distance to the nearest training point has development medians **1.16, 1.20, 1.51, 1.31 times** the TRAIN leave-initial-state-out nearest-neighbor median. The fraction beyond the corresponding TRAIN 95th percentile is **6.25%, 35.94%, 100%, 96.88%** by development state. The last two banks are poorly covered under this particular feature-distance diagnostic. This posthoc proxy does not establish a calibrated OOD boundary or prove distribution shift caused the failure.

## One justified conditional next mechanism

All original families have now failed, satisfying the conditional trigger for **within-state residual contrasts** to distinguish common calibration from candidate-action information. For each TRAIN initial state, subtract the seven-candidate residual mean from its targets and subtract the candidate mean from the action/future feature block. At development scoring, center the analogous feature block over the unchanged 64-candidate bank using features only. Retain the same families, caps, native objective, and physical gates. Use a **within-TRAIN-state permutation** of centered residual targets as the main sham: this preserves each state's residual distribution and removes its action pairing. The original pilot's mean-only correction remains a separately reported calibration comparator; the centered-target mean is analytically zero and is represented by identity in the contrast pilot.

This is a bounded diagnostic proposal, not a predicted success. Naively centering every feature makes the current-context coordinates identically zero and removes state/contact-regime conditioning. Either explicitly label that restricted hypothesis or retain uncentered TRAIN-standardized current context as conditioning while centering action/future contrasts. Without interactions or nonlinear neighborhoods, an additive linear context term alone will not express context-dependent action effects. Centering development features is permissible unlabeled transduction, but its result depends on candidate-bank composition and requires the same bank under every arm.

The primary question is whether centered semantic correction improves **within-state residual contrast error and physical action selection over the within-state sham**, not whether it lowers a shared visual offset. If it still improves forecasts without sham-specific physical wins, the output-correction direction should not advance on that evidence.

This reasoning agrees with [RP1's documented model-exploitation failures](https://arxiv.org/html/2608.18669v1) and [TD-JEPA's poor pure-temporal Push-T control despite stronger temporal ranking](https://arxiv.org/html/2607.25337v1). [LEAP's descriptor-only ablation](https://arxiv.org/html/2609.03294v1) further cautions against replacing native geometry with a single apparently meaningful scalar. None of those papers establishes that our proposed contrast operator will work.

## Completed contrast pilot and final checkpoint

All **nine contrast settings fail the unchanged gates**. Sources are `contrast-49987413/summary.json` (RBF), `contrast-49987414/summary.json` (kNN), and `contrast-linear-49987413/summary.json` (linear). The initial linear run on 49987398 stopped during slow CPU input loading before fitting; the unchanged retry completed on 49987413. No scientific result was discarded. Full per-state semantic/native and semantic/both-sham contrast-MSE comparisons, physical outcomes, gates, and hashes extend `interpretation_diagnostic.json` under `contrast_pilot`.

Positive MSE reduction means improvement; negative means worse than native. Coverage is on the 0–1 scale.

| Family | Cap | Mean coverage gain vs native | Gain vs within-state sham | Contrast-MSE reduction | Full-MSE reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | .005 | 0 | 0 | .313% | .214% |
| Linear | .02 | 0 | 0 | −.139% | −.129% |
| Linear | .1 | 0 | −.005921 | −8.031% | −5.617% |
| RBF | .005 | 0 | 0 | .239% | .171% |
| RBF | .02 | +.006999 | +.001078 | −.089% | −.061% |
| RBF | .1 | +.006999 | +.001078 | −.319% | −.223% |
| kNN | .005 | 0 | 0 | .028% | .031% |
| kNN | .02 | +.006999 | −.001281 | −.422% | −.296% |
| kNN | .1 | +.006999 | −.001281 | −.440% | −.308% |

Only state 0 improves physically for the nonzero RBF/kNN settings; none improves three states. RBF exceeds the within-state sham by .004313 coverage on state 0 and ties on the other states. kNN gains that same .004313 specificity on state 0 but loses .009436 to the sham on state 2. The signed-coordinate sham preserves native outcomes for those settings. Linear changes no selected physical coverage and at cap .1 also fails the full-MSE guard.

There is a **small predictive contrast signal**, not a validated steering effect. At cap .005 the per-state contrast-MSE reductions are linear `[.432%, .281%, .140%, .352%]`, RBF `[.469%, .244%, .097%, .235%]`, and kNN `[.313%, −.120%, −.114%, .030%]`. Aggregate semantic contrast error is respectively .360%, .503%, and .204% below the within-state sham, yet physical choices do not improve. At larger caps semantic contrast MSE exceeds native even when it beats a worse sham. The native within-state contrast component is .129962 of .185897 full MSE, approximately 69.91%; common-bias correction explained previous *improvements*, not most total error.

Across both Push correction pilots and its risk branch, **27 prespecified semantic settings were evaluated and none passed**. Reusing the same four states limits every result to exploration. The result does not warrant a larger blind dose or kernel sweep.

The [Reach H6 diagnostic](../reach/horizon-diagnostic-FINDINGS.md) reinforces the distinction: visual/proprio forecast gains persist on all eight starts (10.25%/38.20% mean reductions), while corrected physical progress is −.1719 mm versus native-bank selection. Its +2.7661 mm versus the original planner is chiefly a bank-selection gain shared with controls. On development state 72, even scoring actual future encodings chooses an action with 5.2328 mm less progress among the selected arms. This is a concrete objective mismatch, with a limited selected-arm scope.

The next useful resource is **better matched state/contact-conditioned action-contrast data plus objective validation**: collect training-only contrasts spanning approach, first contact, and post-contact alignment from otherwise identical states, retain spatial/task-relevant features, and measure both actual encoded goal cost and physical outcome for each action. Compare candidate contrasts with within-state shams and preserve native/no-op baselines. Determine whether goal scoring orders the real outcomes correctly before attributing another forecast gain to steering. This is a targeted data and identification need, not evidence that another operator will succeed; original confirmation remains sealed.
