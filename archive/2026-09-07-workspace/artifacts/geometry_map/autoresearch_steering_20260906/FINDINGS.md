# Autoresearch steering checkpoint — 2026-09-06

No operator qualifies for the full held intervention phase. This round completed **27 Push-T configurations and two Reach-Wall correction pilots**, followed by a fixed-operator horizon diagnostic. Every candidate failed its predeclared physical-outcome promotion gate. Confirmation remained sealed.

The user-authorized cleanup is complete separately: instances `49987396`, `49987400`, and `49987408` contained no research results on actual disk inspection. All bootstrap project files were preserved with matching checksums before destruction; provider absence is confirmed. Removed running rates total **$0.985481/hour, approximately $23.65/day**. See [cleanup receipts](/Users/stevenyang/Documents/gpu_cleanup_2026-09-06/SUMMARY.json).

## What the exploration established

| Branch | Evidence | Decision |
|---|---|---|
| Push-T full visual forecast correction | Three fixed estimators × three caps. Best observed coverage gain versus the native-bank choice was 1.160 percentage points, but that setting lost 0.781 points to its shuffled-target sham. | All 9 fail. Lower forecast error does not establish action-specific usefulness. |
| Push-T forecast-error penalty | Same three estimators × three fixed strengths. Risk ranking was unreliable across states; larger penalties sometimes harmed outcomes. | All 9 fail. |
| Push-T action-contrast correction | Failure-guided within-state residual targets and action/future features, with context retained and a within-state target-permutation sham. Best RBF coverage gain was 0.700 points, restricted to one state and only 0.108 points over the sham. | All 9 fail. |
| Reach pooled → full spatial correction | Full spatial correction lowered measured H1–3 visual error 26.83% and proprio error 49.35% across eight starts. It changed two choices, both physically worse at 15 steps; mean progress −0.734 mm versus native. | Both pilots fail; no full 99-step episode promotion. |
| Reach actual H6 diagnostic | Same frozen choices/model/dose; actual H6 visual error −10.25%, proprio −38.20%, improving both on all 8 native plans. At 30 steps the correction was −0.172 mm versus native-bank selection. | Forecast improvement survives extrapolation; the apparent +2.766 mm versus the original planner mostly comes from a bank-selection effect shared with the sham. The earlier 15-step gate remains failed. |

The strongest new distinction is **shared forecast calibration versus useful differences between candidate actions**. In Push-T, a training-mean correction reduced forecast MSE 7.80%, exceeding the learned absolute estimators at the largest cap. Removing shared bias did not produce reliable contrast prediction or a reproducible physical benefit. The contrast pilot's lowest-dose linear/RBF models reduced contrast error only 0.313%/0.239%, with no selected-outcome gain.

Actual-future oracle diagnostics expose a second limitation. For Push-T, replacing visual forecasts with actual encodings produces the same selected actions as replacing both visual and proprio channels; the native goal objective still misses substantial physical headroom in one state. In Reach episode 72, the actual H6 native objective prefers an action with 5.233 mm less progress and lower accumulated native reward among the at-most-four selected arms. These are privileged explanatory comparisons, never deployable interventions or oracles over unmeasured candidate actions. They demonstrate concrete ranking disagreements without proving that objective mismatch dominates every failure.

## Scope, integrity, and preservation

Push fitting used 16 existing TRAIN groups with 7 actions each. Its four previously seen development states supplied 256 cached candidate plans; the independent unit is the initial state. Reach calibration used 76 measured outcomes from four old DEVELOPMENT episodes, evaluated on all 8 new DEVELOPMENT starts 66–73. The H6 diagnostic added 14 distinct 30-step replays, 420 simulator steps, and 84 actual encoded futures; all reproduced their prior 15-step prefixes exactly.

All doses, sites, comparator definitions, and gates were frozen before their respective new scoring runs. Every trial and failed gate is retained in [TRIAL_LEDGER.jsonl](TRIAL_LEDGER.jsonl). No candidate-level independence, p-value, confirmation efficacy, internal causal site, or positive result is claimed. The 30-step endpoint does not replace the failed 15-step endpoint.

Nine numerical runtime tests passed, along with feature-centering checks, exact zero-edit identity, matched-sham norms, physical replay checks, and recorded model/source checksum checks. Each completed experiment is verified locally as applicable and on its source plus a distinct backup host; the large Reach tensors remain remote. One slow-host contrast attempt was stopped before fitting and rerun unchanged on an idle assigned GPU. Its abort receipt and exact executed code are retained.

Six existing GPUs were used; none were rented. All campaign jobs and transfers finished, and assignments were returned to the shared project board. Measured experimental process time was 215.820 seconds across all GPUs (0.05995 GPU-hours). This excludes setup, CPU work, transfers, and idle rental time and must not be interpreted as the billed duration.

## Failure-informed next step

Collect matched candidate-action outcomes across approach, contact, and post-contact states under an explicit TRAIN/development-calibration split. Check whether actual goal scores order those physical outcomes correctly, then fit task-relevant action contrasts. A future operator should first beat native and matched sham on fresh development snapshots, then complete development episodes. Keep the current confirmation boundary and freeze site, operator, dose, target, and analysis before opening it.

This direction follows the paired [arXiv review](arxiv_notes.md), including [RP1](https://arxiv.org/abs/2608.18669), [LEAP](https://arxiv.org/abs/2609.03294), and [TD-JEPA](https://arxiv.org/abs/2607.25337). Those papers motivated hypotheses and counterarguments; these experiments are not reproductions of their reported systems.

Detailed evidence: [Push interpretation](push/interpretation.md), [Reach pilots](reach/FINDINGS.md),[actual H6 diagnostic](reach/horizon-diagnostic-FINDINGS.md), and [machine-readable summary](SUMMARY.json).
