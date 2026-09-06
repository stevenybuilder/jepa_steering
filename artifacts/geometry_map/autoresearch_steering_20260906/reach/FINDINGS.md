# Reach-Wall development checkpoint — 2026-09-06

Two frozen output-correction pilots improved forecast calibration but failed the physical steering promotion gate. No full episode intervention or original held data was opened.

| Pilot | Actual H1–3 visual MSE | Actual H1–3 proprio MSE | Changed choice versus native bank | Mean 15-step progress versus native planner | Mean progress versus sham |
|---|---:|---:|---:|---:|---:|
| Native reference | 0.931238 | 0.048462 | — | — | — |
| Pooled correction | 0.913617 (−1.89%) | 0.024620 (−49.20%) | 0/8 | −0.6307 mm | 0 mm |
| Full spatial correction | 0.681363 (−26.83%) | 0.024545 (−49.35%) | 2/8 | −0.7337 mm | +0.2440 mm |

MSE is averaged equally across the eight episodes, then their three physically executed horizons. Both corrections improved both channels on all eight initial states. These are exploratory development results, not independent confirmation. The full spatial sham has visual/proprio MSE 0.982847/0.053273; its small mean physical disadvantage does not establish correction efficacy: the correction beats the native planner on zero states and beats the sham on only one.

## What was run

- Calibration used all 76 available actual H1–3 outcomes from fixed native full99 records for old DEVELOPMENT episodes 4, 7, 9, and 10. These are development-calibration data, not the original dataset training partition. No edited-arm outcome was fitted. Source hashes were checked before extraction.
- The inputs are native pooled context and forecast features, proposed action summaries, and horizon. Ridge regularization is fixed at `10 × 76 = 760`, dose at `0.5`. The first model predicts pooled 400-channel errors; the failure-guided second model predicts the entire spatial 256 × 400 residual from exactly the same examples and features.
- Evaluation uses all eight newly allocated DEVELOPMENT initial states 66–73. For each state, the last native CEM population of 300 plans plus the native planner's selected mean forms a fixed 301-plan bank. No candidate-specific test future labels are available to inference.
- The native goal, world-model parameters, visual/proprio weights, objective formula, and proposed plans remain fixed. Corrections are applied at the **output** as a mechanistic comparison; these runs do not identify or steer an internal causal site.
- Signed channel permutation, and an additional spatial permutation for the full-grid model, preserves each candidate's visual and proprio correction norms separately. Native-bank winner and original native-planner selected mean are distinct controls.
- Each selected action executes 15 actual raw simulator steps from the identical prepared start. Identical selected plans share an explicitly marked physical result within a pilot; both pilots re-executed their native references. Real native reward and success information was retained alongside hand-goal progress.

The predeclared full99 gate requires more than 5 mm mean progress over each of native planner, native-bank winner, and sham, plus positive differences from native planner and sham on at least five of eight states. Both pilots fail. No candidate was promoted.

## What the failures reveal

The pooled correction largely preserves the ordering: mean Spearman correlation of corrected versus native model costs is 0.99669, and no native-bank winner changes. Its candidate-dependent cost variation is only 2.48–14.24% of native cost variation across these banks. Large proprio calibration improvement therefore supplies little useful decision change here.

The full spatial correction has greater candidate-dependent cost variation, 10.91–56.81% of native variation; mean rank correlation falls to 0.91199. It changes the native-bank winner on episodes 70 and 71, where actual progress falls by 0.6867 mm and 0.1377 mm respectively. Preserving spatial structure makes the intervention consequential, but these changed choices still fail to help. The +0.2440 mm average versus sham is chiefly sham harm on episode 69, not improvement over native behavior.

This evidence supports a limited conclusion: **better forecast MSE is insufficient as a promotion criterion for these output corrections and initial-state banks.** It does not establish that the native goal metric is the sole cause. Calibration truth covers H1–3, while the unchanged native objective scores H6; applying the correction to H6 extrapolates beyond its measured physical horizon. The entire 301-plan bank lacks physical truth, and the first 15 steps from initial states do not characterize all later contact or recovery situations.

The next useful Reach experiment is a bounded set of later DEVELOPMENT snapshots near observed contact or failure, with physically executed horizon-matched alternatives and an actual-encoding oracle/control. That would help separate long-horizon extrapolation, a poor goal objective, and missing action-contrast information before another full episode campaign.

## Integrity, literature, and compute

Both pilots passed exact zero-dose cost identity, direct-versus-analytic output-correction cost equivalence, separate-channel sham norm equality, original native cost reproduction within the recorded tolerance, exact selected-native command/state/pixel replay, identical paired starting physics/goals, and unchanged model-parameter hashes. All eight states and both trials are retained, including negatives. No original held episodes 12–65 were loaded.

The paired [arXiv decision note](../arxiv_notes.md) motivated this test from prediction-versus-control failures and bounded residual correction. [RP1](https://arxiv.org/abs/2608.18669), [SpikeWorld](https://arxiv.org/abs/2608.07712), and [MBPO](https://arxiv.org/abs/1906.08253) are relevant primary sources; the specific external correction tested here is our hypothesis, not a reproduction or established guarantee from those papers.

Measured aggregate pilot process time: 72.8983 + 75.9329 = **148.8313 seconds, or 0.041342 GPU-hours** across the three exclusively assigned instances. This is model/simulator process time, not total rental billing time; setup, CPU extraction/fitting, transfers, and idle rental time are additional. Full tensors remain on source and distinct backup hosts, with verified manifests; compact reports and receipts are local. No new rental or instance lifecycle change occurred.

Artifacts: `pilot-v1-summary.json`, `pilot-fullgrid-v1-summary.json`, `protocol-v1.json`, `protocol-fullgrid-v1.json`, `worker-ID/margin-diagnostic-v1/summary.json`, and `backup-fullgrid-ID.json`.
