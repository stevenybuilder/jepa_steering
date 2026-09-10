# Reach actual-H6 diagnostic — 2026-09-06

**Forecast gains survive at the native scoring horizon, but the correction still lacks a specific physical advantage over native-bank selection.** The original 15-step steering gate remains failed. This exploratory 30-step endpoint does not promote an operator.

The unchanged full spatial correction was applied to the same four previously selected arms on all eight DEVELOPMENT starts 66–73. Duplicate selected plans share explicitly recorded physical results: 14 distinct plans, 420 actual simulator steps, and 84 actual encoded future observations. There was no refit, new dose, new action selection, full99 episode, or original held-data access.

## Forecast improvement persists through H6

For the original native planner's selected plan, averaging equally over eight starts:

| Actual horizon | Visual MSE reduction | Proprio MSE reduction |
|---|---:|---:|
| H1 | 30.59% | 51.32% |
| H2 | 25.52% | 49.04% |
| H3 | 26.17% | 49.02% |
| H4 | 20.45% | 49.00% |
| H5 | 15.87% | 47.46% |
| H6 | **10.25%** | **38.20%** |

At H6, visual MSE is **1.793510 native, 1.609692 corrected, 1.848148 sham**. Proprio MSE is **0.097133 native, 0.060024 corrected, 0.102037 sham**. Correction improves both channels on all eight original native plans at H6. Gains weaken with horizon but do not disappear. Thus failure of all forecast generalization beyond H3 cannot explain the earlier physical failure. These measurements do not establish perfect calibration or correct action-to-action contrasts.

## Physical gains require the correct comparator

Mean 30-step hand-goal progress change for the frozen corrected selection:

- Versus original native planner: **+2.7661 mm**.
- Versus native-bank selected action: **−0.1719 mm**.
- Versus norm-matched sham selection: **+0.4489 mm**.

The largest positive difference, episode 66 at +20.1929 mm versus native planner, uses exactly the same action for native-bank, corrected, and sham arms. It is a bank-selection effect and cannot be credited to the correction. The two correction-specific native-bank choice changes yield +0.8202 mm on episode 70 and −2.1955 mm on episode 71. All arms have zero native success within these 30 steps.

The original 15-step result remains intact: −0.7337 mm versus native planner, with no positive native comparison. Different fixed-plan horizons can reverse the sign of a short physical proxy, so the new endpoint must not replace the failed prespecified gate.

## A concrete objective mismatch among the selected arms

We computed the unchanged native goal cost using **actual H6 encodings**, for the at-most-four frozen selected arms only. This is not an oracle for all 301 candidates.

On seven of eight starts, the minimum actual-encoding cost selects an action with the best physical progress among these selected arms, including ties. On episode 72, it chooses the worse physical action:

| Episode 72 action | Actual H6 native cost ↓ | 30-step hand-goal progress ↑ | Native reward sum ↑ |
|---|---:|---:|---:|
| Native planner / sham, candidate 300 | **1.749494** | 159.5460 mm | 107.6616 |
| Native-bank / correction, candidate 277 | 1.880360 | **164.7789 mm** | **113.1641** |

The selected-arm objective oracle therefore has **5.2328 mm physical regret** on this state; the equal-state mean is 0.6541 mm. This supplies a measured example where even the actual latent future's native goal score prefers less progress and less accumulated native reward. It does not establish that goal-cost mismatch is the dominant cause across the task or the full candidate bank.

The remaining useful distinction is between task-relevant candidate contrasts and average forecast calibration. Before another steering phase, a targeted later-state or contact-state development test should compare action-conditioned residual contrasts with actual goal scoring and physical outcomes under an unchanged confirmation boundary.

## Verification and preservation

All 14 trajectories reproduced the prior pilot's first 15 actions, states, and pixels exactly. Starting physics and goals, zero-dose identity, direct/analytic native cost equivalence, per-channel sham norms, and model parameter constancy passed. The released checkpoint SHA is `c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8` on every worker. Every state and selected arm is retained.

Aggregate diagnostic process time: **45.3400 seconds = 0.012594 GPU-hours**, below the 0.3 GPU-hour cap. Total Reach campaign process time including the earlier two pilots is 194.1712 seconds = 0.053936 GPU-hours. These are process measurements, not rental billing duration.

All 178,707,514 bytes of diagnostic outputs, code, protocols, and logs were verified on source and distinct backup hosts; compact JSON and receipts are local. Sources are `worker-ID/horizon-diagnostic-v1`, with separate backups `verified-backups/horizon-diagnostic-worker-ID` on the next host. See `horizon-diagnostic-summary-v1.json`, `horizon-diagnostic-backup-ID.json`, and `horizon-diagnostic-CHECKPOINT.json`.
