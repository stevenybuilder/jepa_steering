# Matched-random control audit — September 11, 2026

## Bottom line

The saved MetaWorld controls pass the checks performed here: the paired inputs,
seed assignments, recorded planner budgets, frozen operators, and delivered edit
magnitudes agree with their contracts. No setup or dose mismatch was found that
explains the unusually high Reach random-coupling score. This is not a proof that
the implementation has no possible defects, nor independent confirmation that a
random edit reliably improves planning.

The main unresolved issue is **random-direction robustness**: there is one frozen
control realization per task, not repeated independent random-control draws.

## Paired outcomes, recovered from the original 960 evaluations

“Win” means the control succeeded on a scenario where unsteered failed; “loss”
means the opposite. Each row compares the same 96 scenarios, not separate samples.

| Task | Random control | Control / unsteered successes | Wins | Losses | Net change |
|---|---|---:|---:|---:|---:|
| Reach | Refined fixed-response | 48 / 43 | 25 | 20 | +5.21 percentage points |
| Reach | Vision–action coupling | 58 / 43 | 30 | 15 | +15.63 percentage points |
| Reach-Wall | Refined fixed-response | 23 / 29 | 7 | 13 | −6.25 percentage points |
| Reach-Wall | Vision–action coupling | 27 / 29 | 13 | 15 | −2.08 percentage points |

For Reach random coupling, six of eight logical streams have positive net gains,
one ties, and one is negative; the net gain is not confined to one stream.
Streams are execution groupings, **not eight independent control-direction draws**.

The new control-versus-unsteered calculations are explicitly **post hoc**.
Reach random coupling has an unadjusted exact paired-discordance p-value of
0.03570, but Holm p = **0.14279** across the four supplemental comparisons.
Its four-comparison simultaneous bootstrap interval is **[−1.04, +32.29] points**.
As a broader sensitivity check, Holm correction across the original eight tests
plus these four yields p = **0.42837**. None establishes a confirmed benefit.
These adjustments do not erase the fact that the question was selected after
seeing the results; the original frozen analysis remains unchanged.

## What was checked

- All **6,872 preserved files** passed Drive part, archive, and member checksums.
- All **40 shard reports and 960 episodes** retain their frozen-source bindings;
  episode, action-trace, and forecast-call hashes match.
- All five arms share exactly paired initial state vectors, initial/goal images,
  scenario assignments, and planner seed assignments within each task.
- There are **96 distinct initial/goal scenario clusters per task**.
- Each arm retains 100 elementary simulator steps and 300 planner candidates;
  all recorded forecasts have one backend call. No short-horizon edit was found.
- Coupling random visual/action directions reproduce **bitwise** from the saved
  seeds (2026090703 / 2026090704) using the archived tensor construction code.
  Their norms are approximately one and their cosines with the learned directions
  are below 5×10⁻⁹ in absolute value.
- Coupling learned/random arms have exactly matching sites, times, and scales.
  Their mean delivered total L2 magnitudes agree to better than one part in a
  million. Requested-versus-delivered discrepancies are floating-point-sized.
- Refined learned/random arms have four orthonormal basis vectors over the same
  support and the same dose; **all 2,167,200 recorded candidate edits per arm per
  task were active**. These are internal planner evaluations, not extra episodes.
  The largest individual relative dose-delivery error was below 1.8×10⁻⁶.

## What “matched” does and does not mean

The coupling control replaces learned directions with fixed orthogonal random
directions, holding edit sites and sizes constant. It is an active intervention,
not a no-op, and matching activation norm does not match downstream forecast or
action-selection effects.

The refined control uses a random four-dimensional basis **with response
calibration**, under the same calibration procedure and dose rule as the learned
basis. It is not uncalibrated noise. Its fitted coefficient-map singular values
are not claimed to match those of the learned operator; the saved banks confirm
that they differ. The support and final activation dose are the matched quantities.

No evaluation outcomes were used to change either control in this audit. These
checks do not establish robustness across alternative random directions, new
scenario populations, training seeds, or checkpoints. Those would require a
separate prospectively specified comparison, not relabeling this development panel.

## Reproducible evidence

- [Paired records and delivered-energy audit](../artifacts/offline_study/table-completion-20260911-v1/CONTROL_AUDIT_REPORT.json)
- [Saved direction/bank audit](../artifacts/offline_study/table-completion-20260911-v1/CONTROL_BANK_AUDIT.json)
- [Archive restoration receipt](../artifacts/offline_study/table-completion-20260911-v1/CONTROL_AUDIT_RESTORED.json)
- [CPU record-audit script](../scripts/vast/audit_core_random_controls.py)
- [CPU bank-audit script](../scripts/vast/audit_control_banks.py)
- [Unchanged primary behavioral analysis](CORE_METAWORLD_BEHAVIORAL_RESULTS.md)

This audit made **zero model/GPU calls and ran zero new simulator episodes**.
Scope: the two-task completed core panel, not a new six-task efficacy validation.
