# Results and interpretation

## Protected confirmation

The [README table](../README.md#final-protected-results) reports the complete final
panel: 384 scenarios, eight arms, four tasks, 3,072 evaluations. The committed
[report](../reports/fresh-confirmation/report.json) contains all 32 success rates
and 48 paired contrasts. It is a byte-identical copy of the frozen analyzer output.

All 48 simultaneous intervals include zero. This includes arm-minus-native
comparisons, not only randomized comparators. Positive point estimates remain
positive estimates; wide intervals do not establish equivalence or zero effects.
There is no independent-training-seed replication.

## Earlier results are separate evidence

The original six-task native row was 44.79 / 30.21 / 59.38 / 80.21 / 76.04 / 51.10
for Reach / Reach-Wall / Push-T / PointMaze / Wall / DROID. It remains a historical
measurement. It is not the denominator for fresh intervention effects.

Fresh scenarios and scenario-specific planner RNG streams differ from the earlier
evaluation. Neither hardware noise alone nor interventions alone explain baseline
differences. Push-T and DROID were not rerun in the final protected panel.

Earlier [offline results](../reports/CORRECTED_OFFLINE_RESULTS.md) and
[behavioral development](../reports/CORE_METAWORLD_BEHAVIORAL_RESULTS.md) answer
different questions. The refined successor reduced BF16 H6 proprioceptive embedding
MSE by 2.360% on Reach and 2.192% on Reach-Wall. These are not task-success gains.
They were measured on development populations, not fresh forecast/outcome pairs.

Historical MetaWorld component extensions used a concurrent native reproduction
(45.83 / 29.17) and a disclosed device-equivalence amendment. Those earlier extensions
are not substituted into the final same-physical-GPU confirmation. DROID reports
recorded-plan action agreement, not robot success. Author-paper aggregates are
context and are not directly paired causal comparisons with our interventions.

## Mechanistic diagnostics

All six post-confirmation analysis scripts completed on the real panel. Their
completion is not six independent experiments or a proven mechanism. An independent
recount of all arm records produced the committed [headline audit](../reports/fresh-confirmation/mechanism_audit.json).

| Task | Refined rescues / regressions | Coupling rescues / regressions | Refined coefficient effective dimension |
|---|---:|---:|---:|
| Reach | 21 / 21 | 14 / 20 | 2.23 |
| Reach-Wall | 6 / 14 | 8 / 18 | 1.37 |
| PointMaze | 7 / 5 | 4 / 6 | 3.07 |
| Wall | 4 / 5 | 11 / 9 | 1.92 |

Rescue means native failure/intervention success; regression is the reverse.
Dimension is `(tr M)^2 / tr(M²)` with `M = sum(c cᵀ)`, an uncentered second moment.
It describes concentration within the allowed four-dimensional subspace, not a
one-dimensional physical code or causal explanation.

Mean delivered coupling energy divided by its comparator was approximately one
on all four tasks (maximum deviation below 3 × 10⁻⁷). Both learned interventions
changed first-action hashes in 96/96 scenarios per task. Hashes establish byte
differences, not the size or usefulness of changed actions. These findings do not
support a simple absent-edit explanation; they do not certify every implementation
detail or identify the cause of the negative efficacy result.

**Analysis correction:** the first dose reader expected summed-energy fields while
fresh records stored means. It incorrectly emitted missing-dose warnings. The
corrected reader converts means using the recorded population size, validates that
size, and leaves unknown active-candidate counts unknown. Success counts and frozen
intervals were unchanged. Tests cover both schemas. Raw generated prose with those
stale warnings is not presented as the final interpretation.

Fresh candidate costs, elite IDs and numeric selected-action arrays were not saved.
They cannot be reconstructed from hashes. Candidate-trace and layer-attention code
has CPU tests but **has not produced a new GPU result**. A forecast-to-decision
misalignment remains a hypothesis, not a demonstrated failure mechanism.
