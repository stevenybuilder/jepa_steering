# Corrected primary offline results — 2026-09-07

All five frozen sweeps completed in BF16 (primary) and FP32 (sensitivity) on all three
primary author-validation pools. Exact upstream input, context, rollout and metric
checks passed. This finishes the corrected offline phase, **not the end-to-end study**.
The governing design is [EXPERIMENT_PLAN.md](../docs/EXPERIMENT_PLAN.md), with its
explicit author-validation correction in [author_correction.json](../configs/author_correction.json).

## Coverage and unit of analysis

Reach: 33 trajectories / 33 lineage groups. Reach-Wall: 27 / 27. Push-T: 21 / 21.
These are 81 unique primary trajectories, not 81 times the number of arms, windows,
or precisions. All 30 task/category/precision analysis scopes are verified complete.
Each arm covers 10,590 H6 prefix rollouts across the three tasks (7,200 MW + 3,390 Push-T).
The separate broad unsteered baseline covers 1,120/1,260 rows across all 42 MW tasks.
The remaining protected broad-pool rows are not silently opened.

The seven added primary MW rows and 21 Push-T validation rows were explicitly
authorized for frozen replication. Historical exposure records were preserved.
They are not an untouched confirmation cohort after development selection.

## Main comparisons with the unsteered checkpoint

Positive numbers below mean lower **proprioceptive embedding H6 MSE**, not higher
task success. Intervals are paired-lineage simultaneous 95% intervals within each
task/sweep/precision, jointly covering registered contrasts and both H6 MSE endpoints.
Percent intervals scale the difference interval by the observed native mean; they
are not ratio-parameter confidence intervals. These selected examples are descriptive;
the complete export retains all 672 arm/endpoint/precision comparisons, including controls.

| Task | Frozen arm | Error reduction vs native | Simultaneous 95% interval | Frozen advancement gates |
|---|---|---:|---:|---|
| Reach | Joint vision/action | 2.877% | [1.872%, 3.883%] | Eligible; combined recipe not frozen |
| Reach | Rank 4 | 2.887% | [2.276%, 3.499%] | Eligible; explicit rank-parsimony rule selects rank 4 |
| Reach | Single block 0 | 3.026% | [2.124%, 3.929%] | Eligible; does not uniquely localize the mechanism |
| Reach-Wall | Joint vision/action | 2.327% | [0.825%, 3.828%] | Does not establish the frozen useful-effect threshold |
| Reach-Wall | Rank 4 | 2.137% | [1.645%, 2.629%] | Only eligible rank; selected by the fixed rule |
| Push-T | Joint vision/action | −0.638% | [−1.214%, −0.062%] | Worse than native; not eligible |
| Push-T | Rank 4 | 0.198% | [−0.059%, 0.455%] | Not eligible |

BF16 native H6 errors are 0.0008585703183 (Reach), 0.0008200658457 (Reach-Wall),
and 0.0002192615816 (Push-T). All listed comparisons delivered energy within the
already-frozen FP32 tolerance; the tolerance was not adjusted to obtain this result.

No geometry arm qualifies on any task. No Push-T intervention qualifies in the
primary analysis; retain native, without selecting a more favorable FP32 result.
On Reach-Wall, only rank 4 qualifies across the five sweeps. Reach has eligible
coupling, rank, layer and all-patch choices, requiring the planned combination and
drop-one work. Pairwise uncertainty and unregistered comparisons must not be
silently resolved by a new purportedly predeclared tie breaker.

The explicit rank rule chooses the smallest eligible rank equivalent to the observed
best under the frozen margin. Reach rank 4 and rank 8 meet that test; rank 1 does not.
Failure to reject a rank difference was not treated as equivalence.

## Completion evidence

Remote closure: `/workspace/jepa-runtime/three-task-offline-closure-20260907`.
Durable export: `artifacts/offline_study/primary-durable-20260907/three-task-offline-closure-20260907`.

- Closure report SHA256: `c282b82a32663ebd48447f399d088251758a318a0d0a3707e07be126d0a7db34`.
- Full metrics report: `0c476a2468b4f939a8458c1dc1fec9325306d3b15191b667568fc7e9124cdf50`.
- All-arm CSV: `d001b9678716b5a993121c199156b34b2b7244e329ffbba45c462d729aec27b7`.
- Primary parsimony audit: `d7fbdfdc682636d3e7ca844fde6ca68ff77b9f89a5e72d333fd6a2fccaab531a`.

The source instances retain their originals. Utah BF16 evidence was checksum-verified
locally and on the primary host before Utah was stopped. The primary durable transfer
and its independent verification are recorded separately; a transfer starting is not
proof of durable completion.

## Remaining work

Combined-recipe/drop-one checks, any eligible layer/position compatibility check,
prospective cohort/exposure verification and a frozen policy-analysis contract remain.
The native official CEM has passed non-confirmatory simulator checks on all tasks.
Ninety-six candidate-independent initial/goal scenarios per MW task are prepared;
these used only the official expert to define goals, not learned-policy outcomes.
Static BF16-primary banks also pass fit-only transfer to strict-FP32 planning.
Dynamic rank/support planning transfer is still unverified and must not be bypassed.

No closed-loop success benefit, fresh confirmation, three-trained-seed replication,
or last-ten-epoch aggregation is established by these offline results.
