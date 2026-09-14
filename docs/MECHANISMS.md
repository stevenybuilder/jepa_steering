# From internal edits to planner decisions

The evidence separates four endpoints: activation reconstruction, recorded-action
forecast error, candidate scoring, and closed-loop success. An improvement at one
does not establish an improvement at the next. September 13, 2026 reanalyses are
exploratory; the original offline experiments and frozen protected analysis remain
unchanged.

The earlier layer sweep used independently fitted rank-one operators. The later
fixed-response rank-four successor reduced BF16 development H6 proprioceptive MSE by
2.360% on Reach and 2.192% on Reach-Wall. Neither is a treatment-invariant causal
chain into protected behavior. Fresh Reach has 21 refined rescues and 21 regressions:
both native and refined succeed in 52/96 scenarios. The complete protected panel
contains 384 paired scenarios and 3,072 evaluations, without an established net
success gain. [Results and uncertainty](RESULTS.md).

## Layer response map

The recount verifies **576 task/precision/support/modality/horizon cells** from six
hash-verified reports against the published export. It retains all six singleton
blocks, B2+B3, all-six support, and matched-random comparisons. Native and zero-dose
metrics agree. Independent units remain 33 Reach, 27 Reach-Wall and 21 Push-T
lineages, not cells, windows, or horizons.

![Registered BF16 layer response map](figures/layer_mechanism_bfloat16_native.png)

Singleton H6 proprioceptive-error reductions decline monotonically from B0 to B5
on both MetaWorld tasks in both numerical conditions:

| Task | BF16 B0 / B5 | FP32 B0 / B5 |
|---|---:|---:|
| Reach | 3.026% / 0.475% | 3.050% / 0.494% |
| Reach-Wall | 1.526% / 0.298% | 1.334% / 0.228% |

Original BF16 simultaneous B0-minus-B5 intervals are [1.769, 3.333] and
[0.243, 2.214] percentage points of native MSE. Reach B0 versus B1 remains
unresolved. Push-T does not reproduce the MetaWorld H6 benefit; its BF16 B0
proprioceptive error worsens 2.25% at H3 but is nearly native at H6.

This identifies intervention susceptibility, not a unique physics layer. Fits,
activation scale, delivered dose, and downstream computation vary with support.
H1–H2 precede the H3 intervention. The sweep does not test moving the later
rank-four operator to B0.

[All 576 cells](../paper/data/layer_mechanism_grid.csv) ·
[Matched-random map](figures/layer_mechanism_bfloat16_random.png) ·
[FP32 native](figures/layer_mechanism_float32_native.png) ·
[FP32 random](figures/layer_mechanism_float32_random.png).

## Candidate-specific versus common correction

For scenario s, call j and candidate i, decompose the logged, dose-scaled
four-component coefficient c(s,j,i) into its call mean and candidate-centered
residual. Split call means again into scenario mean and between-call variation:

> Total squared energy = candidate-centered energy + between-call mean variation
> + scenario-mean energy.

The identity is checked numerically, coefficient norms match requested edit norms,
and each arm hash links to its completed scenario report and immutable final report.
An orthonormal basis relates these coefficients to **requested activation edits**,
not to forecast error or goal costs.

| Task | Refined centered | Refined shared | Random-subspace centered |
|---|---:|---:|---:|
| Reach | 1.145% | 98.855% | 0.361% |
| Reach-Wall | 0.385% | 99.615% | 0.774% |
| PointMaze | 15.646% | 84.354% | 12.059% |
| Wall | 8.299% | 91.701% | 5.858% |

Each entry averages 96 scenario-level ratios, after summing energies across eligible
calls. The 768 protected arm records contain 40,320 candidate batches of 300; these
are not additional independent replicates. Marginal descriptive 95% intervals use
20,000 scenario-bootstrap draws, seed 20260913.

Shared does not mean temporally constant. Reach's refined energy splits into
25.04% changing call means and 73.81% scenario mean; Reach-Wall gives 11.76% and
87.86%. PointMaze's centered share falls from 41.28% at the first CEM iteration
to 1.76% at the last; random falls from 31.08% to 1.89%. This is compatible with
candidate convergence, not proof of controller collapse. Later action banks differ
between arms.

For squared goal cost, even a constant final-latent translation d changes cost by

> ||z_i+d−g||² − ||z_i−g||² = 2d·(z_i−g) + ||d||².

The first term varies across candidates. Our internal edit also passes through a
nonlinear predictor. Thus 99% shared coefficient energy does **not** imply 99%
decision-irrelevant information or explain protected rescues and regressions.

[Decomposition figure](figures/candidate_specificity.png) ·
[Hash-bound receipt](../paper/data/candidate_specificity.json) ·
[Scenario ratios](../paper/data/candidate_specificity_scenarios.csv).

## Decision-margin reanalysis

A separate September 10 development archive contains **192 scenarios × five arms
× 300 identical candidate actions**. Its archive SHA and all 192 record hashes
are verified; engineering records are excluded. Four non-native arms yield 768
paired comparisons under a [fixed reanalysis scope](../configs/decision_geometry_reanalysis_20260913.json).

For native winner k, native costs C and changes ΔC, each edited relative margin is

> (C_i−C_k) + (ΔC_i−ΔC_k).

Let m be the native best–runner-up margin and r = max(ΔC)−min(ΔC). If r < m,
every competitor remains more expensive. This is an exact sufficient finite-bank
certificate, not a statistical test; failure to certify does not imply a flip.
Ties retain lowest-index selection.
This diagnostic selects the bank's best candidate; production CEM updates from
ten elites. A stable best index alone does not certify an identical CEM update.

| Task | Arm | Certified / 96 | Changed / 96 | Centered RMS / native SD |
|---|---|---:|---:|---:|
| Reach | Refined | 94 | 0 | 0.247% |
| Reach | Random subspace | 95 | 0 | 0.212% |
| Reach | Coupling | 82 | 1 | 0.762% |
| Reach | Random directions | 92 | 0 | 0.346% |
| Reach-Wall | Refined | 90 | 1 | 0.279% |
| Reach-Wall | Random subspace | 89 | 0 | 0.292% |
| Reach-Wall | Coupling | 81 | 2 | 1.079% |
| Reach-Wall | Random directions | 88 | 1 | 0.475% |

RMS percentages average within-bank ratios. The certificate explains stable
**initial-bank** choices, not full iterative CEM or later replanning. The actual
MetaWorld objective is visual goal MSE **plus 0.1 × proprioceptive goal MSE**;
recorded-future proprioceptive error has a different target.

Refined cost changes have 47.5% / 48.4% centered energy here, unlike the protected
coefficient ratios. Different spaces, populations, and RNG prevent interpreting
their difference as amplification. Protected numeric candidate costs and elite
identities were not retained; hashes cannot recover them or establish replanning
drift as the cause of outcome changes.

Mean score-softmax entropy changes across the eight task/arm comparisons stay
within 0.0018 nats, using native within-bank cost SD as the shared temperature.
This is neither CEM proposal entropy, attention entropy, nor physical uncertainty.

[Figure](figures/decision_geometry.png) · [Receipt](../paper/data/decision_geometry.json) ·
[768 comparison rows](../paper/data/decision_geometry_scenarios.csv).
Marginal 95% scenario-bootstrap intervals use 20,000 draws, seed 2026091321.

## Geometry, scope, and next evidence

The [pathway and interpolation recount](PATHWAY_GEOMETRY.md) verifies 17 sources.
It separates output nonadditivity from squared-loss cross terms and raw
reconstruction from recorded-future error. Cubic reconstruction beats equal-anchor
linear in FP32 but loses in BF16 across five tasks; requested-dose-matched H6
advantages remain small and mixed. Precision-specific fits and realized doses
prevent attributing the contrast solely to rounding. Four symmetric anchors do not
identify third-order dynamics, a dense manifold, or model-native physical axes.

The separate GPU replay is complete on all 64 fixed development contexts. Common-only
learned corrections reconstruct the full candidate-centered cost change with scores
0.9827 / 0.9975 on Reach / Reach-Wall; calibrated-random corrections show the same
pattern. Components retain their natural energy. The separate eight-context actual-CEM
follow-up preserves all ten initial elites yet produces different returned prefixes
after adaptive search. [Pilot mechanisms](PILOT_MECHANISMS.md) reports the complete
attention maps, replay controls, proposal traces, intervals, and preservation boundary.
These measurements do not retrospectively recover missing protected traces or establish
better physical ranking. Attention pictures alone cannot identify a causal circuit.

One checkpoint per task, shared across the two MetaWorld tasks, supplies no
independent-training-seed replication. Whole-episode success factorials measure
closed-loop behavioral interaction, not same-input output interaction after
trajectories diverge. [Primary-source comparison](RELATED_WORK.md) distinguishes
policy steering, physics probes, manifold interpolation, adaptation, and memory
from the endpoints measured here.

## Reproducibility

[Fixed follow-up scope](../configs/mechanism_followup_20260913.json) and linked receipts
bind source, protocol, archive and record hashes. Public aggregates support CPU-only
figures; source replay requires authorized private archives and leaves scientific
records and fits unchanged.

~~~bash
python analysis/mechanism/steering_specificity.py
python -m analysis.mechanism.decision_geometry
python -m analysis.mechanism.pathway_geometry --plots-only
python -m unittest discover -s tests -p test_steering_specificity.py -v
python -m unittest discover -s tests -p test_decision_geometry.py -v
~~~

For the original layer recount, supply the coefficient script with
--recount-layers --results /path/to/results-v2 and the required archived sources.
