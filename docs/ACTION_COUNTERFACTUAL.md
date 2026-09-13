# Action counterfactual C: persistent history and action-range controls

The prospectively fixed experiment and independent analysis are complete:
16/16 registered development contexts, 1,120 full-H6 forecasts, and all
cloud-preservation receipts passed. Execution is a separate source-bound
development experiment, not a new protected test. The protocol and analysis
were fixed before complete-cohort outcomes were inspected.

## Results

All-block persistent condition replacement reproduced the coherent raw H3
action change byte-exactly through H6, with reconstruction R=1. Replacing
only the newest H3 occurrence at all blocks recovered R=0.5029–0.5079 across
the two tasks and two banks. Thus an H3-only internal patch is demonstrably
not the same manipulation as changing the literal H3 input action: the
condition also appears in the next predictor context.

The complete registered primary family gives the following pattern. “Positive”
means the simultaneous interval is entirely above zero; failure to clear zero
is not evidence of equality.

| H6 official contrast | Positive cells / 24 | Pattern across both tasks |
|---|---:|---|
| Persistent minus H3-only donor reconstruction | 16 | B0–B3 in both banks; B4/B5 intervals include zero |
| H3 donor minus norm-matched range-random reconstruction | 16 | B0/B1/B2/B4 in both banks; B3/B5 intervals include zero |
| Range-random minus off-range-random native rank loss | 10 | B1/B4 in both banks, plus B2 in the fresh bank; all others include zero |

The full curves, not a selected layer, are shown in
[the all-72-contrast figure](figures/paper_action_counterfactual.pdf).
The largest observed persistence contrast is at B1: ΔR=0.4991–0.5057 across
tasks/banks. B1 persistent single-layer reconstruction is R=0.7320–0.7421,
versus R=0.2286–0.2387 for its H3-only counterpart. This localizes an
intervention's effect in this predictor and cohort; it does not identify a
unique physical circuit. B3's donor-over-range-random contrast does not clear
the simultaneous interval despite being the location of the older rank-four
intervention. The new experiment does not retroactively justify that choice.

Range-random perturbations cause more native-ranking disruption than their
off-range counterparts at B1 and B4, but the difference is small in absolute
units: approximately 0.0021–0.0025 additional Spearman loss. B0/B3/B5 do not
show a simultaneous positive difference. The evidence supports layer-dependent
sensitivity to the literal action-encoder subspace, not a claim that all
off-range directions are irrelevant or all in-range edits improve planning.

The frozen action matrix has retained rank20 of400 in every case. Maximum
delivered relative norm error is 3.41e−8 (registered bound1e−5). Mean isotropic
in-range energy is 4.995%; projected range/off-range controls have mean
leakage about 3.74e−15/2.48e−16 respectively. These numerical checks establish
well-matched controls, not physical manifold membership. No primary cell is
undefined; negative estimates and intervals remain unclipped.

## Fixed scientific comparison

The [protocol](../paper/data/action_counterfactual_protocol.json), SHA256
`7c16653101ebc9a29bca82b88783e7ae44edac2a134733bc87dc0f6263141f72`,
retains IDs0–7 in each of Reach and Reach-Wall, the same original/fresh banks
as specificity B, and all six layers. Every bank has 35 full-H6 forecasts:
native, zero capture, coherent raw H3 action swap, all-block H3-only donor
patch, all-block history-persistent donor patch, and five treatments at each
of six blocks. Those five are H3-only donor, persistent donor, and donor-norm-
matched random perturbations in the action-encoder range, its orthogonal
complement, or the full ambient space.

Only the H3 action is swapped with the cyclic next candidate. In the actual
two-frame context window its condition appears at H3 local position1 and H4
local position0. Patching both occurrences at all six blocks must reproduce
the coherent raw-action counterfactual byte-exactly through H6; a H3-only
patch is not the same intervention. All arms must preserve H1/H2. Native/zero
and coherent/persistent full-horizon parity are engineering requirements, not
scientific outcome filters.

The literal frozen action encoder is affine Linear20→400. A float64 thin SVD
retains singular values greater than `1e-6*s_max`; the full spectrum, rank,
matrix hash and projector residuals are recorded. The three random controls
share one private Gaussian draw per candidate/layer/bank, projected before
separate donor-norm normalization. Delivered FP32 norms must match within
relative `1e-5`; actual in/off-range energy is retained. **Affine encoder-range
membership is not membership in a physical or global activation manifold.**

## Fixed endpoints and statistics

For each context, arm, modality and H3/H4/H6 endpoint, preserve all 300
candidate squared distances to the coherent counterfactual and all native-to-
counterfactual squared distances. The scenario reconstruction score is

\[
R=1-\frac{\operatorname{mean}_{i}\|z_i^{\rm patch}-z_i^{\rm cf}\|^2/N}
              {\operatorname{mean}_{i}\|z_i^{\rm native}-z_i^{\rm cf}\|^2/N}.
\]

This is a **ratio of candidate means**, not the mean of candidate ratios.
Negative scores remain negative. Zero denominators are explicitly undefined;
no epsilon, clipping, candidate removal, or complete-case context selection
is permitted. The official distance combines visual MSE plus 0.1 times
proprio MSE. H6 visual/proprio/official goal-cost arrays and actual elite IDs
are also preserved, with independently recomputed tie-aware native ranking
agreement.

The three primary contrasts are fixed at H6 under the official endpoint:

1. Persistent donor minus H3-only donor reconstruction.
2. H3-only donor minus range-random reconstruction.
3. Range-random minus off-range-random loss of native Spearman agreement.

The independent unit is a context, n8/task, not 300 candidates or two banks.
Twenty thousand bootstrap resamples use seed20260913 and the same context
weights across banks, layers and conditions. Each task/contrast family has
12 bank×layer cells. The maximum absolute **unstudentized centered bootstrap
deviation** supplies simultaneous bands; six families use quantile
`1−0.05/6`, covering all 72 primary cells at the stated simultaneous level.
Marginal percentile95 intervals and all secondary/modality/horizon results
remain descriptive. If any required context is undefined, the relevant mean
and full-family simultaneous band remain undefined rather than silently
shrinking the cohort.

## Independent verification and output schema

[The CPU auditor](../analysis/mechanism/action_counterfactual_summary.py)
requires exact16 task/episode paths and complete STARTED, scores, actions,
report, DONE and cloud receipts before aggregation. It verifies original input,
execution-source/protocol manifest bindings, physical UUID and cloud member hashes;
recomputes rank/elite and weighted-cost identities; checks candidate and
scenario reconstruction formulas; audits norm and projector-energy identities;
and verifies exact H3/H4 addressing and shared random draws. Full forecast
byte parity remains execution-attested because full tensors are intentionally
not retained. This boundary will remain explicit in the result receipt.

Completed compact CSVs are `action_counterfactual_case_metrics.csv` (whole-context
means and ratio scores), `action_counterfactual_norm_audits.csv` (each targeted
history occurrence), `action_counterfactual_summary.csv` (all arm/task/bank/
modality/horizon summaries), and `action_counterfactual_primary_contrasts.csv`
(all72 primary cells). The [final JSON receipt](../paper/data/action_counterfactual_summary.json)
binds every CSV, analysis source, execution source and cloud archive. No “best
layer” replaces the full table.

Source identities: execution manifest
`d2a364d761d34eddfe0c54ffac2c70f6f2b96c6d403857a3bf71d673f572030a`;
runner `efa4287173f453cf13869aeef3c9660ad37b9feb69bcd3bd61522e854a3977f0`;
frozen CPU auditor
`e1ee17c7c2c6d35061aa3cdf77ecdb3d656c619c5dca142be42aa482bc122f51`;
result receipt
`952826353a989493848f89b209852398295c0ae574c093dc7e1fb6a13bba2c3a`.
The separate plotting module reads only the hash-bound complete summary;
its [figure receipt](../paper/data/action_counterfactual_figure.json) preserves
the analyzer identity rather than replacing its frozen source to add plotting.

## Interpretation limits

A positive donor reconstruction result would concern one identifiable action
counterfactual inside the frozen predictor, not physical correctness of that
counterfactual. A range/off-range difference would concern a constrained
coordinate subspace, not a dormant circuit, emergence of planning, or automatic
benefit from rotations, training or test-time adaptation. Both positive and
negative results are retained. The original goal-cost planner remains external
CEM; no C plan is executed. These contexts were previously evaluated and are
not claimed fit-disjoint or fresh protected confirmation.
