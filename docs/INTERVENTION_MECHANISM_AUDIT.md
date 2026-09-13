# Intervention mechanisms: implementation and reusable evidence audit

Audit date: 2026-09-13. This is a code/receipt inventory and prospective analysis
proposal, not a new execution result. No GPU operations or new outcome-selected
statistical tests were performed for this audit. Current code was checked
against available frozen source bindings rather than treating historical prose
as execution evidence.

## 1. What is actually implemented?

| Mechanism | Actual operation and reference | What the name does not establish |
| --- | --- | --- |
| Static visual/action coupling | `operator_fit.fit_coupled_directions` fits a rank-one maximum-cross-covariance direction pair by centered power iteration. `PredictorIntervention` adds the visual direction to the newest predictor-input visual field and/or the action direction to newest B3 AdaLN condition, at H3. Native, zero, visual, action, joint, equal-standardized-energy joint, orthogonal-random and patch-permuted controls are registered. | Not a learned nonlinear transition model, latent rotation, donor-state replacement, or semantic causal factor. The full joint factorial and lower-dose equal-energy joint answer different questions. |
| Refined fixed rank four | `FixedResponseHook` computes three standardized projections of the native B3/H3 field, prepends an intercept, maps these four features through a fixed 4×4 coefficient map, expands in four fitted output directions, normalizes to a fixed L2 dose, and adds once. The control has its own fitted random-basis response map. | The pre-normalization map is affine. Normalization makes the delivered field state-dependent/nonlinear, but this is not an MLP, nonlinear system identification, coordinate rotation, or dynamics training. |
| Earlier support/rank/layer/spatial operators | `support_fit`/`support_operator` fit a PCA3/OLS H6 visual-residual target and support bases; the original support solver is separate from the later fixed-response map. `interventions` explicitly registers ranks 1/4/8, all six singleton blocks, B2+B3 and all-six supports with matched controls. | Layer-distribution results at total rank one do not retrospectively describe the later B3 rank-four operator. Orthonormal bases are not evidence that a rotation intervention was run. |
| Four-donor action geometry | `action_geometry.donor_actions` perturbs H3 normalized actions along a fixed fitting-bank direction; the same four anchors yield linear mean, cubic center, endpoint-line projection, and reflection about that projection. Raw center/native-fidelity errors are separate from residuals normalized to a common dose. | Reflection about an endpoint line is not a general latent-space rotation. The symmetric cubic-center functional is quadratic-compatible, so it does not identify a third-order response. Armwise norm matching changes the raw geometric construction. |
| Controlled geometry A | `controlled_geometry_pilot` uses literal action coordinate 0, offsets ±0.1/±0.05/0, all six H3 fields, one shared FP32 context, FP32/BF16-autocast/FP32-field-roundtrip conditions, and each condition's own center. No fit bank or dose normalization. | A numerical precision comparison around zero actions, not physical manifold discovery or held-out behavior. Raw fields are not saved; scalar metrics and centered-anchor Gram matrices are. |
| Cached component replay | `candidate_replay_pilot.components` splits the **delivered pre-addition correction** into its mean over 300 candidates and its candidate-centered remainder. `ReplayField` adds the cached component to byte-identical native B3/H3 inputs; full, cached-full and zero controls check equivalence. Components are not renormalized. | This is a controlled component intervention, not merely a regression coefficient attribution. Mean/centered pieces are across candidate corrections, not across spatial patches or physical trajectories. Cost additivity cannot be assumed from field additivity. |
| Action-conditioning specificity B | `action_condition_specificity.edited_condition` cyclically interchanges the newest AdaLN condition vector between candidates **within the same context/action bank**, at one H3 block. Random perturbations match each recipient's actual donor-minus-recipient L2, including zero norms. All six blocks and original/freshly seeded banks are retained. | This is real donor interchange, but donor vectors are not sampled from different episodes or semantic labels. Recipient action inputs remain unchanged. One-block replacement can create inconsistency with the other five blocks; marginally native donor vectors do not guarantee jointly on-distribution internal combinations. |

Implementation references:
[static hook](../src/offline_study/interventions.py),
[coupling fit](../src/offline_study/operator_fit.py),
[fixed response](../src/offline_study/fixed_response.py),
[support fitting](../src/offline_study/support_fit.py),
[geometry](../src/offline_study/action_geometry.py),
[geometry fitting contract](../src/offline_study/geometry_fit.py),
[cached replay](../src/offline_study/candidate_replay_pilot.py),
[specificity B](../src/offline_study/action_condition_specificity.py), and
[controlled A](../src/offline_study/controlled_geometry_pilot.py).

**Rotation finding:** the active `src/offline_study` implementation has no general
orthogonal latent-rotation treatment. Searches of active source/analysis/scripts
found plot-label rotations, a description of changes in correction direction,
and DROID physical Euler-rotation validation, none of which is an executed latent
rotation arm. This bounded audit does not certify every historical archived
experiment. Patch permutation, random-direction projection, reflection, donor
interchange, and rotation must not be used as interchangeable labels.

## 2. Cohorts and endpoints that must remain separate

1. **Recorded offline trajectories:** teacher observations supply scoring targets;
   recursive forecasts use the original recorded action sequence. The same-input
   visual/action/joint decomposition is valid within a frozen task/precision
   source. Different task-specific checkpoints, fit banks, doses and precision
   conditions are not randomized factors in the older cross-source comparison.
2. **Previously evaluated developmental simulator contexts:** the canonical
   Reach/Reach-Wall scenarios underlie the fixed-bank decision diagnostic and the
   64-context attention/replay/A pilots. A fresh action bank in B is action-bank
   holdout, not a fresh episode or fit-disjoint confirmation population.
3. **Prospective protected four-task panel:** `fresh_confirmation` freezes a new
   audited source/input bank, 96 scenarios per task and all eight arms on the same
   physical device per scenario. It resets the planner's scenario-specific RNG
   for each arm. The historical development streams used a different RNG
   organization. Thus development-to-protected differences combine population,
   stochastic-stream, and possibly device/receiving differences; they cannot be
   attributed uniquely to overfitting or OOD states.
4. **Author-reported benchmarks and DROID:** neither is a paired reference for
   our protected native arm. DROID's recorded-action endpoint is not physical
   robot success. Base-model training exposure and intervention-development
   exposure are separate properties: “protected” does not certify unseen world-
   model pretraining data.

Within a single protected scenario, arm-wise outcome contrasts are valid as
frozen paired comparisons. After arms choose different actions, subsequent
observations are different. Matching their seventh planner call, horizon, or
candidate index does **not** create a same-context latent factorial. For output
interaction one needs identical incoming context, action bank and target across
native/visual/action/joint, as in the offline factorial—not merely the same
original episode seed. See [fresh protocol implementation](../src/offline_study/fresh_confirmation.py).

## 3. What is archived, and what is missing?

| Evidence source | Reusable, source-bound fields | Unsupported reconstruction from those fields alone |
| --- | --- | --- |
| Offline pathway sources | Paired per-lineage native/visual/action/joint future MSEs, output-interaction MSE, additive quadratic cross term, full loss interaction/remainder, raw and dose-normalized geometry diagnostics; fit/source/precision identities. | New arbitrary rotations or donor substitutions; new latent density/OOD scores without original features/reference distribution; per-window effects from already lineage-aggregated rows. |
| Fixed-bank 192-context decision diagnostic | Candidate action/cost records, selected IDs, native margins, score-change range and centered RMS, top-ten/rank diagnostics. | A candidate score is not its physical counterfactual value. Choice stability is not proof of latent irrelevance or later closed-loop equivalence. |
| 64-context development replay | Same input/goal hashes; original 6×300×20 actions; native and component objective-cost vectors/elite IDs; full-arm cached rank-four coefficients; actual native 15-iteration CEM candidate/cost/proposal traces; attention distances on candidate-zero; per-case common/centered energy and H6-change summaries. | Complete H1–H6 latent trajectories are not archived. **H6 visual/proprio tensors are saved only for IDs0–3/task (eight cases); later IDs retain scalar/score/coefficient evidence.** Native B3 full fields/correction tensors are not retained by the expansion save routine. |
| Protected closed-loop arm records | Initial/goal/scenario hashes, physical UUID/source freeze, actual action trace and observation digests, result, call schedule/timing, requested/realized edit norms and fixed-response coefficients. | No archived full 300-candidate visual/proprio forecasts or goal-cost arrays per call in this runner. No pre-edit latent field, training-density score, or calibrated support distance. A hash is an identity check, not a geometric observation. |
| Controlled A | All 64×6×3 center MSEs, 4×4 float64 centered-anchor Gram matrices, donor response RMS, unchanged fractions, finite differences, source/input/DONE/cloud identities. | Gram matrices preserve within-anchor inner products, not global field orientation, individual coordinates, latent density, or physical targets. |
| Specificity B | Original and fresh 300-action banks, all native/12 edited per-bank cost vectors for visual/proprio/weighted goal endpoints; native/edited condition hashes/shapes; per-candidate donor and realized/random norms; exact byte/RNG controls. | Condition hashes and norms do not reconstruct condition vectors or Mahalanobis support. The saved forecast scores do not support a new intervention without another forward. |

Source links: [pathway lineage table](../paper/data/pathway_geometry_coupling_lineages.csv),
[decision table](../paper/data/decision_geometry_scenarios.csv),
[development save routine](../src/offline_study/development_mechanism_replay.py),
[protected call recorder](../src/offline_study/fixed_response_behavior.py),
[A source receipts](../paper/data/controlled_geometry_summary.json), and
[pilot source receipts](../paper/data/pilot_summary.json).
“All raw preservation complete” means all files that the corresponding protocol
actually emitted are preserved; it does not imply unrecorded H6 tensors exist
for the later 56 cases.

### Source checks made for this audit

The expansion manifest binds current `development_mechanism_replay.py` SHA256
`1e7317917e8c435fa2d8740bfaf47cabe82411e7e9214a351c9a4e0a52b1e569`;
the original eight-case manifest binds a distinct earlier version. Both bind
`candidate_replay_pilot.py` as
`a2f72320b705fce8d0cb3f9fe93bc0f39f06c37fef49d4e5c0f86ff71638b571`
and `fixed_response.py` as
`949b76600ad2752d6a298e91fb437b57cfcabd02ce7cf864ae3aa4ac277b0bec`.
The A manifest `b27dd0e0409fab9e6ad1251281edb7456a78e95878978fac2781ce1979108e9c`
binds the controlled module
`4869cf1957bd3fdd3ac14f7f31a30b513fe72a4d41c9e51a270ce0bed27c3f32`.
The public B receipt reports complete16 with eight contexts per task, manifest
`4bf08978fb5ae7da9a971527efd796035fda82dacfd44028e2acaff44b526341`;
this audit inspected its completion metadata, not its outcome contrasts.

## 4. CPU-only analyses proposed before execution

No new test in this section has been run. These are prospective, bounded
proposals—not completed evidence or automatic authorizations.

**Strongest available CPU decomposition:** on all eight *already archived*
full-H6 replay contexts, compare common-only and centered-only interventions
against their full correction. Fix both learned/random arms and all 300
candidates. For each visual/proprio endpoint compute the output interaction
`r = z_full − z_common − z_centered + z_native` and additive quadratic cross
term `2〈z_common−z_native, z_centered−z_native〉/N`. Use the fixed weighted
sum of modality terms for the actual goal-cost metric. The saved objective
vectors give `D = C_full − C_common − C_centered + C_native`; its nonadditive
remainder is `D − cross`. Also retain weighted `||r||²/N`, which is not equal
to that signed remainder. Aggregate candidates within context, then all eight
contexts (n4/task, descriptive). This needs no newly encoded goal: `D` is already
saved and the cross/interaction vectors use the archived forecasts.

Do not generalize this tensor subset to n32/task or to physical futures. The
scalar64 results can contextualize, but cannot fill unavailable modality
tensors. Separately splitting **D itself** into visual-target/proprio-target
terms requires verified target embeddings or exact saved modality cost arrays;
raw initial/goal images alone cannot supply encoded targets by CPU algebra.

**Secondary norm diagnostic, only if authorized:** use B's full registered
16-context/two-bank/six-layer/permutation-and-random registry. Predefine the
within-context relationship between donor L2 and absolute candidate goal-cost
change, and the paired permutation-minus-random residual at matched norms.
Keep zero-norm candidates explicit and bootstrap contexts, not 300 candidates.
This assesses norm sensitivity, **not** in-distribution membership or a causal
mediator. No threshold or chosen layer would be tuned to the resulting curve.

Do not label coefficient magnitude or correction norm an OOD score: fixed-dose
normalization largely fixes the latter, while coefficients describe a fitted
operator coordinate system without a calibrated training-reference density.
Reconstructing pre-normalization latent scores from normalized coefficients is
generally underdetermined. A real support audit needs a frozen reference feature
sample, explicit layer/precision metric and held-out calibration, none of which
is supplied by an observation hash.

## 5. One strongest cheap causal follow-up

**Coherent H3 action interchange versus one-block condition mismatch.** Keep
the already-fixed B16 contexts, both banks, and the same cyclic donor map; do
not choose cases/layers after B's result. In addition to B's six single-block
swaps, run (i) the donor H3 action condition consistently through all six blocks
at **every context occurrence of that action**, (ii) the corresponding donor
**H3 raw model-input action** with the
recipient H1/H2 prefix unchanged, and (iii) a preseeded common condition
perturbation, coherent across blocks and matched to each recipient's donor
change norm. Compare the same all-candidate visual/proprio/weighted goal-cost
vectors and native rank/top-ten agreement, with whole-context pairing.

The pinned two-frame unroll window contains that action at H3's newest position
(local index1) **and H4's older position (local index0)**. A newest-only H3 swap
does not implement the same intervention at H6. The pinned AdaLN forward uses
a pointwise affine action encoder and passes the resulting `z` to every block.
Therefore (i) versus (ii), after verifying exact action-encoder
mapping and no other action pathway, supplies a strong computational
consistency/parity check. Their relationship to existing single-block swaps
tests whether B primarily measures coherent action substitution or conflict
between a donor condition at one depth and recipient conditions elsewhere.
The matched coherent random arm controls generic magnitude. This is roughly
three new 300-candidate H6 forwards per context/bank (96 total), plus mandatory
native/parity overhead; actual timing and budget need the deployment owner.

This is more directly grounded than an arbitrary latent rotation: native donor
conditions have a known action counterpart, although the resulting internal
state/action pairing still need not occur naturally. It cannot prove physical
counterfactual accuracy, planning emergence, or benefit from gradient training.
No launch follows from this proposal. A later rotation study would require an
explicit subspace, orthogonal transform, norm/support controls, and new
prospective execution; no existing rotation result should be implied.
