# Frozen-model offline study: active plan

Status (2026-09-07, approximately 11:04 EDT): the corrected primary offline phase is
complete, verified across all 30 task/category/precision scopes: 33 Reach, 27 Reach-Wall,
and 21 Push-T trajectories. All six refits exclude validation families; the seven added
MetaWorld rows and 21 Push-T rows were separately authorized without altering their
historical exposure registries. The original custom runs remain historical development
evidence. Broad unsteered coverage remains 1,120/1,260 rows across 42 MetaWorld tasks.
This is full **primary-task released-pool offline replication**, not reconstruction of
the authors' training histories, full 42-task validation coverage, or end-to-end study
completion. Combined/drop-one work and prospective policy evaluation remain unfinished.
See [corrected results and exact completion evidence](../reports/CORRECTED_OFFLINE_RESULTS.md).

At approximately 11:52 EDT, the predeclared Reach combined/drop-one stage launched
on all eight US GPUs after both precision-specific fit-only checks passed. Its exact
ten arms, source component bindings, complete 33-row previously exposed population
and fixed analysis are recorded in [COMBINED_DEVELOPMENT.md](COMBINED_DEVELOPMENT.md).
This does not open another untouched reserve or establish closed-loop efficacy.

## Author-validation correction requested on 2026-09-07

The initial committed version of this plan (`8f07ad2`, September 6 at 23:25 EDT)
already prescribed a custom split, four windows and planning-wrapper context. Following
that plan did not fulfill the subsequently reiterated request to match the authors'
offline validation. The lineage corrections did not remove this protocol mismatch.
The original settings below remain documented as the legacy study, not quietly replaced
in already frozen protocols or result provenance.

The correction is tracked in [AUTHOR_VALIDATION_AUDIT.md](../reports/AUTHOR_VALIDATION_AUDIT.md)
and `configs/study.json:author_validation_alignment`. The metadata-only
`offline_study.author_validation` command reproduces the published split and slice order,
with direct tests against the pinned upstream implementation. It counts source rows,
lineage groups, clips and rollout prefixes separately and identifies fit overlap and
protected cohorts. It does not execute a model, grant exposure permission or claim parity
of preprocessing, rollout metrics, statistical analysis or original training histories.

Before replacement evaluations: freeze the matched evaluator and existing intervention
arms, verify preprocessing/context/metric parity, refit without any validation-family
overlap, and verify cohort access against the preserved exposure registry. Protected
trajectories cannot be relabelled development to bypass these checks. Do not silently
drop protected trajectories and claim complete official validation coverage.

The authors' full available validation pool has 1,260 MetaWorld rows (33 Reach and
27 Reach-Wall) and 21 Push-T rows. A complete pass over that pool is a stated extension
of their batch-wise validation monitoring, not a reconstruction of the exact samples
behind a published curve. Author-split replication and untouched confirmation remain
different claims. The legacy `authors_split_reproduced: false` remains correct until
new executable parity checks and completed run receipts support a narrower claim.

### Protected-preserving execution amendment

The active replacement settings are [author_correction.json](../configs/author_correction.json).
They supersede only the legacy execution settings below, not historical result records.
Use the published normalized dataset/transform, context capacity three, H6 rollout at
every valid prefix of every included clip, and official visual/proprioceptive embedding
L1/L2 losses at H1–H6. MetaWorld clips have 18 frames; Push-T clips have eight; both use
frame stride five and concatenated recorded actions. bfloat16 is the published-precision
primary analysis and FP32 is a separately reported sensitivity analysis; neither may be
selected based on favorable outcomes. Direct actual-upstream rollout and metric parity
was checked on fit-only examples in both precisions for all three primary tasks.

Each task/precision refit uses 128 previously eligible fitting families, four fixed
clip/prefix examples per family, excluding **every** official-validation lineage and
every protected lineage. The five fixed sweeps retain their registered arms, controls,
fitting algorithms and dose rules. There is no autonomous successor search. Report
requested and realized edit energy because low-precision rounding can prevent exact
equal-delivered-energy comparisons.

The initial protected-excluded replacement covered 29 of 33 Reach rows and 24 of 27
Reach-Wall rows. At that stage, four and three protected rows respectively stayed
unopened; the subsequently authorized additive completion is documented below.
The broad unsteered
baseline includes 1,120 of 1,260 official-validation rows across all 42 MetaWorld tasks,
excluding 140 protected rows. Label these partial author-split development replications,
never complete official validation or untouched confirmation.

The verified Push-T registry marks **all 21 released validation rows protected**, even
though historical evidence records earlier access. The earlier metadata-only intention
to evaluate 21 rows did not grant permission. The original intention is preserved by
hash in the replacement config; the bound executable cohort has zero evaluation rows
and all 21 exclusions. Both Push-T precision-specific fits can be completed safely now.
Opening validation requires a separately verified, frozen access decision; changing a
split label is not an acceptable substitute. This does not make those 21 rows fresh
confirmation, nor does it invalidate the preserved 36-family custom development runs.

Subsequent explicit authorization is now implemented in a separate run root,
`/workspace/jepa-runtime/pusht-author-replication-20260907`. Its access receipt SHA256 is
`a80b37516e5f95f6de649adb7004410f680a6c8ee860a927ddd948310314e559` and authorized cohort
SHA256 is `2376af7881b4e411e8a726f178b16ead707551c5b691fe1c1dfa38553db9b9ed`.
All ten task/precision/category protocols were already frozen before this opening;
the original fits and protocols are retained and no operator is refitted from validation.
This grants the recorded full fixed replication sweep, not fresh confirmation or
permission to alter arms/doses after outcomes. The seven primary MetaWorld protected
rows remain unopened. Next-stage selection follows the existing development rule and
the user's later explicit autonomous-advancement instruction; selection evidence is
not independent confirmation.

Further user approval now authorizes the seven primary MetaWorld rows for the same
frozen replication sweeps. The append-only extension run is
`/workspace/jepa-runtime/metaworld-author-extension-20260907`: execute only the four
missing Reach and three missing Reach-Wall trajectories, then verify original fit,
protocol, shard hashes and complete paired coverage before merging with the original
53 rows. Full primary coverage becomes 33/33 Reach, 27/27 Reach-Wall and 21/21 Push-T
only after those receipts complete. No refitting or additional arm search is permitted;
these seven cease to be an untouched confirmation reserve after replication access.
Other protected MetaWorld rows remain outside this authorization. See
[planning methodology alignment](../reports/PLANNING_METHOD_ALIGNMENT.md) for the
separate simulator, episode-count, seed and fresh-confirmation requirements.

The primary forecast endpoint remains H6 proprioceptive **embedding** MSE. Also report
visual embedding MSE and L1/L2 at every horizon. Point estimates include both
clip-weighted and lineage-weighted means. Intervals use paired lineage bootstrap draws
with simultaneous correction over all registered contrasts and both H6 MSE endpoints
within task/category/precision; the minimum useful effect is 1% of the corresponding
fit-only native error. These runs do not reproduce the authors' training-curve history,
three trained seeds, decoded physical-state errors, or closed-loop task-success results.

## Objective and task coverage

Determine whether interventions in a frozen action-conditioned world model have
reproducible, selective effects on forecasts, and whether useful effects survive
later candidate-ranking and closed-loop evaluation.

Primary task suite:

| Task | Added difficulty | Physical measurements for later validated readouts |
|---|---|---|
| MetaWorld Reach | Robot motion toward a goal | End-effector displacement and goal distance |
| MetaWorld Reach-Wall | Motion constrained by an obstacle | Displacement, goal distance, wall clearance |
| Push-T | Contact, translation, rotation | Pusher/object pose, angle, contact, target overlap |

The broad unedited offline baseline covers all 42 MetaWorld tasks and Push-T.
Initial intervention comparisons cover the three primary tasks. Cross-task analyses
use appropriate labels per task, and report per-task results rather than treating
1,260 mixed-task trajectories as 1,260 Reach-Wall trials.

## Fixed system

- Official jepa_wm_metaworld and jepa_wm_pusht checkpoints; different task-trained
  predictors are not interchangeable. Record local checkpoint SHA256.
- Upstream source pinned to `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.
- DINOv2 ViT-S/14 encoder, 12 blocks; 224x224 images and 256 patches.
- Six-block AdaLN+RoPE predictor; weights frozen; proprioception enabled.
- FP32 is the initial parity/throughput baseline. Mixed precision is a separate,
  parity-checked performance optimization, not silently introduced between arms.
- H6 recorded-action rollout; five elementary actions per model action; measure H1/H3/H6.
- One observed starting frame; subsequent context is the model's own predictions,
  with the official wrapper retaining up to two temporal positions.
- Future recorded observations are encoded only as scoring targets. Never supply
  them to the recursive predictor in this open-loop offline check.
- No planner or simulator is called. Native CEM/L2 settings become relevant only
  during the later candidate-ranking/control phase.

## Dataset scale and splitting

Reported dataset targets: MetaWorld 12,600 episodes across 42 tasks; Push-T 18,500
replays. These are targets to reconcile with downloaded assets, not hard-coded
assertions that the available files contain those counts.

Important source correction: upstream MetaWorld code uses torch.randperm to split
whole trajectories. Upstream Push-T code opens separate train/ and val/ directories;
its split_ratio parameter does not create a 90/10 partition. Therefore our earlier
claim that both released loaders automatically produce 90/10 was incorrect.

The requested study design is an outer 90% fitting / 10% provisional holdout split,
with 10% of the outer fitting pool reserved internally for development (approximately
81/9/10 overall). The inventory names this `study_hash_90_10_with_inner_development`,
not an exact reproduction of the paper's split. It uses deterministic SHA256 ordering
with seed234. Splits are assigned to lineage groups, never individual correlated rows.

### MetaWorld duplicate correction and exposure status (2026-09-07)

The official MetaWorld release contains 12,600 rows, but 160 are redundant numerical
copies: 156 lineage groups contain two rows and two groups contain three. All members
of each such group have byte-identical float32 state and action tensors. The corrected
inventory therefore defines a MetaWorld lineage group by SHA256 of task plus exact
state/action tensor contents, yielding 12,440 independent numerical trajectory groups.
Video bytes are deliberately not part of this identity claim.

The earlier baseline assigned the deterministic split to row IDs. Re-hashing all new
content IDs would reshuffle every row and could falsely relabel measured development
data as holdout. The correction instead reconstructs that exact legacy row split and
coalesces each duplicate group conservatively: development wins because it has already
been measured; otherwise an untouched holdout assignment wins over unused fit. The
result is 10,056 fit, 1,125 development, and 1,259 provisional-holdout groups. Historical
protections exclude six development groups, leaving 1,119 reviewed development groups
for the corrected all-task baseline. Reach contains 295 total groups and 28 reviewed
development groups; Reach-Wall contains 297 total groups and 24 reviewed development
groups. Details and receipt hashes are in
[the MetaWorld lineage correction report](../reports/METAWORLD_LINEAGE_CORRECTION.md).

For Push-T, the released train data contains exactly 185 initial-state families of 101
distinct action/state rollouts each. A family is defined by the SHA256 of its exact
initial state; this was verified directly from the released tensors. The resulting
structural partition is 149 fit, 17 development, and 19 holdout-labelled families. The
supplied val set contains 21 additional initial-state families with no exact initial-state
overlap with train and remains an external reserve. Counts are reported separately.

### Push-T correction and exposure status (2026-09-07)

The first Push-T inventory incorrectly applied that hash split to individual rollout
row IDs. Its development baseline consequently included 1,682 rollouts drawn from all
185 train families. The run remains valid for throughput and descriptive per-rollout
forecast error, but it is not a family-independent efficacy sample or power calculation.

This exposure cannot be undone by recomputing a group split. For this study, the entire
released Push-T train pool is now development-exposed at the family level. In any
correction receipt, the 19 `holdout` labels record the deterministic structural split
only; they are **not** evidence of untouched confirmation eligibility. Before any
intervention execution, those 19 families are reassigned to development. The active
released-data analysis split is therefore 149 fit families, 36 development families,
and zero confirmation families (15,049/3,636/0 rollout rows). This keeps fitting and
development separate without discarding already exposed data. The supplied val
pool was accessed by historical work and likewise is not silently promoted to a fresh
confirmation cohort. Push-T confirmation therefore requires newly collected independent
initial-state families with a prospectively frozen manifest. Until those exist, Push-T
is development/replication evidence only. MetaWorld confirmation remains governed by
its separately audited, protected trajectory split. Exact affected receipt hashes and
implemented safeguards are recorded in
[the Push-T lineage correction report](../reports/PUSHT_LINEAGE_CORRECTION.md).

No tuning or efficiency benchmarking on a genuinely provisional holdout. Previous exposure must
be reconciled using source trajectory IDs before any confirmation claim. Previously
protected legacy trajectories remain protected even if the new hash split would place
them in development: the exposure registry must filter them before real execution.
The benchmark requires a manifest-bound exposure registry with a review basis and
per-trajectory evidence of permitted development use; unreviewed IDs fail closed.
Creating the mapping still requires the downloaded source: old and new row IDs must
be linked through source revision/row/task/collection seed.

Record separately whether the world model was trained on a trajectory. An intervention
holdout can still have been seen during base-model training. No claim of unseen-world-
model generalization follows from the intervention split.

Push-T rows within an initial-state family are repeated observations, not independent
generalization units. Aggregate windows within rollout and rollouts within family;
cluster uncertainty and power calculations by the 185 verified families. The released
files do not identify these families as source demonstrations, so call them
initial-state families rather than making a stronger provenance claim.

MetaWorld likewise aggregates windows within released row and rows within exact
state/action lineage group. Exact duplicate rows do not increase statistical sample size.

Four evenly spaced valid H6 windows per trajectory is the initial throughput budgeting
assumption. All eligible trajectories are retained; too-short trajectories and fewer
than four distinct windows are explicitly reported. Windows can overlap and must be
aggregated within trajectory. Expanding windows does not increase independent n.

## Design-choice selection method

Adopt the clean structure of the JEPA-WM design-choice study without copying its
training-specific assumptions. For each category, define a finite candidate registry
and a category-specific reference protocol, vary only that category's named factor,
and hold checkpoint, inputs, actions, fitting population and algorithm, evaluation
windows, hook/time/spatial scope, operator capacity, and perturbation energy fixed
unless one of them is the factor under study. Evaluate every registered arm on the
same paired development trajectories; do not stop or add arms after inspecting results.

Run the choices in this order: vision-action substrate, action-response geometry,
operator rank, layer distribution, spatial distribution, and finally temporal routing.
Temporal routing is eligible only after a non-routed operator has a reproducible effect
and the fit data establish incremental history value and treatment separation. Each
sweep uses its frozen category reference rather than silently inheriting a favorable
setting found in another sweep.

Selection is per task/checkpoint; a mixed-task average may summarize but cannot choose
an arm. An arm is eligible only if it passes identity/fidelity checks, improves the
predeclared recorded-future endpoint by the smallest useful effect, and beats its
scope/rank/energy-matched random control. Estimate contrasts from paired per-lineage
aggregates and use a predeclared familywise simultaneous interval within each category
and task/checkpoint. "Equivalent" means that interval lies wholly inside a frozen
equivalence margin; failure to reject a difference is not equivalence. Among equivalent
eligible arms, choose the simpler one. Otherwise an unresolved comparison is
inconclusive, and if no arm is eligible retain native/no intervention. After the
one-factor sweeps, combine compatible selected choices on development and run drop-one
ablations. Freeze that combined recipe, controls, endpoints, multiplicity handling,
and sample size before one protected evaluation. Closed-loop success remains the final
primary outcome; offline forecast and mechanism measures are advancement gates.

## Agreed ablation table

| Category | Arms | Primary mechanism comparison |
|---|---|---|
| Vision–action coupling | No edit; visual-only; action-condition-only; joint; equal-standardized-energy joint; spatially permuted joint | Joint versus individual edits; nonadditive output interaction; equal-budget efficacy |
| Action-response geometry | Equal-anchor linear; cubic; cubic projected onto endpoint line; reflected curvature | Contribution of off-line curvature beyond data quantity and line reparameterization |
| Operator dimensionality | Rank 1; rank 4; rank 8; rank-matched random subspaces | Added value of distributed signal beyond a single direction at fixed support and energy |
| Routing across imagined time | Constant gate; memoryless gate; HMM-filtered gate | Conditional HMM versus memoryless using the same underlying operator and comparable dose |
| Spatial distribution | One patch; contiguous group; equally sized scattered group; all patches | Concentration at fixed block/time and total perturbation budget |
| Layer distribution | Every singleton B0–B5; intermediate B2+B3; all six blocks | Single-layer localization versus the intermediate zone and global depth distribution, at fixed spatial scope/time, rank, and energy |

The last two rows form one diagnostic category. Do not cross every row into a large grid.
All categories include zero-dose identity and suitable norm/scope-matched random controls.

### Vision–action coupling

Action-only means editing the internal action-conditioning representation, not changing
the provided recorded action sequence. The factorial must include native, visual, action,
and joint arms with identical inputs and individual component doses.

Measure output interaction r = z_joint - z_visual - z_action + z_native before squaring
errors. A nonzero interaction of MSE scores can occur even for additive outputs, so
decompose that quadratic cross term separately. Joint edits have more total perturbation
than a single component under a factorial. Therefore include a separate joint arm with
both standardized component doses scaled by 1/sqrt(2), so their squared standardized
dose sums to one single-component budget, plus its own matched-random arm. Use the
unscaled factorial for the interaction estimand and the equal-budget arms for efficacy;
do not substitute one comparison for the other.

Reuse candidates: archive/.../scripts/geometry_map/run_joint_visual_action_factorial.py
and run_joint_factorial_64.py. Donor/reference selection, exact hook, signed doses,
and action embedding reconstruction must be pinned before the new scientific run.

### Action-response geometry

Use the same four noncentral action-response anchors for linear/cubic fits; omit the
evaluation coordinate from fitting. Compare projected cubic and reflected curvature
to distinguish speed along a line from a correctly oriented curved deviation.
Report omitted-activation reconstruction and downstream native-forecast fidelity
separately from actual recorded-future error. A closer reproduction of the model
is not automatically a better prediction of physical reality.

Changing action inputs to collect donor responses is permitted within this diagnostic,
but only the original recorded action has a recorded physical future. Never treat that
future as ground truth for arbitrary perturbed actions. New counterfactual physical
targets require later simulator forks or existing matched outcome banks.

Reuse candidate: archive/.../scripts/geometry_map/action_path_geometry.py.
Anchor radius/directions, target coordinate, site, and dimensionality remain to freeze.

### Operator dimensionality

At the category reference site P3/H3 with full spatial support, compare ranks 1, 4,
and 8. The rank set is bounded and motivated by the archived distributed-code evidence;
it is not an invitation to search arbitrary ranks. Fit every rank with the same target,
algorithm, fitting groups, and regularization rule. Match total delivered squared L2
energy across ranks. For each rank, use a random subspace with the same support, rank,
spectrum, and energy. The primary question is whether rank 4 or 8 improves the
recorded-future endpoint beyond rank 1 and its own matched-random control. Control the
rank-contrast family jointly. If a higher rank is equivalent to rank 1 under the frozen
margin, select rank 1. Rank 1 is also the fixed category-reference capacity for the
separate layer and spatial sweeps; a higher-rank winner is introduced only in the later
combined development recipe, not retroactively into those one-factor comparisons.

### HMM across imagined time (agreed choice)

The first comparison uses native unedited imagined H1/H2 activations to gate an edit
at H3. Each candidate rollout starts a new filtered belief; no sharing across candidates,
CEM iterations, or real episodes. No future smoothing at evaluation. The constant,
memoryless, and HMM arms share the edit and fitted emissions where applicable.

Training may fit emissions/transitions to training imagined sequences, but the online
H3 decision sees only its prefix. Keep routing features on a native shadow rollout in
the initial experiment to avoid mixing intervention feedback with the value of history.
Count that additional model work in ablation throughput. The underlying edit must
have a stated target; adding a gate to an ineffective edit need not rescue it.

Reuse candidate: archive/.../scripts/geometry_map/run_reach_hmm_routing.py.
Regime count, fitted operator, gate rule, and comparator budget remain to freeze.

### Spatial and layer distribution

Treat depth and token position as two axes of one localization question, but run them as
separate one-factor sweeps rather than a full layer-by-position grid.

The layer sweep fixes the block-output hook, H3, full 256-patch support, semantic target,
fit procedure, total direct-sum rank at 1, and total perturbation energy. It compares all six
zero-indexed singleton blocks B0 through B5, the archived evidence-derived intermediate
zone B2+B3, and the global B0+B1+B2+B3+B4+B5 arm. Fit a basis in the direct sum of the
registered block spaces and split it into layer-specific slices; do not copy one vector
across blocks. This keeps total operator rank fixed even when support spans more layers.
Every layer-support arm has its own random control with identical block support, rank,
spectrum, and energy. Early edits change later activations, so all multi-block edits are
constructed from and logged against the same specified native reference.

The spatial sweep fixes P3/H3, rank 1, target, and fitting procedure. Select the candidate
single patch and 4x4 contiguous region on fitting data, then freeze them. Compare one
selected patch, the selected contiguous 16-patch region, an equally sized scattered
16-patch set, and all 256 patches. Each support has its own random-direction control with
the same positions, rank, spectrum, and energy. The fit-selected one-patch, contiguous,
and scattered supports additionally have random-position arms preserving cardinality
and topology; the all-patch arm has no position analog. Share total perturbation energy
sum_{block,patch} ||delta||^2 and report downstream effect magnitudes. Control the
predeclared support-contrast family jointly.

A singleton supports layer localization only if it beats its matched random, the
singleton contrast family isolates it from the other blocks, and it is equivalent to
the broader supports within the frozen margin. B2+B3 beating both B2 and B3 by the
smallest useful effect supports an intermediate distributed zone. All six beating the
best singleton and B2+B3 by that effect supports global depth distribution. Analogously,
localized spatial support must beat its direction and position controls and be equivalent
to broader support; all-patch support must improve over the best localized arm to support
spatial distribution. Any wide interval that spans these decision regions is
inconclusive. After both sweeps, run only a secondary development 2x2 compatibility
check crossing the rule-selected best non-global versus global layer support with the
rule-selected best non-global versus all-patch spatial support. Do not launch the full
layer-by-position Cartesian grid or treat this post-selection check as independent
confirmation.

## Evaluation order and scientific claims

1. Baseline throughput on development trajectories from Reach, Reach-Wall, and Push-T.
2. Broad unedited H6 error coverage after inventory/exposure reconciliation.
3. Fit one bounded, explicit operator protocol per category on fitting data.
4. Run every registered arm in each one-factor development sweep with paired trajectory
   aggregation; apply the frozen eligibility and parsimony rule.
5. Combine compatible selected choices on development and run predeclared drop-one
   ablations plus the minimal layer-by-position interaction check.
6. Freeze the combined recipe, primary contrasts, smallest useful effects, sample size,
   multiplicity handling, and analysis.
7. Open verified untouched offline evaluation once. For current MetaWorld data this
   means its protected trajectory split; for Push-T it requires prospectively collected
   independent families because the released train/val pools are development-exposed.
   Later physical forks and closed-loop confirmation assess behavior, not just embedding error.

Use continuous per-rollout differences aggregated within lineage family, confidence
intervals over independent families, and multiplicity handling for the predeclared primary contrasts.
Estimate variance from development data; never set n by copying total training size or
by interpreting thousands of windows as thousands of independent samples. No power
claim is currently justified by the old small-state pilots.

## Runtime benchmark and records

Start with one GPU and a bounded development sample; then test two before eight.
Do not assume linear scaling. Measure model setup, data read/decode, encoding, H6 rollout,
metric computation, optional cache write, windows/second, and peak GPU memory.
Record device, Torch/CUDA versions, precision, source commit/checkpoint hashes, input
manifest, selected IDs, window counts, and zero-dose identity. Partial/failed runs are
not completed studies. Use a new output directory for every run.

The earlier 30–80 GPU-hour range is not a measured budget. configs/study.json deliberately
leaves total GPU-hours null until the actual baseline and representative ablation timers
have been measured. CPU synthetic timings must not fill that field.

## Sources

- [JEPA-WM paper](https://arxiv.org/abs/2512.24497)
- [Pinned upstream repository](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0)
- [Upstream MetaWorld splitting](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/datasets/traj_dset.py)
- [Upstream Push-T loader](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/datasets/pusht_dset.py)
- Historical source tree: [archive](../archive/2026-09-07-workspace/)
