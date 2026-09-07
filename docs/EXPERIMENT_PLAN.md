# Frozen-model offline study: active plan

Status: scope agreed; real strict-FP32 baselines have run; Push-T and MetaWorld
lineage corrections are implemented. Intervention arm families are agreed, but their executable scientific
protocols are not frozen. Completed baseline jobs are throughput/descriptive evidence,
not efficacy experiments or power calculations.

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
scope/rank/energy-matched random control. Among statistically equivalent eligible arms,
choose the simpler one. If none is eligible, retain native/no intervention. After the
one-factor sweeps, combine compatible selected choices on development and run drop-one
ablations. Freeze that combined recipe, controls, endpoints, multiplicity handling,
and sample size before one protected evaluation. Closed-loop success remains the final
primary outcome; offline forecast and mechanism measures are advancement gates.

## Agreed ablation table

| Category | Arms | Primary mechanism comparison |
|---|---|---|
| Vision–action coupling | No edit; visual-only; action-condition-only; joint; spatially permuted joint | Joint versus individual edits; nonadditive output interaction |
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
than a single component under a factorial; add a separately dose-matched comparison
before attributing improved efficacy to coupling rather than budget.

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
recorded-future endpoint beyond rank 1 and its own matched-random control. If higher
rank is statistically equivalent to rank 1, select rank 1.

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
fit procedure, total direct-sum rank, and total perturbation energy. It compares all six
zero-indexed singleton blocks B0 through B5, the archived evidence-derived intermediate
zone B2+B3, and the global B0+B1+B2+B3+B4+B5 arm. Fit a basis in the direct sum of the
registered block spaces and split it into layer-specific slices; do not copy one vector
across blocks. This keeps total operator rank fixed even when support spans more layers.
Every layer-support arm has its own random control with identical block support, rank,
spectrum, and energy. Early edits change later activations, so all multi-block edits are
constructed from and logged against the same specified native reference.

The spatial sweep fixes P3/H3, rank, target, and fitting procedure. Select the candidate
single patch and 4x4 contiguous region on fitting data, then freeze them. Compare one
selected patch, the selected contiguous 16-patch region, an equally sized scattered
16-patch set, and all 256 patches. Each support has its own random-direction control with
the same positions, rank, spectrum, and energy; selected positions additionally compare
against random position sets of equal cardinality. Share total perturbation energy
sum_{block,patch} ||delta||^2 and report downstream effect magnitudes.

A singleton winner supports layer localization; B2+B3 beating B2 and B3 supports an
intermediate distributed zone; all six beating B2+B3 supports global depth distribution.
Likewise, one/contiguous support beating scattered/all supports spatial localization,
while scattered/all beating localized supports spatial distribution. After both sweeps,
run only a minimal 2x2 interaction check crossing the best non-global versus global layer
support with the best non-global versus all-patch spatial support. Do not launch the full
layer-by-position Cartesian grid.

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
