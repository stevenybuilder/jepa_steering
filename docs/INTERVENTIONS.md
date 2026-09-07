# Frozen intervention executor

`jepa-intervene` is the scoped migration boundary between historical exploratory
scripts and the active study. It executes all registered arms for one category, but
it does not select directions, anchors, patches, blocks, HMM states, gates, doses, or
success criteria. Those choices must be fitted on `fit`, reviewed, and frozen first.

The executor supports six predeclared comparisons, with spatial and layer distribution
kept as separate one-factor experiments:

- `vision_action_coupling`
- `action_response_geometry`
- `operator_rank`
- `imagined_time_routing`
- `distribution_spatial`
- `distribution_layer`

Every category has a required native arm, a zero-dose arm that exercises real hook
sites at scale zero, the named mechanism arms from the experiment plan, and its named
matched-random controls. Missing arms fail closed. The native and zero-dose forecasts must be
bitwise identical within every edited batch. On the first batch, native rows are also
checked against a same-shape uninstrumented reference pass with a declared narrow FP32
tolerance (`rtol=1e-5`, `atol=5e-5`), because separate optimized CUDA attention calls
are not bitwise repeatable. The receipt records exact maximum and mean discrepancies.

The zero-dose arm must register every hook location used by any active or matched-random
arm, with every scale set to exactly zero. Extra, post-hoc arms are rejected as strictly
as missing arms.

## Frozen protocol

The protocol is JSON with `schema_version: 1` and `status: "frozen"`. It binds the
manifest, checkpoint, pinned vendor commit, and fit receipt by hash; declares fit and
development splits, tasks, hypothesis, dose budget, arms, and primary contrasts; and
uses only these edit sites:

- `predictor_visual`: newest visual predictor input at a registered H1..H6.
- `block_condition`: newest action-condition input at a registered block/H1..H6.
- `block_output`: the whole block output or an explicit token slice.

An arm edit contains `site`, `horizon`, `tensor`, and `scale`; block sites also require
`block` in 0..5. A block-output edit can specify `token_start` and `token_end`. An
optional `gate` names a scalar tensor in the operator bank. Routing gates therefore
need their own fit-time/native-shadow audit before freezing; this executor does not
infer an HMM from development outcomes.

The protocol must use these arm names:

| Category | Required mechanism/control arms (plus native and zero_dose) |
|---|---|
| Vision/action | visual_only, action_condition_only, joint, joint_equal_standardized_energy, permuted_visual, permuted_joint, matched_random, matched_random_equal_standardized_energy |
| Action geometry | equal_anchor_linear, cubic, projected_cubic, reflected_curvature, matched_random |
| Imagined-time routing | constant_gate, memoryless_gate, hmm_filtered_gate, matched_random |
| Operator rank | rank1, rank4, rank8, plus one rank/spectrum/energy-matched random control for each |
| Spatial distribution | one_patch, contiguous_group, equal_size_scattered_group, all_patches, plus support-specific random-direction controls and equal-cardinality random-position controls |
| Layer distribution | single_block0 through single_block5, intermediate_blocks2_3, all_six_blocks, plus one support-specific matched-random control for each |

Layer names use zero-indexed predictor blocks. The block registry is enforced by the
protocol validator: every singleton B0--B5 is included, B2+B3 is the evidence-derived
intermediate-zone composite, and all six blocks is the global arm. Every mechanism arm
has a random control with exactly the same block support. All layer arms must use the
same hook type, imagined horizon, spatial scope, and total direct-sum operator rank.
Fit one basis in the registered direct-sum layer space and split it into layer-specific
slices; never copy one vector across blocks. Scale slices so total delivered squared L2
energy is equal across arms.

The operator-rank protocol fixes P3/H3, full 256-patch support, semantic target, fitting
groups, algorithm, and total delivered squared L2 energy while varying rank 1, 4, or 8.
Each mechanism arm declares `operator_rank` and has a random subspace with the same
support, rank, spectrum, and energy. The layer protocol also declares `operator_rank`,
but it must be the same positive integer for every arm. Its `dose_budget` must declare
`equal_total_delivered_squared_l2_per_arm`, `fixed_total_direct_sum_rank_per_arm`, and
`same_support_rank_spectrum_and_energy`. These are frozen-protocol requirements; the
fit receipt and run receipt must provide the numerical rank/spectrum and delivered-dose
audit before a result is scientifically eligible.

For spatial distribution, `matched_random_<support>` means a random direction on the
same tokens. `random_position_<support>` retains the arm's token count and topology but
moves it to a fit-frozen random location/set. This separates whether a particular token
region matters from whether an edit of that size matters anywhere. The all-patch arm
has no random-position analog because there is only one complete 256-patch support.

## Operator bank

The operator bank is a data-only `.pt` file loaded with `weights_only=True`:

```python
{
  "schema_version": 1,
  "protocol_sha256": "...",
  "global_tensors": {"shared_direction": tensor_without_batch_dimension},
  "rows": {
    "TRAJECTORY_ID:START": {
      "trajectory_id": "TRAJECTORY_ID",
      "start": START,
      "split": "development",
      "tensors": {"row_specific_direction": tensor_without_batch_dimension}
    }
  }
}
```

Tensor keys resolve row-specific values first and global values second. A scalar gate
multiplies the declared scale. Edits at the same hook location sum only when their
shapes agree. At application time, every dense operator tensor must exactly match its
target site or token slice; broadcasting is rejected.

## Efficient execution

For a batch of `B` windows and `A` arms, contexts/actions are repeated window-major and
all arms run in one predictor unroll with effective batch `B*A`. Images and future
targets are encoded only once per original window. This raises matmul intensity without
model sharding or NCCL collectives, but it also multiplies predictor activation memory;
start with `--batch-size 1` and increase only after measuring reserved memory.

The command accepts only reviewed development trajectories and strict FP32/no-TF32:

```bash
jepa-intervene --vendor vendor/jepa-wms \
  --checkpoint CHECKPOINT --checkpoint-sha256 CHECKPOINT_SHA256 \
  --manifest trajectories.jsonl --exposure-registry exposure.json \
  --data-root DATA_ROOT --tasks TASKS \
  --protocol FROZEN_PROTOCOL.json --operator-bank OPERATORS.pt \
  --fit-receipt FIT_DONE.json --batch-size 1 --device cuda:0 \
  --max-runtime-seconds 1800 --shard-index 0 --num-shards 1 \
  --output DURABLE_RUN_ROOT/category-development
```

For multiple GPUs, launch one process per device with a distinct `--shard-index` and
the same `--num-shards`; assignment is deterministic by trajectory ID, with no padding
or model collectives. Each command fails at the first batch boundary reached after the
runtime cap. A rental supervisor should still impose its own process-level timeout and
copy each shard receipt to durable storage before instance termination.

`jepa-intervention-multigpu` provides that supervisor for GPUs on one host. It launches
one process per device, preserves global shard indices, validates every child receipt,
and aggregates only when the local shard set is globally complete. `--shard-offset`
and `--num-shards` support a Push-T split across hosts; such a partial host writes a
partial receipt rather than misreporting a complete result. After all shard directories
are gathered under one root, `jepa-aggregate-interventions` verifies exact global
coverage and builds the combined report.

The receipt reports each arm separately, paired candidate-minus-control differences
after equal within-trajectory aggregation, delivered L2 norm per edit site, throughput,
peak memory, input hashes, and frozen parameter/buffer version checks. It produces no
p-value, holdout claim, simulator result, or efficacy conclusion.

For vision/action coupling, it additionally forms the output-space residual
`joint - visual_only - action_condition_only + native` before reducing it to MSE. The
receipt separates this from the additive quadratic cross-term and the full factorial
MSE interaction, so score curvature is not mislabeled as a nonadditive model mechanism.

## Fit-only vision/action protocol

`jepa-fit-coupling` implements the first bounded protocol from the plan. It selects one
deterministic released row per fit lineage group, captures the native H3 newest visual
field and P3 (zero-indexed predictor block 3) action condition, and fits one rank-one
maximum-covariance direction pair. It never loads development or holdout outcomes.

The component dose is fixed at 0.1 robust fit-score standard deviations before any
development execution. The factorial joint arm retains both component doses. A separate
equal-standardized-energy joint scales each component by 1/sqrt(2), matching the total
squared standardized dose of either single component; its random control receives the
same scaling. The matched-random arms use orthogonal unit directions at the corresponding
per-site doses, and the spatial sham uses a fixed permutation of the 256 visual patches,
which preserves visual L2 norm.
The fit receipt is written and hashed before the protocol; the resulting operator bank
is then bound to the frozen protocol hash.

Run one fit process per task/checkpoint:

```bash
jepa-fit-coupling --vendor vendor/jepa-wms \
  --checkpoint CHECKPOINT --checkpoint-sha256 CHECKPOINT_SHA256 \
  --manifest trajectories.jsonl --exposure-registry exposure.json \
  --data-root DATA_ROOT --task mw-reach --max-fit-lineage-groups 128 \
  --batch-size 4 --dose-fraction 0.1 --device cuda:0 \
  --output DURABLE_RUN_ROOT/mw-reach-coupling-fit
```

The output contains `fit_selection.json`, `fit_receipt.json`, `protocol.json`,
`operator_bank.pt`, and `DONE.json`. A script exit is only a fit/protocol milestone;
it is not an intervention result or confirmation claim.

After all three development runs complete, `jepa-analyze-development --runs RUN...`
produces deterministic percentile-bootstrap summaries over independent lineage groups.
It never pools tasks, selects an arm, chooses a smallest useful effect, or labels the
development interval confirmatory. Those decisions and multiplicity handling are a
subsequent frozen-analysis gate before any protected MetaWorld evaluation is opened.
