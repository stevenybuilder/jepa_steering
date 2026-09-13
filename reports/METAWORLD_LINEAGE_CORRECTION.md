# MetaWorld exact-duplicate correction — 2026-09-07

> Historical development report. For the completed protected evaluation and current
> interpretation, see [Results](../docs/RESULTS.md). Archived references below are
> retained as paths; bulk evidence is not bundled.


## Finding

The official release has the paper-reported 12,600 MetaWorld rows: 300 for each
of 42 tasks. An exact numerical audit found 12,440 unique task/state/action
trajectories. There are 158 duplicated lineage groups: 156 contain two rows and
two contain three, for 160 redundant rows in total. Within every duplicated group,
the full float32 state and action tensors are byte-identical. This is a 1.27% row
redundancy, not evidence that the underlying files are corrupted.

Reach contains 300 rows and 295 unique groups. Reach-Wall contains 300 rows and
297 unique groups. Duplicate rows do not add independent statistical power.

## Correction policy

The initial inventory split released row IDs. The corrected inventory defines
`lineage_group` as SHA256 of the task name and exact state/action tensor contents.
Manifest validation rejects a group that crosses tasks or splits, and all metric
summaries aggregate windows within row and rows within lineage group.

Historical exposure is preserved instead of re-hashing the new content IDs. The
code reconstructs the original row-ID split, then coalesces duplicate members with
the following precedence:

1. Development wins because any measured member exposes the complete duplicate group.
2. Otherwise holdout wins over unused fit, keeping that entire group provisional.
3. Otherwise the group remains fit.

The exposure registry propagates every historical protected status across the full
duplicate group and fails closed if any development member lacks clearance.

## Corrected inventory

- Dataset revision: `6116f042ae7ae4c8e3f1fd2f194f432615664182`
- Upstream source commit: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`
- Rows: 12,600
- Independent state/action lineage groups: 12,440
- Split rows: 10,154 fit / 1,161 development / 1,285 provisional holdout
- Split groups: 10,056 fit / 1,125 development / 1,259 provisional holdout
- Reviewed development groups after historical protection: 1,119
- Manifest SHA256:
  `1b1a1f7f6f427fefa0eb5dde31dc1d5631dbab17e3eea3e74a0396b76691d44d`
- Exposure registry SHA256:
  `5328961ac2e3ea536cf210ca2d385604f3e420e3a391523e66303760089f882e`

## Corrected all-task baseline receipt

- Baseline report SHA256:
  `5be44e90bbf5513a54a4ff00d7f772246fff3ac7d332bbd11c2ecd3b14f3a2b7`
- Window metrics SHA256:
  `d648329c765a58b4f9b94274871c9a8d7d331bcb9db28b7aa0a7308c09291d11`
- Scale: 1,155 reviewed rollout rows, 1,119 independent state/action lineage
  groups, 4,620 windows, 197.723 seconds wall time on eight RTX 5090 GPUs.
- Reach: 28 rollout rows and 28 independent lineage groups.
- Reach-Wall: 25 rollout rows and 24 independent lineage groups.

All eight shard receipts passed report, window, selection, and disjoint-coverage
validation. This remains an unedited recorded-action forecast baseline: it certifies
the data path, metrics, grouping, and GPU execution, not intervention efficacy.
