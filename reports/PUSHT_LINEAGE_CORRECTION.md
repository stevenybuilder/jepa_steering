# Push-T lineage correction — 2026-09-07

## Finding

The released Push-T train tensors contain 18,685 distinct rollout rows arranged as
185 exact-initial-state families of 101 rollouts each. The 21 released validation rows
have 21 distinct initial states, with no exact initial-state overlap with train. The
release does not establish that an initial-state family is literally one source
demonstration, so this project uses the narrower term `initial-state family`.

The first study inventory split individual rollout IDs instead of those families. All
185 train families consequently appeared in fit, development, and provisional holdout.
The development baseline measured 1,682 rollout rows and therefore exposed every train
family at least once.

## Affected receipts

- Dataset revision: `6116f042ae7ae4c8e3f1fd2f194f432615664182`
- Original manifest SHA256:
  `96c461df57aa44de8122379a48a5806b833512764b7a7652cd1586126e069751`
- Original inventory receipt SHA256:
  `e88ccaaa33e53cb017259983746b6c3e33a449469be07441e489558aa2017283`
- Original baseline report SHA256:
  `df1ece7bcfa6ef0575f2b1d7cbf67f2318b7d8ee8eb7e8356d54c6957d1f1878`
- Original baseline window-metrics SHA256:
  `d9db5d84dcd88da27895f57026581fc2f72baedb4e77f3d239c5b52e909ac7c5`
- Original baseline scale: 1,682 rollout rows, 6,728 windows, 232.667 seconds
  wall time on eight RTX 5090 GPUs.

## Claim boundary

The affected baseline remains valid as a GPU-throughput measurement and as descriptive
per-rollout forecast-error output. It is not a sample of 1,682 independent experimental
units and cannot support a confirmatory or power claim. Its conservative detectable
independence structure contains 185 initial-state families.

Recomputing a 149/17/19 family split does not make its 19 holdout-labelled train families
untouched: the old development baseline already included rows from all 185 families.
Historical work also accessed the released validation pool. Accordingly, current
released Push-T train and validation data are development/replication data for this
study. Confirmation requires fresh, independently collected initial-state families
assigned to a frozen split before their outcomes are inspected.

## Implemented safeguards

- Inventory splitting is by `lineage_group`, never by rollout ID.
- Manifest validation rejects a lineage group spanning multiple splits or tasks.
- Development exposure filtering excludes a whole family if any row in that family
  lacks explicit clearance.
- Baseline and intervention outputs aggregate windows within rollout, then rollouts
  within family, and report equal-family-weighted task summaries.
- Intervention operator-bank rows bind the lineage group as well as trajectory/window.
- Reports use `rollout_trajectories` and `independent_lineage_groups` instead of calling
  every rollout an independent trajectory.
- The active plan and machine-readable study configuration explicitly record that the
  current Push-T pools are not confirmation-eligible.

The MetaWorld inventory and baseline use one lineage group per released trajectory and
are not affected by this Push-T row-family error.
