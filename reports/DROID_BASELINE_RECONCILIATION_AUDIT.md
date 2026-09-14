# DROID baseline reconciliation audit

> Historical development report. For the completed protected evaluation and current
> interpretation, see [Results](../docs/RESULTS.md). Archived references below are
> retained as paths; bulk evidence is not bundled.


Audit date: 2026-09-11. Scope: existing evidence only; no scientific runs,
protocol changes, GPU rentals, or original-result modifications.

## Bottom line

The stored unsteered DROID score **51.09964804706845** reconstructs exactly from
all 64 saved action records, including when using the upstream metric code.
The upstream sampling logic also reproduces all 64 recorded file choices and
frame sequences. No score-calculation or scenario-sampling error was found in
these checks.

This does **not** establish why our score is 2.899648 points above the published
48.2, nor establish an improvement attributable to our interventions. A known
aggregation mismatch and unresolved historical-input/implementation equivalence
prevent a direct superiority claim. An untouched upstream GPU evaluation versus
our wrapper, on identical real inputs and hardware, was not performed in this
audit.

## Reference comparison

The JEPA-WM v4 paper reports DROID CEM-L2
48.2 (1.8) in its main comparison and 46.5 (0.4) for final checkpoints across
three training seeds in Table 12. Its main result uses late-training aggregation;
our audited reference uses one released epoch-315 checkpoint. Neither published
aggregate supplies the score for our particular checkpoint and input manifest.
The paper describes 16 Franka evaluation recordings, while the released configs
select 15. These are recorded-action comparisons, not physical-robot executions.

The existing source-alignment contract (archived reference: `../docs/DROID_METHOD_ALIGNMENT.md`)
records the executable late-window interpretation: selected checkpoints in
epochs [215,315], not a requirement to evaluate 100 distinct checkpoints. This
audit did not launch or complete those training histories.

## Checks and findings

“Pass” below is limited to the particular check, not certification of the entire
evaluation against the authors' historical run.

| Check | Finding | Status / consequence |
|---|---|---|
| Metric and score transformation | Recomputed all 64 XYZ action errors from saved actions, using both our verifier and the upstream evaluator's metric branch. The upstream score transformation gives 51.09964804706845 exactly. | Pass. No per-episode clipping, wrong dimension, or percentage-success substitution detected. |
| Record integrity and completeness | DONE/report/protocol bindings, all 64 episode checksums, expected episode IDs, arm labels, and seed assignments verify. | Pass. No missing or duplicated episode IDs in this reference. |
| Scenario sampling | Executed the upstream DROID sampler on recorded dataset metadata, retaining constructor RNG consumption and persistent logical streams. Every recording index, five-frame window, and four-frame goal segment matches. | Pass for sampling logic and metadata. Images were not redecoded in this audit. Historical author draws are unavailable. |
| Evaluation counts | 64 episodes, 58 unique recording/goal segments, 15 source recordings. | Matches released evaluation count. Repeated segments follow the sampler; these are not 64 independent recordings. |
| CEM budget and objective | H3, context 2, 15 iterations, 300 candidates, 10 elites, visual L2 objective, variance scale 0.1, and source action-norm grouping match. All saved episodes have the expected full CEM call trace. | Pass for config and recorded call budget; no shortened search found. |
| Camera and preprocessing | Native left-camera loader, 256-pixel preprocessing, ImageNet normalization, measured pose deltas, and stride 8 match the released executable path. | Source match. The YAML resize scale is overridden by the official evaluation loader; our scale matches that override, so this is not a detected divergence. |
| Released predictor checkpoint | Existing asset and execution receipts bind the released checkpoint hash, epoch 315, and exact predictor tensors. | Receipt chain passes. Large checkpoint bytes were not downloaded or rehashed afresh locally. |
| Visual encoder | Earlier receipts verify strict mapping of official Hugging Face DINOv3 tensors into the native constructor and feature agreement on two fixed synthetic images. | Partial equivalence evidence. Original native weight-file bytes were unavailable; this is not full historical-weight or full-planner equivalence. |
| Historical evaluation population | Both pinned configs select the same ordered 15 files. The extra released file is absent from those configs and has a checksum-verified internal HDF5 truncation in the original asset audit. | Paper/release discrepancy documented. The authors' actual historical membership and reason for omission are unresolved; do not assume they evaluated the extra file. |
| Training-seed / checkpoint aggregation | Our reference is one released checkpoint, not the paper's multi-seed late-window aggregate. | Not matched. This changes the quantity being estimated, but does not quantify the cause of the observed gap. |
| Full upstream-runner equivalence | Existing engineering checks cover selected native/strict-loader inputs, encoder features, repeated wrapper execution, and zero-dose controls. | Not established end to end. Repeatability of our wrapper alone is not independent upstream equivalence. |

## NeurIPS checklist and methodological guidance

Applied the relevant portions of the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist):
claims and limitations (1–2), reproducibility and experimental settings (4–6),
uncertainty (7), and compute disclosure (8). This is a focused baseline audit,
not a completed submission-wide checklist or license review.

| Requirement | Application to this discrepancy |
|---|---|
| Claims match evidence | Label 51.10 a released-checkpoint reproduced reference, not an intervention gain or robot success percentage. Keep the published aggregate separately labelled. |
| Reproducible settings | Retain source revisions, input order, actual sampled frames, separate RNG streams, planner settings, environment, and weight receipts. Do not imply all historical author artifacts are available. |
| Correct uncertainty | Distinguish source recordings, sampled segments, training seeds, and checkpoints. Published standard deviations are not confidence intervals for our difference from the paper. |
| Limits and compute | Disclose unverified full upstream GPU equivalence and unavailable matched histories. This audit used CPU record checks only; no new paid workloads. It is not a reconciliation of historical project spending. |

The methodological reference is Patterson et al.,
Empirical Design in Reinforcement Learning
(2024), especially sections 2.2, 4.1, 4.4–4.5, and 6. Its guidance supports
comparable baselines, explicitly identified randomness, paired differences when
appropriate, and inspection of implementation details before attributing a
reproduction discrepancy to noise or a new method. Accordingly, neither a
shared seed label nor a higher standalone number is sufficient here.

Our existing DROID comparative analyzer resamples paired source-recording
clusters across arms (20,000 draws; seed 2026090801) and computes simultaneous
intervals for its 16 frozen contrasts. Source inspection and CPU tests support
that implementation; this audit did **not** rerun the entire 576-record
comparative analysis. Such intervals do not supply missing training-seed
variation or resolve the historical-baseline discrepancy. See the existing
DROID findings (archived reference: `KEY_FINDINGS.md#10-droid-supplies-a-completed-additional-endpoint-not-robot-success`)
for the separately completed comparison.

## Fresh validation performed

- Called `offline_study.tasks.droid.verify_droid_replication.verify` directly, without its
  output-writing CLI, on the preserved reference directory below.
- Independently extracted and executed the pinned upstream sampling and DROID
  metric branches on existing metadata/actions: **64/64 exact matches** for
  sampled recording/frame sequences and for XYZ errors. No model forward calls.
- Checked the three frozen project source hashes and nine pinned upstream
  source-file hashes against the run protocol; all match.
- Rechecked saved asset, native-encoder, and full-planner engineering receipt
  chains and their locally present listed payload hashes.
- Ran `.venv/bin/python -m unittest discover -s tests -p 'test_droid*.py' -v`:
  **27 tests passed**, no skips. These are CPU tests, not GPU parity tests.

The CPU audit environment used PyTorch 2.2.2 with CUDA unavailable. The preserved
scientific runtime used PyTorch 2.7.1+cu128 and explicit float32/no-TF32. This audit
does not establish equality with the authors' historical numerical environment.

## Evidence locations and identities

Repository audited on `main` at `01e24dd`; no tracked scientific code was changed.
Upstream checkout was clean at
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.

Primary reference directory:
`artifacts/offline_study/primary-durable-20260907/droid-native-replication-20260907-v1/`.

| Evidence | SHA256 |
|---|---|
| Reference report | `31ff18bb63d54876ccc4ec300b89957bc133253746b2d994f932b655231355f4` |
| Reference protocol | `e8d20156daf9212c8dcf45e6db60fcd71e5f11134b536a8582cc08279d3508be` |
| Asset-audit report | `8fb62e5731bf0e59e1d5d0c2ab0ca9811297b5545f0a9144e99f3610bd666a14` |
| Native-encoder report | `90daf86eaf1a03f8ec0a0a6db9bf8e8b4d5ad209b5c2aa1805fdb52d60a599bf` |
| Engineering report | `4587522fcab5c518465545f524b88b3b4cd28dd780640d765bf7fb1d58ecd7c8` |
| Released predictor bytes, as verified in original receipt | `daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa` |

Local implementation evidence:
[contract](../src/offline_study/tasks/droid/droid_contract.py),
[native wrapper](../src/offline_study/tasks/droid/droid_native.py),
[reference runner](../src/offline_study/tasks/droid/droid_replication.py),
[record verifier](../src/offline_study/tasks/droid/verify_droid_replication.py), and
[paired analysis](../src/offline_study/tasks/droid/droid_coupling_behavior.py).

Pinned upstream evidence:
[evaluation config](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml),
[sampler and metric](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/evals/simu_env_planning/planning/plan_evaluator.py), and
[score/aggregation plotting code](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/plot/logs_plan_joint_per_design_choice.py).

## Reporting disposition

Preserve both published and reproduced numbers with their distinct scopes. Use
the same-setup unsteered and matched-random results for intervention contrasts;
do not replace our measured baseline with 48.2 or interpret the 2.90-point gap
as a steering effect. The exact historical discrepancy remains unresolved.
The outstanding verification item is full upstream-versus-wrapper equivalence
on identical real inputs; it was not silently treated as passed or launched here.
