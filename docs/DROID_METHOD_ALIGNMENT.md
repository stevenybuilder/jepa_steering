# DROID: approved sixth task and source-alignment contract

Decision: **implement**. User authorization: 2026-09-07, after 14:00 EDT.
RoboCasa is excluded; no RoboCasa data, kitchen assets or simulator installation.
Scope is six tasks: Reach, Reach-Wall, Push-T, PointMaze, Wall and DROID.
The existing five-task work continues independently. This document does not claim
that DROID's baseline, interventions or confirmation have already run.

## Reference and claim

JEPA-WM v4 is the governing paper, not a stock DROID imitation-learning benchmark.
DROID supplies real-robot training recordings; the authors evaluate plans on their
own released Franka recordings. Their action-score endpoint does not execute new
physical-robot actions. This addition tests our recipe on real-data planning inputs
and a different model architecture, not real-robot closed-loop success.
Primary evidence: paper Sections 5.1 and 5.3, Appendix E, Table 10 and G.2/Figure 19;
the pinned official repository and linked Hugging Face raw assets.

## Executable sources and input scope

- `configs/droid_assets.json`: immutable dataset/model revisions, 32 recording files
  (16 `episode.h5` files and 16 companions), and one released checkpoint.
- `offline_study.droid_assets`: bounded two-worker download and verification,
  preserving small-file Git blob identity and recording SHA256 for every file.
- `offline_study.droid_contract`: strict source-config checks, 64-episode seed
  schedule, exact action metric, and explicit unresolved paper/code discrepancies.
- Native training config: `configs/vjepa_wm/droid_final_sweep/` followed by
  `droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_noprop_repro_2roll_4n.yaml`.
- Native full evaluation config: `configs/evals/simu_env_planning/droid/jepa-wm/`
  followed by `droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml`.

The raw Franka evaluation files total 2,105,926,429 bytes; the released checkpoint
is 2,746,249,135 bytes. This initial download is **not the DROID training corpus**.
The repository's full non-stereo raw-training download is approximately 5.6 TB;
the paper reports 8,000 training trajectories. The exact training-subset manifest
still needs reconciliation before claiming matched retraining. Do not silently
replace it with the 16 evaluation videos or a conveniently small training subset.

The separate DINOv3 encoder/source must also be verified. The authors expect the
native DINOv3-L checkpoint and local DINOv3 repository, not an unvalidated swap to
a Transformers implementation. Hugging Face access to JEPA-WM does not automatically
establish access to every separately gated encoder. Credentials remain local/in
process memory, never committed or written into artifact receipts.

## Model and native planning design

| Item | DROID setting |
|---|---|
| World-model checkpoint | `jepa_wm_droid.pth.tar`, pinned SHA256 in the asset manifest |
| Encoder | DINOv3 ViT-L/16, 256x256 images, 256 spatial patches |
| Predictor | 12-block AdaLN/RoPE, width 1024; action-conditioned, no proprioceptive input |
| Planning objective | Final visual-embedding L2 distance; alpha=0, not our other tasks' proprioceptive objective |
| Planner | CEM, 15 iterations, 300 candidates, 10 elites, variance scale 0.1 |
| Horizon/context | H3; two-position planning context; execute three model actions |
| Action handling | Seven measured pose-delta dimensions; no action normalization/repeat |
| Norm limits | Joint dimensions 0..5 bounded by 0.1; dimension 6 by 0.75, exactly as the source config |
| Evaluation count | 64 sampled initial/goal segments per checkpoint/condition |
| Display-only omissions | No optional decoder/plots; unchanged stimuli, model forecasts and planner budgets |

Keep the joint position-plus-orientation norm group: do not accidentally interpret
the config as a translation-only limit. The dummy DROID environment returns a
constant success value; that field must NEVER be used as scientific task success.
Only the recorded-plan action endpoint is meaningful for this task.

## Stimuli, sampling and randomness

Use the official `DROIDVideoDataset`/Franka HDF5 path and
`PlanEvaluator.sample_traj_segment_from_dset` behavior. Evaluation camera is
`exterior_image_2_left`; dataset RNG seed is 234. The configured 4 fps is implemented
as `ceil(30/4)=8` raw frames per sample, effectively 3.75 fps under the loader's
30-fps assumption. Preserve this implementation rather than silently rounding to
a different temporal stride.

The native loader samples five frames, then the evaluator chooses a four-frame
initial/goal segment with three intervening action deltas. Preserve constructor
RNG consumption, camera selection, temporal-window draws and segment offset order.
Record source recording, raw frame indices, initial/goal hashes and RNG state
bindings. A read/decode failure must fail the scientific job, not silently substitute
another video as the permissive upstream retry path can do.

Use eight logical RNG streams with local seeds `1 + rank*3000`, rank 0..7, and the
source episode-seed formula `(local_seed**2 + episode*local_seed) % (2**32-2)`.
CPU scenario and CUDA CEM generators are separate from the dataset's NumPy RNG.
Streams persist across their eight episodes; physical GPU assignment must not change
their identities. Pair native, candidate and control on identical stimuli and initial
planner RNG states. Actual historical author episode draws are not available.

### Paper/release population discrepancy

The paper describes 16 evaluation videos. Both the released training-monitoring and
planning configs explicitly select 15; the additional released file is
`franka_custom/data/pick/v0/run_0001/episode.h5`.
Download and retain all 16. Preserve the exact 15-file source order for initial
engineering replication. Audit the extra video's metadata/lineage and freeze the
scientific population decision before inspecting comparative outcomes. Do not call
15 videos an exact reconstruction of the paper's 16, or treat the extra file as an
independent untouched confirmation cohort merely because the config omits it.

Observed source-file audit, 2026-09-07 19:00 UTC: all 15 config-selected files
open and have finite, frame-aligned poses and images. Their exact file/state/initial
state/first-frame fingerprints are distinct; this is not proof of family-level
independence. The extra `pick/v0` recording fails HDF5 opening: actual/published
length 344,981,504 bytes, stored EOF 438,316,577. Its published checksum matches
(`2a3b44dbb4331654dfb207b1e92f41acdcdc670ca838a6f9bed0d8f1ef52cc01`),
so this is an internally truncated upstream asset, not a partial local download.
That provides an objective reason not to add it to the executable population;
the authors' historical reason for omitting it remains undocumented. Preserve the
raw file and failure receipt. Never silently repair/pad it or replace it with a
different recording. Initial engineering continues with the already selected 15.

The native-versus-fail-closed loader comparison passed exact pixel/action/state
and post-sampling RNG equality on recording indices 0, 7 and 14, preserving the
constructor's initial sample. Full model/CEM engineering remains a separate gate.

The official Hugging Face DINOv3-L tensors were also reverse-mapped into the pinned
native DINOv3 constructor with strict learned-tensor coverage. On two fixed
synthetic images, maximum patch-feature difference was `5.1856e-6` and class-token
difference `1.8477e-6`, passing predeclared `atol=rtol=1e-3`. This is equivalent
native tensor serialization, not a claim of possessing the original `.pth` bytes.
Native weight SHA256 is
`fc933429ede48304e832fa11bbcc5d838eb539ca83b1746b93c8625526407b9a`.
The native hub entrypoint requires a filename suffix even for local weights; use
the same verified constructor followed by explicit strict tensor loading.

## Primary metric, hypotheses and analysis

The code's `Act_err_xyz` first sums actions over time, subtracts the recorded summed
actions, takes absolute values, then sums the first three dimensions:

`E = sum_xyz(abs(sum_time(planned_actions) - sum_time(recorded_actions)))`.

The primary checkpoint score is `max(0, 800*(0.1-mean_episode(E)))`. The official
evaluation writes the mean episode error into `eval.csv`; its plotting code then
clips/scales that mean. Average the resulting checkpoint scores across the required
late epochs/seeds. Do NOT clip per episode and average: that changes the statistic
(errors 0 and .11 give official score 36, versus 40 with per-episode clipping).
This is not mean per-step L1 error or all-seven-dimensional error. Perfect agreement
scores 80, not a literal 100% success rate.
Orientation/gripper errors, per-step discrepancy, predicted embedding losses,
action norms and runtime are secondary diagnostics. Some distinct action paths
can have equal net displacement; this limits the endpoint's interpretation.

Hypothesis: the same predeclared geometry-based recipe can improve the native
DROID checkpoint's action score and repeat across trained seeds/late checkpoints.
Alternatives include no improvement, improvement shared by matched-random edits,
better forecast loss without better action score, or an effect confined to a few
recordings/checkpoints. Compare native, zero-dose, applicable fixed candidate and
scope/rank/energy-matched random controls on paired stimuli. Do not admit candidates
only because of significant offline improvement or add methods after outcomes.

The old six-block, 400-feature, proprioceptive-H6 fit is NOT a DROID fit. The existing
H6 planning adapter intentionally returns native at H3, so passing DROID through
it would not test steering. A DROID-specific fit, hook/time mapping, finite arm
registry and validation receipt are required before comparative execution. Fit on
separate permitted DROID training recordings, never these evaluation recordings.

Treat source recording/lineage as the clustering unit, with repeated temporal
segments nested inside it and training-seed/checkpoint structure retained. Report
both source-video and episode counts. The 64 episodes are not 64 independent
recordings; repeated checkpoints are not independent training seeds.

Before comparative reveal, freeze paired contrasts, two-sided familywise 95%
intervals over the finite candidate family and the practical effect rule. Use
1 action-score point as the study's prospective minimum useful native improvement;
also require a positive matched-random contrast. This is our study criterion,
not a threshold specified by JEPA-WM. Retain all seeds and epochs, report per-seed
effects, and distinguish paper-style late-epoch standard deviations from inferential
confidence intervals. With only three seeds and a small recording pool, population
uncertainty remains limited; report detectable-effect sensitivity rather than
asserting adequate power from the nominal episode count. Fresh-recording confirmation
would require a distinct collection, not another crop of an exposed video.

## Required training histories and checkpoint aggregation

Three independent training runs are required. Reproduction seeds are 234/235/236;
only 234 is published in the source configs, so this is not a claim to know the
authors' exact unpublished triplet. DROID trains for 315 epochs, 300 optimizer
updates per epoch, global batch 256, BF16, AdamW betas (0.9,0.995), two-step rollout
loss and the original frozen encoder. Do not reuse the (0.9,0.999) simulation betas.
Preserve the complete source optimization schedule and preprocessing.

The authors explicitly clarify in their released plotting script (lines 1992–1993)
that the rebuttal replaced a dense last-100-checkpoint rule with a late epoch window
`[215,315]`, permitting a subset of checkpoints. Follow that released methodology:
native cadence is every six zero-based loop epochs plus the final checkpoint.
One-based metadata is therefore 217, 223, ..., 313, 315: **18 checkpoints per seed**,
64 episodes per checkpoint/condition. That is 3x18x64 = **3,456** repeated episode
evaluations per condition, not 19,200 and not 3,456 independent recordings.
Retain the final `latest` checkpoint immutably. Source filenames use zero-based
indices while checkpoint metadata and evaluation folder labels are one-based.
The five simulated tasks' paper-level last-ten primary analysis remains 3x10x96
per task (14,400 total); six-task primary work totals 17,856 episode evaluations
per condition. The released plotting defaults additionally use a wider MetaWorld
window (start=20, last_n=30); retain those checkpoints for an explicitly labelled
released-code sensitivity, not an undisclosed replacement of the paper primary.
No training histories have yet been completed. Never choose epochs by outcomes.
Fit the same frozen intervention algorithm separately for each checkpoint rather
than silently transferring a final-checkpoint operator to earlier models.

## Execution order and acceptance

1. Stage/verify the pinned raw evaluation assets and released model; verify encoder.
2. Audit HDF5 schema, source identity, population selection, frames/actions and RNG.
3. Validate the native model/planner and exact action metric on excluded engineering
   inputs with full CEM budget; retain failures. No comparative-outcome selection here.
4. Complete DROID fit-data/training-subset provenance and frozen candidate mappings;
   execute paired released-checkpoint planning development after the above checks.
5. Produce and verify three complete training histories, refits and all required late
   checkpoint comparisons; aggregate without choosing favorable seeds/epochs.
6. Report replication separately from fresh-recording confirmation and physical
   robot task success; neither follows from completed downloads or a dummy simulator.

Respect the latest aggregate $7/hour US-only limit. Download/stage independently of the
running MetaWorld checks and Wall/Maze preparation. Reprofile the larger model;
do not extrapolate small-model timings, overfill disks, or shrink the planner to fit.
Exact completion is proved by run and checksum receipts, not this plan.

## Sources

- [JEPA-WM v4 PDF: methods, Appendix E, Table 10, G.2 and Figure 19](https://arxiv.org/pdf/2512.24497v4)
- [Pinned source repository](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0)
- [Official setup and training/evaluation instructions](https://github.com/facebookresearch/jepa-wms)
- [Released raw Franka inputs](https://huggingface.co/datasets/facebook/jepa-wms/tree/6116f042ae7ae4c8e3f1fd2f194f432615664182/franka_custom)
- [Released model](https://huggingface.co/facebook/jepa-wms/blob/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_droid.pth.tar)
- [Original DROID dataset](https://droid-dataset.github.io/)
