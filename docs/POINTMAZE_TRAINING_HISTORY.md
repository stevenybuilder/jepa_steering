# PointMaze training-history execution contract

This implements the existing required three-seed/history track. It does not add a
new intervention, change an outcome gate, or replace frozen-checkpoint behavioral
comparisons. Wall, MetaWorld, Push-T and DROID retain their separate contracts.

## Source and population

Use the pinned JEPA-WM commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` and
`configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml`.
The entire configuration is bound by SHA256
`c666b4251f72f56c08c69ab223cfe645625f39705f1c84285320edc98c205a66`.
The original 718,363,945-byte PointMaze archive expands to 2,003 required files,
30,117,624,491 bytes. `pointmaze_training_inputs` verifies each file against the
archive, not just a list of filenames or a previously staged evaluation subset.

The authors' loader normalizes the full released dataset before its seed-234
trajectory split. Preserve that behavior and disclose it; intervention fitting
remains a different, training-only operation. The 2,000 trajectories yield 1,800
training and 200 validation rows, 145,800 four-frame training clips and 12,200
eight-frame validation clips. Clips are not independent trajectories.

## Optimizer, validation and checkpoints

| Item | Required execution |
|---|---|
| Training seeds | 234, 235, 236; only 234 is published, not the exact unpublished author triplet |
| Complete training budget | 50 epochs, 1,139 optimizer updates per epoch |
| Effective training batch | Native 16 logical ranks × 8 examples = 128 |
| Native model/loss | Frozen DINOv2-S/14, six-block predictor, BF16, original teacher-forced plus two-step rollout objective |
| Optimizer | Original AdamW, betas 0.9/0.999, gradient clipping, LR/weight-decay schedules unchanged |
| Validation monitoring | Every 200 within-epoch updates: five events per epoch, 250 total |
| Validation sampler | Native 16-rank shuffle/padding; no `drop_last`; 190 batches of 64 followed by one of 48 per complete cycle |
| Validation metrics | Actual upstream `step_model(train=False)`, H6/context3, recorded and noisy-action rollouts |
| Checkpoint retention | Every epoch saved immutably, optimizer/scaler/scheduler and logical RNG/cursor state included |
| Behavioral primary history window | Epochs 41–50, 96 total episodes per checkpoint/condition, three training seeds |

The 12,208 validation clip draws in a complete sampler cycle include eight padded
draws; they do not create additional independent observations. Native validation
sampling is distinct from the disjoint, non-padded behavioral episode allocation.
The persistent validation iterator must retain its final partial batch and cycle
position across epochs and resumes. Wall's 418-update/two-validation-event schedule
cannot be reused for PointMaze.

## Fidelity gates and execution order

`offline_study.pointmaze_training_history` shares the already checked native model
construction and gradient objective, not Wall's task-specific schedule or image
layout. PointMaze images are THWC; the reader must preserve exact upstream pixels,
actions, states, rewards, nested subset mapping and transform RNG after conversion.
All receiving CPU tests passed on Virginia, including the actual upstream reader
on synthetic inputs. This is not GPU gradient parity or a completed training run.

Before full training on each seed/device:

1. Verify all input bytes and the preserved native split/clip ordering.
2. Run the existing five-update PointMaze-specific pilot, comparing actual upstream
   per-logical-rank objectives/gradient mean against accumulation, including weights,
   optimizer moments, scaler and RNG.
3. Complete the seed's first full 1,139-update epoch, including all five native
   validation events; verify a saved/reloaded model with the actual upstream
   checkpoint loader and one matched next update on disposable clones.
4. Resume that exact epoch-one state through epoch 50. Keep all checkpoints and
   validation events. No seed/checkpoint selection by results, no short-run substitution.
5. Separately refit the frozen intervention recipe on permitted fitting data for
   each required checkpoint, run the registered paired behavioral comparisons,
   then aggregate the complete checkpoint/seed panel.

Single-device accumulation preserves the effective batch and objective, not the
authors' undocumented cross-GPU reduction order or exact historical RNG sequence.
The GPU-specific numerical checks are required before scientific advancement.
Full training completion alone does not complete the behavioral history comparison.

## Resource assignments at preparation

Virginia 50189244 GPU0 is reserved for seed234 after Wall234 finishes and all 50
Wall checkpoint hashes verify. Its CPU queue is live; it does not interrupt Wall.
Indiana 50205763 GPU0 is reserved for seed235 after all 64 prior non-rank navigation
shards verify and the preceding job releases the GPU. Training data staging there
adds only absent files after checking existing bytes; it never overwrites active
evaluation inputs. Seed236 remains required and separately schedulable.

These assignments use existing US instances within the latest $7/hour cap. Every
run gets an immutable source/output root; the old Wall and active behavioral
snapshots are not patched. Final status is established by run/verification receipts,
not by this preparation document.

Source: [official PointMaze training configuration](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml),
[native data loaders](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/datasets/utils.py).
