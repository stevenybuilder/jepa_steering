# Author-aligned checkpoint averaging

User reaffirmed September8,2026: match the JEPA-WM late-epoch aggregation. This
does not reduce the required study to a single released checkpoint or substitute
repeated evaluations of that checkpoint for independently trained histories.

For each simulated task and condition, compute its success percentage from all
96 evaluation episodes at each checkpoint. Require epochs41–50 from each of
the three reproduction training seeds234/235/236. The reported mean is the mean
of those30 checkpoint scores, with equal weight per seed/checkpoint:

`mean = (1 / 30) * sum_seed sum_epoch (100 / 96) * sum_episode success`.

These are2,880 episode evaluations, but only three training realizations. Do
not treat epochs or repeated scenarios as independent training seeds. Fit data,
scenario identity, trained checkpoint origin and raw endpoint verification are
separate prerequisites. GPU count never multiplies the96-episode population.

The released plotting function `aggregate_task_data_by_groups` concatenates
the per-seed checkpoint scores and calls NumPy `mean` and population `std`
(`ddof=0`). The standard deviation is descriptive, not a confidence interval or
standard error. Report per-seed means/SD and the SD of seed-averaged epoch scores
separately with explicit labels; do not interchange them with that released-code
SD or with our paired, multiplicity-corrected intervention inference.

DROID remains different: require64 recorded-plan evaluations for each of its
18 native-cadence checkpoints217,223,...,313,315 in the registered215–315
window, for each training seed. First average XYZ action error across each
64-episode evaluation, then compute`max(0,800*(0.1-mean_xyz_error))`, then average
the54 resulting checkpoint scores. Clipping individual episode scores first is
not equivalent and is not the released plotting procedure. This is not robot
success. Sparse released-code cadence is explicitly distinguished from the
paper's last100-epoch wording.

A final-checkpoint-only three-seed simulation summary uses epoch50 and288 total
episode evaluations per task/condition. It is separately named and never replaces
the primary late-epoch result; Table12 in the paper reports this narrower scope.

## Implementation and limitations

`offline_study.checkpoint_aggregation.aggregate_checkpoints` accepts exactly one
task/condition and the complete fixed grid. It rejects missing/duplicate cells,
wrong episode counts, reused checkpoint/evaluation hashes, nonfinite values and
mixed tasks/conditions. It performs no best-epoch, best-planner or outcome-based
selection and does not silently summarize a partial grid.

This is a numerical aggregation component, not a raw-receipt verifier or a
training/evaluation launcher. Its callers must verify the underlying checkpoint,
scenario and result receipts. No complete three-seed/late-checkpoint behavioral
grid is established merely by adding or testing this component. Missing model
histories and evaluations still require work; averaging already saved scores is
cheap CPU work and does not itself require another GPU run.

## Paper/code distinction

Primary simulation window41–50 follows the paper's ten-epoch statement. The
released plotting aliases use a wider30-checkpoint MetaWorld window and the
released Reach-Wall evaluation YAML says48 episodes. Our paper-level primary
is explicitly96 episodes/ten late epochs; we do not describe it as an unchanged
execution of every released configuration. Any wider-window sensitivity must
remain separate and must not change prior results or authorize additional jobs.

Sources: [paper §5.1/G.2](https://arxiv.org/html/2512.24497v4#A7.SS2),
[final-checkpoint Table12](https://arxiv.org/html/2512.24497v4#A7.SS1),
[pinned plotting code](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/plot/logs_plan_joint_per_design_choice.py),
[pinned window aliases](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/plot/aliases.py).
