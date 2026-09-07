# Paper-aligned planning preparation (2026-09-07)

This is a setup/provenance report, not planning results or proof of confirmation.
The governing study remains [EXPERIMENT_PLAN.md](../docs/EXPERIMENT_PLAN.md).
Implementation: `src/offline_study/planning_contract.py`, `planning_env_smoke.py`.

## Sources checked

- [JEPA-WM paper](https://arxiv.org/html/2512.24497v4): Figure 1, Appendix F / Table 10,
  Appendix G.2, Figures 20–21. Table 10 distinguishes planning from offline rollout.
- [Pinned official repository](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0):
  full JEPA-WM evaluation configs, training configs, `pyproject.toml`, `envs/init.py`,
  `envs/metaworld.py`, `envs/pusht_env/pusht_env.py`, `plan_evaluator.py`, `gc_agent.py`.
- [Dataset card](https://huggingface.co/datasets/facebook/jepa-wms): Push-T is rehosted
  DINO-WM data. The small `hf_data` preview is not the complete raw release; the
  downloaded `pusht_noise.zip` and raw MetaWorld files are the study inputs.
- [Model file inventory](https://huggingface.co/facebook/jepa-wms/tree/main): checked
  recursively; one `jepa_wm_metaworld.pth.tar` and one `jepa_wm_pusht.pth.tar` are
  listed. DINO-WM and V-JEPA checkpoints are different models, not extra trained
  seeds of the same JEPA-WM. This inventory does not provide all three training
  seeds or the last ten epoch checkpoints used for paper aggregation.

## Exact distinctions

| Item | Paper/released code | Current study/preparation |
|---|---|---|
| Offline batch | Four validation clips per device; sampled repeatedly during training | Four clips per device, all valid clips/prefixes in the authorized pool; not the original training-time sample history |
| Offline context | Three, H6; MW 18-frame and Push-T 8-frame clips, stride five | Implemented and checked against actual upstream inputs and rollout |
| Primary released validation pool | Split seed 234 yields Reach 33, Reach-Wall 27; Push-T opens separate 21-row `val/` | Original 29/24 MW rows running; seven-row addition authorized and queued; all 21 Push-T rows running |
| Planning episode count | Paper G.2 says 96; full MW release config and MW training config say 48 | Prepared 96 per task/condition following the paper; the 48-to-96 change is explicit |
| Training seeds/history | Three independently trained seeds for final models; aggregate last ten epochs for these tasks | One released checkpoint per task model; extra environment/planner seeds cannot reproduce training variability |
| Planning seeds | Example base seed 1, local seed = base + rank × H × 1000; separate CPU sampling and CUDA planning streams | Same formulas in prepared contract; eight fixed logical streams independent of physical scheduling; not claimed to recover unpublished actual episode identities |
| Simulator, Reach / Reach-Wall | Official MetaWorld V3 wrapper, MuJoCo, expert policy goal generation | Same official wrapper and goal-generation methods; isolated simulator environment being checked |
| Simulator, Push-T | Official PushTEnv, Pymunk 6.8.0, Pygame rendering; dataset-based initial/goal setup | Same implementation and dataset replay path; no substitution of PyBullet |

### Table 10 planning settings

| Setting | Reach / Reach-Wall | Push-T |
|---|---:|---:|
| Candidate sequences per CEM iteration | 300 | 300 |
| Horizon | 6 | 6 |
| Elites | 10 | 10 |
| CEM iterations | 15 | 30 |
| Planning context | 2 | 2 |
| Elementary actions per model action | 5 | 5 |
| Planned actions executed before replanning | 3 | 6 |
| Maximum elementary steps | 100 | 30 |

Planning cost is terminal visual embedding L2 plus 0.1 times proprioceptive
embedding L2 for the released final models. Context two here is intentional, not
a regression to the earlier offline context mistake. Push-T's published setup
executes its complete H6 plan in one 30-step simulator episode; unlike MetaWorld,
it is not a multi-replan feedback-control test.

The full released MW model config is shared between Reach and Reach-Wall. The
repository supplies a full JEPA-WM Reach-Wall example, but not a full JEPA-WM Reach
example; Reach uses that shared model/data configuration with the official Reach
task template. Do not substitute the DINO-WM Reach model config.

## Replication and fresh confirmation

Including the seven reserved MetaWorld rows completes the primary author-validation
pool, under the user's explicit conditional approval. It does not make them new
independent confirmation after using their outcomes for development selection.
Only those seven additional rows are authorized; the other protected MetaWorld
rows are not silently opened. Their original registry labels remain preserved.

Paper-style Push-T planning draws segments from the released validation trajectories.
Ninety-six episodes sampled this way need not represent 96 source families. Report
episode counts, distinct pairs, and source-family counts separately; use family-level
uncertainty for correlated released-data episodes.

Fresh MetaWorld simulator episodes can retain the paper's environment/expert method.
Fresh Push-T initial-state families require a distinct prospectively frozen sampling
protocol. The official wrapper exposes a random-state sampler, but that is not the
paper's dataset-based Push-T protocol, and random goal reachability cannot be assumed.
No fresh-family confirmation has yet been run or claimed. Matching 96 episodes is
not by itself a power calculation.

## Runtime reproducibility and efficiency

Update at approximately 10:07 EDT: official simulator initial/goal setup passed on
all three tasks in `planning-env-smoke-v2`; repeated smoke seeds reproduce byte-identical
initial and goal observations. The unmodified native CEM + simulator loop has also
completed its non-confirmatory smoke on Push-T and Reach-Wall; Reach is running.
These are runtime checks, not efficacy estimates or a 96-episode evaluation.

The released standalone planning entrypoint does not enable an outer mixed-precision
context. Native planning smoke therefore uses strict FP32 with TF32 disabled; do not
silently copy the offline BF16 setting into planning. This is distinct from the paper's
training-time precision and does not reconstruct its full training/evaluation history.
Any later planning precision change needs a separately recorded fidelity check and
must be fixed across conditions before the confirmation reveal.

Simulator dependencies are installed in a separate Python 3.10 environment so
running offline workers are not changed. MetaWorld is pinned at source commit
`6e01ad7e2ffb2302e4dca04f796fcd8837df8540` from the repository named in the authors'
install instructions. The paper does not supply an exact historical simulator lockfile;
we must record actual package versions and pass reset/replay checks, not claim binary
identity with the authors' unpublished runtime.

Use independent episode streams on available GPUs, keep model/data local, batch the
300 CEM candidates, and omit unused decoder/plotting work. Simulator physics remains
on its native CPU implementation; GPU inference and modest graphics work are separate.
These are applications of the [GPU chapter](https://jax-ml.github.io/scaling-book/gpus/),
not permission to change candidate counts, precision, or planner settings mid-study.

Offline work is being redistributed without changing shard membership: the Utah
worker finishes its existing BF16 Push-T jobs; six primary-host GPUs take the untouched
FP32 Push-T jobs after MetaWorld diagnostic completion. Finished shards are checksum
verified and reused. The original two-GPU scheduler is paused only to prevent it from
starting those same FP32 jobs; its live BF16 children finish normally.
