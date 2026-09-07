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
| Primary released validation pool | Split seed 234 yields Reach 33, Reach-Wall 27; Push-T opens separate 21-row `val/` | All 33/27/21 rows complete and verified in BF16 and FP32, including the authorized seven-row addition |
| Planning episode count | Paper G.2 says 96; full MW release config and MW training config say 48 | Prepared 96 per task/condition following the paper; the 48-to-96 change is explicit |
| Training seeds/history | Three independently trained seeds for final models; aggregate last ten epochs for these tasks | One released checkpoint per task model; extra environment/planner seeds cannot reproduce training variability |
| Planning seeds | Example base seed 1, local seed = base + rank × H × 1000; separate CPU sampling and CUDA planning streams | Same formulas in prepared contract; eight fixed logical streams independent of physical scheduling; not claimed to recover unpublished actual episode identities |
| Simulator, Reach / Reach-Wall | Official MetaWorld V3 wrapper, MuJoCo, expert policy goal generation | Same official wrapper and goal-generation methods; initial/goal pairing and native CEM checks passed |
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

Update at approximately 10:30 EDT: official simulator initial/goal setup passed on
all three tasks in `planning-env-smoke-v2`; repeated smoke seeds reproduce byte-identical
initial and goal observations. The unmodified native CEM + simulator loop has also
completed its non-confirmatory smoke on Push-T, Reach-Wall and Reach.
These are runtime checks, not efficacy estimates or a 96-episode evaluation.

The static vision-action planning adapter also passed 60 fit-only comparisons on
Push-T (ten frozen arms, H2/H5/H6, both precisions). Active edits match the existing
frozen hook compilation bitwise. Native, zero-dose and edits outside the realized
horizon match the unsteered rollout exactly. Initial failed checks are preserved:
the standalone zero-dose path needed to skip zero-valued edits entirely because
cloning the context could alter striding and FP32 accumulation at H5. No identity
tolerance was relaxed. This does not validate dynamic geometry/support operators,
select a candidate, or authorize a confirmation reveal.

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

The offline redistribution completed without changing shard membership: Utah finished
BF16 Push-T; six primary-host GPUs executed the untouched FP32 shards. Finished shards
were checksum-verified and reused. The superseded Utah scheduler was terminated only
after its BF16 children completed, and Utah is now stopped. All primary offline pools
are complete; see [corrected results](CORRECTED_OFFLINE_RESULTS.md).

### Subsequent planning preparation (approximately 11:05 EDT)

`planning_scenarios.py` prepared 96 initial/goal scenarios each for Reach and Reach-Wall,
with 96 distinct simulator random-state vectors per task. It invokes the authors'
expert-goal helper, retains expert failures without filtering and does not run a learned
policy. Eight logical seed streams are independent of physical GPU assignment. The
released local-RNG configuration retains global NumPy/Torch seed zero while dedicated
streams use each rank's local seed. These artifacts prepare inputs; they do not certify
an exposure audit, freeze an intervention or constitute confirmation results.

Input-preparation report hashes:

- Reach: `31461954ff88586fcb343ec681eebf0207c05c8015ab3b001c7f0efbfbc7f080`.
- Reach-Wall: `5680fe26a65e5b0c3154473fcc78823c82be15d0bedb7adc7e389a716b835089`.

Static planning transfer now has an additional fit-only check: the **same BF16-primary
fit tensors** on the FP32 planner must match their unchanged hook compilation. A
separately fitted FP32 sensitivity bank is not a substitute for the primary intervention.
Each MW task passes 90 arm/horizon/fit-bank/inference-precision checks, including H2/H5/H6
and all ten frozen arms. Cross-precision transfer report hashes:

- Reach: `52dfac1061e7db5714c76932ce92695a192f6b229d1da9e1e8c4fbe0f0aa2688`.
- Reach-Wall: `67c2a7352a3d0f0cbfe94d2a581813f7cad61ab774dfdcf3eab6e9002fed605a`.

The static adapter completed all four real-CEM integration checks on the already-used,
excluded smoke seed, with joint and matched-random arms on each MW task. Each retained
the published candidate/iteration counts and completed 210 adapter calls in approximately
280–288 seconds. These checks measure integration and throughput only; their task
outcomes cannot select a candidate. Durable, checksum-verified reports are under
`artifacts/offline_study/primary-durable-20260907/planning-static-cem-smoke-v1/`.
They do not promote a Reach-Wall coupling arm that failed its offline advancement gate.
These static checks do not validate dynamic support/rank transfer or the exact
selected combined recipe; their separate checks are described below.

### Rank-transfer implementation and completed fit checks (12:35 EDT)

`planning_support.py` now implements that transfer as an explicitly recorded study
extension. It uses the unchanged BF16-primary rank bank with the actual upstream
two-context, strict-FP32 planning unroll, retaining all 32 source response probes,
the native H3 shadow target and the source common-degeneracy test. The H6 target is
not redefined for H5: when the official planner shortens its horizon below six,
this adapter returns a true native rollout and adds no fictitious actions. The same
rule applies to both components if a combined recipe later advances. This is our
intervention-transfer policy, not a setting claimed to appear in the JEPA-WM paper.

Only response-operator construction uses candidate slices of at most eight elements.
The actual native and edited forecasts retain the full official candidate population;
no reduction, reordering or padding is used. Earlier v1/v2 checks rejected slicing
actual forecasts because batch/stride-dependent FP32 kernels changed predictions
beyond the existing tolerance. An alternative probe-chunk diagnostic also failed
source parity, so the original eight-probe chunk and all 32 probes are retained.
Those failures are preserved, not relabelled as passes or overwritten.

Fit-only v3 passed both MW tasks: H2/H5 true-native behavior, H6 unchanged-source
corrections, exact full-population native/zero identity and independent full-batch
source-hook checks for 19 and 300 candidates. Repeated fit-action timing at 300
candidates is about 102 seconds for rank4 versus 2.9 seconds native. This is not an
actual CEM/closed-loop throughput measurement. Local verified report hashes:

- Reach: `10d5352901f44fc8898aa26ae7ea095545f746ef7fc1d59ddc5261bd1bbb1faf`.
- Reach-Wall: `3c9cc26a1dc3c2491e0cc61a7291785bb7e649b7ca8b20526b8fe3bdc7502253`.

Four follow-up checks completed on GPUs 0–3 of the same US worker: exact Reach
combined and its random control, plus Reach-Wall rank4 and its random control.
They cycle all contiguous H6 action sequences from the existing fit-only fixture,
retaining 300 candidates and a 19-candidate tail test. Their reference assembles
unchanged source rank fields and uses the original static coupling compiler.
No physical targets are attributed to these candidate actions; no new outcomes or
confirmation cohort is opened. The first launch failed at source staging before
Python could import the module; its logs remain under `planning-selected-transfer-fit-v1-*`.
The verified source snapshot is `code-v25`; new `planning-selected-transfer-fit-v2`
outputs all pass, with 52 distinct fit action sequences cycled to 300 candidates.
These repetitions are engineering coverage, not independent observations. Full
candidate predictions match independently assembled source hooks bitwise. The
completed reports and contracts have been checksum-verified in local durable storage:

- Reach combined: `e40c3421f8b3e4fd823538caad99fa70967d89310482193d5a8d94fb1457853e`.
- Reach combined random: `b28336c4c9379eac1ae6d2e5e9501b3efbe7226acea44c583fa3b321c3f0be42`.
- Reach-Wall rank4: `c9c20732f474ce54f35aa2ab6fa8bc968da5ea5a1e82caa4b15eaad1cd4357c8`.
- Reach-Wall random: `37de0eeccdae62483f65e82aa8616479d280ae083bb4b7397c755ee4ad5a50da`.

Actual selected-candidate CEM integration, prospective exposure verification and
the final policy-analysis freeze remain separate gates.
