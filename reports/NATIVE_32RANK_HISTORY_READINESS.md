# MetaWorld and Push-T native 32-rank history readiness

September 8, 2026; implementation update at 15:12 UTC. **A CPU-tested receiving
core and complete raw-input byte audit now exist; full native reader/history
implementation and launch authorization do not.** This documents the existing required three-seed/history track;
it does not alter frozen experiments, authorize protected access, introduce a new
method, or make feature caching a prerequisite. Current-checkpoint behavioral
work remains independent. No new model outputs or GPU timings were collected
for this track.

## Implemented since the initial readiness audit

- [Raw-input verifier](../src/offline_study/robotics_training_inputs.py): all126
  MetaWorld parquets (737,175,770 bytes) and all18,718 Push-T extracted files
  (7,370,447,305 bytes), plus the original2,785,304,515-byte ZIP, verified against
  pinned original download/extraction manifests. Existing complete assets were
  found on Indiana50205763; no new dataset transfer was required. The71.61-second
  receiving audit used low-priority CPU/I/O, no Torch import, and zero GPU calls.
  Thirteen compact source/test/protocol/report/DONE files passed local hash readback.
  [Readback proof](../artifacts/offline_study/robotics-input-verification-20260908-v2/READBACK.json).
- Preserve the failed v1 verifier attempt: MetaWorld passed, then Push-T manifest
  preflight rejected our mistaken `shapes.pkl` assumption. The pinned release
  actually contains `tokens.pth` in both pools; shapes remain native defaults.
  v2 corrected this check and passed all11 receiving CPU tests. The failure did
  not establish data corruption, interrupt a scientific run, or require reruns.
- [32-rank receiving core](../src/offline_study/robotics_training_pilot.py): actual
  pinned caller/sampler and native `step_model`, 32 eight-clip microbatches,
  independent logical CPU/CUDA/Python/NumPy RNG streams, and one averaged
  optimizer/scheduler update. Nineteen local CPU tests pass, including failure
  and provenance guards. The public numerical entrypoints require real native
  CUDA/model/input evidence; synthetic tests cannot authorize execution.
- The complete integration suite ran401 tests with two existing skips. Neither
  these tests nor byte verification establish transformed-input, GPU numerical,
  validation-cursor or epoch-resume equivalence. The native input parity reader
  and complete history orchestration remain missing. Existing behavioral and
  corrected-navigation queues do not wait for this work.

## Pinned evidence

Upstream: `vendor/jepa-wms`, commit
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`. Dataset revision:
`6116f042ae7ae4c8e3f1fd2f194f432615664182`.

Reference configurations, relative to that upstream root:

- MetaWorld: `configs/vjepa_wm/mw_final_sweep/mw_4f_fsk5_ask1_r224_pred_AdaLN_ftprop_depth6_repro_2roll_save.yaml`,
  SHA256 `db0c9cab6c12cb6541c222026358e7ccb482aa2276ed12b5a597ff4b547d0bba`.
- Push-T: `configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml`,
  SHA256 `7a16e8edbf40b260aab303949e90c1879dd5f778d51f3b6d6cb4bea1d99969e2`.

| Pinned source | Evidence used |
|---|---|
| `app/vjepa_wm/train.py` | Seeding at lines 234–235; actual config flattening/caller at 318–361; loader length at 399–403; persistent validation cursor at 727–778; train sampler epoch at 822; native objective at 834–1315; within-epoch validation trigger at 1333; checkpoint/planning cadence at 1424–1460 |
| `app/plan_common/datasets/utils.py` | MetaWorld versus Push-T branches at 123–162; distinct shuffle flags; native train/validation samplers and loaders at 230–260 |
| `app/plan_common/datasets/traj_dset.py` | Exact clip construction/permutation at 101–112; slicing/action concatenation at 134–153; whole-row seeded partition at 157–222 |
| `app/plan_common/datasets/metaworld_hf_dset.py` | 99-step alignment; full-dataset normalization before the split; all-task loader |
| `app/plan_common/datasets/pusht_dset.py` | Separate train/val pools, relative-action scaling, velocity proprioception, fixed normalization constants, native clip loaders |
| `app/vjepa_wm/utils.py` | Planning episode override at 318–320; original optimizer/schedulers/scaler at 844–999 |

SHA256 of `train.py`:
`c1fc4c57b99cab18df14405236adf463363fa4eef23705635a2a48e5d6e98285`;
dataset `utils.py`:
`e264c808ebdaa6bcd653538d7a2a53932a6cd985071e204efb5b966c6dff130a`.

The metadata audit used these complete, existing trajectory manifests:

- `artifacts/offline_study/2026-09-07-metaworld-dedup-correction/metaworld-inventory-v4-deduplicated-exposure-preserving/trajectories.jsonl`,
  SHA256 `1b1a1f7f6f427fefa0eb5dde31dc1d5631dbab17e3eea3e74a0396b76691d44d`.
- `artifacts/offline_study/2026-09-07-pusht-family-correction/pusht-inventory-v3-development-exposed/trajectories.jsonl`,
  SHA256 `e91f421193b18cb96b640b505f6186fa30e7f4647d99c3cb385c7b73d018849f`.

These manifests establish row order, lengths and lineage metadata, not current
availability or checksums of every raw video. Their historical exposure labels
are not replaced by an author-split label.

## Verified native work counts

The audit executed the actual pinned `train.main` config-flattening/call AST and
`init_data` AST, replacing only dataset I/O with metadata-sized `range` objects.
No DataLoader workers were started: loaders were constructed but never iterated;
only their CPU samplers/batch samplers were inspected. This verifies scheduling,
not image preprocessing or numerical training equivalence.

| Requirement | MetaWorld | Push-T |
|---|---:|---:|
| Training / validation source rows | 11,340 / 1,260 | 18,685 / 21 |
| Training / validation clips | 907,200 / 12,600 | 1,981,721 / 1,695 |
| Native logical training batch | 32 ranks × 8 = 256 | 32 ranks × 8 = 256 |
| Optimizer updates per epoch | 3,543 | 7,741 |
| Epochs / updates per seed | 50 / 177,150 | 50 / 387,050 |
| Train and validation sampler shuffle | True | False |
| Native sampler seed | 0 | 0 |
| Training `drop_last` / validation `drop_last` | True / False | True / False |
| Validation trigger | Every 300 updates within each epoch | Same |
| Validation events per epoch / all 50 epochs | 11 / 550 | 25 / 1,250 |
| Batches per validation cursor cycle | 99 | 14 |
| Last validation batch | 2/rank = 64 total clips | 1/rank = 32 total clips |
| Padded validation draws per full cursor cycle | 8 | 1 |

MetaWorld's 99-step trajectories produce 80 four-frame training clips and ten
18-frame validation clips each. Push-T lengths range from 49 to 246 steps in the
training manifest; its validation clips have eight frames. All use frame stride
five and concatenated elementary actions. The exact slice count is
`sum(max(0, T - frames * 5 + 1))`, not a count of nonoverlapping clips.

MetaWorld's sampler has no training padding and the loader drops 192 draws per
epoch. Push-T's sampler adds seven entries, then the final batch drops those seven
and 25 unique clips. Padding is reproduced only for native training monitoring;
it does not increase independent statistical n or change disjoint behavioral
episode assignments. The validation cursor persists across epochs and wraps
without changing the validation sampler's epoch from zero.

One MetaWorld model per training seed serves both Reach and Reach-Wall. It is
trained on all 42 tasks, not on only those two tasks. The missing training work is
three MetaWorld plus three Push-T histories, not a separate Reach-Wall model.

## Minimal execution contract to implement next

1. Keep the pinned native constructors and actual upstream `step_model` reference.
   The existing [navigation pilot](../src/offline_study/training_pilot.py) has a
   parameterized sampler, but its microbatches, accumulation and numerical proof
   hardcode 16 ranks/global 128. Add a separate 32-rank adapter; do not modify
   already frozen navigation source snapshots.
2. Preserve 32 eight-clip forwards and average gradients by **32**, with one
   optimizer, scaler, learning-rate and weight-decay step per global batch.
   A single 256-clip forward is not assumed numerically interchangeable. Keep
   BF16 autocast, frozen DINO, teacher loss divided by three plus the native
   sequential rollout loss, random prefix, stop-gradient and context three.
   Use native AdamW: betas `(0.9, 0.999)`, epsilon `1e-8`, clip norm one,
   learning rate `5e-4`, zero warmup and cosine weight decay `1e-7` to `1e-6`.
   Preserve the existing registered precision contract; no cache/kernel change.
3. Separate model seeds **234/235/236** from fixed data split/clip seed **234**
   and native distributed sampler seed **0**. Only 234 is specified in the
   released selected configs; do not claim the other two are Meta's unpublished
   exact seed IDs. Keep 32 logical CPU/CUDA RNG streams and explicitly test
   native loader initialization/reset RNG bookkeeping and validation wraparound.
   Do not silently inherit physical-worker RNG assumptions from the 16-rank
   runner, or claim bitwise reproduction of an unpublished all-reduce order.
4. Preserve native transformed images/actions/proprioception, including MetaWorld
   normalization before its whole-row split and Push-T's velocity/relative-action
   conventions. A selected-frame reader needs direct pixels, actions, states,
   rewards and RNG parity against the native reader; it is not inherited from
   the navigation-only mmap reader. Preserve native validation train/eval mode
   transitions, both noisy and recorded-action H6 forecasts, all partial batches,
   and the single persistent validation cursor. Validation is not an optimizer step.
5. Bind epoch-only resume to source/config/input hashes, task, model/data seeds,
   sampler policy, optimizer/scaler, scheduler steps, all logical RNGs, loader
   state, module modes, validation cursor and complete prior checkpoint hashes.
   At epoch `e`, require scheduler steps `3543*e` / `7741*e` and validation events
   `11*e` / `25*e`. Use the actual upstream checkpoint loader, then compare the
   next full update and a validation boundary on disposable engineering clones.
6. Save all 50 epoch checkpoints: upstream filenames `jepa-e0` through `jepa-e49`
   carry payload epochs 1–50. The existing primary aggregation requires epochs
   **41–50** across all three seeds, not favorable checkpoint selection. The
   upstream training loop launches planning after each epoch; deferred execution
   must preserve the checkpoint identity and distinguish the required final-window
   results from reproducing a complete 50-epoch behavioral learning curve. Each
   new checkpoint needs the unchanged recipe refitted on permitted fitting data
   and checkpoint-bound engineering, not reuse of a released-checkpoint fit.

## Readiness gates and protected scope

- **Full raw bytes now verified; native transformed inputs still pending.** The recent
  `artifacts/offline_study/pusht-native-recovery-20260908-v1/PUBLIC_INPUTS.json`
  contains 31 files: only one training video and all 21 validation videos. It is
  a planning recovery bundle, not the18,685-video training pool. The separate
  complete pool was found and verified on Indiana in the update above. No need
  to redownload it. Byte identity alone is not native pixels/actions/RNG parity.
- **The 140 other protected MetaWorld validation rows remain protected.** Complete
  native monitoring eventually reads all 1,260 validation rows. Existing approval
  for the seven primary rows is not approval for these other 140. Resolve the
  exposure/access contract explicitly before launch. Neither silent omission nor
  relabelling provides full native coverage. This report grants no new access.
- **MetaWorld has a code/paper episode-count distinction.** Its selected training
  YAML sets `evals.eval_episodes: 48`. `build_plan_eval_args` copies 48 directly to
  `meta.eval_episodes`; `nodes: 2` does not multiply the episode count. The existing
  agreed 96-total behavioral protocol remains an explicitly documented
  paper-matched choice, not literal equality to that YAML. Push-T's selected YAML
  itself specifies 96. No existing episode count is changed by this report.
- **Receiving numerical tests and real throughput remain pending.** Require
  32-rank native loss/gradient/update/RNG parity, partial validation and cursor
  wraparound, input parity, and save/resume evidence. Counts above are not a GPU
  ETA. Do not use navigation/synthetic timings to assert full-history completion.

These are dependencies of the already required study, as described in the
[behavioral amendment](../docs/BEHAVIORAL_EVALUATION_AMENDMENT.md), not new goals.

## Implementation ownership and remaining files

The first two modules/tests now exist. Keep current navigation runners and frozen
snapshots unchanged; no history launch is implied by this table.

| Responsibility | Files | Status |
|---|---|---|
| Complete raw-input binding | `src/offline_study/robotics_training_inputs.py`, `tests/test_robotics_training_inputs.py` | Implemented; full receiving byte audit passed |
| Native reader/order/transformed-input parity | Separate reader module/tests still required | Not implemented by the byte verifier |
|32-rank update, logical RNGs and receiving numerical pilot | `src/offline_study/robotics_training_pilot.py`, `tests/test_robotics_training_pilot.py` | Implemented and CPU-tested; native GPU proof pending |
| Task-specific monitoring, histories, epoch resume | `src/offline_study/robotics_training_history.py`, `tests/test_robotics_training_history.py` | Not yet implemented |

## Reproduce the count checks locally

Run from the repository root. This compact check recomputes metadata counts and
native sampler sizes; the separate audit described above additionally executed
the actual upstream caller/loader AST to verify its selected shuffle/defaults.
It reads no videos or outcomes, creates no workers, and writes no files.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from offline_study.author_validation import official_partition

paths = {
    "metaworld": "2026-09-07-metaworld-dedup-correction/metaworld-inventory-v4-deduplicated-exposure-preserving",
    "pusht": "2026-09-07-pusht-family-correction/pusht-inventory-v3-development-exposed",
}
for task, relative in paths.items():
    path = Path("artifacts/offline_study") / relative / "trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    train, val = official_partition(rows, task, seed=234)
    val_frames = 18 if task == "metaworld" else 8
    sizes = [sum(max(0, row["length"] - f * 5 + 1) for row in pool)
             for pool, f in ((train, 4), (val, val_frames))]
    samplers = [torch.utils.data.DistributedSampler(range(n),
        num_replicas=32, rank=0, shuffle=task == "metaworld") for n in sizes]
    updates = len(samplers[0]) // 8
    val_batches = list(torch.utils.data.BatchSampler(samplers[1], 4, False))
    print(task, {"clips": sizes, "updates_per_epoch": updates,
        "validation_events_per_epoch": updates // 300,
        "validation_batches": len(val_batches),
        "last_validation_global_clips": 32 * len(val_batches[-1])})
PY
```
