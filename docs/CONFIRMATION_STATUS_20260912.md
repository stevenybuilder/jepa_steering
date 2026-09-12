# Protected confirmation: status and what it takes to finish (2026-09-12)

Facts from files in this repo and the Drive archives, not from summaries. Where the only
evidence is a document claim, it says so.

## 1. What the design requires

| Item | Requirement | Source |
|---|---|---|
| Stage | Freeze recipe/contrasts/useful effect/sample size/multiplicity, then **one** protected evaluation on untouched scenarios | `EXPERIMENT_PLAN.md` "Evaluation order", steps 6–7 |
| What is re-tested | The **unchanged** learned methods (fixed-response rank-4 refined edit; equal-budget coupling) plus native and their two matched-random controls = 5 arms. Labelled an *independent re-test*: neither method qualified under the development eligibility rule | `FINAL_STATISTICAL_VALIDATION.md` (FSV) "Authorized execution" |
| Tasks | Reach and Reach-Wall only, released checkpoint `c3297772…` | FSV; `confirmation-20260911-v1/protocol.json` |
| Population | Reserved base-1 schedule (`seed_schedule(1)`): 96 unique initial states per task, expert-defined goals, zero learned-policy exposure at preparation; 8 logical streams × 12 episodes; all five arms of a stream on **one physical GPU** | `offline_study/confirmation.py` (`ReservedGoalBank`), FSV "Actual reserved-input readiness", protocol `logical_streams` |
| Size | 96 episodes per task per arm → **960 complete 100-step episodes** | protocol `total_full_episodes` |
| Endpoint / analysis | Official task success; 4 contrasts × 2 tasks = family of 8; 20,000 paired scenario-cluster bootstrap draws, seed 2026090722, Bonferroni simultaneous 95 %; exact paired discordance p-values with Holm; useful effect 5 pp; **not** powered for 5 pp (FSV table: ~1,000–2,500 pairs would be); `stop_on_significance: false` | protocol `analysis`, `statistical_scope`; `fixed_response_behavior.ANALYSIS` |
| Receiving gate | Four full excluded-engineering episodes per actual GPU (native, native-repeat exact, fixed_rank4, matched random) before any protected outcome; strict FP32, no TF32; coupling arms bind the restored coupling-engineering receipts | protocol `receiving_gate`, `precision`; `confirmation.py` `verify_coupling_engineering` |
| Forbidden | No new fit, dose, direction, arm or subgroup search; no online probes; no pooling with development; development is reported separately | protocol `new_fit_or_dose_search`, `analysis_role`; FSV |

## 2. What was actually run (2026-09-10/11)

- **Authorized** 2026-09-11T02:19Z: $65 total, $7/hr, hard deadline 2026-09-11T11:00Z
  (`confirmation-20260911-v1/AUTHORIZATION.json`). Exposure audit passed (1,075 files,
  1,007 outcome records, 0 reserved matches; `EXPOSURE_AUDIT.json`). Freeze-v2 protocol
  sha `319d5ec5…`, "protected outcomes observed before freeze: false" (`FROZEN.json`).
- **Fleet** (`FLEET_PLAN.json`): one 8×4090 (Florida, $3.21/hr) for all Reach-Wall streams;
  four 2×4090 US boxes for Reach streams (0–1, 2–3, 4–5, 6–7). All RTX 4090 (Ada; passes the
  device-parity checks measured 2026-09-12).
- **Receiving engineering passed** on every GPU that reached it: e.g. 50546169 gpu-0
  `fixed_response_full_cem_engineering_complete`, native_repeat_exact, paired_stimuli_exact,
  full CEM schedule verified, 4 episodes in 1,235 s (≈305 s/episode). The 8-GPU box shows
  engineering for gpu-2…gpu-7 in its archive listing (gpu-0/1 not seen in the `report.json`
  extract; not verified further).
- **User stopped the run** 2026-09-10 23:35 EDT ("Stop the runs for now", `USER_PAUSE.json`
  03:36Z; `PAUSED.json` 03:43Z). The workers' "returncode 1" is **not a failure**: their run
  logs end with `Bounded queue interrupted; preserve outputs` (SIGTERM from the pause).
- **What exists from it:** native-arm episodes only, all **partial streams**, no `DONE.json`:

| Worker | Task / logical ranks | Complete native episode records |
|---|---|---:|
| 50544130 (8×4090) | reach-wall 0–7 | 84 (ranks 0,1,4,5,6,7: 11 each; rank 2: 10; rank 3: 8) |
| 50546172 | reach 0,1 | 8 |
| 50546171 | reach 2,3 | 9 |
| 50546170 | reach 4,5 | 8 |
| 50546169 | reach 6,7 | 8 (4 + 4) |
| **Total** | | **117 native; 0 learned-arm; 0 complete streams** |

  All five worker archives (`JEPA-confirmation-<id>-20260911.tar.gz`, ~350 MB each) are on
  Drive with preservation manifests; two were re-verified byte-for-byte against their disks
  on 2026-09-12 before the instances were destroyed. Nothing further is recoverable from them.
- **Consequence for reuse:** the runner has no resume (`output.mkdir(exist_ok=False)`; whole
  12-episode streams) and the frozen failure policy is "preserve partial stream, no dropping
  or replacement". The 117 records are retained evidence, **not countable**; every stream
  reruns from its seed. Per FSV the base-1 cohort now has recorded *native* exposure and
  "must not be described as wholly unopened again" — a disclosure, not a blocker, since no
  learned-arm outcome was observed.
- **Since then:** the table-completion plan (2026-09-11) set `confirmation_restart_authorized:
  false` in `configs/study.json` and kept the pause; last night's 14-cell work was
  development/replication on the exposed population and did not touch this stage.

## 3. What remains and what it takes

| Quantity | Value | Basis |
|---|---:|---|
| Episodes to run | 960 (all of them) | no complete stream exists |
| Seconds/episode | ~305 on 4090; ~245 on 5090 | confirmation engineering timing; 2026-09-12 MetaWorld rates |
| Scientific GPU-hours | 81 (4090) / 65 (5090) | 960 × rate |
| Engineering + bootstrap per GPU | ~0.35 h + ~0.25 h | 4 episodes per GPU per task; setup |
| GPU-slots | 16 (one per task-stream; 60 episodes each, 5 arms on one GPU) | protocol `logical_streams` |
| Wall-clock, 16 GPUs | ~5.5 h (4090) / ~4.5 h (5090) | one stream per GPU |
| Wall-clock, 8 GPUs | ~11 h / ~9 h | two streams per GPU |
| Cost at ~$0.70/GPU-h | ~$60–70 total (≈ the original $65 cap) | 87–94 GPU-h incl. setup |

Code path: `scripts/vast/manage_confirmation.py` (fleet, fail-closed) → `launch_confirmation.py`
→ `run_confirmation_worker.py --task --deadline --logical-ranks` → `offline_study.confirmation
run` per arm/rank, then `offline_study.confirmation analyze` on the complete two-task panel.
Inputs: `confirmation-20260911-v1/confirmation-inputs.tar.gz` (179 MB, sha `bb94cff4…`, on
Drive as `JEPA-confirmation-source-inputs-20260911.tar.gz`) and `bootstrap_confirmation.sh`.
Last night's fleet tooling (v2 slot driver, multi-GPU setup, MOVED markers) is **not** used
here: the confirmation has its own manager, and its protocol requires all five arms of a
stream on one GPU, so no cross-device spreading. Reusable pieces: offer search by $/GPU-hr,
the dead-provision waiter, collect→Drive→`rclone check` scripts, the quiet monitor.

Gates that must be explicitly given before anything runs:
1. **User authorization of a new total-spend cap and deadline** — the 2026-09-11 authorization
   expired at 11:00Z and FSV says confirmation "needs a new total-spend decision before leasing".
   Write a new `AUTHORIZATION.json` (manager reads `deadline_timestamp` from it).
2. **Lift the pause**: `USER_PAUSE.json` makes the manager refuse rental and restart
   ("User paused this run; no new rental/workload restart authorized"); it is immutable by
   design — a new dated user-authorization receipt must supersede it, not a silent delete.
3. `configs/study.json`: `confirmation_restart_authorized` → true, recorded with the date.
4. **Decide the question** (FSV step 2): current-checkpoint independent re-test at n = 96
   (valid replication, underpowered for 5 pp) vs a larger, separately funded run. The frozen
   protocol is the 96-scenario re-test; changing n means a new freeze, not an edit.
5. **Hardware**: the manager hardcodes `gpu_name == 'RTX 4090'` (refuses anything else).
   5090s passed the same parity checks and are ~20 % faster; using them means an ops-script
   change that must be recorded, not a science change. Ampere/Hopper are excluded either way.
6. Receiving engineering must pass on every new GPU before protected outcomes; no tuning
   from protected outcomes; all five arms of a stream stay on one GPU; whole streams only.
7. Report as an independent re-test alongside (never pooled with) development; disclose the
   117-episode native exposure of the base-1 cohort from the paused run.

## 4. Not determined here

- Whether the 8-GPU box's gpu-0/gpu-1 engineering completed (only gpu-2…7 receipts were seen in
  the archive listing); irrelevant to a rerun, which re-engineers every GPU.
- Current 4090/5090 availability and price (the 2026-09-12 night market had no cheap 4090s).
- Whether the user wants n = 96 (frozen) or a powered design; the docs leave that to the user.
- Whether Codex's 2026-09-10 protected-exposure audit needs re-running after the 14-cell work
  (that work used the exposed development population only; no reserved inputs were read).
