# Claude handoff — JEPA-WM experiments, execution failures, and live jobs

Prepared September 11, 2026. Last live snapshot: **23:50 UTC / 7:50 p.m. EDT**.

## Read this first

The user explicitly requested a handoff to Claude and an account of the failures. **This session is stopping implementation; it has NOT stopped the existing experiments or their billing.** Do not assume this document means the machines are parked. No new model session was launched.

**Delivery failure: zero of the 14 outstanding table cells is complete at this snapshot.** There are real partial rollout records, but earlier updates repeatedly described repairs, launches, tests, or GPU activity without delivering a completed new cell. The user's frustration is justified. Do not repeat those reporting mistakes.

Four US instances, **one GPU each**, remain running. The last account spending rate was **$6.093844/hour**, including retained storage, against a **$7/hour aggregate budget**. Credit was **$14.279348** at the snapshot. The configured $1.50 reserve would be reached in roughly 2.1 hours at that rate, before additional bandwidth charges or further top-ups. This balance does not support an end-to-end completion promise.

Use the existing local checkout. **GitHub main does not contain all the latest work: much of the running code is untracked locally.** Do not clone over it, reset it, or overwrite remote scientific snapshots with this dirty tree.

## User's current experiment scope

Finish the remaining cells in the six-task released-checkpoint comparison: Reach, Reach-Wall, Push-T, PointMaze, Wall, and DROID. Preserve all results and logs; avoid idle paid instances; use US compute only within $7/hour. The immediate instruction is a handoff, not authorization for the outgoing session to launch more work.

Do not restart the old expensive online rank solver. The current refined intervention is a fitted, inexpensive four-direction response operator. Do not reopen HMM, fresh confirmation, training-history replication, RoboCasa, or autonomous method search as prerequisites for this table. Those expansions were paused. DROID is the authors' **recorded-plan action-score endpoint**, not a physical-robot closed-loop success experiment.

The user prefers all project work on **main**, without new branches or PRs unless requested.

## Where to work and what to read

Repository: `/Users/stevenyang/Documents/rep_geometry_transcoder`

Read these before taking over:

1. `/Users/stevenyang/Documents/AGENTS.md`
2. Repository `AGENTS.md`
3. `/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md` — lease ownership is authoritative; read before GPU mutations.
4. `docs/EXPERIMENT_PLAN.md` and `configs/study.json` — read before scientific changes.
5. `docs/TABLE_COMPLETION_20260911.md`
6. `docs/METAWORLD_COMPONENT_COMPLETION.md`
7. `docs/REFINED_SIX_TASK_COMPLETION.md`
8. `reports/SIX_TASK_TABLE_PROGRESS_20260911.md` — timestamped, sometimes superseded snapshots.
9. `reports/SIX_TASK_RESULTS_20260911.md`
10. `reports/MATCHED_RANDOM_CONTROL_AUDIT_20260911.md`, `matched random control.md`, and `reports/DROID_BASELINE_RECONCILIATION_AUDIT.md`.

Other relevant design documents: `rank 4 intervention.md`, `Rank Edit.md`, `docs/FIXED_COMBINED_BEHAVIOR.md`, `docs/MUJOCO_REPLICATION_BEST_PRACTICES.md`, and `jax_scaling_notes.md`.

Reference sources:

- Authors' implementation: https://github.com/facebookresearch/jepa-wms
- Checkpoints: https://huggingface.co/facebook/jepa-wms
- Datasets: https://huggingface.co/datasets/facebook/jepa-wms
- Paper: https://arxiv.org/pdf/2512.24497
- Scaling Book: https://jax-ml.github.io/scaling-book/

## What is complete, and what is not

The existing table has **34 of 48 project cells complete**. Published-author rows are contextual comparisons and are not included in that count. The original 960-episode MetaWorld core comparison and its analysis are prior work, not the newly running extension.

The table below is the inherited completed-results inventory; it is not a fresh recomputation in this handoff. Simulation entries are success percentages. DROID uses a different score and must not be described as robotics success rate. NR means the requested complete cell is not available, even if partial records exist.

| Project condition | Reach | Reach-Wall | Push-T | PointMaze | Wall | DROID score |
|---|---:|---:|---:|---:|---:|---:|
| Released checkpoint, unsteered | 44.79 | 30.21 | 59.38 | 80.21 | 76.04 | 51.10 |
| Refined four-direction edit | 51.04 | 31.25 | NR | NR | NR | NR |
| Its random-subspace control | 50.00 | 23.96 | NR | NR | NR | NR |
| Equal-budget vision–action coupling | 48.96 | 26.04 | 61.46 | 77.08 | 82.29 | 51.07 |
| Its randomized-direction control | 60.42 | 28.13 | 60.42 | 84.38 | 77.08 | 51.27 |
| Unscaled joint vision–action edit | NR | NR | 61.46 | 79.17 | 78.13 | 50.85 |
| Visual-only edit | NR | NR | 60.42 | 81.25 | 73.96 | 50.83 |
| Action-conditioning-only edit | NR | NR | 58.33 | 86.46 | 76.04 | 51.11 |

There are **two distinct random-control rows**, not one. Neither is a random-action policy. Their scientific meaning and delivered edit magnitudes need the existing audit, not comparison to a paper's unrelated random-policy baseline. Do not select whichever published baseline is easiest to beat. Same-runtime, paired unsteered comparisons are the main efficacy reference; randomized edits test whether the chosen structure/directions add value.

The completed original core analysis did not establish a learned intervention improving task success against both its unsteered and matched-random controls. This is not proof of zero effect. Offline prediction/reconstruction gains do not establish planning-success gains. Single released-checkpoint results do not reproduce the paper's multi-training-seed/late-checkpoint aggregation.

### The 14 missing cells and their evaluation workload

| Remaining panel | Missing cells | New scientific evaluations required |
|---|---:|---:|
| Reach and Reach-Wall × visual-only, action-only, joint | 6 | 768: 576 edited + 192 paired-native |
| Push-T × refined and random-subspace | 2 | 288: 192 edited + 96 paired-native |
| PointMaze × refined and random-subspace | 2 | 288: 192 edited + 96 paired-native |
| Wall × refined and random-subspace | 2 | 288: 192 edited + 96 paired-native |
| DROID × refined and random-subspace | 2 | 192: 128 edited + 64 paired-native |
| Total | **14** | **1,824** |

The new native arms are pairing/compatibility work, not additional blank rows. Engineering checks are excluded from these totals. These are evaluation counts, not independent training observations.

### Live partial progress at 23:50 UTC

| Panel | Actual collected records | Current state |
|---|---|---|
| MetaWorld extension | Reach: native 48, visual 43, action 23, joint 12; total **126/768** | No Reach-Wall extension records yet. No complete new 96-episode cell. |
| Push-T refined panel | Native 13, refined 12, random 12; **37/288** | First 12-episode stream complete for each arm; next native stream underway. |
| PointMaze refined panel | Native 12, refined 12, random 5; **29/288** | Native/refined first streams complete; first random stream underway. |
| Wall refined panel | **0/288** | Fit has not started; waiting for Missouri's current intact MetaWorld visual stream, 7/12. |
| DROID refined panel | **0/192** | Fit complete; engineering/behavior not yet started; waiting for Nevada slot 3's MetaWorld action stream, 11/12. |

These records include partial streams and are not automatically validated final outcomes. A `DONE.json` for one stream is not a complete table cell. The last live check showed 100% utilization on each GPU; that does not establish an efficient overall schedule.

## Failure ledger — do not soften these into ordinary research uncertainty

| Failure | What happened | Repair/state and remaining risk |
|---|---|---|
| No completed new cells after hours of paid execution | The count stayed at zero of 14 across repeated user requests. | Still true at this snapshot. Partial progress is not delivery. |
| Incomplete MetaWorld deployment | Only logical slots 0–3 were initially assigned; that could cover at most 384 of 768 evaluations. | Missing task/slot pairs now assigned to continuation queues. Coverage assignments are repaired, but still not executed. |
| Premature readiness claims | Full-suite execution was discussed before fits, environments, and behavioral entrypoints were ready. | Distinguish prepared, queued, actually executing, complete, and analyzed. |
| Fragmented scheduling | Work spread across incomplete arms/tasks, with long whole-stream handoffs. | Some ordering improved; schedule is not proven optimal. Four busy GPUs did not mean short time to a finished cell. |
| Double handoff between fit and behavior | Fit helper resumed MetaWorld; CPU preparer then registered behavior too late, losing another whole batch. | DROID incurred this delay. New Wall direct chain removes the second handoff, but live passage has not yet been observed. |
| PointMaze dependency and source-path mistakes | First attempt failed D4RL's Python packaging restriction; second passed reset checks but referenced the wrong historical source directory. | v3 uses the pinned isolated runtime and correct nested source. Actual behavior is now running. Preserve failed receipts. |
| DROID fitting dependency failure | Initial fitting failed on missing `torchmetrics` in the DINOv3 import stack. | Private dependency overlay repaired; v2 fit completed. Behavioral validation still pending. |
| Wall failed before fitting, without enough error detail | Original handoff logged `ValueError` but not the information needed to reconstruct the exact cause. | More detailed errors and bounded driver-release handling added. Do not invent the original cause or call fit successful. |
| Whole-panel timeout too short | Six-hour insertion timer was marginal/inadequate for measured 288-evaluation panels. | CPU timer wrappers adopted active panels with a bounded longer deadline. This prevents premature interruption; it is not a throughput improvement. |
| Stale metadata mistaken for live process state | I hypothesized old `QUEUE_PLAN.json` deadlines would abort jobs, but actual process arguments already contained renewed deadlines. | Hypothesis disproved before deployment; no coordinator restart for it. Investigation wasted time. |
| Incorrect partial-record counter | I used `episode-*.json` globbing for refined outputs actually stored as `episode-NNN/episode.json`, producing false zero partial counts. | Corrected. The zero-completed-cell count was nevertheless still true. |
| Tests confused with experimental progress | Hundreds of tests appeared in updates while cells remained incomplete. | Software tests do not count as rollout episodes, validity receipts, or efficacy findings. |
| Poor price/performance allocation | Observed MetaWorld timings on expensive Nevada GPUs did not demonstrate a speed gain matching their roughly threefold cost per 100 episodes versus the NJ 5090. | This is an observational comparison, not a controlled hardware benchmark. Cost efficiency remains unresolved. |
| Excess operational rework | Repeated diagnosis, status collection, and piecemeal deployment consumed time. | Next session should prioritize actual complete panels, not another infrastructure expansion. |
| Latest execution code not committed | Numerous scientific adapters and deployment scripts are untracked despite branch being main. | Local checkout is essential. GitHub alone is not a complete handoff. |
| End-to-end analysis readiness not established | Refined panel queue writes `PANEL_DONE` with `analysis_complete=False`; a complete refined analysis entrypoint was not verified. | Audit/finish frozen analysis once valid data exist. Do not claim end-to-end setup is complete. |
| Unsupported completion estimates | Earlier estimates overstated readiness and did not adequately separate setup, scheduling, sample count, analysis, and available balance. | No reliable full ETA is supplied here. Use measured complete-stream rates and the actual critical path. |

No evidence from the latest operational repairs establishes that scientific data were destroyed. Equally, **not all live outputs have been verified in Google Drive**, so do not promise everything is already safely archived.

## Four currently running workers

Refresh provider state before using these endpoints; SSH ports/IPs and PIDs can become stale. All four have one GPU. Only mutate project-owned leased instances.

| ID / shorthand | GPU / location | Approx. $/hour | SSH endpoint | Active GPU work at snapshot |
|---|---|---:|---|---|
| 50626847 / NJ, slot 1 | RTX 5090, New Jersey | 0.75185 | `root@71.104.167.38 -p 53046` | PointMaze refined panel |
| 50632757 / MO, slot 2 | RTX 5000 Ada, Missouri | 0.35556 | `root@154.36.209.172 -p 16758` | MetaWorld visual stream; Wall chain waiting |
| 50638073 / NV0, slot 0 | RTX PRO 6000 WS, Nevada | 2.06222 | `root@184.186.104.194 -p 11099` | Push-T refined panel |
| 50640703 / NV3, slot 3 | RTX PRO 6000 WS, Nevada | 2.21996 | `root@184.186.104.194 -p 10081` | MetaWorld action stream; DROID behavior waiting |

CLI: `/Users/stevenyang/.local/bin/vastai`. SSH key: `/Users/stevenyang/.ssh/id_ed25519`. Do not display private key contents or provider credentials.

Local Python: `.venv/bin/python` (Python 3.10, torch 2.2.2 CPU, no torchvision).

Local lease roots, relative to `artifacts/offline_study/table-completion-20260911-v1/component-extension-v1/`:

- NJ: `rentals/slot-01/LEASE.json`
- MO: `rentals-v3/slot-02/LEASE.json`
- NV0: `rentals-v5/slot-00/LEASE.json`
- NV3: `rentals-v11/slot-03/LEASE.json`

Common remote MetaWorld root, called `MWROOT` below: `/workspace/metaworld-components-20260911-v1`.
Remote Python: `/workspace/component-python/bin/python`.
Vendor: `MWROOT/code/vendor/jepa-wms`.
Frozen MetaWorld scientific source hash: `bcf77f2b2efd8e53b5497c5224b25a452bd70e34fcb91aa78bd4a39fbcd0e217`.

Do not deploy the current local full source tree over that snapshot. Shared worker runtime includes torch 2.7.1+cu128, NumPy 2.2.6, MuJoCo 3.3.0, MetaWorld 3.1.1, Gym 0.23.1, Gymnasium 1.3.0. PointMaze has its own different historical environment.

### Process and scheduling identities

| Worker | Existing MetaWorld coordinator | Inserted panel/helper | Continuation |
|---|---|---|---|
| NJ | PID 7537, suspended | PointMaze parent 13596; adopted timer 17668 | New waiting tail 18247; old childless tail 10330 retired |
| MO | PID 4626, suspended while child completes | Current scientific child 8771; Wall direct chain 13615 | No additional tail assignments |
| NV0 | PID 3410, suspended | Push-T parent 8258; adopted timer 11157 | Tail 7512 |
| NV3 | PID 2598, suspended while child completes | Current scientific child 9382; DROID boundary waiter 9410 | Tail 7032 |

Suspended queue coordinators are intentional while an inserted job or intact child stream owns the GPU. Do not send `SIGCONT` blindly and create overlapping scientific processes.

Original slot `s` owns Reach `s` and Reach-Wall `s`. Added continuation assignments are NJ: Reach 5, Reach 7, Reach-Wall 5; NV0: Reach 4, Reach-Wall 4, Reach-Wall 7; NV3: Reach 6, Reach-Wall 6. Combined original and extra assignments cover the 16 task/slot pairs exactly once.

Actual live coordinator deadline arguments were checked: NJ/MO/NV0 `1789246320`; NV3 `1789246251.857615`. Old saved queue-plan deadlines are stale. Credit guards can stop earlier.

Nevada SSH probes timed out briefly around 23:32 UTC; both hosts subsequently responded with the same endpoints. A timeout alone is not proof of loss or grounds for restart.

## Repairs most recently deployed — operational, not new science

### Wall direct fit-to-behavior chain

Remote directory: `MWROOT/ops/wall-chain-v1`; inspect `PLAN.json`, `ADOPTED.json`, `RETIRED_HELPERS.json`, `LAUNCH.json`, and `chain.log`.

Only waiting CPU helpers 9279 and 9333 were retired after confirming no children/no `FIT_STARTED`; active scientific PID 8771 was preserved. Chain PID 13615 waits for the current whole MetaWorld stream, executes the original fit, freezes CPU inputs, runs unchanged engineering plus the 288-evaluation behavioral panel, then resumes coordinator 4626. It does not resume MetaWorld between fit and behavior.

This removes the **second** unnecessary MetaWorld batch wait, not the current intact batch. Adoption and tests passed; **real fit → freeze → behavior execution remains unverified**.

### Push-T and PointMaze timer adoption

Remote directories: `MWROOT/ops/pusht-timer-repair-v1` and `MWROOT/ops/pointmaze-timer-repair-v1`.

- Push-T: retired boundary wrapper 5951; preserved panel parent 8258; new monitor 11157; deadline `1789210130.892028`.
- PointMaze: retired boundary wrapper 12158; preserved panel parent 13596; new monitor 17668; deadline `1789211025.8466263`.

The new cap is the earlier of 12 hours from the original panel start and the existing lease deadline. Credit/$7-hour guards remain. Scientific children, source, RNG streams, and original job receipts were not restarted or edited. The monitors should write terminal/resume receipts and resume the original MetaWorld coordinators only after the panel has ended and GPU ownership is clear.

Measured 12-episode stream times: Push-T native 1,029 seconds and refined 1,034 seconds; PointMaze native 963 seconds. These are real timings, not proof all remaining arms take exactly that long.

Possible failure path still needs attention: an unexpected adopted-monitor error can leave the queue held while an orphaned panel continues. Do not assume the new monitors are battle-tested.

### NJ ordering amendment

Remote `MWROOT/ops/continuation-v1/ORDER_AMENDMENT.json` binds the old plan's SHA and preserves the same assignment set. New order is Reach 5 → Reach 7 → Reach-Wall 5, instead of Reach 5 → Reach-Wall 5 → Reach 7. Remote code `component_completion_tail_v2.py`, log `queue-v2.log`, receipt `LAUNCH_V2.json`.

This removes 48 Reach-Wall evaluations from ahead of its last Reach group. It does not shorten total work, and original Reach-Wall slot 1 still precedes the continuation. Dedicated new ordering regression coverage was not added; the component suite passed after the change.

## Task-specific roots and setup traps

| Task | Fit root | Behavioral root |
|---|---|---|
| Push-T | `/workspace/refined-pusht-prerequisite-20260911-v1` | `/workspace/refined-pusht-behavior-20260911-v1` |
| PointMaze | `/workspace/refined-nav-pointmaze-prerequisite-20260911-v1` | `/workspace/refined-pointmaze-behavior-20260911-v3` |
| Wall | `/workspace/refined-nav-wall-prerequisite-20260911-v2` | `/workspace/refined-wall-behavior-20260911-v2` |
| DROID | `/workspace/refined-droid-prerequisite-20260911-v2` | `/workspace/refined-droid-behavior-20260911-v1` |

Corresponding local artifacts are under `artifacts/offline_study/table-completion-20260911-v1/`, including `refined-pusht-behavior-worker-v1`, `refined-pointmaze-behavior-worker-v3`, `refined-wall-worker-v2`, and `refined-wall-behavior-worker-v2`.

PointMaze v3 uses `<behavior-root>/runtime/python/bin/python`: historical Python 3.11, D4RL commit `89141a689b0353b0dac3da5cba60da4b1b16254d`, MuJoCo 2.1, mujoco-py 2.1.2.14, Cython 0.29.37, NumPy 1.26.4, Gym 0.23.1, OpenCV headless 4.11.0.86. Installation used `--ignore-requires-python` for stale D4RL packaging metadata. All 97 original reset/goal input hashes matched. The historical scientific source is nested at `baseline-source/src/offline_study`, not a flattened path. Do not replace it with modern MuJoCo without protocol/parity review.

DROID v2 fit completed in about 232 seconds around 23:00–23:03 UTC. Original assets/encoder remain in `/workspace/refined-droid-prerequisite-20260911-v1`; canonical hardlinked assets at `/workspace/jepa-runtime/droid-assets-20260907-v1`. DINOv3 source commit `6876159a11b4df116f30f667f8c9888617df0751`. Its private overlay includes torchmetrics 1.8.2 and lightning-utilities 0.15.2; do not modify the shared active MetaWorld runtime. Imported model and 128-recording/512-prefix CPU input contract passed. DROID's intervention site uses block 6 of 12, horizon 3, 256 patches × 1,024 features; do not reshape another task's bank into it.

## Scientific invariants

- **96 total evaluations per simulation condition, not 96 per GPU. DROID: 64 total per condition.**
- Preserve identical scenario/reset/goal inputs and the paired native baseline for each comparison.
- Keep each scenario's arms on the same physical GPU under the frozen protocol.
- Preserve whole 12-episode simulation RNG streams. Do not naively resume partial streams, reseed, pad duplicate samples, or count candidates/windows as independent trajectories.
- No independent experiment multiplexing on one GPU; unique output directories.
- Behavioral runs use frozen FP32 and the native planner settings, including 300 CEM candidates and 15 iterations. Do not silently change horizon, actions, precision, or candidate count for speed.
- Fits may use the already-approved BF16 setup. Small precomputed response operators are the refined intervention; the previous expensive online operator is not authorized for revival.
- Current evaluation is released-data development/replication, not fresh-family confirmation. Existing exposure does not disappear by reshuffling rows.
- DROID recorded-plan scores, simulation success rates, and offline prediction errors are different endpoints.
- Published JEPA-WM aggregate/final-checkpoint rows are contextual, not interchangeable with same-runtime paired unsteered baselines.
- Do not alter frozen candidate selection, fitting inputs, comparison families, or significance thresholds based on observed outcomes.

## Analysis still required

MetaWorld analysis implementation exists in `src/offline_study/metaworld_component_analysis.py`. Inspect its frozen plan and contrast definitions rather than inventing a new analysis.

The refined panel queue can finish execution while writing `analysis_complete=False`. A complete final refined-analysis entrypoint has **not** been verified. The prior plan calls for a 12-contrast family, 20,000 cluster bootstrap samples, seed `2026091102`, and Bonferroni-adjusted 95% intervals; verify the exact definitions in `docs/REFINED_SIX_TASK_COMPLETION.md` before implementation. MetaWorld uses seed `2026091101` and 12 contrasts across the two tasks; verify unit of clustering and all pairing checks against its plan.

For a finished cell, require complete expected scenarios, valid stream receipts and hashes, no duplicate or missing scenario IDs, paired-input compatibility, and correct denominator. Do not paste partial estimates into final-result cells. Do not equate `PANEL_DONE`, a process exit code, or a fit receipt with completed final statistical analysis.

## Code and Git state

Branch **main**, ahead of `origin/main` by three commits at the check.
HEAD: `01e24ddcb5509c9f8747d127ca5ad9c6c7e7572a`.

Recent commits:

- `01e24dd` Limit public CI to portable CPU tests
- `44b1189` Merge remaining CPU test workflow into main
- `f263063` Merge preserved curved-action follow-up notes into main
- `2ba4803` Present verified JEPA-WM results and ten evidence-backed findings

Tracked dirty files: `README.md`, `configs/study.json`, `docs/EXPERIMENT_PLAN.md`, `jax_scaling_notes.md`. No staged changes at the check. Many untracked docs, reports, scripts, tests, and scientific modules are essential to current execution. Run `git status --short --branch`; do not use a destructive reset or blanket staging.

Latest repair files:

- `scripts/vast/component_fit_behavior_chain.py` — new direct-chain/timer adoption helpers.
- `scripts/vast/deploy_wall_direct_chain.py` — deployment/adoption operations.
- `scripts/vast/component_completion_tail.py` — continuation ordering amendment support.
- `tests/test_component_fit_behavior_chain.py` — new helper tests.
- `reports/SIX_TASK_TABLE_PROGRESS_20260911.md` — repair/status record.
- `/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md` — external shared-resource record.

Other critical untracked execution files include `component_extension_fleet.py`, `component_extension_queue.py`, `component_boundary_job.py`, `deploy_component_completion_tail.py`, `refined_panel_queue.py`, `prepare_remaining_refined_behavior.py`, the task fit/stage/restore scripts, and `collect_refined_prerequisite.py` under `scripts/vast/`.

Critical scientific modules include `metaworld_component_behavior.py`, `metaworld_component_analysis.py`, `refined_task_behavior.py`, `refined_task_fit.py`, `refined_fit_inputs.py`, `navigation_refined_pipeline.py`, `navigation_rank_reconstruction.py`, `droid_fixed_response.py`, `droid_fixed_response_fit.py`, and `droid_fixed_response_behavior.py` under `src/offline_study/`.

No commits or pushes were made for the latest repairs. The handoff document itself is also newly created local work.

### Validation already performed

- Latest full local suite: **748 tests run, 746 passed, 2 skipped**, about 92 seconds. Log `/tmp/jepa-scheduler-final-validation-20260911.log`.
- Latest targeted component suite: **32 passed** after the ordering patch.
- Six new direct-chain/timer tests cover identity checks, unstarted-fit adoption, refusal of unready state, cleanup, no intervening resume, and preserving active-panel execution during timer adoption.
- The two local skips require torchvision, absent on this laptop. Those exact tests previously passed on NJ in an isolated CPU check, zero skips: `test_navigation_input_check.NavigationInputTests.test_selected_frames_preserve_subset_mapping_actions_pixels_and_rng` and `test_pointmaze_training.PointMazeTrainingTests.test_actual_native_validation_pixels_actions_nested_rows_and_rng`.
- Worker receipt: `MWROOT/ops/worker-parity-tests-v1/RESULT.json`.
- `git diff --check` passed before this handoff.

These are software checks, **not** hundreds of completed experimental rollouts.

## Storage, credentials, preservation, and billing guards

The last local disk check showed approximately **14 GiB available, 97% capacity used**. Avoid downloading large duplicate archives or raw datasets. Google Drive storage is authorized; do not use paid GPUs merely as an inefficient storage substitute.

Google Drive folder ID: `14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48`.
Uploader: `scripts/vast/final_preservation_drive.py`, using the configured rclone Google Drive OAuth connection. Read configuration paths from the implementation; never print tokens.

Collector: `scripts/vast/collect_refined_prerequisite.py`. It preserves compact source/ops/fit/reference/engineering/condition/log/result evidence, excludes reproducible public-download duplicates, checks archive entries, and verifies full Drive byte readback. **A registered collector is not proof the current panel has been uploaded.** Push-T fit had a previously verified Drive archive with file ID `1H7UuH_cf1PTVu6hO1MtDpG9-BM46f2CJ`; verify receipts rather than inferring coverage of subsequent behavior.

The Hugging Face token is stored locally in the ignored `.env`. Do not copy it into this document, logs, commits, or command output. Provider/Drive credentials and SSH private keys must remain secret.

Routine worker release must follow result collection, expected marker/checksum verification, and Drive readback. Existing emergency credit/deadline guards may stop machines while **retaining disks** before a full archive succeeds. Do not destroy those disks on the assumption that stopped means safely archived. The user's instruction was to preserve necessary results before routine teardown.

Local `launchctl` jobs supervise rental guards and collectors. **If the Mac sleeps or loses power, local monitoring can stop while remote jobs keep running.** Do not assume remote execution alone enforces every local billing guard.

Relevant labels have prefix `com.steven.jepa.`; inspect live state rather than trusting old PIDs:

- `components.rental.0.v5`, `components.rental.1`, `components.rental.2.v3`, `components.rental.3.v11`
- `pusht.prerequisite.collect.20260911`, `pusht.behavior.collect.20260911`
- `pointmaze.prerequisite.collect.20260911`, `pointmaze.behavior.collect.20260911`, `.v2`, `.v3`
- `wall.prerequisite.collect.20260911`, `wall.prerequisite.collect.v2.20260911`, `wall.behavior.collect.20260911`, `wall.behavior.collect.20260911.v2`
- `droid.prerequisite.collect.20260911`, `droid.prerequisite.collect.v2.20260911`, `droid.behavior.collect.20260911`
- `refined.fit.capacity.20260911` is an older failed Michigan-reuse guard; inspect before touching.

Automatic capacity rental was disabled; no automatic credit purchase is authorized. Michigan instance 50546171 reuse failed availability and was canceled; do not revive old confirmation work. Shanghai instances 50135088/50135089/50135090 are explicitly excluded. Other retained/exited instances are not available project GPUs merely because idle.

## Safe inspection and counting tips

`scripts/vast/component_extension_fleet.py` supplies `provider('show', 'instances')` and `connection(row)` helpers. Print only whitelisted state/cost/location/connection fields: provider objects may contain sensitive fields.

- MetaWorld records: `MWROOT/results/**/episode-*.json`.
- Refined records: `<behavior-root>/conditions/**/episode.json` — **different filename pattern**.
- Complete stream markers: `conditions/<arm>/shard-N/DONE.json` plus valid report hashes.
- Inspect actual `/proc/<pid>/cmdline` and parent/child process relationships; saved plans can be stale.
- `PANEL_STARTED` can include engineering time; it does not prove scientific episodes have begun.
- `PANEL_DONE` does not prove analysis is complete.
- Do not restart a healthy stream merely because its coordinator is suspended or SSH briefly times out.

## Recommended takeover order

1. Confirm sole ownership/takeover, read AGENTS and the GPU board, preserve the dirty checkout, refresh the four providers and available credit.
2. Verify DROID actually transitions into engineering/behavior after its current MetaWorld stream. It was not running behavior at the last snapshot.
3. Observe Wall's new direct chain through its first real fit → freeze → behavior transition. Adoption receipts alone are insufficient.
4. Confirm Push-T and PointMaze continue making progress under adopted timers, without orphaned GPU processes or duplicate arms. Leave healthy scientific children alone.
5. Audit the remaining assignment order and measured time to complete cells. Existing continuation assignments fix missing coverage, not all scheduling inefficiency. Any redeployment must preserve pairing, RNG streams, output ownership, budget, and provenance.
6. Finish and validate complete cohorts; run the frozen analyses. Audit/implement the missing refined-analysis automation without changing the statistical plan in response to results.
7. Update the table with complete, verified values and explicit uncertainty. Report cells complete, scientific episode counts, job purpose, and measured timing separately.
8. Preserve results/source/logs to Drive with full verification before routine teardown. Keep the user informed of billing and actual funding runway; do not promise end-to-end completion from the current balance.
9. After review, make scoped main-branch commits so a future clone can reproduce the implementation. Do not commit secrets, raw videos/checkpoints, unrelated archives, or unreviewed material wholesale.

The outgoing session made no additional experiment changes while preparing this handoff. **Existing jobs and billing remain active subject to their existing guards.**
