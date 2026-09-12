# Closed-loop CEM adapter for the JEPA driving cell — implementation report (2026-09-03)

Author: closed-loop adapter subagent (autonomous session, user away). Box: **box 2 only** (Vast 49766237 "cgs-pilot-2",
`ssh -p 45460 root@70.27.250.55`, root `/root/cgs-pilot`). Box 1 was never touched.
Governing text: `cross model design jepa.md` — "Target hierarchy" (discovery label = closed-loop simulator outcome under the
model's own planner), "Step 1: collect real outcome labels", "Step 8", claim-ladder row **C0** (closed-loop part), the
"Enhanced order" under "Execution order" (item 1), and the BINDING **"Label-first principle"** section (from the COAST paper,
`references/COAST_arXiv_2605.17144.pdf` §3.2 labels by simulation outcome, §4.1 fitting/held-out rollouts, outcome mix by
checkpoint choice). Every number in this report is re-derived from `rollouts.jsonl` records, never from my own summary.

STATUS LINE (kept current at the end of the file, section 9).

**Label amendment during the session.** The first implementation used the spec's researcher-authored progress rule
(`safe_pass` = no contact and progress ≥ 0.9 × the hazard-free rollout's progress). After its smoke test the coordinator
issued **"Label-first principle, amendment 1"** (user decision): the label must be the simulator's NATIVE predicates only,
exactly as COAST used LIBERO's native `is_success` (COAST App. A.8). Section 2b describes the native-label implementation
that now runs; the progress-rule quantities survive only as `descriptive_*` fields. Section 5 has the progress-rule smoke
(seed 3) for the record; section 5b the native smoke (seed 9).

## 1. What was built

| file | role |
|---|---|
| `scripts/cgs_pilot/closed_loop_bridge.py` | MetaDrive-side JSON-lines server (runs in `/opt/conda/envs/metadrive/bin/python`, no torch). Commands `init` / `reset` / `step` / `close`. `reset` replays the generator prefix exactly as `metadrive_hazard_pilot.run_episode` does; `step` executes model-step chunks (3 sim steps each) and returns per-sim-step ego state, contact flags, hazard gap, lane progress, plus the RGB frame and the semantic hazard / corridor masks after every model step. |
| `scripts/cgs_pilot/closed_loop_rollout.py` | Model-side driver (`/opt/conda/bin/python`): loads the frozen predictor (`model_action_sensitivity.load_model`, encoder = frozen DINOv3 via the pilot loader), talks to the bridge through a pipe, runs the CEM in the predictor's currency, records every episode to `rollouts.jsonl`, writes `summary.json` (rates, scene-bootstrap CIs, C0 gate). Modes `run` and `summarize`. Resumable (skips finished `(scene, level, policy, episode seed)`). |
| `scripts/cgs_pilot/run_closed_loop_c0.sh` | Box runner, `setsid nohup`-safe, markers in `logs/drive_closed_loop.log`: `CLOSED_LOOP_SMOKE_DONE` / `CLOSED_LOOP_SMOKE_FAIL`, `CLOSED_LOOP_PILOT12_DONE`, `CLOSED_LOOP_PILOT_DONE`. |

Nothing in `metadrive_hazard_pilot.py`, the registered chain, or the generator's outputs was modified. The bridge imports
`make_env`, `reset_scene`, `lane_pose`, `spawn_hazard_at`, `pose_pedestrian_animation`, `settle_physics`, `apply_row_factors`,
`ego_state`, `ego_lane`, `hazard_gap`, `corridor_mask`, `semantic_rgb` / `colour_mask`, `project`, `frame_rgb`,
`raw_actions_from_chunks`, `prefix_travel_of`, `provenance` from it. The driver reuses `token_groups.groups_from_driving_masks`
and `protocol.sha256_array` / `append_jsonl`. md5 of every synced file was verified identical on both sides
(`closed_loop_bridge.py`, `closed_loop_rollout.py`, `run_closed_loop_c0.sh`, and the reused `metadrive_hazard_pilot.py`,
`token_groups.py`, `model_action_sensitivity.py`, `protocol.py` were already identical between local and box 2).

## 2. Design decisions (and where they deviate from the spec)

**Branching from the context state = deterministic prefix replay.** MetaDrive has no Bullet save/restore
(`DRIVING_METADRIVE_SPIKE.md` §4), so `reset` re-does `reset_scene(seed, map_cfg)` → `set_velocity(v0)` → hazard spawn at
`s_ahead + prefix_travel` (`lane_pose`) → `pose_pedestrian_animation` → `PREFIX_STEPS = 5` steps of `[0, prefix_throttle]`,
with the module factors (`hazard_body`, `fov_deg`, `settle_steps`, `prefix_throttle`) set from the manifest row by
`apply_row_factors` (v0.8 rows → v0.8 values). The hazard object is removed with `clear_objects(force_destroy=True)` before the
next reset, as `run_episode` does at its end (pooled reuse would re-apply `setMass(0)` / the ghost mask → divergent physics).

**Replay verification (every rollout record carries it, key `replay`).** When the scene's cell npz is on the box
(`source = cell_npz`): `ctx_state_max_dev` = max |replayed − stored `ego_states[5]`| over the 7-vector, `initial_state_max_dev`,
and the context-frame pixel difference. When the npz is not there yet (`source = manifest_only`; the merged manifests on
box 2 point at `../../drive_factorial/...`, which arrives with `PULL_DONE`): bit-exact `initial_state_sha256` match
(`protocol.sha256_array` of the replayed initial 7-vector), `context_speed_mps`, `prefix_travel_m`, `hazard_pose_xyzh`,
`context_longitudinal_gap_m`, `context_lateral_m` — together these pin x, y, heading, speed at the context step. `replay_ok`
= sha match and every available deviation ≤ 1e-6; rollouts with `replay_ok == false` are excluded from `summary.json` and
listed under `replay.excluded`. `replay_max_dev` is the max over the checks.

**Planner currency = `steered_planner_ranking.py`'s progress goal**, `cost(a) = ||P(z_ctx, a)[-1] − z_goal||²` summed over
the hazard ∪ corridor tokens. Closed-loop extensions (deviations, all documented in the code header):
- *Rolling goal.* The open-loop currency uses one fixed goal (the h0a1 true future, 3 steps after the context). In the loop
  the ego moves, so `z_goal` at a replan = encoded frame of the scene's **hazard-free always-throttle** rollout at model step
  `executed_steps + HORIZON` ("be where the free-lane throttle rollout is, 3 imagined steps from now"). The reference rollout
  is generated once per scene by the bridge (model-free, `n_replans × execute_steps + 3` model steps).
  Deviation: the goal source is the truly hazard-free scene rather than the level-0 (sidewalk-pedestrian) cell; the
  registered currency calls level 0 "hazard-free" because the ego lane is free. The difference is a few sidewalk tokens.
- *Token set.* `token_groups.groups_from_driving_masks` on the **live context frame** (semantic hazard silhouette dilated by
  one patch ∪ ego-lane corridor polygon) instead of the scene-union of final-frame masks over the 8 cells (unknowable before
  rolling out). Hazard-free rollouts therefore use corridor tokens only. `n_tokens` and `n_hazard_patches` are recorded per
  replan.
- *Action chunk parameterisation* = the generator's: `HORIZON = 3` model steps × `[steer, throttle_brake]`, **steer fixed at
  0** (the generator and every training clip use steer = 0 exactly: `TRAIN_DIST["steer_sigma"] = 0`), `throttle_brake`
  clipped to `[−1, 0.5]` (the generator's brake / throttle values; the released `max_norms` clip is the 7-D DROID analogue).
  `--steer-max > 0` exists as an exploratory, off-distribution option and was **not** used.

**CEM (own compact implementation, same shape as the released jepa-wms CEM).** Released config
(`configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml`):
iterations 15, num_samples 300, num_elites 10, horizon 3, var_scale 0.1, momentum 0, `num_act_stepped 3`, L2 objective.
Mine: Gaussian population over the 3 throttle_brake values (init mean 0 = coast, init std 0.5, std floor 0.02, clipped to
[−1, 0.5]), **samples 64, elites 8, iterations 4**, momentum 0, executed action = final mean (rounded to the record).
Why smaller: the search space is 3-D and a batched 3-step unroll of the width-512 predictor costs ≈ 5 ms per candidate on the
shared 3090 (67 candidates × 3 steps = 0.35 s, 3.1 GB peak), so 300 × 15 would be ≈ 20 s per replan. The two generator
chunks (brake, throttle) and the running mean are evaluated with every population as **reference candidates** (their costs
are recorded → the open-loop safe choice `cost(throttle) > cost(brake)` at every replan) but never enter the elite set.
Every hyper-parameter is stored in each record (`cem_config`) and in `summary.json`.
- `--execute-steps 1` (receding horizon: replan after each executed model step = 0.3 s) instead of the released
  `num_act_stepped 3`, because with the hazard 9 m ahead at 8.3 m/s the throttle contact happens at future sim step 7–8, i.e.
  inside the first 3-step chunk; executing one step gives the planner 2–3 decisions before the body. `--execute-steps 3`
  reproduces the released behaviour if wanted.
- Episode = `--n-replans 8` decisions (8 model steps = 2.4 s; the hazard-free throttle rollout travels ≈ 24 m, hazard at 9 m),
  stopping at the first contact.

**Labels and denominators.** `collision` = `crash_human | crash_object` at any sim step. `safe_pass` = no contact and
`progress_m ≥ 0.9 × denominator`; `failure_progress` otherwise. `progress_m` = longitudinal coordinate along the **context**
lane (`lane.local_coordinates`) minus its context value (the first blocks are straight and collinear; Euclidean `disp_m` is
stored alongside). Two denominators are stored so nothing has to be re-run:
- `label` (primary, the spec's wording "the hazard-free rollout's progress"): the **same model's hazard-free CEM rollout with
  the same episode seed** (benchmark policies use the CEM's episode-seed-0 hazard-free rollout).
- `label_ref`: the hazard-free **always-throttle** progress after the same number of model steps (`progress_ref_m`, model-free).
- `free_goal_success` (hazard-free CEM episode): no contact, no `out_of_road`, `progress_m ≥ 0.9 × progress_ref_m`.
Benchmarks per scene × level: `always_throttle` and `always_brake` (the negative benchmark); their labels are reported in the
same table and never pooled with the CEM rows.

**Temporal precedence fields (coordinator scope addition, design-doc Step 1 "truncate before the first avoidance or
contact response").** Each record has `first_contact_replan_idx`, `first_brake_replan_idx` (first executed
`throttle_brake < 0`), `divergence_replan_idx` = first replan whose executed chunk differs from the hazard-free reference
rollout's chunk at the same index by more than `DIVERGENCE_TOL = 0.1` (reference = the same policy's hazard-free rollout,
same episode seed for CEM; `chunk_dev_per_replan` stores the raw per-index deviation so the threshold can be changed later;
`divergence_replan_idx_vs_throttle` compares with the constant throttle chunk), `pre_outcome_replans` =
`truncation_mask(record)` = replan indices strictly before `min(divergence, first contact)`, and `latent_path` → an npz under
`<out>/latents/` with `z_context [n_replans, 256, 1024] fp16` (frozen-encoder latent of the context frame at each replan),
`token_mask [n_replans, 256]`, `goal_index`, `executed_tb`. `closed_loop_rollout.truncation_mask(rec, tol=None)` is the helper
(re-thresholds from `chunk_dev_per_replan` when `tol` is given).

**Label-first principle (binding).** The driver refuses any selected seed that appears in a `confirmation_seeds*.txt` next to
the stimulus (or `--confirmation-file`); `--allow-confirmation` is reserved for the separate registered confirmation run and
was not used. The pilot uses `drive_factorial_merged/discovery_seeds_common.txt` only. No planner parameter was tuned on
outcomes; a solid-in-lane safe-pass rate outside 0.20–0.90 is recorded as a C0 FAIL (registered remedies: an earlier-epoch
checkpoint of the same run, model scale, more training clips — not planner tuning).

**Bridge design.** One `Bridge` (subprocess with `stdin`/`stdout` pipes; the bridge dup's fd 1 to fd 2 at start so panda3d /
MetaDrive chatter cannot corrupt the protocol) per driver process. Frames travel as base64 raw uint8 (256×256×3 RGB, BGR
already flipped by `frame_rgb`), masks as packbits. Per model step the bridge renders the RGB and the semantic camera and
projects the corridor polygon exactly at the same points in the step sequence as the generator (frame steps only), so the
physics sequence is identical to `run_episode`. Measured: first reset 8.2 s (asset load), later resets ≈ 0.6 s, a 3-sim-step
model step incl. both renders ≈ 25 ms.

**Compute hygiene.** `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`; the pilot runs 2 drivers + 2 bridges (4 CPU processes); GPU
eval batch 64 (≈ 3 GB per driver + 1.2 GB per EGL context). No process on the box was killed.

## 2b. Native label (Label-first principle, amendment 1) — what runs now

- **Termination = MetaDrive's defaults.** The generator's `make_env` sets `crash_human_done / crash_object_done /
  crash_vehicle_done / out_of_road_done = False` for its scripted futures; the bridge's `init` (`closed_loop: true`) sets the
  four back to MetaDrive's default `True` on the env config (the generator file is untouched) and every episode runs until the
  env's own `terminated | truncated`. Horizon: MetaDrive's shipped default is `None` (`base_env.py` line 83), which would let a
  stalled always-brake episode run forever, so the generator's `horizon = 1000` env steps is kept as an **engineering guard**
  (`horizon_guard` in `bridge_provenance.json` / `run_config.json`), surfacing as the native `max_step` predicate; it cannot
  create a success. A driver cap of 400 model steps (1200 env steps) sits above the guard (`max_model_steps_guard`).
- **Label.** `success` = `info["arrive_dest"]` (`TerminationState.SUCCESS`, `_is_arrive_destination`: within ±5 m of the
  final lane's end and inside the road) at the env's done with **no crash predicate at any earlier or the same step**;
  `failure` otherwise, with `failure_reason` = first native predicate by priority `crash_human > crash_object > crash_vehicle
  > crash_building > crash_sidewalk > out_of_road > max_step > driver_cap`. Every record stores the full `done_info` dict at
  the final step (`native_done_info_final`), the OR over steps (`native_done_info_any`), the first env step of each predicate
  (`native_first_step`), `native_route_completion` (`info["route_completion"]`), `env_steps_after_context`,
  `env_episode_step_final`, `terminated`/`truncated`, `arrive_dest_env_step`, and `success_and_crash_same_step` (counted as
  failure). The former progress-rule quantities are kept as `descriptive_progress_m`, `descriptive_progress_ref_m`,
  `descriptive_progress_free_cem_m`, `descriptive_label_progress_rule`, `descriptive_label_ref_progress_rule`,
  `descriptive_free_goal_success`, `descriptive_min_gap_m`, … and never enter the label.
- **Reachability check, never a goal of ours.** Per scene the hazard-free always-throttle reference (model-free) must itself
  reach `arrive_dest`; otherwise the scene is refused and recorded in `skipped_scenes.jsonl` (`reference_reachable = false`,
  with the native reason). With steer fixed at 0 (the generator's chunk space) this refuses every scene whose map contains a
  curved block: `scene_map_config(seed)` = `"S"` + two draws from `{S, C}`, so only `SSS` maps are drivable straight to the
  destination. Of the 54 common v0.7 discovery seeds, 12 are `SSS` (9, 10, 19, 35, 50, 57, 68, 72, 77, 85, 94, 115; 20 SCS,
  13 SCC, 9 SSC). The runner does not pre-filter by map: the reference check decides and the refusals are data.
- **Eligibility for fitting** (COAST App. A.8): ≥ 3 successes and ≥ 3 failures per model × level over the CEM episodes
  (`summary.json → eligibility`). The C0 gate keeps its frozen thresholds but on native success: solid-in-lane success rate
  in [0.20, 0.90] and hazard-free CEM success ≥ 0.50.
- **CEM budget for native episodes**: samples 32, elites 6, iterations 3 (a native episode is 40–330 replans instead of 8;
  ≈ 106 candidate-unrolls per replan ≈ 0.6–1.4 s on the shared GPU). Benchmark policies (no decisions) step 10 model steps
  per bridge call. Everything else (rolling goal, token set, execute-steps 1, steer 0, clip [−1, 0.5]) is unchanged.
- **Reachability / timing test under native termination** (`artifacts/drive_closed_loop/_reach_test.py`, MetaDrive env,
  direct, no model): seed 9 (`SSS`) hazard-free always-throttle from the context state → `arrive_dest` at env step **128**
  (progress 221.6 m, final speed 22.4 m/s, route_completion 0.985, final lane 54.3 m; 1.2 s wall); seeds 10/19/35 (`SSS`):
  122/138/144 steps; hazard-free **coast** stalls → `max_step` at post-context step 995 (= 1000 incl. the 5-step prefix);
  in-lane solid pedestrian + throttle → `crash_human` at step 8 (terminated); + brake → stops (progress −4.1 m: it rolls back
  slightly) → `max_step` at 995 (11.2 s wall); ghost cone / sidewalk pedestrian + throttle → `arrive_dest` at 128 like the
  free lane; seed 3 (`SCC`) → `out_of_road` at step 77 (progress 105.8 m). The guard (1000) is ≈ 8× the arrival step.

**Runner (full mode, `run_closed_loop_c0.sh`).** Stage S native smoke (seed 9, `stop-on-unreachable`) →
`CLOSED_LOOP_NATIVE_SMOKE_DONE|_FAIL`; waits for the legacy progress-rule pilot to finish (process budget), then stage 1 =
seed-0 pair on the v0.7 common discovery list → `CLOSED_LOOP_C0_SEED0_DONE`; stage 2 = seeds 1–2 (two models at a time) →
`CLOSED_LOOP_C0_V07_DONE`; stage 3 = v0.9 randomised set discovery scenes (`drive_v09_merged/rand/arm{A,B}`, waits for the
merge line in `logs/drive_v09_eval.log`) for all six models → `CLOSED_LOOP_C0_V09_DONE`. `c0_table.{json,md}` is rewritten
after every stage (`closed_loop_rollout.py table`). Each driver prints a `PROJECTION` line in its `driver.log` after 20
episodes (s/episode, min/scene, hours left). Layout: `artifacts/drive_closed_loop/<set>/<arm>_seed<s>/…` with `<set>` ∈
{`v07`, `v09`}; native smoke under `smoke_native/armA_seed0`; the legacy progress-rule outputs stay at
`artifacts/drive_closed_loop/{smoke_armA_seed0, armA_seed0, armB_seed0}`.

**Legacy run and the "do not kill" rule.** The progress-rule 12-scene pilot (first 12 common discovery seeds, 64/8/4 CEM,
8 replans) was already running under the first runner when the amendment arrived and was left alone. Its runner script was
overwritten on disk by the full-mode runner (bash reads scripts by byte offset): I verified from `/proc/<pid>/fdinfo` that the
legacy bash (pid 8971) will resume at byte 6008 of the new file, i.e. inside `… fi\nfi\n`, which is a syntax error → it
exits after its already-parsed loop (its `CLOSED_LOOP_PILOT12_DONE` marker is written inside the loop; its "all" stage
launches the new driver, which refuses to append native records to a progress-rule directory — `LABEL-MODE GUARD` — and
exits 0). The new runner (pid 10430) resumes at byte 6015 = the start of `# ---- stage prerequisites`, in sync. Rule for the
future: never edit `run_closed_loop_c0.sh` while a runner is alive; copy to a new file name instead.

## 3. Environment fixes needed on box 2 (deviations from "nothing else changed")

- The GPU fix's `pip install torch==2.7.1 ... --index-url .../cu126` was a no-op (`2.7.1+cu128` already satisfied the pin;
  rc=0 in 1 s) and `torch.cuda.is_available()` was False with error 804 ("forward compatibility was attempted on non
  supported HW"). Diagnosis (mine, confirmed by the coordinator): the image's CUDA forward-compat `libcuda.so.570` in
  `/usr/local/cuda-12.8/compat` was on the `ld.so.conf` path; GeForce cannot use it. The coordinator disabled that ld.so.conf
  entry (`CUDA_FIX` line in `logs/fix_box2_gpu.log`); torch 2.7.1+cu128 then reports CUDA True (matmul verified). I made no
  ldconfig change.
- The model env lacked declared `jepa-wms` dependencies that the generic `hubconf` loader imports even for our driving model
  (`setup_box2.sh` installed them only into the MetaDrive env or not at all). Installed into `/opt/conda` with `pip`
  (each logged with a timestamp in `logs/drive_closed_loop.log`): `gym==0.23.1`, `gymnasium==1.3.0` (+ `gym-notices`,
  `Farama-Notifications`), `pygame 2.6.1`, `pymunk==6.8.0`, `shapely`, `scikit-image` (+ `lazy-loader`, `tifffile`),
  `matplotlib`, `lpips`, `seaborn`, `datasets`, `clusterscope`, `torchmetrics`. Every one is a pin from
  `vendor/jepa-wms/pyproject.toml`; `numpy 2.2.6`, `torch 2.7.1+cu128`, `torchvision 0.22.1+cu128` were verified unchanged
  before and after. Without these the orchestrator's later GPU jobs (identity geometry, v0.9 eval, Panel B) would have
  failed at the same import.

## 4. Standalone bridge test (before any model; `artifacts/drive_closed_loop/_bridge_test.py` on box 2)

Scene seed 3 (`map_cfg SCC`), v0.7 defaults (hazard 9 m, prefix throttle 0.2, static body), model env ↔ MetaDrive env:
- hazard-free reset: context speed **8.285331726074618 m/s**, prefix travel 3.902998924255371 m — the box-1 manifest row of
  seed 3 (`artifacts/cgs_pilot/drive_eval/armA_seed0/manifest.jsonl`) stores `context_speed_mps 8.285331726074618` and
  `context_longitudinal_gap_m 9.000000953674316`; the bridge's in-lane reset returned `lon 9.000000953674316` — identical
  digits across boxes.
- hazard-free throttle 11 model steps: progress 2.54, 5.20, 8.00, 10.92, 13.97, 17.15, 20.46, 23.89, 27.46, 31.15, 34.97 m
  (final speed 12.99 m/s); after 8 model steps: 23.89 m.
- in-lane solid pedestrian, throttle: first contact at sim step 8 after the context (generator: `first_contact_step 7`,
  0-based → same event), min gap 0.0, ego stops on the body (0.41 m/s). Brake: no contact, min gap 3.495 m, progress 2.81 m,
  final 0.109 m/s. Ghost cone, throttle: no contact, min gap 0.0 (passes through), progress 23.89 m (= hazard-free).
- determinism within the process: repeated hazard-free reset 2.3e-13; in-lane vs hazard-free context state 5.7e-14;
  two in-lane resets 1.1e-13; re-rendered context frame 0 differing pixels.

## 5. Smoke test (1 scene × armA_seed0 × {hazard-free, 4 levels} × {always_throttle, always_brake, cem}, 1 episode seed)

Progress-rule label (pre-amendment), scene seed 3 (`map SCC`, hazard 9 m), model `armA_seed0` (arm A physics: pedestrian
solid, cone ghost), CEM 64/8/4, 8 replans × 1 executed model step, goal = hazard-free throttle rollout frame 3 steps ahead.
Runner marker `CLOSED_LOOP_SMOKE_DONE` at 17:32:14 UTC (third launch; the first two died on missing `gym` / `torchmetrics`,
section 3). Every number below is read from `artifacts/drive_closed_loop/smoke_armA_seed0/rollouts.jsonl` (15 records) with
`_inspect.py`; `summary.json` agrees.

Replay verification: all 15 rollouts `replay_ok = true`, source `cell_npz` (the freshly generated seed-3 cells);
`ctx_state_max_dev` ≤ 2.27e-13 (levels 1 and 3: 0.0; always-brake level 3: 7.1e-15); context frame vs stored cell: 0 differing
pixels for every hazard level (the hazard-free reset differs from the h0a0 cell by 1356 px — that is the sidewalk pedestrian
missing from the hazard-free frame, as expected). Generator cross-check: the box-2 regeneration of seed 3 reproduces box 1's
`context_speed_mps 8.285331726074618` and `context_longitudinal_gap_m 9.000000953674316` digit for digit.

| level | policy | contact | progress (m) | label (den = hazard-free CEM 18.24 m) | label_ref (den = throttle 23.89 m) | min gap (m) | wall (s) |
|---|---|---|---|---|---|---|---|
| hazard-free | always_throttle (11 steps) | no | 34.97 (23.89 at step 8) | reference | reference | – | 8.7 (first reset) |
| hazard-free | always_brake | no | 2.81 | failure_progress | failure_progress | – | 0.9 |
| hazard-free | cem | no | 18.24 | failure_progress (0.76 of throttle) | failure_progress | – | 26.0 |
| 0 sidewalk ped | always_throttle | no | 23.89 | safe_pass | safe_pass | 0.974 | 1.0 |
| 0 sidewalk ped | always_brake | no | 2.81 | failure_progress | failure_progress | 3.628 | 1.0 |
| 0 sidewalk ped | cem | no | 18.95 | safe_pass | failure_progress | 0.974 | 26.6 |
| 1 in-lane ped (solid) | always_throttle | **yes, sim step 8** | 6.49 | collision | collision | 0.0 | 0.8 |
| 1 in-lane ped (solid) | always_brake | no | 2.81 | failure_progress | failure_progress | 3.495 | 1.0 |
| 1 in-lane ped (solid) | cem | no | 4.15 | failure_progress | failure_progress | 2.240 | 27.4 |
| 2 mirror ped | always_throttle | no | 23.89 | safe_pass | safe_pass | 0.974 | 1.0 |
| 2 mirror ped | always_brake | no | 2.81 | failure_progress | failure_progress | 3.628 | 1.1 |
| 2 mirror ped | cem | no | 16.09 | failure_progress | failure_progress | 0.974 | 27.4 |
| 3 in-lane cone (ghost) | always_throttle | no | 23.89 | safe_pass | safe_pass | 0.0 (passes through) | 1.0 |
| 3 in-lane cone (ghost) | always_brake | no | 2.81 | failure_progress | failure_progress | 3.495 | 1.0 |
| 3 in-lane cone (ghost) | cem | no | 16.78 | safe_pass | failure_progress | 0.0 | 26.2 |

What the CEM traces show (`replans[*].cem`, costs = squared L2 over 64–95 hazard ∪ corridor tokens):
- **Solid in-lane pedestrian (level 1): the arm-A model brakes at the first replan** — executed throttle_brake −0.79, −0.91,
  −1.00 at replans 0–2 (cost brake 3389 vs throttle 8524 at replan 0; open-loop safe choice True at replans 0–6), the car
  stops at a 3.2 m gap, then creeps (+0.44, +0.49, +0.48, +0.50 at replans 4–7; gap 3.17 → 2.24 m) as the cost landscape
  flattens at standstill (replan 7: throttle 5374 < brake 5702). No contact in 8 replans, progress 4.15 m →
  `failure_progress` (a lane-centred immovable body cannot be passed with steer = 0).
- **Ghost cone in lane (level 3): no brake at replan 0** (cost brake 2954 vs throttle 462, safe choice False — arm A's cone
  is a ghost and the model reflects it), a hard brake at replan 1 (−0.63 at 1.6 m gap) then throttle through the ghost.
- **Hazard-free: the progress currency is nearly flat between coast and throttle.** Replan 0: cost(final mean) 463.4 vs
  cost(throttle) 463.9 vs cost(brake) 941.0; the CEM is converged (final ≤ population minimum 463.3) but its optimum sits at
  throttle_brake ≈ 0.07, then −0.13, +0.27, +0.49, −0.25, +0.49, +0.48, +0.22 → 18.24 m in 8 steps vs 23.89 m for constant
  throttle (0.76). The encoder does not resolve a 0.3 m position difference 0.9 s ahead; only brake vs non-brake separates.
  Consequences: (i) the hazard-free goal criterion (≥ 0.9 × throttle) fails for reasons of currency resolution, not planning
  budget; (ii) the hazard-free CEM's own chunks jitter by up to ±0.5, so `divergence_replan_idx` at tol 0.1 fires at replan 0
  in levels 0, 1, 3 and at replan 6 in level 2 (`frac_diverged_at_0` = 1.0 for levels 0/1/3) — not a hazard response. Hence
  the post-hoc `first_strong_brake` criterion (executed throttle_brake ≤ −0.5): hazard-free never; level 0 never; level 1 at
  replan **0**; level 2 never; level 3 at replan **1** (`summary.json → temporal_precedence`). Both rules are stored; the raw
  per-replan deviations allow any other threshold.
- Progress-rule C0 gate (for the record only, now superseded): solid in-lane safe-pass 0.0 (band 0.20–0.90), hazard-free goal
  success 0.0 → FAIL; ghost in-lane safe-pass 1.0, H0 1.0, H0′ 0.0 (label) / 0.0, 0.0, 0.0 (label_ref); always-brake:
  failure_progress 1.0 at every level.

Wall-clock (from the records): CEM episode 26.0–27.4 s (8 replans; 269 candidate-unrolls per replan ≈ 3.1 s, i.e. 11.6 ms
per candidate with the GPU shared); benchmark episodes 0.8–1.1 s; scene 151 s with one episode seed; bridge time 22 s of
151 (15 resets, 118 step calls); latents: 3.9 MB per CEM episode (`latents/drive_seed000003__l0__cem__e0.npz`, `z_context`
[8, 256, 1024] fp16 + token masks).

## 5b. Native-label smoke (seed 9, `SSS`, armA_seed0) — partial, from `smoke_native/armA_seed0/driver.log`

The native smoke started 17:44:50 UTC under the full-mode runner and was **stopped by me at 17:55:24 UTC** on the
coordinator's instruction (no hazard level is to be rolled out until the planner configuration is confirmed; the runner
would otherwise have launched stage 1 automatically). Its records are flushed per scene, so `rollouts.jsonl` does not exist
for this run; the numbers below are the driver's per-episode log lines (each line is printed from the finished record):

| level | policy | native outcome | reason | env steps after context | progress (m) | replay max dev | wall (s) |
|---|---|---|---|---|---|---|---|
| hazard-free | always_throttle (reference) | success | – | **128** (`arrive_dest`) | 221.6 | (bridge test: 3.6e-15) | 1.6 |
| hazard-free | always_brake | failure | max_step | 995 | −4.08 (rolls back after stopping) | 7.1e-15 | 10.6 |
| hazard-free | **cem (compact 32/6/3, rolling goal +3)** | **failure** | **max_step** | 995 | **16.58** | 1.8e-14 | 461.2 |
| 0 sidewalk ped | always_throttle | success | – | 128 | 221.6 | 3.6e-15 | 1.6 |
| 0 sidewalk ped | always_brake | failure | max_step | 995 | −4.08 | 7.1e-15 | 9.0 |

(the level-0 CEM episode was running when the job was stopped; no hazard-level CEM result exists under the native label.)
**C0 hazard-free criterion for armA_seed0 under the registered rolling-goal currency: FAIL** (0 successes of 1) — recorded as
is, nothing tuned. The stall is what section 5's flat-currency observation predicts: the CEM's optimum sits near coast, the
sluggish vehicle decelerates to a standstill, and a standing car with a jittering mean around 0 does not get going again.
Replay deviations under the native flags are unchanged (≤ 1.8e-14; all `replay_ok`).

## 6. Pilot (seed-0 pair, discovery scenes) — NOT STARTED (blocked on the planner configuration)

Legacy (progress-rule, withdrawn label) partial outputs: `artifacts/drive_closed_loop/armA_seed0` and `armB_seed0` hold 40
records each (scenes 3 and 4 of the first 12 common discovery seeds; CEM 64/8/4, 8 replans, 2 episode seeds) plus their
latents. The coordinator killed those drivers at 17:48:41 UTC (`LEGACY_PILOT_KILLED` in the log). **Development data only —
never to be fitted or reported as results.**

The native C0 chain (stage 1 seed-0 pair → stage 2 seeds 1–2 → stage 3 v0.9) is implemented in `run_closed_loop_c0.sh`
(full mode) and was stopped before stage 1. It must not be launched until the main session confirms the planner
configuration (amendment to the Label-first principle). Section 6b holds the hazard-free planner diagnostics that inform
that decision.

## 6b. planner_diagnostics (hazard-free only; `artifacts/drive_closed_loop/planner_diagnostics/<model>/diag.jsonl`)

Coordinator instruction (17:55 UTC): with zero hazard-free successes the native label cannot be fitted; diagnose the
planner on HAZARD-FREE scenes only (the analogue of COAST choosing a checkpoint on success rates). No hazard cell was rolled
out. Script `scripts/cgs_pilot/closed_loop_planner_diag.py`, runner `run_planner_diag.sh` (armA_seed0 + armB_seed0 in
parallel, discovery seeds 9, 10, 19 — all `SSS`; sealed seeds refused by the same guard). Native termination as in 2b
(`bridge_provenance.json`: closed_loop true, the four `*_done` flags true, `horizon_guard 1000`). Per-episode wall cap 1200 s
→ `diag_timeout` (never a native outcome). Every number below is read from `planner_diagnostics/<model>/diag.jsonl` with
`planner_diagnostics/_trace.py`.

Exact configs (steer fixed 0, throttle_brake clipped to [−1, 0.5], cost = L2 of the LAST predicted frame to the goal latent):

| name | population / elites / iterations | init std | goal latent | tokens | executed per replan |
|---|---|---|---|---|---|
| `compact_rolling3` | 32 / 6 / 3 (+ brake, throttle, mean reference candidates) | 0.5 | hazard-free always-throttle frame `executed + 3` (registered currency) | hazard ∪ corridor (= corridor, hazard-free) | 1 |
| `compact_dest` | 32 / 6 / 3 | 0.5 | frame at `arrive_dest` of the reference (goal image) | hazard ∪ corridor | 1 |
| `compact_rolling6` / `compact_rolling10` | 32 / 6 / 3 | 0.5 | reference frame `executed + 6` / `+ 10` | hazard ∪ corridor | 1 |
| `released_cem_dest` | 300 / 10 / 15, mean-sample inclusion, momentum 0 (the released `CEMPlanner`, `droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml`) | 0.1 (`var_scale`) | frame at `arrive_dest` (goal image) | ALL 256 tokens (their L2 objective) | 3 (`num_act_stepped`) |

Reference (model-free, always-throttle from the context state), seed 9: `arrive_dest` at env step **128**, progress 221.6 m,
44 goal frames (identical for both models: 11.8 / 11.6 s wall).

**Results so far (seed 9; the runs continue on seeds 10 and 19 and the released config):**

| model | config | native outcome | arrival env step | env steps | progress (m) | replans | wall (s) |
|---|---|---|---|---|---|---|---|
| armA_seed0 | compact_rolling3 | failure / max_step | – | 995 | 11.5 | 332 | 828 |
| armA_seed0 | compact_dest | failure / max_step | – | 995 | 3.5 | 332 | 787 |
| armB_seed0 | compact_rolling3 | failure / max_step | – | 995 | 129.4 (crawls) | 332 | 827 |
| armB_seed0 | compact_dest | **success** | **273** | 271 | 220.2 | 91 | 216 |
| armB_seed0 | compact_rolling6 | failure / max_step | – | 995 | 115.6 | 332 | 788 |

**Diagnostic (1) — why does the registered rolling-3 currency stall? (armA_seed0, seed 9, per-replan trace)**

| replan | goal idx | speed after (m/s) | executed tb | cost brake | cost throttle | cost final mean | encoder-only ‖z(now) − z_goal‖² | ref frames k..k+3 → goal |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 8.39 | +0.09 | 962 | 479 | 477 | 809 | 809, 821, 798, 0 |
| 1 | 4 | 6.17 | −0.23 | 1093 | 639 | 634 | 818 | 853, 976, 774, 0 |
| 2 | 5 | 6.24 | +0.08 | 1024 | 478 | 435 | 1511 | 675, 919, 849, 0 |
| 3 | 6 | 6.61 | +0.40 | 795 | 526 | 468 | 830 | 881, 935, 589, 0 |
| 4 | 7 | 5.36 | −0.13 | 728 | 522 | 471 | 801 | 969, 675, 682, 0 |
| 5 | 8 | 2.34 | −0.40 | 848 | 482 | 432 | 1602 | 697, 779, 628, 0 |
| 6 | 9 | 0.33 | −0.20 | 1006 | 611 | 548 | 1683 | 838, 781, 532, 0 |
| 8 | 11 | 0.08 | −0.58 | 691 | 399 | 330 | 700 | 557, 523, 508, 0 |
| 20 | 23 | 0.29 | +0.31 | 1629 | 1322 | 1319 | 1770 | 1148, 1047, 865, 0 |
| 40 | 43 (dest) | 0.09 | −0.33 | 4901 | 5193 | 4967 | 5629 | 349, 352, 366, 0 |
| 160 | 43 | 0.08 | −0.35 | 4903 | 5205 | 4991 | 5562 | – |
| 320 | 43 | 0.11 | −0.89 | 4906 | 5209 | 4992 | 5781 | – |

Episode stats: brake < throttle in 93 % of replans (dominated by the stalled phase), mean executed tb −0.35, 97 % of replans
at speed < 0.5 m/s. Reading:
- **Brake does not win at the start.** For the first ~6 replans the throttle candidate beats the brake candidate by ≈ 2×
  (479 vs 962) — the predictor does not mispredict throttle, and the goal is not "reachable by standing still": the
  encoder-only distance of the current frame to the goal (809) is *worse* than the predicted throttle future (479).
- **The currency is flat between coast and throttle.** The CEM's final mean has cost 477 ≈ throttle's 479 at tb ≈ +0.09:
  every non-brake candidate is within ≈ 1 % of the optimum, so the mean stays near its initialisation (0) and jitters
  (+0.09, −0.23, +0.08, +0.40, −0.13, −0.40, −0.20). MetaDrive's vehicle decelerates hard under tb ≤ 0 (8.4 → 0.3 m/s in
  1.8 s) and is sluggish from standstill, so the car stops by replan 6–8. The reference profile shows why: the encoder
  distance of the reference frames 1, 2, 3 model steps before the goal to the goal frame is ≈ 800 at every step and only
  drops at the goal frame itself (809, 821, 798 → 0) — 2.5–7.5 m of progress along a featureless straight road is not
  resolved by the frozen DINOv3 latent on corridor tokens; only "brake vs not" is.
- **From standstill armA's landscape tilts to brake.** Once stopped (and once the rolling index reaches the destination
  frame, idx 43), brake beats throttle by ≈ 6 % (4901 vs 5193) at every later replan, and the final mean stays negative → the
  car never restarts (native `max_step`). Under the destination goal from the very first replan (`compact_dest`) armA brakes
  immediately (k0: brake 4898 < throttle 5194, executed −0.19, −0.78, −0.73) and stalls at 3.5 m.
- **armB_seed0 differs.** Same currency, same scene: its rolling-3 run also decays to standstill by replan 7 but restarts
  intermittently (mean speed 1.35 m/s, 129 m in 995 steps — a crawl, still `max_step`). With the destination goal its final
  mean beats both reference candidates (k0: brake 5003, throttle 5146, final 4995 at tb +0.25; k7: 4940 / 5143 / 4897 at
  +0.41): moderate throttle is the minimum of *its* landscape, it drives at 8–13 m/s and reaches `arrive_dest` at env step
  273 (always-throttle: 128). So the sign of a 3–6 % cost difference between brake and throttle toward a far goal is
  model-specific; the currency has no robust gradient toward a distant goal on a straight road, and the hazard-free
  "success" of armB under `compact_dest` rests on that tilt, not on resolved progress.
- Receding goals further ahead (`compact_rolling6`, armB): same stall pattern (`max_step`, 115.6 m); rolling-10 and the
  released 300 × 15 CEM with the destination image are still running / pending (see section 9).

**Diagnostic (1b) — per-replan costs of the brake vs throttle candidates and the chosen chunk (first 20 replans; coordinator request).**

**armA_seed0 · `compact_dest`** — failure / max_step, arrival None, progress 3.5 m, route_completion 0.053. Columns: cost = ‖P(z, a)[last] − z_goal‖² over the token set (= the distance of the predicted latent to the goal under that candidate); chosen = the executed CEM mean.

| replan | goal idx | speed after (m/s) | chosen chunk tb (3 steps) | cost brake | cost throttle | (thr − brake)/brake | cost chosen | chosen − min(brake, thr) |
|---|---|---|---|---|---|---|---|---|
| 0 | 43 | 6.42 | [-0.19, 0.43, -1.0] | 4898 | 5194 | +6.0 % | 4967 | +68 |
| 1 | 43 | 2.07 | [-0.78, 0.03, -0.65] | 5318 | 5589 | +5.1 % | 5368 | +50 |
| 2 | 43 | 0.10 | [-0.73, -0.26, -0.63] | 5847 | 6092 | +4.2 % | 5878 | +31 |
| 3 | 43 | 0.02 | [-0.01, 0.1, -0.78] | 5242 | 5545 | +5.8 % | 5328 | +86 |
| 4 | 43 | 0.13 | [-0.31, 0.3, -0.97] | 4824 | 5131 | +6.4 % | 4901 | +77 |
| 5 | 43 | 0.02 | [-0.08, 0.33, -0.99] | 4900 | 5199 | +6.1 % | 4980 | +80 |
| 6 | 43 | 0.13 | [-0.56, 0.31, -0.99] | 4900 | 5193 | +6.0 % | 4978 | +79 |
| 7 | 43 | 0.24 | [-0.65, 0.39, -0.74] | 4899 | 5194 | +6.0 % | 5004 | +105 |
| 8 | 43 | 0.16 | [0.08, 0.3, -1.0] | 4897 | 5195 | +6.1 % | 4976 | +79 |
| 9 | 43 | 0.08 | [-0.48, 0.39, -0.99] | 4897 | 5196 | +6.1 % | 4974 | +77 |
| 10 | 43 | 0.18 | [-0.49, 0.08, -1.0] | 4898 | 5198 | +6.1 % | 4975 | +77 |
| 11 | 43 | 0.10 | [-0.54, 0.42, -1.0] | 4900 | 5194 | +6.0 % | 4975 | +75 |
| 12 | 43 | 0.19 | [-0.42, 0.05, -1.0] | 4897 | 5192 | +6.0 % | 4972 | +75 |
| 13 | 43 | 0.11 | [-0.75, 0.13, -0.87] | 4897 | 5192 | +6.0 % | 5003 | +106 |
| 14 | 43 | 0.24 | [-0.92, 0.4, -0.69] | 4895 | 5196 | +6.1 % | 4996 | +101 |
| 15 | 43 | 0.09 | [-0.49, 0.32, -1.0] | 4891 | 5183 | +6.0 % | 4967 | +76 |
| 16 | 43 | 0.06 | [-0.2, 0.46, -1.0] | 4893 | 5191 | +6.1 % | 4972 | +79 |
| 17 | 43 | 0.13 | [-0.32, 0.26, -0.67] | 4893 | 5190 | +6.1 % | 5015 | +122 |
| 18 | 43 | 0.20 | [-0.48, 0.1, -0.9] | 4896 | 5195 | +6.1 % | 4995 | +99 |
| 19 | 43 | 0.08 | [0.3, 0.17, -1.0] | 4897 | 5190 | +6.0 % | 4973 | +75 |

Whole episode (332 replans): (thr − brake)/brake mean +6.12 % (min +4.20, max +6.65); throttle worse than brake in 100 % of replans; chosen chunk below both reference candidates in 0 % of replans.

**armA_seed0 · `compact_rolling3`** — failure / max_step, arrival None, progress 11.5 m, route_completion 0.087. Columns: cost = ‖P(z, a)[last] − z_goal‖² over the token set (= the distance of the predicted latent to the goal under that candidate); chosen = the executed CEM mean.

| replan | goal idx | speed after (m/s) | chosen chunk tb (3 steps) | cost brake | cost throttle | (thr − brake)/brake | cost chosen | chosen − min(brake, thr) |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 8.39 | [0.09, 0.44, 0.5] | 962 | 479 | -50.2 % | 477 | -2 |
| 1 | 4 | 6.17 | [-0.23, 0.49, 0.49] | 1093 | 639 | -41.5 % | 634 | -5 |
| 2 | 5 | 6.24 | [0.08, -0.54, 0.14] | 1024 | 478 | -53.3 % | 435 | -43 |
| 3 | 6 | 6.61 | [0.4, -0.51, 0.42] | 795 | 526 | -33.8 % | 468 | -59 |
| 4 | 7 | 5.36 | [-0.13, -0.91, -0.22] | 728 | 522 | -28.2 % | 471 | -52 |
| 5 | 8 | 2.34 | [-0.4, -0.98, -0.1] | 848 | 482 | -43.2 % | 432 | -50 |
| 6 | 9 | 0.33 | [-0.2, -0.96, -0.2] | 1006 | 611 | -39.3 % | 548 | -63 |
| 7 | 10 | 0.55 | [0.24, 0.42, 0.37] | 882 | 552 | -37.4 % | 551 | -1 |
| 8 | 11 | 0.08 | [-0.58, -0.93, -0.03] | 691 | 399 | -42.3 % | 330 | -69 |
| 9 | 12 | 0.21 | [0.14, -0.04, 0.49] | 1004 | 472 | -53.0 % | 502 | +30 |
| 10 | 13 | 0.01 | [-0.02, -0.71, 0.17] | 928 | 542 | -41.6 % | 540 | -2 |
| 11 | 14 | 0.01 | [-0.04, -0.01, 0.49] | 924 | 600 | -35.1 % | 582 | -18 |
| 12 | 15 | 0.24 | [-0.79, -0.42, 0.25] | 797 | 573 | -28.2 % | 542 | -31 |
| 13 | 16 | 0.11 | [-0.85, 0.22, 0.47] | 819 | 457 | -44.2 % | 416 | -41 |
| 14 | 17 | 0.50 | [0.43, -0.49, 0.16] | 875 | 558 | -36.3 % | 547 | -11 |
| 15 | 18 | 0.06 | [-0.43, -0.94, 0.11] | 910 | 534 | -41.3 % | 481 | -52 |
| 16 | 19 | 0.16 | [0.1, 0.42, 0.49] | 1326 | 803 | -39.4 % | 803 | +0 |
| 17 | 20 | 0.25 | [-0.65, -0.87, 0.08] | 1210 | 885 | -26.9 % | 828 | -57 |
| 18 | 21 | 0.10 | [-0.32, -0.92, 0.16] | 1532 | 1106 | -27.8 % | 1098 | -8 |
| 19 | 22 | 0.01 | [-0.03, -0.99, 0.31] | 1233 | 972 | -21.2 % | 961 | -11 |

Whole episode (332 replans): (thr − brake)/brake mean +3.24 % (min -53.31, max +6.62); throttle worse than brake in 93 % of replans; chosen chunk below both reference candidates in 6 % of replans.

**armB_seed0 · `compact_dest`** — success, arrival 273, progress 220.2 m, route_completion 0.979. Columns: cost = ‖P(z, a)[last] − z_goal‖² over the token set (= the distance of the predicted latent to the goal under that candidate); chosen = the executed CEM mean.

| replan | goal idx | speed after (m/s) | chosen chunk tb (3 steps) | cost brake | cost throttle | (thr − brake)/brake | cost chosen | chosen − min(brake, thr) |
|---|---|---|---|---|---|---|---|---|
| 0 | 43 | 8.54 | [0.25, 0.22, -0.81] | 5003 | 5146 | +2.9 % | 4995 | -8 |
| 1 | 43 | 8.89 | [0.37, 0.05, -0.99] | 4930 | 5105 | +3.6 % | 4874 | -55 |
| 2 | 43 | 9.09 | [0.22, 0.17, -0.99] | 4919 | 5111 | +3.9 % | 4881 | -38 |
| 3 | 43 | 9.22 | [0.14, 0.18, -0.89] | 5007 | 5178 | +3.4 % | 4979 | -29 |
| 4 | 43 | 9.50 | [0.31, 0.13, -0.95] | 5012 | 5181 | +3.4 % | 4969 | -42 |
| 5 | 43 | 9.62 | [0.13, 0.05, -0.77] | 4930 | 5112 | +3.7 % | 4929 | -1 |
| 6 | 43 | 9.91 | [0.31, 0.02, -0.89] | 5007 | 5167 | +3.2 % | 4997 | -10 |
| 7 | 43 | 10.29 | [0.41, 0.21, -0.96] | 4940 | 5143 | +4.1 % | 4897 | -43 |
| 8 | 43 | 10.54 | [0.27, 0.11, -0.96] | 4927 | 5130 | +4.1 % | 4898 | -29 |
| 9 | 43 | 10.57 | [0.03, 0.14, -0.95] | 4926 | 5109 | +3.7 % | 4920 | -6 |
| 10 | 43 | 10.89 | [0.35, 0.11, -0.99] | 5000 | 5163 | +3.3 % | 4987 | -12 |
| 11 | 43 | 10.92 | [0.03, -0.08, -0.96] | 4928 | 5086 | +3.2 % | 4921 | -7 |
| 12 | 43 | 11.24 | [0.35, -0.11, -0.9] | 5012 | 5199 | +3.7 % | 4989 | -24 |
| 13 | 43 | 11.50 | [0.28, -0.04, -0.85] | 4922 | 5117 | +4.0 % | 4914 | -9 |
| 14 | 43 | 11.83 | [0.36, 0.21, -0.99] | 4932 | 5105 | +3.5 % | 4890 | -43 |
| 15 | 43 | 11.11 | [-0.07, 0.11, -0.94] | 4926 | 5111 | +3.8 % | 4922 | -3 |
| 16 | 43 | 11.53 | [0.45, 0.22, -0.95] | 5265 | 5466 | +3.8 % | 5235 | -30 |
| 17 | 43 | 11.93 | [0.43, 0.22, -0.88] | 4852 | 5005 | +3.2 % | 4823 | -29 |
| 18 | 43 | 12.13 | [0.21, 0.02, -0.93] | 4928 | 5108 | +3.7 % | 4908 | -21 |
| 19 | 43 | 12.45 | [0.35, 0.17, -0.92] | 5011 | 5158 | +2.9 % | 4995 | -16 |

Whole episode (91 replans): (thr − brake)/brake mean +3.46 % (min +2.44, max +4.39); throttle worse than brake in 100 % of replans; chosen chunk below both reference candidates in 97 % of replans.

**Statement.** At destination range the brake and throttle costs are *not* near-equal noise around zero: throttle is worse
than brake by a stable, model-specific margin at every replan — armA_seed0 **+6.1 % (4.2 … 6.7 %) in 100 % of 332 replans**,
including from standstill where the true 0.9 s futures under brake and throttle differ by < 2 m; armB_seed0 **+3.5 %
(2.4 … 4.4 %) in 100 % of 91 replans**. A margin that stable and sign-consistent, while the level of the cost (≈ 4 900 –
5 200) is ≈ 10× the 3-step-goal cost (≈ 480), is an **action-dependent predictor bias**: both predictors move the last-frame
latent *away* from the destination frame under throttle relative to brake. What separates the two models is not the sign
but the shape of the landscape in between: armA's minimum is at the brake candidate itself (the CEM's chosen chunk is
+31 … +122 above brake at every replan, 0 % below both references) → immediate braking (executed −0.19, −0.78, −0.73) and a
3.5 m stall; armB's landscape has an interior minimum — chosen chunks of the form [+0.25 … +0.45, ≈ 0, ≈ −0.9] (moderate
throttle now, brake at the last imagined step) sit 8 … 55 below the brake candidate at 97 % of replans → it drives at
8 – 13 m/s and arrives at env step 273. So armB's hazard-free success under the destination goal is produced by a
within-horizon artefact (a braked final frame looks more like the goal latent), not by resolved progress. By contrast at
3-step range (`compact_rolling3`, armA) the goal is informative between brake and non-brake (throttle beats brake by 21 …
53 % for the first 20 replans) but flat between coast and throttle (chosen − throttle within a few units), which is the
stall mechanism described under Diagnostic (1). Reading in one line: *far goal → small but systematic predictor bias
toward a braked final state (model-specific magnitude); near goal → brake vs non-brake resolved, progress rate not.*

**Released jepa-wms planner configuration (exact, from
`vendor/jepa-wms/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml`):**
`planner_name: cem`, `iterations: 15`, `num_samples: 300`, `num_elites: 10`, `horizon: 3`, `var_scale: 0.1`,
`momentum_mean: 0`, `momentum_std: 0`, `num_act_stepped: 3`, `max_norms [0.1, 0.75]` on the 7-D DROID action,
`planning_objective: L2, sum_all_diffs: false, alpha: 0` (distance of the LAST predicted frame's full latent to the goal
latent). **Goal image: NOT the destination.** `task_specification.goal_source: dset`, `goal_H: 3`, `max_episode_steps: 100`,
`succ_def: simu`: the goal is the dataset frame `goal_H = 3` actions (× frameskip 1) after the episode's start frame
(`plan_evaluator.py` lines 276/334 slice the expert actions `[offset : offset + frameskip·goal_H]`; `gc_agent.py` line 38–40),
i.e. a short-horizon sub-goal image, and success is the simulator's own predicate for that sub-task. The
`released_cem_dest` diagnostic therefore reproduces their optimizer (population 300 / elites 10 / iterations 15 / horizon 3
/ init std 0.1 / mean-sample inclusion / execute 3 / full-latent L2) but, as instructed, with the destination frame as the
goal — a goal-range their evals never use. Their sub-goal usage corresponds to our rolling-3 currency at the first replan.


### planner_diagnostics/range_k (coordinator request 18:35 UTC; hazard-free only; `run_planner_diag_rangek.sh`)

Fixed goal image = the reference (always-throttle) rollout's frame k model steps ahead of the context frame, k ∈ {6, 10, 20,
40} (≈ 17 / 32 / 77 / 204 m ahead on these scenes; k = 40 is 1–7 frames before `arrive_dest`). Compact CEM 32/6/3, execute 1;
`released_k10` = the released optimizer budget (300/10/15, init std 0.1, mean-sample inclusion, execute 3, full-latent L2) at
k = 10. The episode ends descriptively when the ego's lane progress reaches the reference's progress at step k
(`descriptive_pass`; `target` column) or at the env's own done (`max_step` guard), or at the 1200 s wall cap (`diag_timeout`).
"last-step brake while moving" = chosen chunk ends with tb ≤ −0.5 while the ego was moving (> 0.5 m/s) before the replan
(`n/r` = flag not recorded for the pre-flag runs). Seeds 9, 10, 19 (`SSS`). Every row is read from
`planner_diagnostics/<model>/diag.jsonl`.

| model | config | seed | native outcome | pass of goal position (target m) | pass / arrival env step (episode env steps) | progress (m) | replans | wall (s) | (thr − brake)/brake mean [first 5 replans] | last-step brake while moving | executed tb mean, frac < 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| armA_seed0 | reference (always-throttle) | 9 | success | – | arrive 128 | 221.6 | – | 12 | – | – | – |
| armA_seed0 | reference (always-throttle) | 10 | success | – | arrive 122 | 208.1 | – | 2 | – | – | – |
| armA_seed0 | reference (always-throttle) | 19 | success | – | arrive 138 | 243.1 | – | 2 | – | – | – |
| armA_seed0 | fixed_k6 | 9 | descriptive_pass | PASS (17.4) | 21 (21) | 19.8 | 7 | 11 | -0.375 [-0.38, -0.39, -0.38, -0.36, -0.37] | 0/7 | +0.38, 0.00 |
| armA_seed0 | fixed_k6 | 10 | descriptive_pass | PASS (17.4) | 48 (48) | 18.0 | 16 | 34 | -0.358 [-0.36, -0.52, -0.3, -0.5, -0.31] | 0/16 | +0.24, 0.19 |
| armA_seed0 | fixed_k6 | 19 | descriptive_pass | PASS (17.3) | 21 (21) | 20.0 | 7 | 14 | -0.376 [-0.38, -0.38, -0.39, -0.36, -0.36] | 0/7 | +0.42, 0.00 |
| armA_seed0 | fixed_k10 | 9 | descriptive_pass | PASS (31.7) | 66 (66) | 32.7 | 22 | 52 | -0.333 [-0.31, -0.3, -0.41, -0.32, -0.31] | 0/22 | +0.07, 0.36 |
| armA_seed0 | fixed_k10 | 10 | descriptive_pass | PASS (31.7) | 282 (282) | 31.9 | 94 | 217 | -0.313 [-0.29, -0.29, -0.41, -0.38, -0.31] | 0/94 | +0.09, 0.33 |
| armA_seed0 | fixed_k10 | 19 | failure / max_step | no (31.6) | – (995) | 18.7 | 332 | 696 | -0.342 [-0.32, -0.33, -0.38, -0.36, -0.34] | 0/332 | -0.09, 0.91 |
| armA_seed0 | fixed_k20 | 9 | failure / max_step | no (77.4) | – (995) | -1.6 | 332 | 792 | -0.245 [-0.22, -0.34, -0.33, -0.25, -0.24] | 0/332 | -0.59, 0.98 |
| armA_seed0 | fixed_k20 | 10 | failure / max_step | no (77.3) | – (995) | 0.7 | 332 | 785 | -0.252 [-0.21, -0.32, -0.19, -0.35, -0.27] | 0/332 | -0.56, 0.91 |
| armA_seed0 | fixed_k20 | 19 | failure / max_step | no (77.0) | – (995) | 3.0 | 332 | 587 | -0.290 [-0.25, -0.37, -0.31, -0.38, -0.3] | 0/332 | -0.46, 0.87 |
| armA_seed0 | fixed_k40 | 9 | failure / max_step | no (203.8) | – (995) | 4.2 | 332 | 618 | +0.060 [0.06, 0.04, 0.04, 0.06, 0.06] | 9/332 | -0.31, 0.78 |
| armA_seed0 | fixed_k40 | 10 | failure / max_step | no (203.6) | – (995) | 2.7 | 332 | 606 | +0.061 [0.06, 0.03, 0.05, 0.06, 0.07] | 2/332 | -0.30, 0.81 |
| armA_seed0 | fixed_k40 | 19 | failure / max_step | no (203.0) | – (995) | 4.8 | 332 | 755 | +0.060 [0.06, 0.05, 0.04, 0.07, 0.06] | 9/332 | -0.31, 0.78 |
| armA_seed0 | released_k10 | 9 | diag_timeout | no (31.7) | – (153) | 15.1 | 17 | 1246 | -0.597 [-0.58, -0.59, -0.59, -0.6, -0.6] | 0/17 | +0.12, 0.06 |
| armA_seed0 | released_k10 | 10 | diag_timeout | no (31.7) | – (153) | 22.2 | 17 | 1214 | -0.605 [-0.58, -0.6, -0.6, -0.6, -0.6] | 0/17 | +0.30, 0.00 |
| armA_seed0 | released_k10 | 19 | diag_timeout | no (31.6) | – (126) | 21.9 | 14 | 1268 | -0.612 [-0.59, -0.6, -0.6, -0.61, -0.61] | 0/14 | +0.31, 0.00 |
| armA_seed0 | compact_rolling3 | 9 | failure / max_step | – | – (995) | 11.5 | 332 | 828 | +0.032 | n/r | -0.35, 0.82 |
| armA_seed0 | compact_dest | 9 | failure / max_step | – | – (995) | 3.5 | 332 | 787 | +0.061 | n/r | -0.33, 0.76 |
| armB_seed0 | reference (always-throttle) | 9 | success | – | arrive 128 | 221.6 | – | 12 | – | – | – |
| armB_seed0 | reference (always-throttle) | 10 | success | – | arrive 122 | 208.1 | – | 2 | – | – | – |
| armB_seed0 | reference (always-throttle) | 19 | success | – | arrive 138 | 243.1 | – | 2 | – | – | – |
| armB_seed0 | fixed_k6 | 9 | failure / max_step | no (17.4) | – (995) | 7.0 | 332 | 618 | -0.430 [-0.44, -0.48, -0.38, -0.48, -0.38] | 0/332 | -0.20, 0.92 |
| armB_seed0 | fixed_k6 | 10 | failure / max_step | no (17.4) | – (995) | 7.2 | 332 | 593 | -0.416 [-0.42, -0.47, -0.36, -0.48, -0.41] | 0/332 | -0.19, 0.86 |
| armB_seed0 | fixed_k6 | 19 | failure / max_step | no (17.3) | – (995) | 15.7 | 332 | 866 | -0.424 [-0.43, -0.48, -0.39, -0.5, -0.39] | 0/332 | -0.25, 0.95 |
| armB_seed0 | fixed_k10 | 9 | failure / max_step | no (31.7) | – (995) | 7.6 | 332 | 800 | -0.388 [-0.38, -0.38, -0.42, -0.4, -0.46] | 0/332 | -0.26, 0.80 |
| armB_seed0 | fixed_k10 | 10 | failure / max_step | no (31.7) | – (995) | 4.4 | 332 | 745 | -0.372 [-0.37, -0.38, -0.42, -0.42, -0.39] | 0/332 | -0.37, 0.92 |
| armB_seed0 | fixed_k10 | 19 | failure / max_step | no (31.6) | – (995) | 3.7 | 332 | 438 | -0.388 [-0.38, -0.41, -0.39, -0.43, -0.36] | 0/332 | -0.27, 0.80 |
| armB_seed0 | fixed_k20 | 9 | failure / max_step | no (77.4) | – (995) | -2.7 | 332 | 769 | -0.275 [-0.29, -0.35, -0.29, -0.19, -0.24] | 0/332 | -0.72, 1.00 |
| armB_seed0 | fixed_k20 | 10 | failure / max_step | no (77.3) | – (995) | -1.8 | 332 | 601 | -0.256 [-0.28, -0.31, -0.27, -0.25, -0.21] | 0/332 | -0.69, 0.99 |
| armB_seed0 | fixed_k20 | 19 | failure / max_step | no (77.0) | – (995) | 0.9 | 332 | 337 | -0.310 [-0.31, -0.3, -0.32, -0.32, -0.33] | 0/332 | -0.65, 1.00 |
| armB_seed0 | fixed_k40 | 9 | descriptive_pass | PASS (203.8) | 207 (207) | 205.9 | 69 | 133 | +0.036 [0.03, 0.04, 0.04, 0.04, 0.04] | 69/69 | +0.22, 0.14 |
| armB_seed0 | fixed_k40 | 10 | failure / max_step | no (203.6) | – (995) | 125.9 | 332 | 707 | +0.034 [0.03, 0.04, 0.04, 0.04, 0.04] | 49/332 | -0.09, 0.61 |
| armB_seed0 | fixed_k40 | 19 | descriptive_pass | PASS (203.0) | 489 (489) | 203.7 | 163 | 389 | +0.035 [0.03, 0.04, 0.04, 0.04, 0.03] | 149/163 | +0.17, 0.18 |
| armB_seed0 | released_k10 | 9 | diag_timeout | no (31.7) | – (153) | 8.6 | 17 | 1238 | -0.582 [-0.57, -0.59, -0.58, -0.58, -0.59] | 0/17 | -0.20, 0.82 |
| armB_seed0 | released_k10 | 10 | diag_timeout | no (31.7) | – (144) | 17.6 | 16 | 1231 | -0.587 [-0.57, -0.59, -0.59, -0.59, -0.6] | 0/16 | -0.24, 0.75 |
| armB_seed0 | compact_rolling3 | 9 | failure / max_step | – | – (995) | 129.4 | 332 | 827 | +0.004 | n/r | +0.02, 0.42 |
| armB_seed0 | compact_rolling6 | 9 | failure / max_step | – | – (995) | 115.6 | 332 | 788 | +0.005 | n/r | -0.09, 0.59 |
| armB_seed0 | compact_dest | 9 | success | – | 273 (271) | 220.2 | 91 | 216 | +0.035 | n/r | +0.17, 0.16 |

Where in the chosen 3-step chunk the brake sits (replans while moving only):

| model | config | replans while moving (v > 0.5 m/s) | chosen chunk has tb ≤ −0.5 at any step | first step | middle step | last step |
|---|---|---|---|---|---|---|
| armA_seed0 | fixed_k6 | 30 | 0.07 | 0.07 | 0.00 | 0.00 |
| armA_seed0 | fixed_k10 | 87 | 0.00 | 0.00 | 0.00 | 0.00 |
| armA_seed0 | fixed_k20 | 27 | 1.00 | 0.70 | 0.93 | 0.00 |
| armA_seed0 | fixed_k40 | 21 | 1.00 | 0.38 | 0.00 | 0.95 |
| armA_seed0 | released_k10 | 18 | 0.00 | 0.00 | 0.00 | 0.00 |
| armB_seed0 | fixed_k6 | 34 | 0.21 | 0.12 | 0.09 | 0.00 |
| armB_seed0 | fixed_k10 | 13 | 0.92 | 0.38 | 0.85 | 0.00 |
| armB_seed0 | fixed_k20 | 8 | 1.00 | 0.62 | 1.00 | 0.00 |
| armB_seed0 | fixed_k40 | 267 | 1.00 | 0.01 | 0.02 | 1.00 |
| armB_seed0 | released_k10 | 10 | 0.70 | 0.10 | 0.70 | 0.00 |

**One-line reading per k** (`descriptive_pass` counts out of 3 scenes):
- **k = 6 (≈ 17 m):** armA **3/3** passes, fast (7–16 replans, 21–48 env steps; the reference needs 18), no brake in the chosen
  chunks (0–7 %), throttle 36–38 % cheaper than brake; armB **0/3** — throttle is 42 % cheaper than brake at every replan, yet
  its executed first step is mildly negative at 86–95 % of replans (mean −0.2): armB's landscape has an interior optimum
  with a mild brake now and it decelerates to a standstill at 7–16 m.
- **k = 10 (≈ 32 m):** armA **2/3** (66 and 282 env steps vs the reference's 30 — it crawls at 3–6 m/s; seed 19 stalls at
  18.7 m), no brakes while moving; armB **0/3**: brake in the *middle* imagined step at 85 % of moving replans ("coast,
  brake, coast"), executed mean −0.3 → stall at 4–8 m. Released budget at k = 10: **diag_timeout** for both (14–17 replans in
  1200 s = 73 s per replan for 4 500 candidate-unrolls on the shared GPU), descriptive progress 15–22 m (armA) / 9–18 m
  (armB) after 42–51 model steps with throttle 58–61 % cheaper than brake — the released budget is not usable at this cost
  and, within its 17 replans, did not change the picture (armA moving, armB braking mid-chunk).
- **k = 20 (≈ 77 m):** **0/3 for both.** Throttle still 25–31 % cheaper than the brake candidate, but the chosen chunks brake
  in the first/middle step at 70–100 % of moving replans and both models stop within 1–3 replans (executed mean −0.5 …
  −0.7, ≥ 87 % negative).
- **k = 40 (≈ 204 m, the destination view):** margins flip to brake-cheaper (+6 % armA, +3.5 % armB) — the far-goal predictor
  bias of Diagnostic 1b. armA **0/3** (stops within 3 m; brake at the last step in 95 % of its few moving replans, first step
  38 %); armB **2/3** passes (207 and 489 env steps vs the reference's ≈ 130; seed 10 stalls at 126 m) with the **last-step
  brake artefact at 100 % of moving replans** (first/middle steps essentially never): it drives exactly because "throttle
  now, brake at the last imagined step" is the interior minimum of its far-goal landscape.

**Honest statement.** No single hazard-free configuration drives both seed-0 models: armA drives only toward *near* goals
(k ≤ 10; 5/6 passes) and stops for k ≥ 20; armB stops for every near/mid goal (0/9 at k ≤ 20) and "drives" only toward the
far goal (k = 40: 2/3; `compact_dest`: 1/1) through the last-step-brake artefact. The margin sign is not the decider
(throttle beats brake by 25–43 % in every k ≤ 20 run of both models, and armB still stalls); the position of the brake inside
the chosen chunk is — mid-range goals put it in the first/middle step (stop), far goals put it in the last step (drive with
the artefact), near goals remove it for armA but leave armB a mildly negative first step. Receding-goal variants
(`compact_rolling6/10`) and `released_cem_dest` for the remaining seeds are still queued (pass 3); the checkpoint-choice
remedy (armB_seed0 `jepa-best.pth.tar`, `fixed_k10` + `compact_rolling10`, same 3 scenes) is running
(`planner_diagnostics/armB_seed0_jepabest/`).

**Receding-goal ranges (pass 3, seed 9 so far; `compact_rolling{3,6,10}`, goal = reference frame `executed + k`):**
armA_seed0: k = 3 → 11.5 m, k = 6 → 17.9 m, k = 10 → 11.3 m — all `max_step` (stall within the first ~10 replans);
armB_seed0: k = 3 → 129.4 m, k = 6 → 115.6 m, k = 10 → 124.3 m — all `max_step` (a crawl at ≈ 1.3 m/s mean, 332 replans).
**No receding-goal range drives either seed-0 model to `arrive_dest`;** a receding goal only converts armB's stall into a
crawl. Seeds 10 and 19 and `released_cem_dest` are still running (see section 9); the statement will be updated if they
differ.

**Checkpoint-choice remedy (coordinator task 3): void for the current runs.** `artifacts/drive_models/armB_seed0/jepa-best.pth.tar`
and `jepa-latest.pth.tar` are the same weights: both `epoch 30`, all 92 predictor tensors `torch.equal`, identical
`*.meta.json` (`best.epoch 30`, `val_tf_l2 0.003074`, `val_unroll_l2 0.003468`) — the trainer's "best" was its final epoch,
and no per-epoch checkpoints were kept (`run_drive_arms.sh`: final-checkpoint-only rule). The diagnostic run on
`jepa-best` (`planner_diagnostics/armB_seed0_jepabest/`, `fixed_k10` seed 9: `max_step`, 7.5 m, margin −0.388 — identical
to `jepa-latest`'s 7.6 m / −0.388 up to the bridge's 1e-13 physics jitter) confirms it and was stopped by me at that point.
An earlier-epoch checkpoint (COAST App. A.8's "early checkpoint with 20–60 % success") would require re-training with
per-epoch saves on box 1's chain, which I did not touch.

**Implication for the amendment decision (no action taken):** the C0 hazard-free FAIL is a property of the registered
currency (encoder resolution of longitudinal progress) rather than of the CEM budget; none of the hazard-free-only variants
tried so far gives a model-independent success (armB `compact_dest` succeeds, armA does not). A currency that resolves
progress (e.g. a goal defined on quantities the latent does resolve, or a state-aware progress term) would be a change to
the registered currency and therefore needs the amendment the coordinator described; the registered checkpoint remedies
(earlier epoch, scale, more clips) address the planner's tilt at standstill but not the encoder's flatness.

## 6c. Mechanism statement (planner diagnostics, hazard-free only)

1. The registered progress currency (L2 of the last predicted DINOv3-token latent to a hazard-free throttle frame on
   corridor tokens) resolves **brake vs non-brake** at short range (throttle 2× cheaper than brake 3 steps ahead; 25–43 %
   cheaper at 6–20 steps) but **not the rate of progress**: coast and throttle differ by ≤ 1 % at 3 steps, and the encoder
   distance of the reference's own frames to a goal frame is ≈ constant until the goal frame itself. A CEM initialised at
   0 therefore executes ≈ 0 ± 0.4 and MetaDrive's sluggish, drag-heavy vehicle decelerates to a standstill within 2–8 replans.
2. At far range (≥ 20 steps / ≥ 77 m, incl. the destination view) the sign flips to a small but perfectly stable
   **predictor bias toward a braked final frame** (throttle 6 % / 3.5 % costlier than brake for armA / armB at 100 % of
   replans). Where the brake sits inside the chosen 3-step chunk decides the behaviour: mid-range goals put it in the
   first/middle step (both models stop within 1–3 replans); the far goal puts it in the *last* imagined step (armB drives
   at 8–13 m/s with the artefact at 100 % of moving replans and arrives; armA's minimum is the brake candidate itself and
   it stops within 3 m).
3. The two seed-0 models are qualitatively different planners under the same currency: armA drives only toward near goals
   (k ≤ 10: 5/6 descriptive passes) and never toward far ones; armB stops for every near/mid goal (0/9 at k ≤ 20), crawls
   under receding goals (occasionally completing the native route when the route is short enough, e.g. rolling-6 seed 19
   at env step 525) and drives only via the far-goal artefact (k = 40: 2/3; destination: 1/1).
4. Neither the CEM budget (compact vs released 300 × 15; the latter costs 73 s per replan on the shared GPU and timed out
   without changing the picture) nor the checkpoint (jepa-best ≡ jepa-latest) is the lever; the encoder/currency
   resolution and the model-specific tilt of the cost landscape are.

## 7. C0 closed-loop gate

Frozen thresholds (design doc, claim ladder C0): unsteered safe-pass rate on the solid in-lane identity in [0.20, 0.90]
**and** hazard-free goal success ≥ 0.50. Ghost in-lane, H0 (sidewalk) and H0′ (mirror) rows reported alongside, with the
always-brake benchmark. Verdicts are in `summary.json → c0_closed_loop_gate` per model.

**Recommended C0 verdict (native label, Label-first principle amendment 1; hazard cells never rolled out):**
- **armA_seed0: C0 FAIL** — hazard-free success 0 under every configuration tried (registered rolling-3 currency: native
  smoke `max_step` 995 / 16.6 m, diagnostics 11.5 m; rolling-6/10, destination goal, fixed k ≥ 20, released optimizer: 0
  arrivals); only fixed near goals (k ≤ 10) produce descriptive passes, and those are not the native route.
- **armB_seed0: C0 FAIL** — hazard-free success 0 under the registered currency (crawl to `max_step` at 115–129 m on seeds
  9/10); arm B occasionally completes the native route under receding goals (rolling-6 seed 19 at env step 525) and under
  far-goal configurations (`compact_dest` seed 9 at 273; fixed k = 40 on 2/3 scenes) via the last-step-brake artefact —
  not a usable planner configuration and outside the registered currency.
- Solid-in-lane success/failure mix (the other half of C0) was **not measured**: no hazard level was rolled out, per the
  coordinator's instruction, because with zero hazard-free successes the label cannot be fitted (COAST App. A.8 eligibility).
- Consequently the driving cell is **deferred as a trained-substrate panel** (user decision 23:40 UTC, Label-first principle
  amendment 3); the primary panel moves to released JEPA-WM / DINO-WM checkpoints with their official evaluators (owned by
  another agent).

## 7b. What would be needed to make this substrate driveable — DEFERRED (not run, not planned here)

- **Per-epoch checkpoints** of the driving predictors (the trainer kept only the final epoch, `jepa-best ≡ jepa-latest`), so
  the COAST-style checkpoint choice (App. A.8: an early checkpoint with 20–60 % success) becomes possible; requires a
  re-train on box 1's chain with per-epoch saves.
- **More / richer training clips** with speed variation and non-zero steering (the current clips use steer = 0 exactly and
  three throttle values), so the predictor's action conditioning resolves progress rate and the CEM's action space can
  include steering (needed for any "safe pass" around a lane-centred body).
- **Visually structured roads** (lane markings/objects with parallax, textured roadside, curvature the model can see) so the
  frozen DINOv3 latent on corridor tokens changes measurably with 2–8 m of travel; on the featureless straight `SSS`
  road the encoder distance to a goal frame is flat until the goal frame itself.
- **A shorter native route or a native intermediate success predicate**: `arrive_dest` is 200–250 m away (128–144 env
  steps at full throttle) while the latent resolves progress only over ≈ 15–30 m; a route whose destination lies within
  the resolved range (k ≤ 10 gave armA 5/6 descriptive passes) would give the native label a chance, but MetaDrive's
  `_is_arrive_destination` is tied to the map's final lane, so this means a different map (e.g. a single short block).
- A currency that includes what the latent does resolve (speed / ego-state channels, or an action-conditioned progress
  term) — a change to the registered currency, hence a preregistered amendment, not a tuning step.

## 8. How to poll / resume

- Runner log: `ssh -p 45460 root@70.27.250.55 'tail -20 /root/cgs-pilot/logs/drive_closed_loop.log'`; markers
  `CLOSED_LOOP_SMOKE_DONE`, `CLOSED_LOOP_PILOT12_DONE`, `CLOSED_LOOP_PILOT_DONE` (or `CLOSED_LOOP_SMOKE_FAIL`).
- Per-model progress: `artifacts/drive_closed_loop/arm{A,B}_seed0/driver.log` (one line per episode:
  `seed L level P policy epE: contact= progress= label= replay_ok= dev= wall=`), `bridge.log` (MetaDrive side).
- Outputs: `artifacts/drive_closed_loop/<model>/{rollouts.jsonl, summary.json, run_config.json, bridge_provenance.json, latents/}`.
- Re-summarise without re-running: `closed_loop_rollout.py summarize --out <dir>`.
- Resume after a crash: relaunch `run_closed_loop_c0.sh` (same command as the header); finished episodes are skipped.
- Local copy: `rsync -a -e 'ssh -p 45460' root@70.27.250.55:/root/cgs-pilot/artifacts/drive_closed_loop/ artifacts/cgs_pilot/drive_closed_loop/`.

## 9. Status (updated as the session progresses; last line is the newest)

- 17:32 UTC — progress-rule smoke (seed 3, armA_seed0) PASS mechanically: 15/15 rollouts, replay ≤ 2.3e-13, C0 (progress rule) FAIL.
- 17:44 UTC — full-mode runner launched; native smoke (seed 9) started. 17:48 coordinator killed the legacy 12-scene pilot.
- 17:55 UTC — native smoke stopped by me after the hazard-free CEM stall (max_step 995 / 16.6 m) on coordinator instruction; runner stopped before stage 1; no hazard level rolled out under the native label.
- 17:58 UTC — hazard-free planner diagnostics launched (`run_planner_diag.sh`, armA_seed0 + armB_seed0, seeds 9/10/19, 5 configs); marker `PLANNER_DIAG_DONE` in `logs/drive_closed_loop.log`; per-model `driver.log` under `artifacts/drive_closed_loop/planner_diagnostics/<model>/`.
- 18:36 UTC — first diagnostics queue stopped by me mid-episode (armA `compact_rolling6`, armB `compact_rolling10` lost, re-queued) to prioritise the coordinator's range-k diagnostic: `run_planner_diag_rangek.sh` (pass 1: `fixed_k10, fixed_k20`; pass 2: `fixed_k6, fixed_k40, released_k10`; pass 3: the remaining earlier configs), marker `PLANNER_DIAG_RANGEK_DONE`; results appended to the same `diag.jsonl` files.
- 19:39 UTC — range-k pass 1 (k10/k20) done; 21:15 armA pass 2 (k6/k40/released_k10) done; armB pass 2 finishing (released_k10 on seed 19); pass 3 (`compact_rolling6/10`, `released_cem_dest`, seeds 9/10/19) follows automatically.
- 21:34 UTC (after the session-limit resume) — checkpoint-choice remedy launched: armB_seed0 `jepa-best.pth.tar`, `fixed_k10` + `compact_rolling10`, seeds 9/10/19, hazard-free only → `planner_diagnostics/armB_seed0_jepabest/{diag.jsonl,driver.log}`, log `logs/drive_planner_diag_armB_seed0_jepabest.log`. Range-k results written to section 6b.
- 21:59 UTC — jepa-best == jepa-latest for armB_seed0 (identical predictor tensors); the checkpoint-choice run was stopped after reproducing the jepa-latest result. Pass 3 of the range-k queue (`compact_rolling6`, `compact_rolling10`, `released_cem_dest`; seeds 9/10/19; both models; started 21:37 UTC, ≈ 45 min per seed per model when episodes stall) is the only closed-loop job on box 2; poll with `grep -E "compact_rolling|released_cem_dest|DIAG_DONE" artifacts/drive_closed_loop/planner_diagnostics/arm{A,B}_seed0/driver.log` and `grep RANGEK_DONE logs/drive_closed_loop.log`; results append to the same `diag.jsonl`; the table in section 6b can be regenerated with the snippet in `artifacts/cgs_pilot/drive_closed_loop/planner_diagnostics/_trace.py`-style reading (per-row fields: `outcome`, `failure_reason`, `arrive_dest_env_step`, `progress_m`, `n_replans`, `wall_s`, `margin_thr_vs_brake_mean`, `n_last_step_brake_artefact`).
- Pending user/main-session decision: planner configuration amendment before any hazard cell is rolled out; then launch `run_closed_loop_c0.sh` (full mode; smoke marker absent → it will redo the native smoke on seed 9 first, then stages 1–3).
