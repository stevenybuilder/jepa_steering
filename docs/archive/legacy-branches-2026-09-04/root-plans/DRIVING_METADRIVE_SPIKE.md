# MetaDrive feasibility spike — pedestrian-hazard x ego-action factorial (2026-09-02)

**Verdict: GO (conditional)** for MetaDrive 0.4.3 as the driving stimulus simulator for a <= 8-GPU-hour experiment,
with three conditions: (1) rendering must use the NVIDIA EGL path (~1.2 GB VRAM per process; a CPU-only renderer is
NOT available — software `p3tinydisplay` renders garbage, Mesa-EGL fails); (2) hazards must be spawned through the
verified path in `metadrive_spike.py::spawn_hazard` (a `setKinematic` after spawn silently disables collision;
a second `setStatic(True)` call lets the car penetrate); (3) the walking-pedestrian animation must be posed explicitly
per step (it is wall-clock driven otherwise, which breaks frame determinism).

Box: Vast 49155754 (`ssh -p 20566 root@192.220.55.116`), RTX 3090, 64 cores, no `DISPLAY`, no Xvfb.
Env: `/opt/conda/envs/metadrive` (python 3.10, `metadrive-simulator==0.4.3`, panda3d 1.10.13).
Script: `scripts/cgs_pilot/metadrive_spike.py` (synced to `/root/cgs-pilot/code/cgs_pilot/metadrive_spike.py`).
Artifacts + JSON reports: `/root/cgs-pilot/artifacts/metadrive_spike*/` (`report_all.json`, `report_toggle.json`,
`report_contact.json`, `metadrive_spike_det/report_determinism.json`, `metadrive_spike_soft/report_softreset.json`),
logs in `/root/cgs-pilot/logs/metadrive_spike_*.log`. Nothing was deleted; the running GPU job / conversion were untouched.

## 1. Headless offscreen RGB rendering — PASS

| item | value |
|---|---|
| config | `use_render=False, image_observation=True, norm_pixel=False, stack_size=1, sensors={"rgb_camera": (RGBCamera, 256, 256)}, vehicle_config={"image_source": "rgb_camera"}` |
| backend | panda3d `pandagl` fails (no X) and auto-falls back to `p3headlessgl` -> **`eglGraphicsPipe`, OpenGL 4.6 on "NVIDIA GeForce RTX 3090/PCIe/SSE2", driver 580.126.09**. No Xvfb needed. |
| obs | `obs["image"]` shape `(256, 256, 3, 1)` uint8, **BGR** channel order (verified visually; flip to RGB before DINO) |
| env create | 20 s first time (asset load); render-mode `env.reset()` 4.5 s at default `map_region_size=1024` (terrain re-render), **0.48 s at `map_region_size=256`**, 1.38 s at 512 |
| fps | **170–173 env steps/s with 256 px rendering** (147 steps/s at 512 px + cv2 resize), 704 steps/s without rendering |
| VRAM | 1166 MiB for the GL context (`nvidia-smi` type G row) |
| CPU-only | `p3tinydisplay` (software) works mechanically (8.3 steps/s, no GPU row) but the frame is unusable (pink, no PBR shaders/textures). Mesa llvmpipe via `__EGL_VENDOR_LIBRARY_FILENAMES=50_mesa.json` (+`EGL_PLATFORM=surfaceless`, GL overrides) fails: "Not allowed to force software rendering when API explicitly selects a hardware device / failed to create dri2 screen" -> simplepbr tonemap buffer is None. **Rendering needs the GPU.** |

Sample frames: `/root/cgs-pilot/artifacts/metadrive_spike/render_256.png` (correct dashcam view; sky/road/lane lines), `metadrive_spike_tiny/render_256_tiny.png` (garbage).

## 2. Pedestrian / matched object placement and visibility — PASS

APIs (all in `metadrive_spike.py`):
- lane frame: `env.agent.lane.local_coordinates(pos) -> (long, lat)`, `lane.position(long, lat)`; outermost lane = `env.agent.navigation.current_ref_lanes[-1]`. **Positive lateral of the outer lane points to the sidewalk** (sidewalk is 2 m wide, starting at +w/2); negative lateral points to lane 0. Ego spawn lane set with `agent_configs={"default_agent": {"spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, lane_num-1)}}` so the sidewalk hazard is a matched 2.75 m lateral offset.
- pedestrian: `env.engine.spawn_object(Pedestrian, position=(x, y), heading_theta=h)` (`metadrive.component.traffic_participants.pedestrian.Pedestrian`; cylinder r 0.35 m, h 1.75 m, 70 kg; `TYPE_NAME=PEDESTRIAN`, collision group `TrafficParticipants`). Walking: `ped.set_velocity(dir, speed)` (switches to the animated model for speed >= 0.4 m/s) or kinematic `set_position` per step.
- matched inanimate object: `MatchedCone(TrafficCone)` subclass (`metadrive_spike.py::matched_cone_class`) with RADIUS/HEIGHT/MASS copied from `Pedestrian` -> **identical Bullet cylinder (0 % envelope mismatch)**; the visual is rescaled from measured tight bounds to 0.70 m x 1.75 m (the stock cone visual is only 0.64 m tall). Contact sets `crash_object` (TrafficObject group) instead of `crash_human`.
- remove: `env.engine.clear_objects([obj.id])`.

Pixel footprint (rendered-difference mask vs. the same frame without the object, threshold 8/255; includes the shadow; bbox = w x h px at 256 px):

| dist (m) | fov 60 (default) H1 | fov 40 H1 | fov 40 H0 sidewalk |
|---|---|---|---|
| 4 | 101x112 (3462 px) | 59x103 | 59x102 |
| 6 | 70x81 | 112x111 | 97x111 |
| 8 | **54x59** (753 px) | 87x92 | 79x92 |
| 10 | 45x46 | **70x72** (1110 px) | 66x72 |
| 12 | 36x37 | **60x59** (739 px) | 55x59 |
| 15 | 29x30 | **48x47** (480 px) | 46x47 |
| 20 | 21x22 | 36x35 | 34x35 |
| 25 | 19x17 | 29x28 | 28x28 |
| 30 | 11x14 | 20x23 | 19x23 |

- **>= 48x48 px holds for d <= 8 m at the default 60 deg FOV and d <= 15 m at 40 deg FOV** (`env.engine.get_sensor("rgb_camera").get_lens().setFov(40)`). At 8 m/s that is TTC ~1.9 s at 40 deg. Caveat: a pedestrian is thin; at 12 m / 40 deg the silhouette+shadow is 739 px of a 60x59 bbox (~20 %), i.e. ~2 patch columns of solid coverage.
- Matched cone at 11.5 m / 40 deg: **62x61 px (783 px)** vs pedestrian 63x62 (830 px) -> visibility matched.
- Sidewalk H0 (2.75 m lateral) at 12 m / 40 deg: 37x55 px, bbox centre x=223 (in frame). H0' at 4.25 m lateral is at the frame edge (3x24 px) — use a different second pose (e.g. far-side sidewalk or 1.0 m extra offset with flipped heading) for "matched pixel displacement".
- Frames: `ped_H1_d{4..30}_fov{60,40}.png`, `fixed_H0_12m.png`, `vis_cone_fixed_11p5m.png`, `vis_ped_11p5m.png`.

## 3. Ego action interface + contact flags — PASS

- action = `[steer, throttle_brake]`, each in [-1, 1]; 10 Hz control (`physics_world_step_size=0.02` x `decision_repeat=5`). `env.agent.speed` is **m/s** in 0.4.3 (my JSON keys say `speed_kmh_at` — mislabeled, values are m/s).
- From standstill the default vehicle is sluggish (throttle 0.8: 3.5 m/s at 1.5 s, 11 m/s at 5 s), so the context speed is seeded with `env.agent.set_velocity([1, 0], 8.0, in_local_frame=True)` (8 m/s).
- flags: `env.agent.crash_human` (BaseVehicle attribute set in `_body_contact` via `dynamic_world.contactTest(chassis, use_filter=True)` when the contacting node is `PEDESTRIAN`/`CYCLIST`), `env.agent.crash_object` (TrafficObject), `crash_sidewalk`, and the same keys in `info` from `done_function`. Set `crash_human_done=False, crash_object_done=False` to keep stepping. Min distance is computed from `agent.position` / `obj.position` (centre-to-centre).

Cells (hazard 20 m ahead, v0 = 8 m/s, context 10 steps throttle 0.5, chunk 40 steps; A1 = throttle 0.5, A0 = brake -1; solid pedestrian via `spawn_hazard(..., solid=True)`):

| cell | crash_human | first contact step | min centre dist (m) | ego speed m/s t=19/29/49 |
|---|---|---|---|---|
| H1 in-lane, A1 throttle | **True** | 18 | 0.04 | 6.97 / 2.79 / 0.48 (car stops on the body; x@49 = 29.1 m vs 61.0 m unobstructed) |
| H1 in-lane, A0 brake | False | – | 7.33 | 9.38 -> 0.30 (stops within ~1 s) |
| H0 sidewalk (+2.75 m), A1 | False | – | 2.78 | 10.76 / 12.14 / 14.9 (= no-hazard) |
| H0 sidewalk, A0 | False | – | 7.37 | as A0 |
| H0' (+4.25 m), A1 / A0 | False | – | 4.27 / 7.4 | as no-hazard |

-> **contact only in (H1, A1)**. Matched cone: same pattern with `crash_object` True at step 18 (min 0.02 m), ego 6.82 / 1.02 / 0.47 m/s — physics response comparable to the pedestrian's (within ~10 % on the stop profile; identical envelope). Truly dynamic 70 kg body (`spawn_hazard(..., static=False)`): `crash_human` at step 20, pedestrian knocked over (z 0.875 -> 0.31), ego slows only to 10.76 -> 11.41 -> 14.18 m/s.

Gotchas found: `body.setKinematic(True)` after spawn -> **no contact at all** (must `world.remove(body); setKinematic(True); world.attach(body)`); calling `setStatic(True)` twice -> car penetrates (transient 10.76 -> 9.55 m/s, min 0.34 m). `spawn_hazard` uses a single `setStatic(True)` (or the re-attach path with `kinematic=True`), both verified to stop the car.

## Intangible vs solid toggle (coordinator add-on) — PASS

API: Bullet filter is `groups-mask` (`engine_core.py:88`); the object's group is `body.setIntoCollideMask(...)`, pair rules via `dynamic_world.setGroupCollisionFlag(g1, g2, bool)` (`CollisionGroup` bits: Vehicle 1, Terrain 2, TrafficObject 4, Sidewalk 6, TrafficParticipants 10). Intangible = **ghost group bit 12 paired only with Terrain** (`spawn_hazard(..., solid=False)`): render untouched, no `contactTest` hit, no crash flag, no physics response; the body still rests on the ground (z 0.875 constant; `allOff()` also worked in 50 steps but relies on Bullet sleeping, so the ghost group is the safe choice). Works identically for the cone (`solid=False`).

| pair (solid vs intangible, fov 40, 20 m, A1) | solid first contact | intangible crash | intangible min dist (m) | max px diff before contact (n px) | ego state max diff before contact | intangible vs no-hazard: state / frames |
|---|---|---|---|---|---|---|
| ped dynamic vs ghost | 20 | False | 0.40 | 10 (10 px) | 1.8e-12 | 1.8e-12 / jitter only |
| ped static vs ghost | 18 | False | 0.40 | 3 (4 px) | 9.1e-13 | 9.1e-13 |
| cone dynamic vs ghost | 18 | False | 0.40 | 0 (0 px) | 1.8e-12 | 9.1e-13 |
| cone static vs ghost | 18 | False | 0.40 | 10 (9 px) | 2.7e-12 | 1.8e-12 |
| ped/cone at 11.5 m, single frame | – | – | – | 6 (renderer jitter floor) | – | – |

Ego trajectories in no-contact cells are identical to ~1e-12 m (ULP-level, not bit-identical: adding/removing a body changes Bullet's summation order). Frames differ by <= 10 levels in <= 10 of 3.3 M pixels — the same renderer jitter seen between two identical runs.

Walking pedestrian: `ped.set_velocity(dir, 1.2)` each step (ghost, crossing from the sidewalk into the lane; min centre dist 0.85 m). Two identical runs: **max px diff 192 in 22 921 px with the default animation** (wall-clock-driven `loop`) vs **8 in 6 px with explicit `Pedestrian._MODEL[speed].get_anim_control("Take 001").pose(t)`** per step. Ego states between the two runs not bit-identical (1e-12 class). So: walking works for both solid and intangible, but pose the animation explicitly.

## 4. Determinism / replay

- Two fresh processes, same seed + same 50 actions (solid pedestrian, contact at step 18): **ego state trajectories bit-identical (max diff 0.0)**; RGB frames: **max abs pixel diff 8, fraction of differing pixels 1.5e-6** (5 of 3.3 M pixel-channels) -> renderer-only jitter, same class as the egg pilot's v0.3 tolerance (here up to 8/255 on a handful of pixels rather than <= 1/255).
- Within one process, episode-to-episode: physics differs at the 1e-13..1e-12 level (prefix state diff 2.3e-13 in the branch test), frames differ by <= 10 levels on <= 10 pixels.
- Save/restore mid-episode: **MetaDrive has no Bullet state save/restore**; `record_episode`/`replay_episode` replays trajectories (not physics forks). Branching = replay the identical prefix and `spawn_object` the hazard at step T (`check_hazard_spawn_midepisode`: prefix frames max diff 4, prefix state diff 2.3e-13, post-spawn diff 150 as expected). Given the ~0.5 s reset (region 256) this is cheap. Soft reset (reposition the ego without `env.reset`) is 6e-5 s but NOT state-exact (0.4 mm / 0.005 m/s residuals, frame diff 169) -> use hard resets.

## 5. Throughput

| mode | steps/s | reset |
|---|---|---|
| no rendering (lidar obs) | 704 | 0.011 s |
| 256 px RGB | 173 | 4.5 s (region 1024) / **0.48 s (region 256)** |
| 512 px RGB + resize to 256 | 147 | 4.5 s |

10k clips x 20 frames = 200k steps -> 0.32 h of stepping single-process; with one hard reset per clip: +12.6 h at region 1024, **+1.3 h at region 256 -> ~1.7 h single process, ~15 min with 8 processes** (8 x 1.2 GB VRAM = ~9.6 GB; the box currently has ~10 GB free, so 4–6 processes is the safe number while the JEPA jobs run). Cutting several 20-frame clips per episode reduces resets further.

## 6. Footprint

- conda env 1.3 GB (`/opt/conda/envs/metadrive`; metadrive pkg 245 MB incl. assets 168 MB from `github.com/metadriverse/metadrive/releases/download/MetaDrive-0.4.3/assets.zip`). Disk on the box fluctuates 1.3–6.8 GB free because of the running conversion job (not this env).
- Physics (Bullet) is CPU; **rendering is GPU-only in practice** (1.2 GB VRAM per process, negligible compute — the egl context showed no measurable effect on the running 6.8 GB job). Data generation therefore does touch the GPU for the framebuffer; if that is unacceptable the only alternative is a different simulator (CARLA has the same constraint).

## Go / no-go

**GO**, with the caveats above. MetaDrive delivers: deterministic (bit-identical physics across processes, renderer jitter <= 8/255 on ~1e-6 of pixels) headless 256 px RGB at ~170 steps/s; lane-relative pedestrian and envelope-matched cone placement (stationary or walking); clean solid/intangible toggles with identical renders up to contact; `crash_human` / `crash_object` flags with contact only in (H1, A1); >= 48x48 px hazards at <= 15 m with a 40 deg FOV. Not clean: CPU-only rendering (impossible), Bullet state save/restore (absent; use prefix replay), wall-clock animation (must pose explicitly), and the `setKinematic`/double-`setStatic` collision traps (use `spawn_hazard`). A 10k-clip x 20-frame factorial is ~1.7 CPU-process-hours and well inside an 8-GPU-hour budget.
