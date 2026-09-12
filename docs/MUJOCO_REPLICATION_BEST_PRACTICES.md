# MuJoCo replication and validation practices

Reviewed September 11, 2026 using official documentation and pinned JEPA-WM code.
This checklist does not authorize changes to frozen physics, tasks or metrics.

## Terminology and task stacks

The official package supplies **Python bindings** to the native engine/API.
An environment **wrapper** adds reset, observations, action transforms, rewards
and termination rules. Modern `pip install mujoco` includes the native library.
[Python documentation](https://mujoco.readthedocs.io/en/stable/python.html)

MuJoCo exposes a C API usable from C/C++, separating compiled model parameters
(`mjModel`) from changing simulation data (`mjData`). Preserve MJCF/XML and
assets, not only version-specific compiled model binaries.
[Overview](https://mujoco.readthedocs.io/en/stable/overview.html)

| Task | Author implementation used here | Replication rule |
|---|---|---|
| PointMaze | Task wrappers → D4RL/mujoco-py → MuJoCo 2.1 | Restore the legacy dependency path, not a newer maze task. |
| Reach / Reach-Wall | Pinned MetaWorld plus modern `mujoco` | Preserve task version, camera, goal construction, controls and success timing. |
| Push-T | Pymunk/Pygame implementation | A MuJoCo port is not a dependency fix. |
| Wall | Authors' custom dynamics/wrappers | Do not substitute a MuJoCo wall scene. |
| DROID | Recorded images/states/actions and source evaluator | Not a MuJoCo success benchmark or physical-robot closed-loop execution. |

Evidence: pinned [environment dispatcher](../vendor/jepa-wms/evals/simu_env_planning/envs/init.py),
[PointMaze wrapper](../vendor/jepa-wms/evals/simu_env_planning/envs/pointmaze_gym_wrap.py),
and [DROID contract](DROID_METHOD_ALIGNMENT.md). The author README explicitly
identifies PointMaze's MuJoCo 2.1 dependency.
[JEPA-WM setup](https://github.com/facebookresearch/jepa-wms#-mujoco-21-for-pointmaze)

## Physics and model checks

MJCF poses are local to parent bodies; XML angular units depend on compiler
settings, whereas compiled angles use radians. Defaults, inertias, actuators,
contacts and solver settings affect behavior as well as appearance.
[Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html)

Project checklist derived from that guidance:

- Hash XML, meshes/textures, environment source and package versions; record
  timestep, integrator, gravity, solver tolerances, collision/friction settings,
  actuator gains and control ranges.
- Compare action meaning, units, clipping, normalization and frame skip with
  the author adapter. Test known bounded actions on excluded engineering states.
- Inspect zero-input stability, joint direction, contact behavior, penetration
  and numerical warnings. Compiling without errors is not physical validation.
- Preserve the native success criterion and its timing. Switching final-step
  success to any-step success can change the score without improving behavior.
- Never tune physics/cameras/thresholds to make the reference score agree with
  the paper. A justified changed setup needs a separate labelled protocol.

## State, reset and reproducibility

Solver warmstarts can affect results when numerical convergence is incomplete.
Independent per-thread `mjData` with shared read-only `mjModel` supports sampling
parallelism. The split `mj_step1`/`mj_step2` path is not equivalent to RK4;
numerical warnings and automatic simulation resets must be noticed.
[Simulation documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html)

Our acceptance checks:

- Separate training, construction/global, episode/reset, planner-sampling and
  control-bank seeds. Log actual states and images; a matching integer is not
  sufficient to establish paired inputs.
- Preserve persistent logical planner streams and constructor order. Move whole
  streams between GPUs; no padded duplicate scenarios or extra samples per GPU.
- Mid-episode restoration requires the engine's complete relevant state plus
  wrapper counters, controller memory and RNG state—not merely joint positions.
- Replay saved actions on excluded or already observed inputs. Compare states,
  rewards, success, termination and step counts before promoting a new runtime.
- Treat nonfinite states/instability as explicit failures and retain partial
  traces. Do not count silent reset-on-instability as a valid scientific episode.

## Visual verification is model-input verification

Linux rendering can use GLX, headless OSMesa software rendering, or headless EGL
hardware rendering. OpenGL contexts must be current on the appropriate thread.
[Visualization](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)

For a visual world model:

- Verify camera identity/pose/field of view, image orientation, channel order,
  dimensions, crop, resize and normalization.
- Visually inspect fixed initial/goal/action-replay frames: plausible contacts,
  visible robot/object/goal, proper alignment, nonblank images, no asymmetric
  goal overlays or diagnostic annotations in the model input.
- Hash the actual pre-intervention encoder input. Visual plausibility and exact
  pixel identity answer different questions; use both.
- Diagnose state differences and render differences separately. Similar-looking
  images can hide changed physics; equal states can still render differently.
- Do not replace a failed exact-pairing rule with an outcome-selected tolerance.
  If a platform cannot preserve required inputs, keep runs separate or rerun
  the affected paired panel under a justified new frozen runtime.

No new human-style visual inspection of all 97 PointMaze pairs was performed
in this repair. Exact hashes prove agreement with saved tensors, not that the
historical scene/camera was intrinsically correct.

## Python-specific pitfalls and efficiency

Arrays can alias mutable native memory: use copies for logged state snapshots.
Native calls release the GIL, while Python callbacks reacquire it. The `nstep`
argument can avoid repeated Python-call overhead when intervening Python work
is unnecessary. [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)

Do not batch across required control updates, renders, success checks or
termination boundaries. Keep simulation state, renderer contexts and hook state
separate between workers. Test cleanup after exceptions. Pin compatible NumPy,
Cython, Gym and native/binding versions rather than upgrading a running study.

Profile inference, CEM, physics, rendering, host/device transfer and writing
separately. Batch candidate forecasts where the frozen implementation allows;
keep weights/banks resident, limit CPU-thread oversubscription, and distribute
independent complete streams. Moving light physics to a GPU may not accelerate
a forecast-bound planner.

Modern MJX supports accelerator-oriented batching; current documentation also
describes Warp-backed batch rendering. This is useful for future campaigns,
not evidence of equivalence to the frozen legacy engine.
[MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)

No MJX migration, shortened horizon, reduced candidates, changed precision or
different success timing is used to finish these existing panels.
See [scaling notes](../jax_scaling_notes.md) for applied scheduling/I/O changes.

## Access versus missing dependencies

The current MuJoCo repository is Apache-2.0 licensed; separate model/dataset
access and licenses still need their own checks.
[Official repository](https://github.com/google-deepmind/mujoco)

The observed PointMaze failure was **missing `d4rl`**, not an identified license
or authentication failure. The repair followed the author dependency path;
no access controls were bypassed. Diagnose compiler headers, writable build
caches, driver libraries and GL permissions by their actual errors. Do not
replace inputs merely because access or installation is inconvenient.

## What was fixed; what still gates scientific execution

| Item | Evidence / status |
|---|---|
| Missing legacy simulator | Private interpreter with MuJoCo 2.1, mujoco-py 2.1.2.14, pinned D4RL and NumPy 1.26.4; OSMesa build/import passes. Push-T's existing Python environment unchanged. |
| Nested-venv dependencies | Explicit parent-package path after private version pins. |
| Initial checker mismatch | Checker had used episode RNG for global construction; corrected to original construction RNG. Failed receipts retained. This was not evidence of corrupt historical accepted episodes. |
| Saved initial/goal reproduction | **97/97 pairs exact**, including excluded smoke, on CPU without model inference. |
| Receiving planner | Frozen native-repeat, zero-dose, arm/source and planner-budget checks still required on each GPU; input parity does not waive them. |
| Missing controls | Supervisor launched for the same 11 whole streams / 132 episodes. Free devices are used first; occupied Push-T devices are used only after exit. Launch is not completion. |
| Reruns | Dependency-only startup failure created no accepted scientific episodes. Finish missing streams; rerun a completed paired comparison only if an actual validity failure is established. Preserve old evidence. |

Code: [setup](../scripts/vast/prepare_pointmaze_runtime.sh),
[input checker](../scripts/vast/check_pointmaze_runtime.py),
[continuation](../scripts/vast/resume_pointmaze_repaired.py).
The collector includes small repair source/log/receipt files in verified Drive
snapshots, excluding reinstallable virtualenv/binaries. The repaired environment
is not yet a portable container image or a completed full-history replication.
