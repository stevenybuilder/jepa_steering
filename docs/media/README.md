# JEPA-WM simulator replay

[GIF](jepa_prefix_comparison.gif) · [1920×1080 MP4](jepa_prefix_comparison_hd.mp4) · [Verification receipt](replay_receipt.json)

The two rows show Reach and Reach-Wall. Columns show plans selected by unsteered,
learned rank-four, and calibrated random-subspace JEPA-WM. Each clip executes the
saved first fifteen elementary actions from the same recorded simulator state.
These are physical simulator renders, not decoded model predictions.

**Selection:** episode 4, the first case of the registered 56-context extension,
in each task. All three plan arms are shown. The cases were chosen by order before
rendering. They are illustrative, not selected success stories or estimates of
average performance.

Each action advances 0.0125 seconds of simulator time. The 0.1875-second prefixes
are shown at 16× slower playback, with an initial and final hold. The clips cover
one selected prefix, without subsequent replanning or full-task completion.

## Verification

The input export retains the actions, physical state, observation states, rewards,
and flags from generation-pinned, SHA256-verified source archives. The renderer
restores all saved MuJoCo integration fields, including warmstart, and checks the
initial observation and all fifteen subsequent states against the archived values.
The maximum state discrepancy is 1.4211e-14 on Reach and zero on Reach-Wall; all
rewards, success flags, and termination flags match. The camera uses upstream
`corner2` and its vertical flip at a larger rendering resolution.

No model inference, refitting, new scientific evaluation, or cloud GPU rental is
performed. The original analyses and their uncertainty are unchanged.

## Reproduce on CPU

Use a separate Python 3.10 environment to keep rendering dependencies out of the
analysis environment:

```bash
python3.10 -m venv /tmp/jepa-render
/tmp/jepa-render/bin/pip install mujoco==3.3.0 metaworld==3.1.1 gymnasium==1.3.0 numpy==2.2.6 pillow imageio imageio-ffmpeg packaging
/tmp/jepa-render/bin/python scripts/render_qualitative_rollouts.py
```

Run from the repository root. macOS uses the local OpenGL backend; Linux needs a
working MuJoCo rendering backend (for example OSMesa for CPU rendering). The script
fails if state, reward, or flag agreement exceeds its stated tolerances. Rendering
libraries and fonts can change compressed image bytes across systems; compare
replay diagnostics rather than expecting pixel hashes to match across renderers.

[Input records](../../paper/data/qualitative_replay_inputs.json) ·
[Renderer](../../scripts/render_qualitative_rollouts.py) ·
[Full physical-prefix analysis](../PLANNED_PREFIX_REPLAY.md)

## Earlier scripted task illustration

The earlier [task_demonstration.gif](task_demonstration.gif) remains available with a
[1920×720 MP4](task_demonstration_hd.mp4). It shows the MetaWorld scripted Reach
and Reach-Wall policies with the actual target highlighted in green. **It is a
task illustration, not a JEPA-WM rollout or evidence for steering efficacy.**
That distinction appears both in the clip and beside it in the README.

Both tasks use seed 0 and 160 elementary actions without a seed search. Motion
plays at simulation speed (two seconds), followed by a half-second final hold.
The clip shows task completion; the short measured JEPA prefixes above remain
available separately. Full JEPA episode logs preserve action hashes rather than
numeric actions, so they cannot reconstruct an extended measured rollout.

Reproduce with the same rendering environment:

```bash
python scripts/render_task_demonstration.py
```

[Scripted actions and distances](../../paper/data/task_demonstration_records.json) ·
[Generation receipt](task_demonstration_receipt.json).

## Complete JEPA episodes used in the README

[Real-time GIF](jepa_full_episodes.gif) · [1920×1080 MP4](jepa_full_episodes_hd.mp4) ·
[Rendering receipt](jepa_full_episodes_receipt.json)

These are new qualitative runs of the frozen MetaWorld JEPA-WM checkpoint in
strict FP32, without activation steering. Reach and Reach-Wall both use seed 0,
chosen before capture. Reach succeeds; Reach-Wall does not. Both entire native
agent executions are shown, with no seed search or outcome selection.

The native episode limit is 100 steps. The agent loop starts at elapsed step 1,
so each capture contains 99 executed actions and 100 simulator states. The
planner requests 100 actions in total; the environment executes 99 before the
limit. Seven replans each retain fifteen CEM iterations and 300 candidates.
The 99 executed actions cover 1.2375 seconds of simulation time per task. Playback
omits planner computation and adds a half-second hold after each task.
The MP4 retains every captured state at 80 fps; the 1280×720 GIF samples at
40 fps with the format's centisecond timing. It runs at approximately simulation
speed, without the earlier prefix clip's 16× slowdown.

All saved observations are checked against restored MuJoCo physics before
rendering. The task and outcome labels occupy separate bands outside the robot
view. A transparent green marker indicates the actual simulator target. The
presentation takes inspiration from the unobstructed workspace and visible goals
in [Sholto Douglas's robotics videos](https://sholtodouglas.github.io/).

The original logs retained episode summaries and action hashes, which cannot
reconstruct full videos. These new numerical captures are separate from the
published scientific evaluation. The first capture attempt had an incorrect
100-action validation expectation; its 99-action record was preserved, and the
same-seed retry produced identical actions. The corrected validator reads the
native loop's actual remaining step count. Both completed records and technical
logs were archived to Google Cloud and read back with matching SHA256 before
releasing the GPU.

[Capture protocol](../../paper/data/jepa_media_capture_protocol.json) ·
[Reach actions and states](../../paper/data/jepa_episode_reach.json) ·
[Reach-Wall actions and states](../../paper/data/jepa_episode_reach_wall.json).

### Reproduce

Capture on the pinned CUDA runtime:

```bash
python scripts/capture_jepa_episode.py \
  --vendor vendor/jepa-wms --checkpoint /path/to/jepa_wm_metaworld.pth.tar \
  --task reach --seed 0 --output /path/to/new-media-capture
```

Render locally with MuJoCo 3.3.0, MetaWorld 3.1.1, Gymnasium 1.3.0, NumPy 1.26.4,
Pillow, imageio, imageio-ffmpeg, and packaging; the compositor also needs FFmpeg:

```bash
python scripts/render_jepa_episode.py \
  --input paper/data/jepa_episode_reach.json --output /tmp/render-reach
python scripts/render_jepa_episode.py \
  --input paper/data/jepa_episode_reach_wall.json --output /tmp/render-reach-wall
python scripts/compose_jepa_episode_media.py \
  --renders /tmp/render-reach /tmp/render-reach-wall --output /tmp/jepa-media
```
