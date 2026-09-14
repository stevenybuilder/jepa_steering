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

## Full task illustration in the README

The README now embeds [task_demonstration.gif](task_demonstration.gif), with a
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

## Complete JEPA episode capture: prepared, not yet executed

The full evaluation executor returned episode summaries and saved action hashes;
it did not retain the numerical actions or image sequence. Those hashes verify
identity but cannot be inverted into a video. The physical-prefix study retained
only fifteen actions. A full model clip therefore needs a new qualitative run.

[`capture_jepa_episode.py`](../../scripts/capture_jepa_episode.py) captures one
unsteered episode, including all 100 elementary actions and 101 simulator states.
It starts recording after expert goal construction, so the clip contains only
the JEPA-WM agent's execution. Task and seed are fixed before capture, and the
entire episode is retained regardless of success. It writes a separate media
artifact and does not add a new measurement to the published evaluation.

On a qualified CUDA worker with the existing pinned checkpoint and runtime:

```bash
python scripts/capture_jepa_episode.py \
  --vendor vendor/jepa-wms --checkpoint /path/to/jepa_wm_metaworld.pth.tar \
  --task reach --seed 0 --output /path/to/new-media-capture
```

After downloading `episode.json`, render on CPU in the matching simulator environment:

```bash
python scripts/render_jepa_episode.py \
  --input /path/to/new-media-capture/episode.json \
  --output /path/to/new-render-output
```

The renderer checks every saved observation against the restored physics. It
targets a **1920×1080 MP4 at 40 fps** and a **1280×720 GIF**, with real-time
motion and a half-second final hold. The complete 100-action episode represents
1.25 seconds of simulated motion; planner computation time is omitted from
playback. These scripts are prepared, with capture-hook tests passing, but the
GPU capture and full-episode render have not yet run. No replacement clip is
claimed or linked from the README until those checks pass.
