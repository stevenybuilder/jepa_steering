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
