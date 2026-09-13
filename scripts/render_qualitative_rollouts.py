"""Render saved action prefixes on CPU; verify every state against the archive.

Requires mujoco==3.3.0, metaworld==3.1.1, gymnasium==1.3.0, numpy==2.2.6,
pillow, imageio, imageio-ffmpeg and packaging. No model weights or GPU rental.
The JSON export contains only the selected physical replay records, not forecasts.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4')
LABELS = ('Unsteered', 'Learned rank-four', 'Random subspace')
TASKS = ('reach', 'reach-wall')
# Rendering qualification only, not a replacement scientific evaluation.
STATE_ATOL = 1e-6
REWARD_ATOL = 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(env, record):
    saved = record['initial_physics']
    for key, value in saved.items():
        if key == 'time':
            env.data.time = value
        else:
            getattr(env.data, key)[:] = value
    mujoco.mj_forward(env.model, env.data)
    env.data.qacc_warmstart[:] = saved['qacc_warmstart']
    env._prev_obs = np.array(record['initial_state'][18:36])
    obs = env._get_obs().astype(np.float32)
    error = float(np.max(np.abs(obs - np.array(record['initial_state']))))
    if error > STATE_ATOL:
        raise ValueError(f'Initial observation mismatch: {error}')
    return error


def replay(task, case):
    frames, audits = {}, {}
    for arm in ARMS:
        env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[task+'-v3-goal-observable'](seed=0)
        renderer = None
        try:
            env.seeded_rand_vec = False
            env._freeze_rand_vec = False
            env.reset()
            env.seeded_rand_vec = True
            env.seed(case['seed'])
            env.reset()
            env.step(np.zeros(4, dtype=np.float32))  # upstream one-frame warmup
            saved = case['physical'][arm]
            initial_error = restore(env, saved)
            # Same named camera and position as upstream, with a larger framebuffer.
            env.model.cam_pos[2] = [.75, .075, .7]
            env.model.vis.global_.offwidth = 640
            env.model.vis.global_.offheight = 480
            renderer = mujoco.Renderer(env.model, height=440, width=640)
            def capture():
                renderer.update_scene(env.data, camera='corner2')
                # Match the vertical flip in the pinned MetaWorldWrapper.
                return Image.fromarray(renderer.render()[::-1].copy())
            images = [capture()]
            errors, reward_errors = [], []
            actions = np.asarray(saved['executed_actions'], dtype=np.float32)
            if actions.shape != (15, 4):
                raise ValueError('Expected exactly fifteen saved elementary actions')
            for i, action in enumerate(actions):
                obs, reward, terminated, truncated, info = env.step(action.copy())
                error = float(np.max(np.abs(obs.astype(np.float32)-np.array(saved['states'][i]))))
                reward_error = float(abs(float(np.float32(reward))-saved['rewards'][i]))
                if error > STATE_ATOL or reward_error > REWARD_ATOL:
                    raise ValueError(f'{task}/{arm}/step{i+1}: replay mismatch {error}, {reward_error}')
                if bool(terminated) != bool(saved['dones'][i]) or info['success'] != saved['successes'][i]:
                    raise ValueError('Saved termination or success flag differs')
                errors.append(error); reward_errors.append(reward_error)
                images.append(capture())
            if abs(env.data.time-saved['endpoint_physics']['time']) > 1e-12:
                raise ValueError('Simulator time mismatch')
            frames[arm] = images
            audits[arm] = dict(initial_state_max_abs_error=initial_error,
                state_max_abs_error=max(errors), reward_max_abs_error=max(reward_errors),
                all_15_states_checked=True, success_and_termination_equal=True,
                elementary_step_seconds=float(env.dt), actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
        finally:
            if renderer is not None: renderer.close()
            env.close()
    return frames, audits


def font(size, bold=False):
    candidates = [Path('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf')]
    for path in candidates:
        if path.exists(): return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT/'paper/data/qualitative_replay_inputs.json')
    ap.add_argument('--output', type=Path, default=ROOT/'docs/media')
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    for package, version in data['source_versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Require source package {package}=={version}')
    if set(data['tasks']) != set(TASKS): raise ValueError('Both selected tasks are required')
    all_frames, audits = {}, {}
    for task in TASKS:
        case = data['tasks'][task]
        if case['episode'] != 4 or set(case['physical']) != set(ARMS):
            raise ValueError('Example selection or arms changed')
        all_frames[task], audits[task] = replay(task, case)
        print(task, json.dumps(audits[task]), flush=True)
    composites = []
    for step in range(16):
        image = Image.new('RGB', (1920, 1080), '#111923')
        draw = ImageDraw.Draw(image)
        draw.text((24, 8), 'Frozen JEPA-WM · selected action prefixes', font=font(30, True), fill='white')
        for col, label in enumerate(LABELS):
            draw.text((col*640+24, 49), label, font=font(25, True), fill=['#8eb4e5','#79c7be','#e2b276'][col])
        for row, task in enumerate(TASKS):
            y = 84+row*474
            draw.text((24, y), ('Reach' if task=='reach' else 'Reach-Wall')+' · development case 4',
                font=font(23), fill='#c1cedb')
            for col, arm in enumerate(ARMS):
                image.paste(all_frames[task][arm][step], (col*640, y+30))
        draw.text((24, 1040), f'Action {step:02d}/15  ·  {step*.0125:.4f} s simulated  ·  16× slower playback',
            font=font(23), fill='#c1cedb')
        composites.append(image)
    args.output.mkdir(parents=True, exist_ok=True)
    gif = args.output/'jepa_prefix_comparison.gif'
    mp4 = args.output/'jepa_prefix_comparison_hd.mp4'
    durations = [800]+[200]*14+[1000]
    small = [im.resize((960, 540), Image.Resampling.LANCZOS) for im in composites]
    # One palette across the sequence prevents frame-to-frame color shifts.
    contact = Image.new('RGB', (960, 540*len(small)))
    for i, im in enumerate(small): contact.paste(im, (0,i*540))
    palette = contact.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in small]
    indexed[0].save(gif, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=False, disposal=2)
    with imageio.get_writer(mp4, fps=30, codec='libx264', quality=8, macro_block_size=1,
            ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
        for im, duration in zip(composites, durations):
            for _ in range(round(duration/1000*30)): writer.append_data(np.asarray(im))
    composites[0].save(args.output/'jepa_prefix_poster.jpg', quality=92)
    receipt = dict(selection=data['selection'], input_sha256=sha(args.input),
        generator_sha256=sha(__file__), source_versions=data['source_versions'], platform=platform.platform(),
        state_tolerance=STATE_ATOL, reward_tolerance=REWARD_ATOL, audits=audits,
        frames=16, executed_actions_per_plan=15, total_simulated_seconds=.1875,
        playback_slowdown=16, initial_hold_ms=800, final_hold_ms=1000,
        camera='upstream corner2, cam_pos[2]=[.75,.075,.7], vertically flipped; 640x440 render tiles',
        new_model_inference=False, new_scientific_evaluation=False,
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [gif,mp4,args.output/'jepa_prefix_poster.jpg']})
    (args.output/'replay_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt['outputs'],indent=2))


if __name__ == '__main__': main()
