"""Show saved model rollouts, holding each panel at first success or episode end."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

from capture_jepa_episode import validate_capture
from render_qualitative_rollouts import font, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=2, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    records = [json.loads(p.read_text()) for p in args.inputs]
    if [r['task'] for r in records] != ['reach', 'reach-wall']:
        raise ValueError('Require Reach on the left and Reach-Wall on the right')
    for data in records:
        validate_capture(data['capture'])
        for package, version in data['versions'].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f'Requires {package}=={version}')
        if len(data['capture']['frames']) != 100 or not np.isclose(data['capture']['dt'], .0125):
            raise ValueError('Expected the two complete recorded 80 Hz episodes')
    args.output.mkdir(parents=True)
    # A successful reaching task is complete before the native time limit.
    # Hold its actual first-success state rather than replaying post-success
    # controller oscillation. Never manufacture a successful endpoint.
    endpoints = [next((i + 1 for i, success in enumerate(d['capture']['successes'])
                       if success), len(d['capture']['frames']) - 1) for d in records]
    envs = []; renderers = []; errors = [0., 0.]; samples = []
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0., .65, .17]
    camera.distance = 1.4; camera.azimuth = 225.; camera.elevation = -35.
    mp4 = args.output/'jepa_tasks_side_by_side_hd.mp4'
    gif = args.output/'jepa_tasks_side_by_side.gif'
    try:
        for data in records:
            record = data['capture']
            env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[data['task']+'-v3-goal-observable'](seed=0)
            envs.append(env)
            env._last_rand_vec = np.array(record['rand_vec']); env._freeze_rand_vec = True
            env.reset(); env._target_pos = np.array(record['target'])
            # Recolour the existing target site; do not add overlapping markers.
            env.model.site('goal').pos[:] = record['target']
            env.model.site('goal').rgba[:] = [.05, .9, .65, 1.]
            env.model.vis.global_.offwidth = 960; env.model.vis.global_.offheight = 960
            env.model.vis.quality.offsamples = 4
            renderers.append(mujoco.Renderer(env.model, height=960, width=960))
        with imageio.get_writer(mp4, fps=80, codec='libx264', quality=8, macro_block_size=1,
                ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart']) as writer:
            for i in range(100):
                canvas = Image.new('RGB', (1920, 1080), '#101820')
                draw = ImageDraw.Draw(canvas)
                for j, (env, renderer, data) in enumerate(zip(envs, renderers, records)):
                    record = data['capture']; frame_index = min(i, endpoints[j])
                    saved = record['frames'][frame_index]
                    for key, value in saved['physics'].items():
                        if key == 'time': env.data.time = value
                        else: getattr(env.data, key)[:] = value
                    mujoco.mj_forward(env.model, env.data)
                    env._prev_obs = np.array(saved['state'][18:36])
                    error = float(np.max(np.abs(env._get_obs().astype(np.float32)-np.array(saved['state']))))
                    errors[j] = max(errors[j], error)
                    if error > 1e-6:
                        raise ValueError(f'{data["task"]} frame {i} differs: {error}')
                    renderer.update_scene(env.data, camera=camera)
                    canvas.paste(Image.fromarray(renderer.render().copy()), (j*960, 68))
                    draw.text((j*960+24, 14), ['Reach', 'Reach-Wall'][j], font=font(34, True), fill='white')
                    if i >= endpoints[j]:
                        outcome = 'Target reached' if any(record['successes']) else 'Time limit · target not reached'
                        draw.text((j*960+24, 1039), outcome, font=font(25), fill='#ccd7df')
                    else:
                        draw.text((j*960+24, 1039), f'{frame_index*.0125:.2f} s · JEPA-WM planning',
                                  font=font(25), fill='#ccd7df')
                draw.line((960, 0, 960, 1080), fill='#101820', width=6)
                draw.text((1420, 18), 'Green marker: target', font=font(25), fill='#78d9be')
                if i in (0, 25, 50, 75, 99):
                    samples.append(canvas.resize((960, 540), Image.Resampling.LANCZOS))
                if i == 50:
                    canvas.save(args.output/'preview.png')
                for _ in range(40 if i == 99 else 1):
                    writer.append_data(np.asarray(canvas))
    finally:
        for renderer in renderers: renderer.close()
        for env in envs: env.close()
    # 50 fps maps exactly to GIF's 20 ms clock ticks. A global palette without
    # ordered dithering avoids the moving stipple around hands and object edges.
    filters = ('[0:v]fps=50,scale=1600:900:flags=lanczos,split[a][b];'
               '[a]palettegen=stats_mode=full[p];'
               '[b][p]paletteuse=dither=none')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(mp4), '-filter_complex', filters,
                    '-loop', '0', str(gif)], check=True)
    sheet = Image.new('RGB', (960, 540*len(samples)))
    for i, sample in enumerate(samples): sheet.paste(sample, (0, 540*i))
    sheet.save(args.output/'contact_sheet.jpg')
    with Image.open(gif) as image:
        durations = []
        for i in range(image.n_frames):
            image.seek(i); durations.append(image.info['duration'])
    receipt = {'role': 'qualitative_replay_holding_each_task_at_first_success_or_time_limit',
        'tasks': ['reach', 'reach-wall'], 'seed': 0, 'new_model_inference': False,
        'sources': [{'input_sha256': sha(p), 'unique_states_checked': endpoint + 1,
                     'display_endpoint_frame': endpoint, 'state_max_abs_error': e}
                    for p, endpoint, e in zip(args.inputs, endpoints, errors)],
        'renderer_sha256': sha(__file__),
        'camera': {'lookat': list(camera.lookat), 'distance': camera.distance,
                   'azimuth': camera.azimuth, 'elevation': camera.elevation},
        'display_changes': 'Shared camera and green target site; physics and actions restored unchanged',
        'recorded_motion_seconds_per_task': 1.2375, 'final_hold_seconds': .5,
        'displayed_motion_seconds': [endpoint * .0125 for endpoint in endpoints],
        'success_flags': [bool(any(d['capture']['successes'])) for d in records],
        'gif_fps': 50, 'gif_dither': 'none',
        'mp4_resolution': [1920, 1080], 'gif_resolution': [1600, 900],
        'gif_duration_seconds': sum(durations)/1000,
        'outputs': {p.name: sha(p) for p in (mp4, gif)}}
    (args.output/'jepa_tasks_side_by_side_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
