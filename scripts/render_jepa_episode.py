"""Render a complete captured JEPA-WM episode at simulation speed on CPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

from capture_jepa_episode import validate_capture
from render_qualitative_rollouts import font, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.input.read_text()); record = data['capture']
    validate_capture(record)
    for package, version in data['versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Render requires {package}=={version}')
    if not np.isclose(record['dt'], .0125, atol=1e-12):
        raise ValueError('This renderer expects the pinned 80 Hz MetaWorld simulation')
    if args.output.exists():
        raise ValueError('Use a new output directory')
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[data['task']+'-v3-goal-observable'](seed=0)
    renderer = None
    frames = []; max_error = 0.
    try:
        env._last_rand_vec = np.asarray(record['rand_vec']); env._freeze_rand_vec = True
        env.reset()
        env._target_pos = np.asarray(record['target'])
        env.model.cam_pos[2] = [.75, .075, .7]
        env.model.vis.global_.offwidth = 1920; env.model.vis.global_.offheight = 960
        renderer = mujoco.Renderer(env.model, height=960, width=1920)
        for i, saved in enumerate(record['frames']):
            for key, value in saved['physics'].items():
                if key == 'time': env.data.time = value
                else: getattr(env.data, key)[:] = value
            mujoco.mj_forward(env.model, env.data)
            env._prev_obs = np.array(saved['state'][18:36])
            error = float(np.max(np.abs(env._get_obs().astype(np.float32)-np.asarray(saved['state']))))
            max_error = max(max_error, error)
            if error > 1e-6:
                raise ValueError(f'Captured state differs on frame {i}: {error}')
            if i % 2: continue
            renderer.update_scene(env.data, camera='corner2')
            scene = renderer.scene
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([.025]*3), pos=np.asarray(record['target']),
                mat=np.eye(3).ravel(), rgba=np.array([.05,.85,.7,.55]))
            scene.ngeom += 1
            image = Image.new('RGB', (1920,1080), '#111923')
            image.paste(Image.fromarray(renderer.render()[::-1].copy()), (0,75))
            draw = ImageDraw.Draw(image)
            draw.text((25,14), f"Frozen JEPA-WM · {data['task']} · unsteered · seed {data['seed']}",
                      font=font(34,True), fill='white')
            success = i > 0 and bool(record['successes'][i-1])
            draw.text((25,1038), f"{i*record['dt']:.2f} s simulated · real-time motion · " +
                      ('target reached' if success else 'target not reached'), font=font(27), fill='#c1cedb')
            frames.append(image)
    finally:
        if renderer is not None: renderer.close()
        env.close()
    args.output.mkdir(parents=True)
    mp4 = args.output/'jepa_episode_hd.mp4'; gif = args.output/'jepa_episode.gif'
    with imageio.get_writer(mp4, fps=40, codec='libx264', quality=8, macro_block_size=1,
            ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
        for image in frames[:-1]: writer.append_data(np.asarray(image))
        for _ in range(20): writer.append_data(np.asarray(frames[-1]))
    small = [im.resize((1280,720), Image.Resampling.LANCZOS) for im in frames]
    contact = Image.new('RGB',(1280,720*len(small)))
    for i,im in enumerate(small): contact.paste(im,(0,720*i))
    palette = contact.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [im.quantize(palette=palette,dither=Image.Dither.NONE) for im in small]
    durations = [20 if i%2==0 else 30 for i in range(len(frames)-1)] + [500]
    indexed[0].save(gif,save_all=True,append_images=indexed[1:],duration=durations,
                    loop=0,optimize=False,disposal=2)
    receipt = {'input_sha256':sha(args.input),'renderer_sha256':sha(__file__),
               'all_states_checked':len(record['frames']),'state_max_abs_error':max_error,
               'motion_seconds':len(record['actions'])*record['dt'],'final_hold_seconds':.5,
               'mp4_resolution':[1920,1080],'gif_resolution':[1280,720],
               'new_model_inference':False,'outputs':{p.name:sha(p) for p in [mp4,gif]}}
    (args.output/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__': main()
