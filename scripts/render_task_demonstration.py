"""Render task context using MetaWorld scripted policies, not JEPA results.

Fixed seed 0, Reach and Reach-Wall, 160 actions each; no selection search,
model inference, or scientific evaluation. CPU renderer dependencies match
render_qualitative_rollouts.py. The task goal is highlighted in the render only.
"""
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import SawyerReachV3Policy, SawyerReachWallV3Policy
from render_qualitative_rollouts import font, sha
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'docs/media';out.mkdir(exist_ok=True)
    clips=[];records=[]
    for task,policy_type in (('reach',SawyerReachV3Policy),('reach-wall',SawyerReachWallV3Policy)):
        env=ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[task+'-v3-goal-observable'](seed=0)
        renderer=None
        try:
            obs,_=env.reset(seed=0);policy=policy_type()
            env.model.cam_pos[2]=[.75,.075,.7]
            env.model.vis.global_.offwidth=960;env.model.vis.global_.offheight=640
            renderer=mujoco.Renderer(env.model,height=580,width=960)
            frames=[];actions=[];distances=[];successes=[]
            for step in range(161):
                if step:
                    action=np.clip(policy.get_action(obs),-1,1).astype(np.float32)
                    obs,reward,terminated,truncated,info=env.step(action)
                    actions.append(action.tolist());successes.append(float(info['success']))
                distance=float(np.linalg.norm(obs[:3]-obs[-3:]));distances.append(distance)
                if step%2:continue
                renderer.update_scene(env.data,camera='corner2')
                # Highlight the actual environment target; no change to physics.
                scene=renderer.scene
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([.025,.025,.025]),pos=np.asarray(env._target_pos),
                    mat=np.eye(3).ravel(),rgba=np.array([.05,.85,.7,.55]))
                scene.ngeom+=1
                im=Image.fromarray(renderer.render()[::-1].copy())
                frames.append(im)
            clips.append(frames)
            records.append(dict(task=task,seed=0,policy=policy_type.__name__,actions=actions,
                distance_m=distances,success=successes,dt=float(env.dt),steps=160,
                policy_source_sha256=sha(__import__('inspect').getfile(policy_type))))
        finally:
            if renderer:renderer.close()
            env.close()
    composites=[]
    for i in range(81):
        im=Image.new('RGB',(1920,720),'#101923');draw=ImageDraw.Draw(im)
        draw.text((28,12),'Reach the target. Clear the wall.',font=font(36,True),fill='white')
        for col,(label,clip) in enumerate(zip(('Reach','Reach-Wall'),clips)):
            draw.text((col*960+28,66),label,font=font(27,True),fill='#7bddc9')
            im.paste(clip[i],(col*960,105))
            reached=records[col]['distance_m'][i*2]<.05
            draw.text((col*960+28,119), 'TARGET REACHED' if reached else 'MOVE TO GREEN TARGET',font=font(25,True),fill='#101923')
        draw.text((28,685),'Task illustration · MetaWorld scripted policies · not a JEPA-WM rollout · real-time playback',font=font(23),fill='#c2cfda')
        composites.append(im)
    # Two simulation steps per rendered frame = 25 ms; GIF alternates 20/30 ms.
    durations=[20 if i%2==0 else 30 for i in range(81)];durations[-1]=500
    small=[im.resize((960,360),Image.Resampling.LANCZOS) for im in composites[::2]]
    durations=[50]*40+[500]
    sheet=Image.new('RGB',(960,360*len(small)))
    for i,im in enumerate(small):sheet.paste(im,(0,i*360))
    palette=sheet.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    frames=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in small]
    gif=out/'task_demonstration.gif'
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2,optimize=False)
    mp4=out/'task_demonstration_hd.mp4'
    with imageio.get_writer(mp4,fps=40,codec='libx264',quality=8,macro_block_size=1,
                            ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
        for im in composites[:-1]:writer.append_data(np.asarray(im))
        for _ in range(20):writer.append_data(np.asarray(composites[-1]))
    records_path=ROOT/'paper/data/task_demonstration_records.json'
    records_path.write_text(json.dumps(dict(scope='scripted task illustration only',records=records),indent=2)+'\n')
    receipt=dict(scope='scripted task illustration, not JEPA-WM behavior',seed=0,policy_selection='fixed task policies; no seed search',
        new_model_inference=False,new_scientific_evaluation=False,playback_speed=1,total_simulated_seconds=2.0,
        final_hold_ms=500,goal_overlay='actual env._target_pos; render-only translucent sphere',
        input_records_sha256=sha(records_path),generator_sha256=sha(__file__),
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (gif,mp4)})
    (out/'task_demonstration_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for i in (0,10,20,30,40):small[i].save(f'/tmp/task-demo-{i}.png')
    print(json.dumps(receipt,indent=2))
    print([(r['task'],r['distance_m'][0],r['distance_m'][-1],max(r['success'])) for r in records])

if __name__=='__main__':main()
