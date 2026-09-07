#!/usr/bin/env python3
"""One frozen RGB nuisance, same four development starts/actions; no training/CEM/sim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch

CHECKPOINT='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
SIGMA=.01
SEED=2026090631
CONTEXT_MAXABS=1e-4
DINO_SHA='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def exact(a,b,name):
    if not torch.equal(a,b):raise ValueError('Exact guard: '+name)


def initial_rgb(frame):
    if frame.ndim!=4 or frame.shape[:2]!=(1,3) or frame.dtype!=torch.uint8:
        raise ValueError('Require exact cached environment observation [1,3,H,W] uint8')
    return frame.unsqueeze(0).float()


def rank_indices(costs):
    return costs.argsort(dim=1,stable=True)


def noisy_display(raw,seed,sigma=SIGMA):
    """Raw channel-first [0,255] -> display RGB nuisance -> raw float, no requantization."""
    if raw.ndim!=5 or raw.shape[2]!=3:raise ValueError('Require B,T,3,H,W raw RGB')
    raw=raw.detach().cpu().float()
    if not torch.isfinite(raw).all() or raw.min()<0 or raw.max()>255:raise ValueError('Raw RGB range')
    if sigma<0:raise ValueError('Negative sigma')
    if sigma==0:return raw.clone(),dict(sigma=0.,clipped_fraction=0.,realized_display_RMS=0.)
    display=raw/255.
    generator=torch.Generator(device='cpu').manual_seed(seed)
    noise=torch.randn(display.shape,generator=generator,dtype=torch.float32)*sigma
    intended=display+noise;changed=intended.clamp(0,1)
    return changed*255.,dict(sigma=sigma,seed=seed,clipped_fraction=float(((intended<0)|(intended>1)).float().mean()),
        requested_display_RMS=float(noise.double().square().mean().sqrt()),realized_display_RMS=float((changed-display).double().square().mean().sqrt()),
        changed_pixel_channel_fraction=float((changed!=display).float().mean()),display_min=float(changed.min()),display_max=float(changed.max()),
        requantized_to_uint8=False,physical_state_modified=False)


def paired_spread(clean,noisy):
    """Per-H/action MSE; normalize RMS by clean alternative-minus-central RMS, not variance."""
    if clean.shape!=noisy.shape or clean.shape[1]!=9:raise ValueError('Same H,9 actions,features required')
    clean=clean.double();noisy=noisy.double()
    drift=(noisy-clean).square().flatten(2).mean(-1)
    action=(clean[:,1:]-clean[:,:1]).square().flatten(2).mean(-1).mean(1)
    mean=drift.mean(1)
    return dict(mse_by_horizon_candidate=drift.tolist(),mean_mse_by_horizon=mean.tolist(),RMS_by_horizon=mean.sqrt().tolist(),
        clean_alternative_minus_central_mse_by_horizon=action.tolist(),
        RMS_over_native_action_RMS=[None if d<=1e-20 else float((n/d).sqrt()) for n,d in zip(mean,action)],
        denominator_tiny=[bool(d<=1e-20) for d in action],normalization='sqrt(mean noise MSE across all9 / mean clean alternative-minus-central MSE across8); not MSE ratio')


def cpu(value):return {k:v.detach().cpu().clone() for k,v in value.items()}


def parameter_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def source(a,row,wm,prep,first):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from diagnose_pusht_goal_coverage import coverage,native_helper
    start=time.monotonic();path=a.bank/row['path']
    if sha(path)!=row['sha256']:raise ValueError('Source SHA before load')
    bank=torch.load(path,map_location='cpu',weights_only=False)
    episode=int(row['key'][-3:]);candidates=bank['candidates']
    if bank['row']['split']!='development_external' or len(candidates)!=9:raise ValueError('Original near9 DEV only')
    if bank['context']['visual'].shape[:2]!=(1,1):raise ValueError('Only saved single-frame initial context reconstructable; no inferred history')
    frame=candidates[0]['frames_modelsteps'][0]
    raw=initial_rgb(frame)
    states=np.stack([c['states'] for c in candidates])
    for c in candidates:
        exact(c['frames_modelsteps'][0],frame,'all nine plans exact initial pixels')
        if not np.array_equal(c['states'][0],states[0,0]):raise ValueError('All nine plans exact initial state')
    proprio=torch.tensor(states[0,0,[0,1,5,6]],dtype=torch.float32)[None,None]
    zero_raw,_=noisy_display(raw,SEED+episode,0.);exact(zero_raw,raw,'zero RGB exact')
    def encode(rgb):return wm.encode({'visual':rgb,'proprio':proprio},act=False)
    clean_encoded=encode(raw);zero_encoded=encode(zero_raw)
    for key in ('visual','proprio'):exact(clean_encoded[key],zero_encoded[key],'zero-noise encoder identity '+key)
    parity={key:dict(maxabs=float((clean_encoded[key].cpu()-bank['context'][key]).abs().max()),
        MSE=float((clean_encoded[key].cpu().double()-bank['context'][key].double()).square().mean()),
        same_shape=list(clean_encoded[key].shape)==list(bank['context'][key].shape)) for key in ('visual','proprio')}
    write(a.output/(row['key']+'-context-parity.json'),dict(episode=episode,parity=parity,historical_v2_allowed_maxabs=CONTEXT_MAXABS,
        v3_baseline='New authorized fresh-clean paired baseline, historical visual context difference descriptive; not exact old continuation'))
    if any(not p['same_shape'] for p in parity.values()):raise ValueError('Raw source context shape differs')
    if not torch.isfinite(clean_encoded['visual']).all():raise ValueError('Nonfinite fresh native visual encoder')
    exact(clean_encoded['proprio'].cpu(),bank['context']['proprio'],'exact original true-proprio context')
    normalized=prep.normalize_actions(bank['raw_actions'].reshape(9,6,5,2)).reshape(9,6,10)
    exact(normalized,bank['normalized_actions'],'same official action normalization')
    actions=normalized.transpose(0,1).contiguous().to(wm.device);saved_actions=actions.clone()
    goal=TensorDict(bank['goal_encoded'],batch_size=[]).to(wm.device)
    frozen_goal=goal.clone();objective=ReprTargetDistMPCObjective({},goal,sum_all_diffs=False,alpha=.1)
    def unroll(encoded):
        context=TensorDict({k:v.to(wm.device).repeat_interleave(9,dim=0) for k,v in encoded.items()},batch_size=[])
        return wm.unroll(context,act_suffix=actions)
    cached=unroll(bank['context']);clean=unroll(clean_encoded);zero=unroll(zero_encoded)
    for key in ('visual','proprio'):exact(clean[key],zero[key],'zero-noise native H6 identity '+key)
    old_cost=torch.stack([c['native_cost_by_horizon'].flatten() for c in candidates],dim=1)
    cached_cost=objective(cached,actions,keepdims=True).cpu();clean_cost=objective(clean,actions,keepdims=True).cpu()
    old_choice=int(old_cost[-1].argmin());clean_choice=int(clean_cost[-1].argmin());cached_choice=int(cached_cost[-1].argmin())
    if old_choice!=cached_choice or old_choice!=clean_choice:raise ValueError('Clean raw/cached native choice differs from original nine-bank choice')
    # v3 predeclares a fresh paired baseline. Do not encode any nuisance until all clean guards pass.
    noisy_raw,noise_meta=noisy_display(raw,SEED+episode)
    repeated_noise,_=noisy_display(raw,SEED+episode);exact(repeated_noise,noisy_raw,'fixed state noise RNG')
    noisy_encoded=encode(noisy_raw);exact(noisy_encoded['proprio'],clean_encoded['proprio'],'true proprio unchanged')
    noisy=unroll(noisy_encoded)
    forecasts={'cached_context':cpu({k:cached[k][1:] for k in ('visual','proprio')}),
        'clean':cpu({k:clean[k][1:] for k in ('visual','proprio')}),'noisy':cpu({k:noisy[k][1:] for k in ('visual','proprio')})}
    costs={'cached_context':cached_cost[1:],'clean':clean_cost[1:],'noisy':objective(noisy,actions,keepdims=True)[1:].cpu()}
    actual_visual=torch.stack([c['actual_visual'] for c in candidates],dim=1).float()
    raw_proprio=torch.tensor(states[:,::5][:,:,[0,1,5,6]],dtype=torch.float32)
    actual_proprio=wm.model.encode_proprio(prep.normalize_proprios(raw_proprio).to(wm.device)).cpu()[:,1:].transpose(0,1).contiguous()
    actual={'visual':actual_visual,'proprio':actual_proprio}
    goal_state=np.asarray(bank['goal_state']);endpoints=states[:,-1]
    cover=np.array([coverage(s[2:5],goal_state[2:5]) for s in endpoints]);xy=np.linalg.norm(endpoints[:,:4]-goal_state[:4],axis=1)
    native_poly=native_helper(a.repo/'evals/simu_env_planning/envs/pusht_env/pusht_env.py');target_poly=native_poly(goal_state[2:5])
    checked=np.array([native_poly(s[2:5]).intersection(target_poly).area/target_poly.area for s in endpoints])
    if np.max(np.abs(checked-cover))>1e-12:raise ValueError('Requested-goal native polygon parity')
    rows=[]
    for arm,prediction in forecasts.items():
        choice=int(costs[arm][-1].argmin())
        error={k:(prediction[k].double()-actual[k].double()).square().flatten(2).mean(-1) for k in actual}
        rows.append(dict(arm=arm,choice=choice,cost_by_horizon=costs[arm].tolist(),rank_by_horizon=rank_indices(costs[arm]).tolist(),
            actual_MSE_by_horizon_candidate={k:v.tolist() for k,v in error.items()},actual_mean_MSE_by_horizon={k:v.mean(1).tolist() for k,v in error.items()},
            selected_requested_goal_coverage=float(cover[choice]),selected_goal_coverage_delta_vs_clean=float(cover[choice]-cover[clean_choice]),
            selected_combined_agent_block_XY_distance_px=float(xy[choice]),selected_XY_delta_vs_clean_px=float(xy[choice]-xy[clean_choice])))
    encoder_drift={k:dict(L2=float((noisy_encoded[k]-clean_encoded[k]).double().norm()),MSE=float((noisy_encoded[k]-clean_encoded[k]).double().square().mean())) for k in ('visual','proprio')}
    spread={k:paired_spread(forecasts['clean'][k],forecasts['noisy'][k]) for k in ('visual','proprio')}
    cross={k:paired_spread(forecasts['cached_context'][k],forecasts['clean'][k]) for k in ('visual','proprio')}
    exact(actions,saved_actions,'all action plans unchanged')
    for key in ('visual','proprio'):exact(goal[key],frozen_goal[key],'same goal target '+key)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    summary=dict(complete=True,episode=episode,source_sha256=row['sha256'],seconds=seconds,noise=noise_meta,encoder_drift=encoder_drift,
        raw_context_parity=parity,cached_single_vs_batch9_cost_maxabs=float((cached_cost-old_cost).abs().max()),cached_single_choice=old_choice,
        cross_context_numerical_floor=cross,spread=spread,rows=rows,requested_goal_coverage_by_candidate=cover.tolist(),combined_XY_distance_by_candidate=xy.tolist(),
        zero_noise_encoder_and_forecast_exact=True,plans_goals_props_unchanged=True,physical_state_unchanged_by_construction=True)
    p=a.output/(row['key']+'.pt')
    torch.save(dict(summary=summary,raw_clean=raw,raw_noisy=noisy_raw,raw_proprio=proprio,encoded_clean=cpu(clean_encoded),encoded_noisy=cpu(noisy_encoded),
        goal_encoded=cpu(goal),raw_actions=bank['raw_actions'],normalized_actions=normalized,forecasts=forecasts,costs=costs,actual=actual),p)
    report=p.with_suffix('.json');write(report,summary)
    print(json.dumps(dict(event='image_nuisance_state_complete',episode=episode,seconds=seconds,choices={r['arm']:r['choice'] for r in rows},
        visual_noise_action_RMS_ratio=spread['visual']['RMS_over_native_action_RMS'])),flush=True)
    if first:
        estimate=a.initialization_seconds+seconds*4
        write(a.output/'CANARY.json',dict(complete=True,estimated_process_seconds=estimate,limit=a.max_process_seconds,raw_context_parity=parity,zero_exact=True,native_choice_exact=True))
        if estimate>a.max_process_seconds:raise ValueError('First canary estimate exceeds frozen process cap')
    return [dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size) for f in (p,report,a.output/(row['key']+'-context-parity.json'))],summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bank','repo','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-process-seconds',type=float,default=180.)
    a=p.parse_args();start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
    try:
        done=json.loads((a.bank/'DONE.json').read_text());rows=sorted([r for r in done['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['key'])
        if not done['complete'] or [r['key'] for r in rows]!=[f'near-dev-{i:03}' for i in range(4)] or any(r['split']!='development_external' for r in rows):raise ValueError('Exactly original four DEVELOPMENT banks before load')
        if sha(a.checkpoint)!=CHECKPOINT:raise ValueError('Frozen checkpoint SHA')
        dino_path=Path('/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
        if sha(dino_path)!=DINO_SHA:raise ValueError('Frozen DINO weight SHA differs from original source host')
        write(a.output/'protocol.json',dict(exploratory=True,episodes=list(range(4)),actions_per_state=9,sigma_display_RGB=SIGMA,seed_formula=f'{SEED}+episode',
            noise='Single CPUfloat32 Gaussian draw per state, clip displayRGB[0,1], float raw255 to canonical native preprocessing; no uint8 requantization; same input all9actions',
            context='Only exact saved initial RGB plus true agentXY/velocity proprio; no inferred missing history',
            baseline='Explicitly new v3 fresh-clean RGB encoding; clean/zero repeated encoder and H6 exact; original cached9choice exact before noisy encoder; cached visual mismatch descriptive not exact historical continuation',
            historical_v2_context_limit=CONTEXT_MAXABS,frozen_DINO_sha256=DINO_SHA,
            primary='Paired H1-H6 forecast drift normalized by native within-state alternative-minus-central action-response RMS, separately visual/proprio',
            secondary='Actual cached future MSE, native fixed-pool ranks/choice, selected requested-goal coverage and combined agent/blockXY',
            source_rows=rows,source_DONE_sha256=sha(a.bank/'DONE.json'),script_sha256=sha(__file__),checkpoint_sha256=CHECKPOINT,process_pid=os.getpid(),max_process_seconds=a.max_process_seconds,
            protocol_version=3,prior_failed_versions='v1 layout8.8338085seconds andv2 historical context parity9.3800071seconds preserved; new fresh-paired protocol authorized before nuisance outcomes',
            architecture_audit=dict(visual_encoder='Shared frozen DINOv2-vits14, no runtimeEMAteacher',mask='Temporal blockcausal attention, not spatial maskedtarget reconstruction',
                stopgrad='Rollout history/targets detached; one-step learned proprio targets not explicitly detached',RGB_decoder='Excluded; native objective visualMSE +.1proprioMSE',
                historical_training_config='Consistent matching50epoch freeze_encoder:true YAML; not cryptographically bound inside9bec checkpoint',
                training_config_sha256='7a16e8edbf40b260aab303949e90c1879dd5f778d51f3b6d6cb4bea1d99969e2'),
            reference='https://arxiv.org/abs/2608.12939',scope='ACPC-inspired fixed nuisance sensitivity only, not full IR/SR, certification, training, or explanation of original clean rollout failures',
            unit='Four reused development states; one noise draw each, nine dependent plans',held_access=False,training_steps=0,cem_calls=0,simulator_calls=0))
        from model_loader import load_headless
        torch.set_num_threads(2);torch.manual_seed(90505)
        wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
        wm.eval().requires_grad_(False);before=parameter_sha(wm);a.initialization_seconds=time.monotonic()-start
        outputs=[];summaries=[]
        for i,row in enumerate(rows):
            if time.monotonic()-start>a.max_process_seconds-20:raise ValueError('No next state near process cap')
            files,summary=source(a,row,wm,prep,i==0);outputs+=files;summaries.append(summary)
        if parameter_sha(wm)!=before:raise ValueError('Frozen parameter hash changed')
        seconds=time.monotonic()-start
        if seconds>a.max_process_seconds:raise ValueError('Total native process cap exceeded')
        write(a.output/'summary.json',dict(complete=True,states=4,seconds=seconds,rows=summaries,parameters_sha256=before,
            interpretation='One fixed image-nuisance sensitivity diagnostic. No steering promotion, robust-control certification or causal explanation of clean prediction error.'))
        for f in (a.output/'protocol.json',a.output/'CANARY.json',a.output/'summary.json'):outputs.append(dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size))
        write(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=seconds,parameters_sha256=before,provenance=provenance,checkpoint_sha256=CHECKPOINT,
            held_access=False,training_steps=0,cem_calls=0,simulator_calls=0))
        print(json.dumps(dict(event='image_nuisance_DONE',seconds=seconds)),flush=True)
    except Exception:
        write(a.output/'FAILED.json',dict(complete=False,seconds=time.monotonic()-start,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
