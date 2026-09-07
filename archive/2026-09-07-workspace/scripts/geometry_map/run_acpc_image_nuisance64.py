#!/usr/bin/env python3
"""Same frozen RGB nuisance on existing 64-plan banks; no new physics or fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch

try:
    from .run_acpc_image_nuisance import CHECKPOINT,DINO_SHA,SIGMA,SEED,sha,write,exact,noisy_display,rank_indices,parameter_sha,cpu
except ImportError:
    from run_acpc_image_nuisance import CHECKPOINT,DINO_SHA,SIGMA,SEED,sha,write,exact,noisy_display,rank_indices,parameter_sha,cpu


def tensor_sha(x):
    x=torch.as_tensor(x).detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def mse_rows(x,y):
    if x.shape!=y.shape:raise ValueError('Identical full spatial shapes required')
    return torch.stack([(a.double()-b.double()).square().flatten(1).mean(1) for a,b in zip(x,y)])


def spread(x,y,indices=None):
    if x.shape!=y.shape or x.ndim<3 or x.shape[1]!=64:raise ValueError('Require H,64,features paired arrays')
    ids=list(range(64)) if indices is None else list(indices)
    if not ids or ids[0]!=0 or len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('Unique central-first subset required')
    drift=mse_rows(x[:,ids],y[:,ids]);action=mse_rows(x[:,ids[1:]],x[:,:1].expand_as(x[:,ids[1:]])).mean(1)
    mean=drift.mean(1)
    return dict(candidate_indices=ids,mse_by_horizon_candidate=drift.tolist(),mean_mse_by_horizon=mean.tolist(),RMS_by_horizon=mean.sqrt().tolist(),
        clean_alternative_minus_central_mse_by_horizon=action.tolist(),denominator_tiny=(action<=1e-20).tolist(),
        RMS_over_native_action_RMS=[None if d<=1e-20 else float((n/d).sqrt()) for n,d in zip(mean,action)],
        normalization=f'RMS ratio: mean nuisance MSE over {len(ids)} candidates / mean central-relative clean MSE over {len(ids)-1} alternatives; not variance or MSE ratio')


def checked_file(root,receipt,name):
    rows=[r for r in receipt['outputs'] if r['path']==name]
    if len(rows)!=1 or sha(root/name)!=rows[0]['sha256']:raise ValueError('Source SHA before load '+name)
    return rows[0]


def prepare(args):
    """Small extraction only; original verified tensors remain on both original hosts."""
    torch.set_num_threads(1);done=json.loads((args.source/'DONE.json').read_text())
    if not done['complete'] or done.get('held_access',True):raise ValueError('Original complete nonheld study required')
    args.output.mkdir(parents=True,exist_ok=False);outputs=[]
    for episode in range(4):
        name=f'near-dev-{episode:03d}.pt';row=checked_file(args.source,done,name)
        original=torch.load(args.source/name,map_location='cpu',weights_only=False)
        if original['summary']['episode']!=episode or not original['summary']['complete']:raise ValueError('Original identity')
        regenerated,_=noisy_display(original['raw_clean'],SEED+episode)
        exact(regenerated,original['raw_noisy'],'Exact original fixed noise before extraction')
        data={k:original[k] for k in ('raw_clean','raw_noisy','raw_proprio','encoded_clean','encoded_noisy','goal_encoded','raw_actions','normalized_actions','summary','costs')}
        data.update(source_row=row,source_root=str(args.source),original9_source_sha256=original['summary']['source_sha256'],
                    actual_visual_sha256=tensor_sha(original['actual']['visual']))
        path=args.output/f'stimulus-{episode:03d}.pt';torch.save(data,path)
        outputs.append(dict(path=path.name,episode=episode,split='development_external',sha256=sha(path),bytes=path.stat().st_size))
    write(args.output/'DONE.json',dict(complete=True,outputs=outputs,source_done_sha256=sha(args.source/'DONE.json'),
        same_noise_regeneration_exact=True,held_access=False,model_calls=0,simulator_calls=0))
    print(json.dumps(dict(event='tiny_original_stimuli_ready',bytes=sum(r['bytes'] for r in outputs))),flush=True)


def validate_inputs(bank,stimulus,episode):
    if bank['row']['split']!='development_external' or len(bank['candidates'])!=64 or not bank['original9_actions_goal_physics_pixels_exact']:raise ValueError('Only verified existing64 DEV bank')
    if bank['row']['source_id']!=episode or stimulus['summary']['episode']!=episode:raise ValueError('State identity')
    if bank['original9_source']['sha256']!=stimulus['original9_source_sha256']:raise ValueError('Original9 source identity link')
    for k in ('raw_actions','normalized_actions'):exact(bank[k][:9],stimulus[k],'First9 original '+k)
    for k in ('visual','proprio'):exact(bank['goal_encoded'][k],stimulus['goal_encoded'][k],'Original fullprecision goal '+k)
    actual9=torch.stack([c['actual_visual'] for c in bank['candidates'][:9]],1).float()
    if tensor_sha(actual9)!=stimulus['actual_visual_sha256']:raise ValueError('First9 actual encoded physics futures')
    if bank['context']['visual'].shape[:2]!=(1,1):raise ValueError('No inferred missing RGB history')
    states=np.stack([c['states'] for c in bank['candidates']])
    if not np.array_equal(states[:,0],np.repeat(states[:1,0],64,axis=0)):raise ValueError('All64 same actual start')
    props=torch.tensor(states[0,0,[0,1,5,6]],dtype=torch.float32)[None,None]
    exact(props,stimulus['raw_proprio'],'Original physical proprio')
    # Saved SHA-verified pixels, not a fresh host-dependent CPU draw, are the stimulus.
    generated,_=noisy_display(stimulus['raw_clean'],SEED+episode)
    metadata=dict(stimulus['summary']['noise'])
    difference=generated-stimulus['raw_noisy']
    metadata['cross_host_regeneration_diagnostic']=dict(maxabs_raw255=float(difference.abs().max()),
        MSE_raw255=float(difference.double().square().mean()),unequal_pixel_channels=int((difference!=0).sum()),
        torch_version=torch.__version__,used_for_experiment=False,
        actual_input='Original9 saved noisy pixels, SHA-verified before load; never regenerated on this host')
    zero,_=noisy_display(stimulus['raw_clean'],SEED+episode,0.);exact(zero,stimulus['raw_clean'],'Zero raw pixels')
    return states,metadata


@torch.no_grad()
def run(args):
    started=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    try:
        if args.episode not in range(4):raise ValueError('Only four already-seen states')
        done=json.loads((args.bank/'DONE.json').read_text());sdone=json.loads((args.stimuli/'DONE.json').read_text())
        if not done['complete'] or done.get('held_access',True) or not sdone['complete'] or sdone.get('held_access',True):raise ValueError('Sealed receipt guard')
        brow=checked_file(args.bank,done,f'expanded-dev-{args.episode:03d}.pt');srow=checked_file(args.stimuli,sdone,f'stimulus-{args.episode:03d}.pt')
        if brow['split']!='development_external' or srow['split']!='development_external':raise ValueError('Read split before load')
        bank=torch.load(args.bank/brow['path'],map_location='cpu',weights_only=False)
        stimulus=torch.load(args.stimuli/srow['path'],map_location='cpu',weights_only=False)
        states,noise=validate_inputs(bank,stimulus,args.episode)
        if sha(args.checkpoint)!=CHECKPOINT or sha(Path('/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'))!=DINO_SHA:raise ValueError('Frozen predictor/externalDINO SHA')
        if shutil.disk_usage(args.output).free<1_500_000_000:raise ValueError('Preserve1GB disk plus outputs')
        protocol=dict(episode=args.episode,states_total=4,candidates=64,sigma=SIGMA,seed=SEED+args.episode,
            frozen_inputs=dict(bank_path=str(args.bank/brow['path']),bank_sha256=brow['sha256'],stimulus_path=str(args.stimuli/srow['path']),stimulus_sha256=srow['sha256'],
                raw_clean_sha256=tensor_sha(stimulus['raw_clean']),raw_noisy_sha256=tensor_sha(stimulus['raw_noisy']),raw_actions_sha256=tensor_sha(bank['raw_actions']),normalized_actions_sha256=tensor_sha(bank['normalized_actions'])),
            protocol='Fresh-clean paired baseline, exact zero encodings/H6, cached64/fresh64 choice exact before any noisy model call; old9 forecasts/costs cross-batch descriptive only',
            primary='Same-input sensitivity / native central-relative action RMS, full64 and original9-subset samebatch',
            fixed_noise='Authoritative SHA-verified saved original sigma.01 noisy pixels, never replaced by local RNG; clipped display[0,1], no uint8 requantization; substantial clipping means delivered noise is not mean-zero',
            prior_failed_version='v1 stopped before model construction on unnecessary cross-host RNG regeneration guard; saved stimulus hash/action/goal/actual-future guards passed. Preserved immutable; v2 uses exactly saved original pixels by parent approval.',
            unit='Four reused development states, not256 independent samples; broader radius AND rank changes denominator',
            limitations='Not IR/SR replication, certification, steering, or explanation of original clean rollout errors; no different-state separation or new noise search',
            architecture_audit='Shared frozen DINO, no runtime EMAteacher, temporal blockcausal attention not spatial masked-target training; RGB decoder excluded; historical50epoch configuration not cryptographically bound',
            checkpoint_sha256=CHECKPOINT,external_DINO_sha256=DINO_SHA,script_sha256=sha(__file__),max_process_seconds=args.max_seconds,
            held_access=False,model_training_steps=0,cem_calls=0,simulator_calls=0,process_pid=os.getpid())
        write(args.output/'protocol.json',protocol)
        from model_loader import load_headless
        from tensordict import TensorDict
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
        from diagnose_pusht_goal_coverage import coverage,native_helper
        torch.set_num_threads(2);torch.manual_seed(90505)
        wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
        wm.eval().requires_grad_(False);before=parameter_sha(wm)
        def encode(raw):return wm.encode({'visual':raw,'proprio':stimulus['raw_proprio']},act=False)
        clean_encoded=encode(stimulus['raw_clean']);zero_encoded=encode(stimulus['raw_clean'].clone())
        for k in ('visual','proprio'):exact(clean_encoded[k],zero_encoded[k],'Zero noise encoder '+k)
        exact(clean_encoded['proprio'].cpu(),bank['context']['proprio'],'Native true proprio')
        normalized=prep.normalize_actions(bank['raw_actions'].reshape(64,6,5,2)).reshape(64,6,10)
        exact(normalized,bank['normalized_actions'],'Frozen normalized64 plans')
        actions=normalized.transpose(0,1).contiguous().cuda();frozen_actions=actions.clone()
        goal=TensorDict(bank['goal_encoded'],batch_size=[]).cuda();frozen_goal=goal.clone()
        objective=ReprTargetDistMPCObjective({},goal,sum_all_diffs=False,alpha=.1)
        def unroll(encoded):return wm.unroll(TensorDict({k:v.cuda().repeat_interleave(64,dim=0) for k,v in encoded.items()},batch_size=[]),act_suffix=actions)
        cached=unroll(bank['context']);clean=unroll(clean_encoded);zero=unroll(zero_encoded)
        for k in ('visual','proprio'):exact(clean[k],zero[k],'Exact samebatch zero H6 '+k)
        del zero
        ccost=objective(cached,actions,keepdims=True)[1:].cpu();cost=objective(clean,actions,keepdims=True)[1:].cpu()
        old_cost=torch.cat([c['native_cost_by_horizon'] for c in bank['candidates']],dim=1)[1:]
        choices=[int(x[-1].argmin()) for x in (old_cost,ccost,cost)]
        if len(set(choices))!=1:raise ValueError('Cached64/fresh64 native choice before noisy encoding')
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        write(args.output/'CANARY.json',dict(complete=True,seconds=elapsed,clean_choices=choices,exact_zero=True,
            projected_seconds=elapsed+20.,max_seconds=args.max_seconds,first9_source_physics_actions_noise_exact=True))
        print(json.dumps(dict(event='noise64_clean_canary',episode=args.episode,seconds=elapsed,choices=choices)),flush=True)
        if elapsed+10>args.max_seconds:raise ValueError('Insufficient remaining frozen process budget')
        noisy_encoded=encode(stimulus['raw_noisy']);exact(noisy_encoded['proprio'],clean_encoded['proprio'],'Unchanged proprio')
        noisy=unroll(noisy_encoded)
        predictions={n:cpu({k:v[k][1:] for k in ('visual','proprio')}) for n,v in [('clean',clean),('noisy',noisy)]}
        costs={'clean':cost,'noisy':objective(noisy,actions,keepdims=True)[1:].cpu()}
        floors={k:dict(old_context_maxabs=float((clean_encoded[k].cpu()-bank['context'][k]).abs().max()),
            original9_fresh_encoder_maxabs=float((clean_encoded[k].cpu()-stimulus['encoded_clean'][k]).abs().max()),
            original9_noisy_encoder_maxabs=float((noisy_encoded[k].cpu()-stimulus['encoded_noisy'][k]).abs().max()),
            cached_context_vs_fresh_H6_maxabs=float((cached[k][-1]-clean[k][-1]).abs().max())) for k in ('visual','proprio')}
        for n in ('clean','noisy'):
            floors[n+'_first9_cost_maxabs']=float((costs[n][:,:9]-stimulus['costs'][n]).abs().max())
            floors[n+'_first9_choice']=int(costs[n][-1,:9].argmin())
        del cached,clean,noisy
        actual={'visual':torch.stack([c['actual_visual'] for c in bank['candidates']],1).float()}
        props=torch.tensor(states[:,::5][:,:,[0,1,5,6]],dtype=torch.float32)
        actual['proprio']=wm.model.encode_proprio(prep.normalize_proprios(props).cuda()).cpu()[:,1:].transpose(0,1).contiguous()
        gs=np.asarray(bank['goal_state']);future=states[:,5::5]
        cover=np.array([[coverage(s[2:5],gs[2:5]) for s in plan] for plan in future]).T
        xy=np.linalg.norm(future[:,:,:4]-gs[:4],axis=-1).T
        poly=native_helper(args.repo/'evals/simu_env_planning/envs/pusht_env/pusht_env.py');target=poly(gs[2:5])
        checked=np.array([poly(s[2:5]).intersection(target).area/target.area for s in future[:,-1]])
        if np.max(np.abs(checked-cover[-1]))>1e-12:raise ValueError('Native requested-goal polygon parity')
        rows=[];clean_choice=choices[-1]
        for n,prediction in predictions.items():
            chosen=int(costs[n][-1].argmin());errors={k:mse_rows(prediction[k],actual[k]) for k in actual}
            rows.append(dict(arm=n,choice=chosen,cost_by_horizon=costs[n].tolist(),rank_by_horizon=rank_indices(costs[n]).tolist(),
                actual_MSE_by_horizon_candidate={k:v.tolist() for k,v in errors.items()},actual_mean_MSE_by_horizon={k:v.mean(1).tolist() for k,v in errors.items()},
                original9_mean_MSE_by_horizon={k:v[:,:9].mean(1).tolist() for k,v in errors.items()},
                selected_requested_goal_coverage=float(cover[-1,chosen]),selected_goal_coverage_delta_vs_clean=float(cover[-1,chosen]-cover[-1,clean_choice]),
                selected_combined_agent_block_XY_distance_px=float(xy[-1,chosen]),selected_XY_delta_vs_clean_px=float(xy[-1,chosen]-xy[-1,clean_choice])))
        sensitivity={k:dict(full64=spread(predictions['clean'][k],predictions['noisy'][k]),original9_same_batch=spread(predictions['clean'][k],predictions['noisy'][k],range(9))) for k in actual}
        encoder_drift={k:dict(MSE=float((clean_encoded[k]-noisy_encoded[k]).double().square().mean()),L2=float((clean_encoded[k]-noisy_encoded[k]).double().norm())) for k in ('visual','proprio')}
        exact(actions,frozen_actions,'All64 external plans unchanged')
        for k in ('visual','proprio'):exact(goal[k],frozen_goal[k],'Fullprecision objective unchanged')
        if parameter_sha(wm)!=before:raise ValueError('Frozen parameter hash changed')
        torch.cuda.synchronize();model_seconds=time.monotonic()-started
        report=dict(complete=True,episode=args.episode,rows=rows,spread=sensitivity,noise=noise,encoder_drift=encoder_drift,
            cross_batch_context_floors=floors,original9_preservation_exact=True,zero_encoder_H6_exact=True,parameters_sha256=before,
            goal_coverage_by_horizon_candidate=cover.tolist(),combined_XY_distance_by_horizon_candidate=xy.tolist(),candidate_metadata=bank['metadata'],
            protocol=protocol,model_and_analysis_seconds=model_seconds,raw_stimulus_pointer=str(args.stimuli/srow['path']),source_bank_pointer=str(args.bank/brow['path']))
        torch.save(dict(complete=True,report=report,predictions=predictions,costs=costs,encoded_clean=cpu(clean_encoded),encoded_noisy=cpu(noisy_encoded),
            raw_actions=bank['raw_actions'],normalized_actions=normalized,goal_encoded=bank['goal_encoded'],states=states),args.output/'result.pt')
        report['seconds']=time.monotonic()-started
        if report['seconds']>args.max_seconds:raise ValueError('Process budget exceeded')
        write(args.output/'summary.json',report)
        files=[args.output/n for n in ('result.pt','summary.json','CANARY.json','protocol.json')]
        outputs=[dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size) for f in files]
        write(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-started,episode=args.episode,
            parameters_sha256=before,provenance=provenance,held_access=False,training_steps=0,cem_calls=0,simulator_calls=0))
        print(json.dumps(dict(event='noise64_DONE',episode=args.episode,seconds=time.monotonic()-started,choices={r['arm']:r['choice'] for r in rows})),flush=True)
    except Exception:
        write(args.output/'FAILED.json',dict(complete=False,seconds=time.monotonic()-started,exception=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    for name in ('source','output'):prep.add_argument('--'+name,type=Path,required=True)
    job=sub.add_parser('run')
    for name in ('bank','stimuli','repo','checkpoint','output'):job.add_argument('--'+name,type=Path,required=True)
    job.add_argument('--episode',type=int,required=True);job.add_argument('--max-seconds',type=float,default=60.)
    args=p.parse_args();prepare(args) if args.mode=='prepare' else run(args)


if __name__=='__main__':main()
