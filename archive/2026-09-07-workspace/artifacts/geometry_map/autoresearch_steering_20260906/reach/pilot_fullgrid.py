"""Frozen DEVELOPMENT output-correction pilot; no held data or online label fit."""
from pathlib import Path
import argparse,json,hashlib,time,os,traceback,sys
import numpy as np
import torch
from extract_fit import features,pool,sha

class Correction:
    def __init__(self,path,device):
        d=np.load(path);self.d={k:torch.as_tensor(v,device=device)for k,v in d.items()}
    def vector(self,z,pred,actions,h):
        n=actions.shape[1];ctx=torch.cat([pool(z['visual'],384),pool(z['proprio'],16)]).expand(n,-1)
        pooled=torch.cat([pred['visual'][h].reshape(n,-1,384).mean(1),pred['proprio'][h].reshape(n,-1,16).mean(1)],1)
        a=actions[:h].permute(1,0,2).reshape(n,-1,4)
        f=torch.cat([ctx,pooled,a.mean(1),a[:,-1],a.sum(1)/15,torch.full((n,1),min(h,3)/3,device=actions.device)],1)
        zf=(f-self.d['mean'])/self.d['scale'];q=torch.cat([torch.ones((n,1),device=f.device),zf],1)
        return ((q@self.d['train'].T)@self.d['alpha']).reshape(n,256,400)
    def sham(self,delta):return delta[:,self.d['tokenperm']][:,:,self.d['perm']]*self.d['sign']
    def delta_cost(self,pred,goal,delta,h,alpha):
        n=len(delta);v=pred['visual'][h].reshape(n,256,384)-goal['visual'].reshape(1,256,384)
        p=pred['proprio'][h].reshape(n,256,16)-goal['proprio'].reshape(1,256,16)
        return (2*v*delta[:,:,:384]+delta[:,:,:384]**2).mean((1,2))+alpha*(2*p*delta[:,:,384:]+delta[:,:,384:]**2).mean((1,2))

def save(path,v):path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def source(root,e,w):
    base=root/'development-expansion-66-73-v1'
    matches=list(base.glob(f'*/inputs/input-{e:03d}.pt'))
    if w==49155754:return root/'autoresearch-steering-20260906/worker-49155754/input-v1'/f'input-{e:03d}.pt',root/'autoresearch-steering-20260906/worker-49155754/input-v1'/f'episode-{e:03d}-unsteered-replan-00.pt'
    assert len(matches)==1,(e,matches)
    p=matches[0];return p,p.parent.parent/'results-v2'/f'episode-{e:03d}-unsteered-replan-00.pt'

@torch.no_grad()
def run(a):
    sys.path.insert(0,str(a.repo));torch.set_num_threads(2)
    from model_loader import load_headless_metaworld
    from causal_planner_forks import setup_cfg,close_env
    from collect_reach_development_expansion_v1 import reconstruct_prepared
    from capture_horizon_coordinates import exact,same_physics
    from collect_on_policy_bank import physics_snapshot
    from capture_specificity_controls import parameters_sha
    from run_residual_coordinate_time import hand_goal_metrics
    from evals.simu_env_planning.planning.utils import make_td
    out=a.output;out.mkdir(parents=True,exist_ok=False);began=time.monotonic();rows=[];sources=[]
    save(out/'protocol.json',dict(json.load(open(a.protocol)),instance=a.worker,episodes=a.episodes,pid=os.getpid(),start_epoch=time.time(),correction_sha256=sha(a.correction),runner_sha256=sha(__file__)))
    try:
        wm,pre,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False);before=parameters_sha(wm)
        cfg=setup_cfg(a.config,out,wm);corr=Correction(a.correction,'cuda')
        for e in a.episodes:
            assert 66<=e<=73
            inp,ref=source(a.root,e,a.worker); expected=json.load(open(a.protocol.parent/'test-inputs.json')); assert sha(inp)==expected[inp.name] and sha(ref)==expected[ref.name]; prepared=torch.load(inp,map_location='cpu',weights_only=False); t=torch.load(ref,map_location='cpu',weights_only=False)
            assert prepared['episode']==e and t['episode']==e and t['replan']==0 and t['arm']=='unsteered'
            sources.extend([dict(path=str(p),sha256=sha(p))for p in (inp,ref)])
            bank=dict(episode=e,environment_seed=2026090500+e,planner_seed=90500+e,prepared=prepared,prepared_sha256=sha(inp),replans=[dict(observation_proprio=prepared['initial_proprio'])])
            env,agent,z,checks=reconstruct_prepared(cfg,wm,pre,bank)
            assert not agent.objective.sum_all_diffs
            same_physics(physics_snapshot(env),t['initial_physics'],'saved initial physics')
            for k in ('visual','proprio'):exact(z[k],t['encoded_context'][k],'saved encoded context '+k)
            pop=[x for x in t['candidate_iterations']if x['actions_normalized'].shape[1]==300][-1]
            actions=torch.cat([pop['actions_normalized'],t['selected_full_plan'][:,None]],1).cuda();costs={k:[]for k in ('native','edit','sham')};diag=[];canaries=[]
            for lo in range(0,301,16):
                assert time.monotonic()-began<1200,'Per-worker pilot budget reached'; aa=actions[:,lo:lo+16];pred=wm.unroll(z.clone(),act_suffix=aa);h=aa.shape[0];base=agent.objective(pred,aa);delta=.5*corr.vector(z,pred,aa,h);sham=corr.sham(delta)
                assert torch.allclose(delta[:,:,:384].flatten(1).norm(dim=1),sham[:,:,:384].flatten(1).norm(dim=1),atol=1e-6,rtol=1e-6)
                assert torch.allclose(delta[:,:,384:].flatten(1).norm(dim=1),sham[:,:,384:].flatten(1).norm(dim=1),atol=1e-6,rtol=1e-6)
                cedit=base+corr.delta_cost(pred,agent.goal_state_enc,delta,h,agent.objective.alpha);csham=base+corr.delta_cost(pred,agent.goal_state_enc,sham,h,agent.objective.alpha)
                if lo==0:
                    altered=pred.clone()
                    for k,d in [('visual',delta[:,:,:384]),('proprio',delta[:,:,384:])]:altered[k][h]+=d.reshape_as(altered[k][h])
                    direct=agent.objective(altered,aa);assert torch.allclose(direct,cedit,atol=1e-6,rtol=1e-5)
                    zero=base+corr.delta_cost(pred,agent.goal_state_enc,delta*0,h,agent.objective.alpha);assert torch.equal(zero,base)
                    canaries.append(dict(exact_zero_cost=True,analytic_output_correction_equivalence_maxabs=float((direct-cedit).abs().max()),separate_channel_norm_matched=True))
                for k,c in zip(costs,(base,cedit,csham)):costs[k].extend(c.cpu().tolist())
                if lo<=300<lo+len(aa[0]):
                    idx=300-lo
                    for truth in t['actual_executed_prefix_encodings']:
                        hh=truth['horizon'];d=.5*corr.vector(z,pred,aa,hh)[idx];ss=corr.sham(d[None])[0];dr=dict(horizon=hh)
                        for k,sl,dim in [('visual',slice(0,384),384),('proprio',slice(384,400),16)]:
                            pp=pred[k][hh,idx];target=truth[k].cuda().reshape_as(pp);err=pp-target
                            dd=d[:,sl].reshape_as(pp);sd=ss[:,sl].reshape_as(pp)
                            dr[k+'_mse']={arm:float(x.square().mean())for arm,x in [('native',err),('edit',err+dd),('sham',err+sd)]}
                        diag.append(dr)
            close_env(env)
            error=np.max(np.abs(np.array(costs['native'][:300])-pop['costs'].numpy()));assert error<2e-4,('native cost mismatch',error)
            selected={'native_planner':300,'native_bank':int(np.argmin(costs['native'])),'edit':int(np.argmin(costs['edit'])),'sham':int(np.argmin(costs['sham']))}
            physical={};cache={}
            for arm,idx in selected.items():
                if idx in cache:physical[arm]=dict(cache[idx],shared_identical_action_index=idx);continue
                env,ag,z2,ch=reconstruct_prepared(cfg,wm,pre,bank);same_physics(physics_snapshot(env),t['initial_physics'],'paired arm initial physics')
                assert ch['fresh_goal_sha256']==checks['fresh_goal_sha256']
                raw=pre.denormalize_actions(actions[:3,idx].cpu().reshape(-1,4));states=[];frames=[];infos=[];rews=[]
                baseenv=env.proprio_env.unwrapped
                for action in raw:
                    obs,reward,done,info=env.step_multiple(action[None]);states.append(np.asarray(info[0]['state']).copy());frames.append(obs[0].cpu());rr,ii=baseenv.evaluate_state(baseenv._get_obs(),action.numpy());infos.append({k:float(v)for k,v in ii.items()if np.asarray(v).shape==()});rews.append(float(reward[0]))
                assert len(states)==15
                if arm=='native_planner':
                    exact(np.stack(states),t['states'],'native selected15 physical states');exact(torch.stack(frames),t['frames'],'native selected15 physical frames');exact(raw,t['raw_commands_executed'],'native rawcommands')
                m=hand_goal_metrics(prepared['initial_proprio'].reshape(-1)[:3].numpy(),states);m.update(native_ever_success=any(x['success']>0 for x in infos),native_final_success=infos[-1]['success']>0,native_reward_sum=float(sum(rews)),candidate_index=idx)
                physical[arm]=m;cache[idx]=m
                torch.save(dict(episode=e,arm=arm,actions=raw,states=np.stack(states),frames=torch.stack(frames),native_infos=infos,rewards=rews,source_sha256=sha(ref)),out/f'episode-{e:03d}-{arm}-physical.pt');close_env(env)
            row=dict(episode=e,selected=selected,physical=physical,forecast_mse=diag,cost_identity_maxabs=float(error),canaries=canaries,progress_change_vs_native_m=physical['edit']['hand_goal_progress_m']-physical['native_planner']['hand_goal_progress_m'],progress_change_vs_sham_m=physical['edit']['hand_goal_progress_m']-physical['sham']['hand_goal_progress_m'],progress_change_vs_bank_m=physical['edit']['hand_goal_progress_m']-physical['native_bank']['hand_goal_progress_m'])
            save(out/f'episode-{e:03d}.json',row);torch.save(dict(actions=actions.cpu(),costs=costs,selected=selected),out/f'episode-{e:03d}-bank.pt');rows.append(row);save(out/'progress.json',dict(complete=False,rows=rows,seconds=time.monotonic()-began));print(json.dumps(dict(event='episode_complete',episode=e,selected=selected,native=row['progress_change_vs_native_m'],sham=row['progress_change_vs_sham_m'],seconds=time.monotonic()-began)),flush=True)
        assert parameters_sha(wm)==before
        save(out/'summary.json',dict(complete=True,rows=rows,sources=sources,seconds=time.monotonic()-began,parameters_unchanged=True,provenance=provenance,held_access=False))
        outputs=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size)for p in sorted(out.iterdir())if p.is_file()];save(out/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-began));print(json.dumps(dict(event='complete',seconds=time.monotonic()-began)),flush=True)
    except Exception:
        save(out/'FAILED.json',dict(complete=False,exception=traceback.format_exc(),seconds=time.monotonic()-began));raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','repo','config','correction','protocol','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--worker',type=int,required=True);p.add_argument('--episodes',type=int,nargs='+',required=True);run(p.parse_args())
