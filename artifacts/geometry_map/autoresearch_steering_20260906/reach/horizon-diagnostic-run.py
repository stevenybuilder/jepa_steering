"""Extend frozen selected arms to actual H6; exploratory diagnostic, no promotion."""
from pathlib import Path
import argparse,json,hashlib,time,sys,os,traceback
import numpy as np
import torch
from pilot_fullgrid import Correction,source
from extract_fit import sha

def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def verified(p,root,entries):
 assert p.name in entries and sha(p)==entries[p.name]['sha256'],str(p)
 return torch.load(p,map_location='cpu',weights_only=False)

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
 a.output.mkdir(parents=True,exist_ok=False);began=time.monotonic();rows=[]
 manifest=json.load(a.protocol.open());assert sha(a.correction)==manifest['frozen_correction_sha256'];save(a.output/'protocol.json',dict(manifest,pid=os.getpid(),worker=a.worker,actual_episodes=a.episodes,runner_sha256=sha(__file__),start_epoch=time.time()))
 try:
  wm,pre,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False);before=parameters_sha(wm);cfg=setup_cfg(a.config,a.output,wm);corr=Correction(a.correction,'cuda')
  prior=a.parent/'pilot-fullgrid-v1';receipt=json.load((prior/'DONE.json').open());assert receipt['complete'];entries={x['path']:x for x in receipt['outputs']}
  assert sha(prior/'summary.json')==entries['summary.json']['sha256'];previous=json.load((prior/'summary.json').open());assert previous['parameters_unchanged'];priorrows={r['episode']:r for r in previous['rows']}
  for e in a.episodes:
   assert 66<=e<=73
   assert time.monotonic()-began<360,'Worker budget expired'
   inp,ref=source(a.root,e,a.worker);expected=json.load((a.parent/'test-inputs.json').open());assert sha(inp)==expected[inp.name]and sha(ref)==expected[ref.name]
   prepared=torch.load(inp,map_location='cpu',weights_only=False);t=torch.load(ref,map_location='cpu',weights_only=False)
   bank=verified(prior/f'episode-{e:03d}-bank.pt',prior,entries);assert bank['selected']==priorrows[e]['selected'];selected=bank['selected'];ids=list(dict.fromkeys(selected.values()));assert len(ids)<=4
   actions=bank['actions'][:,ids].cuda();assert actions.shape==(6,len(ids),20)
   initial=dict(episode=e,environment_seed=2026090500+e,planner_seed=90500+e,prepared=prepared,prepared_sha256=sha(inp),replans=[dict(observation_proprio=prepared['initial_proprio'])])
   env,agent,z,checks=reconstruct_prepared(cfg,wm,pre,initial);same_physics(physics_snapshot(env),t['initial_physics'],'frozen selected-arm initial')
   for k in ('visual','proprio'):exact(z[k],t['encoded_context'][k],'original initial encodedcontext '+k)
   native=wm.unroll(z.clone(),act_suffix=actions);nativecost=agent.objective(native,actions)
   oldcost=np.array(bank['costs']['native'])[ids];cost_error=float(np.max(np.abs(nativecost.cpu().numpy()-oldcost)));assert cost_error<2e-4
   correction={h:.5*corr.vector(z,native,actions,h)for h in range(1,7)};shams={h:corr.sham(correction[h])for h in range(1,7)}
   for h in range(1,7):
    for sl in [slice(0,384),slice(384,400)]:assert torch.allclose(correction[h][:,:,sl].flatten(1).norm(dim=1),shams[h][:,:,sl].flatten(1).norm(dim=1),rtol=1e-6,atol=1e-6)
   zero=nativecost+corr.delta_cost(native,agent.goal_state_enc,correction[6]*0,6,agent.objective.alpha);assert torch.equal(zero,nativecost)
   altered=native.clone()
   for k,sl in [('visual',slice(0,384)),('proprio',slice(384,400))]:altered[k][6]+=correction[6][:,:,sl].reshape_as(altered[k][6])
   direct=agent.objective(altered,actions);analytic=nativecost+corr.delta_cost(native,agent.goal_state_enc,correction[6],6,agent.objective.alpha);assert torch.allclose(direct,analytic,rtol=1e-5,atol=1e-6)
   goal={k:agent.goal_state_enc[k].cpu().clone()for k in ['visual','proprio']};close_env(env);by_index={};outputs=[]
   for j,idx in enumerate(ids):
    assert time.monotonic()-began<360,'Worker budget expired'
    name=next(n for n,i in selected.items()if i==idx);old=verified(prior/f'episode-{e:03d}-{name}-physical.pt',prior,entries)
    env,ag,z2,ch=reconstruct_prepared(cfg,wm,pre,initial);assert ch['fresh_goal_sha256']==checks['fresh_goal_sha256'];same_physics(physics_snapshot(env),t['initial_physics'],'selected-arm fresh physics')
    raw=pre.denormalize_actions(actions[:,j].cpu().reshape(30,4));exact(raw[:15],old['actions'],'first15 frozen actions')
    states=[];frames=[];infos=[];rewards=[];truths=[];metrics=[];base=env.proprio_env.unwrapped
    for step,action in enumerate(raw):
     obs,reward,done,info=env.step_multiple(action[None]);assert len(obs)==1 and not done[-1]
     states.append(np.asarray(info[0]['state']).copy());frames.append(obs[0].cpu());rr,ii=base.evaluate_state(base._get_obs(),action.numpy());infos.append({k:float(v)for k,v in ii.items()if np.asarray(v).shape==()});rewards.append(float(reward[0]))
     if (step+1)%5==0:
      h=(step+1)//5;enc=wm.encode(make_td(obs[0],info[0]).to(ag.device).unsqueeze(0),act=True)
      actualcost=float(ag.objective(enc,actions[:,j:j+1]).reshape(-1)[0]);m=dict(horizon=h,actual_native_goal_cost=actualcost)
      for k,sl in [('visual',slice(0,384)),('proprio',slice(384,400))]:
       pp=native[k][h,j];tt=enc[k].reshape_as(pp);delta=correction[h][j,:,sl].reshape_as(pp);sham=shams[h][j,:,sl].reshape_as(pp)
       m[k+'_mse']={arm:float((pp+dd-tt).square().mean())for arm,dd in [('native',delta*0),('edit',delta),('sham',sham)]}
      # Goal score for the same frozen plan at this horizon; channels use unchanged nativealpha.
      nscore=float(((native['visual'][h,j]-goal['visual'].to(native['visual']).reshape_as(native['visual'][h,j]))**2).mean()+ag.objective.alpha*((native['proprio'][h,j]-goal['proprio'].to(native['proprio']).reshape_as(native['proprio'][h,j]))**2).mean())
      m['predicted_native_goal_cost']=nscore;m['predicted_corrected_goal_cost']=nscore+float(corr.delta_cost(native,ag.goal_state_enc,correction[h],h,ag.objective.alpha)[j]);m['predicted_sham_goal_cost']=nscore+float(corr.delta_cost(native,ag.goal_state_enc,shams[h],h,ag.objective.alpha)[j]);metrics.append(m);truths.append({k:enc[k].cpu().clone()for k in ['visual','proprio']})
    assert len(states)==30 and len(metrics)==6
    exact(np.stack(states[:15]),old['states'],'first15 states matchpriorpilot');exact(torch.stack(frames[:15]),old['frames'],'first15 pixels matchpriorpilot')
    physical=hand_goal_metrics(prepared['initial_proprio'].reshape(-1)[:3].numpy(),states);physical.update(candidate_index=idx,native_ever_success=any(x['success']>0 for x in infos),native_final_success=infos[-1]['success']>0,native_reward_sum=float(sum(rewards)),raw_steps=30)
    by_index[idx]=dict(physical=physical,forecast=metrics,actual_h6_native_goal_cost=metrics[-1]['actual_native_goal_cost'],first15_prior_replay_exact=True)
    full=dict(episode=e,representative_arm=name,candidate_index=idx,actions=raw,states=np.stack(states),frames=torch.stack(frames),native_infos=infos,rewards=rewards,actual_encoded_h1_h6=truths,native_predictions={k:native[k][:,j].cpu()for k in ['visual','proprio']},correction_h1_h6=torch.stack([correction[h][j].cpu()for h in range(1,7)]),goal=goal,metrics=by_index[idx]);torch.save(full,a.output/f'episode-{e:03d}-candidate-{idx:03d}.pt');close_env(env)
   arms={name:by_index[idx]for name,idx in selected.items()};oracle_idx=min(ids,key=lambda idx:by_index[idx]['actual_h6_native_goal_cost']);physicalbest_idx=max(ids,key=lambda idx:by_index[idx]['physical']['hand_goal_progress_m'])
   row=dict(episode=e,selected=selected,unique_physical_plan_count=len(ids),arms=arms,oracle_scope='Actual H6 native objective among these <=4 frozen selected arms only; not all301bank candidates',selected_arm_actual_objective_oracle_index=oracle_idx,selected_arm_best_physical_progress_index=physicalbest_idx,actual_objective_oracle_physical_regret_m=by_index[physicalbest_idx]['physical']['hand_goal_progress_m']-by_index[oracle_idx]['physical']['hand_goal_progress_m'],progress_change_vs_native_m=arms['edit']['physical']['hand_goal_progress_m']-arms['native_planner']['physical']['hand_goal_progress_m'],progress_change_vs_bank_m=arms['edit']['physical']['hand_goal_progress_m']-arms['native_bank']['physical']['hand_goal_progress_m'],progress_change_vs_sham_m=arms['edit']['physical']['hand_goal_progress_m']-arms['sham']['physical']['hand_goal_progress_m'],canaries=dict(first15_all_physical_replays_exact=True,exact_zero=True,sham_visual_proprio_norm_match=True,analytic_cost_maxabs=float((direct-analytic).abs().max()),native_cost_maxabs=cost_error))
   rows.append(row);save(a.output/f'episode-{e:03d}.json',row);save(a.output/'progress.json',dict(complete=False,rows=rows,seconds=time.monotonic()-began));print(json.dumps(dict(event='episode_complete',episode=e,unique_plans=len(ids),h6_progress_vs_native_m=row['progress_change_vs_native_m'],h6_progress_vs_sham_m=row['progress_change_vs_sham_m'],oracle_regret=row['actual_objective_oracle_physical_regret_m'],seconds=time.monotonic()-began)),flush=True)
  assert parameters_sha(wm)==before
  save(a.output/'summary.json',dict(complete=True,rows=rows,seconds=time.monotonic()-began,parameters_unchanged=True,provenance=provenance,held_access=False,previous_gate_remains_failed=True,no_promotion=True))
  files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size)for p in sorted(a.output.iterdir())if p.is_file()];save(a.output/'DONE.json',dict(complete=True,outputs=files,seconds=time.monotonic()-began));print(json.dumps(dict(event='complete',seconds=time.monotonic()-began)),flush=True)
 except Exception:
  save(a.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc(),seconds=time.monotonic()-began));raise

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','parent','repo','config','correction','protocol','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--worker',type=int,required=True);p.add_argument('--episodes',type=int,nargs='+',required=True);run(p.parse_args())
