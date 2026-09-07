#!/usr/bin/env python3
"""Native P3 edits evaluated on frozen same-action physical consequences."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from capture_pusht_calibration import first_tensor,parameter_sha
from capture_horizon_coordinates import exact,pool_visual
from collect_pusht_bank import config
from complete_cached_geometry import sha256,write_json
from fit_action_consequence import ranking

def settings():
    rows=[{"name":"unsteered","basis":None,"dose":0.,"gate":"all","sign":1},
          {"name":"identity","basis":"readable","dose":0.,"gate":"all","sign":1}]
    for basis in ("readable","action_effect","random"):
        for dose in (.001,.005):
            for gate in ("first","all"):
                rows.append({"name":f"{basis}_{dose}_{gate}","basis":basis,"dose":dose,"gate":gate,"sign":1})
    for basis in ("readable","action_effect"):
        rows.append({"name":basis+"_negative_0.001_all","basis":basis,"dose":.001,"gate":"all","sign":-1})
    return rows

def normalized_delta(error,basis,response,radius):
    coefficients=error@torch.linalg.pinv(basis.T@response,rtol=1e-3)
    delta=coefficients@basis.T
    norm=delta.norm(dim=-1,keepdim=True)
    return delta/norm.clamp_min(1e-12)*radius

class CorrectP3:
    def __init__(self,block,calibration,setting,device):
        self.block=block;self.setting=setting;self.calls=0;self.norms=[];self.support=[]
        self.c={key:torch.as_tensor(value,dtype=torch.float32,device=device) for key,value in calibration.items()}
        self.pinv=[]
        if setting['basis'] is not None:
            self.b=self.c['basis_'+setting['basis']]
            self.pinv=[torch.linalg.pinv(self.b.T@self.c['response_raw'][h],rtol=1e-3) for h in range(6)]
    def __enter__(self):self.handle=self.block.register_forward_hook(self.hook);return self
    def hook(self,_module,_inputs,out):
        raw=first_tensor(out);h=self.calls;self.calls+=1
        if h>=6 or raw.ndim!=3 or raw.shape[-1]!=400 or raw.shape[1]<256:raise ValueError("Wrong native P3 horizon/layout")
        pooled=raw[:,-256:].float().mean(1)
        if self.setting['dose']==0 or (self.setting['gate']=='first' and h>0):
            self.norms.append(0.);self.support.append(pooled.cpu());return out
        e=((pooled-self.c['residual_mean'][h])/self.c['residual_scale'][h])@self.c['residual_coef'][h].T+self.c['residual_intercept'][h]
        delta=(e@self.pinv[h])@self.b.T
        radius=self.setting['dose']*self.c['p3_norm_median']
        delta=self.setting['sign']*delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)*radius
        if not torch.isfinite(delta).all():raise ValueError("Nonfinite runtime residual correction")
        changed=raw.clone();changed[:,-256:]=changed[:,-256:]+delta[:,None].to(raw.dtype)
        realized=(changed[:,-256:].float()-raw[:,-256:].float()).mean(1)
        self.norms.append(float(realized.norm(dim=-1).mean()));self.support.append(pooled.cpu())
        if isinstance(out,tuple):return (changed,*out[1:])
        if isinstance(out,list):return [changed,*out[1:]]
        return changed
    def __exit__(self,*_):self.handle.remove()

def observer(calibration,value):
    return ((value-calibration['observer_mean'])/calibration['observer_scale'])@calibration['observer_coef'].T+calibration['observer_intercept']

@torch.no_grad()
def run_source(args,row,wm,preprocessor,cfg,calibration):
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from tensordict import TensorDict
    path=args.bank/row['path']
    if sha256(path)!=row['sha256']:raise ValueError("Frozen bank hash mismatch before loading")
    bank=torch.load(path,map_location='cpu',weights_only=False)
    if bank['row']['split'] not in ('fit','development_validation','development_external'):raise ValueError("Held bank forbidden")
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=preprocessor)
    agent.set_goal(TensorDict(bank['raw_goal'],batch_size=[]))
    for key in ('visual','proprio'):exact(agent.goal_state_enc[key].float().cpu(),bank['goal_encoded'][key].float(),"same fixed raw goal encoding "+key)
    context=TensorDict(bank['context'],batch_size=[]).to(agent.device)
    conditions=[];base_predictions=[];arrays={};start=time.monotonic()
    true_q=np.stack([np.column_stack([r['states'][5::5,:4],np.sin(r['states'][5::5,4]),np.cos(r['states'][5::5,4])]) for r in bank['candidates']])
    real_cost=np.stack([np.asarray(r['actual_encoded_native_cost_by_horizon']) for r in bank['candidates']])
    physical_xy=np.stack([r['metrics']['goal_xy_distance'] for r in bank['candidates']])
    for setting in settings():
        forecasts=[];pooled=[];mse=[];cost=[];norms=[];surrogate_agreement=[];visual_cost=[];proprio_cost=[]
        for candidate,truth in enumerate(bank['candidates']):
            plan=bank['normalized_actions'][candidate,:,None].to(agent.device)
            if setting['basis'] is None:
                output=wm.unroll(context.clone(),act_suffix=plan);norm=[0.]*6
            else:
                with CorrectP3(wm.model.predictor.predictor_blocks[3],calibration,setting,agent.device) as hook:
                    output=wm.unroll(context.clone(),act_suffix=plan)
                if hook.calls!=6:raise ValueError("Wrong native hook callcount")
                norm=hook.norms
            predicted=output['visual'][1:,0].float().cpu()
            if setting['name'] in ('unsteered','identity'):
                exact(predicted,truth['predicted_visual'],"exact frozen action-bank baseline "+setting['name'])
                exact(output['proprio'][1:,0].float().cpu(),truth['predicted_proprio'],"exact native proprio baseline")
            forecasts.append(predicted);pooled.append(pool_visual(predicted).numpy());norms.append(norm)
            mse.append((predicted-truth['actual_visual']).square().reshape(6,-1).mean(1).numpy())
            cost.append(agent.objective(output,plan,keepdims=True).cpu().numpy())
            visual_cost.append((output['visual']-agent.goal_state_enc['visual']).float().square().reshape(7,-1).mean(1).cpu().numpy())
            proprio_cost.append((output['proprio']-agent.goal_state_enc['proprio']).float().square().reshape(7,-1).mean(1).cpu().numpy())
        pooled=np.stack(pooled);mse=np.stack(mse);cost=np.stack(cost);norms=np.asarray(norms)
        if setting['name']=='unsteered':base_predictions=forecasts;base_pool=pooled.copy();base_mse=mse.copy()
        delta_pool=pooled-base_pool
        decoded=observer(calibration,pooled);observed=observer(calibration,np.stack([r['actual_visual_pooled'].numpy() for r in bank['candidates']]))
        info={**setting,'forecast_visual_mse':float(mse.mean()),'visual_mse_delta_vs_unsteered':float((mse-base_mse).mean()),
            'horizon_visual_mse':mse.mean(0).tolist(),'horizon_latent_pooled_change':np.linalg.norm(delta_pool,axis=-1).mean(0).tolist(),
            'delivered_pooled_norm_by_horizon':norms.mean(0).tolist(),'raw_patch_delta_norm_factor':16,
            'decoded_forecast_rmse_px':float(np.sqrt(np.square(decoded[...,:4]-true_q[...,:4]).mean())),
            'actual_encoded_observer_rmse_px':float(np.sqrt(np.square(observed[...,:4]-true_q[...,:4]).mean())),
            'ranking':ranking(cost[:,-1].reshape(-1),real_cost[:,-1].reshape(-1),physical_xy[:,-1]),
            'native_cost_delta_by_candidate':(cost[:,-1].reshape(-1)-np.stack([np.asarray(r['native_cost_by_horizon']) for r in bank['candidates']])[:,-1].reshape(-1)).tolist()}
        conditions.append(info)
        name=setting['name'];arrays[name+'_predicted_pooled']=pooled;arrays[name+'_visual_mse']=mse;arrays[name+'_native_cost']=cost
        arrays[name+'_visual_goal_cost']=np.stack(visual_cost);arrays[name+'_proprio_goal_cost']=np.stack(proprio_cost);arrays[name+'_delivered_norm']=norms
        print(json.dumps({'event':'action_consequence_edit_complete','key':row['key'],'arm':name,'mse_delta':info['visual_mse_delta_vs_unsteered']}),flush=True)
    result={'complete':True,'key':row['key'],'split':row['split'],'conditions':conditions,'seconds':time.monotonic()-start,
        'source_bank_sha256':row['sha256'],'calibration_sha256':sha256(args.calibration/'calibration.npz'),
        'runtime_edit_inputs':['currentP3','imaginedhorizon','frozenTRAINcalibration'], 'runtime_future_labels_supplied':False,
        'intervention_type':'modelactivationedit with fixedaction bank, followed bynativecost actionselection; no newphysicalrolloutneeded forsameactions',
        'native_goal_objective':{'alpha':float(agent.objective.alpha),'sum_all_diffs':bool(agent.objective.sum_all_diffs)},
        'physical_consequence_status':'exact paired physicaltrajectories cached independently beforeintervention; truefutureonlyoffline scoring'}
    out=args.output/(row['key']+'.json');write_json(out,result)
    np.savez_compressed(out.with_suffix('.npz'),**arrays,actual_encoded_cost=real_cost,actual_physical_xy_distance=physical_xy,actual_q_next=true_q)
    return [{'path':p.name,'sha256':sha256(p),'bytes':p.stat().st_size,'key':row['key'],'split':row['split']} for p in (out,out.with_suffix('.npz'))]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bank','calibration','repo','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--keys',nargs='*');args=p.parse_args()
    receipt=json.loads((args.bank/'DONE.json').read_text());frozen=json.loads((args.calibration/'FROZEN.json').read_text())
    if not receipt['complete'] or not frozen['complete'] or sha256(args.calibration/'calibration.npz')!=frozen['calibration_sha256']:raise ValueError("Incomplete/unverified calibration orbank")
    rows=[r for r in receipt['outputs'] if r['path'].endswith('.pt') and (not args.keys or r['key'] in args.keys)]
    if any(r['split'] not in ('fit','development_validation','development_external') for r in rows):raise ValueError("Forbidden heldrows beforeload")
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',{'rows':rows,'settings':settings(),'calibration_frozen_sha256':sha256(args.calibration/'FROZEN.json'),
        'script_sha256':sha256(__file__),'same_rank_norm_site_capacity':True,'note':'Surrogateoutputresponse istrainingregression, notassumedcausal; actualnativeforecast error/ranking ismeasured'})
    from model_loader import load_headless
    torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(args.repo,args.output)
    wm,preprocessor,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);calibration=dict(np.load(args.calibration/'calibration.npz',allow_pickle=False));start=time.monotonic();outputs=[]
    for row in rows:
        if time.monotonic()-start>3500:raise RuntimeError("Bounded intervention stage exceeded")
        outputs.extend(run_source(args,row,wm,preprocessor,cfg,calibration))
    if parameter_sha(wm)!=before:raise ValueError("Frozen nativeparameters changed")
    write_json(args.output/'DONE.json',{'complete':True,'outputs':outputs,'seconds':time.monotonic()-start,'parameter_sha256':before,'provenance':provenance,
        'confirmation_or_heldstudy_accessed':False,'cem_calls':0})

if __name__=='__main__':main()
