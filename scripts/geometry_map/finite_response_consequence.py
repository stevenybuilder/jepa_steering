#!/usr/bin/env python3
"""Measured first-horizon native P3 response versus training regression response."""
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

def correction(error,basis,response,radius):
    delta=(error@torch.linalg.pinv(response,rtol=1e-3))@basis.T
    return delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)*radius

class FirstDelta:
    def __init__(self,block,delta):self.block=block;self.delta=delta;self.calls=0;self.first=None
    def __enter__(self):self.handle=self.block.register_forward_hook(self.hook);return self
    def hook(self,_module,_inputs,out):
        raw=first_tensor(out);h=self.calls;self.calls+=1
        if h>0:return out
        self.first=raw[:,-256:].float().mean(1).detach().clone()
        if self.delta is None:return out
        changed=raw.clone();changed[:,-256:]+=self.delta[:,None].to(raw.dtype)
        self.realized=(changed[:,-256:].float()-raw[:,-256:].float()).mean(1)
        if isinstance(out,tuple):return (changed,*out[1:])
        if isinstance(out,list):return [changed,*out[1:]]
        return changed
    def __exit__(self,*_):self.handle.remove()

@torch.no_grad()
def run_source(args,row,wm,prep,cfg,c):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    path=args.bank/row['path']
    if sha256(path)!=row['sha256']:raise ValueError("Source bank SHA mismatch")
    bank=torch.load(path,map_location='cpu',weights_only=False)
    if bank['row']['split']!='development_validation':raise ValueError("Only frozen fourdevelopmentvalidation groups")
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);agent.set_goal(TensorDict(bank['raw_goal'],batch_size=[]))
    for key in ('visual','proprio'):exact(agent.goal_state_enc[key].float().cpu(),bank['goal_encoded'][key].float(),"freshfixedgoal")
    z=TensorDict(bank['context'],batch_size=[]).to(agent.device);block=wm.model.predictor.predictor_blocks[3]
    calibration={k:torch.as_tensor(v,dtype=torch.float32,device=agent.device) for k,v in c.items()}
    epsilon=.001*calibration['p3_norm_median'];start=time.monotonic();candidate_results=[];probe_seconds=0.;forward_calls=0
    for candidate,truth in enumerate(bank['candidates']):
        plan=bank['normalized_actions'][candidate,:,None].to(agent.device)
        def forward(delta=None):
            nonlocal forward_calls
            forward_calls+=1
            with FirstDelta(block,delta) as hook:out=wm.unroll(z.clone(),act_suffix=plan)
            if hook.calls!=6:raise ValueError("Wrong nativeH6 callcount")
            return out,hook
        base,hook=forward();basevisual=base['visual'][1:,0].float().cpu()
        exact(basevisual,truth['predicted_visual'],"exactcoarsebankbaseline")
        p3=hook.first
        error=((p3-calibration['residual_mean'][0])/calibration['residual_scale'][0])@calibration['residual_coef'][0].T+calibration['residual_intercept'][0]
        def measurements(out,norm=0.):
            pred=out['visual'][1:,0].float().cpu()
            return {'visual_mse':(pred-truth['actual_visual']).square().reshape(6,-1).mean(1).numpy(),
                'native_cost':agent.objective(out,plan,keepdims=True).cpu().numpy(),
                'predicted_pooled':pool_visual(pred).numpy(),'delivered_pooled_norm':norm}
        records={'unsteered':measurements(base)};responses={}
        for name in ('readable','action_effect','random'):
            basis=calibration['basis_'+name];response=[];asymmetry=[];t0=time.monotonic()
            for axis in range(2):
                shift=epsilon*basis[:,axis][None]
                plus,_=forward(shift);minus,_=forward(-shift)
                pp=pool_visual(plus['visual'][1:,0].float())[0];mm=pool_visual(minus['visual'][1:,0].float())[0]
                bb=pool_visual(base['visual'][1:,0].float())[0]
                response.append((pp-mm)/(2*epsilon));asymmetry.append(float((pp+mm-2*bb).norm()/(pp-mm).norm().clamp_min(1e-12)))
            probe_seconds+=time.monotonic()-t0
            measured=torch.stack(response);surrogate=basis.T@calibration['response_raw'][0]
            cos=torch.nn.functional.cosine_similarity(measured.reshape(1,-1),surrogate.reshape(1,-1)).item()
            responses[name]={'actual_finite_response':measured.cpu().numpy(),'surrogate_response':surrogate.cpu().numpy(),
                'cosine':cos,'relative_surrogate_error':float((surrogate-measured).norm()/measured.norm().clamp_min(1e-12)),
                'singular_values':torch.linalg.svdvals(measured).cpu().tolist(),'central_difference_asymmetry':asymmetry}
            for method,matrix in (('surrogate',surrogate),('finite',measured)):
                for dose in (.001,.005):
                    delta=correction(error,basis,matrix,dose*calibration['p3_norm_median'])
                    out,edited=forward(delta)
                    records[f'{name}_{method}_{dose}']=measurements(out,float(edited.realized.norm()))
            anti=correction(error,basis,measured,-.001*calibration['p3_norm_median']);out,edited=forward(anti)
            records[name+'_finite_negative_0.001']=measurements(out,float(edited.realized.norm()))
        candidate_results.append({'records':records,'response_diagnostics':responses})
    conditions=[];arrays={};real=np.stack([np.asarray(r['actual_encoded_native_cost_by_horizon']) for r in bank['candidates']]);physical=np.stack([r['metrics']['goal_xy_distance'] for r in bank['candidates']])
    base_mse=np.stack([r['records']['unsteered']['visual_mse'] for r in candidate_results])
    for name in candidate_results[0]['records']:
        mses=np.stack([r['records'][name]['visual_mse'] for r in candidate_results]);costs=np.stack([r['records'][name]['native_cost'] for r in candidate_results])
        arrays[name+'_visual_mse']=mses;arrays[name+'_native_cost']=costs
        arrays[name+'_predicted_pooled']=np.stack([r['records'][name]['predicted_pooled'] for r in candidate_results])
        conditions.append({'name':name,'visual_mse':float(mses.mean()),'first_horizon_mse':float(mses[:,0].mean()),
            'mse_delta_vs_unsteered':float((mses-base_mse).mean()),'first_horizon_delta_vs_unsteered':float((mses[:,0]-base_mse[:,0]).mean()),
            'delivered_pooled_norm':float(np.mean([r['records'][name]['delivered_pooled_norm'] for r in candidate_results])),
            'ranking':ranking(costs[:,-1].reshape(-1),real[:,-1].reshape(-1),physical[:,-1])})
    response_summary={}
    for basis in ('readable','action_effect','random'):
        entries=[r['response_diagnostics'][basis] for r in candidate_results]
        response_summary[basis]={k:np.asarray([e[k] for e in entries]).mean(0).tolist() for k in ('cosine','relative_surrogate_error','singular_values','central_difference_asymmetry')}
        arrays[basis+'_actual_finite_response']=np.stack([e['actual_finite_response'] for e in entries]);arrays[basis+'_surrogate_response']=np.stack([e['surrogate_response'] for e in entries])
    out=args.output/(row['key']+'.json')
    write_json(out,{'complete':True,'key':row['key'],'split':row['split'],'conditions':conditions,'response_diagnostics':response_summary,
        'seconds':time.monotonic()-start,'finite_probe_seconds':probe_seconds,'native_forward_calls':forward_calls,
        'futuretruthusedtoedit':False,'edit_target_origin':'unchanged TRAIN-only horizon1 latentresidualprediction fromcurrentP3',
        'local_response_origin':'finite±P3 edits throughsamefrozenmodel/currentactions; no simulatorfutureinput',
        'source_bank_sha256':row['sha256'],'calibration_sha256':sha256(args.calibration/'calibration.npz')})
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    print(json.dumps({'event':'finite_response_complete','key':row['key'],'seconds':time.monotonic()-start,'forward_calls':forward_calls}),flush=True)
    return [{'path':p.name,'sha256':sha256(p),'bytes':p.stat().st_size,'key':row['key'],'split':row['split']} for p in (out,out.with_suffix('.npz'))]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bank','calibration','repo','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--keys',nargs='+',required=True);args=p.parse_args()
    if any(k not in ('train-1818','train-1919','train-2020','train-2121') for k in args.keys):raise ValueError("Only fourpredeclareddevelopmentgroups")
    d=json.loads((args.bank/'DONE.json').read_text());f=json.loads((args.calibration/'FROZEN.json').read_text())
    if not d['complete'] or not f['complete'] or sha256(args.calibration/'calibration.npz')!=f['calibration_sha256']:raise ValueError("Badsource/calibrationreceipt")
    rows=[r for r in d['outputs'] if r['path'].endswith('.pt') and r['key'] in args.keys]
    if len(rows)!=len(args.keys) or any(r['split']!='development_validation' for r in rows):raise ValueError("Wrongsource/split beforeload")
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'protocol.json',{'rows':rows,'site':'P3','rank':2,'horizon':1,
        'epsilon_ratio_to_training_P3norm':.001,'correction_doses':[.001,.005],'response_directions_per_basis':4,'bases':['readable','action_effect','random'],
        'futuretruthusedtoedit':False,'script_sha256':sha256(__file__),'calibration_frozen_sha256':sha256(args.calibration/'FROZEN.json')})
    from model_loader import load_headless
    torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(args.repo,args.output)
    wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);c=dict(np.load(args.calibration/'calibration.npz',allow_pickle=False));start=time.monotonic();outputs=[]
    for row in rows:outputs.extend(run_source(args,row,wm,prep,cfg,c))
    if parameter_sha(wm)!=before:raise ValueError("Weights changed")
    write_json(args.output/'DONE.json',{'complete':True,'outputs':outputs,'seconds':time.monotonic()-start,'parameter_sha256':before,'provenance':provenance,'cem_calls':0})

if __name__=='__main__':main()
