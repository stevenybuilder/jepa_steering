#!/usr/bin/env python3
"""Matched native head spatial-versus-mean ablation on fixed near-plan actions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from residual_search_hooks import ComponentPatch
from capture_horizon_coordinates import exact
from capture_pusht_calibration import parameter_sha
from causal_response_transfer import restore_fullprecision_goal
from complete_cached_geometry import sha256,write_json
from collect_pusht_bank import config
from fit_action_consequence import ranking

def component(x,kind):
    mean=x.mean(1,keepdim=True)
    if kind=='spatial':return x-mean
    if kind=='mean':return mean.expand_as(x)
    raise ValueError('Unknown fixed component')

class FixedNormHead(ComponentPatch):
    def __init__(self,block,head,horizon,kind,radius,sham=False,identity=False):
        spec=dict(site='attention_preproj',head=head,pulse=horizon,mask='all',sign=1,dose=1.)
        super().__init__(block,spec,lambda x,h:x,radius,sham=sham,identity=identity);self.kind=kind
    def _delta(self,current,gate):
        lo,hi=self._head_support(current.shape[-1]);raw=torch.zeros_like(current)
        raw[...,lo:hi]=-component(current[...,lo:hi],self.kind)
        original=self._map(raw,gate).flatten(1).norm(dim=-1)
        if (original<=1e-12).any():raise ValueError('Zero component cannot receive matched dose; no head omission allowed')
        target=torch.full_like(original,self.cap_radius)
        if self.sham:
            gen=torch.Generator(device='cpu').manual_seed(2026090607)
            perm=torch.randperm(hi-lo,generator=gen).to(raw.device)
            signs=(2*torch.randint(0,2,(hi-lo,),generator=gen)-1).to(raw)
            raw[...,lo:hi]=raw[...,lo:hi][...,perm]*signs
        available=self._map(raw,gate).flatten(1).norm(dim=-1)
        if (available<=1e-12).any():raise ValueError('Zero sham mapped norm')
        delta=raw*(target/available)[:,None,None]
        requested=self._map(delta,gate).flatten(1).norm(dim=-1)
        torch.testing.assert_close(requested,target,rtol=2e-6,atol=2e-5)
        return delta,dict(step=self.step,head=self.candidate['head'],kind=self.kind,sham=self.sham,
            requested_residual_norm=requested.cpu(),semantic_requested_residual_norm=target.cpu(),
            unsuppressed_component_norm=original.cpu(),suppression_fraction=(target/original).cpu(),
            normmatch_max_abs=float((requested-target).abs().max()))

def specifications(heads):
    return [dict(name=f'head{head:02d}_H{h}_{kind}_'+('sham' if sham else 'semantic'),head=head,horizon=h,kind=kind,sham=sham)
        for head in range(heads) for h in (1,3,6) for kind in ('spatial','mean') for sham in (False,True)]

def batch_context(context,count):
    from tensordict import TensorDict
    if any(v.shape[:2]!=(1,1) for v in context.values()):raise ValueError('Expected one context frame and one batch')
    # Native EncPredWM context is [batch,time,...], unlike its time-major return.
    return TensorDict({k:v.repeat_interleave(count,dim=0) for k,v in context.items()},batch_size=[])

@torch.no_grad()
def run_source(args,row,wm,prep,cfg,radius,heads):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    source=args.bank/row['path']
    if sha256(source)!=row['sha256']:raise ValueError('Near bank SHA mismatch before loading')
    bank=torch.load(source,map_location='cpu',weights_only=False)
    if bank['row']['split']!='development_external' or len(bank['candidates'])!=9:raise ValueError('Wrong near-panel source')
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);agent.set_goal(TensorDict(bank['raw_goal'],batch_size=[]))
    goal_diff=restore_fullprecision_goal(agent,bank['goal_encoded'])
    z=batch_context(bank['context'],9).to(agent.device);plan=bank['normalized_actions'].transpose(0,1).to(agent.device)
    saved_actions=plan.clone();start=time.monotonic();block=wm.model.predictor.predictor_blocks[3]
    def forward(setting=None,identity=False):
        if setting is None:return wm.unroll(z.clone(),act_suffix=plan),[]
        with FixedNormHead(block,setting['head'],setting['horizon'],setting['kind'],radius,setting['sham'],identity) as hook:
            pred=wm.unroll(z.clone(),act_suffix=plan)
        if hook.step!=6:raise ValueError('Wrong native H6 hook count')
        if not identity:
            if len(hook.records)!=1:raise ValueError('Expected exactly one edited imagined horizon')
            actual=hook.records[0]['actual_rounded_residual_norm'];target=torch.full_like(actual,radius)
            torch.testing.assert_close(actual,target,rtol=2e-5,atol=2e-5)
        return pred,hook.records
    base,_=forward();repeat,_=forward()
    identity,_=forward(specifications(1)[0],True)
    for channel in ('visual','proprio'):
        exact(base[channel],repeat[channel],'fresh batch9 native repeat '+channel)
        exact(base[channel],identity[channel],'fresh batch9 identity '+channel)
    truth=torch.stack([r['actual_visual'] for r in bank['candidates']],dim=1)
    old_pred=torch.stack([r['predicted_visual'] for r in bank['candidates']],dim=1)
    real=np.stack([np.asarray(r['actual_encoded_native_cost_by_horizon']) for r in bank['candidates']])[:,-1].reshape(-1)
    xy=np.stack([r['metrics']['goal_xy_distance'] for r in bank['candidates']])[:,-1]
    angles=np.stack([r['metrics']['wrapped_angle_error'] for r in bank['candidates']])[:,-1]
    blocks=np.stack([r['metrics']['goal_block_distance'] for r in bank['candidates']])[:,-1]
    arrays={};conditions=[]
    def measure(name,out,setting=None,records=()):
        error=(out['visual'][1:].float().cpu()-truth).square()
        mse=error.reshape(6,9,-1).mean(-1).T.numpy()
        patch=error.reshape(6,9,256,384).mean(-1).permute(1,0,2).numpy()
        cost=agent.objective(out,plan,keepdims=True).cpu().numpy();final=cost[-1].reshape(-1)
        score=ranking(final,real,xy);selected=score['selected_candidate']
        row=dict(name=name,forecast_visual_mse=float(mse.mean()),horizon_visual_mse=mse.mean(0).tolist(),ranking=score,
            selected_actual_block_distance_px=float(blocks[selected]),selected_actual_wrapped_angle_rad=float(angles[selected]))
        if setting is not None:row.update(setting)
        if records:
            rec=records[0];row['norm_diagnostics']={k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in rec.items()}
        conditions.append(row);arrays[name+'_visual_mse']=mse;arrays[name+'_patch_visual_mse']=patch;arrays[name+'_native_cost']=cost
    measure('unsteered',base)
    old_cost=np.stack([np.asarray(r['native_cost_by_horizon']) for r in bank['candidates']])[:,-1].reshape(-1)
    batch_differences=dict(full_visual_maxabs=float((base['visual'][1:].float().cpu()-old_pred).abs().max()),
        final_native_cost_maxabs=float(np.abs(arrays['unsteered_native_cost'][-1].reshape(-1)-old_cost).max()),
        old_single_selected=int(old_cost.argmin()),fresh_batch_selected=conditions[0]['ranking']['selected_candidate'],
        old_single_rank=ranking(old_cost,real,xy),historical_bitexact_claimed=False)
    # Fixed first four conditions are timing-only canary; never choose heads by outcomes.
    settings=specifications(heads or 16);t0=time.monotonic()
    for setting in settings[:4]:
        out,records=forward(setting);measure(setting['name'],out,setting,records)
    torch.cuda.synchronize();seconds_per=(time.monotonic()-t0)/4
    if heads is None:
        heads=16 if seconds_per*193*4<=600 else 8
        settings=specifications(heads)
        write_json(args.output/'CANARY.json',dict(complete=True,seconds_per_batch_condition=seconds_per,
            estimated_seconds_16heads=seconds_per*193*4,selected_head_count_by_runtime_only=heads,
            max_cuda_memory_allocated=torch.cuda.max_memory_allocated(),fresh_batch_repeat_and_identity_exact=True,
            old_single_vs_batch=batch_differences,radius=radius,source=row['key']))
        print(json.dumps(dict(event='head_ablation_canary_pass',seconds_per_batch_condition=seconds_per,heads=heads)),flush=True)
    for setting in settings[4:]:
        out,records=forward(setting);measure(setting['name'],out,setting,records)
    exact(plan,saved_actions,'same cached actions for every condition')
    baseline=conditions[0]['forecast_visual_mse']
    for r in conditions:r['visual_mse_delta_vs_unsteered']=r['forecast_visual_mse']-baseline
    out=args.output/(row['key']+'.json')
    write_json(out,dict(complete=True,key=row['key'],split=row['split'],conditions=conditions,seconds=time.monotonic()-start,
        source_sha256=row['sha256'],fullprecision_saved_goal_exact=True,fresh_goal_encoding_difference=goal_diff,
        old_single_vs_fresh_batch=batch_differences,heads=list(range(heads)),native_candidate_batch=9,
        radius=radius,native_objective_alpha=float(agent.objective.alpha),futuretruthusedtoedit=False,
        limitations=['Spatial and mean components differ in token degrees of freedom; equal site/head/norm, not equal dimensionality',
            'Norm-scaled suppression can exceed unit attenuation when requested radius exceeds available component; fractions reported',
            'Full spatial feature errors and actual cached physical consequences are offline scores, not model probabilities']))
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    print(json.dumps(dict(event='head_spatial_mean_complete',key=row['key'],conditions=len(conditions),seconds=time.monotonic()-start)),flush=True)
    return [dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,key=row['key'],split=row['split']) for p in (out,out.with_suffix('.npz'))],heads

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bank','norm','repo','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();d=json.loads((a.bank/'DONE.json').read_text());norm=json.loads(a.norm.read_text())
    rows=[r for r in d['outputs'] if r['path'].endswith('.pt')]
    if not d['complete'] or {r['key'] for r in rows}!={f'near-dev-{i:03}' for i in range(4)} or any(r['split']!='development_external' for r in rows):raise ValueError('Only fixed four near development sources')
    if not norm['complete'] or norm['train_groups']!=16 or norm['dose']!=.005:raise ValueError('Wrong frozen TRAIN norm')
    if sha256(a.checkpoint)!='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb':raise ValueError('Wrong pinned native checkpoint')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',dict(rows=rows,full_settings=specifications(16),norm_sha256=sha256(a.norm),
        norm_source=norm,script_sha256=sha256(__file__),helper_sha256=sha256(Path(__file__).with_name('residual_search_hooks.py')),
        batch_policy='All nine candidates, identical batch/order/precision for every fresh arm; old single-candidate differences descriptive',
        norm_tolerance=dict(requested_rtol=2e-6,actual_rounded_rtol=2e-5,atol=2e-5),timing_only_cap='If16head estimated>600s, fixed heads0..7'))
    from model_loader import load_headless
    torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(a.repo,a.output)
    wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);start=time.monotonic();outputs=[];heads=None
    for r in sorted(rows,key=lambda r:r['key']):
        entries,heads=run_source(a,r,wm,prep,cfg,norm['radius'],heads);outputs.extend(entries)
    if parameter_sha(wm)!=before:raise ValueError('Frozen checkpoint changed')
    write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,heads=heads,
        conditions=1+len(specifications(heads)),parameter_sha256=before,provenance=provenance,cem_calls=0,simulator_calls=0))

if __name__=='__main__':main()
