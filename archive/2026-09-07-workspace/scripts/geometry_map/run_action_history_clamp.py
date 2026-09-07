#!/usr/bin/env python3
"""Locate action-path bending in recurrent imagined state versus action history.

Only fixed model rollouts: no simulator truth, CEM, fitted decoder, or steering.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import torch

from action_path_geometry import sampled_path_geometry
from run_action_path_curvature import BLOCKS, HORIZONS, TIMES, BATCH, ResidualCapture, fixed_action_path

ARMS = ('native', 'central_state_context', 'central_state_and_past_actions')
CHECKPOINT = '9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'


def exact(a,b,name):
    if not torch.equal(a.detach().cpu(),b.detach().cpu()): raise RuntimeError('Exact guard: '+name)


def clamp_arguments(arguments, central, past_actions=False):
    """Preserve native argument strides; change no caller-owned tensor storage."""
    if len(arguments) != 3 or any(not isinstance(x,torch.Tensor) for x in arguments):
        raise ValueError('Expected native AdaLN predictor (visual, action, proprio) tensors')
    visual, actions, proprio = arguments
    if visual.ndim != 6 or actions.ndim not in (3,4) or proprio.ndim not in (3,4):
        raise ValueError('Unexpected batch/time native context axes')
    if visual.shape[:2] != actions.shape[:2] or visual.shape[:2] != proprio.shape[:2]:
        raise ValueError('Visual/action/proprio context windows differ')
    values=[]
    for index, (value, reference) in enumerate(zip(arguments,central)):
        if reference.shape[0] != 1 or reference.shape[1:] != value.shape[1:]:
            raise ValueError('Central source shape mismatch')
        if index == 1 and (not past_actions or value.shape[1] == 1):
            values.append(value); continue
        result=torch.empty_strided(value.shape,value.stride(),device=value.device,dtype=value.dtype)
        result.copy_(value)
        donor=reference.to(value)
        if index == 1: result[:,:-1].copy_(donor[:,:-1].expand_as(result[:,:-1]))
        else: result.copy_(donor.expand_as(result))
        values.append(result)
    return tuple(values)


class PredictorInputs:
    def __init__(self,predictor,central=None,past_actions=False):
        self.predictor=predictor; self.central=central; self.past_actions=past_actions
        self.values=[]; self.shapes=[]; self.calls=0
    def __enter__(self):
        self.handle=self.predictor.register_forward_pre_hook(self.hook); return self
    def hook(self,module,args):
        self.calls+=1
        if len(args)!=3 or any(not torch.is_tensor(x) for x in args): raise ValueError('Pinned AdaLN call signature changed')
        self.shapes.append([dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype)) for x in args])
        if self.central is None:
            self.values.append(tuple(x[2:3].detach().cpu().clone() for x in args)); return None
        reference=self.central[self.calls-1]
        for i,(x,y) in enumerate(zip(args,reference)):
            exact(x[2:3],y,f'central input row H{self.calls}/argument{i}')
        # At H1 all rows share actual initial state and there is no past action;
        # check the clamp is mathematically identity and preserve original storage.
        changed=clamp_arguments(args,reference,self.past_actions)
        if self.calls==1:
            for i,(x,y) in enumerate(zip(args,changed)): exact(x,y,f'H1 context identity argument{i}')
            return None
        return changed
    def __exit__(self,*exc):
        self.handle.remove()
        if exc[0] is None and self.calls != 6: raise ValueError('Expected six native predictor calls')


def padded_plan(prep,raw):
    normalized=prep.normalize_actions(raw.reshape(5,6,5,2)).reshape(5,6,10)
    plan=torch.cat([normalized,normalized[2:3].repeat(2,1,1)]).transpose(0,1).contiguous().to('cuda:0')
    if plan.shape != (6,7,10) or not plan.is_contiguous(): raise ValueError('Pinned contiguous batch7 action layout required')
    return normalized,plan


@torch.no_grad()
def run_source(args,row,wm,prep,first):
    from complete_cached_geometry import sha256,write_json
    from intervene_head_spatial_mean import batch_context
    source=args.bank/row['path']
    if sha256(source)!=row['sha256']: raise ValueError('Source hash mismatch before tensor load')
    bank=torch.load(source,map_location='cpu',weights_only=False)
    if bank['row']['split']!='development_external' or bank['row']['key']!=row['key']: raise ValueError('Forbidden source tensor')
    z=batch_context(bank['context'],BATCH).to('cuda:0'); z_saved={k:z[k].clone() for k in z.keys()}
    center_raw=bank['raw_actions'][0:1].repeat(5,1,1)
    _,central_plan=padded_plan(prep,center_raw)
    with PredictorInputs(wm.model.predictor) as capture:
        center=wm.unroll(z.clone(),act_suffix=central_plan)
    center_repeat=wm.unroll(z.clone(),act_suffix=central_plan)
    for k in ('visual','proprio'): exact(center[k],center_repeat[k],'read-only central input capture '+k)
    central=capture.values
    entries=[]
    for pair in range(4):
        for radius in (1,4):
            start=time.monotonic()
            raw,meta=fixed_action_path(bank['raw_actions'],pair,radius)
            normalized,plan=padded_plan(prep,raw); saved=plan.clone()
            exact(normalized[2],bank['normalized_actions'][0],'frozen central normalized actions')
            predictions={}; residuals={}; contexts={}
            for arm in ARMS:
                kwargs={} if arm=='native' else dict(central=central,past_actions=arm==ARMS[2])
                with PredictorInputs(wm.model.predictor,**kwargs) as input_capture, ResidualCapture(wm.model.predictor.predictor_blocks) as residual:
                    output=wm.unroll(z.clone(),act_suffix=plan)
                for k in ('visual','proprio'):
                    exact(output[k][:,2],center[k][:,2],'central output invariant '+arm+'/'+k)
                    if arm!='native': exact(output[k][:2],predictions['native'][k][:2],'all H1 predictions unchanged '+arm+'/'+k)
                predictions[arm]={k:output[k].float().cpu() for k in ('visual','proprio')}
                residuals[arm]=residual.values
                contexts[arm]=input_capture.shapes
                if arm!='native':
                    for block in BLOCKS: exact(residual.values[block,1],residuals['native'][block,1],'all H1 residuals unchanged '+arm)
            no_history=all(shape[1]['shape'][1]==1 for shape in contexts['native'])
            if no_history:
                for k in ('visual','proprio'): exact(predictions[ARMS[1]][k],predictions[ARMS[2]][k],'no past action slots: outputs exactly equivalent')
                for site in residuals[ARMS[1]]: exact(residuals[ARMS[1]][site],residuals[ARMS[2]][site],'no past action slots: residuals equivalent')
            rows=[]
            for block in BLOCKS:
                for horizon in HORIZONS:
                    for arm in ARMS:
                        rows.append(dict(block=block,horizon=horizon,arm=arm,
                            geometry=sampled_path_geometry(residuals[arm][block,horizon][:5])))
            outputs_by_arm={}
            for arm in ARMS:
                outputs_by_arm[arm]={}
                for key in ('visual','proprio'):
                    current=predictions[arm][key][1:,:5]
                    baseline=predictions['native'][key][1:,:5]
                    outputs_by_arm[arm][key]=dict(mse_to_native_by_horizon_and_t=(current-baseline).square().flatten(2).mean(-1).tolist(),
                        geometry_by_horizon=[sampled_path_geometry(current[h]) for h in range(6)])
            exact(plan,saved,'all raw model actions unchanged')
            for k in z.keys(): exact(z[k],z_saved[k],'initial context unchanged '+k)
            name=f"{row['key']}-pair{pair}-radius{radius}"
            path=args.output/(name+'.pt')
            unique=ARMS[:2] if no_history else ARMS
            torch.save(dict(complete=True,key=row['key'],source_sha256=row['sha256'],path=meta,
                sample_t=TIMES,raw_actions=raw,normalized_actions=normalized,
                central_predictor_inputs=central,central_input_shapes=capture.shapes,
                residuals={arm:{f'P{b}_H{h}':residuals[arm][b,h][:5].clone() for b in BLOCKS for h in HORIZONS} for arm in unique},
                predictions={arm:{k:v[:, :5].clone() for k,v in predictions[arm].items()} for arm in unique},
                arm_alias={ARMS[2]:ARMS[1]} if no_history else {},
                context_shapes=contexts,wm_ctxt_window=wm.ctxt_window,proprio_mode=wm.proprio_mode,
                no_past_action_slots=no_history,no_history_arm_equivalence_exact=no_history,
                initial_context={k:v.cpu() for k,v in z_saved.items()},
                central_exact=True,all_samples_H1_exact=True,simulator_calls=0,cem_calls=0),path)
            torch.cuda.synchronize(); seconds=time.monotonic()-start
            report=dict(complete=True,key=row['key'],source_sha256=row['sha256'],path=meta,seconds=seconds,
                wm_ctxt_window=wm.ctxt_window,proprio_mode=wm.proprio_mode,context_shapes=contexts,
                no_past_action_slots=no_history,third_arm_independent=not no_history,third_arm_exact_duplicate=no_history,
                central_exact=True,all_samples_H1_exact=True,rows=rows,output_changes=outputs_by_arm,
                full_tensor_sha256=sha256(path),full_tensor_bytes=path.stat().st_size,
                interpretation='Intervention on model-imagined history; not actual-state teacher forcing or physical efficacy')
            write_json(path.with_suffix('.json'),report)
            for p in (path,path.with_suffix('.json')): entries.append(dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size))
            print(json.dumps(dict(event='history_clamp_path_complete',key=row['key'],pair=pair,radius=radius,seconds=seconds,ctxt_window=wm.ctxt_window,no_past_action_slots=no_history)),flush=True)
            if first:
                upper=seconds*32+args.init_seconds
                write_json(args.output/'CANARY.json',dict(complete=True,seconds=seconds,estimated_process_seconds=upper,
                    max_seconds=900,all_samples_H1_exact=True,central_exact=True,no_outcome_gate=True))
                print(json.dumps(dict(event='history_clamp_canary',seconds=seconds,estimated_process_seconds=upper)),flush=True)
                if upper>900: raise RuntimeError('Fixed first-path timing exceeds authorized900seconds estimate')
                first=False
            if time.monotonic()-args.started>900: raise RuntimeError('Process budget exceeded; no next path')
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','repo','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); args.started=time.monotonic()
    from complete_cached_geometry import sha256,write_json
    from model_loader import load_headless
    from capture_pusht_calibration import parameter_sha
    done=json.loads((args.bank/'DONE.json').read_text())
    rows=sorted([r for r in done['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['key'])
    if not done['complete'] or [r['key'] for r in rows]!=[f'near-dev-{i:03d}' for i in range(4)] or any(r['split']!='development_external' for r in rows):
        raise ValueError('Exactly four original development sources, checked before tensor load')
    if sha256(args.checkpoint)!=CHECKPOINT: raise ValueError('Pinned checkpoint mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        write_json(args.output/'protocol.json',dict(rows=rows,arms=ARMS,radii=[1,4],pairs=list(range(4)),t=TIMES,
            blocks=BLOCKS,horizons=HORIZONS,batch=7,action_tensor_layout='contiguous H6,B7,A10',
            script_sha256=sha256(__file__),geometry_helper_sha256=sha256(Path(__file__).with_name('action_path_geometry.py')),
            checkpoint_sha256=CHECKPOINT,all_input_truth='Natural central imagined histories freshly computed same runtime/batch',
            no_past_action_policy='If every input window has length1, run and verify third-arm exact duplicate; do not count independent evidence',
            held_access=False,simulator_calls=0,cem_calls=0,max_process_seconds=900,
            unit='Four fixed development starts; pair/radius/site/horizon are dependent probes'))
        torch.set_num_threads(2);torch.manual_seed(90505)
        wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
        wm.eval().requires_grad_(False)
        if wm.model.pred_type!='AdaLN' or wm.proprio_mode!='predict_proprio': raise ValueError('Pinned native recurrent predictor changed')
        before=parameter_sha(wm);args.init_seconds=time.monotonic()-args.started;entries=[]
        for i,row in enumerate(rows): entries.extend(run_source(args,row,wm,prep,i==0))
        if parameter_sha(wm)!=before: raise ValueError('Frozen parameter hash changed')
        write_json(args.output/'DONE.json',dict(complete=True,outputs=entries,paths=32,seconds=time.monotonic()-args.started,
            parameter_sha256=before,provenance=provenance,wm_ctxt_window=wm.ctxt_window,proprio_mode=wm.proprio_mode,
            all_central_and_H1_exact=True,held_access=False,simulator_calls=0,cem_calls=0))
    except Exception:
        write_json(args.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
