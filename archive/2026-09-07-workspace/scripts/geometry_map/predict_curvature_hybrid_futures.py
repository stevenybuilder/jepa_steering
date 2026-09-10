#!/usr/bin/env python3
"""Unedited native forecasts of the already-executed 128 hybrid action plans."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import torch

from collect_curvature_interior_futures import CHECKPOINT, action_sha


def fixed_batch_plan(normalized):
    if normalized.shape!=(6,10) or normalized.dtype!=torch.float32 or not torch.isfinite(normalized).all():
        raise ValueError('Require frozen finite float32 H6 normalized Push action plan')
    return normalized[:,None,:].repeat(1,8,1).contiguous()


def select_hybrids(receipt):
    rows=sorted([r for r in receipt['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['path'])
    expected={f"near-dev-{e:03d}-pair{p}-radius{r}-t{s}025-H{h}.pt" for e in range(4) for p in range(4) for r in (1,4) for s in ('minus','plus') for h in (1,3)}
    if not receipt.get('complete') or not receipt.get('hybrid') or receipt.get('held_access') is not False or {r['path'] for r in rows}!=expected or len(rows)!=128:
        raise ValueError('Exactly128 completed original-development hybrids required before tensor load')
    return rows


def exact(x,y,name):
    if not torch.equal(x.detach().cpu(),y.detach().cpu()): raise ValueError('Exact native identity guard: '+name)


@torch.no_grad()
def run(args):
    from tensordict import TensorDict
    from complete_cached_geometry import sha256,write_json
    from collect_pusht_bank import config
    from model_loader import load_headless
    from capture_pusht_calibration import parameter_sha
    from intervene_head_spatial_mean import batch_context
    started=time.monotonic()
    receipt=json.loads((args.truth/'DONE.json').read_text());rows=select_hybrids(receipt)
    near=json.loads((args.bank/'DONE.json').read_text())
    banks={r['key']:r for r in near['outputs'] if r['path'].endswith('.pt')}
    if not near['complete'] or set(banks)!={f'near-dev-{e:03d}' for e in range(4)} or any(r['split']!='development_external' for r in banks.values()):
        raise ValueError('Only four near-development sources before load')
    if sha256(args.checkpoint)!=CHECKPOINT: raise ValueError('Frozen checkpoint mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        write_json(args.output/'protocol.json',dict(process_pid=os.getpid(),truth_done_sha256=sha256(args.truth/'DONE.json'),
            script_sha256=sha256(__file__),checkpoint_sha256=CHECKPOINT,plans=128,batch=8,selected_row=1,
            action_layout='contiguous H6,B8,A10; all eight rows contain same hybrid action sequence',
            comparison='Unedited full native hybrid forecast versus matching physical hybrid truth; no donor state import',
            goal='Original fullprecision bank target direct; no new encoding',
            full_H6_references='Reused separately by consumer from existing unedited natural_donor_visual; no new H6 plan run',
            simulator_calls=0,cem_calls=0,held_access=False,max_process_seconds=300))
        torch.set_num_threads(2);torch.manual_seed(90505)
        cfg=config(args.repo,args.output)
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
        wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
        wm.eval().requires_grad_(False);before=parameter_sha(wm)
        contexts={}; objectives={}
        for key,row in banks.items():
            path=args.bank/row['path']
            if sha256(path)!=row['sha256']: raise ValueError('Near source checksum mismatch')
            bank=torch.load(path,map_location='cpu',weights_only=False)
            if bank['row']['split']!='development_external': raise ValueError('Near tensor split mismatch')
            contexts[key]=batch_context(bank['context'],8).to('cuda:0')
            for field in ('visual','proprio'):
                if bank['goal_encoded'][field].dtype!=torch.float32: raise ValueError('Require original float32 goal')
            target=TensorDict({k:v.clone().to('cuda:0') for k,v in bank['goal_encoded'].items()},batch_size=[])
            objectives[key]=ReprTargetDistMPCObjective(cfg,target_enc=target,**cfg.planner.planning_objective)
            for field in ('visual','proprio'): exact(objectives[key].target_enc[field],bank['goal_encoded'][field],'saved goal '+field)
        init=time.monotonic()-started;outputs=[];prefix={}
        for index,row in enumerate(rows):
            start=time.monotonic();path=args.truth/row['path']
            if sha256(path)!=row['sha256']: raise ValueError('Physical hybrid source hash mismatch before load')
            truth=torch.load(path,map_location='cpu',weights_only=False)
            key=truth['key']
            if key not in banks or truth['source_sha256']!=banks[key]['sha256'] or truth['split']!='development_external':
                raise ValueError('Hybrid source/context identity mismatch')
            raw=truth['raw_actions'];normalized=truth['normalized_actions']
            if action_sha(raw)!=truth['hybrid_action_sha256']: raise ValueError('Raw hybrid action hash mismatch')
            fresh=prep.normalize_actions(raw.reshape(1,6,5,2)).reshape(6,10)
            exact(fresh,normalized,'saved official normalized hybrid actions')
            plan=fixed_batch_plan(normalized).to('cuda:0');saved=plan.clone();z=contexts[key]
            initial={k:z[k].clone() for k in z.keys()}
            forecast=wm.unroll(z.clone(),act_suffix=plan)
            repeat=wm.unroll(z.clone(),act_suffix=plan)
            for field in ('visual','proprio'): exact(forecast[field],repeat[field],'unedited native repeat '+field)
            for field in z.keys(): exact(z[field],initial[field],'initial context unchanged '+field)
            exact(plan,saved,'fixed hybrid action tensor unchanged')
            pred={field:forecast[field][1:,1].float().cpu() for field in ('visual','proprio')}
            if pred['visual'].shape!=(6,1,16,16,384): raise ValueError('Unexpected native future spatial layout')
            errors={field:(pred[field]-truth['actual_'+field]).square().flatten(1).mean(-1).tolist() for field in ('visual','proprio')}
            floor={field:float((forecast[field][:,0]-forecast[field][:,1]).abs().max()) for field in ('visual','proprio')}
            base=row['path'].rsplit('-H',1)[0]
            if truth['patch_horizon']==1: prefix[base]={k:v[0].clone() for k,v in pred.items()}
            else:
                for field in pred: exact(pred[field][0],prefix[base][field],'same donor first chunk H1 forecast across H1/H3 hybrids')
            costs=objectives[key](forecast,plan,keepdims=True)[1:,1].float().cpu()
            value=dict(complete=True,key=key,pair=truth['pair'],radius=truth['radius'],t=truth['t'],patch_horizon=truth['patch_horizon'],
                hybrid_truth_source_path=str(path),hybrid_truth_sha256=row['sha256'],near_bank_sha256=banks[key]['sha256'],
                raw_actions=raw,normalized_actions=normalized,hybrid_action_sha256=truth['hybrid_action_sha256'],
                predicted_visual=pred['visual'],predicted_proprio=pred['proprio'],
                actual_error_by_horizon=errors,native_goal_cost_by_horizon=costs,
                actual_encoded_native_cost_by_horizon=truth['actual_encoded_native_cost_by_horizon'],
                native_repeat_exact=True,all_actions_unchanged=True,same_H1_donor_prefix_forecast_exact=True,
                batch=8,selected_row=1,row0_vs_row1_maxabs=floor,simulator_calls=0,cem_calls=0)
            out=args.output/row['path'];torch.save(value,out);torch.cuda.synchronize()
            seconds=time.monotonic()-start
            report={k:value[k] for k in ('key','pair','radius','t','patch_horizon','hybrid_truth_sha256','near_bank_sha256','hybrid_action_sha256','actual_error_by_horizon','row0_vs_row1_maxabs')}
            report.update(complete=True,seconds=seconds,native_repeat_exact=True,raw_actions_hash_exact=True,
                native_goal_cost_by_horizon=costs.reshape(-1).tolist(),full_tensor_sha256=sha256(out),full_tensor_bytes=out.stat().st_size)
            write_json(out.with_suffix('.json'),report)
            for p in (out,out.with_suffix('.json')):outputs.append(dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size))
            print(json.dumps(dict(event='native_hybrid_forecast_complete',path=row['path'],seconds=seconds)),flush=True)
            if index==0:
                estimate=seconds*128+init
                write_json(args.output/'CANARY.json',dict(complete=True,seconds=seconds,estimated_process_seconds=estimate,limit_seconds=300,identity_exact=True,no_outcome_gate=True))
                print(json.dumps(dict(event='native_hybrid_canary',estimated_process_seconds=estimate)),flush=True)
                if estimate>300: raise RuntimeError('First fixed native forecast timing exceeds300seconds')
            if time.monotonic()-started>300: raise RuntimeError('Native forecast process budget exhausted; no next plan')
        if parameter_sha(wm)!=before: raise ValueError('Native weights changed')
        write_json(args.output/'DONE.json',dict(complete=True,outputs=outputs,plans=128,process_pid=os.getpid(),
            seconds=time.monotonic()-started,parameter_sha256=before,provenance=provenance,
            all_repeat_action_prefix_identities_exact=True,simulator_calls=0,cem_calls=0,held_access=False))
    except Exception:
        write_json(args.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('truth','bank','repo','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
