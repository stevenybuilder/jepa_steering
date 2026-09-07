#!/usr/bin/env python3
"""Frozen TRAIN causal-response transfer, separate from observational regression."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from complete_cached_geometry import sha256, write_json
from capture_pusht_calibration import parameter_sha
from capture_horizon_coordinates import exact, pool_visual
from finite_response_consequence import FirstDelta, correction
from fit_action_consequence import ranking
from collect_pusht_bank import config

BASES = ('readable', 'action_effect', 'random')
TRAIN_KEYS = tuple(f'train-{i:04}' for i in range(202, 1718, 101))
DEV_KEYS = ('train-1818', 'train-1919', 'train-2020', 'train-2121')
ACTION_IDS = (0, 1, 2)  # reference, zero, +X; fixed before any response capture

def restore_fullprecision_goal(agent, saved):
    differences = {}
    for key in ('visual', 'proprio'):
        if saved[key].dtype != torch.float32:
            raise ValueError('Require original fullprecision goal, not promoted float16')
        differences[key] = float((agent.goal_state_enc[key].float().cpu()-saved[key]).abs().max())
        agent.goal_state_enc[key] = saved[key].to(agent.device).clone()
    agent.objective.target_enc = agent.goal_state_enc
    agent.planner.set_objective(agent.objective)
    for key in ('visual', 'proprio'):
        exact(agent.objective.target_enc[key].cpu(), saved[key], 'effective original fullprecision objective')
    return differences

def select_rows(receipt, mode, keys=None, near_plan=False):
    if near_plan and mode != 'evaluate':raise ValueError('Near-plan is evaluation only')
    allowed = tuple(f'near-dev-{i:03d}' for i in range(4)) if near_plan else (TRAIN_KEYS if mode == 'capture' else DEV_KEYS)
    split = 'development_external' if near_plan else ('fit' if mode == 'capture' else 'development_validation')
    if keys and any(k not in allowed for k in keys):
        raise ValueError('Forbidden group before tensor loading')
    rows = [r for r in receipt['outputs'] if r['path'].endswith('.pt')
            and r['key'] in allowed and (not keys or r['key'] in keys)]
    if not receipt['complete'] or not rows or any(r['split'] != split for r in rows):
        raise ValueError('Incomplete receipt or split mismatch before loading')
    if keys and {r['key'] for r in rows} != set(keys):
        raise ValueError('Missing predeclared source')
    return rows

def fit_response(x, y):
    """One fixed PCA16/ridge100 model; all statistics from training inputs."""
    from sklearn.linear_model import Ridge
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-6] = 1.
    z = (x - mean) / scale
    components = np.linalg.svd(z, full_matrices=False)[2][:16]
    model = Ridge(alpha=100.).fit(z @ components.T, y.reshape(len(y), -1))
    return dict(mean=mean, scale=scale, components=components,
                coef=model.coef_, intercept=model.intercept_, global_mean=y.mean(0))

def predict_response(m, x):
    return ((((x-m['mean'])/m['scale']) @ m['components'].T) @ m['coef'].T + m['intercept']).reshape(-1, 2, 384)

def fit(args):
    start = time.monotonic(); rows = []; xs = []; targets = {b: [] for b in BASES}
    for directory in args.capture_dirs:
        d = json.loads((directory/'DONE.json').read_text())
        if not d['complete']: raise ValueError('Incomplete training capture')
        for r in d['outputs']:
            if r['split'] != 'fit' or r['key'] not in TRAIN_KEYS:
                raise ValueError('Development response forbidden during fitting')
            p = directory/r['path']
            if sha256(p) != r['sha256']: raise ValueError('Training response SHA mismatch')
            b = dict(np.load(p, allow_pickle=False))
            if not np.array_equal(b['candidate_ids'], ACTION_IDS): raise ValueError('Wrong fixed actions')
            rows.append(r); xs.append(b['p3'])
            for basis in BASES: targets[basis].append(b[basis+'_response'])
    if len(rows) != 16 or {r['key'] for r in rows} != set(TRAIN_KEYS):
        raise ValueError('Exactly16 distinct fixed TRAIN groups required')
    args.output.mkdir(parents=True, exist_ok=False)
    x = np.concatenate(xs); arrays = {}
    for basis in BASES:
        m = fit_response(x.astype(float), np.concatenate(targets[basis]).astype(float))
        arrays.update({basis+'_'+k: v for k,v in m.items()})
    np.savez_compressed(args.output/'response_models.npz', **arrays)
    write_json(args.output/'FROZEN.json', dict(complete=True, train_keys=sorted(r['key'] for r in rows),
        train_initial_groups=16, train_action_rows=48, candidate_ids=ACTION_IDS, bases=BASES,
        input='native first-horizon P3 pooled400, TRAIN standardization and PCA16', ridge_alpha=100.,
        targets='actual finite-difference returned-visual response, not observational regression',
        held_development_response_used_for_fit=False, response_models_sha256=sha256(args.output/'response_models.npz'),
        training_sources=rows, script_sha256=sha256(__file__)))
    write_json(args.output/'DONE.json', dict(complete=True, seconds=time.monotonic()-start,
        outputs=[dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(args.output.iterdir())]))
    print(json.dumps(dict(event='causal_response_frozen', groups=16, seconds=time.monotonic()-start)),flush=True)

@torch.no_grad()
def run_source(args,row,wm,prep,cfg,c,models):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    path=args.bank/row['path']
    if sha256(path)!=row['sha256']: raise ValueError('Bank SHA mismatch')
    bank=torch.load(path,map_location='cpu',weights_only=False)
    if bank['row']['split']!=row['split']: raise ValueError('Payload split mismatch')
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep)
    agent.set_goal(TensorDict(bank['raw_goal'],batch_size=[]))
    goal_difference=restore_fullprecision_goal(agent,bank['goal_encoded'])
    z=TensorDict(bank['context'],batch_size=[]).to(agent.device)
    block=wm.model.predictor.predictor_blocks[3]
    tc={k:torch.as_tensor(v,dtype=torch.float32,device=agent.device) for k,v in c.items()}
    epsilon=.001*tc['p3_norm_median']; start=time.monotonic(); calls=0
    p3s=[]; responses={b:[] for b in BASES}; records=[]
    ids=ACTION_IDS if args.mode=='capture' else range(len(bank['candidates']))
    for candidate in ids:
        truth=bank['candidates'][candidate];plan=bank['normalized_actions'][candidate,:,None].to(agent.device)
        def forward(delta=None):
            nonlocal calls
            calls+=1
            with FirstDelta(block,delta) as hook: out=wm.unroll(z.clone(),act_suffix=plan)
            if hook.calls!=6: raise ValueError('Native H6 hook mismatch')
            return out,hook
        base,hook=forward();exact(base['visual'][1:,0].float().cpu(),truth['predicted_visual'],'same frozen native baseline')
        p3=hook.first;p3s.append(p3[0].cpu().numpy())
        if args.mode=='capture':
            for basis in BASES:
                columns=[]
                for axis in range(2):
                    shift=epsilon*tc['basis_'+basis][:,axis][None]
                    plus,_=forward(shift);minus,_=forward(-shift)
                    columns.append(((pool_visual(plus['visual'][1:,0].float())[0]-pool_visual(minus['visual'][1:,0].float())[0])/(2*epsilon)).cpu().numpy())
                responses[basis].append(np.stack(columns))
            continue
        error=((p3-tc['residual_mean'][0])/tc['residual_scale'][0])@tc['residual_coef'][0].T+tc['residual_intercept'][0]
        def measure(out,norm=0.):
            return dict(visual_mse=(out['visual'][1:,0].float().cpu()-truth['actual_visual']).square().reshape(6,-1).mean(1).numpy(),
                native_cost=agent.objective(out,plan,keepdims=True).cpu().numpy(), delivered_pooled_norm=norm)
        rr={'unsteered':measure(base)}
        for basis in BASES:
            m={k.removeprefix(basis+'_'):v for k,v in models.items() if k.startswith(basis+'_')}
            state_response=predict_response(m,p3.cpu().numpy())[0]
            responses[basis].append(state_response)
            for method,matrix in (('causal_mean',m['global_mean']),('causal_ridge',state_response)):
                mt=torch.as_tensor(matrix,dtype=torch.float32,device=agent.device)
                for dose in (.001,.005):
                    delta=correction(error,tc['basis_'+basis],mt,dose*tc['p3_norm_median'])
                    out,edited=forward(delta)
                    rr[f'{basis}_{method}_{dose}']=measure(out,float(edited.realized.norm()))
            anti=correction(error,tc['basis_'+basis],torch.as_tensor(state_response,dtype=torch.float32,device=agent.device),-.001*tc['p3_norm_median'])
            out,edited=forward(anti);rr[f'{basis}_causal_ridge_negative_0.001']=measure(out,float(edited.realized.norm()))
        records.append(rr)
    arrays=dict(p3=np.stack(p3s),candidate_ids=np.asarray(tuple(ids)),
                fresh_goal_encoding_maxabs=np.asarray([goal_difference[k] for k in ('visual','proprio')]))
    arrays.update({b+'_response':np.stack(responses[b]) for b in BASES})
    out=args.output/(row['key']+'.npz')
    if args.mode=='evaluate':
        real=np.stack([np.asarray(r['actual_encoded_native_cost_by_horizon']) for r in bank['candidates']])
        physical=np.stack([r['metrics']['goal_xy_distance'] for r in bank['candidates']])
        base_mse=np.stack([r['unsteered']['visual_mse'] for r in records]);conditions=[]
        for name in records[0]:
            mses=np.stack([r[name]['visual_mse'] for r in records]);costs=np.stack([r[name]['native_cost'] for r in records])
            arrays[name+'_visual_mse']=mses;arrays[name+'_native_cost']=costs
            conditions.append(dict(name=name,visual_mse=float(mses.mean()),first_horizon_mse=float(mses[:,0].mean()),
                mse_delta_vs_unsteered=float((mses-base_mse).mean()),first_horizon_delta_vs_unsteered=float((mses[:,0]-base_mse[:,0]).mean()),
                delivered_pooled_norm=float(np.mean([r[name]['delivered_pooled_norm'] for r in records])),
                ranking=ranking(costs[:,-1].reshape(-1),real[:,-1].reshape(-1),physical[:,-1])))
        write_json(out.with_suffix('.json'),dict(complete=True,key=row['key'],split=row['split'],conditions=conditions,
            seconds=time.monotonic()-start,native_forward_calls=calls,source_bank_sha256=row['sha256'],
            response_model_sha256=sha256(args.response_models/'response_models.npz'),futuretruthusedtoedit=False,
            edit_target_origin='unchanged TRAIN latent residual map; first-horizon P3 only',
            fresh_goal_encoding_difference=goal_difference,effective_fullprecision_saved_goal_exact=True,
            goal_in_forecast_or_response_computation=False))
    np.savez_compressed(out,**arrays)
    files=[out] if args.mode=='capture' else [out,out.with_suffix('.json')]
    print(json.dumps(dict(event='causal_response_'+args.mode,key=row['key'],seconds=time.monotonic()-start,native_forward_calls=calls)),flush=True)
    return [dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,key=row['key'],split=row['split']) for p in files]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('capture','fit','evaluate'))
    for name in ('bank','calibration','repo','checkpoint','output','response-models'):p.add_argument('--'+name,type=Path,required=name=='output')
    p.add_argument('--keys',nargs='+');p.add_argument('--capture-dirs',type=Path,nargs='+');p.add_argument('--near-plan',action='store_true');args=p.parse_args()
    if args.mode=='fit': return fit(args)
    rows=select_rows(json.loads((args.bank/'DONE.json').read_text()),args.mode,args.keys,args.near_plan)
    frozen=json.loads((args.calibration/'FROZEN.json').read_text())
    if not frozen['complete'] or sha256(args.calibration/'calibration.npz')!=frozen['calibration_sha256']:raise ValueError('Bad frozen calibration')
    models=None
    if args.mode=='evaluate':
        f=json.loads((args.response_models/'FROZEN.json').read_text())
        if not f['complete'] or sha256(args.response_models/'response_models.npz')!=f['response_models_sha256']:raise ValueError('Bad response model')
        models=dict(np.load(args.response_models/'response_models.npz',allow_pickle=False))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',dict(mode=args.mode,rows=rows,bases=BASES,rank=2,horizon=1,
        epsilon_ratio=.001,candidate_ids=ACTION_IDS if args.mode=='capture' else list(range(9 if args.near_plan else 7)),
        alpha=100.,pca_rank=16,doses=[.001,.005],script_sha256=sha256(__file__),calibration_sha256=frozen['calibration_sha256'],
        no_runtime_future_truth=True,no_held_study_access=True,
        goal_contract='Original saved float32 objective shared all arms; fresh encoding difference recorded. Goal unused by model unroll/residual target.'))
    from model_loader import load_headless
    torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(args.repo,args.output)
    wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm)
    c=dict(np.load(args.calibration/'calibration.npz',allow_pickle=False));start=time.monotonic();outputs=[]
    for row in rows:outputs.extend(run_source(args,row,wm,prep,cfg,c,models))
    if parameter_sha(wm)!=before:raise ValueError('Frozen weights changed')
    write_json(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,
        parameter_sha256=before,provenance=provenance,cem_calls=0))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
