#!/usr/bin/env python3
"""Unseen interior action donors patched into fixed central-action recipients."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from action_path_geometry import interior_estimates
from run_action_path_curvature import ResidualCapture, ResidualReplacement, METHODS, HORIZONS, tensor_mse_by_horizon, interpolation_diagnostics

TARGETS=(-.25,.25)
CONDITIONS=('exact_self_patch','actual_donor_patch')+METHODS
BATCH=len(CONDITIONS)


def target_actions(raw_actions,pair,radius,t):
    if pair not in range(4) or radius not in (1,4) or t not in TARGETS or raw_actions.shape!=(9,30,2):
        raise ValueError('Only frozen original64 interior targets')
    return raw_actions[0]+(radius*t)*(raw_actions[1+2*pair]-raw_actions[0])


def recipient_replacements(native,donor,estimates):
    if set(estimates)!=set(METHODS):raise ValueError('Exactly six frozen estimators required')
    return torch.stack([native,donor]+[estimates[m] for m in METHODS])


@torch.no_grad()
def source(args,row,wm,prep,cfg,paths,canary):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from capture_horizon_coordinates import exact
    from causal_response_transfer import restore_fullprecision_goal
    from intervene_head_spatial_mean import batch_context
    from complete_cached_geometry import sha256,write_json
    bank_path=args.bank/row['path']
    if sha256(bank_path)!=row['sha256']:raise ValueError('Near-bank SHA mismatch before loading')
    bank=torch.load(bank_path,map_location='cpu',weights_only=False)
    if bank['row']['split']!='development_external':raise ValueError('Held inputs forbidden')
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep)
    agent.set_goal(TensorDict(bank['raw_goal'],batch_size=[]));goal_diff=restore_fullprecision_goal(agent,bank['goal_encoded'])
    z=batch_context(bank['context'],BATCH).to(agent.device)
    central_plan=bank['normalized_actions'][0,:,None].repeat(1,BATCH,1).contiguous().to(agent.device)
    fixed_plan=central_plan.clone();blocks=wm.model.predictor.predictor_blocks
    baseline=wm.unroll(z.clone(),act_suffix=central_plan)
    with ResidualCapture(blocks) as central:
        native_repeat=wm.unroll(z.clone(),act_suffix=central_plan)
    for k in ('visual','proprio'):exact(baseline[k],native_repeat[k],'native batch8 repeat '+k)
    base_visual=baseline['visual'][1:].float().cpu();actual_center=bank['candidates'][0]['actual_visual'][:,None].expand_as(base_visual)
    entries=[]
    for pair in range(4):
        for radius in (1,4):
            old_name=f"{row['key']}-pair{pair}-radius{radius}.pt";old_row=paths[old_name];old_path=args.curvature/old_name
            if sha256(old_path)!=old_row['sha256']:raise ValueError('Frozen noncentral donor SHA mismatch')
            prior=torch.load(old_path,map_location='cpu',weights_only=False)
            if prior['source']['source_sha256']!=row['sha256']:raise ValueError('Path and original-state provenance differ')
            for t in TARGETS:
                start=time.monotonic();raw=target_actions(bank['raw_actions'],pair,radius,t)
                norm=prep.normalize_actions(raw.reshape(6,5,2)).reshape(6,10)
                donor_plan=norm[:,None].repeat(1,BATCH,1).contiguous().to(agent.device)
                with ResidualCapture(blocks) as target:
                    donor_natural=wm.unroll(z.clone(),act_suffix=donor_plan)
                target_repeat=wm.unroll(z.clone(),act_suffix=donor_plan)
                for k in ('visual','proprio'):exact(donor_natural[k],target_repeat[k],'natural interior donor repeat '+k)
                reports=[];forecasts=[];proprio=[];all_replacements=[]
                for j,horizon in enumerate(HORIZONS):
                    # Original five-point capture retained; fit consumes only four nonzero donors.
                    noncentral=prior['residual_samples'][3+j][[0,1,3,4]]
                    estimates=interior_estimates(noncentral,t)
                    donor=target.values[3,horizon][0]
                    replacements=recipient_replacements(central.values[3,horizon][0],donor,estimates)
                    with ResidualReplacement(blocks[3],horizon,replacements.to(agent.device)):
                        result=wm.unroll(z.clone(),act_suffix=central_plan)
                    for k in ('visual','proprio'):
                        exact(result[k][:,0],baseline[k][:,0],'exact central recipient self patch '+k)
                        exact(result[k][:horizon],baseline[k][:horizon],'recipient untouched preceding horizons '+k)
                    visual=result['visual'][1:].float().cpu()
                    oracle=visual[:,1:2].expand_as(visual)
                    oracle_error=tensor_mse_by_horizon(visual,oracle)
                    native_error=tensor_mse_by_horizon(visual,base_visual)
                    actual_center_error=tensor_mse_by_horizon(visual,actual_center)
                    costs=agent.objective(result,central_plan,keepdims=True).float().cpu()
                    methods=[]
                    for i,name in enumerate(METHODS,2):
                        methods.append(dict(method=name,**interpolation_diagnostics(estimates[name],noncentral,donor),
                            downstream_oracle_patch_mse_by_horizon=oracle_error[:,i].tolist(),
                            downstream_oracle_patch_mse_after_patch=float(oracle_error[horizon-1:,i].mean()),
                            downstream_native_recipient_mse_by_horizon=native_error[:,i].tolist(),
                            actual_central_action_future_mse_by_horizon=actual_center_error[:,i].tolist(),
                            native_central_recipient_goal_cost_by_horizon=costs[:,i].reshape(-1).tolist()))
                    reports.append(dict(block=3,intervention_horizon=horizon,methods=methods,
                        actual_donor_patch_vs_native_mse_by_horizon=native_error[:,1].tolist(),
                        actual_donor_patch_vs_native_mse_after_patch=float(native_error[horizon-1:,1].mean()),
                        exact_self_patch=True,oracle_donor_not_used_in_any_fit=True))
                    forecasts.append(visual);proprio.append(result['proprio'][1:].float().cpu());all_replacements.append(replacements)
                exact(central_plan,fixed_plan,'unchanged central recipient actions across all conditions')
                label='minus' if t<0 else 'plus';name=f"{row['key']}-pair{pair}-radius{radius}-{label}025"
                pt=args.output/(name+'.pt')
                meta=dict(key=row['key'],pair=pair,radius=radius,target_t=t,source_sha256=row['sha256'],
                    frozen_noncentral_path_sha256=old_row['sha256'],split='development_external',batch=BATCH,
                    goal_fullprecision_shared=True,fresh_goal_encoding_difference=goal_diff,
                    central_raw_step=0,donor_action_physics_available_in_this_file=False,
                    donor_activation_is_oracle_causal_reference_not_physical_truth=True)
                torch.save(dict(complete=True,meta=meta,raw_target_actions=raw,normalized_target_actions=norm,
                    conditions=CONDITIONS,horizons=HORIZONS,replacements=torch.stack(all_replacements),
                    patched_visual=torch.stack(forecasts),patched_proprio=torch.stack(proprio),
                    natural_donor_visual=donor_natural['visual'][1:,0].float().cpu(),
                    native_central_visual=base_visual[:,0],actual_central_visual=actual_center[:,0]),pt)
                torch.cuda.synchronize();seconds=time.monotonic()-start
                report=pt.with_suffix('.json')
                write_json(report,dict(complete=True,meta=meta,rows=reports,seconds=seconds,
                    full_tensor_sha256=sha256(pt),full_tensor_bytes=pt.stat().st_size,
                    limitations=['Observed interior donor activation used only as oracle patch and scoring target, never estimator fit',
                        'All edited recipients retain central actions; donor-patch output is not native donor-action dynamics',
                        'Central-action physical truth is not donor-action truth',
                        'Four states and dependent directions; exploratory causal interchange, not steering success']))
                for p in (pt,report):entries.append(dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,key=row['key']))
                print(json.dumps(dict(event='interior_interchange_complete',key=row['key'],pair=pair,radius=radius,t=t,seconds=seconds)),flush=True)
                if canary:
                    write_json(args.output/'CANARY.json',dict(complete=True,seconds=seconds,projected64_seconds=seconds*64,
                        self_and_repeat_exact=True,within900=seconds*64<=900,outcomes_not_used_for_gate=True))
                    if seconds*64>900:raise RuntimeError('First target timing exceeds15minute projection; no next target')
                    canary=False
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bank','curvature','repo','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    from complete_cached_geometry import sha256,write_json
    from capture_pusht_calibration import parameter_sha
    from collect_pusht_bank import config
    from model_loader import load_headless
    d=json.loads((args.bank/'DONE.json').read_text());curves=json.loads((args.curvature/'DONE.json').read_text())
    rows=sorted([r for r in d['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['key'])
    if not d['complete'] or not curves['complete'] or [r['key'] for r in rows]!=[f'near-dev-{i:03}' for i in range(4)] or any(r['split']!='development_external' for r in rows):raise ValueError('Wrong frozen development panel')
    if sha256(args.checkpoint)!='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb':raise ValueError('Wrong checkpoint')
    paths={r['path']:r for r in curves['outputs'] if r['path'].endswith('.pt')}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',dict(rows=rows,curvature_done_sha256=sha256(args.curvature/'DONE.json'),
        targets=TARGETS,pairs=list(range(4)),radii=[1,4],blocks=[3],horizons=HORIZONS,conditions=CONDITIONS,
        script_sha256=sha256(__file__),helper_sha256=sha256(Path(__file__).with_name('action_path_geometry.py')),
        fit_inputs=[-1,-.5,.5,1],fit_unchanged=True,actual_target_activation_not_fit=True,
        causal_reference='Actual held-out target donor residual patched into same central-action recipient',
        primary='P3/H3 oracle-patched fullspatial output fidelity at/after intervention, each radius separately',
        batch_policy='All8conditions share contiguous central plan; target natural batch8 repeated identical actions',
        budget_gate='Firsttarget projected64≤900seconds, never outcome gate',cem_calls=0,simulator_calls=0))
    torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(args.repo,args.output)
    wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);start=time.monotonic();outputs=[]
    for i,row in enumerate(rows):outputs.extend(source(args,row,wm,prep,cfg,paths,i==0))
    if parameter_sha(wm)!=before:raise ValueError('Parameters changed')
    write_json(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,
        parameter_sha256=before,provenance=provenance,targets=64,sites_per_target=3,methods=6,cem_calls=0,simulator_calls=0))


if __name__=='__main__':main()
