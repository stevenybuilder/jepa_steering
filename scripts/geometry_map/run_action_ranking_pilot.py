#!/usr/bin/env python3
"""Four seen Push states, nine SAME cached physical plans: native-cost ranking.

One predeclared P3/H3 local-chord smoother versus norm-matched random sham.
This re-ranks a fixed small bank, not a new CEM run or physical steering study.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from complete_cached_geometry import sha256,write_json
from capture_horizon_coordinates import exact,tensor_hash
from run_action_path_curvature import ResidualCapture,ResidualReplacement

KEYS=tuple(f'near-dev-{e:03d}' for e in range(4))
CHECKPOINT='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
ARMS=('native_unhooked','unsteered_identity','geometry_near_chord','norm_matched_sham')


def safe_rows(receipt):
    rows=sorted([r for r in receipt.get('outputs',[]) if r['path'].endswith('.pt')],key=lambda r:r['key'])
    if not receipt.get('complete') or tuple(r['key'] for r in rows)!=KEYS or any(r['split']!='development_external' for r in rows):
        raise ValueError('Only four complete original development near-plan banks before load')
    return rows


def rank_correlation(a,b):
    """Spearman with average ties; constant vectors explicitly unavailable."""
    def ranks(x):
        x=np.asarray(x,dtype=float);order=np.argsort(x,kind='stable');result=np.empty(len(x));i=0
        while i<len(x):
            j=i+1
            while j<len(x) and x[order[j]]==x[order[i]]:j+=1
            result[order[i:j]]=(i+j-1)/2.;i=j
        return result
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=(9,) or b.shape!=(9,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nine finite same-action scores required')
    aa,bb=ranks(a),ranks(b)
    if np.ptp(aa)==0 or np.ptp(bb)==0:return None
    return float(np.corrcoef(aa,bb)[0,1])


def donor_plans(raw,normalized):
    if raw.shape!=(9,30,2) or normalized.shape!=(9,6,10):raise ValueError('Original9 Push H6 plan bank required')
    raw_plans=[];plans=[]
    tolerance=8*torch.finfo(raw.dtype).eps*max(1.,float(raw.abs().max()))
    for pair in range(4):
        delta=raw[1+2*pair]-raw[0]
        if float((delta+raw[2+2*pair]-raw[0]).abs().max())>tolerance:raise ValueError('Saved pair not antithetic')
        ndelta=normalized[1+2*pair]-normalized[0]
        for sign in (-1,1):
            raw_plans.append(raw+(sign*2)*delta[None])
            plans.append(normalized+(sign*2)*ndelta[None])
    return torch.stack(raw_plans),torch.stack(plans)


def local_chord_delta(donors,recipient):
    if donors.shape!=(8,*recipient.shape) or recipient.shape!=(9,256,400):
        raise ValueError('Eight symmetric same-action donor residuals, nine recipients, full256x400 support')
    return donors.double().mean(0).to(recipient)-recipient


def norm_sham(delta,episode):
    g=torch.Generator(device='cpu').manual_seed(9050700+episode)
    random=torch.randn(delta.shape,generator=g,dtype=torch.float64)
    norms=delta.double().flatten(1).norm(dim=1)
    random*= (norms/random.flatten(1).norm(dim=1))[:,None,None]
    return random.to(delta)


def score(cost,states,goal):
    if states.shape!=(9,31,7):raise ValueError('Require native31state/7coordinate fixed30-action trajectories for ALL9 plans')
    if not np.array_equal(states[:,0],np.broadcast_to(states[0,0],states[:,0].shape)):
        raise ValueError('Physical plans do not share exact initial state')
    goal=np.asarray(goal);initial=float(np.linalg.norm(states[0,0,:4]-goal[:4]))
    distances=np.linalg.norm(states[:,-1,:4]-goal[:4],axis=1)
    progress=initial-distances;selected=int(np.argmin(cost))
    return dict(cost_vs_physical_distance_spearman=rank_correlation(cost,distances),
        negative_cost_vs_physical_progress_spearman=rank_correlation(-np.asarray(cost),progress),
        selected_action_index=selected,selected_actual_progress_px=float(progress[selected]),
        selected_actual_goal_xy_distance_px=float(distances[selected]),initial_goal_xy_distance_px=initial,
        physical_progress_px_by_action=progress.tolist(),physical_goal_xy_distance_px_by_action=distances.tolist(),
        original_native_planner_action_index=0,original_native_planner_actual_progress_px=float(progress[0]),
        original_native_planner_actual_goal_xy_distance_px=float(distances[0]),
        fixed_bank_physical_oracle_index=int(np.argmin(distances)))


@torch.no_grad()
def run_source(args,row,wm,prep,cfg):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from causal_response_transfer import restore_fullprecision_goal
    from intervene_head_spatial_mean import batch_context
    path=args.bank/row['path']
    if sha256(path)!=row['sha256']:raise ValueError('Source SHA failed before tensor load')
    b=torch.load(path,map_location='cpu',weights_only=False)
    if b['row']['split']!='development_external' or b['row']['key']!=row['key'] or not b['same_state_actions_and_two_physical_replays_exact']:
        raise ValueError('Source split/physical replay identity failed')
    if len(b['candidates'])!=9 or any(np.asarray(x['states']).shape!=(31,7) for x in b['candidates']):
        raise ValueError('Missing actual physical candidate outcomes; no invented rollout')
    start=time.monotonic();agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep)
    agent.set_goal(TensorDict(b['raw_goal'],batch_size=[]));goal_diff=restore_fullprecision_goal(agent,b['goal_encoded'])
    goal_sha={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')}
    z=batch_context(b['context'],9).to(agent.device);z_sha={k:tensor_hash(z[k]) for k in z.keys()}
    plan=b['normalized_actions'].transpose(0,1).contiguous().to(agent.device);saved=plan.clone()
    blocks=wm.model.predictor.predictor_blocks
    native=wm.unroll(z.clone(),act_suffix=plan)
    with ResidualCapture(blocks) as base_capture:observed=wm.unroll(z.clone(),act_suffix=plan)
    for k in ('visual','proprio'):exact(native[k],observed[k],'read-only capture '+k)
    recipient=base_capture.values[3,3]
    raw_donors,normalized_donors=donor_plans(b['raw_actions'],b['normalized_actions'])
    reconstructed=prep.denormalize_actions(normalized_donors.reshape(8,9,30,2))
    roundoff=float((reconstructed-raw_donors).abs().max())
    tolerance=16*torch.finfo(plan.dtype).eps*max(1.,float(raw_donors.abs().max()),float(normalized_donors.abs().max()))
    torch.testing.assert_close(reconstructed,raw_donors,rtol=0,atol=tolerance)
    donor_residuals=[]
    for donor in normalized_donors:
        with ResidualCapture(blocks) as captured:
            wm.unroll(z.clone(),act_suffix=donor.transpose(0,1).contiguous().to(agent.device))
        donor_residuals.append(captured.values[3,3])
    donors=torch.stack(donor_residuals);delta=local_chord_delta(donors,recipient);sham=norm_sham(delta,row['source_id'])
    requested=delta.double().flatten(1).norm(dim=1)
    semantic_actual=((recipient+delta)-recipient).double().flatten(1).norm(dim=1)
    sham_actual=((recipient+sham)-recipient).double().flatten(1).norm(dim=1)
    torch.testing.assert_close(sham.double().flatten(1).norm(dim=1),requested,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(sham_actual,semantic_actual,rtol=1e-4,atol=1e-4)
    predictions={'native_unhooked':native}
    for arm,edit in (('unsteered_identity',torch.zeros_like(delta)),('geometry_near_chord',delta),('norm_matched_sham',sham)):
        with ResidualReplacement(blocks[3],3,(recipient+edit).to(agent.device)):
            predictions[arm]=wm.unroll(z.clone(),act_suffix=plan)
        for k in ('visual','proprio'):
            exact(predictions[arm][k][:3],native[k][:3],'pre-H3 unchanged '+arm+' '+k)
            if arm=='unsteered_identity':exact(predictions[arm][k],native[k],'identity full unroll '+k)
    costs={arm:agent.objective(pred,plan).reshape(9).cpu().numpy() for arm,pred in predictions.items()}
    exact(costs['native_unhooked'],costs['unsteered_identity'],'native objective exact identity')
    # Ground-truth outcomes are used only here, after every frozen model condition.
    states=np.stack([np.asarray(x['states']) for x in b['candidates']]);scores={arm:score(cost,states,b['goal_state']) for arm,cost in costs.items()}
    baseline=scores['native_unhooked'];rows=[]
    for arm in ARMS:
        s=scores[arm];rho=s['cost_vs_physical_distance_spearman'];brho=baseline['cost_vs_physical_distance_spearman']
        rows.append(dict(arm=arm,**s,native_goal_cost_by_action=costs[arm].tolist(),
            spearman_delta_vs_native=None if rho is None or brho is None else rho-brho,
            selected_actual_progress_delta_vs_native_px=s['selected_actual_progress_px']-baseline['selected_actual_progress_px'],
            selected_action_changed_vs_native=s['selected_action_index']!=baseline['selected_action_index']))
    exact(plan,saved,'original9 actions unchanged')
    if z_sha!={k:tensor_hash(z[k]) for k in z.keys()} or goal_sha!={k:tensor_hash(agent.goal_state_enc[k]) for k in goal_sha}:
        raise ValueError('Context/goal changed')
    pt=args.output/(row['key']+'.pt')
    torch.save(dict(source=row,raw_actions=b['raw_actions'],normalized_actions=b['normalized_actions'],
        new_model_only_raw_donor_actions=raw_donors,new_model_only_normalized_donor_actions=normalized_donors,
        p3_h3_native=recipient,p3_h3_donors=donors,geometry_delta=delta,matched_sham_delta=sham,
        predictions={arm:{k:pred[k].float().cpu() for k in ('visual','proprio')} for arm,pred in predictions.items() if arm!='unsteered_identity'},
        physical_states=states,goal_state=b['goal_state'],costs=costs),pt)
    report=pt.with_suffix('.json');seconds=time.monotonic()-start
    write_json(report,dict(complete=True,key=row['key'],source=row,source_path=str(path),rows=rows,seconds=seconds,
        identity_exact=True,weights_frozen=True,all_actions_context_goals_unchanged=True,actual_truth_used_only_after_model_conditions=True,
        native_objective_sum_all_diffs=bool(agent.objective.sum_all_diffs),native_objective_alpha=float(agent.objective.alpha),
        fresh_vs_saved_goal_encoding_maxabs=goal_diff,effective_original_goal_sha256=goal_sha,
        original_single_vs_batch_visual_maxabs=float((native['visual'][1:,0].cpu()-b['candidates'][0]['predicted_visual']).abs().max()),
        norm_matching=dict(requested_raw_delta_l2=requested.tolist(),actual_rounded_geometry_l2=semantic_actual.tolist(),
            actual_rounded_sham_l2=sham_actual.tolist(),actual_rounded_max_difference=float((semantic_actual-sham_actual).abs().max())),
        raw_action_roundtrip_maxabs=roundoff,tensor_sha256=sha256(pt),tensor_bytes=pt.stat().st_size,
        candidate_definition='P3/H3 fullspatial mean of eight symmetric donor states: allfour original directions at±2delta; no cap or learned readout',
        limitations=['Four already-seen development states; not36 independent episodes or untouched confirmation',
            'Fixed9-action bank reranking, not nativeCEM300 reoptimization or closed-loop physical efficacy',
            'Original native planner chose action0; native bankargmin retained as separate comparator',
            'Selected physical outcomes reuse exact original30-action cached trajectories; no newphysics',
            'New donor paths have model predictions only; no outcome-guided candidate/dose adaptation']))
    print(json.dumps(dict(event='ranking_source_complete',key=row['key'],seconds=seconds,
        geometry_spearman_delta=rows[2]['spearman_delta_vs_native'],geometry_progress_delta=rows[2]['selected_actual_progress_delta_vs_native_px'])),flush=True)
    return [dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,key=row['key']) for p in (pt,report)],seconds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','repo','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();rows=safe_rows(json.loads((a.bank/'DONE.json').read_text()))
    if sha256(a.checkpoint)!=CHECKPOINT:raise ValueError('Wrong pinned Push checkpoint')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',dict(sources=rows,arms=ARMS,site='P3 residual newest256',imagined_step=3,
        action_bank_size=9,source_states=4,original_native_planner_action_index=0,
        geometry='equal average allfour original antithetic directions, offsets±2delta, then residual difference vs recipient',
        donor_radius=4,donor_t=[-.5,.5],native_raw_direction_rms=.01,effective_raw_offset_rms=.02,
        sham='seed9050700+sourceID random fullspatial Gaussian, per-action raw-L2 matched; rounded norm checked',
        primary=['within-state Spearman(native full objective cost,actual endpoint4D XY distance)',
                 'selected same-bank actual30-step progress versus native and sham'],
        secondary=['selected action change','comparison to original CEM selected action0'],
        no_new_cem_or_physics=True,no_held_access=True,canary='first fixed state×4 runtime estimate≤300s, identity/norm guards; no outcome gate',
        checkpoint_sha256=CHECKPOINT,source_code_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ('run_action_ranking_pilot.py','run_action_path_curvature.py','causal_response_transfer.py')}))
    from model_loader import load_headless
    from collect_pusht_bank import config
    from capture_pusht_calibration import parameter_sha
    torch.set_num_threads(2);torch.manual_seed(90507);start=time.monotonic()
    try:
        cfg=config(a.repo,a.output);wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
        wm.eval().requires_grad_(False);before=parameter_sha(wm);outputs=[]
        for i,row in enumerate(rows):
            if time.monotonic()-start>300:raise RuntimeError('Bounded pilot budget reached')
            entries,seconds=run_source(a,row,wm,prep,cfg);outputs.extend(entries)
            if i==0:
                write_json(a.output/'CANARY.json',dict(complete=True,guards_passed=True,seconds=seconds,estimated4state_seconds=seconds*4,
                    outcomes_not_used_as_gate=True,within300seconds=seconds*4<=300))
                if seconds*4>300:raise RuntimeError('Timing canary exceeds300seconds')
        if parameter_sha(wm)!=before:raise ValueError('Native weights changed')
        for name in ('protocol.json','CANARY.json'):
            path=a.output/name;outputs.append(dict(path=name,sha256=sha256(path),bytes=path.stat().st_size))
        write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,
            independent_source_states=4,actions_per_state=9,weights_unchanged=True,parameter_sha256=before,
            provenance=provenance,cem_calls=0,simulator_calls=0,held_access=False))
    except Exception as exc:
        write_json(a.output/'FAILED.json',dict(complete=False,error=repr(exc),seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
