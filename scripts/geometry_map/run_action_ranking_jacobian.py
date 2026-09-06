#!/usr/bin/env python3
"""Action-conditioned native-response metric mutation of a frozen residual edit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from run_action_ranking_pilot import (safe_rows,donor_plans,local_chord_delta,norm_sham,score,CHECKPOINT)
from run_action_path_curvature import ResidualCapture,ResidualReplacement
from capture_horizon_coordinates import exact,tensor_hash
from complete_cached_geometry import sha256,write_json

ARMS=('native_unhooked','unsteered_identity','ordinary_near_chord','jacobian_coordinates','norm_matched_sham')


def action_tangent(donors):
    odd=(donors[1::2].double()-donors[0::2].double())/4
    matrix=odd.flatten(2).permute(1,2,0).contiguous()
    u,s,_=torch.linalg.svd(matrix,full_matrices=False)
    active=(s>s[:,:1]*1e-4)&(s>1e-8)
    return u*active[:,None,:],s,active


def metric_vector(prediction,alpha,sum_all):
    parts=[]
    for k,weight in (('visual',1.),('proprio',alpha)):
        full=prediction[k];value=full if sum_all else full[-1:]
        parts.append(value.double().cpu().transpose(0,1).flatten(1)*(weight/full[0,0].numel())**.5)
    return torch.cat(parts,1)


def whiten_delta(delta,u,gram):
    flat=delta.double().flatten(1);coeff=torch.einsum('bnk,bn->bk',u,flat)
    tangent=torch.einsum('bnk,bk->bn',u,coeff);normal=flat-tangent
    eig,q=torch.linalg.eigh((gram+gram.transpose(1,2))/2)
    eig=eig.clamp_min(0);rank=(u.square().sum(1)>.5).sum(1).clamp_min(1)
    mean=eig.sum(1)/rank;damping=(mean*.01).clamp_min(1e-12)
    factors=((mean+damping)[:,None]/(eig+damping[:,None])).sqrt()
    transform=(q*factors[:,None,:])@q.transpose(1,2)
    newcoeff=torch.einsum('bij,bj->bi',transform,coeff)
    changed=normal+torch.einsum('bnk,bk->bn',u,newcoeff)
    norms=flat.norm(dim=1);newnorms=changed.norm(dim=1)
    if bool(((newnorms<1e-15)&(norms>0)).any()):raise ValueError('Nonzero candidate collapsed to zero')
    changed*=torch.where(norms>0,norms/newnorms.clamp_min(1e-15),torch.zeros_like(norms))[:,None]
    return changed.reshape_as(delta).to(delta),dict(eigenvalues=eig.tolist(),damping=damping.tolist(),
        tangent_fraction_of_candidate_l2=(tangent.norm(dim=1)/norms.clamp_min(1e-15)).tolist(),
        before_radial_rematch_norm=newnorms.tolist(),raw_budget=norms.tolist(),
        whitening_factors=factors.tolist(),metric_identified=(mean>1e-12).tolist())


@torch.no_grad()
def run_source(args,row,wm,prep,cfg):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from causal_response_transfer import restore_fullprecision_goal
    from intervene_head_spatial_mean import batch_context
    path=args.bank/row['path']
    if sha256(path)!=row['sha256']:raise ValueError('Source SHA before load failed')
    b=torch.load(path,map_location='cpu',weights_only=False)
    if b['row']['key']!=row['key'] or b['row']['split']!='development_external' or not b['same_state_actions_and_two_physical_replays_exact']:
        raise ValueError('Source development/physical identity mismatch')
    if len(b['candidates'])!=9 or any(np.asarray(v['states']).shape!=(31,7) for v in b['candidates']):
        raise ValueError('Missing nine actual30-action physical trajectories')
    start=time.monotonic();agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep)
    agent.set_goal(TensorDict(b['raw_goal'],batch_size=[]));restore_fullprecision_goal(agent,b['goal_encoded'])
    goal_sha={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')}
    z=batch_context(b['context'],9).to(agent.device);z_sha={k:tensor_hash(z[k]) for k in z.keys()}
    plan=b['normalized_actions'].transpose(0,1).contiguous().to(agent.device);saved=plan.clone()
    blocks=wm.model.predictor.predictor_blocks;native=wm.unroll(z.clone(),act_suffix=plan)
    with ResidualCapture(blocks) as capture:repeat=wm.unroll(z.clone(),act_suffix=plan)
    for k in ('visual','proprio'):exact(native[k],repeat[k],'native capture repeat '+k)
    recipient=capture.values[3,3];raw_donors,normalized_donors=donor_plans(b['raw_actions'],b['normalized_actions'])
    actualraw=prep.denormalize_actions(normalized_donors.reshape(8,9,30,2))
    tolerance=16*torch.finfo(plan.dtype).eps*max(1.,float(raw_donors.abs().max()),float(normalized_donors.abs().max()))
    torch.testing.assert_close(actualraw,raw_donors,rtol=0,atol=tolerance)
    captures=[]
    for donor in normalized_donors:
        with ResidualCapture(blocks) as cap:wm.unroll(z.clone(),act_suffix=donor.transpose(0,1).contiguous().to(agent.device))
        captures.append(cap.values[3,3])
    donors=torch.stack(captures);ordinary=local_chord_delta(donors,recipient);u,singular,active=action_tangent(donors)
    epsilon=(recipient.double().flatten(1).norm(dim=1)*1e-4).clamp_min(1e-6)
    jacobians=[];probe_norms=[]
    for axis in range(4):
        column=u[:,:,axis].reshape_as(recipient).to(recipient);values=[];norms=[]
        for sign in (-1,1):
            replacement=recipient+sign*epsilon.to(recipient)[:,None,None]*column
            norms.append((replacement-recipient).double().flatten(1).norm(dim=1).tolist())
            with ResidualReplacement(blocks[3],3,replacement.to(agent.device)):
                pred=wm.unroll(z.clone(),act_suffix=plan)
            for k in ('visual','proprio'):exact(pred[k][:3],native[k][:3],'Jacobian probe pre-H3 prefix '+k)
            values.append(metric_vector(pred,float(agent.objective.alpha),bool(agent.objective.sum_all_diffs)))
        jacobians.append((values[1]-values[0])/(2*epsilon[:,None]));probe_norms.append(norms)
    jac=torch.stack(jacobians,2)
    if not torch.isfinite(jac).all():raise ValueError('Nonfinite native finite-difference Jacobian')
    gram=jac.transpose(1,2)@jac;mutant,metric=whiten_delta(ordinary,u,gram);sham=norm_sham(mutant,row['source_id'])
    deltas={'unsteered_identity':torch.zeros_like(ordinary),'ordinary_near_chord':ordinary,
        'jacobian_coordinates':mutant,'norm_matched_sham':sham}
    ordinary_norm=ordinary.double().flatten(1).norm(dim=1)
    rounded={name:((recipient+d)-recipient).double().flatten(1).norm(dim=1) for name,d in deltas.items()}
    for name in ('jacobian_coordinates','norm_matched_sham'):
        torch.testing.assert_close(deltas[name].double().flatten(1).norm(dim=1),ordinary_norm,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(rounded[name],rounded['ordinary_near_chord'],rtol=1e-4,atol=1e-4)
    predictions={'native_unhooked':native}
    for name,delta in deltas.items():
        with ResidualReplacement(blocks[3],3,(recipient+delta).to(agent.device)):
            pred=wm.unroll(z.clone(),act_suffix=plan)
        for k in ('visual','proprio'):
            exact(pred[k][:3],native[k][:3],'fixed pre-H3 '+name+' '+k)
            if name=='unsteered_identity':exact(pred[k],native[k],'exact-self '+k)
        predictions[name]=pred
    costs={name:agent.objective(pred,plan).reshape(9).cpu().numpy() for name,pred in predictions.items()}
    exact(costs['native_unhooked'],costs['unsteered_identity'],'native objective exact self')
    # No physical outcomes enter candidate construction or Jacobian evaluation.
    states=np.stack([np.asarray(v['states']) for v in b['candidates']]);scores={name:score(cost,states,b['goal_state']) for name,cost in costs.items()}
    rows=[]
    for name in ARMS:
        value=scores[name];r=value['cost_vs_physical_distance_spearman'];base=scores['native_unhooked'];ordinary_score=scores['ordinary_near_chord']
        rows.append(dict(arm=name,**value,native_goal_cost_by_action=costs[name].tolist(),
            spearman_delta_vs_native=None if r is None or base['cost_vs_physical_distance_spearman'] is None else r-base['cost_vs_physical_distance_spearman'],
            spearman_delta_vs_ordinary=None if r is None or ordinary_score['cost_vs_physical_distance_spearman'] is None else r-ordinary_score['cost_vs_physical_distance_spearman'],
            selected_actual_progress_delta_vs_native_px=value['selected_actual_progress_px']-base['selected_actual_progress_px'],
            selected_actual_progress_delta_vs_ordinary_px=value['selected_actual_progress_px']-ordinary_score['selected_actual_progress_px'],
            selected_action_changed_vs_native=value['selected_action_index']!=base['selected_action_index']))
    exact(plan,saved,'immutable same9 actions')
    if z_sha!={k:tensor_hash(z[k]) for k in z.keys()} or goal_sha!={k:tensor_hash(agent.goal_state_enc[k]) for k in goal_sha}:
        raise ValueError('Context/goal modified')
    pt=args.output/(row['key']+'.pt')
    torch.save(dict(source=row,raw_actions=b['raw_actions'],normalized_actions=b['normalized_actions'],
        p3_h3_native=recipient,p3_h3_donors=donors,action_tangent_basis=u.float(),action_tangent_singular_values=singular,
        native_finite_difference_jacobian=jac.float(),native_response_metric=gram,deltas=deltas,
        predictions={name:{k:v[k].float().cpu() for k in ('visual','proprio')} for name,v in predictions.items() if name!='unsteered_identity'},
        physical_states=states,goal_state=b['goal_state'],costs=costs,probe_epsilon=epsilon),pt)
    seconds=time.monotonic()-start;report=pt.with_suffix('.json')
    write_json(report,dict(complete=True,key=row['key'],source=row,rows=rows,seconds=seconds,identity_exact=True,
        unchanged_context_goal_actions=True,physical_outcomes_used_after_all_frozen_forwards=True,
        action_tangent_rank=active.sum(1).tolist(),action_tangent_singular_values=singular.tolist(),
        local_metric=metric,jacobian_probe_epsilon=epsilon.tolist(),jacobian_probe_actual_rounded_norms=probe_norms,
        native_response_objective_alpha=float(agent.objective.alpha),native_response_sum_all_diffs=bool(agent.objective.sum_all_diffs),
        requested_same_raw_budget=ordinary_norm.tolist(),actual_rounded_norms={k:v.tolist() for k,v in rounded.items()},
        mutation_delta_cosine_to_ordinary=torch.nn.functional.cosine_similarity(mutant.flatten(1),ordinary.flatten(1)).tolist(),
        tensor_sha256=sha256(pt),tensor_bytes=pt.stat().st_size,
        limitations=['Four already-seen development states, nine same-action plans each; notCEM300 or confirmation',
            'Jacobian consequence means actual native predicted visual/proprio sensitivity, not true physical dynamics',
            'Coordinate mutation is model-local; actual physical trajectories used only for fixed-bank scoring',
            'Same raw residualL2 and newest256 support/H3; tangent anisotropy changed, not output latent',
            'No outcome-guided rank/damping/dose selection; all four states and failures retained']))
    print(json.dumps(dict(event='jacobian_ranking_source_complete',key=row['key'],seconds=seconds,
        mutant_spearman_delta_vs_ordinary=rows[3]['spearman_delta_vs_ordinary'],mutant_progress_delta=rows[3]['selected_actual_progress_delta_vs_native_px'])),flush=True)
    return [dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,key=row['key']) for p in (pt,report)],seconds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','repo','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();rows=safe_rows(json.loads((a.bank/'DONE.json').read_text()))
    if sha256(a.checkpoint)!=CHECKPOINT:raise ValueError('Pinned checkpoint mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',dict(sources=rows,arms=ARMS,site='P3 residual newest256',imagined_step=3,
        ordinary_candidate='eight symmetric near-chord donor residual mean, offsets±2delta on allfour original action directions',
        tangent='raw-orthonormal SVD of four local odd P3/H3 action responses; rankthreshold max(1e-4largest,1e-8), maximum4',
        jacobian='actual frozen native ±epsilon basis residual probes atH3; epsilon1e-4recipient rawfullspatialL2',
        metric='J^TJ of native predicted visual/proprio vector weighted exactly nativeL2 per-feature/alpha and horizon aggregation',
        whitening='eigenbasis inverse-square-root, dampingmax(.01meanEigen,1e-12), trace-normalized scale',
        mutation='retain candidate orthogonal complement, whiten tangent coefficients, then rematch original rawL2',
        sham='same deterministic seed9050700+episode fullspatial Gaussian, same rawL2 as ordinary/mutant',
        primary='within-state native-cost vs physical-endpoint Spearman and selected actual30-step progress',
        no_physical_labels_online=True,no_held=True,cem_calls=0,simulator_calls=0,max_seconds=300,
        checkpoint_sha256=CHECKPOINT,script_sha256=sha256(__file__),parent_candidate_code_sha256=sha256(Path(__file__).with_name('run_action_ranking_pilot.py'))))
    from model_loader import load_headless
    from collect_pusht_bank import config
    from capture_pusht_calibration import parameter_sha
    torch.set_num_threads(2);torch.manual_seed(90507);start=time.monotonic()
    try:
        cfg=config(a.repo,a.output);wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
        wm.eval().requires_grad_(False);before=parameter_sha(wm);outputs=[]
        for i,row in enumerate(rows):
            if time.monotonic()-start>300:raise RuntimeError('Pilot budget exhausted before nextstate')
            entries,seconds=run_source(a,row,wm,prep,cfg);outputs.extend(entries)
            if i==0:
                write_json(a.output/'CANARY.json',dict(complete=True,guards_passed=True,seconds=seconds,estimated4state_seconds=seconds*4,
                    outcomes_not_used_as_gate=True,within300seconds=seconds*4<=300))
                if seconds*4>300:raise RuntimeError('Timing-only canary exceeds300seconds')
        if parameter_sha(wm)!=before:raise ValueError('Native weights changed')
        for name in ('protocol.json','CANARY.json'):
            path=a.output/name;outputs.append(dict(path=name,sha256=sha256(path),bytes=path.stat().st_size))
        write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,
            states=4,actions_per_state=9,weights_unchanged=True,parameter_sha256=before,provenance=provenance,cem_calls=0,simulator_calls=0))
    except Exception as exc:
        write_json(a.output/'FAILED.json',dict(complete=False,error=repr(exc),seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
