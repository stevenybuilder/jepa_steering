#!/usr/bin/env python3
"""PCA64 natural-spline residual path pilot on fixed Push development actions.

Adaptation of Manifold Steering A.3/A.5/A.6: the known raw-action path parameter
is a local one-dimensional coordinate, not an inferred global physical concept.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from run_action_ranking_pilot import safe_rows, donor_plans, score, CHECKPOINT
from run_action_path_curvature import ResidualCapture, ResidualReplacement
from capture_horizon_coordinates import exact, tensor_hash
from complete_cached_geometry import sha256, write_json

KNOTS = np.array([-4., -2., 2., 4.])
QUERIES = np.array([-3., -1., 0., 1., 3.])


def natural_weights(query):
    """Natural cubic interpolation weights; never include the held-out midpoint."""
    x=KNOTS; y=np.eye(4); widths=np.diff(x)
    matrix=np.zeros((4,4)); rhs=np.zeros((4,4));matrix[0,0]=matrix[-1,-1]=1
    for i in (1,2):
        matrix[i,i-1]=widths[i-1];matrix[i,i]=2*(widths[i-1]+widths[i]);matrix[i,i+1]=widths[i]
        rhs[i]=6*((y[i+1]-y[i])/widths[i]-(y[i]-y[i-1])/widths[i-1])
    second=np.linalg.solve(matrix,rhs); result=[]
    for q in np.atleast_1d(query):
        if q<x[0] or q>x[-1]:raise ValueError('No unregistered extrapolation')
        i=min(int(np.searchsorted(x,q,side='right')-1),2);h=widths[i]
        a=(x[i+1]-q)/h;b=(q-x[i])/h
        result.append(a*y[i]+b*y[i+1]+((a**3-a)*second[i]+(b**3-b)*second[i+1])*h*h/6)
    return np.stack(result)


def pca_delta(recipient, target, mean, basis):
    """Only change the PCA coordinates; preserve the off-subspace residual."""
    return ((target-mean)@basis.T-(recipient-mean)@basis.T)@basis


def orthonormalize_rows(basis):
    """Remove float32 SVD roundoff without changing its selected linear span."""
    return torch.linalg.qr(basis.double().T,mode='reduced').Q.T


def local_linear_weights(query):
    result=[]
    for q in np.atleast_1d(query):
        if q<KNOTS[0] or q>KNOTS[-1]:raise ValueError('No extrapolation')
        i=min(int(np.searchsorted(KNOTS,q,side='right')-1),2)
        w=np.zeros(4);w[i+1]=(q-KNOTS[i])/(KNOTS[i+1]-KNOTS[i]);w[i]=1-w[i+1];result.append(w)
    return np.stack(result)


def check_path_coordinate(raw_center, raw_knots):
    # [4 knots,4 directions,9 candidate centers,30,2]
    delta=(raw_knots[-1]-raw_knots[0])/8
    reconstruction=raw_center[None,None]+torch.as_tensor(KNOTS).to(raw_knots)[:,None,None,None,None]*delta[None]
    tolerance=32*torch.finfo(raw_knots.dtype).eps*max(1.,float(raw_knots.abs().max()))
    torch.testing.assert_close(reconstruction,raw_knots,rtol=0,atol=tolerance)
    lengths=delta.double().flatten(2).norm(dim=2)
    if bool((lengths<1e-8).any()):raise ValueError('Intrinsic raw-action coordinate not identifiable: zero direction')
    return dict(coordinate='known signed amplitude of fixed nonzero raw-action direction, separately conditioned on state/candidate/direction',
        reconstruction_maxabs=float((reconstruction-raw_knots).abs().max()),direction_l2=lengths.tolist(),
        global_physical_intrinsic_coordinate_identified=False,local_input_coordinate_identified=True,
        no_action_clipping=True,physical_outcome_not_used=True)


@torch.no_grad()
def run_source(a,row,wm,prep,cfg):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from causal_response_transfer import restore_fullprecision_goal
    from intervene_head_spatial_mean import batch_context
    path=a.bank/row['path']
    if sha256(path)!=row['sha256']:raise ValueError('Input SHA before tensor access')
    b=torch.load(path,map_location='cpu',weights_only=False)
    if b['row']['split']!='development_external' or b['row']['key']!=row['key'] or not b['same_state_actions_and_two_physical_replays_exact']:
        raise ValueError('Wrong development/physical source')
    started=time.monotonic();agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep)
    agent.set_goal(TensorDict(b['raw_goal'],batch_size=[]));restore_fullprecision_goal(agent,b['goal_encoded'])
    z=batch_context(b['context'],9).to(agent.device);plan=b['normalized_actions'].transpose(0,1).contiguous().to(agent.device)
    hashes={k:tensor_hash(z[k]) for k in z.keys()};goal={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')}
    saved=plan.clone();blocks=wm.model.predictor.predictor_blocks
    native=wm.unroll(z.clone(),act_suffix=plan)
    with ResidualCapture(blocks) as capture:repeat=wm.unroll(z.clone(),act_suffix=plan)
    for k in ('visual','proprio'):exact(native[k],repeat[k],'unhooked/capture identity')
    recipient=capture.values[3,3];raw_near,near=donor_plans(b['raw_actions'],b['normalized_actions'])
    # donor_plans order is direction0 negative/positive, direction1 negative/positive...
    signed_raw=(raw_near[1::2]-raw_near[0::2])/4
    signed=(near[1::2]-near[0::2])/4
    raw_knots=b['raw_actions'][None,None]+torch.as_tensor(KNOTS).float()[:,None,None,None,None]*signed_raw[None]
    coordinate=check_path_coordinate(b['raw_actions'],raw_knots)
    knot_captures=[]
    for amplitude in KNOTS:
        direction_captures=[]
        for direction in range(4):
            condition=b['normalized_actions']+float(amplitude)*signed[direction]
            with ResidualCapture(blocks) as cap:wm.unroll(z.clone(),act_suffix=condition.transpose(0,1).contiguous().to(agent.device))
            direction_captures.append(cap.values[3,3])
        knot_captures.append(torch.stack(direction_captures))
    knots=torch.stack(knot_captures) # [4,4,9,256,400]
    # PCA uses only unedited model donor activations, never true future encodings.
    matrix=knots.flatten(0,2).flatten(1).to(agent.device);mean=matrix.mean(0)
    _,singular,vt=torch.linalg.svd(matrix-mean,full_matrices=False)
    if len(singular)<64 or float(singular[63])<=1e-8:raise ValueError('PCA64 is not supported by this fixed donor sample')
    basis=orthonormalize_rows(vt[:64].cpu());mean=mean.double().cpu();del matrix,vt
    torch.testing.assert_close(basis@basis.T,torch.eye(64,dtype=torch.float64),atol=2e-5,rtol=2e-5)
    flat=recipient.double().flatten(1);coordinates=(knots.double().flatten(3)-mean)@basis.T
    weights=torch.from_numpy(natural_weights([0])[0])
    spline_coordinate=torch.einsum('k,kdnc->dnc',weights,coordinates).mean(0)
    chord_coordinate=((coordinates[0]+coordinates[-1])/2).mean(0)
    local_coordinate=((coordinates[1]+coordinates[2])/2).mean(0)
    recipient_coordinate=(flat-mean)@basis.T
    deltas={'exact_self':torch.zeros_like(recipient),
        'pca64_endpoint_chord':((chord_coordinate-recipient_coordinate)@basis).reshape_as(recipient).float(),
        'pca64_local_chord':((local_coordinate-recipient_coordinate)@basis).reshape_as(recipient).float(),
        'pca64_natural_spline':((spline_coordinate-recipient_coordinate)@basis).reshape_as(recipient).float()}
    generator=torch.Generator().manual_seed(9050820+row['source_id']);sham_coeff=torch.randn(9,64,generator=generator,dtype=torch.float64)
    sham=sham_coeff@basis;target_norm=deltas['pca64_natural_spline'].double().flatten(1).norm(dim=1)
    sham*=target_norm[:,None]/sham.norm(dim=1,keepdim=True).clamp_min(1e-30)
    deltas['pca64_spline_norm_sham']=sham.reshape_as(recipient).float()
    # Arc length uses 150 intrinsic-coordinate intervals, matching A.5 numerical construction.
    grid=torch.from_numpy(natural_weights(np.linspace(-4,4,151)))
    curve=torch.einsum('tk,kdnc->tdnc',grid,coordinates)
    arc=(curve[1:]-curve[:-1]).norm(dim=-1).sum(0);endpoint=(coordinates[-1]-coordinates[0]).norm(dim=-1)
    responses={'native_unhooked':native};off_residual={};rounded={}
    for name,delta in deltas.items():
        replacement=recipient+delta;actual=(replacement-recipient).double().flatten(1)
        off=actual-(actual@basis.T)@basis
        off_residual[name]=dict(maxabs=float(off.abs().max()),l2=off.norm(dim=1).tolist(),
            mathematical_off_subspace_preserved=True,float32_reconstruction_rounding_only=True)
        if float(off.abs().max())>5e-4:raise ValueError('Off-PCA component changed beyond float32 reconstruction allowance')
        rounded[name]=actual.norm(dim=1).tolist()
        with ResidualReplacement(blocks[3],3,replacement.to(agent.device)):
            changed=wm.unroll(z.clone(),act_suffix=plan)
        for k in ('visual','proprio'):
            exact(changed[k][:3],native[k][:3],'unchanged pre-H3 prefix')
            if name=='exact_self':exact(changed[k],native[k],'exact self')
        responses[name]=changed
    torch.testing.assert_close(torch.tensor(rounded['pca64_natural_spline']),torch.tensor(rounded['pca64_spline_norm_sham']),rtol=1e-4,atol=1e-4)
    write_json(a.output/'CANARY_IDENTITY.json',dict(complete=True,key=row['key'],seconds=time.monotonic()-started,
        residual_shape=list(recipient.shape),PCA_shape=list(basis.shape),native_self_and_prefix_exact=True,
        projected_future_labels_used=False,rounded_norm_sham_passed=True,script_sha256=sha256(__file__)))
    print(json.dumps(dict(event='density_identity_canary_passed',key=row['key'],seconds=time.monotonic()-started)),flush=True)
    # Existing physical truths are accessed only after frozen path construction and forwards.
    states=np.stack([np.asarray(v['states']) for v in b['candidates']]);truth=torch.stack([v['actual_visual'] for v in b['candidates']],1)
    rows=[]
    for name,pred in responses.items():
        cost=agent.objective(pred,plan).reshape(9).cpu().numpy()
        mse=(pred['visual'][1:].cpu()-truth).double().square().flatten(2).mean(2)
        native_mse=(pred['visual'][1:].cpu()-native['visual'][1:].cpu()).double().square().flatten(2).mean(2)
        rows.append(dict(arm=name,**score(cost,states,b['goal_state']),native_goal_cost=cost.tolist(),
            actual_visual_mse_H1_H6_by_candidate=mse.tolist(),mean_actual_visual_mse_by_horizon=mse.mean(1).tolist(),
            native_forecast_restoration_mse_by_horizon=native_mse.mean(1).tolist()))
    # Interior causal traversal, with the SAME original action supplied throughout.
    # At nonzero coordinates there is no new physical path truth: scores still
    # compare the edited fixed-action forecast with its own cached physical future.
    traversal=[];traversal_coefficients=[]
    for direction in range(4):
        for query in QUERIES:
            sw=torch.from_numpy(natural_weights([query])[0]);lw=torch.from_numpy(local_linear_weights([query])[0])
            targets={'natural_spline':torch.einsum('k,knc->nc',sw,coordinates[:,direction]),
                'local_linear':torch.einsum('k,knc->nc',lw,coordinates[:,direction]),
                'endpoint_chord':((4-query)*coordinates[0,direction]+(query+4)*coordinates[-1,direction])/8}
            target_norm=((targets['natural_spline']-recipient_coordinate)@basis).norm(dim=1)
            random=torch.randn(9,64,generator=generator,dtype=torch.float64);random*=target_norm[:,None]/random.norm(dim=1,keepdim=True)
            targets['spline_norm_sham']=recipient_coordinate+random
            for method,target in targets.items():
                coefficients=target-recipient_coordinate;delta=(coefficients@basis).reshape_as(recipient).float()
                with ResidualReplacement(blocks[3],3,(recipient+delta).to(agent.device)):
                    pred=wm.unroll(z.clone(),act_suffix=plan)
                for key in ('visual','proprio'):exact(pred[key][:3],native[key][:3],'traversal unchanged pre-H3')
                cost=agent.objective(pred,plan).reshape(9).cpu().numpy()
                mse=(pred['visual'][1:].cpu()-truth).double().square().flatten(2).mean(2)
                restoration=(pred['visual'][1:].cpu()-native['visual'][1:].cpu()).double().square().flatten(2).mean(2)
                traversal.append(dict(direction=direction,coordinate=float(query),method=method,**score(cost,states,b['goal_state']),
                    native_goal_cost=cost.tolist(),actual_visual_mse_H1_H6_by_candidate=mse.tolist(),
                    mean_actual_visual_mse_by_horizon=mse.mean(1).tolist(),native_restoration_mse_by_horizon=restoration.mean(1).tolist(),
                    requested_raw_norm=delta.double().flatten(1).norm(dim=1).tolist(),
                    actual_rounded_norm=((recipient+delta)-recipient).double().flatten(1).norm(dim=1).tolist()))
                traversal_coefficients.append(coefficients.float())
    exact(plan,saved,'actions unchanged')
    if hashes!={k:tensor_hash(z[k]) for k in z.keys()} or goal!={k:tensor_hash(agent.goal_state_enc[k]) for k in goal}:
        raise ValueError('Context or goal changed')
    pt=a.output/(row['key']+'.pt')
    torch.save(dict(source=row,p3_h3=recipient,donor_residual_knots=knots,pca_mean=mean,pca_basis=basis.float(),
        pca_singular_values=singular.cpu(),knot_coordinates=coordinates,deltas=deltas,
        traversal_coordinates_and_methods=[(r['direction'],r['coordinate'],r['method']) for r in traversal],
        traversal_pca_coefficients=torch.stack(traversal_coefficients),
        predictions={name:{k:v[k].cpu() for k in ('visual','proprio')} for name,v in responses.items() if name!='exact_self'},
        raw_actions=b['raw_actions'],normalized_actions=b['normalized_actions'],physical_states=states),pt)
    report=pt.with_suffix('.json');seconds=time.monotonic()-started
    write_json(report,dict(complete=True,key=row['key'],source=row,rows=rows,traversal=traversal,seconds=seconds,identity_exact=True,
        intrinsic_coordinate=coordinate,PCA64_singular_values=singular[:64].cpu().tolist(),
        PCA64_explained_variance_fraction=float(singular[:64].square().sum()/singular.square().sum()),
        interpolation_knots=KNOTS.tolist(),native_midpoint_excluded_from_curve_fit=True,
        arc_length_150_intervals=arc.tolist(),endpoint_chord_length=endpoint.tolist(),
        arc_over_chord=(arc/endpoint.clamp_min(1e-30)).tolist(),off_subspace_rounding=off_residual,
        actual_rounded_norms=rounded,tensor_sha256=sha256(pt),tensor_bytes=pt.stat().st_size,
        limitations=['Action-conditioned input coordinate, not a globally identified physical manifold',
            'PCA fit to model-only donor residuals, not simulator future targets; conditional development adaptation',
            'Straight chord and natural spline share PCA64 endpoints and retained off-subspace recipient',
            'Only four already-seen physical states; all nine original plans retained; no new physics or planner search',
            'Native prediction restoration and actual future forecast error are separate outcomes']))
    print(json.dumps(dict(event='density_geometry_state_complete',key=row['key'],seconds=seconds)),flush=True)
    return [dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in (pt,report)],seconds


def main():
    p=argparse.ArgumentParser()
    for key in ('bank','repo','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episode',type=int,choices=range(4),required=True)
    a=p.parse_args();rows=safe_rows(json.loads((a.bank/'DONE.json').read_text()))
    rows=[row for row in rows if row['source_id']==a.episode]
    if len(rows)!=1:raise ValueError('Exactly assigned development state required before load')
    if sha256(a.checkpoint)!=CHECKPOINT:raise ValueError('Wrong checkpoint')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',dict(sources=rows,script_sha256=sha256(__file__),checkpoint_sha256=CHECKPOINT,
        paper='https://arxiv.org/html/2605.05115v1',sections=['A.3','A.5','A.6'],
        adaptation='Conditional one-dimensional known raw-action coordinate; not global concept or behavior-manifold identification',
        knots=KNOTS.tolist(),evaluation_coordinate=0,interior_traversal_coordinates=QUERIES.tolist(),PCA=64,PCA_fit='144 unedited model-only donor residuals per state, no target future',
        site='P3 residual newest256x400 at imaginedH3',physical_truth='same nine cached30raw-action plans per four development states',
        candidate='Mean over four fixed-direction spline midpoints plus each separate direction interior traversal',baseline='Same endpoints, same PCA support, ambient straight chord AND nearest-knot local-linear',
        traversal='Fixed original actions throughout; nonzero coordinate forecast scored to original action physical truth, not invented new path physical truth',
        controls=['native_unhooked','exact_self','PCA64 random spline-norm sham'],no_held=True,cem_calls=0,simulator_calls=0,
        max_seconds=900,primary=['within-state native-cost/physical-endpoint Spearman','selected action physical endpoint'],
        secondary=['actual encoded future MSE H1..H6','native forecast restoration','arc/chord ratio']))
    from model_loader import load_headless
    from collect_pusht_bank import config
    from capture_pusht_calibration import parameter_sha
    torch.set_num_threads(2);torch.manual_seed(90508);began=time.monotonic()
    try:
        cfg=config(a.repo,a.output);wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
        wm.eval().requires_grad_(False);before=parameter_sha(wm);outputs=[]
        for i,row in enumerate(rows):
            if time.monotonic()-began>900:raise RuntimeError('Bounded pilot time exceeded')
            entries,seconds=run_source(a,row,wm,prep,cfg);outputs.extend(entries)
            if i==0:
                write_json(a.output/'CANARY.json',dict(complete=True,seconds=seconds,no_outcome_gate=True,assigned_state=a.episode))
                if seconds>900:raise RuntimeError('Canary timing exceeds900 seconds')
        if parameter_sha(wm)!=before:raise ValueError('Weights changed')
        for name in ('protocol.json','CANARY.json','CANARY_IDENTITY.json'):
            file=a.output/name;outputs.append(dict(path=name,sha256=sha256(file),bytes=file.stat().st_size))
        write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-began,states=1,episode=a.episode,
            actions_per_state=9,weights_unchanged=True,parameter_sha256=before,provenance=provenance,cem_calls=0,simulator_calls=0))
    except Exception as exc:
        write_json(a.output/'FAILED.json',dict(complete=False,error=repr(exc),seconds=time.monotonic()-began));raise


if __name__=='__main__':main()
