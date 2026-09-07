#!/usr/bin/env python3
"""Frozen-control, equal-coordinate and equal-dose graph/RBF causal ablation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from native_chart_control import LearnedChart
from run_behavior_pullback import sha,write,spline_weights,hellinger2,GradPatch,unroll_grad,CHECKPOINT
from run_chart_behavior_pullback import verified_load,raw_lift,support_json


def first_ray_match(evaluate,target,grid_points=33,iterations=40):
    """First observed crossing, then bracketed root; no monotonicity assumption.

    evaluate(alpha[B]) returns B residual vectors. Crossing counts are finite-grid
    observations, not a certificate that no narrower excursions exist.
    """
    target=target.double();batch=len(target);grid=torch.linspace(0,1,grid_points,device=target.device,dtype=target.dtype)
    norms=torch.stack([evaluate(torch.full_like(target,t)).norm(dim=-1) for t in grid])
    if not torch.isfinite(norms).all() or bool((target<0).any()):raise ValueError('Finite nonnegative ray dose required')
    active=target>1e-12
    crossing=(norms[:-1]<target)&(norms[1:]>=target)
    observed=((norms[:-1]-target)*(norms[1:]-target)<0).sum(0)
    if bool((active&~crossing.any(0)).any()):raise ValueError('No bracket for assigned ray dose')
    first=crossing.int().argmax(0);lo=grid[first];hi=grid[first+1]
    for _ in range(iterations):
        mid=(lo+hi)/2;above=evaluate(mid).norm(dim=-1)>=target
        hi=torch.where(above,mid,hi);lo=torch.where(above,lo,mid)
    alpha=torch.where(active,(lo+hi)/2,torch.zeros_like(target));delta=evaluate(alpha)
    error=(delta.norm(dim=-1)-target).abs()
    if bool((error>1e-8+1e-8*target).any()):raise ValueError('Ray dose residual exceeds declared tolerance')
    return delta,dict(alpha=alpha,first_bracket=first,observed_grid_crossings=observed,
        grid_norms=norms,coordinate_norm_error=error,grid_points=grid_points,bisections=iterations,zero_target=~active)


def decode_shift(chart,x,shift,alpha,graph=False):
    c=chart.encode(x);decode=chart.decode_graph if graph else chart.decode
    return decode(c+alpha[:,None]*shift)-decode(c)


def compact_match(value):
    return {k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in value.items() if k!='grid_norms'}


def run(a):
    from model_loader import load_headless
    from collect_pusht_bank import config
    from capture_pusht_calibration import parameter_sha
    from capture_horizon_coordinates import exact,tensor_hash
    from run_action_path_curvature import ResidualCapture
    from causal_response_transfer import restore_fullprecision_goal
    from intervene_head_spatial_mean import batch_context
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from tensordict import TensorDict
    from run_action_ranking_pilot import score
    a.output.mkdir(parents=True,exist_ok=False);began=time.monotonic();torch.set_num_threads(2)
    b,source=verified_load(a.inputs,f'near-dev-{a.episode:03}.pt',a.episode,'development_external')
    side,side_source=verified_load(a.chart_inputs,'chart_inputs.pt',a.episode)
    previous,control_source=verified_load(a.previous,'results.pt',a.episode)
    frozen,target_source=verified_load(a.targets,'FROZEN.pt',a.episode)
    if sha(a.checkpoint)!=CHECKPOINT or side['source_density_sha256']!=b['density_sha256'] or frozen['source_sha256']!=source['sha256']:raise ValueError('Frozen source mismatch')
    if previous['source']['sha256']!=source['sha256'] or previous['side_source']['sha256']!=side_source['sha256']:raise ValueError('Control/chart provenance differs')
    protocol=dict(complete=True,episode=a.episode,script_sha256=sha(__file__),decoder_script_sha256=sha(__import__('native_chart_control').__file__),
        source=source,side_source=side_source,control_source=control_source,target_source=target_source,
        frozen_control_sources=['linear_rank4','nonlinear_RBF_rank4'],waypoints=20,central_query=True,controls_refitted=False,chart_refitted=False,
        site='P3/H3 full newest256x400 residual; PCA64 complement fixed',
        graph='mean+cB+(RBF(c)-mean)(I-B^T B), so re-encoding preserves coordinates',
        arms=['rbf_same_coordinate','graph_same_coordinate','rbf_matched_dose','graph_matched_dose','common_norm_sham'],
        norm_target='.95 times candidate-wise min of original-RBF and graph full-shift Euclidean norms',
        dose_solver='33point first upward crossing scan,40 bisections within first bracket; finite-grid crossing counts not uniqueness proof',
        coordinate_dose_tolerance='1e-8 absolute +1e-8 relative; native realized pair norm max1e-4 absolute +1e-6 relative',
        norm_matching_limitation='Different candidate-wise alpha changes coordinate shifts; on-ray matching avoids off-surface normalization but is not an equal-coordinate contrast',
        primary='Matched-dose graph vs RBF actual H3/H6 spatial forecast error and fixed9action physical ranking/choice',
        secondary='Frozen analyst model-response target loss, exact reencode, support, raw norm and nonmonotonic-ray feasibility',
        decision='Exploratory only; no promotion from4seen states. Report all states and both frozen-control parents; no outcome selection',
        max_state_seconds=225,max_aggregate_seconds=900,optimizer_steps=0,simulator_calls=0,cem_calls=0,held_accessed=False,
        literature=['https://arxiv.org/abs/2205.02304','https://arxiv.org/html/2509.00224v1'],
        literature_scope='Orthogonal nonlinear augmentation is established dimensionality-reduction construction, not evidence of robot steering utility')
    write(a.output/'protocol.json',protocol)
    cfg=config(a.repo,a.output);wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
    wm.eval().requires_grad_(False);weights=parameter_sha(wm)
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);agent.set_goal(TensorDict(b['raw_goal'],batch_size=[]))
    goal_error=restore_fullprecision_goal(agent,b['goal_encoded'])
    z=batch_context(b['context'],9).to(wm.device);actions=b['normalized_actions'].transpose(0,1).contiguous().to(wm.device)
    hashes=dict(actions=tensor_hash(actions),context={k:tensor_hash(z[k]) for k in z.keys()},goal={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')})
    with torch.no_grad():
        native=wm.unroll(z.clone(),act_suffix=actions)
        with ResidualCapture(wm.model.predictor.predictor_blocks) as capture:repeat=wm.unroll(z.clone(),act_suffix=actions)
        for key in ('visual','proprio'):exact(native[key],repeat[key],'fresh native read-only repeat')
        recipient=capture.values[3,3].to(wm.device)
        with GradPatch(wm.model.predictor.predictor_blocks[3],recipient):identity=unroll_grad(wm,z,actions)
        for key in ('visual','proprio'):exact(native[key],identity[key],'native self patch')
    basis=side['pca_basis'].double().to(wm.device);mean=side['pca_mean'].double().to(wm.device)
    chart=LearnedChart(**previous['chart_state']).to(wm.device)
    x=(recipient.double().flatten(1)-mean)@basis.T;c=chart.encode(x)
    target=frozen['target_probabilities'].to(wm.device);temperature=frozen['temperature'].to(wm.device)
    truth=b['actual_visual'].to(wm.device);states=b['physical_states'].numpy();goal=torch.as_tensor(b['goal_state']).double()
    physical=torch.as_tensor(states).double();block_distance=(physical[:,-1,2:4]-goal[2:4]).norm(dim=-1)
    angle=physical[:,-1,4]-goal[4];angle=torch.atan2(angle.sin(),angle.cos()).abs()
    block=wm.model.predictor.predictor_blocks[3]
    generator=torch.Generator().manual_seed(2026090640+a.episode)
    sham_basis=torch.linalg.qr(torch.randn(64,4,generator=generator,dtype=torch.float64)).Q.to(wm.device);sham_direction=sham_basis[:,0]
    def evaluate(delta):
        replacement=raw_lift(recipient,delta,basis)
        with GradPatch(block,replacement):pred=unroll_grad(wm,z,actions)
        cost=agent.objective(pred,actions).reshape(9)
        return (-cost/temperature).softmax(-1),cost,pred,replacement
    with torch.no_grad():
        zero=torch.zeros(4,device=wm.device,dtype=torch.float64)
        for graph in (False,True):
            delta=chart.delta_graph(x,zero) if graph else chart.delta(x,zero)
            exact(delta,torch.zeros_like(delta),'both decoder exact zero')
            _,_,pred,_=evaluate(delta)
            for key in ('visual','proprio'):exact(pred[key],native[key],'both zero native output')
        roundtrip=float((chart.encode(chart.decode_graph(c))-c).abs().max())
        if roundtrip>1e-8:raise ValueError('Graph reencode failed')
    point=zero.detach().requires_grad_(True);prob,*_=evaluate(chart.delta_graph(x,point))
    jac=torch.stack([torch.autograd.grad(prob[i],point,retain_graph=i<8)[0] for i in range(9)])
    direction=torch.tensor([1.,-2.,3.,-4.],device=wm.device,dtype=torch.float64);direction/=direction.norm()
    with torch.no_grad():finite=(evaluate(chart.delta_graph(x,.01*direction))[0]-evaluate(chart.delta_graph(x,-.01*direction))[0])/.02
    analytic=jac@direction;relative=float((finite-analytic).norm()/analytic.norm().clamp_min(1e-10));singular=torch.linalg.svdvals(jac)
    if relative>.15:raise ValueError('Native graph autograd mismatch')
    canary=dict(complete=True,exact_zero=True,graph_roundtrip_maxabs=roundtrip,gradient_fd_relative=relative,
        native_graph_jacobian_singular_values=singular.tolist(),native_graph_rank_relative1e_4=int((singular>singular[0]*1e-4).sum()),seconds=time.monotonic()-began)
    write(a.output/'CANARY.json',canary);print(json.dumps(dict(event='graph_canary',episode=a.episode,**canary)),flush=True)
    rows=[];predictions={};matches=[];deltas=[];old_parity={}
    def make_row(parent,method,index,delta,pred,cost,replacement,requested_shift=None,match=None):
        for key in ('visual','proprio'):exact(pred[key][:3],native[key][:3],'unchanged pre-H3 context')
        raw=(replacement-recipient).double().flatten(1);off=raw-(raw@basis.T)@basis
        if float(off.abs().max())>1e-4:raise ValueError('Realized native edit left common PCA64 support beyond rounding floor')
        row=dict(control_parent=parent,method=method,waypoint=index,native_cost=cost.tolist(),**score(cost.cpu().numpy(),states,b['goal_state']),
            actual_visual_MSE_H1_H6=(pred['visual'][1:]-truth).double().square().flatten(2).mean(2).mean(1).tolist(),
            native_visual_MSE_H1_H6=(pred['visual'][1:]-native['visual'][1:]).double().square().flatten(2).mean(2).mean(1).tolist(),
            requested_raw_norm=delta.norm(dim=-1).tolist(),realized_raw_norm=raw.norm(dim=-1).tolist(),
            off_PCA64_maxabs=float(off.abs().max()),actual_support=support_json(chart,chart.encode(x+delta)))
        selected=row['selected_action_index'];row.update(selected_block_distance_px=float(block_distance[selected]),selected_wrapped_angle_error_rad=float(angle[selected]))
        if isinstance(index,int):row['target_hellinger2']=float(hellinger2((-cost/temperature).softmax(-1),target[index]))
        if requested_shift is not None:
            mismatch=delta@chart.basis.T-requested_shift
            row['coordinate_shift_error_norm']=mismatch.norm(dim=-1).tolist()
            row['native_realized_coordinate_shift_error_norm']=((raw@basis.T)@chart.basis.T-requested_shift).norm(dim=-1).tolist()
            row['requested_support']=support_json(chart,c+requested_shift)
        if match is not None:row['ray_match']=compact_match(match)
        return row
    with torch.no_grad():
        native_cost=agent.objective(native,actions).reshape(9)
        rows.append(make_row('native','native','central',torch.zeros_like(x),native,native_cost,recipient))
        for parent in ('linear_rank4','nonlinear_RBF_rank4'):
            controls=previous['maps'][parent]['control'].to(wm.device)
            central=(spline_weights(torch.linspace(0,1,10),[.5]).to(wm.device)@controls)[0]
            paths=previous['maps'][parent]['path'].to(wm.device)
            for index,u in [*enumerate(paths),('central',central)]:
                if time.monotonic()-began>215:raise RuntimeError('State inference cap reached; preserve partial failure')
                shift=.5*chart.scale*u.tanh();ones=torch.ones(9,device=wm.device,dtype=torch.float64)
                full={g:decode_shift(chart,x,shift,ones,g) for g in (False,True)}
                corrected=full[True]-full[False]
                if float((corrected-(corrected@chart.basis.T)@chart.basis).abs().max())>1e-8:raise ValueError('Graph correction changed unattenuated nonlinear complement')
                target_norm=.95*torch.minimum(full[False].norm(dim=-1),full[True].norm(dim=-1))
                matched={};metadata={}
                for graph in (False,True):
                    matched[graph],metadata[graph]=first_ray_match(lambda alpha:decode_shift(chart,x,shift,alpha,graph),target_norm)
                matches.append(dict(parent=parent,waypoint=index,target_norm=target_norm.cpu(),rbf=metadata[False],graph=metadata[True]))
                conditions=[('rbf_same_coordinate',full[False],shift.expand(9,4),None),('graph_same_coordinate',full[True],shift.expand(9,4),None),
                    ('rbf_matched_dose',matched[False],metadata[False]['alpha'][:,None]*shift,metadata[False]),
                    ('graph_matched_dose',matched[True],metadata[True]['alpha'][:,None]*shift,metadata[True]),
                    ('common_norm_sham',target_norm[:,None]*sham_direction,None,None)]
                realized={}
                for method,delta,requested,meta in conditions:
                    _,cost,pred,replacement=evaluate(delta)
                    row=make_row(parent,method,index,delta,pred,cost,replacement,requested,meta);rows.append(row)
                    realized[method]=(replacement-recipient).double().flatten(1).norm(dim=-1)
                    if method.startswith('graph'):
                        if max(row['coordinate_shift_error_norm'])>1e-8:raise ValueError('Graph delta does not preserve requested coordinates')
                    if index=='central':
                        predictions[parent+'/'+method]={k:pred[k].cpu() for k in ('visual','proprio')}
                        if parent=='nonlinear_RBF_rank4' and method=='rbf_same_coordinate':
                            old=previous['predictions'][parent]
                            old_parity={k:float((pred[k].cpu()-old[k]).abs().max()) for k in ('visual','proprio')}
                            for key in ('visual','proprio'):exact(pred[key].cpu(),old[key],'reuse original RBF central forecast')
                    deltas.append(dict(parent=parent,method=method,waypoint=index,delta64=delta.cpu()))
                for method in ('rbf_matched_dose','graph_matched_dose','common_norm_sham'):
                    error=(realized[method]-target_norm).abs()
                    if bool((error>1e-4+1e-6*target_norm).any()):raise ValueError('Realized native dose does not match assigned target')
                if index=='central':print(json.dumps(dict(event='graph_parent_complete',episode=a.episode,parent=parent,seconds=time.monotonic()-began)),flush=True)
    if parameter_sha(wm)!=weights:raise ValueError('Model weights changed')
    after=dict(actions=tensor_hash(actions),context={k:tensor_hash(z[k]) for k in z.keys()},goal={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')})
    if hashes!=after:raise ValueError('Fixed native inputs changed')
    seconds=time.monotonic()-began
    if seconds>225:raise RuntimeError('225second state budget exceeded')
    report=dict(complete=True,episode=a.episode,seconds=seconds,rows=rows,canary=canary,old_rbf_central_exact=old_parity,
        baseline_support=support_json(chart,c),goal_encoding_difference=goal_error,candidate_promoted=False,
        limitations=['Frozen-control transport ablation, not graph-optimized steering or held generalization',
        'Equal coordinate comparison differs in raw norm; equal dose comparison differs in candidate-specific coordinate attenuation',
        'Ray matching preserves each anchored decoder surface; this surface is not a certified native manifold or training support',
        '33-point crossing counts cannot exclude narrower nonmonotonic excursions; first observed bracket used without outcome information',
        'Actual cached physical labels enter offline scores only; combinedXYprogress is not native Push success',
        'Four seen states; 42pathpoints and9actions perstate are dependent, no pseudoreplication',
        'Model-response softmax is analyst constructed, not native probability'])
    write(a.output/'summary.json',report)
    # Store ray information on CPU; source model/physical banks stay immutable.
    cpu_matches=[dict(parent=m['parent'],waypoint=m['waypoint'],target_norm=m['target_norm'],
        rbf={k:v.cpu() if torch.is_tensor(v) else v for k,v in m['rbf'].items()},
        graph={k:v.cpu() if torch.is_tensor(v) else v for k,v in m['graph'].items()}) for m in matches]
    torch.save(dict(report=report,matches=cpu_matches,deltas=deltas,predictions=predictions,chart_state=previous['chart_state'],control_source=control_source),a.output/'results.pt')
    outputs=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in (a.output/'protocol.json',a.output/'CANARY.json',a.output/'summary.json',a.output/'results.pt')]
    write(a.output/'DONE.json',dict(complete=True,episode=a.episode,seconds=seconds,outputs=outputs,checkpoint_sha256=CHECKPOINT,weights_unchanged=True,provenance=provenance))
    print(json.dumps(dict(event='graph_complete',episode=a.episode,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=range(4),required=True)
    for key in ('inputs','chart-inputs','previous','targets','output','repo','checkpoint'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except Exception:
        import traceback
        if a.output.exists():write(a.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()))
        raise
