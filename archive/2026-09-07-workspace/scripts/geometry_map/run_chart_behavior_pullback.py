#!/usr/bin/env python3
"""Bounded rank4 linear/nonlinear chart-controlled native response experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from run_behavior_pullback import sha,write,spline_weights,hellinger2,GradPatch,unroll_grad,CHECKPOINT
from native_chart_control import LearnedChart


def raw_lift(recipient,delta64,basis64):
    """Exact zero anchor; round only the final native residual replacement."""
    return (recipient.double()+ (delta64.double()@basis64.double()).reshape_as(recipient)).to(recipient)


def control_regularizer(path):
    return 1e-3*torch.tanh(path).square().mean()


def support_json(chart,coordinates):
    value=chart.support_coordinates(coordinates)
    return {k:v.detach().cpu().tolist() for k,v in value.items()}


def verified_load(root,name,episode,split=None):
    receipt=json.loads((root/'DONE.json').read_text())
    if not receipt['complete'] or receipt['episode']!=episode:raise ValueError('Completed assigned source before tensor load')
    if split is not None and receipt.get('split')!=split:raise ValueError('Development split before tensor load')
    row=next(v for v in receipt['outputs'] if v['path']==name)
    if sha(root/name)!=row['sha256']:raise ValueError('Source SHA before tensor load')
    return torch.load(root/name,map_location='cpu',weights_only=False),row


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
    frozen,target_source=verified_load(a.previous,'FROZEN.pt',a.episode)
    if sha(a.checkpoint)!=CHECKPOINT or side['source_density_sha256']!=b['density_sha256'] or frozen['source_sha256']!=source['sha256']:raise ValueError('Checkpoint or frozen source mismatch')
    protocol=dict(complete=True,episode=a.episode,script_sha256=sha(__file__),chart_script_sha256=sha(__import__('native_chart_control').__file__),
        source=source,chart_source=side_source,target_source=target_source,site='P3/H3 newest256x400 residual',maps=['linear_rank4','nonlinear_RBF_rank4'],
        controls=10,waypoints=20,initial_control='EXACT zeros shared4D for every waypoint and recipient',
        trust='shared shift .5*donor-coordinate std*tanh(u); no projection of baseline into training box',
        chart='144 unedited donor rows from this one seen state, PCA4 coordinates and RBF64 decoder; no future labels',
        behavior='Frozen v2 20-waypoint target of analyst softmax over9 native goal costs, not native probability',
        optimizer='LBFGS lr1 max_iter5 strong_wolfe50outer relative1e-3; at most85 optimization seconds/map; restore last accepted on cap',
        regularizer='1e-3 mean squared tanh(u), dimensionless equal across maps',
        bounds_seconds=dict(map=100,shared_per_state=25,aggregate_four_states=900),
        endpoints='Free within same bounded shift; not forced to donors or a train box',
        primary='Frozen native-response model-target matching; physical truth only offline forecast/rank scoring',
        candidate_promoted=False,held_accessed=False,simulator_calls=0,cem_calls=0)
    write(a.output/'protocol.json',protocol)
    cfg=config(a.repo,a.output);wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
    wm.eval().requires_grad_(False);weights_before=parameter_sha(wm)
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);agent.set_goal(TensorDict(b['raw_goal'],batch_size=[]))
    goal_difference=restore_fullprecision_goal(agent,b['goal_encoded'])
    z=batch_context(b['context'],9).to(wm.device);actions=b['normalized_actions'].transpose(0,1).contiguous().to(wm.device)
    hashes=dict(context={k:tensor_hash(z[k]) for k in z.keys()},actions=tensor_hash(actions),goal={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')})
    with torch.no_grad():
        native=wm.unroll(z.clone(),act_suffix=actions)
        with ResidualCapture(wm.model.predictor.predictor_blocks) as capture:repeat=wm.unroll(z.clone(),act_suffix=actions)
        for key in ('visual','proprio'):exact(native[key],repeat[key],'native read-only capture')
        recipient=capture.values[3,3].to(wm.device)
        with GradPatch(wm.model.predictor.predictor_blocks[3],recipient):identity=unroll_grad(wm,z,actions)
        for key in ('visual','proprio'):exact(native[key],identity[key],'native exact self patch')
    basis=side['pca_basis'].double().to(wm.device);mean=side['pca_mean'].double().to(wm.device)
    chart=LearnedChart.fit(side['donor_coordinates']).to(wm.device)
    x=(recipient.double().flatten(1)-mean)@basis.T;coordinates=chart.encode(x)
    temperature=frozen['temperature'].to(wm.device);target=frozen['target_probabilities'].to(wm.device)
    if target.shape!=(20,9):raise ValueError('Unchanged frozen 20x9 target required')
    pathweights=spline_weights(torch.linspace(0,1,10),torch.linspace(0,1,20)).to(wm.device)
    centralweights=spline_weights(torch.linspace(0,1,10),[.5]).to(wm.device)
    block=wm.model.predictor.predictor_blocks[3];truth=b['actual_visual'].to(wm.device);states=b['physical_states'].numpy()
    generator=torch.Generator().manual_seed(2026090640+a.episode)
    sham_basis=torch.linalg.qr(torch.randn(64,4,generator=generator,dtype=torch.float64)).Q.to(wm.device)
    sham_direction=sham_basis[:,0]
    baseline_cost=agent.objective(native,actions).reshape(9)
    baseline_prob=(-baseline_cost/temperature).softmax(-1)
    rows=[];maps={};canaries={};predictions={}

    def forward(u,nonlinear):
        delta=chart.delta(x,u,nonlinear=nonlinear)
        replacement=raw_lift(recipient,delta,basis)
        with GradPatch(block,replacement):pred=unroll_grad(wm,z,actions)
        costs=agent.objective(pred,actions).reshape(9)
        return (-costs/temperature).softmax(-1),costs,pred,replacement,delta

    def result_row(method,waypoint,u,prob,cost,pred,replacement,delta,target_row=None):
        for key in ('visual','proprio'):exact(pred[key][:3],native[key][:3],'unchanged prefix before H3')
        raw=(replacement-recipient).double().flatten(1)
        off=raw-(raw@basis.T)@basis
        result=dict(method=method,waypoint=waypoint,native_cost=cost.tolist(),**score(cost.cpu().numpy(),states,b['goal_state']),
            actual_visual_MSE_H1_H6=(pred['visual'][1:]-truth).double().square().flatten(2).mean(2).mean(1).tolist(),
            native_visual_MSE_H1_H6=(pred['visual'][1:]-native['visual'][1:]).double().square().flatten(2).mean(2).mean(1).tolist(),
            requested_raw_norm=delta.norm(dim=1).tolist(),realized_raw_norm=raw.norm(dim=1).tolist(),offPCA64_maxabs=float(off.abs().max()),
            actual_edited_coordinate_support=support_json(chart,chart.encode(x+delta)))
        if u is not None:
            requested=coordinates+.5*chart.scale*torch.tanh(u)
            result['requested_coordinate_support']=support_json(chart,requested)
            result['dimensionless_control']=u.tolist()
        if target_row is not None:result['target_hellinger2']=float(hellinger2(prob,target_row))
        return result

    setup_seconds=time.monotonic()-began
    for nonlinear in (False,True):
        method='nonlinear_RBF_rank4' if nonlinear else 'linear_rank4';map_start=time.monotonic()
        zero=torch.zeros(4,device=wm.device,dtype=torch.float64)
        with torch.no_grad():
            p,cost,pred,replacement,delta=forward(zero,nonlinear)
            exact(replacement,recipient,method+' exact zero residual')
            for key in ('visual','proprio'):exact(pred[key],native[key],method+' exact zero native output')
        point=zero.detach().requires_grad_(True);p,*_=forward(point,nonlinear)
        jacobian=torch.stack([torch.autograd.grad(p[i],point,retain_graph=i<8)[0] for i in range(9)])
        singular=torch.linalg.svdvals(jacobian);direction=torch.tensor([1.,-2.,3.,-4.],device=wm.device,dtype=torch.float64);direction/=direction.norm()
        epsilon=.01
        with torch.no_grad():finite=(forward(epsilon*direction,nonlinear)[0]-forward(-epsilon*direction,nonlinear)[0])/(2*epsilon)
        analytic=jacobian@direction;relative=float((finite-analytic).norm()/analytic.norm().clamp_min(1e-10))
        if relative>.15:raise ValueError('Native chart gradient finite-difference mismatch '+str(relative))
        canaries[method]=dict(exact_zero_identity=True,jacobian_singular_values=singular.tolist(),rank_relative1e_4=int((singular>singular[0]*1e-4).sum()),
            native_gradient_finite_difference_relative=relative,epsilon=epsilon,seconds=time.monotonic()-map_start)
        write(a.output/'CANARY.json',dict(complete=len(canaries)==2,maps=canaries,baseline_support=support_json(chart,coordinates),setup_seconds=setup_seconds))
        print(json.dumps(dict(event='chart_canary',episode=a.episode,method=method,relative=relative,seconds=time.monotonic()-began)),flush=True)
        control=torch.nn.Parameter(torch.zeros(10,4,device=wm.device,dtype=torch.float64))
        optimizer=torch.optim.LBFGS([control],lr=1,max_iter=5,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-12)
        history=[];closures=[];stop='maximum50outer'
        class BudgetStop(RuntimeError):pass
        def loss(backward):
            total=0.
            for i in range(20):
                if time.monotonic()-map_start>85:raise BudgetStop()
                probability,*_=forward(pathweights[i]@control,nonlinear)
                value=hellinger2(probability,target[i])/20
                if backward:value.backward()
                total+=float(value.detach())
            reg=control_regularizer(pathweights@control)
            if backward:reg.backward()
            return total+float(reg.detach()),float(reg.detach())
        for outer in range(50):
            accepted=control.detach().clone()
            def closure():
                optimizer.zero_grad();value,reg=loss(True)
                closures.append(dict(outer=outer,loss=value,regularizer=reg,gradient_norm=float(control.grad.norm()),seconds=time.monotonic()-map_start))
                return torch.tensor(value,device=wm.device)
            try:
                optimizer.step(closure)
                with torch.no_grad():value,reg=loss(False)
            except BudgetStop:
                with torch.no_grad():control.copy_(accepted)
                stop='85second_cap_last_accepted_control';break
            history.append(dict(outer=outer,loss=value,regularizer=reg,update_norm=float((control-accepted).norm()),seconds=time.monotonic()-map_start))
            if len(history)>1 and abs(history[-2]['loss']-value)/max(abs(history[-2]['loss']),1e-12)<1e-3:
                stop='relative_loss_change_below1e-3';break
        path=(pathweights@control).detach();saved=[]
        with torch.no_grad():
            for i,u in enumerate(path):
                probability,cost,pred,replacement,delta=forward(u,nonlinear)
                row=result_row(method,i,u,probability,cost,pred,replacement,delta,target[i]);rows.append(row);saved.append(delta.cpu())
            middle=(centralweights@control)[0].detach()
            probability,cost,pred,replacement,delta=forward(middle,nonlinear)
            rows.append(result_row(method,'central',middle,probability,cost,pred,replacement,delta))
            predictions[method]={k:pred[k].cpu() for k in ('visual','proprio')}
            sham_delta=delta.norm(dim=1)[:,None]*sham_direction
            sham_replacement=raw_lift(recipient,sham_delta,basis)
            with GradPatch(block,sham_replacement):sham=unroll_grad(wm,z,actions)
            sham_cost=agent.objective(sham,actions).reshape(9);sham_prob=(-sham_cost/temperature).softmax(-1)
            rows.append(result_row(method+'_norm_sham','central',None,sham_prob,sham_cost,sham,sham_replacement,sham_delta))
            predictions[method+'_norm_sham']={k:sham[k].cpu() for k in ('visual','proprio')}
            requested_norm_error=float((sham_delta.norm(dim=1)-delta.norm(dim=1)).abs().max())
            realized_norm_error=float(((sham_replacement-recipient).double().flatten(1).norm(dim=1)-(replacement-recipient).double().flatten(1).norm(dim=1)).abs().max())
        seconds=time.monotonic()-map_start
        maps[method]=dict(control=control.detach().cpu(),path=path.cpu(),delta64=torch.stack(saved),history=history,closures=closures,stop=stop,seconds=seconds,
            requested_sham_norm_max_error=requested_norm_error,realized_sham_norm_max_error=realized_norm_error,
            endpoints_delta64_norm=torch.stack(saved)[[0,-1]].norm(dim=2).tolist())
        print(json.dumps(dict(event='chart_map_complete',episode=a.episode,method=method,stop=stop,seconds=seconds)),flush=True)
        if seconds>100:raise RuntimeError('Predeclared100second map budget exceeded')
    with torch.no_grad():
        zero_delta=torch.zeros_like(x)
        rows.append(result_row('native','central',None,baseline_prob,baseline_cost,native,recipient,zero_delta))
    if parameter_sha(wm)!=weights_before:raise ValueError('Frozen weights changed')
    after=dict(context={k:tensor_hash(z[k]) for k in z.keys()},actions=tensor_hash(actions),goal={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')})
    if hashes!=after:raise ValueError('Original actions/context/goal changed')
    seconds=time.monotonic()-began
    if seconds>225:raise RuntimeError('Predeclared225second state budget exceeded')
    report=dict(complete=True,episode=a.episode,seconds=seconds,setup_seconds=setup_seconds,rows=rows,
        maps={k:{kk:vv for kk,vv in v.items() if kk not in ('control','path','delta64')} for k,v in maps.items()},
        baseline_target_hellinger2=float(hellinger2(baseline_prob[None],target).mean()),canaries=canaries,
        baseline_support=support_json(chart,coordinates),fresh_vs_saved_goal_encoding_maxabs=goal_difference,
        source_density_recipient_maxabs=float((recipient.cpu()-b['original_density_recipient']).abs().max()),
        candidate_promoted=False,limitations=['Bounded local anchored decoder surface, not a certified native manifold or on-support guarantee',
        '144 model-only donors per one state are dependent; four repeatedly seen development states total',
        'Same nine-action cost softmax and fixed model response targets; no native probability or true utility optimization',
        'Actual physical labels are offline scoring only, no new simulator/CEM or held performance',
        'Nonlinear chart restricted to4 coordinates but local response rank is measured, not assumed identifiable',
        'Same coordinate trust and capacity; linear/nonlinear raw edit norms may differ and each has its own norm-matched random4D sham'])
    write(a.output/'summary.json',report)
    torch.save(dict(report=report,maps=maps,predictions=predictions,chart_state=chart.cpu().state_dict(),source=source,side_source=side_source,target_source=target_source),a.output/'results.pt')
    outputs=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in (a.output/'protocol.json',a.output/'CANARY.json',a.output/'summary.json',a.output/'results.pt')]
    write(a.output/'DONE.json',dict(complete=True,episode=a.episode,seconds=seconds,outputs=outputs,checkpoint_sha256=CHECKPOINT,provenance=provenance,weights_unchanged=True))
    print(json.dumps(dict(event='chart_complete',episode=a.episode,seconds=seconds)),flush=True)


def prepare(a):
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.density/'DONE.json').read_text())
    if not receipt['complete'] or receipt['episode']!=a.episode:raise ValueError('Assigned completed density state before load')
    row=next(v for v in receipt['outputs'] if v['path']==f'near-dev-{a.episode:03}.pt')
    if sha(a.density/row['path'])!=row['sha256']:raise ValueError('Density hash before load')
    d=torch.load(a.density/row['path'],map_location='cpu',weights_only=False)
    basis=torch.linalg.qr(d['pca_basis'].double().T,mode='reduced').Q.T
    mean=d['pca_mean'].double()
    coordinates=(d['donor_residual_knots'].flatten(0,2).flatten(1).double()-mean)@basis.T
    path=a.output/'chart_inputs.pt'
    torch.save(dict(complete=True,episode=a.episode,source_density_sha256=row['sha256'],source_bank_sha256=d['source']['sha256'],
        pca_basis=basis.float(),pca_mean=mean.float(),donor_coordinates=coordinates.float(),
        donor_layout='4knots x4directions x9candidates; no physical future labels',donor_count=144,site='P3H3 full256x400',
        independent_initial_states=1,physical_outcomes_accessed=False),path)
    write(a.output/'DONE.json',dict(complete=True,episode=a.episode,outputs=[dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size)],
        source_density_sha256=row['sha256'],source_density_done_sha256=sha(a.density/'DONE.json'),physical_labels_used=False))
    print(json.dumps(dict(event='chart_inputs_prepared',episode=a.episode,bytes=path.stat().st_size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--episode',type=int,choices=range(4),required=True)
    for key in ('density','output','inputs','chart-inputs','previous','repo','checkpoint'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    try:prepare(a) if a.mode=='prepare' else run(a)
    except Exception:
        import traceback
        if a.output.exists():write(a.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()))
        raise
