#!/usr/bin/env python3
"""A.8-inspired native-goal-response pullback, not an identified physical manifold."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch

CHECKPOINT='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
KNOTS=(-4.,-2.,2.,4.)

def sha(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def spline_weights(knots,queries):
    """Exact natural cubic interpolation, linear in donor/control values."""
    x=torch.as_tensor(knots,dtype=torch.float64);q=torch.as_tensor(queries,dtype=torch.float64)
    if x.ndim!=1 or len(x)<2 or not bool((x[1:]>x[:-1]).all()):raise ValueError('Strictly ordered knots')
    if bool(((q<x[0])|(q>x[-1])).any()):raise ValueError('No extrapolation')
    n=len(x);y=torch.eye(n,dtype=torch.float64);h=x[1:]-x[:-1]
    matrix=torch.zeros(n,n,dtype=torch.float64);rhs=torch.zeros_like(matrix);matrix[0,0]=matrix[-1,-1]=1
    for i in range(1,n-1):
        matrix[i,i-1]=h[i-1];matrix[i,i]=2*(h[i-1]+h[i]);matrix[i,i+1]=h[i]
        rhs[i]=6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
    second=torch.linalg.solve(matrix,rhs);idx=(torch.searchsorted(x,q,right=True)-1).clamp(0,n-2)
    width=h[idx];a=(x[idx+1]-q)/width;b=(q-x[idx])/width
    return a[:,None]*y[idx]+b[:,None]*y[idx+1]+width[:,None].square()/6*((a.pow(3)-a)[:,None]*second[idx]+(b.pow(3)-b)[:,None]*second[idx+1])

def sphere_log(base,point):
    dot=(point*base).sum(-1,keepdim=True).clamp(-1,1)
    orthogonal=point-dot*base;norm=orthogonal.norm(dim=-1,keepdim=True)
    angle=torch.atan2(norm,dot)
    return orthogonal*(angle/norm.clamp_min(1e-15))

def sphere_exp(base,tangent):
    tangent=tangent-(tangent*base).sum(-1,keepdim=True)*base
    norm=tangent.norm(dim=-1,keepdim=True)
    return norm.cos()*base+torch.sinc(norm/torch.pi)*tangent

def behavior_curve(probabilities,queries):
    roots=probabilities.double().sqrt();base=roots.mean(0);base/=base.norm()
    tangent=sphere_log(base,roots)
    curved=sphere_exp(base,spline_weights(KNOTS,queries)@tangent)
    # Squaring then square-root is explicit if a spline crosses the positive orthant.
    decoded=curved.square();decoded/=decoded.sum(-1,keepdim=True)
    return decoded,dict(min_signed_sphere_component=float(curved.min()),orthant_crossing=bool((curved<0).any()),
        sphere_unit_max_error=float((curved.norm(dim=-1)-1).abs().max()))

def projected_replacement(recipient,mean,basis,coordinate):
    flat=recipient.flatten(1)
    current=(flat-mean)@basis.T
    return (flat+(coordinate.to(flat)-current)@basis).reshape_as(recipient)

def scale_prepare(a):
    a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.density/'DONE.json').read_text())
    if not receipt['complete'] or receipt['episode']!=a.episode:raise ValueError('Density receipt before load')
    row=next(v for v in receipt['outputs'] if v['path']==f'near-dev-{a.episode:03}.pt')
    if sha(a.density/row['path'])!=row['sha256']:raise ValueError('Density hash before load')
    d=torch.load(a.density/row['path'],map_location='cpu',weights_only=False)
    basis=torch.linalg.qr(d['pca_basis'][:32].double().T,mode='reduced').Q.T
    features=d['donor_residual_knots'].flatten(0,2).flatten(1).double()
    c=(features-d['pca_mean'].double())@basis.T
    std=c.std(0,unbiased=False);floor=.01*std.median();scale=std.clamp_min(floor)
    write(a.output/'scale.json',dict(complete=True,episode=a.episode,source_density_sha256=row['sha256'],
        model_only_donor_count=len(c),standard_deviation=std.tolist(),floor=float(floor),scale=scale.tolist(),
        source='All144 unedited model donor residuals from same seen development state; no physical labels',actual_future_accessed=False))
    print(json.dumps(dict(event='scale_prepared',episode=a.episode,min_scale=float(scale.min()),max_scale=float(scale.max()))),flush=True)

def hellinger2(probability,target):
    return .5*(probability.sqrt()-target.sqrt()).square().sum(-1)

def prepare(a):
    """One state only; receipt and split are checked before loading each payload."""
    from run_action_ranking_pilot import safe_rows
    from capture_horizon_coordinates import tensor_hash
    a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.density/'DONE.json').read_text())
    if not receipt['complete'] or receipt['episode']!=a.episode:raise ValueError('Density receipt state mismatch')
    name=f'near-dev-{a.episode:03}.pt';row=next(v for v in receipt['outputs'] if v['path']==name)
    if sha(a.density/name)!=row['sha256']:raise ValueError('Density SHA before load')
    source=next(v for v in safe_rows(json.loads((a.bank/'DONE.json').read_text())) if v['source_id']==a.episode)
    if sha(a.bank/source['path'])!=source['sha256']:raise ValueError('Bank SHA before load')
    d=torch.load(a.density/name,map_location='cpu',weights_only=False)
    b=torch.load(a.bank/source['path'],map_location='cpu',weights_only=False)
    if d['source']['sha256']!=source['sha256'] or b['row']['split']!='development_external':raise ValueError('Cross-source mismatch')
    if not b['same_state_actions_and_two_physical_replays_exact']:raise ValueError('No valid actual physical truth')
    # QR of saved float32 rows only removes roundoff within the same top32 span.
    basis=torch.linalg.qr(d['pca_basis'][:32].double().T,mode='reduced').Q.T
    mean=d['pca_mean'].double();donors=d['donor_residual_knots'][:,0].double().flatten(2)
    coordinates=((donors-mean)@basis.T).mean(1)
    out=a.output/name
    torch.save(dict(episode=a.episode,split='development_external',source=source,density_sha256=row['sha256'],
        density_done_sha256=sha(a.density/'DONE.json'),context=b['context'],raw_goal=b['raw_goal'],goal_encoded=b['goal_encoded'],
        normalized_actions=b['normalized_actions'],raw_actions=b['raw_actions'],goal_state=b['goal_state'],
        actual_visual=torch.stack([v['actual_visual'] for v in b['candidates']],1),
        physical_states=torch.as_tensor(__import__('numpy').stack([v['states'] for v in b['candidates']])),
        pca_basis=basis,pca_mean=mean,donor_centroid_coordinates=coordinates,
        original_density_recipient=d['p3_h3'],source_action_sha256=tensor_hash(b['normalized_actions'])),out)
    write(a.output/'DONE.json',dict(complete=True,episode=a.episode,split='development_external',outputs=[dict(path=out.name,sha256=sha(out),bytes=out.stat().st_size)],
        source_bank_sha256=source['sha256'],source_density_sha256=row['sha256'],source_density_done_sha256=sha(a.density/'DONE.json'),no_physical_labels_in_coordinate_fit=True))
    print(json.dumps(dict(event='prepared',episode=a.episode,bytes=out.stat().st_size)),flush=True)


def unroll_grad(wm,context,actions):
    """Unwrap only the native unroll decorator, preserving its executed body."""
    function=inspect.unwrap(type(wm).unroll)
    return function(wm,context.clone(),act_suffix=actions)

class GradPatch:
    def __init__(self,block,replacement):self.block=block;self.replacement=replacement;self.calls=0
    def __enter__(self):self.handle=self.block.register_forward_hook(self.patch);return self
    def patch(self,module,args,out):
        self.calls+=1
        if self.calls==3:
            if out.shape[0]!=9 or out.shape[-1]!=400:raise ValueError('Native patch shape changed')
            return torch.cat((out[:,:-256],self.replacement.to(out)),dim=1)
    def __exit__(self,*exc):
        self.handle.remove()
        if exc[0] is None and self.calls!=6:raise ValueError('Exactly six native horizons required')


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
    import numpy as np
    a.output.mkdir(parents=True,exist_ok=False);began=time.monotonic();torch.set_num_threads(2)
    receipt=json.loads((a.inputs/'DONE.json').read_text());entry=receipt['outputs'][0]
    if not receipt['complete'] or receipt['split']!='development_external' or receipt['episode']!=a.episode:raise ValueError('Prepared state receipt')
    if sha(a.inputs/entry['path'])!=entry['sha256'] or sha(a.checkpoint)!=CHECKPOINT:raise ValueError('Inputs/checkpoint changed')
    protocol=dict(paper='https://arxiv.org/html/2605.05115v1',author_code='https://github.com/goodfire-ai/causalab/tree/manifold_steering',
        script_sha256=sha(__file__),source=entry,episode=a.episode,exploratory=True,held_accessed=False,
        site='P3 residual newest256x400 imaginedH3',direction=0,knots=list(KNOTS),rank=32,controls=10,waypoints=20,
        behavior='Analyst-constructed softmax over9 native fullH6 visual+proprio goal costs, NOT native probabilities',
        target_fit='Natural cubic sphere-tangent spline of unedited donor action cost distributions; no activation distances or true future labels',
        temperature='Standard deviation of all36 donor costs, minimum1e-3; frozen before intervention',
        path='Shared32D coordinate across9 fixed candidate actions; donor centroids average all9contexts; remainingPCA32 and orthogonal residual fixed per recipient',
        optimizer='L-BFGS lr1 max_iter5 strong_wolfe,50outer,relative loss change1e-3 after consecutive evaluated outersteps',
        regularizer='1e-3 mean squared difference from interpolated endpoint coordinate norm, divided by mean endpoint squarednorm',
        endpoints='All10controls free, includingendpoints; reportendpointdrift',
        comparators=['native','exact_self','samePCA32_endpoint_chord','samePCA32_local_chord','samePCA32_natural_spline','pullback','pullback_norm_sham'],
        primary='Behavior-target Hellinger loss; physical9-action rank and forecastaccuracy separate offline secondary; no utility promotion',
        identifiability='9-way simplex Jacobian rank at most8, cannot identify32D activation geometry',
        max_state_seconds=a.max_seconds,aggregate_four_state_cap_seconds=1800,simulator_calls=0,cem_calls=0,
        conditioning='donor_std_FP64' if a.scale else 'raw_FP32')
    write(a.output/'protocol.json',protocol)
    b=torch.load(a.inputs/entry['path'],map_location='cpu',weights_only=False)
    cfg=config(a.repo,a.output);wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm)
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);agent.set_goal(TensorDict(b['raw_goal'],batch_size=[]))
    goal_difference=restore_fullprecision_goal(agent,b['goal_encoded'])
    z=batch_context(b['context'],9).to(wm.device);actions=b['normalized_actions'].transpose(0,1).contiguous().to(wm.device)
    fingerprint={k:tensor_hash(z[k]) for k in z.keys()};action_hash=tensor_hash(actions);goals={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')}
    with torch.no_grad():
        native=wm.unroll(z.clone(),act_suffix=actions)
        with ResidualCapture(wm.model.predictor.predictor_blocks) as capture:repeat=wm.unroll(z.clone(),act_suffix=actions)
        for key in ('visual','proprio'):exact(native[key],repeat[key],'native read-only capture')
        recipient=capture.values[3,3].to(wm.device);basis=b['pca_basis'].float().to(wm.device);mean=b['pca_mean'].float().to(wm.device)
        baseline_cost=agent.objective(native,actions).reshape(9)
        native_gradbody=unroll_grad(wm,z,actions)
        for key in ('visual','proprio'):exact(native_gradbody[key],native[key],'same native unwrapped body')
        with GradPatch(wm.model.predictor.predictor_blocks[3],recipient):identity=unroll_grad(wm,z,actions)
        for key in ('visual','proprio'):exact(identity[key],native[key],'native exact self patch')
        donor_costs=[];delta=b['normalized_actions'][1]-b['normalized_actions'][0]
        for amplitude in KNOTS:
            donor_actions=(b['normalized_actions']+amplitude*delta[None]).transpose(0,1).contiguous().to(wm.device)
            donor=wm.unroll(z.clone(),act_suffix=donor_actions)
            donor_costs.append(agent.objective(donor,donor_actions).reshape(9))
        donor_costs=torch.stack(donor_costs);temperature=donor_costs.std(unbiased=False).clamp_min(1e-3)
        donor_probs=(-donor_costs/temperature).softmax(-1)
    coordinates=b['donor_centroid_coordinates'].float().to(wm.device)
    fractions=torch.linspace(0,1,20,device=wm.device);queries=(fractions*8-4).double().cpu()
    target,curve_guard=behavior_curve(donor_probs.cpu(),queries);target=target.float().to(wm.device)
    pathweights=spline_weights(torch.linspace(0,1,10),torch.linspace(0,1,20)).float().to(wm.device)
    initial=torch.lerp(coordinates[0],coordinates[-1],torch.linspace(0,1,10,device=wm.device)[:,None])
    if a.scale:
        scale_receipt=json.loads(a.scale.read_text())
        if not scale_receipt['complete'] or scale_receipt['episode']!=a.episode or scale_receipt['source_density_sha256']!=b['density_sha256']:raise ValueError('Scale source differs')
        scale=torch.tensor(scale_receipt['scale'],dtype=torch.float64,device=wm.device)
        offset=coordinates.mean(0).double();control=torch.nn.Parameter((initial.double()-offset)/scale)
        pathweights=pathweights.double()
    else:
        scale=torch.ones(32,device=wm.device);offset=torch.zeros(32,device=wm.device);control=torch.nn.Parameter(initial.clone())
    def decode(value):return offset+value*scale
    block=wm.model.predictor.predictor_blocks[3]
    target_norm=torch.lerp(coordinates[0].norm(),coordinates[-1].norm(),fractions)
    norm_scale=((coordinates[0].square().sum()+coordinates[-1].square().sum())/2).clamp_min(1e-12)

    def forward(point):
        replacement=projected_replacement(recipient,mean,basis,point)
        with GradPatch(block,replacement):pred=unroll_grad(wm,z,actions)
        costs=agent.objective(pred,actions).reshape(9)
        return (-costs/temperature).softmax(-1),costs,pred,replacement

    # Numerical causal response check before optimization, fixed direction and step.
    center=((coordinates[0]+coordinates[-1])/2).detach().requires_grad_(True)
    probability,_,_,_=forward(center)
    if not probability.requires_grad:raise RuntimeError('Native model path detached; no surrogate fallback')
    jac=torch.stack([torch.autograd.grad(probability[i],center,retain_graph=i<8)[0] for i in range(9)])
    singular=torch.linalg.svdvals(jac.double());g=torch.Generator().manual_seed(9050910+a.episode)
    direction=torch.randn(32,generator=g).to(wm.device);direction/=direction.norm();epsilon=.01*center.norm().clamp_min(1.)
    with torch.no_grad():
        plus=forward(center+epsilon*direction)[0];minus=forward(center-epsilon*direction)[0]
    finite=(plus-minus)/(2*epsilon);analytic=jac@direction
    relative=float((finite-analytic).norm()/analytic.norm().clamp_min(1e-10))
    if relative>.15:raise ValueError('Autograd finite difference mismatch: '+str(relative))
    write(a.output/'CANARY.json',dict(complete=True,seconds=time.monotonic()-began,native_identity_exact=True,gradient_fd_relative_error=relative,
        gradient_fd_epsilon=float(epsilon),jacobian_singular_values=singular.tolist(),numerical_rank_relative1e_4=int((singular>singular[0]*1e-4).sum()),
        maximum_rank=8,ambient_rank=32,behavior_sphere=curve_guard,temperature=float(temperature)))
    print(json.dumps(dict(event='pullback_gradient_canary',episode=a.episode,seconds=time.monotonic()-began,relative=relative)),flush=True)
    frozen=dict(donor_costs=donor_costs.cpu(),temperature=temperature.cpu(),target_probabilities=target.cpu(),initial_controls=initial.cpu(),
        source_sha256=entry['sha256'],pca_basis=b['pca_basis'],pca_mean=b['pca_mean'],jacobian=jac.cpu(),
        conditioning_scale=scale.cpu(),conditioning_offset=offset.cpu(),initial_parameter=control.detach().cpu())
    torch.save(frozen,a.output/'FROZEN.pt')
    optimizer=torch.optim.LBFGS([control],lr=1,max_iter=5,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-12)
    history=[];closures=[];stop='maximum50outer'
    class BudgetStop(RuntimeError):pass

    def loss(backward):
        total=0.;individual=[]
        # Sequential batch9 waypoints keep memory bounded and preserve same batch/layout.
        for i in range(20):
            if time.monotonic()-began>a.max_seconds-30:raise BudgetStop('Optimization cap reached')
            probability,*_=forward(decode(pathweights[i]@control));value=hellinger2(probability,target[i])/20
            if backward:value.backward()
            total+=float(value.detach());individual.append(float(value.detach()*20))
        path=decode(pathweights@control)
        reg=1e-3*((path.norm(dim=-1)-target_norm).square()/norm_scale).mean()
        if backward:reg.backward()
        return total+float(reg.detach()),individual,float(reg.detach())

    for outer in range(50):
        if time.monotonic()-began>a.max_seconds-30:stop='predeclared_state_time_cap';break
        def closure():
            optimizer.zero_grad();value,individual,reg=loss(True)
            gradient=control.grad.detach();raw_gradient=gradient/scale
            extra={}
            if not closures:
                raw_control=decode(control).detach()
                before_replace=projected_replacement(recipient,mean,basis,raw_control[0]).detach()
                raw_trial=projected_replacement(recipient,mean,basis,raw_control[0]-raw_gradient[0]).detach()
                scaled_trial=projected_replacement(recipient,mean,basis,decode(control-gradient)[0]).detach()
                ulp=(torch.nextafter(before_replace,torch.full_like(before_replace,float('inf')))-before_replace).abs()
                extra=dict(raw_lr1_replacement_maxabs=float((raw_trial-before_replace).abs().max()),
                    raw_lr1_replacement_change_fraction=float((raw_trial!=before_replace).float().mean()),
                    conditioned_lr1_replacement_maxabs=float((scaled_trial-before_replace).abs().max()),
                    conditioned_lr1_replacement_change_fraction=float((scaled_trial!=before_replace).float().mean()),
                    median_native_replacement_ULP=float(ulp.median()),raw_control_norm=float(raw_control.norm()))
            closures.append(dict(outer=outer,loss=value,regularizer=reg,seconds=time.monotonic()-began,
                parameter_gradient_norm=float(gradient.norm()),raw_gradient_norm=float(raw_gradient.norm()),**extra))
            return torch.tensor(value,device=wm.device)
        accepted=control.detach().clone()
        try:
            optimizer.step(closure)
            with torch.no_grad():value,individual,reg=loss(False)
        except BudgetStop:
            with torch.no_grad():control.copy_(accepted)
            stop='predeclared_state_time_cap_last_accepted_control';break
        history.append(dict(outer=outer,loss=value,regularizer=reg,waypoint_losses=individual,seconds=time.monotonic()-began,
            parameter_update_norm=float((control.detach()-accepted).norm()),raw_control_update_norm=float(((control.detach()-accepted)*scale).norm())))
        print(json.dumps(dict(event='pullback_outer',episode=a.episode,outer=outer,loss=value,seconds=time.monotonic()-began)),flush=True)
        if len(history)>1 and abs(history[-2]['loss']-value)/max(abs(history[-2]['loss']),1e-12)<1e-3:
            stop='relative_loss_change_below1e-3';break
    if time.monotonic()-began>a.max_seconds:raise RuntimeError('State budget exceeded')
    optimized=decode(pathweights@control).detach();chord=torch.lerp(coordinates[0],coordinates[-1],fractions[:,None])
    spline=spline_weights(KNOTS,queries).float().to(wm.device)@coordinates
    local=[]
    for q in queries:
        idx=min(max(int(torch.searchsorted(torch.tensor(KNOTS),q,right=True))-1,0),2)
        local.append(torch.lerp(coordinates[idx],coordinates[idx+1],float((q-KNOTS[idx])/(KNOTS[idx+1]-KNOTS[idx]))))
    paths=dict(endpoint_chord=chord,local_chord=torch.stack(local),natural_spline=spline,pullback=optimized)
    current=(recipient.flatten(1)-mean)@basis.T
    sham_direction=torch.randn(32,generator=g).to(wm.device);sham_direction/=sham_direction.norm()
    results=[];saved_predictions={}
    with torch.no_grad():
        truth=b['actual_visual'].to(wm.device);states=b['physical_states'].numpy()
        for method,path in paths.items():
            for i,point in enumerate(path):
                prob,cost,pred,replacement=forward(point)
                if not torch.equal(pred['visual'][:3],native['visual'][:3]):raise ValueError('Prefix changed')
                mse=(pred['visual'][1:]-truth).double().square().flatten(2).mean(2)
                delta=(replacement-recipient).double().flatten(1);off=delta-(delta@basis.double().T)@basis.double()
                results.append(dict(method=method,waypoint=i,coordinate=float(queries[i]),hellinger=float(hellinger2(prob,target[i])),
                    native_cost=cost.tolist(),**score(cost.cpu().numpy(),states,b['goal_state']),
                    actual_visual_MSE_H1_H6=mse.mean(1).tolist(),raw_norm=delta.norm(dim=1).tolist(),offPCA_maxabs=float(off.abs().max())))
        # Exact central query0 is evaluated separately;20 uniform path samples omit0.
        middleweights=spline_weights(torch.linspace(0,1,10),[.5]).to(control)
        middle=dict(endpoint_chord=(coordinates[0]+coordinates[-1])/2,local_chord=(coordinates[1]+coordinates[2])/2,
                    natural_spline=spline_weights(KNOTS,[0]).float().to(wm.device)[0]@coordinates,pullback=decode(middleweights@control)[0])
        for method,point in middle.items():
            prob,cost,pred,replacement=forward(point)
            saved_predictions[method]={k:pred[k].cpu() for k in ('visual','proprio')}
            results.append(dict(method=method,waypoint='central',coordinate=0.,native_cost=cost.tolist(),**score(cost.cpu().numpy(),states,b['goal_state']),
                actual_visual_MSE_H1_H6=(pred['visual'][1:]-truth).double().square().flatten(2).mean(2).mean(1).tolist()))
        # Candidate-wise exact-norm randomPCA32 sham at the primary central query.
        lengths=(middle['pullback']-current).norm(dim=1)
        sham_coordinates=current+lengths[:,None]*sham_direction
        prob,cost,pred,_=forward(sham_coordinates)
        saved_predictions['pullback_norm_sham']={k:pred[k].cpu() for k in ('visual','proprio')}
        results.append(dict(method='pullback_norm_sham',waypoint='central',coordinate=0.,native_cost=cost.tolist(),**score(cost.cpu().numpy(),states,b['goal_state']),
            actual_visual_MSE_H1_H6=(pred['visual'][1:]-truth).double().square().flatten(2).mean(2).mean(1).tolist()))
        results.append(dict(method='native',waypoint='central',coordinate=0.,native_cost=baseline_cost.tolist(),**score(baseline_cost.cpu().numpy(),states,b['goal_state']),
            actual_visual_MSE_H1_H6=(native['visual'][1:]-truth).double().square().flatten(2).mean(2).mean(1).tolist()))
    if parameters_changed := (parameter_sha(wm)!=before):raise ValueError('Weights changed')
    if fingerprint!={k:tensor_hash(z[k]) for k in z.keys()} or tensor_hash(actions)!=action_hash or goals!={k:tensor_hash(agent.goal_state_enc[k]) for k in goals}:raise ValueError('Inputs changed')
    report=dict(complete=True,episode=a.episode,seconds=time.monotonic()-began,history=history,closures=closures,stop=stop,rows=results,
        endpoint_drift=(optimized[[0,-1]]-coordinates[[0,-1]]).norm(dim=1).tolist(),
        fresh_vs_saved_goal_encoding_maxabs=goal_difference,source_density_recipient_maxabs=float((recipient.cpu()-b['original_density_recipient']).abs().max()),
        native_objective_sum_all_diffs=bool(agent.objective.sum_all_diffs),native_objective_alpha=float(agent.objective.alpha),
        limitations=['Conditional same-state fitting to model-generated donor behavior; not independent physical training labels',
            'Behavior softmax is analyst constructed, not native probability; its32D inverse is nonidentifiable',
            'No new physical truth for donor actions; patched original9actions scored against their own unchanged cachedtruth',
            'Four reused development states, not confirmation; no CEM or physical steering efficacy'],candidate_promoted=False)
    write(a.output/'summary.json',report)
    torch.save(dict(report=report,paths={k:v.cpu() for k,v in paths.items()},control=control.detach().cpu(),predictions=saved_predictions,frozen=frozen),a.output/'results.pt')
    outputs=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in (a.output/'protocol.json',a.output/'CANARY.json',a.output/'FROZEN.pt',a.output/'summary.json',a.output/'results.pt')]
    write(a.output/'DONE.json',dict(complete=True,outputs=outputs,episode=a.episode,seconds=time.monotonic()-began,weights_unchanged=True,checkpoint_sha256=CHECKPOINT,provenance=provenance))
    print(json.dumps(dict(event='pullback_complete',episode=a.episode,seconds=time.monotonic()-began)),flush=True)


def analyze(a):
    """CPU-only geometry/forecast separation, after complete source SHA verification."""
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.results/'DONE.json').read_text())
    if not receipt['complete'] or receipt['episode']!=a.episode:raise ValueError('Completed assigned source only')
    for row in receipt['outputs']:
        if sha(a.results/row['path'])!=row['sha256']:raise ValueError('Result SHA before analysis')
    inputs=json.loads((a.inputs/'DONE.json').read_text());source=inputs['outputs'][0]
    if not inputs['complete'] or inputs['episode']!=a.episode or inputs['split']!='development_external':raise ValueError('Input split before load')
    if sha(a.inputs/source['path'])!=source['sha256']:raise ValueError('Input SHA before load')
    data=torch.load(a.results/'results.pt',map_location='cpu',weights_only=False)
    b=torch.load(a.inputs/source['path'],map_location='cpu',weights_only=False)
    report=data['report'];dense=spline_weights(KNOTS,torch.linspace(-4,4,151))@b['donor_centroid_coordinates'].double()
    left=dense[:-1];step=dense[1:]-left;denom=step.square().sum(-1).clamp_min(1e-24)
    geometry={}
    for name,value in data['paths'].items():
        path=value.double();offset=path[:,None]-left
        position=((offset*step).sum(-1)/denom).clamp(0,1)
        distances=(offset-position[:,:,None]*step).square().sum(-1).min(1).values.sqrt()
        variance=(path-path.mean(0)).square().sum()
        chord=path[-1]-path[0];length=chord.norm();arc=(path[1:]-path[:-1]).norm(dim=1).sum()
        displacement=path-path[0];projected=((displacement*chord).sum(1)/chord.square().sum().clamp_min(1e-24))[:,None]*chord
        geometry[name]=dict(mean_closest_reference_distance=float(distances.mean()),rms_closest_reference_distance=float(distances.square().mean().sqrt()),
            reference_recovery_R2=None if variance<1e-20 else float(1-distances.square().sum()/variance),
            path_chord_ratio=None if length<1e-12 else float(arc/length),
            maximum_orthogonal_endpoint_chord_deviation=float((displacement-projected).norm(dim=1).max()),
            endpoint_chord_length=float(length),arc_length=float(arc))
    primary={}
    for name in data['paths']:
        rows=[r for r in report['rows'] if r['method']==name and isinstance(r['waypoint'],int)]
        primary[name]=dict(mean_target_hellinger2=sum(r['hellinger'] for r in rows)/len(rows),waypoints=len(rows))
    central={r['method']:r for r in report['rows'] if r['waypoint']=='central'}
    native=central['native'];physical={}
    states=b['physical_states'].double();goal=torch.as_tensor(b['goal_state']).double()
    block_distance=(states[:,-1,2:4]-goal[2:4]).norm(dim=1)
    angle_delta=states[:,-1,4]-goal[4]
    angle_error=torch.atan2(angle_delta.sin(),angle_delta.cos()).abs()
    for name,row in central.items():
        physical[name]=dict(spearman=row['cost_vs_physical_distance_spearman'],selected_action=row['selected_action_index'],
            selected_progress_delta_vs_native_px=row['selected_actual_progress_px']-native['selected_actual_progress_px'],
            forecast_H3_MSE=row['actual_visual_MSE_H1_H6'][2],forecast_H6_MSE=row['actual_visual_MSE_H1_H6'][5],
            selected_block_goal_distance_px=float(block_distance[row['selected_action_index']]),
            selected_wrapped_angle_error_radians=float(angle_error[row['selected_action_index']]))
    result=dict(complete=True,episode=a.episode,source_done_sha256=sha(a.results/'DONE.json'),script_sha256=sha(__file__),
        model_matching=primary,activation_geometry=geometry,physical_cached_original_actions=physical,
        initial_optimizer_loss=report['closures'][0]['loss'],last_accepted_optimizer_loss=report['history'][-1]['loss'] if report['history'] else None,
        initial_optimizer_diagnostics=report['closures'][0],endpoint_drift=report['endpoint_drift'],stop=report['stop'],
        raw_control_update_norms=[r.get('raw_control_update_norm') for r in report['history']],process_seconds=receipt['seconds'],
        candidate_promoted=False,limitations=['Known action-coordinate centroid spline is a fitted reference, not a validated physical manifold',
            'One pair per each of four reused development states; dependent candidate actions are not independent episodes',
            'Nine-way cost softmax has rank at most8; no unique32D geometry identification',
            'Reference distance/R2 uses the common full32D PCA space, not the papers99percent intrinsic-variance projection',
            'Primary physical progress uses original combined agent/blockXY criterion; block distance and robust angle are separate diagnostic units',
            'No physical target entered optimization; model target fit does not establish actual-world utility',
            'Norm sham is mathematically matched in PCA coordinates; floating-point realized norm was not independently saved'])
    write(a.output/'analysis.json',result)
    write(a.output/'DONE.json',dict(complete=True,outputs=[dict(path='analysis.json',sha256=sha(a.output/'analysis.json'),bytes=(a.output/'analysis.json').stat().st_size)]))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','scale','run','analyze']);p.add_argument('--episode',type=int,choices=range(4),required=True)
    for key in ('output','bank','density','inputs','checkpoint','repo','scale','results'):p.add_argument('--'+key,type=Path)
    p.add_argument('--max-seconds',type=int,default=450)
    a=p.parse_args()
    try:prepare(a) if a.mode=='prepare' else scale_prepare(a) if a.mode=='scale' else analyze(a) if a.mode=='analyze' else run(a)
    except Exception:
        import traceback
        if a.output.exists():write(a.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()))
        raise
