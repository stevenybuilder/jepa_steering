#!/usr/bin/env python3
"""Bounded model-native coordinates on cached five-plan Reach development banks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

EPISODES = (0, 4, 7)
ARMS = ('unsteered', 'broad0-067-edit', 'broad0-067-sham', 'broad0-126-edit', 'broad0-126-sham')
DOSES = (-1., -.5, .5, 1.)
CHECKPOINT = 'c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def write(path, value): path.write_text(json.dumps(value, indent=2)+'\n')


def action_basis():
    result = torch.zeros(4, 15, 4, dtype=torch.float64)
    for i in range(3): result[i, :, i] = 1
    result[3, :5, 0] = -1; result[3, 10:, 0] = 1
    result = result.flatten(1)
    return (result/result.norm(dim=1, keepdim=True)).float()


def principal_basis(values, rank=3):
    values = values.double(); mean = values.mean(0)
    _, singular, basis = torch.linalg.svd(values-mean, full_matrices=False)
    return mean, basis[:rank], singular[:rank]


def fit_error_target(train_features, train_errors, test_features):
    mean, basis, singular = principal_basis(train_features)
    score = (train_features.double()-mean)@basis.T
    scale = score.std(0, unbiased=False).clamp_min(1e-12)
    design = torch.cat([torch.ones(len(score), 1), score/scale], 1)
    weight = torch.linalg.pinv(design)@train_errors.double()
    test = torch.cat([torch.ones(len(test_features), 1), ((test_features.double()-mean)@basis.T)/scale], 1)
    return test@weight, basis, dict(singular_values=singular.tolist(), training_examples=len(score),
        training_rank=int(torch.linalg.matrix_rank(design)), method='TRAIN-only PCA3 + ordinary least squares with intercept')


def correction(basis, response, target, scales=None):
    b = basis.double(); r = response.double()
    if scales is not None: b = b*scales[:, None]; r = r*scales[:, None]
    gram = r@r.T; damping = .01*torch.diag(gram).mean().clamp_min(1e-18)
    coefficient = torch.linalg.solve(gram+damping*torch.eye(len(gram)), r@target.double())
    return coefficient@b, float(damping)


class P3:
    def __init__(self, block, delta=None): self.block=block; self.delta=delta; self.calls=0; self.value=None; self.pooled=[]
    def __enter__(self): self.handle=self.block.register_forward_hook(self.hook); return self
    def hook(self, module, args, value):
        self.calls += 1
        self.pooled.append(value[:, -256:].detach().float().mean(1).cpu())
        if self.calls != 3: return None
        self.value = value[:, -256:].detach().float().cpu().clone()
        if self.delta is None: return None
        result = value.clone(); result[:, -256:] += self.delta.reshape(-1, 256, 400).to(value)
        return result
    def __exit__(self, *exc):
        self.handle.remove()
        if exc[0] is None and self.calls != 6: raise ValueError('Pinned fullH6 predictor call count changed')


def ranks(x):
    return torch.tensor([float((x < value).sum())+(float((x == value).sum())-1)/2 for value in x], dtype=torch.float64)


def rank_metrics(cost, actual_distance):
    c=ranks(cost); q=ranks(actual_distance); c-=c.mean(); q-=q.mean(); denominator=c.norm()*q.norm()
    chosen=int(cost.argmin()); best=float(actual_distance.min())
    return dict(spearman=float(c@q/denominator) if denominator>0 else None, selected_candidate=chosen,
        selected_actual_distance_m=float(actual_distance[chosen]), regret_m=float(actual_distance[chosen])-best,
        actual_distance_min_ties=int((actual_distance==actual_distance.min()).sum()), predicted_min_ties=int((cost==cost.min()).sum()))


def freeze_inputs(root, goals):
    directories = {0:root.parent/'residual-search-v1/full99-pop0-v2-worker-49766237',
                   4:root/'inputs-ep4', 7:root.parent/'residual-search-v1/remote-backups/from-49902461/full99-pop0-v2'}
    goal_manifest=json.loads((goals/'input_manifest.json').read_text())
    gr={row['episode']:row for row in goal_manifest['sources']}
    rows=[]
    for episode in EPISODES:
        plans=[]
        for arm in ARMS:
            path=directories[episode]/f'episode-{episode:03}-{arm}-replan-00.pt'
            receipt=json.loads(path.with_suffix('.DONE.json').read_text())
            if not receipt['complete'] or len(receipt['outputs'])!=1: raise ValueError('Completed individual plan receipt required')
            expected=receipt['outputs'][0]
            if sha(path)!=expected['sha256']: raise ValueError('Source SHA before loading')
            plans.append(dict(path=str(path),sha256=expected['sha256'],arm=arm))
        goal=gr[episode]
        if goal['split']!='development' or goal['sealed']: raise ValueError('Sealed goal source before load')
        path=goals/goal['path']
        if sha(path)!=goal['sha256']: raise ValueError('Goal source SHA')
        rows.append(dict(episode=episode,split='development',plans=plans,goal_path=str(path),goal_sha256=goal['sha256']))
    result=dict(complete=True,rows=rows,episodes=list(EPISODES),plan_arms=list(ARMS),physical_horizons=[1,2,3],
                held_sources=0,selection='All five previously optimized development first-replan plans; no outcome selection')
    write(root/'INPUTS_FROZEN.json',result)
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ('root','goals','repo','config'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=True); output=a.root/'results-v1'; output.mkdir(exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(9050607)
    sources=freeze_inputs(a.root,a.goals)
    protocol=dict(complete=False,episodes=list(EPISODES),rank=3,block=3,patch_horizon=3,tokens='all newest256,400 channels',
        doses=list(DOSES),residual_radius_fraction=.005,response_probe_fraction=.001,raw_action_probe_xyz_rms=.025,
        primary='H3 native objective ranking versus matching actual15-step hand-goal distance; n3 states',
        secondary=['Full spatial actual-encoded H3 MSE','H1/H2 unchanged','Action tangent selectivity','Raw applied perturbation norms'],
        released_H6_cost_is_secondary_prefix_association_not_horizonmatched_truth=True,
        target='Shared leave-one-initial-state-out PCA3/OLS actual-encoded H3 forecast residual; held state labels never enter edits',
        families=['train_pca','local_action_jacobian','local_action_jacobian_whitened'],
        whitening='Geometric-mean singular sensitivity divided by singular value, floor1e-3 of maximum; fixed .01 relative ridge damping',
        rotation_null='Orthogonal basis rotation and inverse preserve ridge patch; verified algebraically before reusing identical field',
        action_jacobian='MODEL derivative to raw nominal commands, not measured simulator Jacobian; original clipping mismatch reported',
        conditions=18,simulator_calls=0,cem_calls=0,held_data=False,script_sha256=sha(__file__),max_process_seconds=900,
        literature=['https://arxiv.org/html/2608.18746v1','https://arxiv.org/abs/2608.12828'])
    write(output/'protocol.json',protocol)
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from causal_planner_forks import setup_cfg
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from tensordict import TensorDict
    from intervene_head_spatial_mean import batch_context
    wm,prep,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False)
    if sha(provenance['checkpoint'])!=CHECKPOINT: raise ValueError('Wrong checkpoint')
    before=parameters_sha(wm); cfg=setup_cfg(a.config,output,wm); began=time.monotonic(); data=[]; block=wm.model.predictor.predictor_blocks[3]
    ab=action_basis(); probe=.025*(45**.5)
    for source in sources['rows']:
        plans=[]
        for row in source['plans']:
            if sha(row['path'])!=row['sha256']: raise ValueError('Input changed')
            value=torch.load(row['path'],map_location='cpu',weights_only=False)
            if value['episode']!=source['episode'] or value['replan']!=0 or value['arm']!=row['arm']: raise ValueError('Development identity failed')
            plans.append(value)
        for value in plans[1:]:
            for key in ('visual','proprio'):
                if not torch.equal(value['encoded_context'][key],plans[0]['encoded_context'][key]): raise ValueError('Shared initial context changed')
        if sha(source['goal_path'])!=source['goal_sha256']: raise ValueError('Goal input changed')
        goal_source=torch.load(source['goal_path'],map_location='cpu',weights_only=False)
        raw_goal={k:goal_source['goal'][k].clone() for k in ('visual','proprio')}
        agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);agent.set_goal(TensorDict(raw_goal,batch_size=[]))
        normalized=torch.stack([v['selected_full_plan'] for v in plans]); raw=prep.denormalize_actions(normalized.reshape(5,30,4))
        for i,value in enumerate(plans):
            if not torch.equal(raw[i,:15],value['raw_commands_executed']): raise ValueError('Same executed15 raw commands required')
        z=batch_context(plans[0]['encoded_context'],5).to(wm.device); act=normalized.transpose(0,1).contiguous().to(wm.device)
        native=wm.unroll(z.clone(),act_suffix=act)
        with P3(block) as capture: repeat=wm.unroll(z.clone(),act_suffix=act)
        for key in ('visual','proprio'):
            if not torch.equal(native[key],repeat[key]): raise ValueError('Native capture parity failed')
        actual=torch.stack([v['actual_executed_prefix_encodings'][2]['visual'].reshape(1,16,16,384) for v in plans])
        state=torch.stack([torch.as_tensor(v['states'][-1]).double() for v in plans])
        distance=(state[:,:3]-state[:,-3:]).norm(dim=1)
        jacobians=[]; singular_values=[]; cross_batch=[]
        for candidate in range(5):
            deltas=torch.cat([torch.zeros(1,60),probe*ab,-probe*ab]).reshape(9,15,4)
            perturbed=raw[candidate].repeat(9,1,1);perturbed[:,:15]+=deltas
            official=prep.normalize_actions(perturbed.reshape(9,6,5,4)).reshape(9,6,20)
            center=prep.normalize_actions(raw[candidate].reshape(6,5,4)).reshape(6,20)
            normalized_path=normalized[candidate][None]+official-center[None]
            normalized_path[0]=normalized[candidate]
            z9=batch_context(plans[0]['encoded_context'],9).to(wm.device)
            with P3(block) as captured: forecast=wm.unroll(z9,act_suffix=normalized_path.transpose(0,1).contiguous().to(wm.device))
            j=(captured.value[1:5].double()-captured.value[5:9].double()).flatten(1)/(2*probe)
            _,s,b=torch.linalg.svd(j,full_matrices=False);jacobians.append(b[:3]);singular_values.append(s[:3])
            cross_batch.append(float((captured.value[0]-capture.value[candidate]).abs().max()))
        data.append(dict(source=source,plans=plans,agent=agent,z=z,actions=act,raw=raw,native=native,
            residual=capture.value.flatten(1),actual=actual,distance=distance,jac_basis=jacobians,jac_s=singular_values,
            cross_batch_residual_maxabs=cross_batch,native_p3_pooled_history=torch.stack(capture.pooled)))
        print(json.dumps(dict(event='native_and_action_jacobian_capture_done',episode=source['episode'],seconds=time.monotonic()-began)),flush=True)
        if len(data)==1:
            estimate=3*(time.monotonic()-began)
            write(output/'CAPTURE_CANARY.json',dict(complete=True,projected_capture_seconds=estimate,limit=300))
            if estimate>300:raise RuntimeError('Fixed capture timing budget exceeded')
    outputs=[];summary=[]
    for index,d in enumerate(data):
        tick=time.monotonic(); train=[other for j,other in enumerate(data) if j!=index]
        x=torch.cat([v['residual'] for v in train]); errors=torch.cat([v['actual']-v['native']['visual'][3].cpu() for v in train]).flatten(1)
        requested,pca,target_fit=fit_error_target(x,errors,d['residual']); corrections={k:[] for k in protocol['families']}; response_records=[]
        for candidate in range(5):
            norm=float(d['residual'][candidate].double().norm()); local=d['jac_basis'][candidate]; local_s=d['jac_s'][candidate]
            responses={}
            for family,basis in (('train_pca',pca),('local_action_jacobian',local)):
                radius=.001*norm
                delta=torch.cat([torch.zeros(1,len(basis[0])),radius*basis,-radius*basis]).float()
                plan=d['actions'][:,candidate:candidate+1].repeat(1,7,1).contiguous()
                z7=batch_context(d['plans'][0]['encoded_context'],7).to(wm.device); native=wm.unroll(z7.clone(),act_suffix=plan)
                with P3(block,delta) as patch: shifted=wm.unroll(z7.clone(),act_suffix=plan)
                for key in ('visual','proprio'):
                    if not torch.equal(shifted[key][:,0],native[key][:,0]) or not torch.equal(shifted[key][:3],native[key][:3]): raise ValueError('Response exact self/prefix guard')
                response=(shifted['visual'][3,1:4].double()-shifted['visual'][3,4:7].double()).flatten(1).cpu()/(2*radius)
                responses[family]=response
                vector,damping=correction(basis,response,requested[candidate]);corrections[family].append(vector)
                response_records.append(dict(candidate=candidate,family=family,radius=radius,damping=damping,
                    signed_curvature_l2=float((shifted['visual'][3,1:4]+shifted['visual'][3,4:7]-2*native['visual'][3,:3]).double().norm())))
            stable=local_s.clamp_min(local_s.max()*1e-3); scales=stable.log().mean().exp()/stable
            vector,damping=correction(local,responses['local_action_jacobian'],requested[candidate],scales)
            corrections['local_action_jacobian_whitened'].append(vector)
            generator=torch.Generator().manual_seed(9050600+d['source']['episode']*10+candidate)
            rotation=torch.linalg.qr(torch.randn(3,3,generator=generator,dtype=torch.float64)).Q
            null,_=correction(rotation@pca,rotation@responses['train_pca'],requested[candidate])
            torch.testing.assert_close(null,corrections['train_pca'][-1],atol=1e-9,rtol=1e-8)
        fields={}; norm_records={}; residual_norm=d['residual'].double().norm(dim=1)
        for family,vectors in corrections.items():
            value=torch.stack(vectors); length=value.norm(dim=1)
            unit=value/length[:,None].clamp_min(1e-20)
            fields[family]=(unit*(.005*residual_norm[:,None])).float()
            torch.testing.assert_close(fields[family].double().norm(dim=1),.005*residual_norm,rtol=5e-6,atol=1e-8)
            norm_records[family]=dict(unscaled_norm=length.tolist(),zero_direction=(length<1e-20).tolist())
        generator=torch.Generator().manual_seed(9050619+d['source']['episode'])
        random=torch.randn(3,d['residual'].shape[1],generator=generator,dtype=torch.float64)
        random_basis=torch.linalg.qr(random.T,mode='reduced').Q.T
        coefficients=torch.randn(5,3,generator=generator,dtype=torch.float64); sham=coefficients@random_basis
        sham=sham/sham.norm(dim=1,keepdim=True)*(.005*residual_norm[:,None]);fields['rank3_random_sham']=sham.float()
        conditions=[('exact_self',torch.zeros_like(d['residual']))]
        for family,field in fields.items():
            for dose in DOSES:conditions.append((family+f'/dose{dose:+g}',field*dose))
        conditions.append(('pca_rotation_null',fields['train_pca']))
        rows=[];predictions=[]
        native_cost=d['agent'].objective(d['native'],d['actions'],keepdims=True).float().cpu().reshape(7,5)
        for name,delta in conditions:
            if time.monotonic()-began>900:raise RuntimeError('Process budget exhausted before next independent condition')
            with P3(block,delta) as patch: changed=wm.unroll(d['z'].clone(),act_suffix=d['actions'])
            for key in ('visual','proprio'):
                if not torch.equal(changed[key][:3],d['native'][key][:3]): raise ValueError('Unchanged H1/H2 guard')
                if name=='exact_self' and not torch.equal(changed[key],d['native'][key]): raise ValueError('Exact self failed')
            cost=d['agent'].objective(changed,d['actions'],keepdims=True).float().cpu().reshape(7,5)
            visual=changed['visual'][1:].float().cpu(); actual_error=(visual[2]-d['actual']).square().flatten(1).mean(1)
            actual_delta=(patch.value.reshape(5,-1)+delta)-patch.value.reshape(5,-1)
            selectivity=[float((actual_delta[i].double()@d['jac_basis'][i].T).norm()/actual_delta[i].double().norm())
                         if actual_delta[i].norm()>0 else None for i in range(5)]
            rows.append(dict(condition=name,H3_native_cost=cost[3].tolist(),H6_native_cost=cost[6].tolist(),
                H3_physical_ranking=rank_metrics(cost[3],d['distance']),H6_cost_vs15step_prefix_only=rank_metrics(cost[6],d['distance']),
                actual_H3_visual_mse=actual_error.tolist(),mean_actual_H3_visual_mse=float(actual_error.mean()),
                raw_requested_norm=delta.double().norm(dim=1).tolist(),raw_realized_norm=actual_delta.double().norm(dim=1).tolist(),
                action_tangent_norm_fraction=selectivity,H1_H2_exact=True))
            predictions.append(visual)
        native_error=(d['native']['visual'][3].cpu()-d['actual']).square().flatten(1).mean(1)
        report=dict(complete=True,episode=d['source']['episode'],training_episodes=[v['source']['episode'] for v in train],
            rows=rows,native_H3_ranking=rank_metrics(native_cost[3],d['distance']),native_H3_cost=native_cost[3].tolist(),
            native_actual_H3_visual_mse=native_error.tolist(),actual_H3_goal_distance_m=d['distance'].tolist(),
            target_fit=target_fit,response_records=response_records,direction_norms=norm_records,
            local_action_jacobian_singular_values=[s.tolist() for s in d['jac_s']],
            cross_batch_residual_maxabs=d['cross_batch_residual_maxabs'],
            original_raw_command_abs_gt1_counts=(d['raw'][:,:15,:3].abs()>1).sum((1,2)).tolist(),
            held_labels_used_in_fit=False,rotation_null_algebra_passed=True,seconds=time.monotonic()-tick)
        pt=output/f'episode-{d["source"]["episode"]:03}.pt'
        torch.save(dict(report=report,conditions=[n for n,_ in conditions],patched_visual=torch.stack(predictions),
            native_visual=d['native']['visual'][1:].float().cpu(),actual_H3_visual=d['actual'],residual=d['residual'],
            fields=fields,pca_basis=pca,local_action_basis=d['jac_basis'],requested_output_error=requested,
            native_p3_pooled_history=d['native_p3_pooled_history'],selected_plans=d['actions'].cpu(),source=d['source']),pt)
        report['tensor_sha256']=sha(pt);write(pt.with_suffix('.json'),report)
        outputs.extend(dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size) for f in (pt,pt.with_suffix('.json')))
        summary.append(report)
        print(json.dumps(dict(event='coordinate_state_complete',episode=d['source']['episode'],seconds=time.monotonic()-began)),flush=True)
        if index==0:
            estimate=(time.monotonic()-began)+2*(time.monotonic()-tick)
            write(output/'CANARY.json',dict(complete=True,estimated_process_seconds=estimate,limit=900,no_outcome_gate=True))
            if estimate>900:raise RuntimeError('Timing-only15min estimate exceeded')
    if parameters_sha(wm)!=before:raise ValueError('Weights changed')
    write(output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-began,model=provenance,
        parameter_sha256=before,states=3,conditions=18,simulator_calls=0,cem_calls=0,held_data=False,
        limitations=['Three adaptively explored development states; not confirmation',
            'Five historical optimized action plans per state, not random independent candidates or full nativeCEM evaluation',
            'Actual truth only through15raw/H3; H6 cost is only associated with executed prefix outcomes',
            'Action Jacobian differentiates model nominal commands, not true physical dynamics; native environment clips some commands',
            'Output correction target is TRAIN-only associational OLS; causal finite response is separately measured',
            'Whitening changes regularized metric; a pure orthogonal rotation leaves mapped patch invariant',
            'Fixed norm projection can amplify a poor predicted target; all signed/null/sham outcomes retained']))


if __name__=='__main__':
    try:main()
    except Exception:
        import sys,traceback
        if '--root' in sys.argv:
            failed=Path(sys.argv[sys.argv.index('--root')+1])/'results-v1'/'FAILED.json'
            if failed.parent.exists() and not failed.exists():write(failed,dict(complete=False,exception=traceback.format_exc()))
        raise
