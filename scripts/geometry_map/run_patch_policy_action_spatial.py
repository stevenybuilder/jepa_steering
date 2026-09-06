#!/usr/bin/env python3
"""Fixed-state/current-AdaLN spatial response ablation; no CEM or simulator."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

CHECKPOINT = '9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
KINDS = ('dense', 'mean', 'spatial_permutation')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def exact(a, b, label):
    if not torch.equal(a, b): raise ValueError('Exact guard: '+label)


def scratch(value):
    result = torch.empty_strided(value.shape, value.stride(), device=value.device, dtype=value.dtype)
    return result.copy_(value)


def central_current_condition(z):
    if z.ndim != 3 or z.shape[0] < 2: raise ValueError('Expected batch,time,AdaLN-condition')
    result = scratch(z)
    result[:, -1].copy_(z[0:1, -1].expand_as(result[:, -1]))
    exact(result[:, :-1], z[:, :-1], 'past action conditions unchanged')
    exact(result[0], z[0], 'central condition unchanged')
    return result


def response_variants(delta, minimum_mean_fraction=.01):
    """Same raw L2; mean comparator unavailable when it needs >100x scaling."""
    if delta.ndim != 3 or delta.shape[1] != 256: raise ValueError('Need native256 patch response')
    if not torch.isfinite(delta).all(): raise ValueError('Nonfinite response')
    norm = delta.double().flatten(1).norm(dim=-1)
    mean = delta.mean(1, keepdim=True).expand_as(delta)
    mean_norm = mean.double().flatten(1).norm(dim=-1)
    fraction = torch.where(norm > 0, mean_norm/norm.clamp_min(1e-30), torch.ones_like(norm))
    available = (norm == 0) | (fraction >= minimum_mean_fraction)
    scale = torch.where((norm > 0) & available, norm/mean_norm.clamp_min(1e-30), torch.zeros_like(norm))
    generator = torch.Generator(device='cpu').manual_seed(2026090617)
    permutation = torch.randperm(256, generator=generator).to(delta.device)
    variants = {'dense': delta, 'mean': mean*scale.to(delta)[:, None, None], 'spatial_permutation': delta[:, permutation]}
    for kind, value in variants.items():
        take = available if kind == 'mean' else torch.ones_like(available)
        torch.testing.assert_close(value.double().flatten(1).norm(dim=-1)[take], norm[take], rtol=2e-6, atol=2e-6)
    energy = delta.double().square().sum((1, 2)).clamp_min(1e-30)
    stats = dict(raw_l2=norm.cpu().tolist(), mean_l2_fraction=fraction.cpu().tolist(),
        mean_energy_fraction=(mean_norm.square()/energy).cpu().tolist(), mean_scale=scale.cpu().tolist(),
        mean_available=available.cpu().tolist(), maximum_channel_energy_fraction=(delta.square().sum(1).max(1).values/energy).cpu().tolist(),
        centered_spatial_energy_fraction=((delta-mean).square().sum((1, 2))/energy).cpu().tolist(),
        spatial_permutation=permutation.cpu().tolist(), minimum_mean_fraction=minimum_mean_fraction)
    return variants, available, stats


class CurrentConditionResponse:
    """Recompute one block at exact same x; change only newest condition slot."""
    def __init__(self, block, record=None, kind=None, dose=0., horizon=3):
        self.block, self.record, self.kind, self.dose = block, record, kind, dose
        self.horizon = horizon
        self.calls = 0; self.reentry = False; self.rounding = None

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.hook, with_kwargs=True)
        return self

    def hook(self, module, args, kwargs, output):
        if self.reentry: return None
        self.calls += 1
        if self.calls != self.horizon: return None
        if len(args) != 2 or output.shape[1] != min(self.horizon,2)*256 or output.shape[-1] != 400:
            raise ValueError('Expected native one/two-frame residual window with400 channels')
        x, z = args
        if self.record is None:
            before_x, before_z = x.clone(), z.clone()
            self.reentry = True
            try:
                same = module(*args, **kwargs)
                exact(same, output, 'same-x same-z native block repeat')
                changed_z = central_current_condition(z)
                counter = module(x, changed_z, **kwargs)
            finally: self.reentry = False
            exact(x, before_x, 'caller block input unchanged')
            exact(z, before_z, 'caller action condition unchanged')
            exact(counter[:, :-256], output[:, :-256], 'past output tokens unchanged by newest condition')
            delta = output[:, -256:]-counter[:, -256:]
            exact(delta[0], torch.zeros_like(delta[0]), 'central current-condition null')
            variants, available, stats = response_variants(delta)
            self.record = dict(x=x.detach().clone(), z=z.detach().clone(), counter_z=changed_z.detach().clone(),
                original=output.detach().clone(), counter=counter.detach().clone(), delta=delta.detach().clone(),
                variants=variants, mean_available=available, stats=stats)
            return None
        exact(x, self.record['x'], 'all arms same block state before edit')
        exact(z, self.record['z'], 'all arms same condition before edit')
        exact(output, self.record['original'], 'all arms same native block output before edit')
        if self.dose == 0: return None
        delta = self.record['variants'][self.kind]*self.dose
        result = output.clone(); result[:, -256:] += delta
        rounded = result[:, -256:]-output[:, -256:]
        exact(result[:, :-256], output[:, :-256], 'old tokens not edited')
        exact(result[0], output[0], 'central result exactly unchanged')
        self.rounding = dict(requested_l2=delta.flatten(1).norm(dim=-1).cpu().tolist(),
            delivered_l2=rounded.flatten(1).norm(dim=-1).cpu().tolist(),
            residual_l2=(rounded-delta).flatten(1).norm(dim=-1).cpu().tolist())
        return result

    def __exit__(self, *exc):
        self.handle.remove()
        if exc[0] is None and self.calls != 6: raise ValueError('Expected six native block calls')


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
    return value


@torch.no_grad()
def source(args, row, wm, prep, first):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from diagnose_pusht_goal_coverage import coverage, native_helper
    path = args.bank/row['path']
    if sha(path) != row['sha256']: raise ValueError('Bank SHA mismatch before tensor load')
    bank = torch.load(path, map_location='cpu', weights_only=False)
    if bank['row']['split'] != 'development_external' or len(bank['candidates']) != 9:
        raise ValueError('Expected original near9 development bank')
    context = TensorDict({k:v.repeat_interleave(9, dim=0) for k,v in bank['context'].items()}, batch_size=[]).cuda()
    saved_context = context.clone()
    normalized = prep.normalize_actions(bank['raw_actions'].reshape(9,6,5,2)).reshape(9,6,10)
    exact(normalized, bank['normalized_actions'], 'official action normalization same')
    actions = normalized.transpose(0,1).contiguous().cuda(); saved_actions = actions.clone()
    goal = TensorDict(bank['goal_encoded'], batch_size=[]).cuda()
    objective = ReprTargetDistMPCObjective({}, goal, sum_all_diffs=False, alpha=.1)
    start = time.monotonic()
    baseline = wm.unroll(context.clone(), act_suffix=actions)
    block = wm.model.predictor.predictor_blocks[args.block]
    with CurrentConditionResponse(block,horizon=args.horizon) as capture:
        identity = wm.unroll(context.clone(), act_suffix=actions)
    for key in ('visual','proprio'): exact(identity[key], baseline[key], 'read-only capture identity '+key)
    with CurrentConditionResponse(block, capture.record, 'dense', 0.,horizon=args.horizon):
        zero = wm.unroll(context.clone(), act_suffix=actions)
    for key in ('visual','proprio'): exact(zero[key], baseline[key], 'zero-dose identity '+key)
    predictions = {'native':cpu({k:baseline[k][1:] for k in ('visual','proprio')})}
    costs = {'native':objective(baseline, actions, keepdims=True)[1:].cpu()}
    rounding = {}
    for kind in KINDS:
        for dose in (-.1,.1):
            label = kind+('_minus' if dose < 0 else '_plus')
            with CurrentConditionResponse(block, capture.record, kind, dose,horizon=args.horizon) as patch:
                output = wm.unroll(context.clone(), act_suffix=actions)
            for key in ('visual','proprio'):
                exact(output[key][:args.horizon], baseline[key][:args.horizon], 'pre-edit outputs exact '+label+'/'+key)
                exact(output[key][:,0], baseline[key][:,0], 'center plan exact '+label+'/'+key)
            predictions[label] = cpu({k:output[k][1:] for k in ('visual','proprio')})
            costs[label] = objective(output, actions, keepdims=True)[1:].cpu(); rounding[label] = patch.rounding
    # Offline-only actual targets. No future states are consulted by any edit.
    actual_visual = torch.stack([c['actual_visual'] for c in bank['candidates']], 1).float()
    states = np.stack([c['states'] for c in bank['candidates']])
    raw_proprio = torch.tensor(states[:,::5][:,:,[0,1,5,6]], dtype=torch.float32)
    actual_proprio = wm.model.encode_proprio(prep.normalize_proprios(raw_proprio).cuda()).cpu()[:,1:].transpose(0,1).contiguous()
    actual = {'visual':actual_visual, 'proprio':actual_proprio}
    future_states = states[:,5::5]
    goal_state = np.asarray(bank['goal_state'])
    aligned = np.array([[coverage(s[2:5], goal_state[2:5]) for s in c] for c in future_states]).T
    painted = np.array([[coverage(s[2:5], [256.,256.,np.pi/4]) for s in c] for c in future_states]).T
    native_polygon = native_helper(args.repo/'evals/simu_env_planning/envs/pusht_env/pusht_env.py')
    reference_polygon = native_polygon(goal_state[2:5])
    checked = np.array([native_polygon(s[2:5]).intersection(reference_polygon).area/reference_polygon.area for s in future_states[:,-1]])
    if np.max(np.abs(checked-aligned[-1])) > 1e-12: raise ValueError('Official goal polygon parity')
    xy = np.linalg.norm(future_states[:,:,:4]-goal_state[:4], axis=-1).T
    base_choice = int(costs['native'][-1].argmin()); rows = []
    available = bool(capture.record['mean_available'].all())
    for label, prediction in predictions.items():
        valid = not label.startswith('mean_') or available
        errors = {key:(prediction[key].double()-actual[key].double()).square().flatten(2).mean(-1) for key in actual}
        variation = {key:(prediction[key].double()-prediction[key][:,0:1].double()).square().flatten(2).mean(-1) for key in actual}
        native_change = {key:(prediction[key].double()-predictions['native'][key].double()).square().flatten(2).mean(-1) for key in actual}
        cost = costs[label]; choice = int(cost[-1].argmin()) if valid else None
        rows.append(dict(arm=label, comparison_available=valid,
            unavailable_candidate_indices=[i for i,a in enumerate(capture.record['stats']['mean_available']) if not a] if label.startswith('mean_') else [],
            cost_by_horizon=cost.tolist(), rank_by_horizon=cost.argsort(dim=1,stable=True).tolist(), choice=choice,
            mse_to_actual={k:v.tolist() for k,v in errors.items()},
            equal_candidate_mse_by_horizon={k:v.mean(1).tolist() for k,v in errors.items()},
            mse_to_native={k:v.tolist() for k,v in native_change.items()},
            action_centered_variation_mse={k:v.tolist() for k,v in variation.items()},
            selected_goal_coverage=float(aligned[-1,choice]) if valid else None,
            selected_goal_coverage_delta=float(aligned[-1,choice]-aligned[-1,base_choice]) if valid else None,
            selected_xy_distance_px=float(xy[-1,choice]) if valid else None,
            selected_xy_delta_px=float(xy[-1,choice]-xy[-1,base_choice]) if valid else None,
            selected_painted_goal_coverage=float(painted[-1,choice]) if valid else None,
            selected_native_reward=float(min(painted[-1,choice]/.95,1.)) if valid else None,
            rounding=rounding.get(label)))
    for key in context.keys(): exact(context[key],saved_context[key], 'initial context unchanged '+key)
    exact(actions,saved_actions,'native normalized action array unchanged')
    name=row['key']; tensor=args.output/(name+'.pt')
    torch.save(dict(source_sha256=row['sha256'],raw_actions=bank['raw_actions'],normalized_actions=normalized,
        initial_context=bank['context'],goal_encoded=bank['goal_encoded'],goal_state=bank['goal_state'],
        predictions=predictions,costs=costs,response=cpu(capture.record),actual=actual,states=states,rows=rows),tensor)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    summary=dict(complete=True,key=name,source_sha256=row['sha256'],seconds=seconds,rows=rows,response=capture.record['stats'],
        native_choice=base_choice,goal_aligned_coverage_by_horizon=aligned.tolist(),xy_distance_by_horizon=xy.tolist(),
        native_goal_coverage_headroom=float(aligned[-1].max()-aligned[-1,base_choice]),
        native_xy_headroom_px=float(xy[-1,base_choice]-xy[-1].min()),all_identity_guards_exact=True,
        actual_proprio_source='Saved exact observed states raw0,5,...30 fields[0,1,5,6], official normalize_proprios and frozen encode_proprio; offline after edits',
        full_tensor_sha256=sha(tensor),full_tensor_bytes=tensor.stat().st_size)
    report=tensor.with_suffix('.json');write(report,summary)
    print(json.dumps(dict(event='patch_policy_spatial_state_complete',key=name,seconds=seconds,
        choices={r['arm']:r['choice'] for r in rows},mean_available=available)),flush=True)
    if first:
        estimate=seconds*4+args.init_seconds
        write(args.output/'CANARY.json',dict(complete=True,state_seconds=seconds,estimated_process_seconds=estimate,max_seconds=300))
        print(json.dumps(dict(event='patch_policy_canary',estimated_process_seconds=estimate)),flush=True)
        if estimate>300:raise ValueError('First-state timing exceeds authorized300seconds estimate')
    return [dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in (tensor,report)],summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('bank','repo','checkpoint','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--block',type=int,default=3);parser.add_argument('--horizon',type=int,default=3)
    args=parser.parse_args();started=time.monotonic()
    if (args.block,args.horizon) not in ((3,3),(1,1),(5,3)):
        raise ValueError('Only predeclared primary P3/H3 and followups P1/H1,P5/H3')
    done=json.loads((args.bank/'DONE.json').read_text())
    rows=sorted([r for r in done['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['key'])
    if not done['complete'] or [r['key'] for r in rows]!=[f'near-dev-{i:03d}' for i in range(4)] or any(r['split']!='development_external' for r in rows):
        raise ValueError('Exactly four permitted near-development sources required before load')
    if sha(args.checkpoint)!=CHECKPOINT:raise ValueError('Frozen checkpoint mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        write(args.output/'protocol.json',dict(rows=rows,block=args.block,imagined_horizon=args.horizon,arms=KINDS,doses=[-.1,.1],
            delta='B(x,z_current)-B(x,z_central_current), same full x/past-z/mask/positions; newest256tokens',
            attribution='Conditional current-action effect at selected block; x already includes prior action/state influences, not pure visual interaction',
            mean_minimum_norm_fraction=.01,mean_unavailable_policy='No amplification above100x; zero edit placeholder excluded from comparisons/ranking',
            selected_action_truth='Same immutable candidate physical trajectories, no new simulation',
            goal='Original full-precision encoded goal restored verbatim; goal-matched physical polygon separated from painted native reward',
            checkpoint_sha256=CHECKPOINT,script_sha256=sha(__file__),max_process_seconds=300,
            unit='Four original development states, not36 independent plans',cem_calls=0,simulator_calls=0,held_access=False))
        from model_loader import load_headless
        torch.set_num_threads(2);torch.manual_seed(90505)
        wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
        wm.eval().requires_grad_(False)
        def parameter_sha():
            h=hashlib.sha256()
            for name,p in wm.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before=parameter_sha();args.init_seconds=time.monotonic()-started;outputs=[];summaries=[]
        for i,row in enumerate(rows):
            if time.monotonic()-started>300:raise ValueError('Process budget exhausted; no next state')
            entries,summary=source(args,row,wm,prep,i==0);outputs.extend(entries);summaries.append(summary)
        if parameter_sha()!=before:raise ValueError('Frozen parameter hash changed')
        aggregate=[]
        for index,base in enumerate(summaries[0]['rows']):
            selected=[s['rows'][index] for s in summaries];valid=[r for r in selected if r['comparison_available']]
            aggregate.append(dict(arm=base['arm'],states_available=len(valid),
                choices=[r['choice'] for r in selected],
                mean_goal_coverage_delta=float(np.mean([r['selected_goal_coverage_delta'] for r in valid])) if valid else None,
                mean_xy_delta_px=float(np.mean([r['selected_xy_delta_px'] for r in valid])) if valid else None,
                equal_state_mse_by_horizon={k:np.mean([r['equal_candidate_mse_by_horizon'][k] for r in valid],axis=0).tolist() for k in ('visual','proprio')} if valid else None))
        write(args.output/'summary.json',dict(complete=True,states=4,aggregate=aggregate,
            state_reports=[s['key']+'.json' for s in summaries],seconds=time.monotonic()-started,
            interpretation='Spatial arrangement of a block-local action-condition response; no superposition, all-patch necessity or deployed steering claim'))
        for p in (args.output/'summary.json',args.output/'protocol.json',args.output/'CANARY.json'):
            outputs.append(dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size))
        write(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-started,
            parameter_sha256=before,provenance=provenance,all_identity_guards_exact=True,cem_calls=0,simulator_calls=0,held_access=False))
    except Exception:
        write(args.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
