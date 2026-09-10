#!/usr/bin/env python3
"""Two exact physical replays of 64 fixed, unseen-interior PushT action plans.

No predictor unroll, CEM, fitted readout, or intervention. Original prepared raw
reset inputs are mandatory: PushT reset itself advances physics by 0.01 seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

CHECKPOINT = '9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
TIMES = (-.25, .25)


def hybrid_actions(donor, center, patch_horizon):
    if donor.shape != (30,2) or center.shape != (30,2) or patch_horizon not in (1,3):
        raise ValueError('Only H1/H3 hybrids; full H6 truth must be reused by hash')
    result=center.clone()
    result[:5*patch_horizon]=donor[:5*patch_horizon]
    exact(result[:5*patch_horizon],donor[:5*patch_horizon],'hybrid donor prefix')
    exact(result[5*patch_horizon:],center[5*patch_horizon:],'hybrid central suffix')
    return result


def action_sha(actions):
    return hashlib.sha256(actions.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def missing_path_actions(raw,pair,radius,t):
    allowed=(-.5,.5) if radius==1 else (-1.,-.5,.5,1.) if radius==4 else ()
    if t not in allowed: raise ValueError('Only missing coefficients ±.5/±2/±4; reuse0/±1')
    _,meta=interior_actions(raw,pair,radius,.25)
    delta=raw[1+2*pair]-raw[0]
    actions=raw[0]+(radius*t)*delta
    meta.update(t=t,raw_delta_coefficient=radius*t,action_offset_l2=float((actions-raw[0]).norm()),
        action_sha256=action_sha(actions),action_sha_format='contiguous float32 raw30x2 bytes')
    return actions,meta


def select_sources(near_receipt, prepared_receipt):
    """Freeze/filter receipt splits before opening any tensor, including reset."""
    if not near_receipt.get('complete') or not prepared_receipt.get('complete'):
        raise ValueError('Incomplete source receipt')
    rows = sorted([r for r in near_receipt['outputs'] if r['path'].endswith('.pt')], key=lambda r: r['key'])
    if [r['key'] for r in rows] != [f'near-dev-{i:03d}' for i in range(4)]:
        raise ValueError('Exactly four original near-development sources required')
    selected = []
    for episode, row in enumerate(rows):
        reset = [r for r in prepared_receipt['episodes'] if r['episode'] == episode]
        if row['split'] != 'development_external' or len(reset) != 1 or reset[0]['split'] != 'development':
            raise ValueError('Forbidden or ambiguous split before tensor loading')
        selected.append((episode, row, reset[0]))
    return selected


def interior_actions(raw, pair, radius, t):
    if raw.shape != (9, 30, 2) or raw.dtype != torch.float32 or not torch.isfinite(raw).all():
        raise ValueError('Require original finite float32 nine-plan near bank')
    if pair not in range(4) or radius not in (1, 4) or t not in TIMES:
        raise ValueError('Only frozen four directions, radii1/4, unseen t±.25')
    center = raw[0]
    delta = raw[1 + 2*pair] - center
    opposite = raw[2 + 2*pair] - center
    tolerance = 4*torch.finfo(raw.dtype).eps*max(1., float(raw.abs().max()))
    if float((delta + opposite).abs().max()) > tolerance:
        raise ValueError('Source direction is not antithetic within saved rounding')
    actions = center + (radius*t)*delta
    return actions, dict(pair=pair, radius=radius, t=t, positive_candidate=1+2*pair,
        delta_l2=float(delta.norm()), action_offset_l2=float((actions-center).norm()),
        antithetic_rounding_maxabs=float((delta+opposite).abs().max()), clipping_applied=False)


def exact(a, b, name):
    if torch.is_tensor(a): a = a.detach().cpu().numpy()
    if torch.is_tensor(b): b = b.detach().cpu().numpy()
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise RuntimeError('Exact replay guard failed: '+name)


def require_reset_link(bank, reset, reset_sha, episode):
    if bank['reset_input_sha256'] != reset_sha or reset['split'] != 'development' or reset['episode'] != episode:
        raise ValueError('Original prepared reset source mismatch')
    if 'initial_state' not in reset or 'environment_seed' not in reset:
        raise ValueError('Raw prepared initial state and seed are required')
    # Deliberately never fall back to observed candidate states[0].
    return np.asarray(reset['initial_state']).copy(), int(reset['environment_seed'])


@torch.no_grad()
def replay(env, observation, info, actions, goal):
    from collect_pusht_bank import snapshot
    from precompute_native_replay import PushContactCapture
    frames, states, proprios = [observation.cpu().clone()], [np.asarray(info['state']).copy()], [torch.as_tensor(info['proprio']).cpu().clone()]
    physics, controls, contacts, rewards, dones, success = [snapshot(env)], [], [], [], [], []
    with PushContactCapture(env.unwrapped) as capture:
        for step, command in enumerate(actions):
            capture.records.clear()
            obs, reward, done, infos = env.step_multiple(command[None])
            if len(obs) != 1 or env.elapsed_steps() != step+1:
                raise RuntimeError('Unexpected physical raw-step count or early stop')
            state = np.asarray(infos[0]['state']).copy()
            frames.append(obs[0].cpu().clone()); states.append(state)
            proprios.append(torch.as_tensor(infos[0]['proprio']).cpu().clone())
            physics.append(snapshot(env)); controls.append(np.asarray(env.unwrapped.latest_action).copy())
            rewards.append(float(reward[0])); dones.append(None if done[0] is None else bool(done[0]))
            success.append(bool(env.eval_state(goal, state)['success']))
            contacts.append(dict(raw_step=step+1, records=list(capture.records),
                agent_block_contact=any(r['agent_block_contact'] for r in capture.records) if capture.available else None))
        available = capture.available
    return dict(frames=torch.stack(frames).to(torch.uint8), states=np.stack(states),
        proprios=torch.stack(proprios), physics=physics, native_applied_targets_xy=np.stack(controls),
        native_rewards=np.asarray(rewards), native_dones=dones, native_goal_success=np.asarray(success),
        contacts=contacts, contact_api_available=available,
        native_reward_goal_pose=np.asarray(env.unwrapped.goal_pose).copy())


def compare_replays(a, b):
    for key in ('frames', 'states', 'proprios', 'native_applied_targets_xy', 'native_rewards', 'native_goal_success', 'native_reward_goal_pose'):
        exact(a[key], b[key], key)
    for key in ('physics', 'contacts', 'native_dones', 'contact_api_available'):
        if json.dumps(a[key], sort_keys=True) != json.dumps(b[key], sort_keys=True):
            raise RuntimeError('Exact replay guard failed: '+key)


@torch.no_grad()
def collect_source(args, selected, wm, prep, cfg, first):
    from tensordict import TensorDict
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from capture_pusht_horizon import encoder_inputs
    from action_consequence_diagnostics import physical_metrics
    from complete_cached_geometry import sha256, write_json
    episode, row, reset_row = selected
    source, reset_path = args.bank/row['path'], args.prepared_inputs/reset_row['path']
    if sha256(source) != row['sha256'] or sha256(reset_path) != reset_row['sha256']:
        raise ValueError('Source checksum mismatch before tensor loading')
    bank = torch.load(source, map_location='cpu', weights_only=False)
    reset = torch.load(reset_path, map_location='cpu', weights_only=False)
    initial, seed = require_reset_link(bank, reset, reset_row['sha256'], episode)
    if bank['row']['split'] != 'development_external' or bank['row']['key'] != row['key']:
        raise ValueError('Near-bank tensor split mismatch')
    goal = np.asarray(bank['goal_state'])
    exact(goal, reset['goal_state'], 'original physical goal')
    for key in ('visual', 'proprio'):
        if bank['goal_encoded'][key].dtype != torch.float32:
            raise ValueError('Original fullprecision goal required; no promoted half precision')
    target = TensorDict({k:v.clone().to('cuda:0') for k,v in bank['goal_encoded'].items()}, batch_size=[])
    if cfg.planner.planning_objective.objective_type != 'L2': raise ValueError('Pinned native L2 objective changed')
    objective = ReprTargetDistMPCObjective(cfg, target_enc=target, **cfg.planner.planning_objective)
    for key in ('visual', 'proprio'): exact(objective.target_enc[key], bank['goal_encoded'][key], 'fullprecision effective goal '+key)
    env = make_env(cfg)
    env_info = reset.get('env_info', {'shape':'T'})
    def initialize():
        env.update_env(env_info)
        obs, info = env.prepare(seed, initial.copy(), env_info)
        exact(obs, bank['candidates'][0]['frames_modelsteps'][0], 'near-bank initial pixels')
        exact(info['state'], bank['candidates'][0]['states'][0], 'near-bank initial observed state')
        if env.unwrapped.window_size != 512 or not env.unwrapped.relative or env.unwrapped.action_scale != 100:
            raise ValueError('Native Push relativeXY control domain changed')
        return obs, info
    outputs = []
    if args.path_truth:
        # Supply central observations without another physical rollout. State
        # layout is independently checked against native saved proprio below.
        central_record=bank['candidates'][0]
        probe_name=f"{row['key']}-pair0-radius4-tplus025.pt"
        probe=torch.load(args.donor_truth/probe_name,map_location='cpu',weights_only=False)
        state_proprio=probe['truth']['states'][:,[0,1,5,6]]
        exact(state_proprio,probe['truth']['proprios'].reshape(31,4),'native state-to-proprio mapping')
        raw_proprio=torch.as_tensor(central_record['states'][::5][:,[0,1,5,6]]).float().reshape(1,7,4)
        encoded_prop=wm.encode_proprio(prep.normalize_proprios(raw_proprio).to('cuda:0')).float().cpu()[0]
        if encoded_prop.shape!=(7,256,16): raise ValueError('Central proprio feature layout changed')
        encoded_prop[0]=bank['context']['proprio'][0,0]
        encoded_vis=torch.cat([bank['context']['visual'][0,:1],central_record['actual_visual']],dim=0)
        central_path=args.output/(row['key']+'-central-reused.pt')
        torch.save(dict(complete=True,key=row['key'],source_sha256=row['sha256'],source_path=str(source),
            original_context=bank['context'],actual_visual_with_initial=encoded_vis,
            actual_proprio_with_initial=encoded_prop,raw_proprio_with_initial=raw_proprio[0],
            observed_states=central_record['states'][::5],frames_modelsteps=central_record['frames_modelsteps'],
            raw_actions=bank['raw_actions'][0],normalized_actions=bank['normalized_actions'][0],
            raw_delta_coefficient=0.,physical_rollouts_added=0,state_to_native_proprio_mapping_exact=True,
            initial_context_reused_bitexact=True,source_actual_visual_reused=True,
            proprio_method='Observed [agentX,agentY,agentVX,agentVY] through official normalize_proprios + encode_proprio; cached initial overrides only index0'),central_path)
        central_report=dict(complete=True,key=row['key'],source_sha256=row['sha256'],physical_rollouts_added=0,
            full_tensor_sha256=sha256(central_path),full_tensor_bytes=central_path.stat().st_size)
        write_json(central_path.with_suffix('.json'),central_report)
        for path in (central_path,central_path.with_suffix('.json')):
            outputs.append(dict(path=path.name,sha256=sha256(path),bytes=path.stat().st_size))
    try:
        for pair in range(4):
            for radius in (1, 4):
                targets=([(t,None) for t in ((-.5,.5) if radius==1 else (-1.,-.5,.5,1.))] if args.path_truth else
                    [(t,h) for t in TIMES for h in (1,3)] if args.hybrid else [(t,None) for t in TIMES])
                for t, patch_horizon in targets:
                    started = time.monotonic()
                    actions, meta = (missing_path_actions if args.path_truth else interior_actions)(bank['raw_actions'], pair, radius, t)
                    donor_name=f"{row['key']}-pair{pair}-radius{radius}-t{'minus' if t<0 else 'plus'}025.pt"
                    donor=None
                    if args.hybrid:
                        donor_row=args.donor_entries[donor_name]
                        donor_path=args.donor_truth/donor_name
                        if sha256(donor_path)!=donor_row['sha256']: raise ValueError('Saved full-donor truth hash mismatch')
                        donor=torch.load(donor_path,map_location='cpu',weights_only=False)
                        if donor['source_sha256']!=row['sha256'] or donor['reset_input_sha256']!=reset_row['sha256']:
                            raise ValueError('Donor truth source/reset mismatch')
                        exact(actions,donor['raw_actions'],'same saved full-donor plan')
                        actions=hybrid_actions(actions,bank['raw_actions'][0],patch_horizon)
                        meta.update(patch_horizon=patch_horizon,donor_prefix_steps=5*patch_horizon,
                            central_suffix_steps=30-5*patch_horizon,full_donor_path=str(donor_path),
                            full_donor_sha256=donor_row['sha256'],donor_action_sha256=action_sha(donor['raw_actions']),
                            central_action_sha256=action_sha(bank['raw_actions'][0]),
                            hybrid_action_sha256=action_sha(actions),action_sha_format='contiguous float32 raw30x2 bytes',
                            action_prefix_suffix_exact=True)
                    actions_copy = actions.clone()
                    obs, info = initialize(); truth = replay(env, obs, info, actions, goal)
                    obs, info = initialize(); repeat = replay(env, obs, info, actions, goal)
                    compare_replays(truth, repeat); exact(actions, actions_copy, 'raw commands unchanged')
                    if args.hybrid:
                        end=5*patch_horizon+1
                        for key in ('frames','states','proprios'): exact(truth[key][:end],donor['truth'][key][:end],'saved donor physical prefix '+key)
                        if json.dumps(truth['physics'][:end],sort_keys=True)!=json.dumps(donor['truth']['physics'][:end],sort_keys=True):
                            raise RuntimeError('Saved donor full-body physical prefix mismatch')
                        meta['saved_donor_physical_prefix_exact']=True
                    encoded_input = encoder_inputs(truth['frames'][::5], truth['proprios'][::5])
                    actual = wm.encode(encoded_input)
                    actual_repeat = wm.encode(encoder_inputs(repeat['frames'][::5], repeat['proprios'][::5]))
                    for key in ('visual','proprio'): exact(actual[key], actual_repeat[key], 'paired frozen encoder '+key)
                    normalized = prep.normalize_actions(actions.reshape(1,6,5,2)).reshape(6,10)
                    actual_tb = TensorDict({k:actual[k].transpose(0,1) for k in ('visual','proprio')}, batch_size=[])
                    costs = objective(actual_tb, normalized[:,None].to('cuda:0'), keepdims=True).float().cpu()[1:]
                    actual_visual = actual['visual'][0,1:].float().cpu()
                    if actual_visual.shape != (6,1,16,16,384): raise ValueError('Unexpected native spatial truth layout')
                    metrics = physical_metrics(truth['states'][5::5], goal)
                    name = f"{row['key']}-pair{pair}-radius{radius}-t{'minus' if t<0 else 'plus'}{round(abs(t)*100):03d}"
                    if args.hybrid: name+=f'-H{patch_horizon}'
                    pt = args.output/(name+'.pt')
                    value = dict(complete=True, episode=episode, key=row['key'], split='development_external', **meta,
                        source_sha256=row['sha256'], reset_input_sha256=reset_row['sha256'], environment_seed=seed,
                        raw_prepared_initial_state=initial, reset_contract='Original raw prepared state; native reset advances physics once',
                        raw_actions=actions, normalized_actions=normalized, truth=truth,
                        actual_visual=actual_visual, actual_proprio=actual['proprio'][0,1:].float().cpu(),
                        actual_encoded_native_cost_by_horizon=costs, goal_state=goal,
                        raw_goal=bank['raw_goal'], goal_encoded=bank['goal_encoded'], metrics=metrics,
                        raw_offsets=[5,10,15,20,25,30], replay_identity_exact=True, encoder_repeat_exact=True,
                        goal_mode='Immutable original fullprecision encoded target; never re-encoded',
                        reward_scope='Native coverage reward uses recorded environment goal_pose; native_goal_success uses original expert goal_state',
                        predictor_calls=0, cem_calls=0, simulator_raw_steps=60)
                    if args.path_truth:
                        exact(truth['states'][:,[0,1,5,6]],truth['proprios'].reshape(31,4),'actual state-to-proprio mapping')
                        value.update(original_context=bank['context'],
                            actual_visual_with_initial=torch.cat([bank['context']['visual'][0,:1],value['actual_visual']]),
                            actual_proprio_with_initial=torch.cat([bank['context']['proprio'][0,:1],value['actual_proprio']]),
                            initial_context_reused_bitexact=True,state_to_native_proprio_mapping_exact=True)
                    torch.save(value, pt)
                    torch.cuda.synchronize(); seconds = time.monotonic()-started
                    report = dict(complete=True, episode=episode, key=row['key'], **meta, seconds=seconds,
                        full_tensor_sha256=sha256(pt), full_tensor_bytes=pt.stat().st_size,
                        source_sha256=row['sha256'], reset_input_sha256=reset_row['sha256'],
                        replay_identity_exact=True, encoder_repeat_exact=True, fullprecision_goal_exact=True,
                        actual_encoded_native_cost_by_horizon=costs.reshape(-1).tolist(),
                        goal_state=goal.tolist(), final_state=truth['states'][-1].tolist(),
                        native_reward_sum=float(truth['native_rewards'].sum()),
                        native_reward_goal_pose=truth['native_reward_goal_pose'].tolist(),
                        final_native_goal_success=bool(truth['native_goal_success'][-1]),
                        ever_native_goal_success=bool(truth['native_goal_success'].any()),
                        metrics={k:np.asarray(v).tolist() for k,v in metrics.items()})
                    write_json(pt.with_suffix('.json'), report)
                    for path in (pt,pt.with_suffix('.json')):
                        outputs.append(dict(path=path.name, sha256=sha256(path), bytes=path.stat().st_size))
                    print(json.dumps(dict(event='interior_truth_complete', name=name, seconds=seconds)), flush=True)
                    if first:
                        upper = seconds*args.plan_count + args.initialization_seconds
                        canary = dict(complete=True, paired_replay_exact=True, seconds=seconds,
                            estimated_total_process_seconds=upper, limit_seconds=args.max_seconds, no_outcome_gate=True)
                        write_json(args.output/'CANARY.json', canary)
                        print(json.dumps(dict(event='interior_canary', **canary)), flush=True)
                        if upper > args.max_seconds: raise RuntimeError('First fixed-plan canary exceeds authorized process estimate')
                        first = False
                    if time.monotonic()-args.process_start > args.max_seconds:
                        raise RuntimeError('Authorized process time exhausted; no next plan launched')
    finally:
        env.close()
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bank','prepared-inputs','repo','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hybrid',action='store_true',help='H1/H3 donor-prefix/central-suffix truth; H6 reused, never rerun')
    p.add_argument('--path-truth',action='store_true',help='Only96 missing coefficients±.5/±2/±4; reuse0/±1')
    p.add_argument('--donor-truth',type=Path)
    args = p.parse_args()
    from complete_cached_geometry import sha256, write_json
    from collect_pusht_bank import config
    from capture_pusht_calibration import parameter_sha
    from model_loader import load_headless
    args.process_start = time.monotonic()
    near = json.loads((args.bank/'DONE.json').read_text())
    prepared = json.loads((args.prepared_inputs/'DONE.json').read_text())
    selected = select_sources(near, prepared)
    if args.hybrid and args.path_truth: raise ValueError('Hybrid and full-path protocols are distinct')
    args.plan_count=128 if args.hybrid else 96 if args.path_truth else 64
    args.max_seconds=300 if args.hybrid or args.path_truth else 900
    reused=[]
    if args.hybrid or args.path_truth:
        if args.donor_truth is None: raise ValueError('Hybrid truth requires immutable full-donor reference')
        donor_receipt=json.loads((args.donor_truth/'DONE.json').read_text())
        if not donor_receipt.get('complete') or donor_receipt['plans']!=64 or donor_receipt.get('held_access') is not False:
            raise ValueError('Only complete original64 development donor truths may be reused')
        args.donor_entries={r['path']:r for r in donor_receipt['outputs'] if r['path'].endswith('.pt')}
        expected={f"near-dev-{ep:03d}-pair{pair}-radius{radius}-t{sign}025.pt" for ep in range(4) for pair in range(4) for radius in (1,4) for sign in ('minus','plus')}
        if set(args.donor_entries)!=expected: raise ValueError('Wrong frozen64 donor source set')
        for name,row in args.donor_entries.items():
            if sha256(args.donor_truth/name)!=row['sha256']: raise ValueError('Full H6 reuse checksum mismatch')
            if args.hybrid or '-radius4-' in name:
                reused.append(dict(path=str(args.donor_truth/name),sha256=row['sha256'],bytes=row['bytes'],
                    patch_horizon=6 if args.hybrid else None,raw_delta_coefficient=(-1. if 'tminus' in name else 1.) if args.path_truth else None,replayed_again=False))
    if sha256(args.checkpoint) != CHECKPOINT: raise ValueError('Pinned checkpoint checksum mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(args.output/'protocol.json', dict(episodes=list(range(4)), pair_indices=list(range(4)), radii=[1,4],
            t=list(TIMES), plans=args.plan_count, raw_steps_per_plan=30, exact_replays_per_plan=2, max_process_seconds=args.max_seconds,
            action_formula='Donor a0+(radius*t)*(positive-a0) through5H, then original central actions' if args.hybrid else 'a0+(radius*t)*(saved_positive_candidate-a0)', source_rows=selected,
            hybrid=args.hybrid,path_truth=args.path_truth,process_pid=os.getpid(),patch_horizons=[1,3] if args.hybrid else None,
            reused_H6=reused if args.hybrid else [],reused_coefficient_one=reused if args.path_truth else [],
            missing_coefficients=[-.5,.5,-2.,2.,-4.,4.] if args.path_truth else None,
            central_supplement_no_physics=args.path_truth,
            action_novelty='Withheld from each curve fit, not globally new: radius4*t±.25 revisits original radius1 endpoints up to saved antithetic rounding',
            checkpoint_sha256=CHECKPOINT, script_sha256=sha256(__file__),
            held_access=False, cem_calls=0, predictor_calls=0, future_truth_only_for_offline_evaluation=True,
            unit='Four fixed original development states; directions/radii/interiors are dependent probes',
            encoded_goal='Saved fullprecision original goal, no recomputation'))
        torch.set_num_threads(2); torch.manual_seed(90505)
        cfg = config(args.repo,args.output)
        wm, prep, provenance = load_headless(args.repo, model_name='jepa_wm_pusht', checkpoint_override=args.checkpoint)
        wm.eval().requires_grad_(False); before = parameter_sha(wm)
        args.initialization_seconds = time.monotonic()-args.process_start
        outputs = []
        for i, source in enumerate(selected): outputs.extend(collect_source(args,source,wm,prep,cfg,i==0))
        if parameter_sha(wm) != before: raise RuntimeError('Frozen weights changed')
        write_json(args.output/'DONE.json', dict(complete=True, outputs=outputs, plans=args.plan_count, exact_replays=2*args.plan_count,
            hybrid=args.hybrid,path_truth=args.path_truth,process_pid=os.getpid(),reused_H6=reused if args.hybrid else [],
            reused_coefficient_one=reused if args.path_truth else [],central_supplements=4 if args.path_truth else 0,
            seconds=time.monotonic()-args.process_start, parameter_sha256=before, provenance=provenance,
            all_replays_and_encoder_exact=True, original_initial_pixels_states_exact=True,
            goal_encoded_restored_exact=True, no_goal_reencoding=True, held_access=False, cem_calls=0, predictor_calls=0))
    except Exception:
        write_json(args.output/'FAILED.json', dict(complete=False, exception=traceback.format_exc()))
        raise


if __name__ == '__main__': main()
