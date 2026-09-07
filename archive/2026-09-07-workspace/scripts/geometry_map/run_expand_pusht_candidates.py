#!/usr/bin/env python3
"""Outcome-blind 64-plan Push development bank, with exact paired physics.

Original nine plans plus 27 antithetic normalized-space pairs and raw-zero.
No CEM, fitting, steering, held source, or outcome-based candidate filtering.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

SCALES = (.25, .5, 1.)
CHECKPOINT = '9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'


def expanded_actions(raw, normalized, episode, prep):
    if episode not in range(4) or raw.shape != (9,30,2) or normalized.shape != (9,6,10):
        raise ValueError('Only original nine-plan banks from development0..3')
    if raw.dtype != torch.float32 or normalized.dtype != torch.float32:
        raise ValueError('Original action precision required')
    if not torch.isfinite(raw).all() or not torch.isfinite(normalized).all():
        raise ValueError('Nonfinite source actions')
    rng = np.random.default_rng(2026090600+episode)
    directions = rng.normal(size=(9,6,10))
    directions /= np.sqrt(np.mean(directions**2, axis=(1,2), keepdims=True))
    raw_rows = list(raw.clone()); norm_rows = list(normalized.clone())
    metadata = [dict(candidate=i, family='original_nine', original_index=i) for i in range(9)]
    for direction in range(9):
        v = torch.from_numpy(directions[direction]).float()
        for scale in SCALES:
            for sign in (-1,1):
                proposed = normalized[0]+sign*scale*v
                command = prep.denormalize_actions(proposed.reshape(30,2)).float()
                effective = prep.normalize_actions(command).reshape(6,10).float()
                metadata.append(dict(candidate=len(raw_rows), family='normalized_antithetic',
                    direction=direction, normalized_rms_radius=scale, sign=sign,
                    nominal_normalized_sha256=hashlib.sha256(proposed.numpy().tobytes()).hexdigest(),
                    normalize_roundtrip_maxabs=float((effective-proposed).abs().max())))
                raw_rows.append(command); norm_rows.append(effective)
    zero = torch.zeros((30,2),dtype=torch.float32)
    metadata.append(dict(candidate=63,family='zero_raw_command',note='Native relative target equals current agent position'))
    raw_rows.append(zero); norm_rows.append(prep.normalize_actions(zero).reshape(6,10).float())
    return torch.stack(raw_rows), torch.stack(norm_rows), metadata, directions


def spearman(a,b):
    def rank(x):
        order=np.argsort(x,kind='stable');result=np.empty(len(x));i=0
        while i<len(x):
            j=i+1
            while j<len(x) and x[order[j]]==x[order[i]]: j+=1
            result[order[i:j]]=(i+j-1)/2; i=j
        return result
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite equal-length scalar score vectors required')
    a,b=rank(a),rank(b)
    return None if np.ptp(a)==0 or np.ptp(b)==0 else float(np.corrcoef(a,b)[0,1])


def summarize(rows, goal, initial):
    states=np.asarray([r['final_state'] for r in rows]); goal=np.asarray(goal); initial=np.asarray(initial)
    cost=np.asarray([r['native_full_objective_cost'] for r in rows])
    encoded=np.asarray([r.get('actual_encoded_full_objective_cost',r['native_full_objective_cost']) for r in rows])
    result={}
    for name,indices in [('full7d_state_l2',list(range(7))),('joint_xy_distance_px',list(range(4)))]:
        distance=np.linalg.norm(states[:,indices]-goal[indices],axis=1)
        start=float(np.linalg.norm(initial[indices]-goal[indices])); choice=int(np.argmin(cost))
        result[name]=dict(distance_by_action=distance.tolist(), progress_by_action=(start-distance).tolist(),
            native_cost_vs_actual_distance_spearman=spearman(cost,distance),
            actual_encoded_cost_vs_actual_distance_spearman=spearman(encoded,distance),
            predicted_cost_vs_actual_encoded_cost_spearman=spearman(cost,encoded),
            native_bank_argmin=choice, native_bank_argmin_actual_distance=float(distance[choice]),
            original_cem_action0_actual_distance=float(distance[0]), oracle_index=int(np.argmin(distance)),
            oracle_headroom_vs_native_argmin=float(distance[choice]-distance.min()),
            oracle_headroom_vs_original_action0=float(distance[0]-distance.min()),
            original9_oracle_headroom_vs_original_action0=float(distance[0]-distance[:9].min()),
            new64_oracle_gain_vs_original9=float(distance[:9].min()-distance.min()))
    if all('native_reward_final' in r for r in rows):
        coverage=np.asarray([r['native_reward_final'] for r in rows])
        result['native_reward_goal_coverage']=dict(negative_predicted_cost_vs_coverage_spearman=spearman(-cost,coverage),
            negative_actual_encoded_cost_vs_coverage_spearman=spearman(-encoded,coverage),
            native_selected_coverage=float(coverage[np.argmin(cost)]),oracle_coverage=float(coverage.max()),
            oracle_headroom=float(coverage.max()-coverage[np.argmin(cost)]),coverage_by_action=coverage.tolist(),
            scope='Native environment reward goal_pose may differ from original expert physical/encoded goal')
    return result


@torch.no_grad()
def run(a):
    from complete_cached_geometry import sha256,write_json
    from collect_curvature_interior_futures import select_sources,require_reset_link,replay,compare_replays,exact
    from capture_pusht_horizon import encoder_inputs
    from capture_horizon_coordinates import CaptureP3Horizons,tensor_hash
    from run_action_path_curvature import ResidualReplacement
    from collect_pusht_bank import config
    from capture_pusht_calibration import parameter_sha
    from action_consequence_diagnostics import actual_cost_state,physical_metrics
    from model_loader import load_headless
    from tensordict import TensorDict
    selected=select_sources(json.loads((a.bank/'DONE.json').read_text()),json.loads((a.prepared_inputs/'DONE.json').read_text()))
    _,source,reset_row=selected[a.episode]
    for directory,row in ((a.bank,source),(a.prepared_inputs,reset_row)):
        if sha256(directory/row['path'])!=row['sha256']: raise ValueError('Source hash failed before load')
    if sha256(a.checkpoint)!=CHECKPOINT: raise ValueError('Pinned checkpoint mismatch')
    protocol=dict(episode=a.episode,split='development_external',source=source,reset_source=reset_row,
        plan_count=64, original9_exact=True, new_candidates=55, directions=9, antithetic_signs=[-1,1],
        scales_normalized_rms=SCALES, seed=2026090600+a.episode, proposal_center='original selected H6 normalized plan',
        proposal_rule='nine independent Gaussian60D directions normalized to RMS1, allthree radii and both signs, plus raw-zero',
        action_contract='Unclipped native relative XY multiplied100; declared Box0..512 is absolute target metadata, not delta bound',
        saved_cem_populations_available=False, paper_comparison='Motivated by2608.18746; NOT its random/mid/elite CEM-stage study',
        raw_steps=30,horizons=[1,2,3,4,5,6],frame_skip=5,p3_capture='block3 residual newest256 spatial tokens at imagined3',
        primary='Candidate variation/headroom and within-state native full objective vs actual full7D state L2',
        secondary='Joint4D XY distance/progress, native coverage reward, source-goal success, allhorizon errors',
        statistical_units='Four previously seen development states, not256 independent episodes',
        paired_physics_repeats=2,no_outcome_filter=True,held_access=False,operator_fit=False,cem_calls=0,
        max_process_seconds=a.max_seconds, checkpoint_sha256=CHECKPOINT,
        code_sha256={n:sha256(Path(__file__).with_name(n)) for n in
            ('run_expand_pusht_candidates.py','collect_curvature_interior_futures.py','capture_horizon_coordinates.py','model_loader.py')})
    write_json(a.output/'protocol.json',protocol)
    bank=torch.load(a.bank/source['path'],map_location='cpu',weights_only=False)
    reset=torch.load(a.prepared_inputs/reset_row['path'],map_location='cpu',weights_only=False)
    initial,seed=require_reset_link(bank,reset,reset_row['sha256'],a.episode)
    if bank['row']['split']!='development_external' or not bank['same_state_actions_and_two_physical_replays_exact']:
        raise ValueError('Bad development source')
    goal=np.asarray(bank['goal_state']);exact(goal,reset['goal_state'],'original physical goal')
    cfg=config(a.repo,a.output)
    wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm)
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    for key in ('visual','proprio'):
        if bank['goal_encoded'][key].dtype!=torch.float32: raise ValueError('Fullprecision source goal required')
    target=TensorDict({k:v.clone().cuda() for k,v in bank['goal_encoded'].items()},batch_size=[])
    objective=ReprTargetDistMPCObjective(cfg,target_enc=target,**cfg.planner.planning_objective)
    z=TensorDict({k:v.clone().cuda() for k,v in bank['context'].items()},batch_size=[])
    context_hash={k:tensor_hash(z[k]) for k in z.keys()};goal_hash={k:tensor_hash(target[k]) for k in target.keys()}
    raw,normalized,meta,directions=expanded_actions(bank['raw_actions'],bank['normalized_actions'],a.episode,prep)
    exact(raw[:9],bank['raw_actions'],'original nine raw actions');exact(normalized[:9],bank['normalized_actions'],'original nine normalized actions')
    frozen=a.output/'FROZEN_INPUTS.pt'
    torch.save(dict(raw_actions=raw,normalized_actions=normalized,metadata=meta,directions=directions,
        original_context=bank['context'],raw_goal=bank['raw_goal'],goal_encoded=bank['goal_encoded'],goal_state=goal,
        original_raw_reset=reset,source=source,reset_source=reset_row,episode=a.episode,split='development_external'),frozen)
    write_json(a.output/'FROZEN_INPUTS.json',dict(complete=True,path=frozen.name,sha256=sha256(frozen),bytes=frozen.stat().st_size,
        frozen_before_any_new_physical_outcome=True,metadata=meta,raw_min=float(raw.min()),raw_max=float(raw.max()),
        normalized_min=float(normalized.min()),normalized_max=float(normalized.max()),clipping_applied=False,
        context_sha256=context_hash,goal_sha256=goal_hash))
    env=make_env(cfg);env_info=reset.get('env_info',{'shape':'T'});rows=[];files=[frozen,a.output/'FROZEN_INPUTS.json',a.output/'protocol.json']
    def initialize():
        env.update_env(env_info);obs,info=env.prepare(seed,initial.copy(),env_info)
        exact(obs,bank['candidates'][0]['frames_modelsteps'][0],'source initial pixels')
        exact(info['state'],bank['candidates'][0]['states'][0],'source initial observed state')
        if env.unwrapped.window_size!=512 or not env.unwrapped.relative or env.unwrapped.action_scale!=100:
            raise ValueError('Native relative control domain changed')
        return obs,info
    try:
        for i in range(64):
            start=time.monotonic();saved=raw[i].clone();obs,info=initialize();truth=replay(env,obs,info,raw[i],goal)
            obs,info=initialize();repeat=replay(env,obs,info,raw[i],goal);compare_replays(truth,repeat)
            exact(raw[i],saved,'raw actions unchanged')
            if i<9:
                old=bank['candidates'][i]
                for key in ('states','native_applied_targets_xy'): exact(truth[key],old[key],'original nine '+key)
                exact(truth['frames'][::5],old['frames_modelsteps'],'original nine physical pixels')
                if json.dumps(truth['physics'],sort_keys=True)!=json.dumps(old['physics'],sort_keys=True):
                    raise ValueError('Original nine full physics differs')
            plan=normalized[i,:,None].contiguous().cuda();native=wm.unroll(z.clone(),act_suffix=plan)
            with CaptureP3Horizons(wm.model.predictor.predictor_blocks[3]) as cap:
                observed=wm.unroll(z.clone(),act_suffix=plan)
            for k in ('visual','proprio'): exact(native[k],observed[k],'read-only hook '+k)
            if len(cap.values)!=6: raise ValueError('Six imagined horizons required')
            p3=cap.values[2].float().cpu()
            if p3.shape!=(1,256,400): raise ValueError('Unexpected P3/H3 spatial shape')
            if i==0:
                with ResidualReplacement(wm.model.predictor.predictor_blocks[3],3,p3.cuda()):
                    identity=wm.unroll(z.clone(),act_suffix=plan)
                for k in ('visual','proprio'): exact(native[k],identity[k],'exact P3H3 self '+k)
            actual=wm.encode(encoder_inputs(truth['frames'][::5],truth['proprios'][::5]))
            again=wm.encode(encoder_inputs(repeat['frames'][::5],repeat['proprios'][::5]))
            for k in ('visual','proprio'): exact(actual[k],again[k],'paired actual encoding '+k)
            av=actual['visual'][0,1:].float().cpu();pv=native['visual'][1:,0].float().cpu()
            ac=actual_cost_state(actual,native)
            cost=objective(native,plan).reshape(-1)
            if cost.numel()!=1: raise ValueError('Native scalar objective expected')
            byh=objective(native,plan,keepdims=True).float().cpu();actualcost=objective(ac,plan,keepdims=True).float().cpu()
            metrics=physical_metrics(truth['states'][5::5],goal)
            report=dict(complete=True,episode=a.episode,**meta[i],final_state=truth['states'][-1].tolist(),
                native_full_objective_cost=float(cost.item()),native_cost_by_horizon=byh.reshape(-1).tolist(),
                actual_encoded_full_objective_cost=float(objective(ac,plan).item()),
                actual_encoded_native_cost_by_horizon=actualcost.reshape(-1).tolist(),
                visual_mse_to_actual_by_horizon=(pv-av).double().square().reshape(6,-1).mean(1).tolist(),
                full7d_distance_by_horizon=np.linalg.norm(truth['states'][5::5]-goal,axis=1).tolist(),
                metrics={k:np.asarray(v).tolist() for k,v in metrics.items()},native_reward_sum=float(truth['native_rewards'].sum()),
                native_reward_final=float(truth['native_rewards'][-1]),native_reward_max=float(truth['native_rewards'].max()),
                ever_native_goal_success=bool(truth['native_goal_success'].any()),final_native_goal_success=bool(truth['native_goal_success'][-1]),
                replay_exact=True,encoder_repeat_exact=True,readonly_capture_exact=True,original_physics_exact=True if i<9 else None,
                historical_model_visual_maxabs=float((pv-bank['candidates'][i]['predicted_visual']).abs().max()) if i<9 else None,
                raw_command_min=float(raw[i].min()),raw_command_max=float(raw[i].max()),
                applied_pixel_targets_outside_window_fraction=float(np.mean((truth['native_applied_targets_xy']<0)|(truth['native_applied_targets_xy']>512))),
                no_clip=True,seconds=time.monotonic()-start)
            pt=a.output/f'candidate-{i:03d}.pt'
            torch.save(dict(episode=a.episode,split='development_external',candidate=i,metadata=meta[i],
                source_sha256=source['sha256'],reset_input_sha256=reset_row['sha256'],frozen_inputs_sha256=sha256(frozen),
                raw_actions=raw[i],normalized_actions=normalized[i],truth=truth,p3_h3=p3,
                predicted_visual=pv,predicted_proprio=native['proprio'][1:,0].float().cpu(),actual_visual=av,
                actual_proprio=actual['proprio'][0,1:].float().cpu(),native_cost_by_horizon=byh,
                actual_encoded_native_cost_by_horizon=actualcost,metrics=metrics),pt)
            report.update(tensor_sha256=sha256(pt),tensor_bytes=pt.stat().st_size)
            write_json(pt.with_suffix('.json'),report);files.extend([pt,pt.with_suffix('.json')]);rows.append(report)
            torch.cuda.synchronize();seconds=time.monotonic()-start
            print(json.dumps(dict(event='candidate_complete',episode=a.episode,candidate=i,seconds=seconds)),flush=True)
            if i==0:
                estimate=time.monotonic()-a.started+seconds*63
                canary=dict(complete=True,first_source_physics_exact=True,self_identity_exact=True,paired_replay_exact=True,
                    seconds=seconds,estimated_process_seconds=estimate,max_seconds=a.max_seconds,no_outcome_gate=True)
                write_json(a.output/'CANARY.json',canary);files.append(a.output/'CANARY.json')
                print(json.dumps(dict(event='canary',**canary)),flush=True)
                if estimate>a.max_seconds: raise RuntimeError('Canary timing exceeds authorized budget')
            if time.monotonic()-a.started>a.max_seconds: raise RuntimeError('Authorized process budget reached')
    finally: env.close()
    if before!=parameter_sha(wm): raise ValueError('Frozen weights changed')
    if context_hash!={k:tensor_hash(z[k]) for k in z.keys()} or goal_hash!={k:tensor_hash(target[k]) for k in target.keys()}:
        raise ValueError('Context or original fullprecision goal changed')
    write_json(a.output/'summary.json',dict(complete=True,episode=a.episode,plan_count=64,rows=rows,
        scores=summarize(rows,goal,bank['candidates'][0]['states'][0]),native_objective_sum_all_diffs=bool(objective.sum_all_diffs),
        native_objective_alpha=float(objective.alpha),goal_state=goal.tolist(),
        reward_scope='Native coverage uses saved environment goal_pose; goal_success compares original expert goal_state',
        states_count=1,independent_states_total=4,source9_exact=True,all64_paired_physics_exact=True,held_access=False))
    files.append(a.output/'summary.json')
    write_json(a.output/'DONE.json',dict(complete=True,episode=a.episode,plans=64,physics_replays=128,raw_simulator_steps=3840,
        seconds=time.monotonic()-a.started,parameter_sha256=before,provenance=provenance,held_access=False,cem_calls=0,
        outputs=[dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in files]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','prepared-inputs','repo','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--episode',type=int,choices=range(4),required=True);p.add_argument('--max-seconds',type=float,default=450)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(20260906)
    try: run(a)
    except Exception as error:
        (a.output/'FAILED.json').write_text(json.dumps(dict(complete=False,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-a.started),indent=2))
        raise


if __name__=='__main__': main()
