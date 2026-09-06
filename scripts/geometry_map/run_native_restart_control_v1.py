#!/usr/bin/env python3
"""Two fixed native CEM restart seeds; only planner RNG changes, no selection."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from coordinate_calibration_operator import CoordinateCalibration
from run_coordinate_dose_timing import run_arm,exact,DoseTimingHook
from run_planner_compensation_v1 import restore_fixed_encoded_goal,CHECKPOINT_SHA
from protocol import file_sha256,write_json_atomic

SEEDS=(817,1801)
EPISODES=(0,4,7)


def apply_restart_seed(agent,snapshot,seed):
    if seed not in SEEDS:raise ValueError('Only two predetermined restart seeds')
    agent.local_gpu_generator.manual_seed(seed)
    snapshot=dict(snapshot);snapshot['planner_rng']=agent.local_gpu_generator.get_state().cpu().clone()
    return snapshot


def checked_baseline(directory,episode):
    d=json.loads((directory/'DONE.json').read_text())
    entries=[x for x in d['outputs'] if x.get('episode')==episode and x.get('arm')=='unsteered']
    if not d['complete'] or d['held_sources_opened'] or len(entries)!=1:raise RuntimeError('Frozen development baseline required')
    e=entries[0];p=directory/e['path']
    if not p.resolve().is_relative_to(directory.resolve()) or file_sha256(p)!=e['sha256']:raise RuntimeError('Baseline SHA/path mismatch')
    return torch.load(p,map_location='cpu',weights_only=False),e


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','bank','baseline-dir','candidate-dir','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start_time=time.monotonic();outputs=[]
    from causal_planner_forks import load_development_bank,setup_cfg
    from model_loader import load_headless_metaworld
    import run_coordinate_steering_pilot as pilot
    bank,banksha=load_development_bank(a.bank);episode=bank['episode']
    if episode not in EPISODES:raise RuntimeError('Only fixed restart development starts0/4/7')
    baseline,entry=checked_baseline(a.baseline_dir,episode)
    candidate=a.candidate_dir/'coordinate_candidate.npz';frozen=json.loads((a.candidate_dir/'FROZEN.json').read_text())
    if file_sha256(candidate)!=frozen['candidate_sha256']:raise RuntimeError('Frozen observer/candidate SHA mismatch')
    protocol={'episode':episode,'restart_seeds':SEEDS,'intervention':'none: unchanged native model and objective',
              'only_changed_variable':'native planner local CUDA generator seed AFTER fixed environment/model initialization',
              'candidate_populations':'NOT matched to original baseline or other restart seeds',
              'native_cem':{'H':6,'samples':300,'iterations':15},'raw_steps_per_seed':15,
              'source_baseline':entry,'bank_sha256':banksha,'candidate_sha256':frozen['candidate_sha256'],
              'checkpoint_sha256':CHECKPOINT_SHA,'goal_mode':'same immutable baseline raw and full-precision encoded target',
              'held_sources_opened':False,'all_seeds_retained':True,'best_restart_selection':False,
              'budget_seconds':1500,'script_sha256':file_sha256(Path(__file__))}
    write_json_atomic(a.output_dir/'protocol.json',protocol)
    try:
        wm,preprocessor,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False)
        if file_sha256(Path(provenance['checkpoint']))!=CHECKPOINT_SHA:raise RuntimeError('Frozen checkpoint changed')
        cfg=setup_cfg(Path(provenance['config']),a.output_dir,wm);block=wm.model.predictor.predictor_blocks[3]
        calibration=CoordinateCalibration(dict(np.load(candidate)),device='cuda:0')
        original_start=pilot.fresh_start;current_seed=None
        def restart_start(cfg,wm,preprocessor,bank):
            env,agent,td,snapshot,mode=original_start(cfg,wm,preprocessor,bank)
            mode=mode|restore_fixed_encoded_goal(agent,snapshot,baseline['initial_snapshot'])
            pilot.paired_start_checks(snapshot,baseline['initial_snapshot'])
            if current_seed is not None:snapshot=apply_restart_seed(agent,snapshot,current_seed)
            return env,agent,td,snapshot,mode
        pilot.fresh_start=restart_start
        # Source-host/runtime preservation check before using the historical baseline comparison.
        env,agent,td,snapshot,mode=restart_start(cfg,wm,preprocessor,bank)
        try:
            with torch.no_grad():
                z=wm.encode(td.cuda().unsqueeze(0),act=True);actions=baseline['selected_plan_fullH6'][:,None].cuda()
                native=wm.unroll(z.clone(),act_suffix=actions)
                with DoseTimingHook(wm,block,calibration,0.) as hook:
                    hook.reset();observed=wm.unroll(z.clone(),act_suffix=actions)
                for k in ('visual','proprio'):exact(native[k],observed[k],'native zero identity '+k)
                obs,_,_,infos=env.step_multiple(baseline['raw_actions'].clone())
                exact(torch.stack(obs).cpu(),baseline['frames'],'source baseline command replay pixels')
                exact(np.stack([np.asarray(i['state']) for i in infos]),baseline['states'],'source baseline command replay states')
        finally:pilot.close_env(env)
        print(json.dumps({'event':'restart_native_identity_and_baseline_replay_passed','episode':episode}),flush=True)
        for seed in SEEDS:
            if time.monotonic()-start_time>1500:raise RuntimeError('Fixed restart worker budget elapsed')
            current_seed=seed
            expected=copy.deepcopy(baseline['initial_snapshot'])
            generator=torch.Generator(device='cuda:0').manual_seed(seed);expected['planner_rng']=generator.get_state().cpu().clone()
            print(json.dumps({'event':'restart_native_cem_started','episode':episode,'seed':seed}),flush=True)
            result=run_arm(cfg,wm,preprocessor,bank,calibration,block,(f'restart_{seed}',0.,False,None),expected,None)
            result['restart_seed']=seed;result['candidate_populations_matched']=False
            costs=[]
            for source in (result,baseline):
                forecast=source['selected_prediction'];snapshot=source['initial_snapshot']
                visual=(forecast['visual']-snapshot['goal_encoded_visual']).square().flatten(2).mean(-1)
                proprio=(forecast['proprio']-snapshot['goal_encoded_proprio']).square().flatten(2).mean(-1)
                costs.append(float((visual+.1*proprio)[-1,0]))
            result['selected_native_terminal_goal_cost']=costs[0]
            path=a.output_dir/f'restart-{seed}.pt';torch.save(result,path)
            row={'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size,'episode':episode,'seed':seed,
                 'goal_progress_m':result['goal_progress_m'],'progress_change_vs_original_baseline_mm':1000*(result['goal_progress_m']-baseline['goal_progress_m']),
                 'ever_success':result['ever_success'],'first300_min_native_cost':float(result['first_candidates']['costs'].min()),
                 'selected_native_terminal_goal_cost':costs[0],'original_baseline_native_terminal_goal_cost':costs[1],
                 'seconds':result['seconds']}
            outputs.append(row);write_json_atomic(a.output_dir/'progress.json',{'outputs':outputs})
            print(json.dumps({'event':'restart_complete',**row}),flush=True)
        write_json_atomic(a.output_dir/'DONE.json',protocol|{'complete':True,'outputs':outputs,'model':provenance,'seconds':time.monotonic()-start_time})
    except Exception as exc:
        write_json_atomic(a.output_dir/'FAILED.json',{'error':repr(exc),'outputs':outputs});raise


if __name__=='__main__':main()
