#!/usr/bin/env python3
"""Eight new development baselines, using the unchanged tested full99 loop.

The reserve stimuli are explicitly allocated by a NEW manifest; their original
reserve receipts and all previous held panels remain immutable.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

EPISODES = tuple(range(66, 74))
SHARDS = {49902461: list(range(66, 70)), 49982193: list(range(70, 74))}
CHECKPOINT = 'c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'
CONFIG = '37400233186d343e7a830ce820b75e86ad134bc26fa1adf43ebc3fd7ed4591c3'


def validate_inputs(manifest, receipt, worker):
    if manifest.get('episode_ids') != list(EPISODES) or manifest.get('split') != 'development_expansion_66_73_v1':
        raise ValueError('Only the frozen eight NEW development starts66..73')
    if worker not in SHARDS or manifest['shards'].get(str(worker)) != SHARDS[worker]:
        raise ValueError('Wrong exclusive four-start shard')
    if manifest.get('previous_full_episode_ids') != list(range(66)):
        raise ValueError('Previous full rollouts0..65 must remain excluded')
    if not receipt.get('complete') or receipt.get('model_execution') is not False or receipt.get('planner_execution') is not False:
        raise ValueError('Require completed model-free prepared reserve before tensor load')
    rows = [r for r in receipt['outputs'] if r['episode'] in SHARDS[worker]]
    if sorted(r['episode'] for r in rows) != SHARDS[worker]:
        raise ValueError('Missing or duplicate prepared source')
    for row in rows:
        e = row['episode']
        if row['environment_seed'] != 2026090500+e or row['planner_seed'] != 90500+e or row['path'] != f'input-{e:03d}.pt':
            raise ValueError('Frozen native seed/path rule changed')
    return sorted(rows, key=lambda r:r['episode'])


@contextmanager
def replace_initializer(module, replacement):
    original = module.reconstruct_fresh
    module.reconstruct_fresh = replacement
    try:
        yield
    finally:
        module.reconstruct_fresh = original


def reconstruct_prepared(cfg, wm, preprocessor, bank):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed
    from collect_on_policy_bank import initialize_episode, physics_snapshot
    from capture_horizon_coordinates import exact, same_physics, tensor_hash
    from causal_planner_forks import restore_cached_goal, close_env
    set_seed(cfg.local_seed)
    env = make_env(cfg)
    try:
        agent = GC_Agent(cfg, wm, dset=None, preprocessor=preprocessor)
        evaluator = PlanEvaluator(cfg, agent)
        td, _, generated_goal, _, _ = initialize_episode(cfg, agent, env, evaluator, bank['episode'], bank['environment_seed'])
        prepared = bank['prepared']
        exact(td['visual'], prepared['initial_visual'], 'original prepared initial pixels')
        exact(td['proprio'], prepared['initial_proprio'], 'original prepared initial proprio')
        same_physics(physics_snapshot(env), prepared['initial_physics'], 'original prepared initial physics')
        goal = restore_cached_goal(generated_goal, {'visual':prepared['goal_visual'], 'proprio':prepared['goal_proprio']})
        agent.set_goal(goal)
        agent.local_gpu_generator.manual_seed(bank['planner_seed'])
        z = wm.encode(td.to(agent.device).unsqueeze(0), act=True)
        agent.capture_raw_context = {k:td[k].cpu().clone() for k in ('visual','proprio')}
        checks = {'initial_prepared_pixels_proprio_physics_exact':True,
                  'fresh_goal_sha256':{k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')},
                  'raw_goal_sha256':{k:tensor_hash(goal[k]) for k in ('visual','proprio')},
                  'goal_mode':'Original prepared raw goal restored identically; fresh native fullprecision encoding',
                  'prepared_source_sha256':bank['prepared_sha256'],
                  'generated_goal_pixel_maxabs_descriptive':float((generated_goal['visual'].float().cpu()-prepared['goal_visual'].float()).abs().max())}
        if 'repeat_reference' in bank:
            ref = bank['repeat_reference']
            for k in ('visual','proprio'): exact(z[k], ref['initial_context'][k], 'pre-CEM repeat context '+k)
            if checks['fresh_goal_sha256'] != ref['fresh_goal_sha256']:
                raise ValueError('Pre-CEM repeat encoded goal changed')
        return env, agent, z, checks
    except Exception:
        close_env(env)
        raise


@torch.no_grad()
def run(args):
    from protocol import file_sha256, write_json_atomic
    manifest = json.loads(args.manifest.read_text())
    receipt = json.loads((args.inputs/'DONE.json').read_text())
    rows = validate_inputs(manifest, receipt, args.instance_id)
    if file_sha256(args.inputs/'DONE.json') != manifest['prepared_done_sha256'] or file_sha256(args.config) != CONFIG:
        raise ValueError('Original prepared receipt or native config differs')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+4500
    outputs, results = [], []
    protocol = dict(manifest, process_pid=os.getpid(), instance_id=args.instance_id,
                    actual_episode_ids=SHARDS[args.instance_id], script_sha256=file_sha256(Path(__file__)),
                    budget_seconds=4500, no_steered_rollouts=True)
    write_json_atomic(args.output/'protocol.json', protocol)
    try:
        sys.path.insert(0, str(args.repo))
        from model_loader import load_headless_metaworld
        from causal_planner_forks import setup_cfg, close_env
        from capture_specificity_controls import parameters_sha
        import capture_horizon_coordinates as reconstruction
        from run_residual_search_full_v1 import run_arm
        torch.set_num_threads(2)
        wm, preprocessor, provenance = load_headless_metaworld(args.repo)
        if file_sha256(Path(provenance['checkpoint'])) != CHECKPOINT:
            raise ValueError('Frozen checkpoint differs')
        wm.eval().requires_grad_(False); before = parameters_sha(wm)
        cfg = setup_cfg(args.config, args.output, wm)
        for row in rows:
            source = args.inputs/row['path']
            if file_sha256(source) != row['sha256']:
                raise ValueError('Prepared source hash mismatch before load')
            prepared = torch.load(source, map_location='cpu', weights_only=False)
            if prepared['episode'] != row['episode'] or prepared['model_execution'] or prepared['planner_execution']:
                raise ValueError('Prepared source identity differs')
            bank = {**{k:row[k] for k in ('episode','environment_seed','planner_seed')},
                    'prepared':prepared, 'prepared_sha256':row['sha256'],
                    'replans':[{'observation_proprio':prepared['initial_proprio']} ]}
            # Cheap reset-only reference validates exact prepared/native initialization
            # and repeated goal/context without spending another planning episode.
            env, agent, z, checks = reconstruct_prepared(cfg, wm, preprocessor, bank)
            canary = {'initial_physics':prepared['initial_physics'],
                      'initial_context':{k:z[k].cpu().clone() for k in ('visual','proprio')},
                      'fresh_goal_sha256':checks['fresh_goal_sha256']}
            close_env(env)
            bank['repeat_reference'] = canary
            with replace_initializer(reconstruction, reconstruct_prepared):
                value = run_arm(cfg, wm, preprocessor, bank, None, None, None, False, None, deadline, args.output)
            from capture_horizon_coordinates import same_physics, exact
            same_physics(value['initial_physics'], canary['initial_physics'], 'reset-only reference physics')
            for k in ('visual','proprio'): exact(value['initial_context'][k], canary['initial_context'][k], 'reset-only reference context '+k)
            if value['fresh_goal_sha256'] != canary['fresh_goal_sha256']:
                raise ValueError('Reset-only reference encoded goal changed')
            value.update(prepared_initial=prepared, prepared_source_sha256=row['sha256'],
                         split=manifest['split'], environment_seed=row['environment_seed'],planner_seed=row['planner_seed'],
                         prepared_initial_and_repeat_context_goal_exact=True)
            value['metrics'].pop('same_initial_physics_and_first300', None)
            value['metrics']['repeat_initial_context_goal_physics_exact'] = True
            value['metrics']['first300_pair_comparison'] = 'Not applicable: one unsteered arm, no paired CEM claimed'
            path = args.output/f"episode-{row['episode']:03d}-unsteered.pt"; torch.save(value, path)
            entry = {'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size}
            write_json_atomic(path.with_suffix('.DONE.json'), {'complete':True,'outputs':[entry]})
            outputs.extend([entry]+value['metrics']['replans']); results.append(value['metrics'])
            write_json_atomic(args.output/'progress.json', {'complete':False,'rows':results,'outputs':outputs})
            print(json.dumps({'event':'new_development_episode_complete','episode':row['episode'],
                              'completed':len(results),'seconds':time.monotonic()-started,
                              'metrics':value['metrics']}), flush=True)
        if parameters_sha(wm) != before:
            raise ValueError('Frozen model weights changed')
        summary = args.output/'summary.json'
        write_json_atomic(summary, {'complete':True,'rows':results,'seconds':time.monotonic()-started,
                                   'provenance':provenance,'parameters_sha256':before,'held_access':False})
        for p in (summary,args.output/'protocol.json'):
            outputs.append({'path':p.name,'sha256':file_sha256(p),'bytes':p.stat().st_size})
        write_json_atomic(args.output/'DONE.json', {'complete':True,'outputs':outputs,'episodes':SHARDS[args.instance_id],
                          'seconds':time.monotonic()-started,'full99':True,'held_access':False})
    except Exception:
        write_json_atomic(args.output/'FAILED.json', {'complete':False,'exception':traceback.format_exc(),'seconds':time.monotonic()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','inputs','repo','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--instance-id',type=int,required=True)
    run(parser.parse_args())
