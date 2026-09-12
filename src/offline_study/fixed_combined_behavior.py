"""Paired Reach extension: two new combined arms, five unchanged references."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .author_fit import source_hash
from .backends import JepaBackend
from .behavioral_analysis import shard_paths
from .behavioral_development import DEVELOPMENT_SEED, assigned_rows, schedule, validate_coverage, verified_report
from .fixed_combined import ARM_COMPONENTS, METHOD, CombinedFixedResponseIntervention
from .fixed_combined_check import exclusive_json, model_signature, verify_bindings, verify_files
from .fixed_combined_smoke import EPISODES, ObservedCombined, validate_record, verify_numerical
from .fixed_response_behavior import (ARMS as REFERENCE_ARMS, CONTRASTS as REFERENCE_CONTRASTS,
    checked_source, coupling_binding, device_uuid, stimulus_contract, validate_protocol, verify_cem)
from .fixed_response_check import CountedBackend
from .fixed_response_smoke import trace_actor, verify_episode, verify_pair
from .planning_contract import prepare
from .planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from .planning_scenarios import close_expert_environments
from .protocol import sha256, write_json
from .vendor import use_vendor


TASK = 'reach'
NEW_ARMS = tuple(ARM_COMPONENTS)
ARMS = (*REFERENCE_ARMS, *NEW_ARMS)
CONTRASTS = (*REFERENCE_CONTRASTS, (NEW_ARMS[0], 'native'), (NEW_ARMS[0], NEW_ARMS[1]),
             (NEW_ARMS[0], 'fixed_rank4'), (NEW_ARMS[0], 'coupling_only'))
ROLE = 'paired_refined_combination_reach_development_extension'
ANALYSIS = {'family': 8, 'replicates': 20000, 'seed': 2026090722,
    'intervals': 'paired whole-scenario cluster bootstrap Bonferroni95%',
    'p_values': 'cluster-homogeneous exact discordance; Holm across eight',
    'minimum_useful_success_gain_percentage_points': 5,
    'complete_seven_arm_panel_required': True, 'selection_from_partial_results': False}


def prepared(args):
    old = validate_protocol(args.reference_freeze)
    checked_source(args.reference_code, old['source_sha256'])
    legacy, _, stimulus = stimulus_contract(args.stimuli, TASK)
    coupling, _, coupling_hashes = coupling_binding(args.original, legacy, TASK)
    _, _, _, _, _, files = verify_bindings(args.fits/TASK, coupling, 'mw-reach', CHECKPOINTS['metaworld'])
    planning = prepare(args.vendor, TASK); planning['config']['meta']['seed'] = DEVELOPMENT_SEED
    previous = old['tasks'][TASK]
    if (planning != previous['planning'] or stimulus != previous['stimuli'] or
            coupling_hashes != previous['coupling'] or
            sha256(args.fits/TASK/'DONE.json') != previous['fit_done_sha256'] or
            sha256(args.fits/TASK/'operator_bank.pt') != previous['fit_bank_sha256']):
        raise ValueError('Original component/scenario/planner binding changed')
    return {'role': ROLE, 'method': METHOD, 'task': TASK, 'source_sha256': source_hash(),
        'reference_freeze_sha256': sha256(args.reference_freeze/'protocol.json'),
        'reference_source_sha256': old['source_sha256'], 'task_binding': previous,
        'fit_files': files, 'arms': list(ARMS), 'new_arms': list(NEW_ARMS),
        'reference_arms': list(REFERENCE_ARMS), 'episodes': schedule(),
        'episodes_per_condition_total': 96, 'new_scientific_episodes': 192,
        'reference_episodes': 480, 'primary_contrasts': [list(c) for c in CONTRASTS],
        'analysis': ANALYSIS, 'precision': 'float32_strict_no_tf32',
        'checkpoint_sha256': CHECKPOINTS['metaworld'], 'requested_tasks': 6, 'this_panel_tasks': 1,
        'reference_reuse': 'same physical GPU, canonical stimuli, logical RNG stream and verified raw traces',
        'component_doses_unchanged': True, 'drop_one_is_equal_total_energy': False,
        'fresh_confirmation': False, 'confirmation_access_authorized': False,
        'training_histories_complete': False, 'offline_significance_required': False}


def read_protocol(args):
    expected = prepared(args)
    path = args.freeze/'protocol.json'
    if (json.loads(path.read_text()) != expected or
            json.loads((args.freeze/'FROZEN.json').read_text())['protocol_sha256'] != sha256(path)):
        raise ValueError('Combined behavioral freeze/source/bindings changed')
    return expected


def verify_combined_cem(root, code, numerical_root, numerical_code, fit_files, expected_uuid):
    report, digest = verified_report(root)
    protocol = json.loads((root/'protocol.json').read_text())
    if ((root/'FAILED.json').exists() or report.get('status') != 'fixed_combined_full_cem_engineering_complete' or
            report.get('task') != TASK or report.get('method') != METHOD or
            report.get('protocol_sha256') != sha256(root/'protocol.json') or
            report.get('device_uuid') != expected_uuid or report.get('full_simulator_episodes') != 4 or
            report.get('scenario_count') != 1 or report.get('scientific_efficacy_measurement') is not False or
            report.get('fresh_confirmation') is not False or
            report.get('backend', {}).get('checkpoint_sha256') != CHECKPOINTS['metaworld'] or
            any(report.get(k) is not True for k in ('native_repeat_exact', 'paired_stimuli_exact',
                'full_cem_schedule_verified', 'parameter_buffer_bytes_modes_unchanged')) or
            protocol.get('method') != METHOD or protocol.get('task') != TASK or
            protocol.get('engineering_seed') != SMOKE_SEED or protocol.get('episode_order') != list(EPISODES) or
            protocol.get('precision') != 'float32_strict_no_tf32' or
            protocol.get('protected_outcomes_access_authorized') is not False):
        raise ValueError('Missing bound same-device full combined engineering')
    checked_source(code, protocol['source_sha256'])
    numerical = verify_numerical(numerical_root, protocol['numerical_check']['report_sha256'],
        numerical_code, fit_files, CHECKPOINTS['metaworld'], expected_uuid)
    if numerical != protocol['numerical_check']:
        raise ValueError('Full-CEM numerical provenance changed')
    if protocol['planning_contract'] != prepare(Path(protocol['planning_contract']['vendor']), TASK):
        raise ValueError('Full-CEM author planning contract changed')
    for name, key in (('model-before.json','model_before_sha256'), ('model-after.json','model_after_sha256')):
        if sha256(root/name) != report[key]: raise ValueError('Full-CEM model byte record changed')
    if (root/'model-before.json').read_bytes() != (root/'model-after.json').read_bytes():
        raise ValueError('Full-CEM model bytes/modes changed')
    rows = {}
    if set(report['episodes']) != set(EPISODES): raise ValueError('Incomplete engineering episode registry')
    for name in EPISODES:
        directory = root/name
        if sha256(directory/'DONE.json') != report['episodes'][name]:
            raise ValueError('Engineering episode receipt changed')
        value, _ = verified_report(directory)
        arm = 'native' if name == 'native_repeat' else name
        if (value['arm'] != arm or value['episode_name'] != name or
                value['engineering_only'] is not True or value['scientific_efficacy_measurement'] is not False):
            raise ValueError('Engineering labels changed')
        for filename, key in (('unroll_calls.json','unroll_calls_sha256'), ('action_trace.json','action_trace_sha256')):
            if sha256(directory/filename) != value[key]: raise ValueError('Engineering raw trace changed')
        calls = json.loads((directory/'unroll_calls.json').read_text())
        verify_episode(value['result'], calls)
        for call in calls: validate_record(call['record'], arm, call['horizon'], call['candidates'])
        rows[name] = {'result': value['result'], 'action_trace': json.loads((directory/'action_trace.json').read_text())}
        if name != 'native': verify_pair(rows['native'], rows[name], native_repeat=name == 'native_repeat')
    return report, digest


def reference_records(args, protocol, arm, ranks=None):
    if arm not in REFERENCE_ARMS: raise ValueError('Unknown unchanged reference arm')
    expected = assigned_rows(schedule(), list(range(8)) if ranks is None else ranks)
    wanted = {row['episode'] for row in expected}
    rows, bindings = [], {}
    for root in shard_paths(args.reference/TASK/arm):
        launch = json.loads((root/'protocol.json').read_text())
        assigned = {r['episode'] for r in launch['expected_episodes']}
        if not assigned & wanted: continue
        report, digest = verified_report(root)
        if ((root/'FAILED.json').exists() or report.get('status') != 'fixed_response_behavioral_shard_complete' or
                report['task'] != TASK or report['arm'] != arm or
                report['protocol_sha256'] != sha256(root/'protocol.json') or
                launch['freeze_sha256'] != protocol['reference_freeze_sha256'] or
                launch['source_sha256'] != protocol['reference_source_sha256'] or
                report['parameters_unchanged'] is not True or report['fresh_confirmation'] is not False or
                report['scientific_efficacy_measurement'] is not True or report['episodes'] != len(assigned)):
            raise ValueError('Invalid original completed reference shard')
        worker = args.reference_workers/('gpu-'+root.name.removeprefix('shard-gpu'))
        proof = verify_cem(worker, args.fits/TASK, args.reference_code, TASK)
        meta = json.loads((worker/'WORKER.json').read_text())
        if (proof != launch['engineering_report_sha256'] or
                sha256(worker/'WORKER.json') != launch['receiving_worker_sha256'] or
                meta['device_uuid'] != launch['device_uuid']):
            raise ValueError('Original receiving-worker proof changed')
        shard_rows = []
        for name, expected_hash in report['episode_files_sha256'].items():
            if Path(name).name != name or not name.startswith('episode-') or sha256(root/name) != expected_hash:
                raise ValueError('Unsafe/changed reference episode')
            row = json.loads((root/name).read_text())
            calls_root = root/f"calls-{row['episode']:03d}"
            for filename, key in (('unroll_calls.json','unroll_calls_sha256'), ('action_trace.json','action_trace_sha256')):
                if sha256(calls_root/filename) != row[key]: raise ValueError('Reference action/call trace changed')
            calls = json.loads((calls_root/'unroll_calls.json').read_text()); verify_episode(row['result'], calls)
            if row['arm'] != arm or any(c['backend_calls'] != 1 for c in calls):
                raise ValueError('Reference treatment or model work changed')
            shard_rows.append(row)
            if row['episode'] in wanted:
                rows.append({**row, 'device_uuid': launch['device_uuid'], 'reference_record': str(root/name),
                    'reference_record_sha256': expected_hash, 'runtime_retained_from_original': True})
        validate_coverage(sorted(shard_rows, key=lambda r:r['episode']), launch['expected_episodes'])
        bindings[str(root/'report.json')] = digest
    rows.sort(key=lambda row: row['episode']); validate_coverage(rows, expected)
    return rows, bindings


@torch.no_grad()
def run(args):
    protocol = read_protocol(args)
    if args.arm not in NEW_ARMS: raise ValueError('Only the two new combined arms run here')
    expected = assigned_rows(schedule(), args.logical_ranks)
    native, reference_files = reference_records(args, protocol, 'native', args.logical_ranks)
    uuid = device_uuid()
    if any(row['device_uuid'] != uuid for row in native):
        raise ValueError('Cross-device reference reuse is not authorized')
    legacy, goals, _ = stimulus_contract(args.stimuli, TASK)
    coupling_root, _, _ = coupling_binding(args.original, legacy, TASK)
    bank, _, coupling_protocol, coupling_bank, _, files = verify_bindings(
        args.fits/TASK, coupling_root, 'mw-reach', CHECKPOINTS['metaworld'])
    _, engineering_hash = verify_combined_cem(args.engineering, args.engineering_code,
        args.numerical_check, args.numerical_code, files, uuid)
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    args.output.mkdir(parents=True, exist_ok=False)
    started, records = time.monotonic(), []
    try:
        exclusive_json(args.output/'protocol.json', {'role': ROLE, 'method': METHOD, 'task': TASK,
            'arm': args.arm, 'freeze_sha256': sha256(args.freeze/'protocol.json'),
            'source_sha256': source_hash(), 'device_uuid': uuid, 'expected_episodes': expected,
            'logical_ranks': args.logical_ranks, 'engineering_report_sha256': engineering_hash,
            'reference_reports_sha256': reference_files, 'fresh_confirmation': False})
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS['metaworld'], 'metaworld', 'cuda:0', 'float32')
        if backend.model.ctxt_window != 2: raise ValueError('Wrong planning context')
        initial_model, counted = model_signature(backend.model), CountedBackend(backend)
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        native = {row['episode']: row for row in native}
        torch.cuda.reset_peak_memory_stats()
        for rank in sorted(args.logical_ranks):
            rows = [r for r in expected if r['logical_rank'] == rank]
            random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
            cfg = OmegaConf.create(protocol['task_binding']['planning']['config'])
            cfg.local_seed = rows[0]['local_seed']
            agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
            env = make_env(cfg)
            try:
                for row in rows:
                    before = time.monotonic(); directory = args.output/f"calls-{row['episode']:03d}"
                    directory.mkdir(exist_ok=False)
                    adapter = CombinedFixedResponseIntervention(counted, bank, coupling_protocol, coupling_bank, args.arm)
                    observed = ObservedCombined(counted, args.arm, directory, adapter)
                    agent.planner.unroll = observed
                    original_act = agent.act; trace = trace_actor(agent)
                    try:
                        with close_expert_environments(plan_evaluator), goals.deliver(row):
                            result = run_episode(cfg, backend, agent, env, row['environment_seed'])
                    finally:
                        agent.act = original_act
                        exclusive_json(directory/'unroll_calls.json', observed.calls)
                        exclusive_json(directory/'action_trace.json', trace)
                    verify_episode(result, observed.calls)
                    metadata, _ = goals.load(row)
                    initial = np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist()
                    if initial != metadata['rand_vec'] or initial != native[row['episode']]['initial_state_vector']:
                        raise ValueError('Canonical initial state changed')
                    for key in ('initial_sha256','goal_sha256'):
                        if result[key] != metadata[key] or result[key] != native[row['episode']]['result'][key]:
                            raise ValueError('New combined/reference stimuli are unpaired')
                    record = {**row, 'arm': args.arm, 'result': result, 'initial_state_vector': initial,
                        'seconds': time.monotonic()-before, 'device_uuid': uuid,
                        'unroll_calls_sha256': sha256(directory/'unroll_calls.json'),
                        'action_trace_sha256': sha256(directory/'action_trace.json')}
                    validate_coverage([record], [row]); records.append(record)
                    exclusive_json(args.output/f"episode-{row['episode']:03d}.json", record)
                    progress = {'task': TASK, 'arm': args.arm, 'completed': len(records),
                        'target': len(expected), 'seconds': time.monotonic()-started, 'fresh_confirmation': False}
                    write_json(args.output/'progress.json', progress); print(json.dumps(progress), flush=True)
            finally: env.close()
        validate_coverage(records, expected); verify_files(files)
        if model_signature(backend.model) != initial_model or source_hash() != protocol['source_sha256']:
            raise ValueError('Frozen model/source changed')
        exclusive_json(args.output/'report.json', {'status': 'combined_fixed_response_behavioral_shard_complete',
            'task': TASK, 'arm': args.arm, 'method': METHOD, 'episodes': len(records), 'device_uuid': uuid,
            'protocol_sha256': sha256(args.output/'protocol.json'), 'parameter_buffer_bytes_modes_unchanged': True,
            'episode_files_sha256': {f"episode-{r['episode']:03d}.json": sha256(args.output/f"episode-{r['episode']:03d}.json") for r in records},
            'seconds': time.monotonic()-started, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
            'scientific_efficacy_measurement': True, 'fresh_confirmation': False, 'comparative_analysis_complete': False})
        exclusive_json(args.output/'DONE.json', {'report_sha256': sha256(args.output/'report.json')})
    except BaseException as error:
        exclusive_json(args.output/'FAILED.json', {'error': repr(error), 'partial_not_complete': True})
        raise


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze','run'))
    for name in ('vendor','fits','original','stimuli','reference','reference-freeze','reference-code','reference-workers','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('freeze','checkpoint','engineering','engineering-code','numerical-check','numerical-code'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--arm', choices=NEW_ARMS)
    parser.add_argument('--logical-ranks', type=int, nargs='+')
    return parser.parse_args()


def main():
    args = arguments()
    if args.mode == 'freeze':
        protocol = prepared(args)
        args.output.mkdir(parents=True, exist_ok=False)
        exclusive_json(args.output/'protocol.json', protocol)
        exclusive_json(args.output/'FROZEN.json', {'protocol_sha256': sha256(args.output/'protocol.json'),
            'new_combined_behavioral_outcomes_observed_before_freeze': False,
            'existing_population_previously_used_for_development': True})
    else: run(args)


if __name__ == '__main__': main()
