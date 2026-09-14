"""Native full-planner paired evaluation of the fixed map on six-block tasks.

Separate from the frozen MetaWorld package. No old online solver, changed
planner budget, offline admission gate, partial RNG resume or confirmation.
"""
from offline_study._paths import source_path
import argparse
from contextlib import nullcontext
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import open_normalized_dataset, validate_cohort
from offline_study.models.backends import JepaBackend
from offline_study.evaluation.behavioral_development import DEVELOPMENT_SEED, assigned_rows, schedule, verified_report
from offline_study.interventions.fixed_response import FixedResponseIntervention, load_fitted_bank
from offline_study.evaluation.fixed_response_behavior import device_uuid
from offline_study.validation.fixed_response_check import CountedBackend, reference_fields
from offline_study.interventions.interventions import PredictorIntervention
from offline_study.data.inventory import _initial_state_group
from offline_study.tasks.navigation.navigation_coupling_behavior import action_hashes, source_digest
from offline_study.tasks.navigation.navigation_replication import validate_records
from offline_study.tasks.navigation.navigation_smoke import comparison_record
from offline_study.fitting.operator_fit import _model_versions
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_env_smoke import SingleFitTrajectory
from offline_study.planning.planning_native_smoke import SMOKE_SEED, run_episode
from offline_study.tasks.metaworld.refined_task_fit import CHECKPOINTS
from offline_study.core.protocol import sha256, write_json
from offline_study.tasks.pusht.pusht_coupling_behavior import paired_inputs as paired_push, validate_push_records
from offline_study.tasks.pusht.pusht_planning_replication import TracedDataset, checked_cohort, input_hashes, trace_segment
from offline_study.interventions.support_operator import NativeFieldCapture
from offline_study.models.vendor import use_vendor

TASKS = ('pusht', 'pointmaze', 'wall')
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4')
ENGINEERING = ('native', 'native_repeat', 'zero_dose', *ARMS[1:])
METHOD = 'fixed_response_six_task_completion_20260911_v1'
ANALYSIS = {'replicates': 20000, 'seed': 2026091102, 'family': 12,
    'interval': 'paired_cluster_percentile_bootstrap_bonferroni_95',
    'family_tasks': ['pusht', 'pointmaze', 'wall', 'droid'],
    'contrasts': ['fixed_rank4-native', 'matched_random_fixed_rank4-native',
        'fixed_rank4-matched_random_fixed_rank4'],
    'automatic_selection': False, 'confirmation': False}


def read(path):
    return json.loads(Path(path).read_text())


def task_inputs(args):
    if args.task not in TASKS:
        raise ValueError('DROID needs its own architecture and action-score adapter')
    bank = load_fitted_bank(args.fit, task=args.task, checkpoint_sha256=CHECKPOINTS[args.task])
    cohort_path = args.fit / 'cohort.json'
    cohort = read(cohort_path)
    validate_cohort(cohort)
    if args.task == 'pusht':
        # Preserve the original separate access-authorized cohort and its sidecars.
        cohort = checked_cohort(args.cohort)
        if sha256(args.cohort) != sha256(cohort_path):
            raise ValueError('Push-T access cohort differs from the fitted cohort')
    native_path = args.reference / 'freeze/protocol.json'
    native = read(native_path)
    if read(args.reference / 'freeze/FROZEN.json')['protocol_sha256'] != sha256(native_path):
        raise ValueError('Historical native freeze changed')
    if (native.get('episodes') != schedule() or native.get('fresh_confirmation') is not False
            or source_digest(args.reference_source) != native.get('source_sha256')):
        raise ValueError('Unverified original scenario schedule or native source')
    for name in ('backends.py', 'model_loader.py', 'planning_native_smoke.py', 'planning_contract.py'):
        if sha256(args.reference_source / name) != sha256(source_path(name)):
            raise ValueError('Proven native execution changed: ' + name)
    use_vendor(args.vendor)
    planning = prepare(args.vendor, args.task)
    planning['config']['meta']['seed'] = DEVELOPMENT_SEED
    if planning != native.get('planning_contract'):
        raise ValueError('Original task/planner settings changed')
    data_binding = None
    if args.task == 'pusht':
        data_binding = input_hashes(args.data_root / 'val')
        if data_binding != native.get('input_files_sha256'):
            raise ValueError('Original released Push-T behavioral bytes changed')
    return bank, cohort, {'planning': planning, 'fit_done_sha256': sha256(args.fit / 'DONE.json'),
        'dose': bank['dose'],
        'fit_bank_sha256': sha256(args.fit / 'operator_bank.pt'),
        'cohort_sha256': sha256(cohort_path), 'checkpoint_sha256': CHECKPOINTS[args.task],
        'native_freeze_sha256': sha256(native_path), 'behavioral_inputs': data_binding}


def make_protocol(task, binding):
    if task not in TASKS:
        raise ValueError('Unknown six-block task')
    return {'method': METHOD, 'task': task, 'source_sha256': source_hash(),
        'binding': binding, 'arms': list(ARMS), 'engineering': list(ENGINEERING),
        'episodes': schedule(), 'episodes_per_condition': 96, 'evaluations': 288,
        'analysis': ANALYSIS, 'precision': 'float32_strict_no_tf32',
        'fresh_confirmation': False, 'protected_access': False,
        'historical_native_reuse': False, 'same_physical_gpu_for_paired_scenario': True,
        'legacy_solver': False, 'offline_significance_gate': False}


def validate_protocol(root):
    p = read(root / 'protocol.json')
    if sha256(root / 'protocol.json') != read(root / 'FROZEN.json')['protocol_sha256']:
        raise ValueError('Behavioral protocol changed')
    if p != make_protocol(p['task'], p['binding']):
        raise ValueError('Task/source/sample/arm/statistical contract changed')
    return p


def freeze(args):
    _, _, binding = task_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', make_protocol(args.task, binding))
    write_json(args.output / 'FROZEN.json', {'protocol_sha256': sha256(args.output / 'protocol.json'),
        'new_candidate_outcomes_observed': False, 'historical_results_already_observed': True})


def paired(reference, candidate, task, repeat=False):
    for key in ('episode', 'logical_rank', 'local_seed', 'environment_seed'):
        if reference[key] != candidate[key]:
            raise ValueError('Changed scenario identity: ' + key)
    for key in ('initial_sha256', 'goal_sha256'):
        if reference['result'][key] != candidate['result'][key]:
            raise ValueError('Changed paired simulator input: ' + key)
    if task == 'pusht' and 'source_segment' in reference:
        paired_push(reference, candidate)
    if repeat and comparison_record(reference['result'], action_hashes(reference['planned_actions'])) != comparison_record(
            candidate['result'], action_hashes(candidate['planned_actions'])):
        raise ValueError('Native-repeat/zero-dose action and outcome identity failed')


def validate_calls(calls, arm, dose):
    if [(c['horizon'], c['candidates']) for c in calls] != [(6, 300), (6, 1)] * 30:
        raise ValueError('Full native CEM call budget changed')
    for call in calls:
        r = call['record']
        if r['backend_calls'] != 1 or r['response_probe_rollouts'] or r['native_shadow_rollouts']:
            raise ValueError('Extra deployed model work or legacy solver')
        if arm in ARMS[1:]:
            n = call['candidates']
            requested = np.asarray(r['requested_l2'])
            realized = np.asarray(r['realized_l2'])
            active = np.asarray(r['active'], dtype=bool)
            coeff = np.asarray(r['coefficients'])
            if (requested.shape != (n,) or realized.shape != (n,) or active.shape != (n,)
                    or coeff.shape != (n, 4) or not np.isfinite(coeff).all()
                    or not np.isfinite(requested).all() or not np.isfinite(realized).all()
                    or not np.allclose(requested, active * dose, rtol=1e-5, atol=1e-8)
                    or not np.allclose(realized, requested, rtol=5e-4, atol=1e-8)):
                raise ValueError('Refined/control delivered dose or coefficient contract changed')
        elif any(k in r for k in ('coefficients', 'requested_l2', 'realized_l2')):
            raise ValueError('An unedited reference received an intervention')


@torch.no_grad()
def one_episode(cfg, backend, agent, env, row, arm, bank, output, engineering=False):
    started = time.monotonic()
    counted = CountedBackend(backend)
    adapter = FixedResponseIntervention(counted, bank, arm)
    calls, actions, checked = [], [], set()
    original_act, original_unroll = agent.act, agent.planner.unroll
    def observed(context, act_suffix=None, **kwargs):
        horizon, count = act_suffix.shape[:2]
        before = counted.calls
        prediction = adapter(context, act_suffix, **kwargs)
        if counted.calls != before + 1:
            raise ValueError('Extra backend calls')
        if engineering and count not in checked:
            if arm in ('native', 'zero_dose'):
                expected = backend.predict(context, act_suffix)
            else:
                with NativeFieldCapture(backend.predictor) as capture:
                    backend.predict(context, act_suffix)
                fields = reference_fields(capture.values[3], adapter.bank, arm)
                with PredictorIntervention(backend.predictor, fields):
                    expected = backend.predict(context, act_suffix)
            if any(not torch.equal(prediction[k], expected[k]) for k in ('visual', 'proprio')):
                raise ValueError('Independent static-map reference parity failed')
            checked.add(count)
        record = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
            for k, v in adapter.last_record.items()}
        calls.append({'horizon': horizon, 'candidates': count, 'record': record})
        write_json(output / 'heartbeat.json', {'completed_calls': len(calls),
            'episode': row['episode'], 'arm': arm, 'engineering': engineering})
        return prediction
    def actor(*args, **kwargs):
        result = original_act(*args, **kwargs)
        actions.append(result.detach().cpu().tolist())
        return result
    agent.planner.unroll, agent.act = observed, actor
    try:
        result = run_episode(cfg, backend, agent, env, row['environment_seed'])
    finally:
        agent.planner.unroll, agent.act = original_unroll, original_act
        write_json(output / 'unroll_calls.json', calls)
    record = {**row, 'arm': arm, 'result': result, 'planned_actions': actions,
        'unroll_calls': [(c['horizon'], c['candidates']) for c in calls],
        'call_trace_sha256': sha256(output / 'unroll_calls.json'),
        'static_reference_counts': sorted(checked), 'seconds': time.monotonic() - started}
    validate_records([record], [row])
    validate_calls(calls, arm, bank['dose'])
    if engineering and checked != {1, 300}:
        raise ValueError('Incomplete independent-reference candidate coverage')
    return record


def verify_engineering(root, protocol, freeze_hash, worker):
    report, digest = verified_report(root)
    if ((root / 'FAILED.json').exists() or report.get('status') != METHOD + '_engineering_complete'
            or report.get('source_sha256') != protocol['source_sha256']
            or report.get('freeze_sha256') != freeze_hash or report.get('device_uuid') != worker
            or report.get('task') != protocol['task'] or report.get('parameters_unchanged') is not True
            or report.get('dose') != protocol['binding']['dose']
            or report.get('scientific_efficacy_measurement') is not False
            or set(report['episodes']) != set(ENGINEERING)):
        raise ValueError('Missing exact task/source/receiving-device validation')
    native = None
    for name in ENGINEERING:
        path = root / name
        record = read(path / 'episode.json')
        if (sha256(path / 'episode.json') != report['episodes'][name]
                or sha256(path / 'unroll_calls.json') != record['call_trace_sha256']
                or record['static_reference_counts'] != [1, 300]
                or record['arm'] != ('native' if name == 'native_repeat' else name)):
            raise ValueError('Receiving validation record changed')
        expected = dict(episode=0, logical_rank=0, local_seed=SMOKE_SEED, environment_seed=SMOKE_SEED)
        validate_records([record], [expected])
        validate_calls(read(path / 'unroll_calls.json'), record['arm'], report['dose'])
        if native is None:
            native = record
        paired(native, record, protocol['task'], name in ('native_repeat', 'zero_dose'))
    return digest


def load_native_stream(root, protocol, freeze_hash, worker, expected):
    """Reject bad references BEFORE spending GPU time on an edited stream."""
    report, digest = verified_report(root)
    launch = read(root / 'protocol.json')
    names = [f"episode-{row['episode']:03d}" for row in expected]
    if ((root / 'FAILED.json').exists() or report.get('status') != METHOD + '_shard_complete'
            or report.get('device_uuid') != worker or report.get('freeze_sha256') != freeze_hash
            or report.get('source_sha256') != protocol['source_sha256']
            or report.get('task') != protocol['task'] or report.get('arm') != 'native'
            or report.get('dose') != protocol['binding']['dose']
            or report.get('parameters_unchanged') is not True or report.get('fresh_confirmation') is not False
            or report.get('scientific_efficacy_measurement') is not True
            or report.get('protocol_sha256') != sha256(root / 'protocol.json')
            or launch.get('expected_episodes') != expected or launch.get('engineering') is not False
            or launch.get('device_uuid') != worker or launch.get('freeze_sha256') != freeze_hash
            or set(report['episodes']) != set(names)):
        raise ValueError('No complete exact same-device native reference stream')
    rows = {}
    for key in names:
        path = root / key
        record = read(path / 'episode.json')
        if (report['episodes'][key] != sha256(path / 'episode.json') or record['arm'] != 'native'
                or record['call_trace_sha256'] != sha256(path / 'unroll_calls.json')):
            raise ValueError('Native raw record changed')
        validate_calls(read(path / 'unroll_calls.json'), 'native', report['dose'])
        rows[key] = record
    validate_records(list(rows.values()), expected)
    return rows, digest


def dataset_for(args, backend, cohort, engineering):
    if args.task != 'pusht':
        return None
    if engineering:
        full = open_normalized_dataset('pusht', args.data_root, cohort['reference_config'], True)
        row = cohort['fit'][0]
        if (_initial_state_group(full.states[row['index'], 0]) != row['lineage_group']
                or full.get_seq_length(row['index']) != row['length']):
            raise ValueError('Excluded fitting stimulus changed')
        return SingleFitTrajectory(full, row['index'])
    from app.plan_common.datasets.pusht_dset import PushTDataset
    full = PushTDataset(data_path=str(args.data_root / 'val'), transform=backend.preprocessor.transform,
        normalize_action=True, with_velocity=True)
    rows = {row['index']: row for row in cohort['evaluation']}
    if (set(rows) != set(range(21)) or len(full) != 21 or any(
            _initial_state_group(full.states[i, 0]) != row['lineage_group']
            or full.get_seq_length(i) != row['length'] for i, row in rows.items())):
        raise ValueError('Actual released Push-T population changed')
    return TracedDataset(full, rows)


@torch.no_grad()
def execute(args):
    p = validate_protocol(args.freeze)
    bank, cohort, binding = task_inputs(args)
    if p['task'] != args.task or p['binding'] != binding:
        raise ValueError('Frozen task inputs changed')
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from omegaconf import OmegaConf
    engineering = args.command == 'engineer'
    worker = device_uuid()
    freeze_hash = sha256(args.freeze / 'protocol.json')
    proof = None if engineering else verify_engineering(args.engineering, p, freeze_hash, worker)
    expected = None if engineering else assigned_rows(schedule(), args.logical_ranks)
    if not engineering and args.arm not in ARMS:
        raise ValueError('Unknown behavioral condition')
    native_rows, native_hash = {}, None
    if not engineering and args.arm != 'native':
        native_rows, native_hash = load_native_stream(args.native, p, freeze_hash, worker, expected)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / 'protocol.json', {'freeze_sha256': freeze_hash,
            'source_sha256': source_hash(), 'task': args.task, 'engineering': engineering,
            'arm': None if engineering else args.arm, 'expected_episodes': expected,
            'logical_ranks': None if engineering else args.logical_ranks,
            'device_uuid': worker, 'engineering_report_sha256': proof, 'fresh_confirmation': False})
        if native_hash:
            write_json(args.output / 'NATIVE_REFERENCE.json', {'report_sha256': native_hash})
        seed = 0 if args.task == 'pusht' else DEVELOPMENT_SEED
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS[args.task], args.task, 'cuda:0', 'float32')
        if backend.model.ctxt_window != 2 or backend.provenance['normalization_dataset'] != args.task:
            raise ValueError('Wrong native context or normalization')
        versions = _model_versions(backend.model)
        dataset = dataset_for(args, backend, cohort, engineering)
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        episodes = {}
        native_engineering = None
        for name in (ENGINEERING if engineering else [args.arm]):
            arm = 'native' if name == 'native_repeat' else name
            for rank in ([None] if engineering else sorted(args.logical_ranks)):
                if engineering:
                    random.seed(SMOKE_SEED); np.random.seed(SMOKE_SEED); torch.manual_seed(SMOKE_SEED)
                    rows = [dict(episode=0, logical_rank=0, local_seed=SMOKE_SEED, environment_seed=SMOKE_SEED)]
                else:
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                    rows = [row for row in expected if row['logical_rank'] == rank]
                cfg = OmegaConf.create(p['binding']['planning']['config'])
                cfg.local_seed = rows[0]['local_seed']
                if engineering:
                    cfg.meta.seed = SMOKE_SEED
                agent = GC_Agent(cfg, backend.model, dset=dataset, preprocessor=backend.preprocessor)
                env = make_env(cfg)
                try:
                    for row in rows:
                        key = name if engineering else f"episode-{row['episode']:03d}"
                        output = args.output / key
                        output.mkdir(exist_ok=False)
                        with (trace_segment(dataset) if args.task == 'pusht' and not engineering else nullcontext()) as segments:
                            record = one_episode(cfg, backend, agent, env, row, arm, bank, output, engineering)
                        if args.task == 'pusht' and not engineering:
                            if len(segments) != 1:
                                raise ValueError('Expected one original source segment')
                            record['source_segment'] = segments[0]
                            validate_push_records([record], [row], cohort)
                        if engineering:
                            if native_engineering is None:
                                native_engineering = record
                            paired(native_engineering, record, args.task, name in ('native_repeat', 'zero_dose'))
                        elif arm != 'native':
                            paired(native_rows[key], record, args.task)
                        write_json(output / 'episode.json', record)
                        episodes[key] = sha256(output / 'episode.json')
                        progress = {'task': args.task, 'arm': arm, 'completed': len(episodes),
                            'engineering': engineering, 'seconds': time.monotonic() - started}
                        write_json(args.output / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
                finally:
                    env.close()
        if _model_versions(backend.model) != versions or source_hash() != p['source_sha256']:
            raise ValueError('Frozen parameters/source changed')
        if not engineering:
            validate_records([read(args.output / key / 'episode.json') for key in episodes], expected)
        write_json(args.output / 'report.json', {'status': METHOD + ('_engineering_complete' if engineering else '_shard_complete'),
            'task': args.task, 'arm': None if engineering else args.arm, 'dose': bank['dose'],
            'freeze_sha256': freeze_hash, 'source_sha256': source_hash(), 'device_uuid': worker,
            'protocol_sha256': sha256(args.output / 'protocol.json'), 'episodes': episodes,
            'parameters_unchanged': True, 'fresh_confirmation': False,
            'scientific_efficacy_measurement': not engineering, 'seconds': time.monotonic() - started})
        write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})
        if engineering:
            verify_engineering(args.output, p, freeze_hash, worker)
    except Exception as exc:
        write_json(args.output / 'FAILED.json', {'error': str(exc), 'partial_not_eligible': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'engineer', 'run'))
    parser.add_argument('--task', choices=TASKS, required=True)
    for name in ('vendor', 'checkpoint', 'fit', 'cohort', 'reference', 'reference-source', 'data-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('freeze', 'engineering', 'native'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--logical-ranks', type=int, nargs='+')
    args = parser.parse_args()
    if args.command != 'freeze' and args.freeze is None:
        parser.error('--freeze required for execution')
    if args.command == 'run' and (args.engineering is None or args.logical_ranks is None or args.arm is None
            or (args.arm != 'native' and args.native is None)):
        parser.error('Run requires engineering, whole logical ranks, arm and paired native when edited')
    freeze(args) if args.command == 'freeze' else execute(args)
