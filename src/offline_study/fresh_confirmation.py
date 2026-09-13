"""Prospective four-task confirmation on the September 12 input banks.

This is a separate protocol, not permission to open an old protected cohort.
One job owns a whole scenario and all eight arms on one physical GPU. Each
scenario starts its own explicitly frozen planner RNG stream; no historical
outcomes or partially consumed streams are spliced into the new panel.
"""
import argparse
import contextlib
import copy
import fcntl
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .author_evaluate import verify_fit
from .author_fit import source_hash
from .backends import JepaBackend
from .fixed_response import FixedResponseIntervention, load_fitted_bank
from .fixed_response_behavior import ObservePlanner, device_uuid
from .fixed_response_check import reference_fields
from .fixed_response_smoke import trace_actor, verify_episode, verify_pair
from .intervention_runner import _model_versions
from .interventions import PredictorIntervention
from .metaworld_component_behavior import verify_energy
from .navigation_smoke import CHECKPOINTS as NAV_CHECKPOINTS, validate_complete
from .planning_contract import prepare
from .planning_env_smoke import observation_digest
from .planning_goal_bank import restore_goal
from .planning_native_smoke import CHECKPOINTS as MW_CHECKPOINTS, run_episode
from .planning_panel_engineering import H6StaticPlanningIntervention
from .planning_scenarios import close_expert_environments
from .planning_support_check import source_coupling_fields
from .protocol import sha256, write_json
from .support_operator import NativeFieldCapture
from .vendor import use_vendor

TASKS = ('reach', 'reach-wall', 'pointmaze', 'wall')
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only',
        'matched_random_coupling', 'joint', 'visual_only', 'action_condition_only')
COUPLING = {'coupling_only': 'joint_equal_standardized_energy',
            'matched_random_coupling': 'matched_random_equal_standardized_energy',
            **{a: a for a in ARMS[5:]}}
ENGINEERING = ('native', 'native_repeat')
ENGINEERING_POLICY = {'full_episodes': list(ENGINEERING),
    'numerical_arms': [*ARMS, 'zero_dose'],
    'coverage': 'every distinct visual/proprio context shape, horizon and candidate count observed during full native episode',
    'reference': 'bitwise independent source-hook/native forecasts; energy, input, parameter and hook identity',
    'scientific_efficacy_measurement': False}
CHECKPOINTS = {**NAV_CHECKPOINTS, 'reach': MW_CHECKPOINTS['metaworld'],
               'reach-wall': MW_CHECKPOINTS['metaworld']}
METHOD = 'fresh_four_task_eight_arm_confirmation_20260912_v2'
BANK = 'artifacts/offline_study/fresh-simulator-banks-20260912-v1'
AUDIT = 'artifacts/offline_study/four-task-source-exposure-20260912-v1'
CONTRASTS = {f'{a}-native': {a: 1, 'native': -1} for a in ARMS[1:]}
CONTRASTS.update({
    'refined-random': {'fixed_rank4': 1, 'matched_random_fixed_rank4': -1},
    'coupling-random': {'coupling_only': 1, 'matched_random_coupling': -1},
    'joint-visual': {'joint': 1, 'visual_only': -1},
    'joint-action': {'joint': 1, 'action_condition_only': -1},
    'factorial-interaction': {'joint': 1, 'visual_only': -1,
                              'action_condition_only': -1, 'native': 1}})
ANALYSIS = {'endpoint': 'official_binary_success', 'independent_unit': 'scenario',
    'bootstrap_draws': 20000, 'bootstrap_seed': 2026091221, 'family_size': 48,
    'interval': 'paired_scenario_percentile_bootstrap_bonferroni_95',
    'minimum_useful_gain_pp': 5, 'complete_panel_required': True,
    'automatic_selection': False, 'historical_results_pooled': False}


def read(path):
    return json.loads(Path(path).read_text())


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes project root')
    return path


def audit_binding(project):
    root = project / AUDIT
    original, cached, historical = (read(root / n) for n in
        ('report.json', 'CACHED_RECHECK.json', 'HISTORICAL_JSON_RECHECK.json'))
    if original['failures'] or historical['matches'] or historical['errors']:
        raise ValueError('Exposure reconciliation has failures or matches')
    if {r['check'] for r in cached['results']} != {'metaworld', 'pointmaze', 'wall'}:
        raise ValueError('Incomplete source-family exposure coverage')
    for result in cached['results']:
        if not result['pass'] or not result['known_positive_detected']:
            raise ValueError('Unverified source comparison')
        for row in result.get('tasks', [result]):
            if row['matches'] or row['candidate_scenarios'] != 104:
                raise ValueError('Exposure mismatch or incomplete candidate bank')
    if read(root / 'protocol.json')['input_bank_sha256'] != sha256(project / BANK / 'protocol.json'):
        raise ValueError('Exposure audit is for another input bank')
    return {str((root / name).relative_to(project)): sha256(root / name) for name in
        ('protocol.json', 'report.json', 'CACHED_RECHECK.json', 'HISTORICAL_JSON_RECHECK.json')}


def bank_records(project, task):
    root = project / BANK
    protocol = read(root / 'protocol.json')
    if sha256(root / 'protocol.json') != read(root / 'FROZEN.json')['protocol_sha256']:
        raise ValueError('Input bank freeze changed')
    records, files = [], {}
    for rank in range(8):
        folder = root / task / f'rank-{rank:02d}'
        report = read(folder / 'report.json')
        if ((folder / 'FAILED.json').exists() or report['protocol_sha256'] != sha256(root / 'protocol.json')
                or report['records_sha256'] != sha256(folder / 'records.json')
                or report['task'] != task or report['rank'] != rank
                or report['scientific_candidates'] != 12 or report['excluded_engineering'] != 1):
            raise ValueError('Unverified input shard')
        for row in read(folder / 'records.json'):
            path = checked_path(folder, row['tensor_file'])
            if sha256(path) != row['tensor_sha256']:
                raise ValueError('Input tensor changed')
            tensors = torch.load(path, map_location='cpu', weights_only=True)
            if any(observation_digest(tensors[k]) != row[k + '_sha256'] for k in ('initial', 'goal')):
                raise ValueError('Input tensor/metadata mismatch')
            records.append({**row, 'tensor_path': str(path.relative_to(project))})
            files[str(path.relative_to(project))] = row['tensor_sha256']
        for name in ('records.json', 'report.json'):
            files[str((folder / name).relative_to(project))] = sha256(folder / name)
    for role, n in (('scientific_candidates', 96), ('excluded_engineering', 8)):
        rows = [r for r in records if r['role'] == role]
        keys = ('episode', 'logical_rank', 'local_seed', 'environment_seed')
        actual = sorted([{k: r[k] for k in keys} for r in rows], key=lambda r: r['episode'])
        if len(rows) != n or actual != protocol['tasks'][task][role]:
            raise ValueError('Missing, duplicate or changed prospective scenario')
    if len({r['initial_sha256'] for r in records}) != 104:
        raise ValueError('Repeated scientific/engineering initial inputs')
    return records, files


def load_assets(project, task, spec):
    refined = checked_path(project, spec['refined'])
    coupling = checked_path(project, spec['coupling'])
    fitted = load_fitted_bank(refined, task='mw-' + task if task.startswith('reach') else task,
                              checkpoint_sha256=CHECKPOINTS[task])
    receipt = read(coupling / 'fit_receipt.json')
    _, protocol = verify_fit(coupling, receipt['cohort_sha256'], CHECKPOINTS[task], 'bfloat16')
    if receipt['development_outcomes_accessed'] is not False:
        raise ValueError('Coupling fitting used evaluation outcomes')
    bank = torch.load(coupling / 'operator_bank.pt', map_location='cpu', weights_only=True)
    if bank['protocol_sha256'] != sha256(coupling / 'protocol.json'):
        raise ValueError('Coupling bank/protocol mismatch')
    if not set(COUPLING.values()).issubset({r['name'] for r in protocol['arms']}):
        raise ValueError('Required frozen coupling arms absent')
    return fitted, protocol, bank


def freeze(args):
    assets = read(args.assets)
    if set(assets) != set(TASKS):
        raise ValueError('All four task assets must be bound before outcomes')
    files = audit_binding(args.project)
    tasks = {}
    for task in TASKS:
        load_assets(args.project, task, assets[task])
        records, inputs = bank_records(args.project, task)
        files.update(inputs)
        for key in ('refined', 'coupling'):
            directory = checked_path(args.project, assets[task][key])
            for path in directory.iterdir():
                if path.is_file():
                    files[str(path.relative_to(args.project))] = sha256(path)
            if key == 'coupling':
                path = directory.parent / 'PARITY.json'
                files[str(path.relative_to(args.project))] = sha256(path)
        tasks[task] = {'assets': assets[task], 'records': records,
                       'planning': prepare(args.project / 'vendor/jepa-wms', task),
                       'checkpoint_sha256': CHECKPOINTS[task]}
    for name in ('protocol.json', 'FROZEN.json'):
        files[BANK + '/' + name] = sha256(args.project / BANK / name)
    # Bind actual vendor code/config bytes, not only its nominal git revision.
    for path in (args.project / 'vendor/jepa-wms').rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
            files[str(path.relative_to(args.project))] = sha256(path)
    protocol = {'method': METHOD, 'tasks': tasks, 'arms': list(ARMS),
        'engineering': list(ENGINEERING), 'engineering_policy': ENGINEERING_POLICY, 'source_sha256': source_hash(),
        'files_sha256': files, 'analysis': ANALYSIS, 'contrasts': CONTRASTS,
        'fresh_confirmation': True, 'same_physical_gpu_per_scenario': True,
        'precision': 'float32_strict_no_tf32', 'scientific_episodes': 3072,
        'rng': 'global Python/NumPy/Torch seed 0; per-scenario GC_Agent local_seed=environment_seed, reset for every arm',
        'exposure_limit': 'checked released source identities and retained histories; not base-model pretraining or unlogged external work',
        'native_goal_delivery': 'unchanged source generation; exact delivered bank hashes; MetaWorld initial and goal images rehydrated only with exact physical/proprio checks and existing one-level/64-value render bound; PointMaze renderer initialized using original prepare_for_render before inherited reset',
        'authority': 'User September 12: execute four available tasks, all eight arms, on audited unseen inputs; old protected cohorts unchanged'}
    args.freeze.mkdir(parents=True, exist_ok=False)
    write_json(args.freeze / 'protocol.json', protocol)
    write_json(args.freeze / 'FROZEN.json', {'protocol_sha256': sha256(args.freeze / 'protocol.json'),
        'scientific_outcomes_observed': False, 'receiving_engineering_pending': True})


def validate_freeze(root, project, check_files=True):
    p = read(root / 'protocol.json')
    if sha256(root / 'protocol.json') != read(root / 'FROZEN.json')['protocol_sha256']:
        raise ValueError('Scientific freeze changed')
    if (p['method'] != METHOD or p['arms'] != list(ARMS) or p['engineering'] != list(ENGINEERING)
            or p['engineering_policy'] != ENGINEERING_POLICY
            or p['source_sha256'] != source_hash() or p['analysis'] != ANALYSIS
            or p['contrasts'] != CONTRASTS or set(p['tasks']) != set(TASKS)):
        raise ValueError('Unrecognized source/arm/analysis freeze')
    if check_files:
        for name, digest in p['files_sha256'].items():
            if sha256(checked_path(project, name)) != digest:
                raise ValueError('Frozen artifact changed: ' + name)
    return p


def restore_initial(initial, saved, metadata):
    """Deliver fixed bank pixels; never substitute physical state or a scenario.

    CUDA-context initialization can change four rasterized channel values by one
    level on the receiving host. Apply the existing goal-render diagnostic bound
    to the initial image too, retaining exact delivered observation hashes.
    The caller separately checks rand_vec, full expert goal and expert actions.
    """
    if (observation_digest(saved['initial']) != metadata['initial_sha256']
            or not torch.equal(initial['proprio'].cpu(), saved['initial']['proprio'])):
        raise ValueError('Initial physical/proprio input differs from frozen bank')
    difference = (initial['visual'].cpu().to(torch.int16) - saved['initial']['visual'].to(torch.int16)).abs()
    if int(difference.max()) > 1 or int(difference.ne(0).sum()) > 64:
        raise ValueError('Initial renderer mismatch exceeds unchanged one-level/64-value bound')
    result = initial.clone()
    for key in ('visual', 'proprio'):
        result[key] = saved['initial'][key].to(initial[key]).clone()
    if observation_digest(result) != metadata['initial_sha256']:
        raise ValueError('Initial delivery is not bitwise equal to frozen bank')
    return result


def initialize_renderer(task, env):
    if task == 'pointmaze':
        # The bank's first prepare() initializes the camera at [3,3,0].
        # An earlier generic reset instead centers it on a random state.
        # Use the identical upstream initializer before that generic reset.
        env.proprio_env.unwrapped.prepare_for_render()


@contextlib.contextmanager
def deliver(project, task, row):
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    tensors = torch.load(checked_path(project, row['tensor_path']), map_location='cpu', weights_only=True)
    original = PlanEvaluator.set_episode
    def bound(evaluator, cfg, agent, env, seed, task_idx=-1):
        if seed != row['environment_seed']:
            raise ValueError('Wrong fresh scenario seed')
        initial, goal, expert, success = original(evaluator, cfg, agent, env, seed, task_idx)
        if task.startswith('reach'):
            if (np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist() != row['rand_vec']
                    or not torch.equal(evaluator.expert_actions.cpu(), tensors['expert_actions'])
                    or float(success) != row['expert_goal_success']
                    or not np.array_equal(np.asarray(evaluator.state_g), np.asarray(row['goal_state']))):
                raise ValueError('Native expert physics/actions differ from frozen inputs')
            initial = restore_initial(initial, tensors, row)
            goal = restore_goal(initial, goal, evaluator.state_g, tensors, row)
        if any(observation_digest(value) != row[k + '_sha256'] for k, value in
               (('initial', initial), ('goal', goal))):
            raise ValueError('Receiving initial/goal bytes differ from audited inputs')
        return initial, goal, expert, success
    PlanEvaluator.set_episode = bound
    try:
        yield
    finally:
        PlanEvaluator.set_episode = original


class CheckedObserver(ObservePlanner):
    def __init__(self, adapter, backend, output, arm, fitted, cp, cb, engineering, all_arms=False):
        super().__init__(adapter, backend, output)
        self.arm, self.fitted, self.cp, self.cb = arm, fitted, cp, cb
        self.engineering, self.checked = engineering, set()
        self.all_arms, self.numerical_checks, self.context_keys = all_arms, [], set()

    def __call__(self, context, act_suffix=None, **kwargs):
        result = super().__call__(context, act_suffix, **kwargs)
        horizon, count = act_suffix.shape[:2]
        if self.engineering:
            from .fresh_engineering import context_shape_key, validate_all_arms
            key = context_shape_key(context, act_suffix)
            self.calls[-1]['context_key'] = json.loads(json.dumps(key))
            if self.all_arms and key not in self.context_keys:
                self.numerical_checks.extend(validate_all_arms(context, act_suffix, self.backend,
                                                               self.fitted, self.cp, self.cb))
                self.context_keys.add(key)
        if self.engineering and (horizon, count) not in self.checked:
            if self.arm in ('native', 'zero_dose') or horizon < 6:
                expected = self.backend.predict(context, act_suffix)
            elif self.arm in ARMS[1:3]:
                with NativeFieldCapture(self.backend.predictor) as capture:
                    self.backend.predict(context, act_suffix)
                fields = reference_fields(capture.values[3], self.adapter.bank, self.arm)
                with PredictorIntervention(self.backend.predictor, fields):
                    expected = self.backend.predict(context, act_suffix)
            else:
                fields = source_coupling_fields((self.cp, self.cb, COUPLING[self.arm]), count, self.backend.device)
                with PredictorIntervention(self.backend.predictor, fields):
                    expected = self.backend.predict(context, act_suffix)
            if any(not torch.equal(result[k], expected[k]) for k in ('visual', 'proprio')):
                raise ValueError('Independent original-hook/native reference parity failed')
            self.checked.add((horizon, count))
        return result


def verify_record(record, row, task):
    result, calls = record['result'], record['calls']
    if record['scenario'] != row or any(result[k] != row[k] for k in ('initial_sha256', 'goal_sha256')):
        raise ValueError('Outcome is not bound to its frozen input')
    if type(result['native_success']) is not bool or not all(math.isfinite(result[k]) for k in
            ('native_state_distance', 'native_reward', 'expert_success')):
        raise ValueError('Invalid scientific outcome')
    if task.startswith('reach'):
        verify_episode(result, calls)
    else:
        validate_complete(result, [(r['horizon'], r['candidates']) for r in calls])
    if any(c['backend_calls'] != 1 for c in calls):
        raise ValueError('Extra model work in scientific adapter')


def verify_refined_energy(calls, arm, dose):
    for call in calls:
        energy = call['energy']
        if energy['response_probe_rollouts'] or energy['native_shadow_rollouts']:
            raise ValueError('Legacy solver or shadow rollout used')
        if arm in ARMS[1:3] and call['horizon'] == 6:
            requested, realized = (np.asarray(energy[k]) for k in ('requested_l2', 'realized_l2'))
            active = np.asarray(energy['active'], dtype=bool)
            coefficients = np.asarray(energy['coefficients'])
            if (requested.shape != (call['candidates'],) or realized.shape != requested.shape
                    or active.shape != requested.shape or coefficients.shape != (call['candidates'], 4)
                    or not np.isfinite(coefficients).all() or not np.isfinite(realized).all()
                    or not np.allclose(requested, active * dose, rtol=1e-5, atol=1e-8)
                    or not np.allclose(realized, requested, rtol=5e-4, atol=1e-8)):
                raise ValueError('Refined edit dose/coefficient contract changed')
        elif any(k in energy for k in ('requested_l2', 'realized_l2', 'coefficients')):
            raise ValueError('Native/shortened forecast received a refined edit')


def verify_bundle(root, protocol, task, row, freeze_hash, engineering, uuid=None):
    report = read(root / 'report.json')
    if sha256(root / 'report.json') != read(root / 'DONE.json')['report_sha256']:
        raise ValueError('Incomplete/changed result receipt')
    names = ENGINEERING if engineering else ARMS
    if (report['task'] != task or report['scenario'] != row or report['freeze_sha256'] != freeze_hash
            or report['engineering'] != engineering or set(report['records_sha256']) != set(names)
            or (uuid is not None and report['device_uuid'] != uuid)):
        raise ValueError('Result task/input/device/role mismatch')
    records = {}
    for name in names:
        path = root / (name + '.json')
        if sha256(path) != report['records_sha256'][name]:
            raise ValueError('Changed episode record')
        record = read(path)
        if (record['arm'] != name or record['device_uuid'] != report['device_uuid']
                or record['freeze_sha256'] != freeze_hash
                or record['scientific_efficacy_measurement'] != (not engineering)):
            raise ValueError('Mixed-device or mislabeled paired arms')
        verify_record(record, row, task)
        records[name] = record
    if engineering:
        for name in ENGINEERING[1:]:
            verify_pair(records['native'], records[name], native_repeat=name in ('native_repeat', 'zero_dose'))
        for r in records.values():
            if not {(6, 1), (6, 300)}.issubset({tuple(v) for v in r['source_parity']}):
                raise ValueError('Incomplete engineering source parity')
        verify_numerical_coverage(records['native'])
    return records


def verify_numerical_coverage(native):
    keys = {json.dumps(call['context_key']) for call in native['calls']}
    expected = {(key, arm) for key in keys for arm in ENGINEERING_POLICY['numerical_arms']}
    checks = native['numerical_checks']
    observed = [(json.dumps(check['context_key']), check['arm']) for check in checks]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Missing or duplicate all-arm context/horizon/count engineering')
    for check in checks:
        if check['backend_calls'] != 1 or any(check.get(key) is not True for key in
            ('source_parity_bitwise', 'energy_valid', 'inputs_unchanged', 'parameters_unchanged', 'hooks_restored')):
            raise ValueError('Failed independent all-arm engineering check')


@torch.no_grad()
def execute(args):
    p = validate_freeze(args.freeze, args.project)
    task = p['tasks'][args.task]
    rows = [r for r in task['records'] if r['role'] ==
            ('excluded_engineering' if args.mode == 'engineering' else 'scientific_candidates')]
    row = next(r for r in rows if r['episode'] == args.episode)
    engineering = args.mode == 'engineering'
    digest = sha256(args.freeze / 'protocol.json')
    uuid = device_uuid()
    if not engineering:
        erow = next(r for r in task['records'] if r['role'] == 'excluded_engineering' and r['episode'] == args.engineering_episode)
        verify_bundle(args.engineering, p, args.task, erow, digest, True, uuid)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'DONE.json').exists():
            verify_bundle(args.output, p, args.task, row, digest, engineering, uuid)
            return
        identity = {'task': args.task, 'scenario': row, 'freeze_sha256': digest,
                    'device_uuid': uuid, 'engineering': engineering}
        intent = args.output / 'STARTED.json'
        if intent.exists() and read(intent) != identity:
            raise ValueError('Partial scenario cannot move device or change protocol')
        if not intent.exists():
            write_json(intent, identity)
        vendor = args.project / 'vendor/jepa-wms'
        use_vendor(vendor)
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        from evals.simu_env_planning.planning import plan_evaluator
        fitted, cp, cb = load_assets(args.project, args.task, task['assets'])
        checkpoint = checked_path(args.project, task['assets']['checkpoint'])
        if sha256(checkpoint) != task['checkpoint_sha256']:
            raise ValueError('Receiving checkpoint changed')
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        backend = JepaBackend(vendor, checkpoint, task['checkpoint_sha256'],
            'metaworld' if args.task.startswith('reach') else args.task, 'cuda:0', 'float32')
        versions, hashes = _model_versions(backend.model), {}
        baseline = None
        for name in ENGINEERING if engineering else ARMS:
            target = args.output / (name + '.json')
            if target.exists():
                record = read(target)
                verify_record(record, row, args.task)
                if (record['device_uuid'] != uuid or record['arm'] != name
                        or record['freeze_sha256'] != digest
                        or record['scientific_efficacy_measurement'] != (not engineering)):
                    raise ValueError('Invalid partial-arm resume')
            else:
                arm = 'native' if name == 'native_repeat' else name
                random.seed(0); np.random.seed(0); torch.manual_seed(0)
                cfg = OmegaConf.create(copy.deepcopy(task['planning']['config']))
                cfg.meta.seed = read(args.project / BANK / 'protocol.json')['tasks'][args.task]['planning_contract']['config']['meta']['seed']
                cfg.local_seed = row['environment_seed']
                agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
                env = make_env(cfg)
                initialize_renderer(args.task, env)
                output = args.output / name
                output.mkdir(exist_ok=True)
                adapter = (H6StaticPlanningIntervention(backend, cp, cb, COUPLING[arm]) if arm in COUPLING
                           else FixedResponseIntervention(backend, fitted, arm))
                observer = CheckedObserver(adapter, backend, output, arm, fitted, cp, cb, engineering,
                                           all_arms=engineering and name == 'native')
                agent.planner.unroll = observer
                trace = trace_actor(agent)
                start = time.monotonic()
                try:
                    with close_expert_environments(plan_evaluator), deliver(args.project, args.task, row):
                        result = run_episode(cfg, backend, agent, env, row['environment_seed'])
                finally:
                    env.close()
                    write_json(output / 'calls.json', observer.calls)
                    write_json(output / 'actions.json', trace)
                if arm in COUPLING:
                    verify_energy(observer.calls, cp, cb, COUPLING[arm])
                else:
                    verify_refined_energy(observer.calls, arm, fitted['dose'])
                record = {'scenario': row, 'arm': name, 'result': result, 'calls': observer.calls,
                    'action_trace': trace, 'source_parity': sorted(observer.checked), 'device_uuid': uuid,
                    'numerical_checks': observer.numerical_checks,
                    'seconds': time.monotonic() - start, 'freeze_sha256': digest,
                    'scientific_efficacy_measurement': not engineering}
                verify_record(record, row, args.task)
                temporary = target.with_suffix('.json.tmp')
                write_json(temporary, record)
                temporary.replace(target)
            if baseline is None:
                baseline = record
            else:
                verify_pair(baseline, record, native_repeat=name in ('native_repeat', 'zero_dose'))
            hashes[name] = sha256(target)
            print(json.dumps({'task': args.task, 'episode': args.episode, 'arm_completed': name,
                               'engineering': engineering}), flush=True)
        if versions != _model_versions(backend.model) or uuid != device_uuid() or source_hash() != p['source_sha256']:
            raise ValueError('Frozen model/source/device changed during execution')
        write_json(args.output / 'report.json', {**identity, 'records_sha256': hashes,
            'backend': backend.provenance, 'parameters_unchanged': True,
            'engineering_report_sha256': None if engineering else sha256(args.engineering / 'report.json')})
        write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})
        verify_bundle(args.output, p, args.task, row, digest, engineering, uuid)


def assigned_episodes(worker_id, workers):
    if not 1 <= workers <= 96 or not 0 <= worker_id < workers:
        raise ValueError('Invalid fixed worker assignment')
    return list(range(worker_id, 96, workers))


def worker(args):
    """One GPU, one task, fixed disjoint scenario stripe, no arm splitting."""
    if args.task is None or args.output is None:
        raise ValueError('worker requires task and campaign output root')
    episodes = assigned_episodes(args.worker_id, args.workers)
    campaign = args.output
    uuid = device_uuid()
    engineering = campaign / 'engineering' / args.task / uuid
    # Host-wide lock survives differing output roots and excludes duplicate
    # launchers for the entire stripe, including gaps between scenario jobs.
    with Path('/tmp', 'fresh-confirmation-' + uuid + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check = copy.copy(args)
        check.mode, check.episode, check.output = 'engineering', 0, engineering
        execute(check)
        for episode in episodes:
            job = copy.copy(args)
            job.mode, job.episode = 'run', episode
            job.engineering, job.engineering_episode = engineering, 0
            job.output = campaign / args.task / f'scenario-{episode:03d}'
            execute(job)


def analyze(args):
    p = validate_freeze(args.freeze, args.project)
    digest = sha256(args.freeze / 'protocol.json')
    matrix, sources = {}, {}
    for task in TASKS:
        rows = sorted((r for r in p['tasks'][task]['records'] if r['role'] == 'scientific_candidates'),
                      key=lambda r: r['episode'])
        values = []
        for row in rows:
            root = args.results / task / f'scenario-{row["episode"]:03d}'
            records = verify_bundle(root, p, task, row, digest, False)
            report = read(root / 'report.json')
            engineering = args.results / 'engineering' / task / report['device_uuid']
            erow = read(engineering / 'report.json')['scenario']
            if erow not in [r for r in p['tasks'][task]['records'] if r['role'] == 'excluded_engineering']:
                raise ValueError('Engineering scenario was not prospectively excluded')
            verify_bundle(engineering, p, task, erow, digest, True, report['device_uuid'])
            if report['engineering_report_sha256'] != sha256(engineering / 'report.json'):
                raise ValueError('Science/engineering provenance chain changed')
            values.append([records[a]['result']['native_success'] for a in ARMS])
            sources[str(root)] = sha256(root / 'report.json')
        matrix[task] = np.asarray(values, dtype=float)
    rng = np.random.default_rng(ANALYSIS['bootstrap_seed'])
    results = {}
    alpha = .05 / ANALYSIS['family_size']
    for task, values in matrix.items():
        indices = rng.integers(0, 96, size=(ANALYSIS['bootstrap_draws'], 96))
        contrasts = {}
        for name, weights in CONTRASTS.items():
            differences = values @ np.asarray([weights.get(a, 0) for a in ARMS])
            interval = np.quantile(differences[indices].mean(1), [alpha / 2, 1 - alpha / 2]) * 100
            contrasts[name] = {'difference_pp': float(differences.mean() * 100), 'simultaneous_95_interval_pp': interval.tolist()}
        results[task] = {'n': 96, 'success_percent': dict(zip(ARMS, (values.mean(0) * 100).tolist())), 'contrasts': contrasts}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'report.json', {'method': METHOD, 'freeze_sha256': digest,
        'results': results, 'analysis': ANALYSIS, 'source_reports_sha256': sources,
        'scientific_evaluations': 3072, 'single_released_checkpoint_per_task': True})
    write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'validate', 'engineering', 'run', 'worker', 'analyze'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--episode', type=int)
    parser.add_argument('--engineering', type=Path)
    parser.add_argument('--engineering-episode', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    args.project = args.project.resolve()
    if args.mode == 'freeze':
        if args.assets is None:
            parser.error('freeze requires --assets JSON mapping every task to refined/coupling/checkpoint paths')
        freeze(args)
    elif args.mode == 'validate':
        validate_freeze(args.freeze, args.project)
        print(json.dumps({'freeze_valid': True, 'receiving_runtime_not_implied': True}))
    elif args.mode == 'analyze':
        if args.results is None or args.output is None:
            parser.error('analyze requires --results and --output')
        analyze(args)
    elif args.mode == 'worker':
        worker(args)
    else:
        if args.task is None or args.episode is None or args.output is None or (args.mode == 'run' and args.engineering is None):
            parser.error('execution requires --task, --episode, --output and science requires --engineering')
        execute(args)


if __name__ == '__main__':
    main()
