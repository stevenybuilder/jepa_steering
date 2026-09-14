"""Prospectively frozen two-task fixed-checkpoint confirmation, not development.

Separate access contract; the old development runner still rejects protected
inputs. No new fitting, expensive online probes, arm search or extra tasks.
"""
from offline_study._paths import PACKAGE_ROOT
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.models.backends import JepaBackend
from offline_study.evaluation.behavioral_development import validate_coverage, verified_report
from offline_study.planning.decision_runtime import source_binding
from offline_study.interventions.fixed_response import METHOD, FixedResponseIntervention, load_fitted_bank
from offline_study.evaluation.fixed_response_behavior import ANALYSIS, ARMS, CONTRASTS, TASKS, ObservePlanner, coupling_binding, device_uuid, frozen_protocol, verify_cem, verify_coupling_engineering
from offline_study.evaluation.fixed_response_behavior_analysis import analyze, validate_panel
from offline_study.validation.fixed_response_smoke import trace_actor, verify_episode
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.planning.planning_contract import prepare, seed_schedule
from offline_study.planning.planning_env_smoke import observation_digest
from offline_study.planning.planning_goal_bank import GoalBank
from offline_study.planning.planning_native_smoke import CHECKPOINTS, run_episode
from offline_study.planning.planning_panel_engineering import COUPLING_ARMS, H6StaticPlanningIntervention
from offline_study.planning.planning_scenarios import close_expert_environments
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor

ROLE = 'independent_fixed_checkpoint_five_arm_confirmation_v1'
SCENARIOS = seed_schedule(1)
RUN_NAME = 'confirmation-20260911-v1'


class ReservedGoalBank(GoalBank):
    """Same checked native delivery callback; only the verified input reader differs."""
    def __init__(self, root, task):
        self.root, self.task, self.cache = root, task, {}
        self.report, self.report_hash = verified_report(root)
        report = self.report
        contract = json.loads((root / 'input_contract.json').read_text())
        if ((root/'FAILED.json').exists() or report['status'] != 'paper_scale_metaworld_inputs_prepared'
            or report['task'] != task or report['learned_policy_episodes'] != 0
            or report['prepared_scenarios'] != 96 or report['unique_initial_state_vectors'] != 96
            or report['input_contract_sha256'] != sha256(root/'input_contract.json')
            or report['scenarios_sha256'] != sha256(root/'scenarios.json')
            or contract['learned_policy_execution'] is not False
            or contract['outcome_filtering'] is not False
            or contract['contract']['episodes'] != SCENARIOS):
            raise ValueError('Unverified reserved cohort')
        rows = json.loads((root/'scenarios.json').read_text())
        self.rows = {r['episode']: r for r in rows}
        if len(rows) != 96 or len(self.rows) != 96 or len({tuple(r['rand_vec']) for r in rows}) != 96:
            raise ValueError('Missing or repeated independent initial states')
        if set(report['tensor_files_sha256']) != {f'logical-rank-{r:02d}.pt' for r in range(8)}:
            raise ValueError('Incomplete tensor inventory')
        for name, digest in report['tensor_files_sha256'].items():
            if sha256(root/name) != digest:
                raise ValueError('Changed protected input bytes')

    def load(self, row):
        if row not in SCENARIOS:
            raise ValueError('Only the frozen reserved stream is authorized')
        rank = row['logical_rank']
        if rank not in self.cache:
            self.cache[rank] = torch.load(self.root/f'logical-rank-{rank:02d}.pt',
                                         map_location='cpu', weights_only=True)
        metadata, tensors = self.rows[row['episode']], self.cache[rank][row['episode']]
        if any(metadata[k] != row[k] for k in row):
            raise ValueError('Protected input seed mismatch')
        for key in ('initial', 'goal'):
            if observation_digest(tensors[key]) != metadata[key+'_sha256']:
                raise ValueError('Protected tensor observation hash mismatch')
        return metadata, tensors


def layout(project):
    artifacts = project/'artifacts/offline_study'
    return {'fixed': artifacts/'fixed-response-20260908-v1',
            'original': artifacts/'primary-durable-20260907',
            'stimuli': artifacts/'restored-behavioral-inputs-20260908-v1',
            'reserve': artifacts/'primary-durable-20260907/planning-scenarios-20260907'}


def freeze(args):
    paths = layout(args.project)
    audit_path = args.output/'EXPOSURE_AUDIT.json'
    audit = json.loads(audit_path.read_text())
    if not audit['passed'] or audit['reserved_outcome_matches'] or audit['actual_outcome_records_examined'] < 960:
        raise ValueError('Protected exposure gate failed')
    legacy = frozen_protocol(paths['stimuli']/'behavioral-development-freeze-20260907-v1')
    tasks = {}
    for task in TASKS:
        bank = ReservedGoalBank(paths['reserve']/(task+'-v1'), task)
        for row in SCENARIOS:
            bank.load(row)
        if bank.report_hash != audit['reserved_reports_sha256'][task]:
            raise ValueError('Exposure audit binds another cohort')
        fit = paths['fixed']/'fits'/task
        load_fitted_bank(fit, task='mw-'+task, checkpoint_sha256=CHECKPOINTS['metaworld'])
        _, _, coupling = coupling_binding(paths['original'], legacy, task)
        tasks[task] = {'planning': prepare(args.vendor, task), 'reserve_report_sha256': bank.report_hash,
            'fixed_fit_done_sha256': sha256(fit/'DONE.json'), 'fixed_bank_sha256': sha256(fit/'operator_bank.pt'),
            'coupling': coupling}
    protocol = {'role': ROLE, 'method': METHOD, 'source_sha256': source_hash(),
        'source_binding': source_binding(args.vendor), 'tasks': tasks, 'arms': list(ARMS),
        'analysis': ANALYSIS, 'contrasts': [list(c) for c in CONTRASTS], 'episodes': SCENARIOS,
        'episodes_per_task_condition': 96, 'total_full_episodes': 960,
        'checkpoint_sha256': CHECKPOINTS['metaworld'], 'exposure_audit_sha256': sha256(audit_path),
        'confirmation_outcomes_authorized': True, 'fresh_confirmation': True,
        'scope': 'independent re-test of unchanged methods, not development-qualified winners',
        'statistical_scope': 'valid independent replication; NOT adequately powered to settle a 5pp gain',
        'joint_80_percent_power_claimed': False, 'training_histories_reproduced': False,
        'maximum_compute_and_transfer_usd': 65, 'maximum_fleet_hourly_usd': 7,
        'precision': 'float32_strict_no_tf32', 'legacy_expensive_operator_enabled': False,
        'online_response_probes': 0, 'native_shadow_rollouts': 0,
        'new_fit_or_dose_search': False, 'stop_on_significance': False,
        'failure_policy': 'preserve partial stream; no outcome-driven dropping or replacement',
        'goals': 'original reserved expert goals, unchanged native physical sampling; exact saved pixel delivery',
        'logical_streams': '8/task; 12 episodes/stream; all five arms on same physical GPU',
        'completed_development_results_reused': False,
        'analysis_role': 'confirmation-only contrasts; report development separately, no pooled primary test',
        'receiving_gate': 'four full excluded-engineering episodes per actual GPU, before protected policy outcomes'}
    if (args.output/'FROZEN.json').exists():
        raise ValueError('Do not overwrite a scientific freeze')
    write_json(args.output/'protocol.json', protocol)
    write_json(args.output/'FROZEN.json', {'protocol_sha256': sha256(args.output/'protocol.json'),
        'protected_policy_outcomes_observed_before_freeze': False})


def validate_freeze(root, vendor=None):
    p = frozen_protocol(root)
    if (p['role'] != ROLE or p['method'] != METHOD or p['arms'] != list(ARMS)
        or p['episodes'] != SCENARIOS or p['analysis'] != ANALYSIS
        or tuple(p['tasks']) != TASKS or p['total_full_episodes'] != 960
        or p['confirmation_outcomes_authorized'] is not True or p['fresh_confirmation'] is not True
        or p['legacy_expensive_operator_enabled'] is not False
        or p['online_response_probes'] != 0 or p['native_shadow_rollouts'] != 0
        or p['exposure_audit_sha256'] != sha256(root/'EXPOSURE_AUDIT.json')):
        raise ValueError('Invalid confirmation access contract')
    audit = json.loads((root/'EXPOSURE_AUDIT.json').read_text())
    if not audit['passed'] or audit['reserved_outcome_matches']:
        raise ValueError('Exposure gate failed')
    if vendor is not None and (p['source_sha256'] != source_hash() or p['source_binding'] != source_binding(vendor)):
        raise ValueError('Frozen source changed')
    return p


class ConfirmationObserver(ObservePlanner):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        write_json(self.output/'progress.json', {'completed_calls': len(self.calls), 'role': ROLE,
                                                'fresh_confirmation': True})
        return result


@torch.no_grad()
def run(args):
    p = validate_freeze(args.freeze, args.vendor)
    paths = layout(args.project)
    task = p['tasks'][args.task]
    fit = paths['fixed']/'fits'/args.task
    bank = load_fitted_bank(fit, task='mw-'+args.task, checkpoint_sha256=CHECKPOINTS['metaworld'])
    if sha256(fit/'DONE.json') != task['fixed_fit_done_sha256'] or sha256(fit/'operator_bank.pt') != task['fixed_bank_sha256']:
        raise ValueError('Frozen fit changed')
    goals = ReservedGoalBank(paths['reserve']/(args.task+'-v1'), args.task)
    if goals.report_hash != task['reserve_report_sha256']:
        raise ValueError('Frozen reserve changed')
    legacy = frozen_protocol(paths['stimuli']/'behavioral-development-freeze-20260907-v1')
    coupling_root, coupling_protocol, bindings = coupling_binding(paths['original'], legacy, args.task)
    if bindings != task['coupling']:
        raise ValueError('Frozen coupling changed')
    proof = verify_cem(args.engineering, fit, PACKAGE_ROOT, args.task)
    worker = json.loads((args.engineering/'WORKER.json').read_text())
    if worker['cem_report_sha256'] != proof or worker['device_uuid'] != device_uuid() or not worker['same_device_before_after']:
        raise ValueError('Missing receiving-device full native-repeat check')
    coupling_check = None
    if args.arm in COUPLING_ARMS:
        coupling_check = verify_coupling_engineering(paths['stimuli']/'planning-panel-coupling-engineering-20260907-v1'/
            (args.task+'-'+args.arm), coupling_root, args.task, args.arm)
    if prepare(args.vendor, args.task) != task['planning']:
        raise ValueError('Native full planner changed')
    rows = [r for r in SCENARIOS if r['logical_rank'] == args.logical_rank]
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    args.output.mkdir(parents=True, exist_ok=False)
    started, records = time.monotonic(), []
    try:
        write_json(args.output/'protocol.json', {'role': ROLE, 'task': args.task, 'arm': args.arm,
            'freeze_sha256': sha256(args.freeze/'protocol.json'), 'source_sha256': source_hash(),
            'expected_episodes': rows, 'device_uuid': device_uuid(), 'engineering_report_sha256': proof,
            'worker_sha256': sha256(args.engineering/'WORKER.json'), 'coupling_engineering_sha256': coupling_check})
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS['metaworld'], 'metaworld', 'cuda:0', 'float32')
        versions = _model_versions(backend.model)
        coupling_bank = torch.load(coupling_root/'operator_bank.pt', map_location='cpu', weights_only=True)
        cfg = OmegaConf.create(task['planning']['config'])
        cfg.local_seed = rows[0]['local_seed']
        agent, env = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor), make_env(cfg)
        torch.cuda.reset_peak_memory_stats()
        try:
            for row in rows:
                before = time.monotonic()
                episode = args.output/f"calls-{row['episode']:03d}"
                episode.mkdir()
                adapter = (H6StaticPlanningIntervention(backend, coupling_protocol, coupling_bank, COUPLING_ARMS[args.arm])
                    if args.arm in COUPLING_ARMS else FixedResponseIntervention(backend, bank, args.arm))
                observed = ConfirmationObserver(adapter, backend, episode)
                agent.planner.unroll = observed
                original_act = agent.act
                trace = trace_actor(agent)
                try:
                    with close_expert_environments(plan_evaluator), goals.deliver(row):
                        result = run_episode(cfg, backend, agent, env, row['environment_seed'])
                finally:
                    agent.act = original_act
                    write_json(episode/'unroll_calls.json', observed.calls)
                    write_json(episode/'action_trace.json', trace)
                verify_episode(result, observed.calls)
                metadata, _ = goals.load(row)
                initial = np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist()
                if initial != metadata['rand_vec'] or any(result[k] != metadata[k] for k in ('initial_sha256','goal_sha256')):
                    raise ValueError('Delivered scenario differs from protected bank')
                record = {**row, 'arm': args.arm, 'result': result, 'initial_state_vector': initial,
                    'seconds': time.monotonic()-before, 'unroll_calls_sha256': sha256(episode/'unroll_calls.json'),
                    'action_trace_sha256': sha256(episode/'action_trace.json')}
                validate_coverage([record], [row])
                records.append(record)
                write_json(args.output/f"episode-{row['episode']:03d}.json", record)
                progress = {'role': ROLE, 'task': args.task, 'arm': args.arm, 'completed': len(records), 'target': 12}
                write_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        finally:
            env.close()
        validate_coverage(records, rows)
        if versions != _model_versions(backend.model) or p['source_sha256'] != source_hash():
            raise ValueError('Frozen parameters/source changed')
        write_json(args.output/'report.json', {'status': 'confirmation_shard_complete', 'task': args.task,
            'arm': args.arm, 'episodes': len(records), 'device_name': torch.cuda.get_device_name(),
            'device_uuid': device_uuid(), 'protocol_sha256': sha256(args.output/'protocol.json'),
            'episode_files_sha256': {f"episode-{r['episode']:03d}.json": sha256(args.output/f"episode-{r['episode']:03d}.json") for r in records},
            'seconds': time.monotonic()-started, 'parameters_unchanged': True,
            'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'fresh_confirmation': True})
        write_json(args.output/'DONE.json', {'report_sha256': sha256(args.output/'report.json')})
    except Exception as exc:
        write_json(args.output/'FAILED.json', {'error': str(exc), 'partial_not_complete': True})
        raise


def load_panel(root, freeze):
    p, panel, hashes = validate_freeze(freeze), {}, {}
    for task in TASKS:
        panel[task] = {}
        devices = {}
        for arm in ARMS:
            records = []
            for rank in range(8):
                shard = root/task/arm/f'rank-{rank}'
                report, digest = verified_report(shard)
                launch = json.loads((shard/'protocol.json').read_text())
                expected = [r for r in SCENARIOS if r['logical_rank'] == rank]
                if ((shard/'FAILED.json').exists() or report['status'] != 'confirmation_shard_complete'
                    or report['fresh_confirmation'] is not True or report['parameters_unchanged'] is not True
                    or report['episodes'] != 12 or report['task'] != task or report['arm'] != arm
                    or report['protocol_sha256'] != sha256(shard/'protocol.json')
                    or launch['role'] != ROLE or launch['freeze_sha256'] != sha256(freeze/'protocol.json')
                    or launch['source_sha256'] != p['source_sha256'] or launch['expected_episodes'] != expected):
                    raise ValueError('Unbound/incomplete confirmation shard')
                identity = report['device_name'], report['device_uuid']
                if devices.setdefault(rank, identity) != identity:
                    raise ValueError('Paired arms must run on the same GPU')
                hashes[str(shard/'report.json')] = digest
                shard_rows = []
                for name, digest in sorted(report['episode_files_sha256'].items()):
                    if Path(name).name != name or not name.startswith('episode-') or sha256(shard/name) != digest:
                        raise ValueError('Changed episode record')
                    row = json.loads((shard/name).read_text())
                    calls = shard/f"calls-{row['episode']:03d}"
                    for filename, key in (('unroll_calls.json','unroll_calls_sha256'), ('action_trace.json','action_trace_sha256')):
                        if sha256(calls/filename) != row[key]:
                            raise ValueError('Changed call/action trace')
                    trace = json.loads((calls/'unroll_calls.json').read_text())
                    verify_episode(row['result'], trace)
                    if any(c['backend_calls'] != 1 for c in trace):
                        raise ValueError('Extra forecasts in confirmation')
                    shard_rows.append(row)
                validate_coverage(shard_rows, expected)
                records.extend(shard_rows)
            panel[task][arm] = records
    validate_panel(panel, SCENARIOS)
    return panel, hashes


def analyze_confirmation(args):
    panel, hashes = load_panel(args.root, args.freeze)
    result = analyze(panel, expected=SCENARIOS)
    result.update(status='independent_fixed_checkpoint_confirmation_complete', fresh_confirmation=True,
        confirmation_outcomes_authorized=True, selection_on_development_only=False,
        analysis_role='unchanged_method_independent_retest_no_candidate_search',
        freeze_sha256=sha256(args.freeze/'protocol.json'), input_reports_sha256=hashes)
    for value in result['tasks'].values():
        value['method_meeting_frozen_gain_rule'] = value.pop('selected_for_later_confirmation_freeze')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'report.json', result)
    write_json(args.output/'DONE.json', {'report_sha256': sha256(args.output/'report.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze','run','analyze'))
    for name in ('project','vendor','output','freeze','checkpoint','engineering','root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--logical-rank', type=int, choices=range(8))
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'analyze': analyze_confirmation}[args.mode](args)


if __name__ == '__main__':
    main()
