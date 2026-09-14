"""Frozen DROID coupling panel: native recorded-action endpoint, never robot success.

Fits are separate raw recordings. All64 source-config episodes per arm retain
eight persistent native RNG streams. No offline efficacy or partial-panel gate.
"""
from offline_study._paths import source_path
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.tasks.droid.droid_contract import action_metrics, checkpoint_score, prepare
from offline_study.tasks.droid.droid_coupling import ARMS, METHOD, Hook, Intervention, arm_fields
from offline_study.tasks.droid.droid_native import assert_same, full_episode, load_model, make_dataset, verified_report
from offline_study.tasks.droid.droid_replication import assigned_rows
from offline_study.evaluation.fixed_response_behavior import device_uuid
from offline_study.planning.planning_native_smoke import SMOKE_SEED
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


PAIRS = [(arm + '_vs_native', arm, 'native') for arm in ARMS[1:]] + [
    ('joint_vs_visual_only', 'joint', 'visual_only'),
    ('joint_vs_action_condition_only', 'joint', 'action_condition_only'),
    ('joint_vs_matched_random', 'joint', 'matched_random'),
    ('joint_vs_permuted_joint', 'joint', 'permuted_joint'),
    ('equal_energy_joint_vs_visual_only', 'joint_equal_standardized_energy', 'visual_only'),
    ('equal_energy_joint_vs_action_condition_only', 'joint_equal_standardized_energy', 'action_condition_only'),
    ('equal_energy_joint_vs_matched_random', 'joint_equal_standardized_energy', 'matched_random_equal_standardized_energy')]
INTERACTION = {'joint': 1, 'visual_only': -1, 'action_condition_only': -1, 'native': 1}


def paired_inputs(native, candidate):
    for key in ('episode', 'logical_rank', 'local_seed', 'environment_seed'):
        if native[key] != candidate[key]:
            raise ValueError('DROID episode/RNG pairing changed: ' + key)
    for key in ('initial_sha256', 'goal_sha256', 'dataset_sample', 'recorded_actions'):
        if native['result'][key] != candidate['result'][key]:
            raise ValueError('DROID recorded stimulus pairing changed: ' + key)


def validate_records(records, expected):
    if len(records) != len(expected):
        raise ValueError('Incomplete DROID episode population')
    for row, want in zip(records, expected, strict=True):
        if any(row[key] != value for key, value in want.items()):
            raise ValueError('Wrong persistent DROID stream order')
        result = row['result']
        if result['robot_executions'] != 0 or not result['dummy_success_intentionally_omitted']:
            raise ValueError('Do not interpret the dummy environment as robot success')
        if result['unroll_calls'] != [[3, 300], [3, 1]] * 15:
            raise ValueError('Changed native CEM budget')
        actions = torch.tensor(result['planned_actions'], dtype=torch.float64)
        recorded = torch.tensor(result['recorded_actions'], dtype=torch.float64)
        metrics = action_metrics(actions, recorded)
        if set(metrics) != set(result['metrics']) or any(not np.isfinite(value) or not np.isclose(value, float(metrics[name]), atol=1e-6, rtol=1e-6)
               for name, value in result['metrics'].items()):
            raise ValueError('Changed or nonfinite DROID action metric')


def native_reference(root, contract):
    report, digest = verified_report(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    if (report['status'] != 'released_droid_native_replication_shard_complete' or report['episodes'] != 64 or
            protocol['contract'] != contract or not protocol['native_baseline_only'] or
            protocol['logical_ranks'] != list(range(8))):
        raise ValueError('Require the full released native DROID reference')
    for name, wanted in protocol['source_sha256'].items():
        if sha256(source_path(name)) != wanted:
            raise ValueError('Original DROID native reference source changed')
    names = [f"episode-{row['episode']:03d}.json" for row in contract['episodes']]
    if set(names) != set(report['episode_files_sha256']):
        raise ValueError('Wrong native reference coverage')
    records = []
    for name in names:
        if sha256(root / name) != report['episode_files_sha256'][name]:
            raise ValueError('Native reference episode changed')
        record = json.loads((root / name).read_text())
        if record['arm'] != 'native':
            raise ValueError('Non-native reference arm')
        records.append(record)
    validate_records(records, contract['episodes'])
    return records, digest


def checked_inputs(args):
    use_vendor(args.vendor)
    contract = prepare(args.vendor, json.loads(args.manifest.read_text()))
    fit, fit_hash = verified_report(args.fit)
    fp = json.loads((args.fit / 'protocol.json').read_text())
    if (fit['status'] != 'droid_coupling_fit_complete_not_behavioral_clearance' or
            fit['fit_recordings'] != 128 or fit['fit_prefixes'] != 512 or fp['method'] != METHOD or
            fp['checkpoint_sha256'] != contract['checkpoint_sha256'] or
            fp['evaluation_observations_used_for_fit'] or not fit['parameters_unchanged']):
        raise ValueError('Require the separate complete DROID fit')
    for name, wanted in fit['files_sha256'].items():
        if sha256(args.fit / name) != wanted:
            raise ValueError('DROID fit evidence changed')
    bank = torch.load(args.fit / 'operator_bank.pt', map_location='cpu', weights_only=True)
    if bank['protocol_sha256'] != sha256(args.fit / 'protocol.json'):
        raise ValueError('Operator is not bound to the fixed fit protocol')
    for arm in ARMS:
        arm_fields(bank, arm, 'cpu')
    records, reference_hash = native_reference(args.reference, contract)
    old, old_hash = verified_report(args.native_engineering)
    op = json.loads((args.native_engineering / 'protocol.json').read_text())
    if (old['status'] != 'native_droid_full_cem_engineering_passed' or op['contract'] != contract or
            op['source_sha256'] != sha256(source_path('droid_native.py')) or
            not old['same_seed_actions_and_metrics_exact']):
        raise ValueError('Original native engineering is not equivalent')
    for name, wanted in old['files_sha256'].items():
        if sha256(args.native_engineering / name) != wanted:
            raise ValueError('Original native engineering evidence changed')
    assets, asset_hash = verified_report(args.assets)
    if assets['status'] != 'official_droid_inputs_staged_and_verified':
        raise ValueError('Invalid DROID asset stage')
    bindings = {'fit_report_sha256': fit_hash, 'reference_report_sha256': reference_hash,
        'native_engineering_report_sha256': old_hash, 'assets_report_sha256': asset_hash}
    return contract, bank, records, bindings


def freeze(args):
    contract, _, _, bindings = checked_inputs(args)
    args.freeze.mkdir(parents=True, exist_ok=False)
    protocol = {'role': 'droid_static_coupling_recorded_plan_development', 'method': METHOD,
        'source_sha256': source_hash(), 'bindings': bindings, 'planning': contract,
        'arms': list(ARMS), 'episodes_per_arm_total': 64, 'total_panel_endpoints': 576,
        'pairs': PAIRS, 'factorial_score_interaction': INTERACTION,
        'analysis': {'bootstrap_draws': 20000, 'bootstrap_seed': 2026090801, 'family_size': 16,
            'resampling_unit': 'recording', 'primary': 'difference of native checkpoint scores',
            'simultaneous_interval': 'Bonferroni95 paired recording-cluster percentile bootstrap',
            'minimum_useful_score_gain': 1, 'secondary': 'recorded net XYZ action error and runtime'},
        'precision': 'float32_no_tf32', 'engineering_seed': SMOKE_SEED,
        'native_reference_policy': 'receiving_paired_native' if args.new_native_reference else 'reuse_exact_original',
        'reuse_reference_requires_full_same_seed_native_action_metric_identity': True,
        'reuse_runtime_hardware_equivalence_not_assumed': True,
        'candidate_admission_requires_offline_significance': False, 'selection_from_partial_results': False,
        'fresh_confirmation': False, 'robot_executions': 0, 'automatic_confirmation_authorized': False}
    write_json(args.freeze / 'protocol.json', protocol)
    write_json(args.freeze / 'FROZEN.json', {'protocol_sha256': sha256(args.freeze / 'protocol.json')})


def read_freeze(args):
    contract, bank, native, bindings = checked_inputs(args)
    digest = sha256(args.freeze / 'protocol.json')
    if json.loads((args.freeze / 'FROZEN.json').read_text())['protocol_sha256'] != digest:
        raise ValueError('Behavioral freeze changed')
    protocol = json.loads((args.freeze / 'protocol.json').read_text())
    if (protocol['source_sha256'] != source_hash() or protocol['planning'] != contract or
            protocol['bindings'] != bindings or protocol['arms'] != list(ARMS) or
            protocol['pairs'] != [list(x) for x in PAIRS]):
        raise ValueError('Source/inputs/arm registry drifted after freezing')
    if (protocol['native_reference_policy'] not in ('reuse_exact_original', 'receiving_paired_native') or
            protocol['episodes_per_arm_total'] != 64 or protocol['total_panel_endpoints'] != 576 or
            protocol['factorial_score_interaction'] != INTERACTION or protocol['precision'] != 'float32_no_tf32' or
            protocol['analysis']['bootstrap_draws'] != 20000 or protocol['analysis']['bootstrap_seed'] != 2026090801 or
            protocol['analysis']['family_size'] != 16 or protocol['analysis']['minimum_useful_score_gain'] != 1 or
            protocol['fresh_confirmation'] or protocol['robot_executions'] or
            protocol['candidate_admission_requires_offline_significance'] or protocol['selection_from_partial_results']):
        raise ValueError('Frozen DROID population, analysis or reference policy changed')
    return protocol, digest, bank, native


def scientific_shard(root, protocol, digest, rank, arm, expected_device=None):
    report, report_hash = verified_report(root)
    launch = json.loads((root / 'protocol.json').read_text())
    assigned = assigned_rows(protocol['planning']['episodes'], [rank])
    expected_files = {f"{kind}-{row['episode']:03d}.json" for row in assigned for kind in ('episode', 'trace')}
    if (report['status'] != 'droid_coupling_scientific_shard_complete' or report['freeze_sha256'] != digest or
            launch['freeze_sha256'] != digest or launch['source_sha256'] != protocol['source_sha256'] or
            launch['role'] != 'scientific_development' or launch['arm'] != arm or launch['logical_ranks'] != [rank] or
            report['episodes'] != len(assigned) or set(report['files_sha256']) != expected_files or
            not report['parameters_unchanged'] or report['fresh_confirmation'] or report['robot_executions'] or
            not launch['engineering_report_sha256'] or launch['device_uuid'] != report['device_uuid'] or
            (expected_device is not None and report['device_uuid'] != expected_device)):
        raise ValueError('Incomplete or changed DROID shard/source/device binding')
    for name, wanted in report['files_sha256'].items():
        if sha256(root / name) != wanted:
            raise ValueError('DROID panel member changed')
    records = [json.loads((root / f"episode-{row['episode']:03d}.json").read_text()) for row in assigned]
    if any(row['arm'] != arm or row['device_uuid'] != report['device_uuid'] for row in records):
        raise ValueError('DROID record arm/device mislabeled')
    validate_records(records, assigned)
    return records, report, report_hash


def receiving_native(panel, protocol, digest, ranks):
    records = []
    for rank in sorted(ranks):
        root = panel / 'conditions/native' / f'shard-{rank}'
        found, _, _ = scientific_shard(root, protocol, digest, rank, 'native', device_uuid())
        records.extend(found)
    validate_records(records, assigned_rows(protocol['planning']['episodes'], ranks))
    return records


class IndependentFields:
    """Engineering-only reference implementation; no online checks in scientific runs."""
    def __init__(self, predictor, visual, action):
        self.predictor, self.visual, self.action = predictor, visual, action
        self.step, self.handles = 0, []
    def __enter__(self):
        def incoming(module, args):
            self.step += 1
            if self.step == 3 and self.visual is not None:
                tensor = args[0].clone()
                tensor[:, -1:] = tensor[:, -1:] + self.visual.reshape(1, 1, 1, 16, 16, 1024)
                return (tensor, *args[1:])
        def block(module, args):
            if self.step == 3 and self.action is not None:
                condition = args[1].clone()
                condition[:, -1:] = condition[:, -1:] + self.action.reshape(1, 1, 1024)
                return (args[0], condition, *args[2:])
        self.handles = [self.predictor.register_forward_pre_hook(incoming),
                        self.predictor.predictor_blocks[6].register_forward_pre_hook(block)]
        return self
    def __exit__(self, kind, *args):
        for handle in self.handles:
            handle.remove()
        if kind is None and self.step != 3:
            raise ValueError('Independent native H3 coverage failed')


def reference_fields(bank, arm, device):
    if arm == 'zero_hook':
        return torch.zeros_like(bank['visual'], device=device), torch.zeros_like(bank['action'], device=device)
    if arm == 'native':
        return None, None
    visual_key = 'visual'
    if arm in ('permuted_visual', 'permuted_joint'):
        visual_key = 'permuted_visual'
    if arm in ('matched_random', 'matched_random_equal_standardized_energy'):
        visual_key = 'random_visual'
    action_key = 'random_action' if 'matched_random' in arm else 'action'
    visual = bank[visual_key].float() * bank['visual_dose']
    action = bank[action_key].float() * bank['action_dose']
    if 'equal_standardized_energy' in arm:
        visual = visual * (1 / np.sqrt(2))
        action = action * (1 / np.sqrt(2))
    return (None if arm == 'action_condition_only' else visual.to(device),
            None if arm in ('visual_only', 'permuted_visual') else action.to(device))


def episode(cfg, model, dset, prep, bank, arm, target, row, agent=None, engineering=False):
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    if agent is None:
        agent = GC_Agent(cfg, model, dset=dset, preprocessor=prep)
    base = agent.planner.unroll
    operator = None if arm == 'zero_hook' else Intervention(model, bank, arm)
    seen, trace = set(), []
    def unroll(context, act_suffix=None, **kwargs):
        batch = act_suffix.shape[1]
        reference = None
        if engineering and batch not in seen:
            v, a = reference_fields(bank, arm, model.device)
            with IndependentFields(model.model.predictor, v, a):
                reference = model.unroll(context, act_suffix=act_suffix, **kwargs)
            seen.add(batch)
        if arm == 'zero_hook':
            v, a = reference_fields(bank, arm, model.device)
            with Hook(model.model.predictor, v, a):
                result = model.unroll(context, act_suffix=act_suffix, **kwargs)
            energy = {}
        else:
            result = operator(context, act_suffix=act_suffix, **kwargs)
            if operator.backend_calls != 1:
                raise ValueError('Unexpected extra runtime model pass')
            energy = {name: {'requested_l2': item['requested_l2'],
                'realized_l2_min': float(item['realized_l2'].min()),
                'realized_l2_max': float(item['realized_l2'].max())}
                for name, item in operator.last_energy.items()}
        if reference is not None and not torch.equal(result, reference):
            raise ValueError('Independent DROID field prediction parity failed')
        trace.append({'horizon': act_suffix.shape[0], 'candidates': batch, 'energy': energy,
                      'scientific_backend_calls': 1})
        return result
    agent.planner.unroll = unroll
    try:
        result = full_episode(cfg, model, dset, prep, target, row['episode'],
            seed=row['environment_seed'], agent=agent,
            role='engineering_not_efficacy' if engineering else 'droid_coupling_recorded_plan_development')
    finally:
        agent.planner.unroll = base
    if len(trace) != 30 or (engineering and seen != {1, 300}):
        raise ValueError('Incomplete field/candidate audit')
    return result, trace


@torch.no_grad()
def execute(args):
    protocol, digest, bank, native = read_freeze(args)
    engineering = args.command == 'engineer'
    engineering_hash = None
    if not engineering:
        proof, engineering_hash = verified_report(args.engineering)
        if (proof['status'] != 'droid_coupling_full_cem_engineering_passed' or
                proof['freeze_sha256'] != digest or proof['device_uuid'] != device_uuid() or
                (protocol['native_reference_policy'] == 'reuse_exact_original' and not proof['old_native_full_episode_exact']) or
                not proof['same_device_repeat_zero_exact'] or proof['episodes'] != 11 or
                not proof['independent_fields_at_1_and_300'] or not proof['parameters_unchanged']):
            raise ValueError('Require receiving-device full engineering before scientific execution')
        for name, wanted in proof['files_sha256'].items():
            if sha256(args.engineering / name) != wanted:
                raise ValueError('Engineering evidence changed')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'freeze_sha256': digest, 'source_sha256': source_hash(),
        'role': 'engineering' if engineering else 'scientific_development', 'arm': args.arm,
        'logical_ranks': args.logical_ranks, 'device_uuid': device_uuid(),
        'engineering_report_sha256': engineering_hash, 'fresh_confirmation': False})
    started = time.monotonic()
    try:
        torch.cuda.set_device(0); torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        contract = protocol['planning']
        _, prep = make_dataset(args.assets, contract)
        model, provenance = load_model(args.vendor, args.assets, args.encoder_source, args.encoder_root, contract, prep)
        versions = [(name, p._version) for name, p in model.named_parameters()]
        from omegaconf import OmegaConf
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        records, files = [], []
        rows = assigned_rows(contract['episodes'], args.logical_ranks) if not engineering else []
        native_by_id = {row['episode']: row for row in native}
        if not engineering and args.arm != 'native' and protocol['native_reference_policy'] == 'receiving_paired_native':
            native_by_id = {row['episode']: row for row in receiving_native(args.freeze.parent, protocol, digest, args.logical_ranks)}
        original_exact = True
        jobs = [(arm, None) for arm in ('native', 'native', 'zero_hook', *ARMS[1:])] if engineering else [
            (args.arm, rank) for rank in sorted(args.logical_ranks)]
        for index, (arm, rank) in enumerate(jobs):
            seed = SMOKE_SEED if engineering else 1
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            dset, prep = make_dataset(args.assets, contract)
            cfg = OmegaConf.create(copy.deepcopy(contract['config']))
            rank_rows = [{'episode': index, 'environment_seed': SMOKE_SEED}] if engineering else [r for r in rows if r['logical_rank'] == rank]
            cfg.local_seed = SMOKE_SEED if engineering else rank_rows[0]['local_seed']
            if engineering:
                cfg.meta.seed = SMOKE_SEED
            agent = GC_Agent(cfg, model, dset=dset, preprocessor=prep)
            for row in rank_rows:
                before = time.monotonic()
                result, trace = episode(cfg, model, dset, prep, bank, arm, args.output, row, agent, engineering)
                if engineering and index < 3:
                    original = json.loads((args.native_engineering / 'repetition-0.json').read_text())['result']
                    if protocol['native_reference_policy'] == 'reuse_exact_original':
                        assert_same(original, result)
                    else:
                        # A new GPU may differ numerically. Preserve source inputs,
                        # require same-device repeat/actual-zero identity, and run
                        # new paired native streams instead of reusing old outcomes.
                        for key in ('initial_sha256', 'goal_sha256', 'dataset_sample', 'recorded_actions'):
                            assert_same(original[key], result[key])
                        if index > 0:
                            assert_same(records[0]['result'], result)
                        original_exact = original_exact and original == result
                if not engineering:
                    paired_inputs(native_by_id[row['episode']], {**row, 'result': result})
                record = {**row, 'arm': arm, 'result': result, 'seconds': time.monotonic() - before,
                          'device_uuid': device_uuid()}
                records.append(record)
                for name, payload in ((f"episode-{row['episode']:03d}.json", record),
                                      (f"trace-{row['episode']:03d}.json", trace)):
                    write_json(args.output / name, payload); files.append(name)
                write_json(args.output / 'progress.json', {'completed': len(records),
                    'target': 11 if engineering else len(rows), 'arm': arm,
                    'engineering_only': engineering, 'seconds': time.monotonic() - started})
        if not engineering:
            validate_records(records, rows)
        if versions != [(name, p._version) for name, p in model.named_parameters()] or source_hash() != protocol['source_sha256']:
            raise ValueError('Frozen parameters or source changed')
        write_json(args.output / 'report.json', {
            'status': 'droid_coupling_full_cem_engineering_passed' if engineering else 'droid_coupling_scientific_shard_complete',
            'protocol_sha256': sha256(args.output / 'protocol.json'), 'freeze_sha256': digest,
            'device_uuid': device_uuid(), 'episodes': len(records), 'model_provenance': provenance,
            'files_sha256': {name: sha256(args.output / name) for name in files},
            'old_native_full_episode_exact': engineering and original_exact,
            'same_device_repeat_zero_exact': engineering, 'independent_fields_at_1_and_300': engineering,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
            'parameters_unchanged': True, 'seconds': time.monotonic() - started,
            'fresh_confirmation': False, 'robot_executions': 0})
        write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})
    except Exception as exc:
        write_json(args.output / 'FAILED.json', {'error': str(exc), 'partial_not_complete': True}); raise


def analyze(panels, episodes):
    if tuple(panels) != ARMS:
        raise ValueError('Require all nine frozen DROID arms')
    for arm, rows in panels.items():
        validate_records(rows, episodes)
        for native, row in zip(panels['native'], rows, strict=True):
            paired_inputs(native, row)
    groups = {}
    for index, row in enumerate(panels['native']):
        groups.setdefault(row['result']['dataset_sample']['path'], []).append(index)
    if not 2 <= len(groups) <= 15:
        raise ValueError('Unrecognized DROID recording clusters')
    clusters = list(groups.values()); sizes = np.array([len(g) for g in clusters])
    draws = np.random.default_rng(2026090801).integers(len(clusters), size=(20000, len(clusters)))
    denominator = sizes[draws].sum(1)
    scores, sampled, summaries = {}, {}, {}
    for arm, rows in panels.items():
        errors = np.array([r['result']['metrics']['action_error_xyz'] for r in rows])
        score = float(checkpoint_score(errors))
        totals = np.array([errors[g].sum() for g in clusters])
        sampled[arm] = np.maximum(0, 800 * (.1 - totals[draws].sum(1) / denominator))
        scores[arm] = score
        summaries[arm] = {'episodes': 64, 'recordings': len(groups), 'checkpoint_score': score,
            'mean_xyz_action_error': float(errors.mean()),
            'mean_episode_seconds': float(np.mean([r['seconds'] for r in rows])),
            'runtime_hardware_parity_not_assumed': True, 'score_is_not_success_percentage': True}
    contrasts = [(name, scores[a] - scores[b], sampled[a] - sampled[b]) for name, a, b in PAIRS]
    contrasts.append(('factorial_score_interaction', sum(scores[a] * w for a, w in INTERACTION.items()),
                      sum(sampled[a] * w for a, w in INTERACTION.items())))
    results = [{'contrast': name, 'score_difference': float(point),
        'simultaneous_95_interval_score_points': np.quantile(values, [.05/32, 1-.05/32]).tolist()}
        for name, point, values in contrasts]
    return {'status': 'complete_droid_coupling_recorded_plan_development_analysis',
        'arm_summaries': summaries, 'contrasts': results, 'recording_draw_counts': {k: len(v) for k, v in groups.items()},
        'bootstrap_draws': 20000, 'interval_family_size': 16,
        'few_recordings_limit_interval_accuracy': True, 'fresh_confirmation': False,
        'robot_executions': 0, 'dummy_success_reported': False, 'training_seed_history_complete': False}


def analyze_files(args):
    protocol, digest, _, native = read_freeze(args)
    recollect_native = protocol['native_reference_policy'] == 'receiving_paired_native'
    panels, receipts = ({} if recollect_native else {'native': native}), {}
    devices = {}
    for arm in (ARMS if recollect_native else ARMS[1:]):
        rows = []
        for rank in range(8):
            root = args.panel / 'conditions' / arm / f'shard-{rank}'
            expected_device = devices.get(rank) if recollect_native else None
            found, report, report_hash = scientific_shard(root, protocol, digest, rank, arm, expected_device)
            if recollect_native:
                if arm == 'native':
                    devices[rank] = report['device_uuid']
                elif report['device_uuid'] != devices[rank]:
                    raise ValueError('Candidate/native device pairing changed')
            rows.extend(found); receipts[str(root)] = report_hash
        panels[arm] = rows
    result = analyze(panels, protocol['planning']['episodes'])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'freeze_sha256': digest, 'source_sha256': source_hash(), 'receipts': receipts})
    write_json(args.output / 'report.json', {**result, 'protocol_sha256': sha256(args.output / 'protocol.json')})
    write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'engineer', 'run', 'analyze'))
    for name in ('vendor', 'assets', 'manifest', 'encoder-source', 'encoder-root', 'fit', 'reference', 'native-engineering', 'freeze'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--engineering', type=Path); p.add_argument('--output', type=Path); p.add_argument('--panel', type=Path)
    p.add_argument('--arm', choices=ARMS); p.add_argument('--logical-ranks', nargs='+', type=int)
    p.add_argument('--new-native-reference', action='store_true', help='Freeze a new64-total receiving-device native panel')
    args = p.parse_args()
    {'freeze': freeze, 'engineer': execute, 'run': execute, 'analyze': analyze_files}[args.command](args)


if __name__ == '__main__':
    main()
