"""DROID fixed-map recorded-plan panel:64 total per arm, never robot success."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .author_fit import source_hash
from .droid_assets import verify_download
from .droid_contract import prepare
from .droid_coupling_behavior import native_reference, paired_inputs, validate_records
from .droid_fixed_response import FieldHook, Intervention, load_bank
from .droid_native import assert_same, full_episode, load_model, make_dataset, verified_report
from .droid_replication import assigned_rows
from .fixed_response_behavior import device_uuid
from .planning_native_smoke import SMOKE_SEED
from .protocol import sha256, write_json
from .refined_task_behavior import ANALYSIS, ARMS, ENGINEERING
from .vendor import use_vendor

METHOD = 'droid_fixed_response_behavior_20260911_v1'


def read(path):
    return json.loads(path.read_text())


def inputs(a):
    use_vendor(a.vendor)
    manifest = read(a.manifest)
    contract = prepare(a.vendor, manifest)
    bank = load_bank(a.fit, contract['checkpoint_sha256'])
    historical, reference_hash = native_reference(a.reference, contract)
    expected_paths = {str(a.assets / 'downloads/dataset' / name) for name in contract['released_config_recordings']}
    if not {row['result']['dataset_sample']['path'] for row in historical} <= expected_paths:
        raise ValueError('Restore the original DROID asset path layout before launch; do not discover path mismatches on GPU')
    native, native_hash = verified_report(a.native_engineering)
    if (native['status'] != 'native_droid_full_cem_engineering_passed'
            or read(a.native_engineering / 'protocol.json')['contract'] != contract
            or native['same_seed_actions_and_metrics_exact'] is not True):
        raise ValueError('Original complete DROID native receiving proof missing')
    for name, digest in native['files_sha256'].items():
        if Path(name).name != name or sha256(a.native_engineering / name) != digest:
            raise ValueError('Original native engineering member changed')
    assets, assets_hash = verified_report(a.assets)
    if assets['status'] != 'official_droid_inputs_staged_and_verified':
        raise ValueError('Require verified official DROID asset stage')
    for entry in manifest['assets']:
        verify_download(a.assets / 'downloads' / entry['repo_type'] / entry['filename'], entry)
    binding = {'fit_done_sha256': sha256(a.fit / 'DONE.json'),
        'operator_bank_sha256': sha256(a.fit / 'operator_bank.pt'), 'dose': bank['dose'],
        'reference_report_sha256': reference_hash, 'native_engineering_report_sha256': native_hash,
        'assets_report_sha256': assets_hash, 'manifest_sha256': sha256(a.manifest)}
    return contract, bank, historical, binding


def protocol_for(contract, binding):
    if len(contract['episodes']) != 64:
        raise ValueError('DROID has64 total sampled segments per condition')
    return {'method': METHOD, 'source_sha256': source_hash(), 'planning': contract,
        'binding': binding, 'arms': list(ARMS), 'engineering_arms': list(ENGINEERING),
        'episodes_per_condition': 64, 'total_panel_evaluations': 192, 'analysis': ANALYSIS,
        'precision': 'float32_no_tf32', 'engineering_seed': SMOKE_SEED,
        'native_reference_policy': 'receiving_paired_native_same_physical_device',
        'historical_native_reuse': False, 'protected_access': False,
        'fresh_confirmation': False, 'robot_executions': 0,
        'endpoint': 'official_recorded_plan_action_score_not_physical_task_success',
        'training_seed_count': 1, 'checkpoint_count': 1,
        'legacy_solver': False, 'candidate_admission_requires_offline_improvement': False}


def freeze(a):
    contract, _, _, binding = inputs(a)
    a.freeze.mkdir(parents=True, exist_ok=False)
    write_json(a.freeze / 'protocol.json', protocol_for(contract, binding))
    write_json(a.freeze / 'FROZEN.json', {'protocol_sha256': sha256(a.freeze / 'protocol.json')})


def checked(a):
    contract, bank, native, binding = inputs(a)
    protocol = read(a.freeze / 'protocol.json')
    if (read(a.freeze / 'FROZEN.json')['protocol_sha256'] != sha256(a.freeze / 'protocol.json')
            or protocol != protocol_for(contract, binding)):
        raise ValueError('Source, method, input, analysis or sample registry changed')
    return protocol, bank, native


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return {k: plain(v) for k, v in value.items()} if isinstance(value, dict) else value


def validate_calls(calls, arm, dose):
    if [(c['horizon'], c['candidates']) for c in calls] != [(3, 300), (3, 1)] * 15:
        raise ValueError('Changed native DROID CEM work')
    for call in calls:
        record, count = call['record'], call['candidates']
        if (record['backend_calls'] != 1 or record['response_probe_rollouts'] != 0
                or record['native_shadow_rollouts'] != 0):
            raise ValueError('Unexpected online model work')
        if arm in ('native', 'zero_dose', 'native_repeat'):
            if set(record) != {'backend_calls', 'response_probe_rollouts', 'native_shadow_rollouts', 'horizon'}:
                raise ValueError('Native/zero arm cannot deliver an edit')
            continue
        if arm not in ARMS[1:]:
            raise ValueError('Unregistered edit')
        active = np.asarray(record['active'])
        requested = np.asarray(record['requested_l2'])
        realized = np.asarray(record['realized_l2'])
        coefficients = np.asarray(record['coefficients'])
        if (active.shape != (count,) or active.dtype != np.bool_ or requested.shape != (count,)
                or realized.shape != (count,) or coefficients.shape != (count, 4)
                or not np.isfinite(requested).all() or not np.isfinite(realized).all()
                or not np.isfinite(coefficients).all()
                or not np.allclose(requested, active * dose, rtol=5e-4, atol=1e-7)
                or not np.allclose(realized, requested, rtol=5e-4, atol=1e-7)):
            raise ValueError('Dose/control/coefficient contract differs')


class StaticReference:
    """Independent engineering-only fixed field application at B6/H3."""
    def __init__(self, predictor, field):
        self.predictor, self.field, self.step, self.handles = predictor, field, 0, []
    def __enter__(self):
        def count(module, args): self.step += 1
        def apply(module, args, output):
            if self.step == 3:
                result = output.clone()
                result[:, -256:] += self.field.to(output.dtype)
                return result
        self.handles = [self.predictor.register_forward_pre_hook(count),
            self.predictor.predictor_blocks[6].register_forward_hook(apply)]
        return self
    def __exit__(self, kind, *_):
        for handle in self.handles: handle.remove()
        if kind is None and self.step != 3:
            raise ValueError('Independent reference horizon differs')


def reference_delta(field, bank, arm):
    scores = ((field.float().reshape(len(field), -1) - bank['mean']) @ bank['projection'].T) / bank['scale']
    phi = torch.cat([torch.ones_like(scores[:, :1]), scores], 1)
    spec = bank['operators'][arm]
    coefficients = phi @ spec['map'].T
    delta = (coefficients @ spec['basis'].reshape(4, -1)).reshape_as(field)
    norm = delta.flatten(1).norm(dim=1)
    return delta * torch.where(norm > 1e-10, bank['dose'] / norm.clamp_min(1e-10), 0.)[:, None, None]


def episode(cfg, model, dset, prep, bank, arm, output, row, agent, engineering):
    adapter = Intervention(model, bank, 'native' if arm == 'native_repeat' else arm)
    original = agent.planner.unroll
    seen, calls = set(), []
    def unroll(context, act_suffix=None, **kwargs):
        count = act_suffix.shape[1]
        reference = None
        if engineering and count not in seen:
            if arm in ARMS[1:]:
                with FieldHook(model.model.predictor) as capture:
                    model.unroll(context, act_suffix=act_suffix)
                delta = reference_delta(capture.field, adapter.bank, arm)
                with StaticReference(model.model.predictor, delta):
                    reference = model.unroll(context, act_suffix=act_suffix)
            else:
                reference = model.unroll(context, act_suffix=act_suffix)
            seen.add(count)
        result = adapter(context, act_suffix=act_suffix, **kwargs)
        if reference is not None and not torch.equal(reference, result):
            raise ValueError('Independent static-field reference differs from thin-map forecast')
        calls.append({'horizon': len(act_suffix), 'candidates': count, 'record': plain(adapter.last_record)})
        return result
    agent.planner.unroll = unroll
    try:
        result = full_episode(cfg, model, dset, prep, output, row['episode'],
            seed=row['environment_seed'], agent=agent,
            role='engineering_not_efficacy' if engineering else METHOD)
    finally:
        agent.planner.unroll = original
    validate_calls(calls, arm, bank['dose'])
    if engineering and seen != {1, 300}:
        raise ValueError('Missing full candidate-count reference checks')
    return result, calls


def load_shard(root, protocol, freeze_hash, uuid, arm, rank, receiving_hash):
    report, digest = verified_report(root)
    launch = read(root / 'protocol.json')
    expected = assigned_rows(protocol['planning']['episodes'], [rank])
    names = {f'{kind}-{r["episode"]:03d}.json' for r in expected for kind in ('episode', 'trace')}
    if (report['status'] != METHOD + '_shard_complete' or (root / 'FAILED.json').exists()
            or report['protocol_sha256'] != sha256(root / 'protocol.json')
            or launch['freeze_sha256'] != freeze_hash or launch['source_sha256'] != protocol['source_sha256']
            or launch['role'] != 'scientific_development' or launch['device_uuid'] != uuid
            or launch['engineering_report_sha256'] != receiving_hash
            or launch['arm'] != arm or launch['logical_ranks'] != [rank]
            or report['episodes'] != 8 or report['device_uuid'] != uuid
            or report['parameters_unchanged'] is not True or report['robot_executions'] != 0
            or report['fresh_confirmation'] is not False or set(report['files_sha256']) != names):
        raise ValueError('Missing or unbound complete paired DROID stream')
    for name, want in report['files_sha256'].items():
        if sha256(root / name) != want:
            raise ValueError('DROID raw evidence changed')
    rows = [read(root / f'episode-{r["episode"]:03d}.json') for r in expected]
    validate_records(rows, expected)
    for row in rows:
        if row['arm'] != arm or row['device_uuid'] != uuid:
            raise ValueError('Mislabeled DROID condition/device')
        validate_calls(read(root / f'trace-{row["episode"]:03d}.json'), arm, protocol['binding']['dose'])
    return rows, digest


def engineering_receipt(root, freeze_hash, uuid, dose):
    report, digest = verified_report(root)
    launch = read(root / 'protocol.json')
    names = {f'{kind}-{i:03d}.json' for i in range(5) for kind in ('episode', 'trace')}
    if (report['status'] != METHOD + '_engineering_passed' or (root / 'FAILED.json').exists()
            or report['protocol_sha256'] != sha256(root / 'protocol.json')
            or launch['freeze_sha256'] != freeze_hash or launch['device_uuid'] != uuid
            or launch['source_sha256'] != source_hash() or launch['role'] != 'engineering'
            or report['device_uuid'] != uuid or report['episodes'] != 5
            or not report['parameters_unchanged'] or set(report['files_sha256']) != names):
        raise ValueError('Missing same-device full receiving gate')
    for name, want in report['files_sha256'].items():
        if sha256(root / name) != want:
            raise ValueError('Receiving raw evidence changed')
    rows = [read(root / f'episode-{i:03d}.json') for i in range(5)]
    validate_records(rows, [{'episode': i, 'environment_seed': SMOKE_SEED} for i in range(5)])
    for i, (row, arm) in enumerate(zip(rows, ENGINEERING)):
        if row['arm'] != arm or row['environment_seed'] != SMOKE_SEED or row['device_uuid'] != uuid:
            raise ValueError('Changed receiving sequence or device')
        validate_calls(read(root / f'trace-{i:03d}.json'), arm, dose)
        if i in (1, 2): assert_same(rows[0]['result'], row['result'])
        for key in ('initial_sha256', 'goal_sha256', 'dataset_sample', 'recorded_actions'):
            assert_same(rows[0]['result'][key], row['result'][key])
    return digest


@torch.no_grad()
def execute(a):
    protocol, bank, historical = checked(a)
    freeze_hash = sha256(a.freeze / 'protocol.json')
    engineering = a.command == 'engineer'
    uuid, receiving = device_uuid(), None
    expected = [] if engineering else assigned_rows(protocol['planning']['episodes'], a.logical_ranks)
    native = {r['episode']: r for r in historical}
    if not engineering:
        if len(a.logical_ranks) != 1:
            raise ValueError('Launch one intact logical DROID stream per process')
        receiving = engineering_receipt(a.engineering, freeze_hash, uuid, bank['dose'])
        if a.arm != 'native':
            if len(a.logical_ranks) != 1 or a.native is None:
                raise ValueError('Require one complete same-device paired native stream')
            found, _ = load_shard(a.native, protocol, freeze_hash, uuid, 'native', a.logical_ranks[0], receiving)
            native = {r['episode']: r for r in found}
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / 'protocol.json', {'freeze_sha256': freeze_hash, 'source_sha256': protocol['source_sha256'],
        'role': 'engineering' if engineering else 'scientific_development', 'arm': a.arm,
        'logical_ranks': a.logical_ranks, 'device_uuid': uuid, 'engineering_report_sha256': receiving})
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        contract = protocol['planning']
        _, prep = make_dataset(a.assets, contract)
        model, provenance = load_model(a.vendor, a.assets, a.encoder_source, a.encoder_root, contract, prep)
        versions = [(name, p._version) for name, p in model.named_parameters()]
        from omegaconf import OmegaConf
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        jobs = [(arm, None) for arm in ENGINEERING] if engineering else [(a.arm, r) for r in sorted(a.logical_ranks)]
        records, files = [], []
        for i, (arm, rank) in enumerate(jobs):
            seed = SMOKE_SEED if engineering else 1
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            dset, prep = make_dataset(a.assets, contract)
            cfg = OmegaConf.create(copy.deepcopy(contract['config']))
            rows = [{'episode': i, 'environment_seed': SMOKE_SEED}] if engineering else [r for r in expected if r['logical_rank'] == rank]
            cfg.local_seed = SMOKE_SEED if engineering else rows[0]['local_seed']
            if engineering: cfg.meta.seed = SMOKE_SEED
            agent = GC_Agent(cfg, model, dset=dset, preprocessor=prep)
            for row in rows:
                before = time.monotonic()
                result, calls = episode(cfg, model, dset, prep, bank, arm, a.output, row, agent, engineering)
                record = {**row, 'arm': arm, 'device_uuid': uuid, 'seconds': time.monotonic() - before, 'result': result}
                if engineering:
                    old = read(a.native_engineering / 'repetition-0.json')['result']
                    for key in ('initial_sha256', 'goal_sha256', 'dataset_sample', 'recorded_actions'):
                        assert_same(old[key], result[key])
                    if i in (1, 2): assert_same(records[0]['result'], result)
                else:
                    paired_inputs(native[row['episode']], record)
                records.append(record)
                for name, value in ((f'episode-{row["episode"]:03d}.json', record), (f'trace-{row["episode"]:03d}.json', calls)):
                    write_json(a.output / name, value); files.append(name)
                print(json.dumps({'arm': arm, 'completed': len(records), 'target': 5 if engineering else len(expected),
                    'engineering': engineering, 'seconds': time.monotonic() - started}), flush=True)
        if not engineering: validate_records(records, expected)
        if versions != [(name, p._version) for name, p in model.named_parameters()] or source_hash() != protocol['source_sha256']:
            raise ValueError('Frozen checkpoint/source changed')
        write_json(a.output / 'report.json', {'status': METHOD + ('_engineering_passed' if engineering else '_shard_complete'),
            'protocol_sha256': sha256(a.output / 'protocol.json'), 'device_uuid': uuid, 'episodes': len(records),
            'files_sha256': {name: sha256(a.output / name) for name in files}, 'model_provenance': provenance,
            'parameters_unchanged': True, 'seconds': time.monotonic() - started,
            'fresh_confirmation': False, 'robot_executions': 0})
        write_json(a.output / 'DONE.json', {'report_sha256': sha256(a.output / 'report.json')})
        if engineering: engineering_receipt(a.output, freeze_hash, uuid, bank['dose'])
    except Exception as error:
        write_json(a.output / 'FAILED.json', {'error': str(error), 'partial_not_eligible': True})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'engineer', 'run'))
    for name in ('vendor', 'fit', 'assets', 'manifest', 'reference', 'native-engineering', 'freeze', 'encoder-source', 'encoder-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('engineering', 'native'): p.add_argument('--' + name, type=Path)
    p.add_argument('--arm', choices=ARMS, default='native')
    p.add_argument('--logical-ranks', nargs='+', type=int, default=[0])
    args = p.parse_args()
    freeze(args) if args.command == 'freeze' else execute(args)
