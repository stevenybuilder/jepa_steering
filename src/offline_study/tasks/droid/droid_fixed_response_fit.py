"""Bounded DROID-only fit of the registered thin-map extension, not a search."""
from offline_study._paths import source_path
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import yaml

from offline_study.tasks.droid.droid_contract import prepare, TRAIN_CONFIG
from offline_study.tasks.droid.droid_fit_audit import make_raw_dataset, reset
from offline_study.tasks.droid.droid_fixed_response import ARMS, METHOD, FieldHook, Intervention, make_bank, load_bank
from offline_study.tasks.droid.droid_native import array_hash, load_model, verified_report
from offline_study.core.protocol import sha256, write_json
from offline_study.interventions.support_operator import SEED, fit_target, principal_basis, random_basis
from offline_study.models.vendor import use_vendor

SELECTION_SEED = 2026090802


def selection(recordings):
    if len(recordings) != 128 or len({r['directory'] for r in recordings}) != 128:
        raise ValueError('Require the original128 distinct fitting recordings')
    order = sorted(range(128), key=lambda i: hashlib.sha256(
        (str(SELECTION_SEED) + ':' + recordings[i]['directory']).encode()).hexdigest())
    return {'calibration': order[:24], 'response_audit': order[24:32]}


def validate_prefix(actual, original, directory, raw_root):
    for key in ('indices', 'state_sha256', 'visual_sha256', 'actions_sha256'):
        if actual[key] != original[key]:
            raise ValueError('Original native prefix changed: ' + key)
    # loadvideo_decord is passed the trajectory DIRECTORY, not a video filename.
    if actual['path'] != str(raw_root / directory) or not original['path'].endswith('/' + directory):
        raise ValueError('Relocation selected a different native trajectory directory')


def input_contract(a):
    audit, audit_hash = verified_report(a.audit)
    audit_protocol = json.loads((a.audit / 'protocol.json').read_text())
    inputs, input_hash = verified_report(a.inputs)
    if (audit['status'] != 'native_droid_fit_inputs_parity_and_exact_separation_passed'
            or audit['recordings'] != 128 or audit['prefixes'] != 512
            or audit['native_pixels_actions_states_rng_exact'] is not True
            or any(audit['exact_evaluation_state_overlap'].values()) or any(audit['duplicate_fit_groups'].values())
            or audit_protocol['source_sha256'] != sha256(source_path('droid_fit_audit.py'))
            or audit_protocol['inputs_report_sha256'] != input_hash
            or inputs['files_sha256'] != sha256(a.inputs / 'FILES.json')):
        raise ValueError('Original fitting audit/input provenance mismatch')
    for name, digest in audit['files_sha256'].items():
        if Path(name).name != name or sha256(a.audit / name) != digest:
            raise ValueError('Original prefix/audit member changed')
    files = json.loads((a.inputs / 'FILES.json').read_text())
    for name, spec in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe raw source name')
        path = a.raw / name
        if path.is_symlink() or path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError('Raw fitting bytes differ')
    source = json.loads((a.inputs / 'protocol.json').read_text())
    selected = selection(source['recordings'])
    prefixes = json.loads((a.audit / 'prefixes.json').read_text())
    if [(p['recording_index'], p['prefix']) for p in prefixes] != [(i, j) for i in range(128) for j in range(4)]:
        raise ValueError('Original prefix order/count differs')
    return source, prefixes, selected, {'audit_report_sha256': audit_hash,
        'input_report_sha256': input_hash, 'raw_files_sha256': sha256(a.inputs / 'FILES.json')}


def collect_responses(model, context, actions, bases, radius):
    if not 0 < radius < float('inf'):
        raise ValueError('Invalid response radius')
    probes = [(key, index, sign) for key in ('rank', 'rank_random') for index in range(4) for sign in (-1, 1)]
    results = {}
    for start in range(0, 16, 4):
        chunk = probes[start:start + 4]
        fields = torch.stack([bases[key][index] * radius * sign for key, index, sign in chunk])
        with FieldHook(model.model.predictor, lambda field: field + fields.to(field.dtype)):
            prediction = model.unroll(context, act_suffix=actions.repeat_interleave(4, dim=1))
        for i, probe in enumerate(chunk):
            results[probe] = prediction[3, i].float().flatten().clone()
    responses = {key: torch.stack([(results[key, i, 1] - results[key, i, -1]) / (2 * radius)
        for i in range(4)]) for key in ('rank', 'rank_random')}
    if any(not torch.isfinite(value).all() for value in responses.values()):
        raise ValueError('Nonfinite response; cannot discard a fitting example')
    return responses


@torch.no_grad()
def run(a):
    use_vendor(a.vendor)
    source, prefixes, selected, input_binding = input_contract(a)
    contract = prepare(a.vendor, json.loads(a.manifest.read_text()))
    a.output.mkdir(parents=True, exist_ok=False)
    binding = {'method': METHOD, 'task': 'droid', 'checkpoint_sha256': contract['checkpoint_sha256'],
        **input_binding, 'fit_precision': 'bfloat16', 'block': 6, 'horizon': 3,
        'features': 1024, 'patches': 256, 'context_capacity': 2,
        'source_files_sha256': {name: sha256(source_path(name)) for name in
            ('droid_fixed_response.py', 'droid_fixed_response_fit.py', 'droid_fit_audit.py',
             'droid_native.py', 'support_operator.py', 'fixed_response.py')},
        'native_training_config_sha256': sha256(a.vendor / TRAIN_CONFIG)}
    configuration = {'binding': binding, 'selection': selected, 'selection_seed': SELECTION_SEED,
        'random_basis_seed': SEED + 1, 'fit_recordings': 128, 'fit_prefixes': 512,
        'response_calibration_recordings': 24, 'response_audit_recordings': 8,
        'probes_per_response_example': 16, 'probe_batch_size': 4,
        'dose_rule': '0.005_centered_B6_H3_field_RMS_L2', 'response_radius_equals_dose': True,
        'readout': 'native_H3_visual_residual_PCA3_OLS_intercept',
        'response_inverse': 'inverse_of_mean_response_0.01_gram_mean_diagonal_damping',
        'online_probes': 0, 'online_native_shadows': 0, 'runtime_arms': list(ARMS),
        'response_audit_is_independent_confirmation': False, 'max_fit_seconds': 3600,
        'behavioral_launch_ready': False, 'author_proposed_steering_method': False}
    write_json(a.output / 'contract.json', configuration)
    started = time.monotonic()
    def check_time():
        if time.monotonic() - started > 3600:
            raise TimeoutError('Fitting time cap reached; retain partial, do not promote')
    try:
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        # Relocation changes paths, not stimuli or the existing audit receipt.
        paths = [str(a.raw / r['directory']) for r in source['recordings']]
        if any(' ' in path or '\n' in path for path in paths):
            raise ValueError('Native CSV cannot represent this relocation safely')
        csv = a.output / 'relocated_paths.csv'
        csv.write_text('\n'.join(paths) + '\n')
        reset()
        dataset = make_raw_dataset(csv, a.vendor, True)
        from app.plan_common.datasets.preprocessor import Preprocessor
        from app.plan_common.datasets.transforms import make_inverse_transforms
        train = yaml.safe_load((a.vendor / TRAIN_CONFIG).read_text())
        preprocessor = Preprocessor(**{name: getattr(dataset, name) for name in
            ('action_mean', 'action_std', 'state_mean', 'state_std', 'proprio_mean', 'proprio_std')},
            transform=dataset.transform, inverse_transform=make_inverse_transforms(img_size=256, **train['data_aug']))
        model, provenance = load_model(a.vendor, a.assets, a.encoder_source, a.encoder_root, contract, preprocessor)
        versions = [(name, p._version) for name, p in model.named_parameters()]
        reset()
        dataset = make_raw_dataset(csv, a.vendor, True)
        fields, errors, cached, seen = [], [], {}, []
        chosen = set(selected['calibration'] + selected['response_audit'])
        for item in prefixes:
            check_time()
            obs, actions, _, _ = dataset[item['recording_index']]
            actual = {**dataset.last_sample, 'visual_sha256': array_hash(obs['visual'].numpy()),
                'actions_sha256': array_hash(actions)}
            relative = source['recordings'][item['recording_index']]['directory']
            validate_prefix(actual, item, relative, a.raw)
            acts = torch.as_tensor(actions[:3], device='cuda:0', dtype=torch.float32)[:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                context = model.model.encode_obs({'visual': obs['visual'][None, :1].to('cuda:0')})['visual']
                target = model.model.encode_obs({'visual': obs['visual'][None, 3:4].to('cuda:0')})['visual'][:, 0]
                with FieldHook(model.model.predictor) as capture:
                    prediction = model.unroll(context, act_suffix=acts)
                if not seen:
                    native = model.unroll(context, act_suffix=acts)
                    if not torch.equal(native, prediction):
                        raise ValueError('Passive native capture changes forecast')
                    write_json(a.output / 'PARITY.json', {'passive_capture_exact': True,
                        'native_single_frame_context_encoding': True, 'future_target_not_in_context': True})
            fields.append(capture.field.float().cpu())
            errors.append((target.float() - prediction[3].float()).flatten(1).cpu())
            if item['recording_index'] in chosen:
                cached[(item['recording_index'], item['prefix'])] = (context.cpu(), acts.cpu(), target.cpu())
            seen.append(item)
            if len(seen) % 16 == 0:
                print(json.dumps({'stage': 'native_fit_capture', 'prefixes': len(seen),
                    'seconds': time.monotonic() - started}), flush=True)
        field_tensor, error_tensor = torch.cat(fields).to('cuda:0'), torch.cat(errors).to('cuda:0')
        torch.save({'field': field_tensor.cpu(), 'errors': error_tensor.cpu()}, a.output / 'native_captures.pt')
        features = field_tensor.flatten(1)
        _, basis = principal_basis(features, 8)
        bases = {'rank': basis.reshape(8, 256, 1024)}
        bases['rank_random'] = random_basis({'basis': bases['rank']}, SEED + 1)['basis']
        target_fit = fit_target(features, error_tensor)
        dose = .005 * float((features - features.mean(0)).square().sum(1).mean().sqrt())
        means, audit_rows = {}, []
        for role, indices in selected.items():
            count = 0
            for index in indices:
                for prefix in range(4):
                    check_time()
                    context, acts, target = (v.to('cuda:0') for v in cached[index, prefix])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        responses = collect_responses(model, context, acts, bases, dose)
                        if role == 'calibration':
                            for key, value in responses.items():
                                means[key] = means.get(key, 0) + value.double()
                        else:
                            native = adapters['native'](context, act_suffix=acts)
                            for arm, adapter in adapters.items():
                                prediction = native if arm == 'native' else adapter(context, act_suffix=acts)
                                if arm == 'zero_dose' and not torch.equal(prediction, native):
                                    raise ValueError('Zero-dose identity failed')
                                row = {'recording_index': index, 'prefix': prefix, 'arm': arm,
                                    'visual_mse_h3': float((prediction[3].float() - target.float()).square().mean())}
                                if arm in ARMS[2:]:
                                    key = 'rank_random' if arm.startswith('matched_random') else 'rank'
                                    row.update(requested_l2=float(adapter.last_record['requested_l2'][0]),
                                        realized_l2=float(adapter.last_record['realized_l2'][0]),
                                        relative_response_deviation=float((responses[key].double() - means[key]).norm() / means[key].norm().clamp_min(1e-18)))
                                audit_rows.append(row)
                    count += 1
                print(json.dumps({'stage': role, 'prefixes': count, 'seconds': time.monotonic() - started}), flush=True)
            if role == 'calibration':
                means = {key: value / count for key, value in means.items()}
                bank = make_bank(target_fit, bases, means, dose, binding)
                torch.save(bank, a.output / 'operator_bank.pt')
                torch.save({key: value.cpu() for key, value in means.items()}, a.output / 'mean_responses.pt')
                adapters = {arm: Intervention(model, bank, arm) for arm in ARMS}
        if versions != [(name, p._version) for name, p in model.named_parameters()]:
            raise ValueError('Checkpoint changed during fitting')
        for name, digest in binding['source_files_sha256'].items():
            if sha256(source_path(name)) != digest:
                raise ValueError('Frozen fitting implementation changed')
        write_json(a.output / 'prefixes.json', seen)
        write_json(a.output / 'audit_metrics.json', audit_rows)
        write_json(a.output / 'report.json', {'status': 'droid_fixed_fit_complete_not_behavioral_clearance',
            'binding': binding, 'model_provenance': provenance, 'fit_recordings': 128, 'fit_prefixes': 512,
            'calibration_prefixes': 96, 'response_audit_prefixes': 32, 'dose': dose,
            'zero_dose_identity': True, 'evaluation_outcomes_accessed': False,
            'parameters_unchanged': True, 'runtime_online_probes': 0, 'runtime_native_shadows': 0,
            'behavioral_launch_ready': False, 'fresh_confirmation': False,
            'seconds': time.monotonic() - started})
        files = ('contract.json', 'operator_bank.pt', 'mean_responses.pt', 'audit_metrics.json',
            'report.json', 'prefixes.json', 'native_captures.pt', 'PARITY.json')
        write_json(a.output / 'DONE.json', {'status': 'droid_fixed_fit_complete_not_behavioral_clearance',
            'files_sha256': {name: sha256(a.output / name) for name in files}})
        load_bank(a.output, contract['checkpoint_sha256'])
    except Exception as error:
        write_json(a.output / 'FAILED.json', {'error': str(error), 'partial_not_eligible': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('vendor', 'assets', 'manifest', 'encoder-source', 'encoder-root', 'inputs', 'raw', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
