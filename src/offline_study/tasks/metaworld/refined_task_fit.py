"""Same fixed-response fit for the three remaining six-block task checkpoints.

Imports the proven arithmetic, but never the old expensive runtime solver.
Requires separate task/source/cohort/config bindings; MetaWorld banks cannot pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import AuthorBackend, encoded_batches, open_normalized_dataset, prefix_batches, validate_cohort
from offline_study.interventions.fixed_response import ARMS, METHOD, FixedResponseIntervention, make_bank, load_fitted_bank
from offline_study.fitting.fixed_response_fit import collect_responses, check_examples
from offline_study.fitting.operator_fit import _model_versions
from offline_study.planning.planning_native_smoke import CHECKPOINTS as PRIMARY_CHECKPOINTS
from offline_study.tasks.navigation.navigation_smoke import CHECKPOINTS as NAVIGATION_CHECKPOINTS
from offline_study.core.protocol import sha256, write_json

TASKS = ('pusht', 'pointmaze', 'wall')
CHECKPOINTS = {**PRIMARY_CHECKPOINTS, **NAVIGATION_CHECKPOINTS}
SELECTION_SEED = 2026090802


def selection(cohort):
    validate_cohort(cohort)
    if cohort['task'] not in TASKS or cohort['dataset'] != cohort['task']:
        raise ValueError('Require an original task-specific six-block fitting cohort')
    rows = sorted(cohort['fit'], key=lambda row: hashlib.sha256(
        (str(SELECTION_SEED) + ':' + row['lineage_group']).encode()).hexdigest())
    return {'calibration': rows[:24], 'response_audit': rows[24:32]}


def source_inputs(args):
    cohort = json.loads(args.cohort.read_text())
    selected = selection(cohort)
    task = cohort['task']
    if (args.fit / 'RECONSTRUCTION.json').exists():
        from offline_study.tasks.navigation.navigation_rank_reconstruction import load
        protocol, source = load(args.fit, args.cohort)
        return cohort, selected, protocol, source
    receipt, protocol = verify_fit(args.fit, sha256(args.cohort), CHECKPOINTS[task], 'bfloat16')
    if (receipt['task'] != task or receipt['fit_lineage_groups'] != [r['lineage_group'] for r in cohort['fit']]
            or protocol['category'] != 'operator_rank'):
        raise ValueError('Source task/population/category differs')
    source = torch.load(args.fit / 'operator_bank.pt', weights_only=True, map_location='cpu')
    if source['protocol_sha256'] != sha256(args.fit / 'protocol.json'):
        raise ValueError('Source bank does not bind the frozen protocol')
    fitted = source['support_bank']
    if fitted['target']['mean'].shape != (102400,):
        raise ValueError('Only six-block400-feature sources are covered by this adapter')
    for key in ('rank', 'rank_random'):
        spec = fitted['bases'][key]
        if spec['blocks'] != [3] or spec['positions'] != list(range(256)) or spec['basis'].shape != (8, 1, 256, 400):
            raise ValueError('Source directions do not match the unchanged B3/H3 contract')
    return cohort, selected, protocol, source


@torch.no_grad()
def run(args):
    cohort, selected, protocol, source = source_inputs(args)
    from offline_study.tasks.metaworld.refined_fit_inputs import verify as verify_inputs
    input_binding = verify_inputs(args.cohort, args.data_root)
    task = cohort['task']
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    binding = {'method': METHOD, 'task': task, 'checkpoint_sha256': CHECKPOINTS[task],
        'cohort_sha256': sha256(args.cohort), 'source_bank_sha256': sha256(args.fit / 'operator_bank.pt'),
        'source_protocol_sha256': sha256(args.fit / 'protocol.json'),
        'source_fit_receipt_sha256': sha256(args.fit / 'fit_receipt.json'),
        'source_sha256': source_hash(), 'fit_precision': 'bfloat16',
        'source_tensor_reconstruction': source.get('reconstruction'),
        'raw_input_binding': input_binding,
        'extension': 'remaining_six_block_task_fits_20260911_v1'}
    config = {'response_calibration_families': 24, 'response_audit_families': 8,
        'examples_per_family': 4, 'selection_seed': SELECTION_SEED, 'probe_chunk_size': 4,
        'runtime_arms': list(ARMS), 'max_fit_seconds_per_task': 3600,
        'development_or_protected_outcomes_accessed': False}
    write_json(args.output / 'contract.json', {**config, 'binding': binding, 'selection': selected,
        'response_audit_is_independent_of_inherited_basis_fit': False})
    write_json(args.output / 'cohort.json', cohort)
    try:
        backend = AuthorBackend(args.vendor, args.checkpoint, CHECKPOINTS[task], task, args.device, 'bfloat16')
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset(task, args.data_root, cohort['reference_config'], True)
        total, metadata, audit_rows, parity = {}, {}, [], None
        radius = protocol['support_operator']['response_radius']
        for role, rows in selected.items():
            seen = []
            for clips, encoded in encoded_batches(backend, dataset, rows, cohort['reference_config'], fitting=True):
                if parity is None:
                    parity = backend.verify_reference(encoded)
                    write_json(args.output / 'PARITY.json', {'fit_only': True, 'checks': parity})
                for meta, context, actions, target in prefix_batches(backend, clips, encoded):
                    if time.monotonic() - started > 3600:
                        raise TimeoutError('Bounded fit timed out; no partial bank promoted')
                    responses = collect_responses(backend, context, actions, source['support_bank'], radius, 4)
                    seen.extend(meta)
                    if role == 'calibration':
                        for key, response in responses.items():
                            total[key] = total.get(key, 0) + response.double().sum(0)
                    else:
                        native = adapters['native'](context, actions)
                        for arm, adapter in adapters.items():
                            prediction = native if arm == 'native' else adapter(context, actions)
                            if arm == 'zero_dose' and any(not torch.equal(native[k], prediction[k]) for k in native):
                                raise ValueError('Zero-dose identity failed')
                            metrics = backend.metrics(prediction, target)
                            for i, item in enumerate(meta):
                                row = {**item, 'arm': arm, 'metrics': metrics[i]}
                                if arm in ARMS[2:]:
                                    key = 'rank_random' if arm.startswith('matched_random') else 'rank'
                                    row.update(relative_response_deviation=float((responses[key][i].double() - means[key]).norm() / means[key].norm().clamp_min(1e-18)),
                                        requested_l2=float(adapter.last_record['requested_l2'][i]),
                                        realized_l2=float(adapter.last_record['realized_l2'][i]))
                                audit_rows.append(row)
                print(json.dumps({'task': task, 'stage': role, 'examples': len(seen),
                    'seconds': time.monotonic() - started}), flush=True)
            check_examples(seen, rows, cohort['reference_config'])
            metadata[role] = seen
            if role == 'calibration':
                means = {k: value / len(seen) for k, value in total.items()}
                bank = make_bank(source, means, protocol['support_operator']['delivered_l2'], binding)
                torch.save(bank, args.output / 'operator_bank.pt')
                torch.save({k: value.cpu() for k, value in means.items()}, args.output / 'mean_responses.pt')
                adapters = {arm: FixedResponseIntervention(backend, bank, arm) for arm in ARMS}
        if (_model_versions(backend.model) != versions or source_hash() != binding['source_sha256']
                or verify_inputs(args.cohort, args.data_root) != input_binding):
            raise ValueError('Frozen source/model changed during fit')
        write_json(args.output / 'audit_metrics.json', audit_rows)
        write_json(args.output / 'report.json', {'status': 'fit_only_calibration_and_response_audit_complete',
            'binding': binding, 'calibration_families': 24, 'response_audit_families': 8,
            'calibration_examples': len(metadata['calibration']), 'response_audit_examples': len(metadata['response_audit']),
            'source_basis_readout_fit_families': 128, 'response_audit_is_untouched_confirmation': False,
            'development_or_protected_outcomes_accessed': False, 'zero_dose_identity': True,
            'runtime_probe_forecasts': 0, 'runtime_native_shadow_forecasts': 0,
            'behavioral_launch_ready': False, 'fresh_confirmation': False,
            'wall_seconds': time.monotonic() - started})
        files = ('contract.json', 'cohort.json', 'operator_bank.pt', 'mean_responses.pt', 'PARITY.json', 'audit_metrics.json', 'report.json')
        write_json(args.output / 'DONE.json', {'status': 'fit_only_complete_not_behavioral_clearance',
            **{name: sha256(args.output / name) for name in files}})
        load_fitted_bank(args.output, task=task, checkpoint_sha256=CHECKPOINTS[task])
    except Exception as exc:
        write_json(args.output / 'FAILED.json', {'error': str(exc), 'partial_not_eligible': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cohort', 'fit', 'vendor', 'checkpoint', 'data-root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', required=True)
    run(parser.parse_args())
