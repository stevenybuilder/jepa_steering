"""Recreate only missing navigation PCA/readout tensors on the original fit pool.

This is explicitly a new fitting artifact, never a byte-identical archive restore.
No geometry, spatial/layer search, validation predictions, or online response solve.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import AuthorBackend, encoded_batches, prefix_batches, open_normalized_dataset
from offline_study.fitting.fixed_response_fit import check_examples
from offline_study.tasks.navigation.navigation_cohort import verify_inputs
from offline_study.tasks.navigation.navigation_smoke import CHECKPOINTS
from offline_study.fitting.operator_fit import _model_versions
from offline_study.core.protocol import sha256, write_json
from offline_study.interventions.support_operator import SEED, principal_basis, fit_target, random_basis


class Capture:
    def __init__(self, predictor):
        self.predictor, self.horizon, self.handles, self.value = predictor, 0, [], None
    def __enter__(self):
        if len(self.predictor.predictor_blocks) != 6 or any(m._forward_hooks or m._forward_pre_hooks for m in self.predictor.modules()):
            raise ValueError('Require uninstrumented six-block predictor')
        def before(*_): self.horizon += 1
        def after(module, args, output):
            if self.horizon == 3:
                if self.value is not None or output.shape[-1] != 400 or output.shape[1] < 256:
                    raise ValueError('Unexpected B3/H3 native field')
                self.value = output[:, -256:].detach().float().clone()
        self.handles = [self.predictor.register_forward_pre_hook(before),
            self.predictor.predictor_blocks[3].register_forward_hook(after)]
        return self
    def __exit__(self, kind, exc, trace):
        for h in self.handles: h.remove()
        if kind is None and (self.horizon != 6 or self.value is None):
            raise ValueError('Incomplete six-step native capture')


def original_contract(original, cohort_path):
    done = json.loads((original / 'DONE.json').read_text())
    for name, key in [('protocol.json', 'protocol_sha256'), ('fit_receipt.json', 'fit_receipt_sha256')]:
        if sha256(original / name) != done[key]:
            raise ValueError('Archived original fitting receipt changed')
    protocol = json.loads((original / 'protocol.json').read_text())
    receipt = json.loads((original / 'fit_receipt.json').read_text())
    cohort = json.loads(cohort_path.read_text())
    if (cohort['task'] not in ('wall', 'pointmaze') or len(cohort['fit']) != 128
            or receipt['fit_lineage_groups'] != [r['lineage_group'] for r in cohort['fit']]
            or receipt['cohort_sha256'] != sha256(cohort_path)
            or receipt['checkpoint_sha256'] != CHECKPOINTS[cohort['task']]
            or protocol['category'] != 'operator_rank' or receipt['precision'] != 'bfloat16'):
        raise ValueError('Original fitting task/population/checkpoint differs')
    return cohort, protocol, done


@torch.no_grad()
def run(a):
    cohort, original, old_done = original_contract(a.original, a.cohort)
    verify_inputs(a.cohort, a.data_root)
    a.output.mkdir(parents=True, exist_ok=False)
    task, started = cohort['task'], time.monotonic()
    binding = {'role': 'limited_native_rank_tensor_reconstruction_not_original_bank_recovery',
        'task': task, 'checkpoint_sha256': CHECKPOINTS[task], 'cohort_sha256': sha256(a.cohort),
        'source_sha256': source_hash(), 'original_missing_bank_sha256': old_done['operator_bank_sha256'],
        'original_protocol_sha256': sha256(a.original / 'protocol.json'),
        'original_fit_receipt_sha256': sha256(a.original / 'fit_receipt.json'),
        'original_input_files_sha256': sha256(a.cohort.parent / 'input_files.json'),
        'fit_families': 128, 'fit_prefixes': 512, 'precision': 'bfloat16',
        'block': 3, 'horizon': 3, 'target_horizon': 6, 'basis_rank': 8, 'runtime_rank': 4,
        'random_seed': SEED + 1, 'readout_scores': 3, 'original_dose_retained': True,
        'validation_outcomes_accessed': False, 'max_seconds': 3600}
    protocol = {**binding, 'category': 'operator_rank', 'support_operator': original['support_operator']}
    write_json(a.output / 'protocol.json', protocol)
    write_json(a.output / 'fit_receipt.json', binding)
    try:
        backend = AuthorBackend(a.vendor, a.checkpoint, CHECKPOINTS[task], task, 'cuda:0', 'bfloat16')
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset(task, a.data_root, cohort['reference_config'], True)
        fields, errors, seen, parity = [], [], [], None
        for clips, encoded in encoded_batches(backend, dataset, cohort['fit'], cohort['reference_config'], fitting=True):
            if parity is None:
                parity = backend.verify_reference(encoded)
                write_json(a.output / 'PARITY.json', {'actual_upstream': parity, 'fit_only': True})
            for metadata, context, actions, target in prefix_batches(backend, clips, encoded):
                if time.monotonic() - started > 3600:
                    raise TimeoutError('Bounded reconstruction incomplete')
                with Capture(backend.predictor) as capture:
                    prediction = backend.predict(context, actions)
                if not fields:
                    passive = backend.predict(context, actions)
                    if any(not torch.equal(prediction[k], passive[k]) for k in prediction):
                        raise ValueError('Passive native capture changed forecasts')
                fields.append(capture.value.cpu())
                errors.append((target['visual'][:, 6].float() - prediction['visual'][6].float()).flatten(1).cpu())
                seen.extend(metadata)
            print(json.dumps({'task': task, 'native_prefixes': len(seen), 'seconds': time.monotonic() - started}), flush=True)
        check_examples(seen, cohort['fit'], cohort['reference_config'])
        if len(seen) != 512:
            raise ValueError('Original512-prefix fitting coverage differs')
        field, error = torch.cat(fields), torch.cat(errors)
        torch.save({'B3_fields': field, 'errors': error, 'metadata': seen}, a.output / 'native_fit.pt')
        features, error = field.flatten(1).to('cuda:0'), error.to('cuda:0')
        _, basis = principal_basis(features, 8)
        rank = {'blocks': [3], 'positions': list(range(256)), 'basis': basis.reshape(8, 1, 256, 400)}
        support = {'target': fit_target(features, error), 'bases': {'rank': rank, 'rank_random': random_basis(rank, SEED + 1)}}
        from offline_study.fitting.support_fit import cpu_tree
        bank = {'protocol_sha256': sha256(a.output / 'protocol.json'), 'support_bank': cpu_tree(support), 'reconstruction': binding}
        torch.save(bank, a.output / 'operator_bank.pt')
        recalculated = .005 * float((features - features.mean(0)).square().sum(1).mean().sqrt())
        if _model_versions(backend.model) != versions or source_hash() != binding['source_sha256']:
            raise ValueError('Source or model changed')
        verify_inputs(a.cohort, a.data_root)
        write_json(a.output / 'report.json', {**binding, 'seconds': time.monotonic() - started,
            'new_fit_only_dose_diagnostic': recalculated, 'delivered_l2_used': original['support_operator']['delivered_l2'],
            'validation_outcomes_accessed': False, 'byte_identical_original_bank_claimed': False})
        files = ('protocol.json', 'fit_receipt.json', 'operator_bank.pt', 'native_fit.pt', 'PARITY.json', 'report.json')
        write_json(a.output / 'RECONSTRUCTION.json', {'status': 'limited_native_rank_reconstruction_complete',
            'files': {name: sha256(a.output / name) for name in files}})
    except Exception as error:
        write_json(a.output / 'FAILED.json', {'error': str(error), 'partial_not_eligible': True})
        raise


def load(original, cohort_path):
    proof = json.loads((original / 'RECONSTRUCTION.json').read_text())
    required = {'protocol.json', 'fit_receipt.json', 'operator_bank.pt', 'native_fit.pt', 'PARITY.json', 'report.json'}
    if (proof.get('status') != 'limited_native_rank_reconstruction_complete' or
            (original / 'FAILED.json').exists() or set(proof['files']) != required):
        raise ValueError('Incomplete limited reconstruction')
    for name, digest in proof['files'].items():
        if sha256(original / name) != digest:
            raise ValueError('Reconstructed fitting bytes changed')
    protocol = json.loads((original / 'protocol.json').read_text())
    cohort = json.loads(cohort_path.read_text())
    if (protocol['cohort_sha256'] != sha256(cohort_path) or protocol['task'] not in CHECKPOINTS
            or protocol['task'] != cohort['task'] or protocol['checkpoint_sha256'] != CHECKPOINTS[cohort['task']]
            or protocol['validation_outcomes_accessed'] is not False or protocol['fit_prefixes'] != 512):
        raise ValueError('Wrong reconstructed source population/task')
    bank = torch.load(original / 'operator_bank.pt', weights_only=True, map_location='cpu')
    if bank['protocol_sha256'] != sha256(original / 'protocol.json'):
        raise ValueError('New bank does not bind its new reconstruction protocol')
    return protocol, bank


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'cohort', 'data-root', 'vendor', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
