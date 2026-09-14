"""CPU-only DROID interpreter/dependency and all fixed-prefix parity preflight."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

from offline_study.tasks.droid.droid_fit_audit import make_raw_dataset, reset
from offline_study.tasks.droid.droid_native import array_hash, verified_report
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


def run(args):
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise ValueError('Runtime preflight must be CUDA-hidden')
    use_vendor(args.vendor)
    audit, digest = verified_report(args.audit)
    if audit['recordings'] != 128 or audit['prefixes'] != 512:
        raise ValueError('Require the complete DROID native-input audit')
    for name, wanted in audit['files_sha256'].items():
        if sha256(args.audit / name) != wanted:
            raise ValueError('Audited member changed')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'role': 'cpu_runtime_and_native_prefix_parity',
        'audit_report_sha256': digest, 'source_sha256': sha256(Path(__file__)),
        'python': sys.version, 'torch': torch.__version__, 'interpreter': sys.executable})
    try:
        # The native DINO hub imports segmentation/training utilities even when
        # only its visual encoder is requested. Validate that complete import.
        import torchmetrics
        import decord
        sys.path.insert(0, str(args.encoder_source))
        spec = importlib.util.spec_from_file_location('checked_dino_hub', args.encoder_source / 'hubconf.py')
        hub = importlib.util.module_from_spec(spec); spec.loader.exec_module(hub)
        if not callable(hub.dinov3_vitl16):
            raise ValueError('Native DINOv3 encoder constructor unavailable')
        reset(); dataset = make_raw_dataset(args.audit / 'native_paths.csv', args.vendor, True)
        prefixes = json.loads((args.audit / 'prefixes.json').read_text())
        for i, item in enumerate(prefixes):
            obs, actions, _, _ = dataset[item['recording_index']]
            actual = {**dataset.last_sample, 'visual_sha256': array_hash(obs['visual'].numpy()),
                      'actions_sha256': array_hash(actions)}
            if actual != {key: item[key] for key in actual}:
                raise ValueError('DROID interpreter changed a frozen native prefix')
            if (i + 1) % 64 == 0:
                print(json.dumps({'prefixes_verified': i + 1, 'gpu_calls': 0}), flush=True)
        if torch.cuda.is_initialized():
            raise ValueError('CPU preflight initialized GPU')
        write_json(args.output / 'report.json', {'status': 'droid_runtime_and_all512_native_prefixes_verified',
            'protocol_sha256': sha256(args.output / 'protocol.json'), 'prefixes': len(prefixes),
            'native_hub_import_passed': True, 'torchmetrics': torchmetrics.__version__,
            'decord': decord.__version__, 'python': sys.version, 'torch': torch.__version__,
            'model_forward_calls': 0, 'gpu_initialized': False})
        write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})
    except Exception as exc:
        write_json(args.output / 'FAILED.json', {'error': str(exc), 'gpu_job_launched': False}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('vendor', 'audit', 'encoder-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
