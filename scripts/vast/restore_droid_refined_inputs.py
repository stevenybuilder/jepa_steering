"""Recover exactly the386 pinned objects for the original128 DROID fit recordings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from offline_study.droid_fit_download import download_object
from offline_study.droid_native import verified_report
from offline_study.protocol import sha256, write_json


def main(a):
    report, digest = verified_report(a.inputs)
    protocol = json.loads((a.inputs / 'protocol.json').read_text())
    if report['files_sha256'] != sha256(a.inputs / 'FILES.json') or report['recordings'] != 128:
        raise ValueError('Original fit input receipt changed')
    files = json.loads((a.inputs / 'FILES.json').read_text())
    objects = protocol['objects']
    if len({o['name'] for o in objects}) != len(objects) or set(files) != {o['name'] for o in objects}:
        raise ValueError('Original object inventory changed')
    a.output.mkdir(parents=True, exist_ok=True)
    missing = sum(int(o['size']) for o in objects if not (a.output / 'raw' / o['name']).exists())
    if shutil.disk_usage(a.output).free < missing + 8 * 1024**3:
        raise ValueError('Require remaining raw bytes plus eight-GiB reserve')
    def one(obj):
        path = a.output / 'raw' / obj['name']
        if not path.exists():
            download_object(obj, a.output / 'raw')
        expected = files[obj['name']]
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Recovered bytes differ from original audited raw object')
        return obj['name']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, _ in enumerate(pool.map(one, objects), 1):
            if count % 16 == 0:
                print(json.dumps({'verified_objects': count, 'required': len(objects)}), flush=True)
    write_json(a.output / 'RECOVERED.json', {'original_report_sha256': digest,
        'files_sha256': sha256(a.inputs / 'FILES.json'), 'objects': len(objects),
        'recordings': 128, 'model_calls': 0, 'population_changed': False})
    print(json.dumps({'status': 'original_fit_objects_recovered', 'model_calls': 0}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
