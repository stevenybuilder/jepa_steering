#!/usr/bin/env python3
"""Download, hash, and unpack official Push-T assets without analyzing outcomes."""
import argparse
import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revisions = {kind: api.repo_info('facebook/jepa-wms', repo_type=kind).sha
                 for kind in ('dataset', 'model')}
    specs = [('dataset', 'pusht/pusht_noise.zip'),
             ('dataset', 'pusht/hf_data/train-00000-of-00001.parquet'),
             ('model', 'jepa_wm_pusht.pth.tar')]

    def fetch(spec):
        kind, name = spec
        print(json.dumps({'event': 'download_started', 'file': name}), flush=True)
        path = Path(hf_hub_download('facebook/jepa-wms', filename=name,
                    repo_type=kind, revision=revisions[kind],
                    local_dir=root / kind))
        row = {'repo_type': kind, 'revision': revisions[kind], 'file': name,
               'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
        print(json.dumps({'event': 'download_verified', **row}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(fetch, specs))
    unpack = root / 'unpacked'
    unpack.mkdir(exist_ok=True)
    with zipfile.ZipFile(root / 'dataset/pusht/pusht_noise.zip') as archive:
        size = sum(item.file_size for item in archive.infolist())
        if size > shutil.disk_usage(root).free - (1 << 30):
            raise RuntimeError('Insufficient disk space to unpack Push-T data')
        for item in archive.infolist():
            if not (unpack / item.filename).resolve().is_relative_to(unpack):
                raise RuntimeError('Unexpected archive member path')
        print(json.dumps({'event': 'extract_started', 'bytes': size}), flush=True)
        archive.extractall(unpack)
    receipt = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'task': 'Push-T', 'stage': 'official assets downloaded, hashed, unpacked',
               'outcome_analysis_performed': False, 'outputs': outputs,
               'unpacked_bytes': size, 'unpacked_path': str(unpack)}
    temporary = root / 'DONE.json.tmp'
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(root / 'DONE.json')
    print(json.dumps({'event': 'done', 'receipt': str(root / 'DONE.json')}), flush=True)


if __name__ == '__main__':
    main()
