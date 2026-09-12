"""Independent asset/dependency preparation; ephemeral official URL in memory only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT = Path('/workspace/metaworld-components-20260911-v1')
PYTHON = '/workspace/component-python/bin/python'


def main(payload):
    try:
        def dependencies():
            with (ROOT / 'bootstrap.log').open('x') as log:
                subprocess.run(['bash', str(ROOT / 'ops/component_rental_bootstrap.sh')],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        def download():
            target = ROOT / 'checkpoints/jepa_wm_metaworld.pth.tar'
            target.parent.mkdir()
            partial = target.with_suffix('.partial')
            h = hashlib.sha256()
            with urllib.request.urlopen(payload['checkpoint_url'], timeout=90) as response, partial.open('xb') as out:
                while block := response.read(4 << 20):
                    h.update(block)
                    out.write(block)
            assert h.hexdigest() == 'c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'
            partial.rename(target)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(dependencies), pool.submit(download)]
            for job in jobs:
                job.result()
        payload.pop('checkpoint_url', None)
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'code/src'), PYTHONDONTWRITEBYTECODE='1')
        with (ROOT / 'encoder.log').open('x') as log:
            subprocess.run([PYTHON, '-c', "import torch;torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True);from offline_study.model_loader import verified_local_dino_cache\nwith verified_local_dino_cache() as proof: print(proof)"],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        with (ROOT / 'queue.log').open('x') as log:
            slot_args = (['--logical-slots', *map(str, payload['logical_slots'])]
                if 'logical_slots' in payload else [])
            subprocess.run([PYTHON, '-u', str(ROOT / 'ops/component_extension_queue.py'),
                '--root', str(ROOT), '--checkpoint', str(ROOT / 'checkpoints/jepa_wm_metaworld.pth.tar'),
                '--gpus', str(payload['gpus']), '--total-gpus', str(payload['total_gpus']),
                '--gpu-offset', str(payload['gpu_offset']), '--deadline', str(payload['deadline']), *slot_args],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    except Exception as e:
        # Never log urllib's exception string: it can contain an expiring URL.
        if not (ROOT / 'TERMINAL.json').exists():
            with (ROOT / 'TERMINAL.json').open('x') as f:
                json.dump({'status': 'preparation_failed', 'error_type': type(e).__name__, 'time': time.time()}, f)
        print(json.dumps({'error_type': type(e).__name__, 'partial_outputs_retained': True}), flush=True)
