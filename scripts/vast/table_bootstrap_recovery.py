"""Authenticated URLs resolved locally; no HF token or signed URL saved remotely.

Input transport repair only. The same archived scientific code/freezes run next.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile
import shutil

ROOT=Path('/workspace/table-completion-20260911-v1')
PYTHON='/workspace/table-python-inherited/bin/python'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def main(payload):
    args=payload['queue_args']; jobs=[]
    try:
        def dependencies():
            with (ROOT/'expansion-runtime-recovery-v2.log').open('x') as f:
                p=subprocess.Popen(['bash',str(ROOT/'code/scripts/vast/bootstrap_table_runtime_v2.sh')],stdout=f,stderr=subprocess.STDOUT)
                jobs.append(p)
                if p.wait(timeout=1200):raise RuntimeError('Runtime dependency repair failed')
        def download(a):
            path=Path(a['target']);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                assert sha(path)==a['sha256'];return
            partial=path.with_suffix(path.suffix+'.recovery-v2.partial')
            # Only official signed CDN capabilities, short-lived in process memory.
            with urllib.request.urlopen(a['url'],timeout=90) as r,partial.open('xb') as f:
                shutil.copyfileobj(r,f,4<<20)
            if sha(partial)!=a['sha256']:raise ValueError('Recovered official asset hash differs')
            partial.rename(path)
            print(json.dumps({'verified_asset':a['label'],'bytes':path.stat().st_size}),flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            dep=pool.submit(dependencies)
            list(pool.map(download,payload['assets']))
            dep.result()
        if payload['task']=='pusht':
            proof=json.loads((ROOT/'code/artifacts/offline_study/pusht-native-recovery-20260908-v1/PUBLIC_INPUTS.json').read_text())
            selected={}
            with zipfile.ZipFile(ROOT/'downloads/pusht_noise.zip') as z:
                for old,spec in proof.items():
                    if '/source/data/' not in old:continue
                    name=old.split('/source/data/',1)[1]
                    assert not Path(name).is_absolute() and '..' not in Path(name).parts
                    target=ROOT/'data'/name;target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():
                        with z.open(name) as r,target.open('xb') as f:shutil.copyfileobj(r,f,4<<20)
                    assert sha(target)==spec['sha256'] and target.stat().st_size==spec['bytes']
                    selected[name]=spec
            with (ROOT/'PUSHT_INPUTS_VERIFIED.json').open('x') as f:json.dump({'files':selected,'archive_sha256':sha(ROOT/'downloads/pusht_noise.zip')},f)
        env=dict(os.environ,PYTHONPATH=str(ROOT/'code/src'),PYTHONDONTWRITEBYTECODE='1')
        code="import torch;torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True);from offline_study.model_loader import verified_local_dino_cache\nwith verified_local_dino_cache() as p:print(p)"
        with (ROOT/'expansion-dino-recovery-v2.log').open('x') as f:
            subprocess.run([PYTHON,'-c',code],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
        with (ROOT/'expansion-recovered-queue.log').open('x') as f:
            p=subprocess.Popen([PYTHON,'-u',str(ROOT/'code/scripts/vast/table_resume_streams.py')]+args,stdout=f,stderr=subprocess.STDOUT)
            jobs.append(p)
            if p.wait(timeout=5*3600):raise RuntimeError('Receiving or scientific queue failed')
        with (ROOT/'EXPANSION_RECOVERY_DONE.json').open('x') as f:json.dump({'queue_terminal':True,'full_study_complete':False},f)
    except Exception as e:
        for p in jobs:
            if p.poll() is None:p.terminate()
        with (ROOT/'EXPANSION_RECOVERY_FAILED.json').open('x') as f:json.dump({'error_type':type(e).__name__,'partial_outputs_retained':True},f)
        raise


if __name__=='__main__':main(json.load(sys.stdin))
