"""Read/copy stopped JEPA disks for user-authorized preservation; no lifecycle API.

Vast's modern rsync modules require C.<instance>. Credentials stay local.
Every transfer checks the actual rsync exit status (the installed CLI does not).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / 'artifacts/offline_study/final-storage-release-20260910-v1'
ALLOWED = {50125440,50135088,50135089,50135090,50159352,50189244,
           50195621,50205763,50229002,50231985,50233992,50239185,50259194,50245262}
KEY = '/tmp/jepa_vast_50123620_ed25519'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)


def client():
    from vastai.api.client import VastClient
    return VastClient(api_key=Path('/Users/stevenyang/.config/vastai/vast_api_key')
                      .read_text().strip(), retry=1, timeout=35)


def command(instance, source):
    if instance not in ALLOWED or not source.startswith('/workspace/') or '..' in source.split('/'):
        raise ValueError('Require explicitly in-scope workspace source')
    from vastai.api.storage import copy
    response = copy(client(), f'C.{instance}', None, source, str(BASE / 'scratch'))
    if not response.get('success'):
        raise RuntimeError('Provider did not authorize disk read')
    addr, port = response['src_addr'], int(response['src_port'])
    if not re.fullmatch(r'[a-zA-Z0-9.:-]+', addr) or not 0 < port < 65536:
        raise ValueError('Invalid provider endpoint')
    ssh = f'ssh -i {KEY} -p {port} -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new'
    return ['rsync', '--timeout=45', '-e', ssh], f'vastai_kaalia@{addr}::C.{instance}/{source}'


def inventory(instance):
    from vastai.api.instances import execute
    root = BASE / str(instance)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result = {'instance':instance, 'utc':stamp, 'no_restart':True}
    for label, cmd in [('root','ls -la'), ('workspace','ls -la workspace')]:
        try:
            output = execute(client(), instance, cmd)
            (root / f'{stamp}-{label}.txt').write_text(str(output))
        except Exception as error:
            result[label+'_error'] = type(error).__name__
    try:
        cmd, source = command(instance, '/workspace/')
        cmd += ['--list-only', '-r', '--exclude=.venv/', '--exclude=venv/',
                '--exclude=__pycache__/', '--exclude=.git/', '--exclude=.cache/', source]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        (root / f'{stamp}-rsync-list.txt').write_text(p.stdout)
        (root / f'{stamp}-rsync-stderr.txt').write_text(p.stderr)
        result.update(returncode=p.returncode, lines=len(p.stdout.splitlines()))
        entries = []
        for line in p.stdout.splitlines():
            match = re.match(r'^([dl-][rwxstST-]{9})\s+([0-9,]+)\s+(\S+)\s+(\S+)\s+(.+)$',line)
            if match:
                mode,size,date,time,name = match.groups()
                entries.append({'path':name, 'bytes':int(size.replace(',','')),
                                'mode':mode,'mtime':date+' '+time})
        write(root / f'{stamp}-inventory.json', {'entries':entries, **result})
        result.update(files=sum(x['mode'].startswith('-') for x in entries),
                      bytes=sum(x['bytes'] for x in entries if x['mode'].startswith('-')))
    except Exception as error:
        result['copy_error'] = type(error).__name__
    write(root / f'{stamp}-STATUS.json', result)
    return result


def bootstrap_archive(instance):
    if instance not in {50135088,50135089,50135090,50229002}:
        raise ValueError('Bootstrap-only bounded archive')
    root=BASE / str(instance)
    inv=json.loads(sorted(root.glob('*-inventory.json'))[-1].read_text())
    if instance != 50229002 and inv['returncode'] != 0:
        raise ValueError('Incomplete inventory')
    if sum(x['bytes'] for x in inv['entries']) > 100 << 20:
        raise ValueError('Unexpected bootstrap disk content')
    stage=root/'bootstrap-stage'; stage.mkdir(exist_ok=False)
    if instance != 50229002:
        cmd,source=command(instance,'/workspace/')
        cmd += ['-rlt', '--exclude=jepa-python/', '--exclude=.venv/',
                '--exclude=.cache/', '--exclude=.git/', '--exclude=__pycache__/',
                '--exclude=.env', '--exclude=.env.*', source, str(stage)+'/']
        with (root/'bootstrap-copy.log').open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=log,check=True,timeout=240)
    manifest={}
    for path in sorted(stage.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected bootstrap symlink')
        if path.is_file():
            manifest['workspace/'+path.relative_to(stage).as_posix()] = {
                'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(root.glob('*')):
        if path.is_file() and path.suffix in {'.json','.txt','.log'}:
            manifest['audit/'+path.name]={'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    write(root/'BOOTSTRAP_FILES.json',manifest)
    archive=root/f'instance-{instance}-bootstrap-and-inventory.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name in manifest:
            path=stage/name.removeprefix('workspace/') if name.startswith('workspace/') else root/name.removeprefix('audit/')
            tar.add(path,arcname=name,recursive=False)
        tar.add(root/'BOOTSTRAP_FILES.json',arcname='BOOTSTRAP_FILES.json')
    proof={'instance':instance,'archive':str(archive),'bytes':archive.stat().st_size,
           'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(manifest),
           'no_scientific_results_found':True,'scope':'workspace bootstrap/code/logs and inventory',
           'excluded':'reconstructible empty Python environment; credentials outside workspace'}
    write(root/'BOOTSTRAP_ARCHIVE.json',proof)
    return proof


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instances', nargs='+', type=int, choices=sorted(ALLOWED))
    parser.add_argument('--bootstrap-archive',action='store_true')
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(bootstrap_archive if args.bootstrap_archive else inventory,x) for x in args.instances]):
            print(json.dumps(future.result()), flush=True)
