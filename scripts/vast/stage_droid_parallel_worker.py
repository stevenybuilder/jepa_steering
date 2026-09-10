"""Stage the verified DROID relay and frozen v3 source on owned Texas50259194."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from stage_droid_relay import INVENTORY, PATHS

ROOT = Path(__file__).resolve().parents[2]
RELAY = '/workspace/jepa-runtime/droid-relay-20260908-v2'
CODE = 'droid-coupling-code-20260908-v3'


def main():
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    if sum(row['instance']['totalHour'] for row in rows) > 7:
        raise ValueError('Current hourly budget exceeded')
    workers = {row['id']: row for row in rows}
    def ssh(number, label, region):
        row = workers[number]
        if row['actual_status'] != 'running' or row['label'] != label or row['geolocation'] != region:
            raise ValueError('Explicit owned US endpoint changed')
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
            '-o', 'StrictHostKeyChecking=accept-new', '-p', str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    source = ssh(50239185, 'jepa-pusht-wall-history-us-v1', 'New Jersey, US')
    target = ssh(50259194, 'jepa-droid-parallel-us-v3', 'Texas, US')
    key = '/tmp/jepa_vast_50123620_ed25519'
    source_direct, target_direct = source[:1]+['-i',key]+source[1:], target[:1]+['-i',key]+target[1:]
    proof = ROOT / 'artifacts/offline_study/droid-parallel-worker-20260908-v3-retry-v2'
    proof.mkdir(exist_ok=False)
    manifest = json.loads((ROOT / 'artifacts/offline_study/droid-relay-20260908-v2/FILES.json').read_text())
    if any(not name.startswith('workspace/') or '..' in Path(name).parts for name in manifest):
        raise ValueError('Relay member outside explicit workspace payload')
    def inventory(connection, base):
        return json.loads(subprocess.check_output(connection + [shlex.join(['python', '-c', INVENTORY, base, json.dumps(PATHS)])], text=True))
    if inventory(source_direct, RELAY) != manifest:
        raise ValueError('Verified relay changed')
    subprocess.run(target_direct + ['test ! -e /workspace/jepa-droid-python && test ! -e /workspace/jepa-runtime'], check=True)
    env = dict(os.environ)
    agent = subprocess.check_output(['ssh-agent', '-s'], text=True)
    for name in ('SSH_AUTH_SOCK','SSH_AGENT_PID'):
        env[name] = re.search(name+r'=([^;]+);', agent).group(1)
    try:
        subprocess.run(['ssh-add',key],env=env,stdout=subprocess.DEVNULL,check=True)
        receiving = 'tar --keep-old-files --no-same-owner --strip-components=1 -C /workspace -xf -'
        command = 'set -o pipefail; nice -n 10 tar --no-recursion --null -C '+shlex.quote(RELAY)+' -cf - -T - | '+shlex.join(target+[receiving])
        print(json.dumps({'stage':'copy_verified_inputs','members':len(manifest),'bytes':sum(x.get('bytes',0) for x in manifest.values())}),flush=True)
        subprocess.run(source[:1]+['-A','-i',key]+source[1:]+[command],env=env,
            input=b'\0'.join(name.encode() for name in manifest)+b'\0',check=True,timeout=2400)
        command = 'set -o pipefail; tar --exclude=__pycache__ -C /workspace/jepa-runtime -cf - '+CODE+' droid-coupling-behavior-20260908-v3/freeze | '+shlex.join(target+[
            'tar --keep-old-files --no-same-owner -C /workspace/jepa-runtime -xf -'])
        subprocess.run(source[:1]+['-A','-i',key]+source[1:]+[command],env=env,check=True,timeout=300)
    finally:
        subprocess.run(['ssh-agent','-k'],env=env,stdout=subprocess.DEVNULL,check=False)
    if inventory(target_direct, '/') != manifest or inventory(source_direct,RELAY) != manifest:
        raise ValueError('Receiving input bytes differ; no GPU launch')
    source_manifest = json.loads((ROOT / 'artifacts/offline_study/droid-coupling-preparation-20260908-v3/FILES.json').read_text())
    check = '''import hashlib,json,pathlib,sys
r=pathlib.Path('/workspace/jepa-runtime');files=json.load(sys.stdin)
for name,want in files.items():
 p=r/name
 if p.is_symlink() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Receiving source member differs')
print(json.dumps({'source_members_verified':len(files)}))'''
    output = subprocess.check_output(target_direct+[shlex.join(['python','-c',check])],input=json.dumps(source_manifest),text=True)
    (proof/'SOURCE.json').write_text(output)
    receipt = {'status':'droid_parallel_inputs_source_staged_verified_not_gpu_clearance','instance':50259194,
        'input_members':len(manifest),'source_members':len(source_manifest),'source_sha256':'249a8cdb9326cd79f3b9180a830e409cdd0ffa4f2f3be2ac393c32e30ab81a54',
        'freeze_sha256':'369d632792b6b482f5c63bbcd599dd7576cf949001ac005baa1cb5f600681862','gpu_calls':0}
    (proof/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
