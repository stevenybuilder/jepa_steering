"""Verified CPU-only relay of fixed DROID inputs between two owned US leases.

Never copies credentials, executes a GPU model, or overwrites an existing relay.
The short-lived forwarded SSH agent is removed when transfer/verification ends.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEST = '/workspace/jepa-runtime/droid-relay-20260908-v2'
PATHS = ['workspace/jepa-droid-python'] + ['workspace/jepa-runtime/' + x for x in (
    'droid-assets-20260907-v1', 'dinov3-native-20260907-v2',
    'droid-coupling-fit-20260908-v2', 'droid-native-replication-20260907-v1/shard-all',
    'droid-native-engineering-20260907-v7')]

INVENTORY = '''
import hashlib,json,os,pathlib,sys
base=pathlib.Path(sys.argv[1]); roots=json.loads(sys.argv[2]); result={}
for name in roots:
 root=base/name
 if not root.exists():raise ValueError('Missing explicit input '+name)
 for p in ([root] if root.is_file() else sorted(root.rglob('*'))):
  if '__pycache__' in p.parts or p.name.startswith('._'):continue
  public_ca = str(p.relative_to(base)) == 'workspace/jepa-droid-python/lib/python3.11/site-packages/pip/_vendor/certifi/cacert.pem'
  if p.name in ('.env','rclone.conf','credentials.json') or (p.suffix in ('.pem','.key') and not public_ca):
   raise ValueError('Credential-like input refused')
  if p.is_symlink():
   result[str(p.relative_to(base))]={'symlink':os.readlink(p)}
  elif p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(4<<20),b''):h.update(b)
   result[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(result,sort_keys=True))
'''


def main():
    global PATHS, DEST
    parser = argparse.ArgumentParser(); parser.add_argument('--pointmaze', action='store_true'); args = parser.parse_args()
    label = 'droid-relay-20260908-v2'
    if args.pointmaze:
        label = 'pointmaze-history-relay-20260908-v1'
        DEST = '/workspace/jepa-runtime/' + label
        PATHS = ['workspace/jepa-runtime/' + name for name in (
            'pointmaze-history-code-20260908-v1', 'pointmaze-training-accumulation-pilot-20260908-v1',
            'pointmaze-training-inputs-20260908-v1', 'navigation-input-check-20260907-v1',
            'navigation-assets-20260907-v1/protocol.json', 'navigation-assets-20260907-v1/report.json',
            'navigation-assets-20260907-v1/DONE.json',
            'navigation-assets-20260907-v1/downloads/dataset/point_maze/point_maze.zip',
            'pointmaze-training-history-20260908-v1/seed-234/epoch-one-engineering',
            'pointmaze-training-history-20260908-v1/seed-234/remaining-epochs')]
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    if sum(x['instance']['totalHour'] for x in rows) > 7:
        raise ValueError('Current aggregate budget exceeded')
    workers = {x['id']: x for x in rows}
    def ssh(number, label):
        row = workers[number]
        if row['actual_status'] != 'running' or not row['geolocation'].endswith(', US') or row['label'] != label:
            raise ValueError('Explicit owned US endpoint changed')
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
            '-o', 'StrictHostKeyChecking=accept-new', '-p', str(row['ports']['22/tcp'][0]['HostPort']),
            'root@' + row['public_ipaddr']]
    source = ssh(50189244, 'jepa-us-droid-history-20260907')
    destination = ssh(50239185, 'jepa-pusht-wall-history-us-v1')
    key = '/tmp/jepa_vast_50123620_ed25519'
    direct_source, direct_dest = source[:1] + ['-i', key] + source[1:], destination[:1] + ['-i', key] + destination[1:]
    proof = ROOT / 'artifacts/offline_study' / label
    proof.mkdir(exist_ok=False)
    def inventory(connection, base):
        return json.loads(subprocess.check_output(connection + [shlex.join(['python', '-c', INVENTORY, base, json.dumps(PATHS)])], text=True))
    manifest = inventory(direct_source, '/')
    size = sum(x.get('bytes', 0) for x in manifest.values())
    (proof / 'FILES.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'stage': 'fixed_input_manifest', 'members': len(manifest), 'bytes': size}), flush=True)
    preflight = "import pathlib,shutil; p=pathlib.Path(" + repr(DEST) + "); assert not p.exists(); assert shutil.disk_usage(p.parent).free>" + str(size + 15*1024**3) + "; p.mkdir()"
    subprocess.run(direct_dest + [shlex.join(['python', '-c', preflight])], check=True)
    text = subprocess.check_output(['ssh-agent', '-s'], text=True)
    env = dict(os.environ)
    for name in ('SSH_AUTH_SOCK', 'SSH_AGENT_PID'):
        env[name] = re.search(name + r'=([^;]+);', text).group(1)
    try:
        subprocess.run(['ssh-add', key], env=env, stdout=subprocess.DEVNULL, check=True)
        receiving = 'tar --keep-old-files --no-same-owner -C ' + shlex.quote(DEST) + ' -xf -'
        transfer = 'set -o pipefail; nice -n 10 tar --no-recursion --null -C / -cf - -T - | ' + shlex.join(destination + [receiving])
        subprocess.run(source[:1] + ['-A', '-i', key] + source[1:] + [transfer], env=env,
            input=b'\0'.join(x.encode() for x in manifest) + b'\0', check=True, timeout=2400)
    finally:
        subprocess.run(['ssh-agent', '-k'], env=env, stdout=subprocess.DEVNULL, check=False)
    if inventory(direct_source, '/') != manifest or inventory(direct_dest, DEST) != manifest:
        raise ValueError('Source or receiving relay content differs; preserve both')
    receipt = {'status': 'all_droid_relay_members_verified', 'source': 50189244, 'destination': 50239185,
        'members': len(manifest), 'bytes': size, 'source_files_unchanged': True, 'gpu_jobs_untouched': True,
        'manifest_sha256': hashlib.sha256((proof / 'FILES.json').read_bytes()).hexdigest(), 'relay_root': DEST}
    (proof / 'VERIFIED.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    for name in ('FILES.json', 'VERIFIED.json'):
        command = "import pathlib,sys; p=pathlib.Path(" + repr(DEST + '/' + name) + "); f=p.open('xb'); f.write(sys.stdin.buffer.read()); f.close()"
        subprocess.run(direct_dest + [shlex.join(['python', '-c', command])], input=(proof / name).read_bytes(), check=True)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
