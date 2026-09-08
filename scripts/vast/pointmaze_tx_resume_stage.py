"""CPU-only immutable PointMaze234 relocation; never start a GPU workload."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LABEL = 'pointmaze-tx-resume-20260908-v1'
REMOTE = '/workspace/jepa-runtime/' + LABEL
RELAY = '/workspace/jepa-runtime/pointmaze-history-relay-20260908-v1'
KEY = '/tmp/jepa_vast_50123620_ed25519'
RUNTIME_ROOTS = [
    'workspace/jepa-python', 'workspace/jepa-planning-python',
    'root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu',
    'root/.cache/torch/hub/facebookresearch_dinov2_main',
    'root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth']
PUBLIC_CAS = [
    'workspace/jepa-python/lib/python3.10/site-packages/certifi/cacert.pem',
    'workspace/jepa-planning-python/lib/python3.10/site-packages/pip/_vendor/certifi/cacert.pem',
    'root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/site-packages/pip/_vendor/certifi/cacert.pem']
INVENTORY = '''
import hashlib,json,os,pathlib,sys
base=pathlib.Path(sys.argv[1]); roots=json.loads(sys.argv[2]); cas=json.loads(sys.argv[3]); result={}
for name in roots:
 root=base/name
 if not root.exists():raise ValueError('Missing explicit input '+name)
 for p in ([root] if root.is_file() else sorted(root.rglob('*'))):
  if '__pycache__' in p.parts or p.name.startswith('._'):continue
  rel=str(p.relative_to(base))
  if p.name in ('.env','rclone.conf','credentials.json','id_rsa','id_ed25519') or (p.suffix in ('.pem','.key') and rel not in cas):raise ValueError('Credential-like input refused: '+rel)
  if p.is_symlink():result[rel]={'symlink':os.readlink(p)}
  elif p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(4<<20),b''):h.update(b)
   result[rel]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(result,sort_keys=True))
'''


def validate_manifest(manifest, roots):
    for name, row in manifest.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or not any(name == r or name.startswith(r + '/') for r in roots):
            raise ValueError('Manifest member outside explicit payload')
        if set(row) not in ({'symlink'}, {'bytes', 'sha256'}):
            raise ValueError('Unexpected manifest schema')


def main():
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    if sum(r['instance']['totalHour'] for r in rows) > 7:
        raise ValueError('Aggregate budget exceeded')
    workers = {r['id']: r for r in rows}
    def ssh(number, label, region):
        row = workers[number]
        if row['actual_status'] != 'running' or row['label'] != label or row['geolocation'] != region:
            raise ValueError('Explicit owned US worker changed')
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'StrictHostKeyChecking=accept-new',
            '-p', str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    source = ssh(50239185, 'jepa-pusht-wall-history-us-v1', 'New Jersey, US')
    target = ssh(50259194, 'jepa-droid-parallel-us-v3', 'Texas, US')
    # The source-to-target hop explicitly uses only the forwarded agent's project key.
    target_hop = list(target)
    option = target_hop.index('IdentitiesOnly=yes')
    del target_hop[option - 1:option + 1]
    sd, td = source[:1] + ['-i', KEY] + source[1:], target[:1] + ['-i', KEY] + target[1:]
    proof = ROOT / 'artifacts/offline_study' / LABEL
    proof.mkdir(exist_ok=False)
    def inventory(connection, base, roots):
        return json.loads(subprocess.check_output(connection + [shlex.join([
            'python', '-c', INVENTORY, base, json.dumps(roots), json.dumps(PUBLIC_CAS)])], text=True))
    old = ROOT / 'artifacts/offline_study/pointmaze-history-relay-20260908-v1/FILES.json'
    if hashlib.sha256(old.read_bytes()).hexdigest() != 'e7112c30a4e3b2f4e9d5d9819d90a8c7ec25a7b8bcaf90698a438a1abd8b71a9':
        raise ValueError('Original relay manifest changed')
    history = json.loads(old.read_text())
    history_roots = sorted({'/'.join(name.split('/')[:3]) for name in history})
    validate_manifest(history, history_roots)
    if inventory(sd, RELAY, history_roots) != history:
        raise ValueError('Previously verified history relay changed')
    runtime = inventory(sd, '/', RUNTIME_ROOTS)
    validate_manifest(runtime, RUNTIME_ROOTS)
    for name, manifest in [('HISTORY_FILES.json', history), ('RUNTIME_FILES.json', runtime)]:
        (proof / name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    size = sum(r.get('bytes', 0) for r in history.values()) + sum(r.get('bytes', 0) for r in runtime.values())
    check = '''import pathlib,shutil,json,sys
roots=json.loads(sys.argv[1]); out=pathlib.Path(sys.argv[2]); need=int(sys.argv[3])
for name in roots:
 p=pathlib.Path('/')/name
 if p.exists() or p.is_symlink():raise ValueError('Destination already exists; preserve '+str(p))
if out.exists() or shutil.disk_usage('/workspace').free<need+30117624491+24*1024**3:raise ValueError('Existing proof or insufficient reserve')
out.mkdir()
'''
    subprocess.run(td + [shlex.join(['python', '-c', check, json.dumps(history_roots + RUNTIME_ROOTS), REMOTE, str(size)])], check=True)
    agent = subprocess.check_output(['ssh-agent', '-s'], text=True)
    env = dict(os.environ)
    for name in ('SSH_AUTH_SOCK', 'SSH_AGENT_PID'):
        env[name] = re.search(name + r'=([^;]+);', agent).group(1)
    try:
        subprocess.run(['ssh-add', KEY], env=env, stdout=subprocess.DEVNULL, check=True)
        for base, manifest, label in [(RELAY, history, 'history'), ('/', runtime, 'runtime')]:
            print(json.dumps({'stage': 'copy_' + label, 'members': len(manifest), 'bytes': sum(r.get('bytes', 0) for r in manifest.values()), 'gpu_calls': 0}), flush=True)
            receive = 'tar --keep-old-files --no-same-owner -C / -xf -'
            command = 'set -o pipefail; nice -n 10 tar --no-recursion --null -C ' + shlex.quote(base) + ' -cf - -T - | ' + shlex.join(target_hop + [receive])
            subprocess.run(source[:1] + ['-A', '-i', KEY] + source[1:] + [command], env=env,
                input=b'\0'.join(n.encode() for n in manifest) + b'\0', check=True, timeout=2400)
    finally:
        subprocess.run(['ssh-agent', '-k'], env=env, stdout=subprocess.DEVNULL, check=False)
    if (inventory(sd, RELAY, history_roots) != history or inventory(td, '/', history_roots) != history or
            inventory(sd, '/', RUNTIME_ROOTS) != runtime or inventory(td, '/', RUNTIME_ROOTS) != runtime):
        raise ValueError('Source or receiving bytes changed; preserve all and do not launch')
    receipt = {'status': 'pointmaze_cpu_relocation_bytes_verified_not_gpu_clearance', 'source': 50239185,
        'target': 50259194, 'history_members': len(history), 'runtime_members': len(runtime), 'bytes': size,
        'history_manifest_sha256': hashlib.sha256((proof / 'HISTORY_FILES.json').read_bytes()).hexdigest(),
        'runtime_manifest_sha256': hashlib.sha256((proof / 'RUNTIME_FILES.json').read_bytes()).hexdigest(),
        'gpu_calls': 0, 'prior_results_unchanged': True, 'droid_jobs_untouched': True}
    (proof / 'STAGED.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    for name in ('HISTORY_FILES.json', 'RUNTIME_FILES.json', 'STAGED.json'):
        put = 'import pathlib,sys; p=pathlib.Path(' + repr(REMOTE + '/' + name) + '); p.open("xb").write(sys.stdin.buffer.read())'
        subprocess.run(td + [shlex.join(['python', '-c', put])], input=(proof / name).read_bytes(), check=True)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
