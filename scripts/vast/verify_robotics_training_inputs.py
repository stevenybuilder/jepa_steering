"""Stage a minimal CPU verifier and audit existing complete author raw inputs."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

import navigation_redistribution_stage as stage
from navigation_redistribution_common import digest, write


PROJECT = Path(__file__).resolve().parents[2]
ROOT = '/workspace/jepa-runtime/robotics-input-verification-20260908-v2'
OUTPUT = PROJECT / 'artifacts/offline_study/robotics-input-verification-20260908-v2'
NAMES = ('src/offline_study/__init__.py', 'src/offline_study/protocol.py',
         'src/offline_study/robotics_training_inputs.py', 'tests/test_robotics_training_inputs.py')

AUDIT = r'''import hashlib,json,os,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be hidden')
if sha(root/'SOURCE_FILES.json')!=expected:raise ValueError('Source manifest changed')
manifest=json.loads((root/'SOURCE_FILES.json').read_text())
for name,row in manifest.items():
 p=root/name
 if p.is_symlink() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
  raise ValueError('Receiving source changed: '+name)
env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
with (root/'cpu-tests.log').open('x') as log:
 subprocess.run(['/usr/bin/python3','-m','unittest','discover','-s',str(root/'tests'),
  '-p','test_robotics_training_inputs.py','-v'],env=env,stdout=log,stderr=subprocess.STDOUT,
  stdin=subprocess.DEVNULL,check=True,timeout=120)
sys.path.insert(0,str(root/'src'))
from offline_study.robotics_training_inputs import verify_inputs
from offline_study.protocol import write_json
began=time.monotonic(); results={}
base=pathlib.Path('/workspace/jepa-runtime')
mw=base/'fixed-response-assets-20260908-v1'
pt=base/'pusht-planning-assets-20260908-v2'
try:
 results['metaworld']=verify_inputs('metaworld',mw/'metaworld/data',
  provenance=mw/'DOWNLOAD_MANIFEST.json',output=root/'metaworld')
 print(json.dumps({'stage':'metaworld_full_raw_bytes_verified','files':results['metaworld']['files']}),flush=True)
 results['pusht']=verify_inputs('pusht',pt/'data/pusht_noise',provenance=pt/'files.json',
  archive=pt/'downloads/datasets/pusht_noise.zip',output=root/'pusht')
 if 'torch' in sys.modules:raise ValueError('Raw audit unexpectedly loaded torch')
 for name,row in manifest.items():
  if sha(root/name)!=row['sha256']:raise ValueError('Verifier changed during audit')
 write_json(root/'DONE.json',{'status':'both_full_robotics_raw_pools_verified_no_execution_access',
  'source_manifest_sha256':expected,'cpu_test_log_sha256':sha(root/'cpu-tests.log'),
  'task_reports_sha256':{task:sha(root/task/'report.json') for task in results},
  'torch_imported':False,'gpu_calls':0,'seconds':time.monotonic()-began})
 print(json.dumps({'status':'complete_raw_input_verification','results':results,
  'done_sha256':sha(root/'DONE.json')}),flush=True)
except Exception as error:
 write_json(root/'FAILED.json',{'error':str(error),'completed_tasks':list(results),
  'gpu_calls':0,'training_or_validation_authorized':False})
 raise
'''


def collect(ssh):
    """Read back compact, complete evidence; never retrieve raw training data."""
    names = ['SOURCE_FILES.json', 'cpu-tests.log', 'DONE.json', *NAMES]
    names += [f'{task}/{name}.json' for task in ('metaworld', 'pusht')
              for name in ('protocol', 'report', 'DONE')]
    program = '''import base64,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); names=json.loads(sys.argv[2]); result={}
if (root/'FAILED.json').exists():raise ValueError('Failed receiving audit')
for name in names:
 p=root/name
 if p.is_symlink() or p.stat().st_size>131072:raise ValueError('Unexpected receipt')
 data=p.read_bytes()
 result[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
  'base64':base64.b64encode(data).decode()}
print(json.dumps(result))
'''
    result = subprocess.check_output(ssh + [shlex.join(['/usr/bin/python3', '-c',
        program, ROOT, json.dumps(names)])], text=True, timeout=40)
    fetched = json.loads(result)
    if set(fetched) != set(names):
        raise ValueError('Incomplete receiving proof readback')
    payload = {}
    for name, row in fetched.items():
        data = base64.b64decode(row['base64'], validate=True)
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Receiving receipt changed: ' + name)
        payload[name] = data
    document = lambda name: json.loads(payload[name])
    done = document('DONE.json')
    if (done['status'] != 'both_full_robotics_raw_pools_verified_no_execution_access' or
            done['source_manifest_sha256'] != digest(OUTPUT / 'SOURCE_FILES.json') or
            done['source_manifest_sha256'] != fetched['SOURCE_FILES.json']['sha256'] or
            done['cpu_test_log_sha256'] != fetched['cpu-tests.log']['sha256'] or
            done['gpu_calls'] != 0 or done['torch_imported'] is not False):
        raise ValueError('Receiving completion binding changed')
    source = document('SOURCE_FILES.json')
    for name in NAMES:
        if source[name] != {key: fetched[name][key] for key in ('bytes', 'sha256')}:
            raise ValueError('Receiving source evidence differs')
    for task in ('metaworld', 'pusht'):
        report = document(f'{task}/report.json')
        if (done['task_reports_sha256'][task] != fetched[f'{task}/report.json']['sha256'] or
                document(f'{task}/DONE.json')['report_sha256'] != fetched[f'{task}/report.json']['sha256'] or
                report['protocol_sha256'] != fetched[f'{task}/protocol.json']['sha256'] or
                report['source_sha256'] != source['src/offline_study/robotics_training_inputs.py']['sha256'] or
                report['training_or_validation_authorized'] is not False):
            raise ValueError('Task receipt binding changed')
    destination = OUTPUT / 'readback'
    destination.mkdir(exist_ok=False)
    for name, data in payload.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        if digest(target) != fetched[name]['sha256']:
            raise ValueError('Local readback differs')
    write(OUTPUT / 'READBACK.json', {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'remote_root': ROOT, 'status': 'complete_receiving_receipts_hash_verified',
        'files': {name: {key: row[key] for key in ('bytes', 'sha256')}
                  for name, row in fetched.items()}, 'gpu_calls': 0,
        'training_or_validation_authorized': False})
    return done


def main():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    if ROOT not in board or 'native 32-rank input byte audit' not in board:
        raise ValueError('Exact CPU audit reservation required')
    connections, authority = stage.connections()
    ssh = connections[50205763]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest = {name: {'bytes': (PROJECT / name).stat().st_size,
                       'sha256': digest(PROJECT / name)} for name in NAMES}
    write(OUTPUT / 'SOURCE_FILES.json', manifest)
    write(OUTPUT / 'INTENT.json', {'root': ROOT, 'instance': 50205763, 'provider': authority,
        'source_manifest_sha256': digest(OUTPUT / 'SOURCE_FILES.json'),
        'program': AUDIT, 'gpu_calls': 0, 'training_or_validation_authorized': False})
    create = 'import pathlib,sys;pathlib.Path(sys.argv[1]).mkdir(exist_ok=False)'
    subprocess.run(ssh + [shlex.join(['/usr/bin/python3', '-c', create, ROOT])], check=True, timeout=30)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name in NAMES:
                archive.add(PROJECT / name, arcname=name, recursive=False)
            archive.add(OUTPUT / 'SOURCE_FILES.json', arcname='SOURCE_FILES.json', recursive=False)
        stream.seek(0)
        subprocess.run(ssh + [shlex.join(['tar', '--keep-old-files', '-xzf', '-', '-C', ROOT])],
                       stdin=stream, check=True, timeout=30)
    command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
        '/usr/bin/ionice', '-c', '3', '/usr/bin/nice', '-n', '19', '/usr/bin/python3', '-u', '-c',
        AUDIT, ROOT, digest(OUTPUT / 'SOURCE_FILES.json')]
    result = subprocess.run(ssh + [shlex.join(command)], text=True, capture_output=True, timeout=900)
    with (OUTPUT / 'receiving.log').open('x') as log:
        log.write(result.stdout + result.stderr)
    if result.returncode:
        write(OUTPUT / 'FAILED.json', {'exit_code': result.returncode,
            'receiving_log_sha256': digest(OUTPUT / 'receiving.log'), 'remote_root': ROOT,
            'failed_and_partial_artifacts_preserved': True, 'gpu_calls': 0})
        raise RuntimeError('CPU audit failed; inspect preserved receiving.log without retrying this root')
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    write(OUTPUT / 'REMOTE_RESULT.json', actual)
    collect(ssh)
    print(json.dumps(actual), flush=True)


if __name__ == '__main__':
    main()
