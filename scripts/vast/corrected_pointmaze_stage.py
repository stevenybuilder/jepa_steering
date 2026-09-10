"""Stage reviewed corrected source on Texas; CPU tests only, no runnable plan."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import corrected_pointmaze_queue as q


TARGET = 50259194
LABEL = 'jepa-droid-parallel-us-v3'
ROOT = '/workspace/jepa-runtime/pointmaze-corrected-queue-20260908-v1'
KEY = '/tmp/jepa_vast_50123620_ed25519'
FILES = ['source.tar.gz', 'SOURCE_FILES.json', 'EXECUTION_APPROVAL.json', 'PROPOSAL.json', 'PREPARED.json']
RECEIVE = r'''import hashlib,importlib.util,json,os,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path(sys.argv[1]); wanted=json.loads(sys.argv[2]); began=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def write(path,data):
 with pathlib.Path(path).open('x') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
try:
 for name,digest in wanted.items():
  if pathlib.Path(name).name!=name or sha(root/name)!=digest:raise ValueError('Received bundle member changed: '+name)
 proof=read(root/'PREPARED.json'); proposal=read(root/'PROPOSAL.json'); manifest=read(root/'SOURCE_FILES.json')
 if (proof['source_archive_sha256']!=sha(root/'source.tar.gz') or proof['source_files_sha256']!=sha(root/'SOURCE_FILES.json') or
  proof['proposal_sha256']!=sha(root/'PROPOSAL.json') or proof['execution_approval_sha256']!=sha(root/'EXECUTION_APPROVAL.json') or
  proposal['source_root']!=str(root/'code') or 'assignments' in proposal or (root/'PLAN.json').exists()):
  raise ValueError('Not the reviewed unbound preparation')
 vendor=pathlib.Path(proposal['vendor'])
 if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()!='13cf1d9c7e476f53c17714d2e0f1dc239a883ce0' or
  subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True).strip()):raise ValueError('Pinned vendor differs')
 seen=set()
 with tarfile.open(root/'source.tar.gz','r:gz') as archive:
  for member in archive:
   relative=pathlib.Path(member.name)
   if (member.name not in manifest or member.name in seen or relative.is_absolute() or '..' in relative.parts or
    not member.name.startswith('code/')):raise ValueError('Unregistered source archive path')
   target=root/relative
   for parent in target.parents:
    if parent==root:break
    if parent.is_symlink():raise ValueError('Source extraction through link')
   target.parent.mkdir(parents=True,exist_ok=True)
   row=manifest[member.name]
   if member.issym():
    if member.name!='code/vendor/jepa-wms' or member.linkname!=str(vendor) or row!={'symlink':str(vendor)}:
     raise ValueError('Unexpected source symlink')
    target.symlink_to(vendor,target_is_directory=True)
   elif member.isfile() and set(row)=={'bytes','sha256'} and member.size==row['bytes']:
    with archive.extractfile(member) as src,target.open('xb') as dst:
     for block in iter(lambda:src.read(4<<20),b''):dst.write(block)
    if sha(target)!=row['sha256']:raise ValueError('Extracted source changed')
   else:raise ValueError('Invalid source member type')
   seen.add(member.name)
 if seen!=set(manifest):raise ValueError('Incomplete source archive')
 path=root/'code/scripts/vast/corrected_pointmaze_queue.py'
 spec=importlib.util.spec_from_file_location('corrected_pointmaze_queue',path); worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
 worker.verify_static(proposal)
 env=worker.environment(proposal)
 if env['CUDA_VISIBLE_DEVICES']!='':raise ValueError('CPU tests must hide CUDA')
 version=subprocess.check_output([proposal['python'],'-c','import sys;print(".".join(map(str,sys.version_info[:2])))'],env=env,text=True).strip()
 if version!='3.10':raise ValueError('Do not use live DROID Python3.11')
 results={}
 for pattern in ('test_native_training_sampler.py','test_pointmaze_training.py','test_corrected_pointmaze_queue.py'):
  logfile=root/('cpu-'+pattern+'.log')
  with logfile.open('x') as log:
   subprocess.run([proposal['python'],'-m','unittest','discover','-s',str(root/'code/tests'),'-p',pattern,'-v'],
    env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
  results[pattern]=sha(logfile)
 # Recheck every immutable source/test member and pinned existing reference after tests.
 for name,row in manifest.items():
  path=root/name
  if 'symlink' in row:
   if not path.is_symlink() or os.readlink(path)!=row['symlink']:raise ValueError('Vendor link changed')
  elif path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Source mutated during tests')
 worker.verify_static(proposal)
 if (root/'PLAN.json').exists() or (root/'corrected-histories').exists():raise ValueError('Unexpected runnable plan or training output')
 receipt={'status':'corrected_pointmaze_source_received_cpu_tests_passed_not_activated','instance':50259194,
  'source_commit':proof['source_commit'],'source_sha256':proof['source_sha256'],'source_files':len(manifest),
  'source_archive_sha256':sha(root/'source.tar.gz'),'source_manifest_sha256':sha(root/'SOURCE_FILES.json'),
  'proposal_sha256':sha(root/'PROPOSAL.json'),'execution_approval_sha256':sha(root/'EXECUTION_APPROVAL.json'),
  'cpu_tests':results,'python_version':version,'gpu_calls':0,'runnable_plan_present':False,
  'queue_launched':False,'existing_runtime_or_library_files_modified':False,'seconds':time.monotonic()-began}
 write(root/'RECEIVING.json',receipt)
 print(json.dumps({'receiving_sha256':sha(root/'RECEIVING.json'),**receipt}))
except Exception as error:
 write(root/'STAGING_FAILED.json',{'error':str(error),'gpu_calls':0,'partial_files_preserved':True})
 raise
'''


def connection():
    rows = json.loads(subprocess.check_output(['/Users/stevenyang/.local/bin/vastai', 'show', 'instances', '--raw'], text=True))
    target = next(row for row in rows if row['id'] == TARGET)
    total = sum(row['dph_total'] if row['actual_status'] == 'running' else row['storage_total_cost'] for row in rows)
    if (target['label'] != LABEL or target['actual_status'] != 'running' or target['intended_status'] != 'running' or
            not target['geolocation'].endswith(', US') or total > 7):
        raise ValueError('Owned US receiver/running state/budget differs')
    # Authentication stays local: never copy a key or print provider credentials.
    ports = {int(row['HostPort']) for row in target['ports'].get('22/tcp', [])}
    if len(ports) != 1:
        raise ValueError('Missing or ambiguous published SSH endpoint')
    return target['public_ipaddr'], ports.pop(), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous staging receipts')
    proof = q.read(args.prepared / 'PREPARED.json')
    if proof['source_sha256'] != 'c1116183b4f00f2ea6afebf4c344c1b69f6ede75809f69d83f5f63b6880e8a00':
        raise ValueError('Only the root-reviewed immutable scientific source may be staged')
    expected = {name: q.digest(args.prepared / name) for name in FILES}
    host, port, total = connection()
    ssh = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', str(port), 'root@' + host]
    args.output.mkdir(parents=True)
    q.write(args.output / 'INTENT.json', {'target': TARGET, 'root': ROOT, 'files_sha256': expected,
        'aggregate_hourly': total, 'gpu_calls': 0, 'runnable_plan_copied': False})
    try:
        initialize = 'from pathlib import Path;Path(' + repr(ROOT) + ').mkdir(parents=False,exist_ok=False)'
        subprocess.run(ssh + [shlex.join(['/usr/bin/python3', '-c', initialize])], check=True, timeout=30)
        command = ['scp', '-q', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', str(port)]
        subprocess.run(command + [str(args.prepared / name) for name in FILES] + ['root@' + host + ':' + ROOT + '/'], check=True, timeout=120)
        result = subprocess.check_output(ssh + [shlex.join(['/usr/bin/python3', '-c', RECEIVE, ROOT, json.dumps(expected)])], text=True, timeout=1000)
        receipt = json.loads(result.strip().splitlines()[-1])
        subprocess.run(command + ['root@' + host + ':' + ROOT + '/RECEIVING.json', str(args.output / 'RECEIVING.json')], check=True, timeout=30)
        if q.digest(args.output / 'RECEIVING.json') != receipt['receiving_sha256']:
            raise ValueError('Receiving proof differs on readback')
        for pattern, sha in receipt['cpu_tests'].items():
            name = 'cpu-' + pattern + '.log'
            subprocess.run(command + ['root@' + host + ':' + ROOT + '/' + name, str(args.output / name)], check=True, timeout=30)
            if q.digest(args.output / name) != sha:
                raise ValueError('CPU test log readback changed')
        if expected != {name: q.digest(args.prepared / name) for name in FILES}:
            raise ValueError('Local prepared payload changed during staging')
        q.write(args.output / 'VERIFIED.json', {'status': 'receiving_proof_and_cpu_logs_readback_verified',
            'target': TARGET, 'root': ROOT, 'receiving_sha256': receipt['receiving_sha256'],
            'gpu_calls': 0, 'runnable_plan_copied': False, 'queue_launched': False})
        print(json.dumps({'status': 'cpu_staging_verified_not_activated', 'target': TARGET,
            'receiving_sha256': receipt['receiving_sha256'], 'gpu_calls': 0}))
    except Exception as error:
        q.write(args.output / 'FAILED.json', {'error': str(error), 'partial_files_preserved': True})
        raise


if __name__ == '__main__':
    main()
