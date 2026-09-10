"""Activate the three reviewed CPU waiters; never alter their frozen training code.

One-shot operational entrypoint. On any partial failure, inspect and preserve the
receipts rather than retrying into the same output directory. GPU ownership passes
only inside the frozen queue after its complete navigation dependency verifies.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import corrected_pointmaze_queue as q
from corrected_pointmaze_stage import KEY, ROOT, TARGET, connection


PLAN_SHA = '6554fb7e9a3b4e47443fa55eacaa5b05b9d1e79a8604c7450425692e789efa3d'
READY_SHA = '281efdb80d717de5ee7b668f7e4a1ea552e226d844318900e39c4ffde32a6ce8'
ACTIVATE = r'''import hashlib,importlib.util,json,os,pathlib,shutil,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); expected_plan=sys.argv[2]; expected_ready=sys.argv[3]
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Parent must hide CUDA')
def digest(path):
 result=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4<<20),b''):result.update(block)
 return result.hexdigest()
if digest(root/'PLAN.json')!=expected_plan or digest(root/'PLAN_RECEIVING_READY.json')!=expected_ready:
 raise ValueError('Root-reviewed plan/receiving proof changed')
ready=json.loads((root/'PLAN_RECEIVING_READY.json').read_text())
if digest(root/'SOURCE_FILES.json')!=ready['source_manifest_sha256']:
 raise ValueError('Frozen source manifest changed')
manifest=json.loads((root/'SOURCE_FILES.json').read_text())
if 'code/scripts/vast/corrected_pointmaze_queue.py' not in manifest:
 raise ValueError('Queue source is not bound')
for name,row in manifest.items():
 path=root/name
 if 'symlink' in row:
  if not path.is_symlink() or os.readlink(path)!=row['symlink']:raise ValueError('Source link changed')
 elif path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
  raise ValueError('Frozen source member changed: '+name)
script=root/'code/scripts/vast/corrected_pointmaze_queue.py'
spec=importlib.util.spec_from_file_location('corrected_pointmaze_queue',script)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
plan=q.read(root/'PLAN.json')
q.require_fields(ready,{'plan_sha256':expected_plan,'static_verification_passed':True,
 'all_seed_plan_validation_passed':True,'queue_launched':False,'instance':50259194},'Receiving proof')
q.verify_static(plan)
for seed in q.SEEDS:
 item=q.validate_plan(plan,seed); before=item['predecessor']
 if item['instance']!=50259194 or q.digest(before['launch'])!=before['launch_sha256']:
  raise ValueError('Actual predecessor launch changed')
 if not q.predecessor_pending(before):q.verify_predecessor(before,plan)
if pathlib.Path(plan['output_root']).exists():raise ValueError('Do not duplicate corrected histories')
free=shutil.disk_usage(root).free
if free<36*1024**3:raise ValueError('Insufficient receiving disk')
activation=root/'root-activation'
activation.mkdir(exist_ok=False)
q.write(activation/'AUTHORIZATION.json',{'owner':'rep_geometry_transcoder/root',
 'plan_sha256':expected_plan,'receiving_ready_sha256':expected_ready,
 'source_sha256':plan['source_sha256'],'seeds':q.SEEDS,'disk_free_bytes':free,
 'execution':'native_accumulation_uncached','duplicate_launch_forbidden':True,
 'gpu_owner_until_completion':'DROID then complete assigned navigation streams',
 'gpu_calls_by_launcher':0})
launches=[]
try:
 for seed in q.SEEDS:
  command=['/usr/bin/python3','-u',str(script),'--plan',str(root/'PLAN.json'),
   '--plan-sha256',expected_plan,'--seed',str(seed)]
  with (activation/('seed-'+str(seed)+'.log')).open('x') as log:
   child=subprocess.Popen(command,env=q.environment(plan),stdin=subprocess.DEVNULL,
    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row={'seed':seed,'pid':child.pid,'start_ticks':None,'argv':command,'identity_pending':True}
  launches.append(row)
  q.write(activation/('seed-'+str(seed)+'-SPAWNED.json'),row)
  identity=q.process_identity(child.pid)
  if identity is None or identity['argv']!=command or identity['state']=='Z':
   raise ValueError('CPU waiter failed at launch: '+str(seed))
  row.update(start_ticks=identity['start_ticks'],identity_pending=False)
  q.write(activation/('seed-'+str(seed)+'-PROCESS.json'),row)
 for row in launches:
  queue=pathlib.Path(plan['output_root'])/('seed-'+str(row['seed']))/'queue'
  deadline=time.monotonic()+180
  while not (queue/'LAUNCH.json').exists():
   identity=q.process_identity(row['pid'])
   if ((queue/'FAILED.json').exists() or identity is None or identity['state']=='Z' or
    identity['start_ticks']!=row['start_ticks'] or identity['argv']!=row['argv']):
    raise ValueError('CPU receiving test/queue failed: '+str(row['seed']))
   if time.monotonic()>deadline:raise TimeoutError('CPU waiter launch proof missing')
   time.sleep(1)
  q.require_fields(q.read(queue/'LAUNCH.json'),{'pid':row['pid'],'seed':row['seed'],
   'plan_sha256':expected_plan,'gpu_job_started':False},'Actual CPU waiter launch')
  if (queue/'FAILED.json').exists():raise ValueError('Queue failed after launch')
  row['queue_launch_sha256']=q.digest(queue/'LAUNCH.json')
  row['prepared_sha256']=q.digest(queue/'PREPARED.json')
 q.write(activation/'ACTIVATED.json',{'status':'three_corrected_pointmaze_cpu_waiters_activated',
  'plan_sha256':expected_plan,'source_sha256':plan['source_sha256'],'launches':launches,
  'gpu_calls_by_launcher':0,'scientific_cache_enabled':False})
 print(json.dumps(q.read(activation/'ACTIVATED.json')))
except Exception as error:
 q.write(activation/'FAILED.json',{'error':str(error),'launched_processes':launches,
  'preserved_no_signals_or_retries':True})
 raise
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior activation; do not rerun')
    host, port, total = connection()
    args.output.mkdir(parents=True)
    q.write(args.output / 'INTENT.json', {'instance': TARGET, 'plan_sha256': PLAN_SHA,
        'ready_sha256': READY_SHA, 'aggregate_hourly': total,
        'launcher_sha256': q.digest(__file__), 'program': ACTIVATE})
    ssh = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-p', str(port), 'root@' + host]
    result = subprocess.check_output(ssh + [shlex.join(['env', 'CUDA_VISIBLE_DEVICES=',
        'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-c', ACTIVATE, ROOT, PLAN_SHA, READY_SHA])],
        text=True, timeout=600)
    receipt = json.loads(result.strip().splitlines()[-1])
    q.write(args.output / 'ACTIVATED.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
