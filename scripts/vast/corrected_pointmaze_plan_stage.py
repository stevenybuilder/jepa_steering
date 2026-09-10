"""CPU-only publication of the actual-navigation-bound PointMaze plan.

No queue entrypoint is called: even its prepare-only mode would consume a future
training output directory. This operation instead runs its pure static checks.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import corrected_pointmaze_queue as q
from corrected_pointmaze_stage import connection, KEY, ROOT, TARGET


RECEIVE = r'''import hashlib,importlib.util,json,os,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); wanted=json.loads(sys.argv[2]); began=time.monotonic()
stage=root/'final-plan-receiving'
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be hidden')
if sha(root/'SOURCE_FILES.json')!='1ba732e6cf1c438806afd1570577635d2f9c2a879458b9c9d679f80a8807335f':
 raise ValueError('Previously verified immutable source manifest changed')
if sha(root/'RECEIVING.json')!='4f1ce79540b06355d5b1d32a064e74295c32d63e5a21eea7aca5d0d44e682119':
 raise ValueError('Previously reviewed initial CPU receiving proof changed')
manifest=json.loads((root/'SOURCE_FILES.json').read_text())
for name,row in manifest.items():
 path=root/name
 if 'symlink' in row:
  if not path.is_symlink() or os.readlink(path)!=row['symlink']:raise ValueError('Source link changed')
 elif path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
  raise ValueError('Previously staged source/test changed: '+name)
script=root/'code/scripts/vast/corrected_pointmaze_queue.py'
spec=importlib.util.spec_from_file_location('corrected_pointmaze_queue',script)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
for name,sha in wanted.items():
 if name not in ('PLAN.json','FINALIZED.json') or q.digest(stage/name)!=sha:
  raise ValueError('Received final plan member changed')
plan=q.read(stage/'PLAN.json'); proof=q.read(stage/'FINALIZED.json')
q.require_fields(proof,{'plan_sha256':wanted['PLAN.json'],'source_sha256':plan['source_sha256'],
 'source_commit':plan['source_commit'],'gpu_calls':0},'Local finalization')
q.verify_static(plan)
assignments=[]
for seed in q.SEEDS:
 item=q.validate_plan(plan,seed); before=item['predecessor']
 pending=q.predecessor_pending(before)
 if not pending:q.verify_predecessor(before,plan)
 assignments.append({'seed':seed,'gpu':item['gpu'],'gpu_uuid':item['gpu_uuid'],
  'navigation_pid':before['pid'],'navigation_start_ticks':before['start_ticks'],
  'navigation_argv':before['argv'],'navigation_pending':pending,
  'required_navigation_endpoints':before['expected_done']['endpoints'],
  'navigation_launch_sha256':q.digest(before['launch'])})
if pathlib.Path(plan['output_root']).exists():raise ValueError('Training output already exists')
env=q.environment(plan)
if env['CUDA_VISIBLE_DEVICES']!='':raise ValueError('CPU test environment exposes CUDA')
log=stage/'actual-upstream-sampler-tests.log'
with log.open('x') as stream:
 subprocess.run([plan['python'],'-m','unittest','discover','-s',str(root/'code/tests'),
  '-p','test_native_training_sampler.py','-v'],env=env,stdin=subprocess.DEVNULL,
  stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
q.verify_static(plan)
# Hard links publish existing verified bytes without an overwrite window.
for name in ('PLAN.json','FINALIZED.json'):os.link(stage/name,root/name)
receipt={'status':'corrected_pointmaze_actual_plan_received_static_verified_not_activated',
 'instance':50259194,'plan_sha256':q.digest(root/'PLAN.json'),
 'finalized_sha256':q.digest(root/'FINALIZED.json'),'source_commit':plan['source_commit'],
 'source_sha256':plan['source_sha256'],'source_manifest_sha256':q.digest(root/'SOURCE_FILES.json'),
 'initial_receiving_sha256':q.digest(root/'RECEIVING.json'),'source_members_verified':len(manifest),
 'fixed_files_verified':len(plan['fixed_files']),'assignments':assignments,
 'actual_upstream_test_log_sha256':q.digest(log),'static_verification_passed':True,
 'all_seed_plan_validation_passed':True,'gpu_calls':0,'queue_launched':False,
 'training_outputs_created':False,'existing_runtime_or_library_files_modified':False,
 'seconds':time.monotonic()-began}
q.write(root/'PLAN_RECEIVING_READY.json',receipt)
print(json.dumps({'ready_sha256':q.digest(root/'PLAN_RECEIVING_READY.json'),**receipt}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous publication receipts')
    names = ('PLAN.json', 'FINALIZED.json')
    expected = {name: q.digest(args.finalized / name) for name in names}
    plan = q.read(args.finalized / 'PLAN.json')
    for seed in q.SEEDS:
        q.validate_plan(plan, seed)
    host, port, total = connection()
    ssh = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-p', str(port), 'root@' + host]
    scp = ['scp', '-q', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', str(port)]
    args.output.mkdir(parents=True)
    q.write(args.output / 'INTENT.json', {'target': TARGET, 'root': ROOT,
        'files_sha256': expected, 'aggregate_hourly': total, 'gpu_calls': 0,
        'queue_launch_authorized': False, 'receiving_program': RECEIVE})
    initialize = ('from pathlib import Path;root=Path(' + repr(ROOT) + ');'
        'assert root.is_dir();assert not (root/"PLAN.json").exists();'
        'assert not (root/"PLAN_RECEIVING_READY.json").exists();'
        '(root/"final-plan-receiving").mkdir(exist_ok=False)')
    subprocess.run(ssh + [shlex.join(['/usr/bin/python3', '-c', initialize])], check=True, timeout=30)
    subprocess.run(scp + [str(args.finalized / name) for name in names] +
        ['root@' + host + ':' + ROOT + '/final-plan-receiving/'], check=True, timeout=60)
    result = subprocess.check_output(ssh + [shlex.join(['env', 'CUDA_VISIBLE_DEVICES=',
        'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-c', RECEIVE, ROOT,
        json.dumps(expected)])], text=True, timeout=600)
    receipt = json.loads(result.strip().splitlines()[-1])
    readback = {'PLAN.json': expected['PLAN.json'], 'FINALIZED.json': expected['FINALIZED.json'],
        'PLAN_RECEIVING_READY.json': receipt['ready_sha256'],
        'final-plan-receiving/actual-upstream-sampler-tests.log': receipt['actual_upstream_test_log_sha256']}
    for name, sha in readback.items():
        local = args.output / Path(name).name
        subprocess.run(scp + ['root@' + host + ':' + ROOT + '/' + name, str(local)], check=True, timeout=30)
        if q.digest(local) != sha:
            raise ValueError('Readback differs: ' + name)
    q.write(args.output / 'VERIFIED.json', {'status': 'final_pointmaze_plan_and_ready_readback_verified_not_activated',
        'target': TARGET, 'root': ROOT, 'plan_sha256': expected['PLAN.json'],
        'ready_sha256': receipt['ready_sha256'], 'gpu_calls': 0, 'queue_launched': False})
    print(json.dumps({'status': 'cpu_plan_staging_verified_not_activated',
        'plan_sha256': expected['PLAN.json'], 'ready_sha256': receipt['ready_sha256'], 'gpu_calls': 0}))


if __name__ == '__main__':
    main()
