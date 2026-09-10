"""Complete only the reviewed v2 post-commit watcher-exit administrative failure."""
import json
from pathlib import Path
import shlex
import subprocess

import navigation_redistribution_common as c
import navigation_redistribution_stage as stage

PLAN = '378cec55b21b84cf3d91b35fe0cfbc4f4b793a0f33b4cc6a726d17f1a4a46a63'
FAILURE = '770f7b1dd0385030bea1b2e22f72ffd2e8afc1dbd1ac6c5ddcd1fe81d7db8b64'

REMOTE = r'''import json,os,pathlib,signal,sys,time
sys.path.insert(0,sys.argv[1]); import navigation_redistribution_common as c
authorization=json.load(sys.stdin)
assert c.digest(c.CONTROL/'PLAN.json')==authorization['plan_sha256']
assert c.digest(c.CONTROL/'boundary/FAILED.json')==authorization['failure_sha256']
assert not (c.CONTROL/'CUTOVER.json').exists()
assert not (c.CONTROL/'workers').exists()
assert not (c.CONTROL/'RUN_AUTHORIZATION.json').exists()
assert not (c.CONTROL/'results').exists()
ready=c.read(c.CONTROL/'READY.json')
assert c.digest(c.CONTROL/'FILES.json')==ready['files_sha256']
c.verify_members(c.CONTROL,c.read(c.CONTROL/'FILES.json'))
c.verify_source()
plan=c.read(c.CONTROL/'PLAN.json'); c.validate_plan(plan)
completed,pristine=c.inventory()
assert completed==plan['completed']
assert {c.key(j) for w in plan['workers'].values() for j in w['jobs']}=={c.key(j) for j in pristine}
assert c.digest(c.CONTROL/'READY.json')==plan['ready_sha256']['50231985']
watcher=ready['watcher']; current=c.process(watcher['pid'])
assert current is None or (current['starttime']==watcher['starttime'] and current['state'] in ('Z','X'))
failed=c.read(c.CONTROL/'boundary/FAILED.json')
assert failed['ownership_committed'] and failed['error']=='Bound process command changed'
root=c.CONTROL/'cutover-recovery';root.mkdir(exist_ok=False)
c.write(root/'AUTHORIZATION.json',authorization)
handles=[]; observations=[]
try:
 # Bind BOTH stopped, childless producers and physical devices before signaling.
 for name in ('ne0','ne1'):
  worker=ready['workers'][name]; expected=worker['parent']; current=c.process(expected['pid'])
  assert current is not None and current['starttime']==expected['starttime'] and current['command']==expected['command'] and current['state'] in ('T','t')
  assert c.gpu_uuid(worker['gpu'])==worker['gpu_uuid'] and not c.gpu_rows(worker['gpu'])
  c.wait_gpu_release(worker['gpu'],worker['gpu_uuid'],paused_parent=expected)
  ids=(pathlib.Path('/proc')/str(expected['pid'])/'task'/str(expected['pid'])/'children').read_text().split()
  for value in ids:
   child=c.process(int(value));assert child is None or child['state'] in ('Z','X')
  descriptor=os.pidfd_open(expected['pid'])
  current=c.process(expected['pid'])
  assert current is not None and current['starttime']==expected['starttime'] and current['command']==expected['command'] and current['state'] in ('T','t')
  handles.append((descriptor,expected))
 c.write(root/'SIGNAL_INTENT.json',{'parents':[v for _,v in handles],'signals':['SIGTERM','SIGCONT'],'children_not_signaled':True})
 for descriptor,expected in handles:
  current=c.process(expected['pid'])
  assert current is not None and current['starttime']==expected['starttime'] and current['command']==expected['command'] and current['state'] in ('T','t')
  ids=(pathlib.Path('/proc')/str(expected['pid'])/'task'/str(expected['pid'])/'children').read_text().split()
  for value in ids:
   child=c.process(int(value));assert child is None or child['state'] in ('Z','X')
 assert not c.gpu_rows(0) and not c.gpu_rows(1)
 for descriptor,expected in handles:
  current=c.process(expected['pid'])
  assert current is not None and current['starttime']==expected['starttime'] and current['command']==expected['command'] and current['state'] in ('T','t')
  # One pidfd guarantees these two signals can never target a reused numericPID.
  signal.pidfd_send_signal(descriptor,signal.SIGTERM)
  signal.pidfd_send_signal(descriptor,signal.SIGCONT)
 for descriptor,expected in handles:
  deadline=time.monotonic()+30
  while True:
   current=c.process(expected['pid']);observations.append({'pid':expected['pid'],'observed':current})
   if current is None:break
   assert current['starttime']==expected['starttime']
   if current['state'] in ('Z','X'):break
   # Exit may briefly expose emptyargv before the state becomes zombie. No new
   # signals are authorized by this observation, and a changed nonemptyargv fails.
   assert not current['command'] or current['command']==expected['command']
   if time.monotonic()>deadline:raise TimeoutError('Original producer did not become terminal')
   time.sleep(.05)
 for name in ('ne0','ne1'):
  worker=ready['workers'][name]
  c.wait_gpu_release(worker['gpu'],worker['gpu_uuid'])
 assert c.digest(c.CONTROL/'PLAN.json')==authorization['plan_sha256']
 receipt={'status':'exact_committed_partition_original_producers_terminal',
  'plan_sha256':authorization['plan_sha256'],'preserved_failure_sha256':authorization['failure_sha256'],
  'old_watcher_terminal':True,'old_parents_terminal':True,'children_not_signaled':True,
  'original_producers_never_resumed_for_scheduling':True,'exit_observations':observations}
 c.write(root/'DONE.json',receipt)
 c.write(c.CONTROL/'CUTOVER.json',{'plan_sha256':authorization['plan_sha256'],
  'old_parents_terminal':True,'old_watcher_terminal':True,'old_children_not_interrupted':True,
  'completed_streams':len(completed),'assigned_streams':len(pristine),
  'administrative_recovery_sha256':c.digest(root/'DONE.json'),
  'original_boundary_failure_preserved':True})
 print(json.dumps({'cutover_sha256':c.digest(c.CONTROL/'CUTOVER.json'),**receipt}))
except BaseException as error:
 c.write(root/'FAILED.json',{'error':str(error),'ownership_remains_committed':True,'observations':observations})
 raise
finally:
 for descriptor,_ in handles:os.close(descriptor)
'''


def main():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    if 'navigation v2 exact committed-cutover recovery' not in board:
        raise ValueError('New explicit root recovery authorization required')
    connections, provider = stage.connections()
    lifecycle_test = r'''import os,signal,subprocess,time,json,sys
sys.path.insert(0,sys.argv[1]);import navigation_redistribution_common as c
command=['/usr/bin/python3','-c','import time;time.sleep(30)']
child=subprocess.Popen(command);descriptor=None
try:
 identity=c.started(child.pid,command)
 descriptor=os.pidfd_open(child.pid)
 signal.pidfd_send_signal(descriptor,signal.SIGSTOP)
 deadline=time.monotonic()+3
 while c.process(child.pid)['state'] not in ('T','t'):
  assert time.monotonic()<deadline;time.sleep(.01)
 assert c.process(child.pid)['starttime']==identity['starttime']
 signal.pidfd_send_signal(descriptor,signal.SIGTERM)
 signal.pidfd_send_signal(descriptor,signal.SIGCONT)
 assert child.wait(timeout=3)==-signal.SIGTERM
 proof={'status':'owned_cpu_term_cont_pidfd_lifecycle_passed','pid':child.pid,'identity':identity,'gpu_calls':0,'scientific_processes_signaled':False}
 c.write(c.CONTROL/'ROOT_RECOVERY_CPU_TEST.json',proof);print(json.dumps(proof))
finally:
 if descriptor is not None:os.close(descriptor)
 if child.poll() is None:child.kill();child.wait()
'''
    test_output = subprocess.check_output(connections[50231985] + [shlex.join(['/usr/bin/python3', '-c', lifecycle_test, str(c.CONTROL)])], text=True, timeout=30)
    c.write(stage.PROOF / 'ROOT_RECOVERY_CPU_TEST.json', json.loads(test_output))
    authority = {'owner': 'rep_geometry_transcoder/root', 'plan_sha256': PLAN,
        'failure_sha256': FAILURE, 'provider': provider, 'operation': 'finish_exact_committed_cutover',
        'script_sha256': c.digest(Path(__file__)),
        'receiving_program_sha256': __import__('hashlib').sha256(REMOTE.encode()).hexdigest()}
    output = subprocess.check_output(connections[50231985] + [shlex.join(['/usr/bin/python3', '-c', REMOTE, str(c.CONTROL)])],
        input=json.dumps(authority), text=True, timeout=120)
    receipt = json.loads(output)
    c.write(stage.PROOF / 'ROOT_CUTOVER_RECOVERY.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
