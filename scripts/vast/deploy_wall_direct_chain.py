"""Replace only two waiting CPU helpers; preserve the active scientific process."""
import json
import argparse
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

import component_extension_fleet as fleet


def main():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    assert 'September11 direct fit-to-behavior scheduling repair' in board
    rows = fleet.provider('show', 'instances')
    row = next(r for r in rows if r['id'] == 50632757)
    assert row['label'] == 'jepa-mw-components-0911-slot2-v3' and row['geolocation'] == 'Missouri, US'
    assert row['actual_status'] == 'running' and fleet.account_hourly(rows) <= 7
    local = fleet.BASE / 'direct-wall-chain-v1'
    local.mkdir(exist_ok=False)
    lease = fleet.read(fleet.BASE / 'rentals-v3/slot-02/LEASE.json')
    remote = fleet.REMOTE + '/ops/wall-chain-v1'
    ssh = fleet.connection(row)
    subprocess.run(ssh + ['test ! -e ' + remote + ' && mkdir ' + remote], check=True, timeout=30)
    names = ('component_fit_behavior_chain.py', 'component_boundary_job.py',
             'component_queue_handoff.py', 'refined_panel_queue.py')
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w') as archive:
            for name in names: archive.add(Path(__file__).with_name(name), arcname=name, recursive=False)
        stream.seek(0)
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -x -C ' + remote], stdin=stream, check=True, timeout=30)
    payload = {'deadline': lease['deadline'], 'files': {n: fleet.sha(Path(__file__).with_name(n)) for n in names}}
    script = '''import json,sys,hashlib,os,signal,subprocess,time
from pathlib import Path
r=Path(sys.argv[1]);payload=json.load(sys.stdin);sys.path.insert(0,str(r))
from component_fit_behavior_chain import validate_adoption
from component_boundary_job import validate_job
from component_queue_handoff import process,direct_children,write
for n,digest in payload['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==digest
fit=Path('/workspace/refined-nav-wall-prerequisite-20260911-v2');panel=Path('/workspace/refined-wall-behavior-20260911-v2')
assert not (fit/'FIT_STARTED.json').exists() and not (panel/'fit').exists() and not (panel/'BOUNDARY_LAUNCH.json').exists()
assert (panel/'RUNTIME_READY.json').exists() and json.loads((panel/'RESET_PARITY.json').read_text())['all97_original_inputs_exact']
waiting=json.loads((fit/'WAITING_BOUNDARY.json').read_text());queue=process(waiting['queue_pid']);science=process(waiting['continuing_scientific_pid'])
assert science and science['state'] not in ('Z','X')
spec={'component_root':'/workspace/metaworld-components-20260911-v1','fit_root':str(fit),'panel_root':str(panel),
 'queue_pid':waiting['queue_pid'],'queue_args':queue['args'],'scientific_pid':waiting['continuing_scientific_pid'],
 'scientific_args':science['args'],'scientific_output':waiting['continuing_output'],'deadline':payload['deadline']}
validate_adoption(spec);validate_job(json.loads((fit/'JOB.json').read_text()),fit)
helpers=[(json.loads((fit/'BOUNDARY_LAUNCH.json').read_text())['pid'],str(fit/'ops/component_boundary_job.py')),
 (json.loads((panel/'PREPARATION_LAUNCH.json').read_text())['pid'],str(panel/'ops/prepare_remaining_refined_behavior.py'))]
states=[];stopped=[];retired=False
try:
 for pid,script in helpers:
  item=process(pid);assert item and script in item['args'] and not direct_children(pid)
  states.append({'pid':pid,'args':item['args']});os.kill(pid,signal.SIGSTOP);stopped.append(pid)
 validate_adoption(spec)
 assert not (fit/'FIT_STARTED.json').exists() and not (panel/'fit').exists()
 for pid,_ in helpers:assert not direct_children(pid)
 write(r/'PLAN.json',spec);write(r/'RETIRED_HELPERS.json',{'time':time.time(),'helpers':states,'no_scientific_process_signaled':True})
 for pid,_ in helpers:os.kill(pid,signal.SIGKILL)
 retired=True
 with (r/'chain.log').open('x') as log:
  child=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'component_fit_behavior_chain.py'),'--root',str(r)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 receipt={'pid':child.pid,'time':time.time(),'scientific_pid_unchanged':spec['scientific_pid']}
 write(r/'LAUNCH.json',receipt)
 for _ in range(30):
  if (r/'ADOPTED.json').exists():break
  if child.poll() is not None:raise RuntimeError('Chain failed to adopt')
  time.sleep(1)
 else:raise TimeoutError('Chain adoption not acknowledged')
 print(json.dumps(receipt))
except BaseException:
 if not retired:
  for pid in stopped:
   if process(pid):os.kill(pid,signal.SIGCONT)
 elif not (r/'ADOPTED.json').exists():os.kill(spec['queue_pid'],signal.SIGCONT)
 raise
'''
    result = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', script, remote])],
                                                input=json.dumps(payload), text=True, timeout=60))
    fleet.write(local / 'LAUNCH.json', {**result, 'instance': row['id'], 'files': payload['files']})
    print(json.dumps(result))


def adopt_timer(task):
    assert 'September11 panel timer repair' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    instance, generation, slot, attempt = {'pusht': (50638073, 5, 0, 1), 'pointmaze': (50626847, 1, 1, 3)}[task]
    rows = fleet.provider('show', 'instances'); row = next(r for r in rows if r['id'] == instance)
    local_lease = fleet.BASE / ('rentals' if generation == 1 else f'rentals-v{generation}') / f'slot-{slot:02d}'
    lease = fleet.read(local_lease / 'LEASE.json')
    assert row['label'] == lease['label'] and row['geolocation'] == lease['geolocation'] and row['geolocation'].endswith(', US')
    assert row['actual_status'] == 'running' and fleet.account_hourly(rows) <= 7
    local = fleet.BASE / f'{task}-timer-repair-v1'; local.mkdir(exist_ok=False)
    remote = fleet.REMOTE + f'/ops/{task}-timer-repair-v1'; ssh = fleet.connection(row)
    subprocess.run(ssh + ['test ! -e ' + remote + ' && mkdir ' + remote], check=True, timeout=30)
    names = ('component_fit_behavior_chain.py', 'component_boundary_job.py', 'component_queue_handoff.py', 'refined_panel_queue.py')
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w') as archive:
            for name in names: archive.add(Path(__file__).with_name(name), arcname=name, recursive=False)
        stream.seek(0)
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -x -C ' + remote], stdin=stream, check=True, timeout=30)
    payload = {'deadline': lease['deadline'], 'panel': f'/workspace/refined-{task}-behavior-20260911-v{attempt}',
               'files': {n: fleet.sha(Path(__file__).with_name(n)) for n in names}}
    script = '''import json,sys,hashlib,subprocess,time
from pathlib import Path
r=Path(sys.argv[1]);payload=json.load(sys.stdin);sys.path.insert(0,str(r))
from component_queue_handoff import process,direct_children,write
from component_boundary_job import validate_job
for n,d in payload['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==d
p=Path(payload['panel']);job=json.loads((p/'JOB.json').read_text());validate_job(job,p)
assert (p/'PANEL_STARTED.json').exists() and not (p/'TERMINAL.json').exists()
waiting=json.loads((p/'WAITING_BOUNDARY.json').read_text());queue=process(waiting['queue_pid'])
boundary=json.loads((p/'BOUNDARY_LAUNCH.json').read_text())['pid'] if (p/'BOUNDARY_LAUNCH.json').exists() else None
if boundary is None:
 candidates=[]
 for path in Path('/proc').iterdir():
  if path.name.isdigit():
   item=process(int(path.name))
   if item and str(p/'ops/component_boundary_job.py') in item['args']:candidates.append(int(path.name))
 assert len(candidates)==1;boundary=candidates[0]
state=process(boundary);children=direct_children(boundary)
assert len(children)==1 and str(p/'ops/refined_panel_queue.py') in children[0][1]['args']
panel_pid,panel=children[0]
deadline=min(payload['deadline'],json.loads((p/'PANEL_STARTED.json').read_text())['time']+12*3600)
spec={'component_root':'/workspace/metaworld-components-20260911-v1','panel_root':str(p),
 'queue_pid':waiting['queue_pid'],'queue_args':queue['args'],'boundary_pid':boundary,'boundary_args':state['args'],
 'panel_pid':panel_pid,'panel_args':panel['args'],'deadline':deadline}
assert queue['state'] in ('T','t') and not direct_children(waiting['queue_pid'])
write(r/'PLAN.json',spec)
with (r/'timer.log').open('x') as log:
 child=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'component_fit_behavior_chain.py'),
  '--root',str(r),'--adopt-panel-timer'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'pid':child.pid,'active_panel_pid':panel_pid,'time':time.time(),'deadline':deadline}
write(r/'LAUNCH.json',receipt)
for _ in range(30):
 if (r/'ADOPTED.json').exists():break
 if child.poll() is not None:raise RuntimeError('Timer adoption failed')
 time.sleep(1)
else:raise TimeoutError('Timer adoption unacknowledged')
print(json.dumps(receipt))'''
    result = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', script, remote])],
                                               input=json.dumps(payload), text=True, timeout=60))
    fleet.write(local / 'LAUNCH.json', {**result, 'files': payload['files'], 'instance': instance})
    print(json.dumps(result))


def prioritize_tail():
    assert 'September11 completion-first continuation order' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    row = next(r for r in fleet.provider('show', 'instances') if r['id'] == 50626847)
    assert row['label'] == 'jepa-mw-components-0911-slot1-v1' and row['geolocation'] == 'New Jersey, US' and row['actual_status'] == 'running'
    ssh = fleet.connection(row); remote = fleet.REMOTE + '/ops/continuation-v1'
    source = Path(__file__).with_name('component_completion_tail.py')
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w') as archive:
            archive.add(source, arcname='component_completion_tail_v2.py', recursive=False)
        stream.seek(0)
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -x -C ' + remote], stdin=stream, check=True, timeout=30)
    script = '''import json,sys,hashlib,subprocess,os,signal,time
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r.parent/'pointmaze-timer-repair-v1'))
from component_queue_handoff import process,direct_children,write
assert hashlib.sha256((r/'component_completion_tail_v2.py').read_bytes()).hexdigest()==sys.argv[2]
old=json.loads((r/'PLAN.json').read_text());pid=json.loads((r/'LAUNCH.json').read_text())['pid'];item=process(pid)
assert item and str(r/'component_completion_tail.py') in item['args'] and not direct_children(pid)
assert not (r/'TERMINAL.json').exists() and not (Path(old['component_root'])/'TERMINAL.json').exists()
assert not list(r.glob('COMPLETE-*.json')) and old['extra']==[['reach',5],['reach-wall',5],['reach',7]]
os.kill(pid,signal.SIGSTOP)
try:
 assert not direct_children(pid) and not list(r.glob('COMPLETE-*.json'))
 write(r/'ORDER_AMENDMENT.json',{'time':time.time(),'prior_plan_sha256':hashlib.sha256((r/'PLAN.json').read_bytes()).hexdigest(),
  'extra':sorted(old['extra'],key=lambda item:item[0]!='reach'),'reason':'Finish remaining Reach streams before extra Reach-Wall streams; no outcome-based selection',
  'original_pid':pid,'scientific_process_signaled':False})
 os.kill(pid,signal.SIGKILL)
 with (r/'queue-v2.log').open('x') as log:
  child=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'component_completion_tail_v2.py'),'--root',str(r)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 receipt={'pid':child.pid,'time':time.time(),'ordered_streams':[['reach',5],['reach',7],['reach-wall',5]]}
 write(r/'LAUNCH_V2.json',receipt);print(json.dumps(receipt))
except BaseException:
 if process(pid) and process(pid)['state'] not in ('Z','X'):os.kill(pid,signal.SIGCONT)
 raise
'''
    receipt = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', script, remote, fleet.sha(source)])], text=True, timeout=30))
    fleet.write(fleet.BASE / 'direct-wall-chain-v1' / 'CONTINUATION_ORDER.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--adopt-timer', choices=('pusht', 'pointmaze'))
    parser.add_argument('--prioritize-tail', action='store_true')
    args = parser.parse_args()
    if args.prioritize_tail: prioritize_tail()
    elif args.adopt_timer: adopt_timer(args.adopt_timer)
    else: main()
