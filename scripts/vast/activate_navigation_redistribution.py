"""Root-reviewed v2 activation; preserves append-only authorizations and receipts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import navigation_redistribution_common as c
import navigation_redistribution_stage as stage


def activate(operation):
    if operation not in ('early', 'boundary', 'run'):
        raise ValueError('Unreviewed operation')
    connections, authority = stage.connections()
    marker = 'navigation v2 ' + operation + ' activation'
    if marker not in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text():
        raise ValueError('Exact root activation reservation required')
    files = c.read(stage.PROOF / 'FILES.json')
    if any(c.digest(stage.PROJECT / ('tests' if name.startswith('test_') else 'scripts/vast') / name) != item['sha256']
           for name, item in files.items()):
        raise ValueError('Reviewed operations changed after receiving tests')
    if operation == 'early':
        assignments = {50205763: [None]}
    elif operation == 'boundary':
        assignments = {50231985: [None]}
    else:
        plan = c.read(stage.PROOF / 'PLAN.json'); c.validate_plan(plan)
        cutover = c.read(stage.PROOF / 'CUTOVER.json')
        if cutover['plan_sha256'] != c.digest(stage.PROOF / 'PLAN.json') or not cutover['old_parents_terminal']:
            raise ValueError('Successful reviewed cutover required')
        assignments = {number: [name for name, row in plan['workers'].items() if row['instance'] == number]
                       for number in stage.HOSTS}
    for number, workers in assignments.items():
        ready_sha = c.digest(stage.PROOF / f'READY-{number}.json')
        authorization = {'owner': 'rep_geometry_transcoder/root', 'operation': operation,
            'ready_sha256': ready_sha, 'provider_check': authority,
            'scientific_source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES}
        if operation == 'run':
            authorization['plan_sha256'] = c.digest(stage.PROOF / 'PLAN.json')
        program = '''import json,os,pathlib,subprocess,sys
sys.path.insert(0,sys.argv[1]); import navigation_redistribution_common as c
value=json.loads(sys.stdin.read()); operation=value['operation']
c.verify_members(c.CONTROL,c.read(c.CONTROL/'FILES.json'))
assert c.digest(c.CONTROL/'READY.json')==value['ready_sha256']
if operation=='early':
 assert c.gpu_uuid(0)=='GPU-1541ee72-c8fc-ca75-7fea-470ae793c81e' and not c.gpu_processes(0)
 assert not c.alive({'pid':9472,'starttime':17126225,'command':[]})
 c.write(c.CONTROL/'PROFILE_RELEASE.json',{'owner':'rep_geometry_transcoder/root','gpu_uuid':c.gpu_uuid(0),
  'profiling_finished':True,'bounded_cache_pilot_terminal':True,
  'cache_receipts_sha256':'77f2f4ea0bd16e473f830b6fedb3f17aeceff0666eb31f584ef12ce9c82da5ff',
  'scientific_cache_adoption':False})
if operation=='run':assert c.digest(c.CONTROL/'PLAN.json')==value['plan_sha256']
c.write(c.CONTROL/(operation.upper()+'_AUTHORIZATION.json'),value)
receipts=[]
for worker in json.loads(sys.argv[2]):
 label=operation if worker is None else 'run-'+worker
 command=['/usr/bin/python3','-u',str(c.CONTROL/'navigation_redistribution_control.py'),operation]
 if worker is not None:command+=['--worker',worker]
 with (c.CONTROL/(label+'.log')).open('x') as log:
  environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  child=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  identity=c.started(child.pid,command)
 receipt={'operation':operation,'worker':worker,'identity':identity,'authorization_sha256':c.digest(c.CONTROL/(operation.upper()+'_AUTHORIZATION.json'))}
 c.write(c.CONTROL/(label+'-ACTIVATED.json'),receipt);receipts.append(receipt)
print(json.dumps(receipts))
'''
        output = subprocess.check_output(connections[number] + [shlex.join(['/usr/bin/python3', '-c', program,
            str(c.CONTROL), json.dumps(workers)])], input=json.dumps(authorization).encode(), timeout=60)
        receipts = json.loads(output)
        c.write(stage.PROOF / f'{operation}-ACTIVATED-{number}.json', receipts)
        print(json.dumps({'instance': number, 'activation': receipts}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('early', 'boundary', 'run'))
    activate(parser.parse_args().operation)
