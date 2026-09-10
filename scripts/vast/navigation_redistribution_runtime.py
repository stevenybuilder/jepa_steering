"""Root-approved absent-only TX OSMesa repair; no GPU or existing library edits."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import navigation_redistribution_common as c
import navigation_redistribution_stage as stage

LIBRARIES = ('libOSMesa.so.8', 'libglapi.so.0', 'libLLVM-15.so.1', 'libdrm.so.2')
OBJECT = ('payload/maze-python/lib/python3.10/site-packages/mujoco_py/generated/'
          '_pyxbld_2.1.2.14_310_linuxcpuextensionbuilder/temp.linux-x86_64-3.10/workspace/'
          'jepa-maze-python/lib/python3.10/site-packages/mujoco_py/cymj.o')
ORIGINAL_OBJECT = '5bfe4405a5643b28897822e1c425110708f7dd17bc51a754c3a364a99d0f726c'
FAILED_OBJECT = '27b5fab991a47f9c56ed9f02eb46ba5c41fe788bde32ca27896f6ec415987baf'


def main():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    if 'libOSMesa' not in board or '/opt/conda/lib' not in board:
        raise ValueError('Narrow public runtime repair authorization required')
    ssh, authority = stage.connections()
    proof = stage.PROOF / 'runtime-repair'; proof.mkdir(exist_ok=False)
    c.write(proof / 'AUTHORITY.json', authority)
    remote = c.CONTROL / 'runtime-repair'
    inventory = '''import hashlib,json,pathlib,sys
result={}
for name in json.loads(sys.argv[1]):
 p=pathlib.Path('/usr/lib/x86_64-linux-gnu')/name
 result[name]={'resolved_source':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(result,sort_keys=True))
'''
    manifest = json.loads(stage.get(ssh[50231985], ['/usr/bin/python3', '-c', inventory, json.dumps(LIBRARIES)]))
    c.write(proof / 'LIBRARIES.json', manifest)
    check = '''import pathlib,sys,json
for name in json.loads(sys.argv[1]):
 for prefix in ('/opt/conda/lib','/usr/lib/x86_64-linux-gnu','/lib/x86_64-linux-gnu'):
  p=pathlib.Path(prefix)/name
  if p.exists() or p.is_symlink():raise ValueError('Never replace/shadow existing library '+str(p))
pathlib.Path(sys.argv[2]).mkdir(exist_ok=False)
'''
    subprocess.run(ssh[50259194] + [shlex.join(['/usr/bin/python3', '-c', check, json.dumps(LIBRARIES), str(remote)])], check=True)
    receive = '''import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); wanted=json.loads(sys.argv[2])
with p.open('xb') as f:
 while True:
  b=sys.stdin.buffer.read(4<<20)
  if not b:break
  f.write(b)
if p.stat().st_size!=wanted['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted['sha256']:raise ValueError('Runtime bytes changed')
p.chmod(0o644)
'''
    for name, wanted in manifest.items():
        with tempfile.TemporaryFile() as packet:
            subprocess.run(ssh[50231985] + [shlex.join(['cat', '/usr/lib/x86_64-linux-gnu/' + name])], stdout=packet, check=True)
            packet.seek(0)
            subprocess.run(ssh[50259194] + [shlex.join(['/usr/bin/python3', '-c', receive,
                '/opt/conda/lib/' + name, json.dumps(wanted)])], stdin=packet, check=True)
        print(json.dumps({'copied_absent_library': name, 'sha256': wanted['sha256'], 'gpu_calls': 0}), flush=True)
    if json.loads(stage.get(ssh[50231985], ['/usr/bin/python3', '-c', inventory, json.dumps(LIBRARIES)])) != manifest:
        raise ValueError('Public source libraries changed during transfer')
    original = c.read(stage.PROOF / 'INPUTS.json')[OBJECT]
    if original['sha256'] != ORIGINAL_OBJECT:
        raise ValueError('Original compiled object binding changed')
    with tempfile.TemporaryFile() as packet:
        subprocess.run(ssh[50231985] + [shlex.join(['cat', str(c.CONTROL / OBJECT)])], stdout=packet, check=True)
        packet.seek(0)
        subprocess.run(ssh[50259194] + [shlex.join(['/usr/bin/python3', '-c', receive,
            str(remote / 'original-cymj.o'), json.dumps(original)])], stdin=packet, check=True)
    restore = '''import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]); target=root/sys.argv[2]; repair=root/'runtime-repair'
if hashlib.sha256(target.read_bytes()).hexdigest()!=sys.argv[3]:raise ValueError('Unexpected changed build object')
if hashlib.sha256((repair/'original-cymj.o').read_bytes()).hexdigest()!=sys.argv[4]:raise ValueError('Restoration bytes differ')
target.rename(repair/'failed-rebuild-cymj.o')
(repair/'original-cymj.o').rename(target)
'''
    subprocess.run(ssh[50259194] + [shlex.join(['/usr/bin/python3', '-c', restore, str(c.CONTROL), OBJECT,
        FAILED_OBJECT, ORIGINAL_OBJECT])], check=True)
    verify = '''import hashlib,json,pathlib,subprocess,sys
sys.path.insert(0,sys.argv[1]); import navigation_redistribution_common as c
manifest=json.loads(sys.argv[2])
for name,wanted in manifest.items():
 p=pathlib.Path('/opt/conda/lib')/name
 assert p.stat().st_size==wanted['bytes'] and c.digest(p)==wanted['sha256']
c.verify_members(c.CONTROL,c.read(c.CONTROL/'INPUTS.json'))
extension=pathlib.Path('/workspace/jepa-maze-python/lib/python3.10/site-packages/mujoco_py/generated/cymj_2.1.2.14_310_linuxcpuextensionbuilder_310.so')
assert c.digest(extension)=='5def61ce98c4da003907e7033e44496bd0c3db6d21430e525369dbaee621dfce'
ldd=subprocess.check_output(['ldd',str(extension)],env=c.environment(50259194),text=True)
assert 'not found' not in ldd
proof={'status':'absent_only_public_runtime_repair_verified','libraries':manifest,'ldd':ldd,'original_input_manifest_reverified':True,'original_extension_unchanged':True,'failed_generated_object_preserved':str(c.CONTROL/'runtime-repair/failed-rebuild-cymj.o'),'gpu_calls':0,'existing_libraries_overwritten':False}
c.write(c.CONTROL/'runtime-repair/VERIFIED.json',proof)
print(json.dumps(proof))
'''
    result = json.loads(stage.get(ssh[50259194], ['/usr/bin/python3', '-c', verify, str(c.CONTROL), json.dumps(manifest)]))
    c.write(proof / 'VERIFIED.json', result)
    # Original CPU preparation is unchanged. Add the newly required runtime
    # supplement to its append-only READY at first publication (no prior TXREADY).
    prepare = '''import sys
sys.path.insert(0,sys.argv[1]); import navigation_redistribution_common as c
import navigation_redistribution_control as control
original=c.write
def write(path,value):
 if path==c.CONTROL/'READY.json':value={**value,'runtime_supplement_sha256':c.digest(c.CONTROL/'runtime-repair/VERIFIED.json')}
 original(path,value)
c.write=write
control.prepare(50259194)
'''
    print(stage.get(ssh[50259194], ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
        '/usr/bin/python3', '-c', prepare, str(c.CONTROL)]), flush=True)
    ready = json.loads(stage.get(ssh[50259194], ['cat', str(c.CONTROL / 'READY.json')]))
    c.write(stage.PROOF / 'READY-50259194.json', ready)
    all_ready = {str(n): c.read(stage.PROOF / f'READY-{n}.json') for n in stage.HOSTS}
    c.write(stage.PROOF / 'ALL_READY.json', all_ready)
    for number in stage.HOSTS:
        stage.put_json(ssh[number], c.CONTROL / 'ALL_READY.json', all_ready)
    c.write(stage.PROOF / 'DONE.json', {'status': 'navigation_redistribution_cpu_staged_tested_ready_for_root_review',
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES, 'files_sha256': c.digest(stage.PROOF / 'FILES.json'),
        'all_ready_sha256': c.digest(stage.PROOF / 'ALL_READY.json'), 'runtime_supplement_sha256': c.digest(proof / 'VERIFIED.json'),
        'parent_signals': 0, 'gpu_calls': 0})


if __name__ == '__main__':
    main()
