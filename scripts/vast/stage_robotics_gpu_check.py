"""Stage/activate one bounded native32 check after Indiana navigation completes.

Only a new isolated source snapshot and CPU waiter are created. No rentals,
predecessor signals, dataset downloads, active-source edits or history launches.
Preserve any failed/partially observed attempt and inspect its exact handles.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

import navigation_redistribution_stage as navigation

PROJECT = Path(__file__).resolve().parents[2]
ROOT = "/workspace/jepa-runtime/robotics-training-gpu-check-20260908-v1"
LOCAL = PROJECT / "artifacts/offline_study/robotics-training-gpu-check-20260908-v1"
VENDOR = "/workspace/jepa_steering/vendor/jepa-wms"
BATCH = "/workspace/jepa-runtime/robotics-training-batch-20260908-v1/evidence"
PYTHON = "/workspace/jepa-planning-python/bin/python"
OVERLAY = ("/workspace/jepa-planning-python/lib/python3.10/site-packages:"
           "/workspace/jepa-python/lib/python3.10/site-packages")
UUID = "GPU-1541ee72-c8fc-ca75-7fea-470ae793c81e"
QUEUE = ROOT + "/scripts/vast/robotics_gpu_check_queue.py"
FIXED_INPUTS = {
    BATCH + "/batch.pt": "2ada0bc241424e26899300b77da849f38c9c056ecab14295c6c99be27526bcc4",
    BATCH + "/report.json": "4b14268d293bb708ad8b96c074624a4d52c4b70ca57988c9858273a8eaa1b7aa",
    BATCH + "/protocol.json": "a5975c987b5cd614de7b56e1943f3927f0365ccdd0175dd5e62c4a36fa01e0ef",
    BATCH + "/comparisons.json": "684e69bb22ff4f512cf8074c7faeb50b38a0efedfc5a1e19aed69b245f4d2e80",
    BATCH + "/selected_inputs.json": "58d857c202fe88dd543c6ce8299c30ef9a19558a3b138a447f5941a296144105",
}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def connection():
    board = Path("/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md").read_text()
    if ROOT not in board or "native32 bounded GPU check" not in board:
        raise ValueError("Exact board reservation required")
    hosts, authority = navigation.connections()
    return hosts[50205763], authority


def remote(ssh, program, *args, timeout=60):
    argv = ["env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1",
            "/usr/bin/python3", "-c", program, *map(str, args)]
    return subprocess.check_output(ssh + [shlex.join(argv)], text=True, timeout=timeout)


def make_manifest(members):
    return {name: {"bytes": len(data), "sha256": digest(data)} for name, data in members.items()}


def science_hash(members):
    value = hashlib.sha256()
    paths = sorted(name for name in members if Path(name).parent == Path("src/offline_study")
                   and name.endswith(".py"))
    if not paths:
        raise ValueError("Missing source package")
    for name in paths:
        value.update(Path(name).name.encode() + b"\0" + members[name])
    return value.hexdigest()


def _stage():
    ssh, authority = connection()
    LOCAL.mkdir(exist_ok=False)
    paths = sorted((PROJECT / "src/offline_study").rglob("*.py"))
    paths += sorted((PROJECT / "tests").glob("test_robotics_training_*.py"))
    paths += [PROJECT / "scripts/vast/robotics_gpu_check_queue.py",
              PROJECT / "tests/test_robotics_gpu_check_queue.py"]
    members = {str(path.relative_to(PROJECT)): path.read_bytes() for path in paths}
    manifest = make_manifest(members)
    write(LOCAL / "SOURCE_FILES.json", manifest)
    write(LOCAL / "INTENT.json", {"instance": 50205763, "root": ROOT,
        "authority": authority, "source_manifest_sha256": digest(encoded(manifest)),
        "gpu_calls": 0, "full_history_authorized": False,
        "navigation_assignment_required": 180, "batch_downloaded": False})
    # Read existing small receipts/runtime and hash the existing batch in place.
    inspect = '''import hashlib,json,os,pathlib,shutil,sys
expected=json.loads(sys.argv[1]); result={}
for name,wanted in expected.items():
 p=pathlib.Path(name); h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 row={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 if p.is_symlink():row['symlink']=os.readlink(p)
 if wanted is not None and row['sha256']!=wanted:raise ValueError('Pinned input differs: '+name)
 result[name]=row
if shutil.disk_usage('/workspace/jepa-runtime').free<4*(1<<30):raise ValueError('Receiving disk reserve')
print(json.dumps(result))
'''
    expected = {**FIXED_INPUTS, BATCH + "/DONE.json": None, PYTHON: None}
    fixed = json.loads(remote(ssh, inspect, json.dumps(expected)))
    if json.loads(remote(ssh, "import json,pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
                         BATCH + "/DONE.json"))["report_sha256"] != FIXED_INPUTS[BATCH + "/report.json"]:
        raise ValueError("First-batch completion changed")
    plan = {"schema": 1, "root": ROOT, "source_sha256": science_hash(members),
        "source_manifest": {"path": ROOT + "/SOURCE_FILES.json", "sha256": digest(encoded(manifest))},
        "fixed_files": fixed, "runtime": {"python": PYTHON, "overlay": OVERLAY,
            "ld_library_path": "/root/.mujoco/mujoco210/bin:/opt/conda/lib"},
        "batch_root": BATCH, "vendor": VENDOR, "instance": 50205763, "gpu": 0, "gpu_uuid": UUID}
    write(LOCAL / "PLAN.json", plan)
    # Staging does not itself create a runnable authorization or CPU waiter.
    all_members = {**members, "SOURCE_FILES.json": encoded(manifest), "PLAN.json": encoded(plan)}
    create = '''import pathlib,sys
root=pathlib.Path(sys.argv[1]); root.mkdir(exist_ok=False)
(root/'vendor').mkdir(); (root/'vendor/jepa-wms').symlink_to(sys.argv[2],target_is_directory=True)
'''
    remote(ssh, create, ROOT, VENDOR)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, data in all_members.items():
                info = tarfile.TarInfo(name); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        stream.seek(0)
        subprocess.run(ssh + [shlex.join(["tar", "--keep-old-files", "-C", ROOT, "-xzf", "-"])],
                       stdin=stream, check=True, timeout=60)
    verify = '''import base64,hashlib,json,pathlib,re,subprocess,sys,os
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if hashlib.sha256((root/'SOURCE_FILES.json').read_bytes()).hexdigest()!=expected:raise ValueError('Manifest differs')
manifest=json.loads((root/'SOURCE_FILES.json').read_text())
for name,row in manifest.items():
 p=root/name
 if p.is_symlink() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source differs: '+name)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'src')+':'+sys.argv[4],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.pop('LD_PRELOAD',None)
with (root/'receiving-tests.log').open('x') as log:
 subprocess.run([sys.argv[3],'-m','unittest','discover','-s',str(root/'tests'),'-p','test_robotics*.py','-v'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
data=(root/'receiving-tests.log').read_bytes()
if not re.search(rb'Ran 109 tests in ',data) or not re.search(rb'OK \\(skipped=1\\)\\s*$',data):
 raise ValueError('Receiving test count/result differs from reviewed109; only archival-readback fixture may skip')
print(json.dumps({'source_manifest_sha256':expected,'tests_sha256':hashlib.sha256(data).hexdigest(),
 'tests':109,'skipped':1,'skip_basis':'local archival readback not staged; original remote inputs are separately hash-bound',
 'tests_log_base64':base64.b64encode(data).decode(),'source_files':len(manifest),'gpu_calls':0}))
'''
    result = json.loads(remote(ssh, verify, ROOT, digest(encoded(manifest)), PYTHON, OVERLAY, timeout=360))
    log = base64.b64decode(result.pop("tests_log_base64"), validate=True)
    if digest(log) != result["tests_sha256"]:
        raise ValueError("Receiving test log changed during readback")
    with (LOCAL / "receiving-tests.log").open("xb") as stream:
        stream.write(log)
    if digest((LOCAL / "receiving-tests.log").read_bytes()) != result["tests_sha256"]:
        raise ValueError("Receiving test log local bytes differ")
    write(LOCAL / "RECEIVING_TESTED.json", result)
    print(json.dumps({"status": "native32_source_staged_cpu_tests_passed_not_activated", **result}))


def stage():
    if LOCAL.exists():
        raise ValueError("Inspect existing attempt; never restage/relaunch blindly")
    try:
        _stage()
    except BaseException as error:
        if LOCAL.exists():
            write(LOCAL / "STAGING_OBSERVATION_FAILED.json", {"error": str(error),
                "remote_root_may_exist": True, "no_gpu_launch_performed": True, "do_not_retry": True})
        raise


ACTIVATE = r'''import argparse,hashlib,importlib.util,json,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); expected_plan=sys.argv[2]; expected_tests=sys.argv[3]
authority=json.loads(sys.argv[4])
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def write(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
if (root/'AUTHORIZATION.json').exists():raise ValueError('Inspect existing authorization/launch; do not retry')
if sha(root/'PLAN.json')!=expected_plan or sha(root/'receiving-tests.log')!=expected_tests:
 raise ValueError('Reviewed plan/receiving tests changed')
plan=json.loads((root/'PLAN.json').read_text())
if sha(root/'SOURCE_FILES.json')!=plan['source_manifest']['sha256']:raise ValueError('Source manifest differs')
files=json.loads((root/'SOURCE_FILES.json').read_text())
for name,row in files.items():
 p=root/name
 if p.is_symlink() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Source member differs')
script=root/'scripts/vast/robotics_gpu_check_queue.py'
spec=importlib.util.spec_from_file_location('robotics_gpu_check_queue',script)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.verify_static(plan);q.verify_navigation_static()
# Full900-second science verification belongs to the waiter after handoff, not
# this bounded activation observation. The waiter still verifies every endpoint.
q.predecessor_pending()
issued=time.time()
auth={'status':'root_authorized_native32_gpu_check','owner':'rep_geometry_transcoder/root',
 'plan_sha256':expected_plan,'instance':50205763,'gpu':0,'gpu_uuid':q.GPU_UUID,
 'issued_unix':issued,'expires_unix':issued+12*3600,
 'child_timeout_seconds':1210,'kill_grace_seconds':5,'full_history_authorized':False,
 'live_operator_authority':authority,'no_provider_transition':True}
write(root/'AUTHORIZATION.json',auth)
auth_sha=sha(root/'AUTHORIZATION.json')
command=['/usr/bin/python3','-u',str(script),'--plan',str(root/'PLAN.json'),
 '--plan-sha256',expected_plan,'--authorization',str(root/'AUTHORIZATION.json'),
 '--authorization-sha256',auth_sha]
child=None
try:
 subprocess.run(command+['--prepare-only'],env=q.environment(plan),stdin=subprocess.DEVNULL,
  stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=180)
 prepared=json.loads((root/'PREPARED.json').read_text())
 with (root/'waiter.log').open('x') as log:
  child=subprocess.Popen(command,env=q.environment(plan),stdin=subprocess.DEVNULL,
   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(root/'SPAWNED.json',{'pid':child.pid,'argv':command,'plan_sha256':expected_plan,
   'authorization_sha256':auth_sha,'identity_pending':True})
  deadline=time.monotonic()+10; binding=None
  while time.monotonic()<deadline:
   if child.poll() is not None:raise ValueError('Waiter exited during launch; inspect existing attempt')
   current=q.process(child.pid)
   if current is not None:
    if binding is not None and current['starttime']!=binding['starttime']:raise ValueError('Launch PID reused')
    binding=current
    if current['command']==command and current['state'] not in ('Z','X'):break
   time.sleep(.05)
  else:raise TimeoutError('Waiter identity not yet observed; preserve existing child')
  value={'status':'bounded_native32_cpu_waiter_activated','pid':child.pid,
   'start_ticks':binding['starttime'],'argv':command,'plan_sha256':expected_plan,
   'authorization_sha256':auth_sha,'prepared_sha256':sha(root/'PREPARED.json'),
   'source_sha256':plan['source_sha256'],'gpu_calls_by_launcher':0,
   'navigation_endpoints_required':180,'history_launch_authorized':False}
  write(root/'ACTIVATED.json',value)
 print(json.dumps({'activation':value,'authorization':auth,'prepared':prepared}))
except BaseException as error:
 write(root/'ACTIVATION_FAILED.json',{'error':str(error),'error_type':type(error).__name__,
  'may_have_live_waiter':child is not None,'pid':None if child is None else child.pid,
  'argv':command,'do_not_retry_or_signal':True})
 raise
'''


def activate():
    ssh, authority = connection()
    board = Path("/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md").read_text()
    if "native32 bounded GPU check activation" not in board:
        raise ValueError("Separate reviewed waiter activation reservation required")
    plan_sha = digest((LOCAL / "PLAN.json").read_bytes())
    ready = json.loads((LOCAL / "RECEIVING_TESTED.json").read_text())
    write(LOCAL / "ACTIVATION_INTENT.json", {"plan_sha256": plan_sha,
        "authority": authority, "program_sha256": digest(ACTIVATE.encode()),
        "gpu_calls_by_launcher": 0})
    try:
        result = json.loads(remote(ssh, ACTIVATE, ROOT, plan_sha, ready["tests_sha256"],
                                   json.dumps(authority), timeout=240))
        write(LOCAL / "ACTIVATED.json", result)
        print(json.dumps(result))
    except BaseException as error:
        write(LOCAL / "ACTIVATION_OBSERVATION_FAILED.json", {"error": str(error),
            "may_have_live_waiter": True, "inspect_same_handle_do_not_retry": True})
        raise


def status():
    ssh, authority = connection()
    program = '''import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out={'observed_receipt_errors':{}}
for name in ('PREPARED.json','SPAWNED.json','ACTIVATED.json','ACTIVATION_FAILED.json','queue/LAUNCH.json','queue/DONE.json','queue/FAILED.json','check/progress.json','check/FAILED.json','check/DONE.json'):
 p=root/name
 if p.exists():
  try:out[name]=json.loads(p.read_text())
  except (OSError,ValueError) as error:out['observed_receipt_errors'][name]=str(error)
row=None;bound=False
if 'ACTIVATED.json' in out:
 row=out['ACTIVATED.json'];bound=True
elif 'queue/LAUNCH.json' in out:
 identity=out['queue/LAUNCH.json']['identity']
 row={'pid':identity['pid'],'start_ticks':identity['starttime'],'argv':identity['command']};bound=True
elif 'SPAWNED.json' in out:row=out['SPAWNED.json']
elif out.get('ACTIVATION_FAILED.json',{}).get('pid') is not None:row=out['ACTIVATION_FAILED.json']
if row is not None:
 out['waiter_pid']=row['pid'];out['identity_bound']=bound
 proc=pathlib.Path('/proc',str(row['pid']))
 try:
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\\0') if x]
  out['waiter_live']=(int(stat[19])==row['start_ticks'] and argv==row['argv'] and stat[0] not in ('Z','X')) if bound else None
  out['observed_waiter_identity']={'pid':row['pid'],'start_ticks':int(stat[19]),'argv':argv}
  out['waiter_state']=stat[0]
 except FileNotFoundError:out['waiter_live']=False
print(json.dumps(out))
'''
    print(json.dumps({"authority": authority, "status": json.loads(remote(ssh, program, ROOT))}))


def collect_preparation():
    """Read back immutable activation evidence while the CPU waiter is alive."""
    ssh, _ = connection()
    names = ["PLAN.json", "SOURCE_FILES.json", "receiving-tests.log", "AUTHORIZATION.json",
             "PREPARED.json", "SPAWNED.json", "ACTIVATED.json", "queue/LAUNCH.json", "queue/PREPARED.json"]
    program = '''import base64,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);out={}
for name in json.loads(sys.argv[2]):
 p=root/name
 if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('Expected compact immutable preparation evidence')
 data=p.read_bytes();out[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
print(json.dumps(out))
'''
    records = json.loads(remote(ssh, program, ROOT, json.dumps(names)))
    if set(records) != set(names):
        raise ValueError("Incomplete activation evidence")
    values = {}
    for name, record in records.items():
        value = base64.b64decode(record["base64"], validate=True)
        if len(value) != record["bytes"] or digest(value) != record["sha256"]:
            raise ValueError("Corrupted receiving readback: " + name)
        values[name] = value
    docs = {name: json.loads(value) for name, value in values.items() if name.endswith(".json")}
    local_activation = json.loads((LOCAL / "ACTIVATED.json").read_text())
    activation = docs["ACTIVATED.json"]
    identity = docs["queue/LAUNCH.json"]["identity"]
    queue_expected = {"plan_sha256": activation["plan_sha256"],
        "authorization_sha256": activation["authorization_sha256"],
        "source_manifest_sha256": records["SOURCE_FILES.json"]["sha256"],
        "instance": 50205763, "gpu_uuid": UUID, "gpu_work_started": False}
    prepared_expected = {key: docs["PREPARED.json"][key] for key in
        ("source_sha256", "source_manifest_sha256", "fixed_files_verified", "model_imports", "gpu_calls")}
    if (activation != local_activation["activation"] or
            docs["AUTHORIZATION.json"] != local_activation["authorization"] or
            docs["PREPARED.json"] != local_activation["prepared"] or
            activation["plan_sha256"] != records["PLAN.json"]["sha256"] or
            activation["authorization_sha256"] != records["AUTHORIZATION.json"]["sha256"] or
            activation["prepared_sha256"] != records["PREPARED.json"]["sha256"] or
            identity["pid"] != activation["pid"] or identity["starttime"] != activation["start_ticks"] or
            identity["command"] != activation["argv"] or
            any(docs["queue/LAUNCH.json"].get(k) != v for k, v in queue_expected.items()) or
            any(docs["queue/PREPARED.json"].get(k) != v for k, v in prepared_expected.items()) or
            docs["SPAWNED.json"]["pid"] != activation["pid"] or
            docs["SPAWNED.json"]["argv"] != activation["argv"] or
            any(values[name] != (LOCAL / name).read_bytes()
                for name in ("PLAN.json", "SOURCE_FILES.json", "receiving-tests.log"))):
        raise ValueError("Receiving activation/source/identity binding changed")
    destination = LOCAL / "readback"
    destination.mkdir(exist_ok=False)
    for name, value in values.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
        if path.stat().st_size != records[name]["bytes"] or digest(path.read_bytes()) != records[name]["sha256"]:
            raise ValueError("Local activation readback changed: " + name)
    write(LOCAL / "PREPARATION_READBACK.json", {"status": "native32_waiter_activation_receipts_verified",
        "remote_root": ROOT, "files": {name: {k: row[k] for k in ("bytes", "sha256")}
        for name, row in records.items()}, "waiter_pid": activation["pid"],
        "gpu_check_completion_claimed": False, "history_launch_authorized": False})
    print(json.dumps({"status": "native32_waiter_activation_receipts_verified", "files": len(names),
                      "waiter_pid": activation["pid"], "gpu_check_completion_claimed": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "activate", "status", "collect-preparation"))
    {"stage": stage, "activate": activate, "status": status,
     "collect-preparation": collect_preparation}[parser.parse_args().mode]()
