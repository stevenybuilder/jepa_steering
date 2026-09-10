"""Isolated receiving CPU input check; no GPU/model/history launch or raw pull."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

import navigation_redistribution_stage as stage
from navigation_redistribution_common import digest, write

PROJECT = Path(__file__).resolve().parents[2]
ROOT = "/workspace/jepa-runtime/robotics-training-batch-20260908-v1"
OUTPUT = PROJECT / "artifacts/offline_study/robotics-training-batch-20260908-v1"
VENDOR = "/workspace/jepa_steering/vendor/jepa-wms"
PYTHON = "/workspace/jepa-planning-python/bin/python"
LIBS = ("/workspace/jepa-planning-python/lib/python3.10/site-packages:"
        "/workspace/jepa-python/lib/python3.10/site-packages")
RAW_REPORT_SHA = "1d8f7bd7c08543b261816a311ea0cd21098eaf326c6f4d92d4e54dedf683591f"

RUNNER = r'''import hashlib,json,os,pathlib,re,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def write(name,value):
 with (root/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
started=time.monotonic()
try:
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be hidden')
 if sha(root/'SOURCE_FILES.json')!=expected:raise ValueError('Source manifest changed')
 manifest=json.loads((root/'SOURCE_FILES.json').read_text())
 def verify():
  for name,row in manifest.items():
   p=root/name
   if p.is_symlink() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
    raise ValueError('Receiving source changed: '+name)
 verify()
 raw=pathlib.Path('/workspace/jepa-runtime/robotics-input-verification-20260908-v2/pusht')
 if sha(raw/'report.json')!='1d8f7bd7c08543b261816a311ea0cd21098eaf326c6f4d92d4e54dedf683591f':
  raise ValueError('Original full raw-input proof changed')
 with (root/'cpu-tests.log').open('x') as log:
  subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),
   '-p','test_robotics_training_*.py','-v'],stdout=log,stderr=subprocess.STDOUT,
   stdin=subprocess.DEVNULL,check=True,timeout=300)
 if not re.search(r'Ran 56 tests in ',(root/'cpu-tests.log').read_text()):raise ValueError('Receiving test population changed')
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA initialization')
 from offline_study.robotics_training_batch import produce_pusht_batch
 assets=pathlib.Path('/workspace/jepa-runtime/pusht-planning-assets-20260908-v2')
 _,report=produce_pusht_batch(pathlib.Path('/workspace/jepa_steering/vendor/jepa-wms'),
  assets/'data/pusht_noise',raw_inputs_receipt=raw,provenance=assets/'files.json',
  output=root/'evidence',model_seed=234)
 verify()
 if torch.cuda.is_initialized():raise ValueError('CPU input check initialized CUDA')
 write('DONE.json',{'status':'receiving_native_pusht_batch_verified_cpu_only',
  'source_manifest_sha256':expected,'cpu_tests_sha256':sha(root/'cpu-tests.log'),'cpu_tests':56,
  'input_report_sha256':sha(root/'evidence/report.json'),
  'raw_input_report_sha256':sha(raw/'report.json'),'gpu_calls':0,
  'cuda_initialized':False,'validation_or_confirmation_access':False,
  'history_launch_authorized':False,'seconds':time.monotonic()-started})
 print(json.dumps({'status':'receiving_native_pusht_batch_verified_cpu_only',
  'done_sha256':sha(root/'DONE.json'),'seconds':time.monotonic()-started}),flush=True)
except BaseException as error:
 write('FAILED.json',{'error':str(error),'error_type':type(error).__name__,
  'seconds':time.monotonic()-started,'no_retry_or_scientific_launch':True})
 raise
'''


def connection():
    board = Path("/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md").read_text()
    if ROOT not in board or "native Push-T training batch CPU parity" not in board:
        raise ValueError("Exact receiving training-only CPU reservation required")
    connections, authority = stage.connections()
    return connections[50205763], authority


def _launch():
    ssh, authority = connection()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    names = sorted(str(p.relative_to(PROJECT)) for p in (PROJECT / "src/offline_study").rglob("*.py"))
    names += sorted(str(p.relative_to(PROJECT)) for p in (PROJECT / "tests").glob("test_robotics_training_*.py"))
    members = {name: (PROJECT / name).read_bytes() for name in names}
    members["RUNNER.py"] = RUNNER.encode()
    manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in members.items()}
    write(OUTPUT / "SOURCE_FILES.json", manifest)
    manifest_sha = digest(OUTPUT / "SOURCE_FILES.json")
    write(OUTPUT / "INTENT.json", {"remote_root": ROOT, "instance": 50205763,
        "authority": authority, "source_manifest_sha256": manifest_sha,
        "raw_input_report_sha256": RAW_REPORT_SHA, "training_clips": 256,
        "cpu_only": True, "validation_or_confirmation_access": False,
        "gpu_or_history_launch_authorized": False, "raw_batch_download": False})
    create = '''import pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); vendor=pathlib.Path(sys.argv[2])
if not vendor.is_dir() or shutil.disk_usage(root.parent).free < 4*(1<<30):raise ValueError('Source/disk preflight')
root.mkdir(exist_ok=False);(root/'vendor').mkdir();(root/'vendor/jepa-wms').symlink_to(vendor,target_is_directory=True)
'''
    subprocess.run(ssh + [shlex.join(["/usr/bin/python3", "-c", create, ROOT, VENDOR])], check=True, timeout=45)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, data in {**members, "SOURCE_FILES.json": (OUTPUT / "SOURCE_FILES.json").read_bytes()}.items():
                info = tarfile.TarInfo(name); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        stream.seek(0)
        subprocess.run(ssh + [shlex.join(["tar", "--keep-old-files", "-xzf", "-", "-C", ROOT])],
                       stdin=stream, check=True, timeout=45)
    argv = ["env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=" + ROOT + "/src:" + LIBS, "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "/usr/bin/ionice", "-c", "3", "/usr/bin/nice", "-n", "19", PYTHON,
        "-u", ROOT + "/RUNNER.py", ROOT, manifest_sha]
    start = '''import hashlib,json,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]);argv=json.loads(sys.argv[2]);expected=sys.argv[3]
if hashlib.sha256((root/'SOURCE_FILES.json').read_bytes()).hexdigest()!=expected:raise ValueError('Manifest differs')
manifest=json.loads((root/'SOURCE_FILES.json').read_text())
for name,row in manifest.items():
 p=root/name
 if p.is_symlink() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source differs')
if (root/'LAUNCH.json').exists():raise ValueError('Never relaunch this exact root')
with (root/'receiving.log').open('xb') as log:
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 with (root/'SPAWN.json').open('x') as f:json.dump({'pid':child.pid,'argv':argv,'unix':time.time()},f)
 proc=pathlib.Path('/proc',str(child.pid));deadline=time.monotonic()+10
 while True:
  if child.poll() is not None:raise ValueError('Child exited before launch identity bound; inspect SPAWN/FAILED/log, never relaunch')
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  observed=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\\0') if x]
  if observed==argv[-5:] and stat[0] not in ('Z','X'):break
  if time.monotonic()>deadline:raise ValueError('Post-exec identity not yet verified; preserve spawned process, never relaunch')
  time.sleep(.05)
 value={'pid':child.pid,'start_ticks':int(stat[19]),'argv':argv,'observed_argv':observed,'source_manifest_sha256':expected,'cpu_only':True,'unix':time.time()}
 with (root/'LAUNCH.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
 print(json.dumps(value))
'''
    result = subprocess.check_output(ssh + [shlex.join(["/usr/bin/python3", "-c", start,
        ROOT, json.dumps(argv), manifest_sha])], text=True, timeout=45)
    value = json.loads(result)
    write(OUTPUT / "LAUNCH.json", value)
    print(json.dumps(value))


def launch():
    # A failed observation is not permission to start another child. Preserve
    # local intent and inspect this same root's SPAWN/process/receipts instead.
    if OUTPUT.exists():
        raise ValueError("Existing attempt must be inspected, never relaunched")
    try:
        _launch()
    except BaseException as error:
        if OUTPUT.exists():
            write(OUTPUT / "LAUNCH_OBSERVATION_FAILED.json", {"error": str(error),
                "error_type": type(error).__name__, "remote_root": ROOT,
                "may_have_live_receiving_child": True, "do_not_retry_launch": True})
        raise


def status():
    ssh, authority = connection()
    program = '''import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);bound=(root/'LAUNCH.json').exists()
record='LAUNCH.json' if bound else 'SPAWN.json'
if not (root/record).exists():
 print(json.dumps({'launch_identity_available':False,'bound_process_alive':None,'inspect_existing_attempt':True}));sys.exit(0)
launch=json.loads((root/record).read_text())
proc=pathlib.Path('/proc',str(launch['pid']));running=False if bound else None;argv=[];state=None
if (proc/'stat').exists():
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split();state=stat[0]
 argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\\0') if x]
 if bound:running=int(stat[19])==launch['start_ticks'] and state not in ('Z','X') and argv==launch['observed_argv']
result={'pid':launch['pid'],'launch_identity_available':bound,'bound_process_alive':running,'process_state':state,'argv':argv}
for name in ('DONE.json','FAILED.json','evidence/progress.json','evidence/FAILED.json'):
 p=root/name
 if p.exists():result[name]=json.loads(p.read_text())
print(json.dumps(result))
'''
    value = json.loads(subprocess.check_output(ssh + [shlex.join(["/usr/bin/python3", "-c", program, ROOT])],
                                              text=True, timeout=45))
    print(json.dumps({"authority": authority, "receiving": value}))


def collect():
    ssh, _ = connection()
    names = ["SOURCE_FILES.json", "LAUNCH.json", "cpu-tests.log", "DONE.json"]
    names += ["evidence/" + name for name in
              ("protocol.json", "selected_inputs.json", "comparisons.json", "report.json", "DONE.json")]
    program = '''import base64,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);names=json.loads(sys.argv[2]);result={}
if (root/'FAILED.json').exists() or (root/'evidence/FAILED.json').exists():raise ValueError('Receiving proof failed')
launch=json.loads((root/'LAUNCH.json').read_text());stat=pathlib.Path('/proc',str(launch['pid']),'stat')
if stat.exists():
 fields=stat.read_text().rsplit(')',1)[1].split()
 if int(fields[19])==launch['start_ticks'] and fields[0] not in ('Z','X'):raise ValueError('Receiving process still live; wait for exact terminal state')
for name in names:
 p=root/name
 if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('Unexpected compact receipt')
 data=p.read_bytes();result[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
print(json.dumps(result))
'''
    fetched = json.loads(subprocess.check_output(ssh + [shlex.join(["/usr/bin/python3", "-c", program,
        ROOT, json.dumps(names)])], text=True, timeout=45))
    if set(fetched) != set(names):
        raise ValueError("Incomplete receiving evidence")
    payload = {}
    for name, row in fetched.items():
        data = base64.b64decode(row["base64"], validate=True)
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Receiving receipt changed: " + name)
        payload[name] = data
    doc = lambda name: json.loads(payload[name])
    done, report, protocol = doc("DONE.json"), doc("evidence/report.json"), doc("evidence/protocol.json")
    if (doc("LAUNCH.json") != json.loads((OUTPUT / "LAUNCH.json").read_text()) or
            doc("LAUNCH.json")["source_manifest_sha256"] != digest(OUTPUT / "SOURCE_FILES.json") or
            done["status"] != "receiving_native_pusht_batch_verified_cpu_only" or
            done["source_manifest_sha256"] != digest(OUTPUT / "SOURCE_FILES.json") or
            done["source_manifest_sha256"] != fetched["SOURCE_FILES.json"]["sha256"] or
            done["raw_input_report_sha256"] != RAW_REPORT_SHA or done["gpu_calls"] != 0 or
            done["cpu_tests"] != 56 or
            done["cuda_initialized"] is not False or done["history_launch_authorized"] is not False or
            done["cpu_tests_sha256"] != fetched["cpu-tests.log"]["sha256"] or
            done["input_report_sha256"] != fetched["evidence/report.json"]["sha256"] or
            doc("evidence/DONE.json")["report_sha256"] != fetched["evidence/report.json"]["sha256"] or
            report["protocol_sha256"] != fetched["evidence/protocol.json"]["sha256"] or
            report["comparisons_sha256"] != fetched["evidence/comparisons.json"]["sha256"] or
            protocol["selected_inputs_sha256"] != fetched["evidence/selected_inputs.json"]["sha256"] or
            report["global_batch"] != 256 or report["validation_or_confirmation_access"] is not False):
        raise ValueError("Receiving evidence binding differs")
    readback = OUTPUT / "readback"; readback.mkdir(exist_ok=False)
    for name, data in payload.items():
        destination = readback / name; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        if destination.stat().st_size != fetched[name]["bytes"] or digest(destination) != fetched[name]["sha256"]:
            raise ValueError("Local receiving-proof readback differs: " + name)
    write(OUTPUT / "READBACK.json", {"status": "compact_native_batch_receipts_hash_verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "remote_root": ROOT,
        "files": {name: {k: row[k] for k in ("bytes", "sha256")} for name, row in fetched.items()},
        "batch_stays_on_receiving_worker": True, "gpu_calls": 0,
        "training_history_authorized": False})
    print(json.dumps({"status": "compact_native_batch_receipts_hash_verified", "files": len(names)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "status", "collect"))
    {"launch": launch, "status": status, "collect": collect}[parser.parse_args().mode]()
