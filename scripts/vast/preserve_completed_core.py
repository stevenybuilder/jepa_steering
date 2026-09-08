"""Wait for the existing core panel, reverify it, and stream its evidence to GCS.

No GPU launch, method change, outcome-based selection, rental restart or deletion.
The root agent must separately review the completed backup before stopping LA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

from backup_results_to_google import verify_archive
from snapshot_live_results import HASH_FILES

PROJECT = Path(__file__).resolve().parents[2]
ROOT = '/workspace/jepa-runtime/'
PANEL = 'fixed-response-behavior-20260908-v1'
ANALYSIS = 'fixed-response-behavior-analysis-20260908-v1'
SSH = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-p', '21938', 'root@98.142.241.142']

STATUS = r'''
import json,pathlib
r=pathlib.Path('/workspace/jepa-runtime'); p=r/'fixed-response-behavior-20260908-v1'
paths=[p/t/a/('shard-gpu'+str(g)) for g in range(8)
       for t in [('reach' if g<4 else 'reach-wall')]
       for a in ('native','fixed_rank4','matched_random_fixed_rank4','coupling_only','matched_random_coupling')]
failed=[str(x) for x in paths if (x/'FAILED.json').exists()]
live=[]
for pid in [*range(5202,5210),5929]:
 f=pathlib.Path('/proc')/str(pid)
 if f.exists():
  raw=(f/'stat').read_text(); state=raw[raw.rfind(')')+2:].split()[0]
  command=(f/'cmdline').read_bytes().decode().replace('\0',' ')
  if state not in ('Z','X'): live.append({'pid':pid,'state':state,'command':command})
print(json.dumps({'complete_shards':sum((x/'DONE.json').exists() for x in paths),
 'published_files':sum(len(list(x.glob('episode-*.json'))) for x in paths),
 'analysis_done':(r/'fixed-response-behavior-analysis-20260908-v1/DONE.json').exists(),
 'failed':failed,'live':live}))
'''

VERIFY = r'''
import json,pathlib,sys
r=pathlib.Path('/workspace/jepa-runtime'); p=r/'fixed-response-behavior-20260908-v1'
code=r/'fixed-response-code-20260908-v4/src'; sys.path.insert(0,str(code))
from offline_study.fixed_response_behavior_analysis import load_panel,analyze
from offline_study.behavioral_development import verified_report
from offline_study.protocol import sha256
import offline_study.fixed_response_behavior_analysis as module
freeze=r/'fixed-response-behavior-evidence-20260908-v1/artifacts/offline_study/fixed-response-20260908-v1/behavioral-freeze-v1'
published,digest=verified_report(r/'fixed-response-behavior-analysis-20260908-v1')
panel,inputs=load_panel(p,freeze); recomputed=analyze(panel)
if any(published.get(k)!=v for k,v in recomputed.items()): raise ValueError('Frozen paired analysis recomputation differs')
if (published['input_reports_sha256']!=inputs or published['freeze_sha256']!=sha256(freeze/'protocol.json')
 or published['analysis_source_sha256']!=sha256(pathlib.Path(module.__file__))): raise ValueError('Analysis binding differs')
print(json.dumps({'status':'all960_core_records_and_frozen_analysis_recomputed_exact',
 'report_sha256':digest,'freeze_sha256':published['freeze_sha256'],
 'analysis_source_sha256':published['analysis_source_sha256'],'input_reports_sha256':inputs,
 'new_gpu_jobs':0,'full_six_task_study_complete':False}))
'''

SELECT = r'''
import json,pathlib
r=pathlib.Path('/workspace/jepa-runtime')
roots=['fixed-response-behavior-20260908-v1','fixed-response-behavior-analysis-20260908-v1',
 'fixed-response-code-20260908-v4','fixed-response-worker-checks-20260908-v1',
 'fixed-response-behavior-evidence-20260908-v1','fixed-response-assets-20260908-v1/jepa_wm_metaworld.pth.tar',
 'hmm-fixed-response-behavior-20260908-v3','hmm-fixed-response-code-20260908-v3',
 'hmm-fixed-response-evidence-20260908-v3','routing-priority-20260908-v3',
 'core-priority-pause-20260908-v1','finish_fixed_behavior_panel_v1.py','pause_hmm_for_core.py']
names=set()
for name in roots:
 path=r/name
 if not path.exists(): raise ValueError('Missing explicit evidence root '+name)
 for p in (path.rglob('*') if path.is_dir() else [path]):
  if '__pycache__' in p.parts or not p.is_file(): continue
  if p.is_symlink() or p.name in ('.env','rclone.conf') or p.suffix in ('.key','.pem'): raise ValueError('Unsafe archive member')
  names.add(str(p.relative_to(r)))
print(json.dumps(sorted(names)))
'''


def remote(code, *, research=False, stdin=None, timeout=180):
    command = (['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'PYTHONPATH=' + ROOT + 'fixed-response-code-20260908-v4/src:/workspace/jepa-python/lib/python3.10/site-packages',
        'nice', '-n', '19', '/workspace/jepa-planning-python/bin/python'] if research else ['/usr/bin/python3'])
    return json.loads(subprocess.check_output(SSH + [shlex.join(command + ['-c', code])],
                      input=stdin, text=True, timeout=timeout))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (args.output / name).open('x') as f: json.dump(value, f, indent=2, sort_keys=True)
    write('LAUNCH.json', {'operation':'existing_core_collection_no_gpu_or_lifecycle_changes',
        'instance':50233992,'pid':os.getpid(),'time':time.time(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    deadline = time.monotonic() + 6 * 3600
    while True:
        try:
            status = remote(STATUS, timeout=45)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # A broken observation is not a terminal worker or permission to retry
            # an experiment. Preserve every run and repeat the read-only check.
            if time.monotonic() > deadline: raise
            print(json.dumps({'status':'observation_failed_no_scientific_action'}),flush=True)
            time.sleep(45); continue
        print(json.dumps({**{k:status[k] for k in ('complete_shards','published_files','analysis_done','failed')},
                          'live_bound_pids':[p['pid'] for p in status['live']]}),flush=True)
        if status['failed']: raise ValueError('Required core job failed; no scientific retry')
        if status['complete_shards'] == 40 and status['analysis_done'] and not status['live']: break
        if not args.wait or time.monotonic() > deadline: raise ValueError('Core collection incomplete')
        if not status['live']: raise ValueError('Expected workers terminal before required completion')
        time.sleep(45)
    verified = remote(VERIFY, research=True, timeout=1800)
    write('SCIENCE_VERIFIED.json', verified)
    names = remote(SELECT)
    manifest = remote(HASH_FILES, stdin=json.dumps(names), timeout=900)
    write('FILES.json', manifest)
    cloud = 'gs://rgt-jepa-archive-2026/rep_geometry_transcoder/' + args.output.name + '/core-and-paused-hmm.tar.gz'
    write('PLAN.json', {'cloud_uri':cloud,'selected_roots_only':True,'source_volume_retained':True,
        'files':len(names),'bytes':sum(x['bytes'] for x in manifest.values()),'hmm_is_partial_not_complete':True})
    tar = upload = None
    hashed, size = hashlib.sha256(), 0
    try:
        with tempfile.TemporaryFile() as listing:
            listing.write(b'\0'.join(n.encode() for n in names)+b'\0'); listing.seek(0)
            tar = subprocess.Popen(SSH + ['nice -n 19 tar -C /workspace/jepa-runtime --hard-dereference --no-recursion --null -czf - -T -'],stdin=listing,stdout=subprocess.PIPE)
            upload = subprocess.Popen(['gcloud','storage','cp','--if-generation-match=0','-',cloud],stdin=subprocess.PIPE)
            for block in iter(lambda:tar.stdout.read(4<<20),b''):
                hashed.update(block); size+=len(block); upload.stdin.write(block)
            upload.stdin.close()
            if tar.wait() or upload.wait(): raise ValueError('Evidence stream upload incomplete')
    finally:
        for process in (tar,upload):
            if process is not None and process.poll() is None: process.terminate(); process.wait()
    verify_archive(['gcloud','storage','cat',cloud],hashed.hexdigest(),size,manifest)
    if remote(HASH_FILES,stdin=json.dumps(names),timeout=900)!=manifest: raise ValueError('Source files changed during preservation')
    write('VERIFIED.json',{'status':'completed_core_and_paused_hmm_full_cloud_member_readback_verified',
        'cloud_uri':cloud,'archive_sha256':hashed.hexdigest(),'archive_bytes':size,'files':len(names),
        'report_sha256':verified['report_sha256'],'new_gpu_jobs':0,'rental_stopped':False,'full_study_complete':False})
    subprocess.run(['gcloud','storage','cp','--if-generation-match=0',str(args.output/'FILES.json'),
        str(args.output/'SCIENCE_VERIFIED.json'),str(args.output/'VERIFIED.json'),cloud.rsplit('/',1)[0]+'/receipts/'],check=True)
    print(json.dumps({'status':'verified_core_ready_for_root_shutdown_review','cloud_uri':cloud}),flush=True)


if __name__=='__main__':
    main()
