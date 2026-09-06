from dispatch import *
import time
from datetime import datetime,timezone
# Keep concise aggregate findings twice as well as the full underlying verified trees.
reports=['FINDINGS.md','pilot-v1-summary.json','pilot-fullgrid-v1-summary.json']
report_hashes={n:hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()for n in reports}
for w in [49155754,49902461]:
 r=root(w)+'/checkpoint-report-v1';run(w,'mkdir '+r)
 for n in reports:
  upload(w,LOCAL/n,r+'/'+n);assert run(w,'sha256sum '+r+'/'+n).split()[0]==report_hashes[n]
status=[]
for w in HOSTS:
 ids=[json.load((LOCAL/n).open())['pid']for n in [f'launch-{w}.json',f'launch-fullgrid-{w}.json']]
 output=run(w,'nvidia-smi --query-compute-apps=pid --format=csv,noheader');assert not output.strip()
 for pid in ids:
  proc=run(w,'ps -p '+str(pid)+' -o args= || true');assert not proc.strip(),proc
 receipt=json.load((LOCAL/f'backup-fullgrid-{w}.json').open());assert receipt['complete']and receipt['source_and_backup_identical']
 status.append(dict(worker=w,owned_pids=ids,owned_pids_absent=True,gpu_compute_empty=True,backup_bytes=receipt['bytes'],backup_manifest_sha256=hashlib.sha256((LOCAL/f'backup-fullgrid-{w}.json').read_bytes()).hexdigest()))
now=datetime.now(timezone.utc).isoformat();d=dict(complete=True,utc=now,instances=status,full_tensors_local=False,held_access=False,full99_promoted=False,aggregate_gpu_process_seconds=148.83125630399445,aggregate_gpu_process_hours=148.83125630399445/3600,report_hashes=report_hashes,report_copies=[root(w)+'/checkpoint-report-v1'for w in [49155754,49902461]],no_owned_jobs_remaining=True)
(LOCAL/'CHECKPOINT.json').write_text(json.dumps(d,indent=2)+'\n')
board=Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md')
with board.open('a')as f:f.write('\n\n## Reach autoresearch steering checkpoint — '+now+'\n\nAll three campaign assignments **49155754,49902461,49982193 RELEASED** to the project after two frozen DEVELOPMENT output-correction pilots complete. Owner `rep_geometry_transcoder/autoresearch_steering_20260906/reach_steering`; all six owned GPU process PIDs absent, all three GPU compute lists empty at this timestamp. Source + distinct-backup SHA verified across266,249,838 bytes; local receipts `artifacts/geometry_map/autoresearch_steering_20260906/reach/backup-fullgrid-ID.json`, final `CHECKPOINT.json`. Fullgrid correctedH1–3 visualMSE−26.83%, proprio−49.35%, but physicalprogress−0.7337mm vsnativeplanner and0/8positive; gatefailed, nofull99/held12–65 access. Pooledfirstpilot retained, totals148.831processseconds=.041342GPUh. Compact report doubly verified on491+461 under`checkpoint-report-v1`. Every owned job/transfer complete; project leases retained, no instance lifecycle change. This entry supersedes this branch\'s prior reservations only; other session assignments untouched.\n')
print(json.dumps(d,indent=2))
