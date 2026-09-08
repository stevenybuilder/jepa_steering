"""Preserve LA logs, earlier routing work and exact vendor working tree omitted by core archive.

Read-only remote inventory and tar stream, no experiment or lifecycle action.
Skip only separately preserved archive copies, rebuildable bytecode and Git internals.
"""
import json
from pathlib import Path
import shutil
import subprocess

from preserve_completed_core import remote, SSH
from backup_results_to_google import digest, verify_archive
from navigation_redistribution_common import write

PROJECT = Path(__file__).resolve().parents[2]
CORE = PROJECT / 'artifacts/offline_study/core-completion-preservation-20260908-v1'
OUT = CORE / 'la-supplement-v2'
INVENTORY = r'''
import hashlib,json,pathlib,sys
root=pathlib.Path('/workspace'); existing=set(json.load(sys.stdin))
excluded={'core-priority-archive-20260908-v1', 'core-completion-preservation-20260908-v1',
          'core-completion-preservation-transfer-check-20260908-v1'}
selected={}; skipped=[]; official_inputs={}
for scope in ('jepa-runtime','jepa_steering/vendor/jepa-wms'):
 for path in sorted((root/scope).rglob('*')):
  relative=path.relative_to(root); parts=relative.parts
  if '__pycache__' in parts or '.git' in parts or path.suffix=='.pyc': continue
  if scope=='jepa-runtime' and (parts[1] in excluded or str(path.relative_to(root/scope)) in existing): continue
  if path.is_symlink():
   skipped.append({'path':str(relative),'reason':'symlink','target':str(path.readlink())}); continue
  if not path.is_file(): continue
  if (path.name in ('.env','rclone.conf') or path.suffix in ('.pem','.key')
      or '\n' in str(relative) or '\0' in str(relative)):
   raise ValueError('Unexpected unsafe evidence path')
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(4<<20),b''): h.update(block)
  value={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
  if (str(relative).startswith('jepa-runtime/fixed-response-assets-20260908-v1/metaworld/data/')
      and path.suffix=='.parquet'):
   official_inputs[str(relative)]=value
  else: selected[str(relative)]=value
print(json.dumps({'files':selected,'skipped_symlinks':skipped,
 'released_dataset_cache_retained_on_source_volume':official_inputs,
 'excluded_duplicate_archive_roots':sorted(excluded)}))
'''


def main():
    OUT.mkdir(exist_ok=False)
    main = json.loads((CORE / 'FILES.json').read_text())
    inventory = remote(INVENTORY, stdin=json.dumps(list(main)), timeout=180)
    write(OUT / 'INVENTORY.json', inventory)
    manifest = inventory['files']
    if inventory['skipped_symlinks']:
        raise ValueError('Review skipped symlink targets before archiving')
    if sum(x['bytes'] for x in manifest.values()) > 100 << 20 or shutil.disk_usage(OUT).free < 1 << 30:
        raise ValueError('Supplement size/reserve bound')
    write(OUT / 'FILES.json', manifest)
    archive = OUT / 'la-work-and-logs-supplement.tar.gz'
    with archive.open('xb') as output:
        subprocess.run(SSH + ['tar -C /workspace -czf - --null --files-from=-'],
            input=b'\0'.join(name.encode() for name in manifest) + b'\0', stdout=output,
            check=True, timeout=180)
    expected, size = digest(archive), archive.stat().st_size
    verify_archive(['cat', str(archive)], expected, size, manifest)
    if remote(INVENTORY, stdin=json.dumps(list(main)), timeout=180) != inventory:
        raise ValueError('Supplement source changed during copy')
    write(OUT / 'LOCAL_VERIFIED.json', {'archive_bytes':size,'archive_sha256':expected,
        'files':len(manifest),'manifest_sha256':digest(OUT/'FILES.json'),
        'status':'all_selected_supplement_members_verified_and_source_rehashed',
        'no_source_deletions':True, 'full_source_volume_retained':True})
    print(json.dumps({'status':'la_supplement_local_verified','files':len(manifest),'bytes':size}),flush=True)


if __name__ == '__main__':
    main()
