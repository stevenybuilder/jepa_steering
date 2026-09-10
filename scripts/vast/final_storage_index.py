"""Publish restoration/deletion receipts, never rental lifecycle operations.

An immutable timestamped snapshot includes only preservation metadata, the
explicit restoration reports and helper source. It excludes staging/results
payloads, credentials, virtual environments and symlinks. Source metadata is
hashed and the archive is verified before upload and full Drive readback.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile

from final_storage_release import BASE, PROJECT, write
from final_preservation_drive import digest, upload_direct, verify
from upload_pause_archive import verify_stream


def publish():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=BASE/'global-index'/stamp;root.mkdir(parents=True,exist_ok=False)
    selected={}
    def add(path, name):
        if path.is_symlink() or not path.is_file():return
        if path.stat().st_size>64<<20:raise ValueError('Unexpected large audit file')
        selected[name]=path
    forbidden={'global-index','stage','bootstrap-stage','link-inventory','connector-parts','scratch'}
    for path in sorted(BASE.rglob('*')):
        rel=path.relative_to(BASE)
        if any(x in forbidden for x in rel.parts):continue
        if path.suffix in ('.json','.txt','.log'):
            add(path,'preservation-receipts/'+rel.as_posix())
    for entry in json.loads((BASE/'PRIOR_DRIVE_ARCHIVES_REVERIFIED.json').read_text()):
        proof=PROJECT/entry['proof']
        for path in (proof,proof.parent/'FILES.json'):
            add(path,'prior-source-proof/'+path.relative_to(PROJECT).as_posix())
    for path in (PROJECT/'reports/VAST_FINAL_STORAGE_RELEASE_20260910.md',
                 PROJECT/'reports/CORE_METAWORLD_BEHAVIORAL_RESULTS.md',
                 PROJECT/'reports/GOOGLE_RESULTS_ARCHIVE_README.md'):
        add(path,'reports/'+path.name)
    add(PROJECT.parent/'GPU_RESOURCE_BOARD.md','GPU_RESOURCE_BOARD.md')
    for path in sorted((PROJECT/'scripts/vast').glob('final_*.py')):
        add(path,'preservation-code/'+path.name)
    manifest={name:{'bytes':path.stat().st_size,'sha256':digest(path)}
              for name,path in selected.items()}
    manifest_path=root/'FILES.json';write(manifest_path,manifest)
    archive=root/f'JEPA-final-storage-recovery-index-{stamp}.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for name,path in selected.items():tar.add(path,arcname=name,recursive=False)
        tar.add(manifest_path,arcname='FILES.json',recursive=False)
    expected={**manifest,'FILES.json':{'bytes':manifest_path.stat().st_size,'sha256':digest(manifest_path)}}
    size=archive.stat().st_size;sha=digest(archive)
    with archive.open('rb') as source:verify_stream(source,sha,size,expected)
    obj=upload_direct(archive,archive.name,root/'DRIVE_UPLOADED.json')
    result=verify(obj['id'],archive.name,size,sha)
    write(root/'DRIVE_VERIFIED.json',{'timestamp':stamp,'drive_file_id':obj['id'],
          'bytes':size,'sha256':sha,'manifest_files':len(manifest),**result})
    print(json.dumps({'global_restore_index_id':obj['id'],'bytes':size,
                      'full_drive_readback':True,'snapshot':stamp}),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    publish()
