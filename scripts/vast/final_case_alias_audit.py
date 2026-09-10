"""Preserve Linux paths that alias on case-insensitive macOS staging.

Download each exact source to a separately numbered file. The restoration tar
retains the original case-sensitive source names. No source writes/deletions.
"""
import argparse
from collections import defaultdict
import json
import subprocess
import tarfile
from final_storage_release import BASE,command,write
from final_preservation_plan import excluded
from final_preservation_drive import digest,upload_direct,verify
from upload_pause_archive import verify_stream


def audit(instance):
    root=BASE/str(instance)
    inv=json.loads(sorted(root.glob('*-inventory.json'))[-1].read_text())
    if inv['returncode']!=0:raise ValueError('Incomplete inventory')
    groups=defaultdict(list)
    for row in inv['entries']:
        if row['mode'].startswith('-') and not excluded(row['path']):
            groups[row['path'].casefold()].append(row)
    selected=[x for group in groups.values() if len(group)>1 for x in group]
    if sum(x['bytes'] for x in selected)>64<<20:
        raise ValueError('Case audit exceeds bounded scratch allowance')
    work=root/'case-alias-corrections';work.mkdir(exist_ok=True)
    if (work/'DRIVE_VERIFIED.json').exists():return
    if not selected:
        write(work/'DRIVE_VERIFIED.json',{'verified':True,'full_byte_readback':True,
              'manifest':{},'reason':'No case-alias paths in full required inventory'})
        print(json.dumps({'instance':instance,'case_alias_files':0}),flush=True);return
    stage=work/'stage';stage.mkdir(exist_ok=True)
    cmd,source=command(instance,'/workspace/')
    manifest={};paths={}
    for i,row in enumerate(selected):
        name=row['path'];target=stage/f'file-{i:04d}'
        subprocess.run(cmd+['-t',source+name,str(target)],check=True,timeout=120,
                       stdout=subprocess.DEVNULL)
        if target.stat().st_size!=row['bytes']:raise ValueError('Source size changed')
        manifest[name]={'bytes':row['bytes'],'sha256':digest(target)};paths[name]=target
    write(work/'FILES.json',manifest)
    archive=work/f'instance-{instance}-case-sensitive-source-corrections.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name,path in paths.items():tar.add(path,arcname=name,recursive=False)
    size=archive.stat().st_size;sha=digest(archive)
    with archive.open('rb') as f:verify_stream(f,sha,size,manifest)
    obj=upload_direct(archive,archive.name,work/'DRIVE_UPLOADED.json')
    result=verify(obj['id'],archive.name,size,sha)
    write(work/'DRIVE_VERIFIED.json',{'instance':instance,'manifest':manifest,
          'bytes':size,'sha256':sha,'drive_file_id':obj['id'],
          'overrides_case_aliased_batch_members':True,**result})
    print(json.dumps({'instance':instance,'case_alias_files':len(manifest),
                      'source_paths_individually_preserved':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('instance',type=int)
    audit(p.parse_args().instance)
