"""Add only missing pinned PointMaze training files; never overwrite live inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

ROOT = Path('/workspace/jepa-runtime')
TRANSFER = ROOT/'pointmaze-history-transfer-20260908-v1'
OUTPUT = ROOT/'pointmaze-training-receiving-20260908-v1'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    def write(name,row):
        (OUTPUT/name).write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
    started=time.monotonic()
    try:
        receipt=TRANSFER/'pointmaze-training-inputs-20260908-v1'
        report=json.loads((receipt/'report.json').read_text())
        if (digest(receipt/'report.json')!=json.loads((receipt/'DONE.json').read_text())['report_sha256'] or
                report['status']!='complete_pointmaze_inputs_match_pinned_archive' or
                report['files_sha256']!=digest(receipt/'files.json') or
                report['protocol_sha256']!=digest(receipt/'protocol.json')):
            raise ValueError('Wrong original complete PointMaze source receipt')
        files=json.loads((receipt/'files.json').read_text())
        required={'states.pth','actions.pth','seq_lengths.pth'}|{f'obses/episode_{i:03d}.pth' for i in range(2000)}
        if set(files)!=required:
            raise ValueError('Incomplete native PointMaze source population')
        rel='navigation-assets-20260907-v1/downloads/dataset/point_maze/point_maze.zip'
        archive=TRANSFER/rel
        if digest(archive)!='6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc':
            raise ValueError('Wrong complete official archive')
        target=ROOT/'navigation-assets-20260907-v1/extracted/pointmaze/point_maze'
        existing,missing=[],[]
        for name,row in files.items():
            p=target/name
            if p.is_symlink(): raise ValueError('Symlink input not allowed')
            if p.exists():
                if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
                    raise ValueError('Existing live PointMaze input differs; do not overwrite: '+name)
                existing.append(name)
            else: missing.append(name)
        need=sum(files[n]['bytes'] for n in missing)
        if shutil.disk_usage(target).free<need+16*1024**3:
            raise ValueError('Insufficient space for complete dataset and checkpoint reserve')
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            for i,name in enumerate(missing,1):
                member='point_maze/'+name
                if names.count(member)!=1 or z.getinfo(member).file_size!=files[name]['bytes']:
                    raise ValueError('Missing or duplicate official member')
                p=target/name; p.parent.mkdir(parents=True,exist_ok=True)
                temp=p.with_name(p.name+'.receiving_tmp')
                h=hashlib.sha256()
                with z.open(member) as source,temp.open('xb') as destination:
                    for b in iter(lambda:source.read(8<<20),b''): destination.write(b);h.update(b)
                if h.hexdigest()!=files[name]['sha256'] or temp.stat().st_size!=files[name]['bytes'] or p.exists():
                    raise ValueError('New input failed binding or appeared concurrently; preserve partial file')
                temp.rename(p)
                if i%200==0:
                    progress={'added_files':i,'missing_files':len(missing),'existing_files_untouched':len(existing)}
                    write('progress.json',progress); print(json.dumps(progress),flush=True)
        # Ground every copied metadata file; existing original receipts must agree.
        for directory in ('navigation-assets-20260907-v1','navigation-input-check-20260907-v1',
                          'pointmaze-training-inputs-20260908-v1','pointmaze-history-code-20260908-v1'):
            for source in (TRANSFER/directory).rglob('*'):
                if source.is_symlink():
                    if source.name!='vendor': raise ValueError('Unexpected transfer symlink')
                    continue
                if not source.is_file(): continue
                relpath=source.relative_to(TRANSFER); dest=ROOT/relpath
                if dest.exists():
                    if dest.is_symlink() or digest(dest)!=digest(source):
                        raise ValueError('Existing source/receipt differs: '+str(relpath))
                    continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with source.open('rb') as src,dest.open('xb') as out: shutil.copyfileobj(src,out,8<<20)
                if digest(dest)!=digest(source): raise ValueError('Transfer copy changed')
        code=ROOT/'pointmaze-history-code-20260908-v1'
        if not (code/'vendor').exists(): (code/'vendor').symlink_to('/workspace/jepa_steering/vendor',target_is_directory=True)
        write('report.json',{'status':'receiving_complete_pointmaze_inputs_verified_existing_files_untouched',
            'original_input_report_sha256':digest(receipt/'report.json'),'data_files':len(files),
            'data_bytes':sum(r['bytes'] for r in files.values()),'added_files':len(missing),
            'existing_files_untouched':len(existing),'archive_sha256':digest(archive),'seconds':time.monotonic()-started,
            'training_started':False,'other_navigation_tasks_full_assets_not_claimed':True})
        write('DONE.json',{'report_sha256':digest(OUTPUT/'report.json')})
        print(json.dumps({'status':'complete_pointmaze_receiving_inputs_verified'}),flush=True)
    except Exception as exc:
        write('FAILED.json',{'error':str(exc),'existing_files_not_overwritten':True}); raise


if __name__=='__main__': main()
