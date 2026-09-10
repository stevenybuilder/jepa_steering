"""Bounded existing-auth Drive checks and stopped-disk missing-object relay.

No rental lifecycle operations. Only this script's verified scratch copy is
removed; source disks and all prior artifacts remain intact.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from final_storage_release import BASE, PROJECT, command, write

FOLDER='14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48'
# Existing JEPA GCS archive project (bucket projectNumber 31043195041).
# Own request quota avoids rclone's globally saturated shared OAuth project;
# no new credentials, IAM grants, file sharing, or compute are involved.
QUOTA_PROJECT='project-flash-490419'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def request(file_id,suffix):
    for attempt in range(5):
        cfg=configparser.ConfigParser(interpolation=None)
        cfg.read('/Users/stevenyang/.config/rclone/rclone.conf')
        token=json.loads(cfg['gdrive']['token'])['access_token']
        try:
            return urllib.request.urlopen(urllib.request.Request(
                'https://www.googleapis.com/drive/v3/files/'+file_id+suffix,
                headers={'Authorization':'Bearer '+token,'X-Goog-User-Project':QUOTA_PROJECT}),timeout=90)
        except urllib.error.HTTPError as error:
            code=error.code
            if code==401 and attempt<4:
                subprocess.run(['rclone','about','gdrive:','--json'],stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,check=True,timeout=60)
                continue
            if code not in (403,429,500,502,503) or attempt==4:
                raise RuntimeError(f'Drive HTTP {code}; retain source') from None
            time.sleep(2**attempt)


def verify(file_id,name,size,sha256,folder=FOLDER,stream=True):
    with request(file_id,'?fields=id,name,size,sha256Checksum,parents,trashed') as response:
        metadata=json.load(response)
    if (metadata.get('id')!=file_id or metadata.get('name')!=name or metadata.get('trashed')
        or metadata.get('parents')!=[folder] or int(metadata.get('size',-1))!=size
        or metadata.get('sha256Checksum')!=sha256):
        raise ValueError('Drive metadata identity/size/hash differs')
    if stream:
        h=hashlib.sha256(); total=0
        with request(file_id,'?alt=media') as response:
            while block:=response.read(4<<20): h.update(block); total+=len(block)
        if total!=size or h.hexdigest()!=sha256:
            raise ValueError('Drive full byte readback differs')
    return {'metadata':metadata,'full_byte_readback':stream,'verified':True}


def bootstrap(instance,file_id):
    root=BASE/str(instance)
    proof=json.loads((root/'BOOTSTRAP_ARCHIVE.json').read_text())
    path=Path(proof['archive'])
    if digest(path)!=proof['sha256'] or path.stat().st_size!=proof['bytes']:
        raise ValueError('Local bootstrap archive changed')
    result=verify(file_id,path.name,proof['bytes'],proof['sha256'])
    write(root/'DRIVE_VERIFIED.json',{**proof,**result,'drive_file_id':file_id,
                                    'folder_id':FOLDER,'status':'full_drive_bytes_verified'})
    print(json.dumps({'instance':instance,'drive_full_readback':True}),flush=True)


def upload_direct(path, name, receipt):
    """Upload into grounded private folder without quota-heavy folder listing.

    One resumable session only; same-range PUT retries are identity bound.
    Session capability stays in this process and is never logged.
    """
    if receipt.exists():
        return json.loads(receipt.read_text())
    cfg=configparser.ConfigParser(interpolation=None)
    cfg.read('/Users/stevenyang/.config/rclone/rclone.conf')
    access=json.loads(cfg['gdrive']['token'])['access_token']
    size=path.stat().st_size
    req=urllib.request.Request('https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable',
        method='POST',data=json.dumps({'name':name,'parents':[FOLDER]}).encode(),
        headers={'Authorization':'Bearer '+access,'X-Goog-User-Project':QUOTA_PROJECT,
                 'Content-Type':'application/json; charset=UTF-8',
                 'X-Upload-Content-Length':str(size),'X-Upload-Content-Type':'application/octet-stream'})
    def open_upload(request):
        for attempt in range(7):
            try:return urllib.request.urlopen(request,timeout=120)
            except urllib.error.HTTPError as error:
                if error.code==308:raise
                transient=error.code in (429,500,502,503)
                if error.code==403:
                    try:
                        body=json.loads(error.read())
                        transient=any(x.get('reason') in ('rateLimitExceeded','userRateLimitExceeded')
                            for x in body.get('error',{}).get('errors',[]))
                    except (ValueError,TypeError):pass
                if not transient or attempt==6:
                    raise RuntimeError(f'Drive upload HTTP {error.code}; source retained') from None
                delay=min(2**(attempt+1),32)
                print(json.dumps({'stage':'drive_quota_backoff','seconds':delay,'name':name}),flush=True)
                time.sleep(delay)
    with open_upload(req) as r: session=r.headers['Location']
    offset=0; output=None
    with path.open('rb') as source:
        while offset<size:
            source.seek(offset); block=source.read(min(64<<20,size-offset))
            request_headers={'Content-Type':'application/octet-stream','X-Goog-User-Project':QUOTA_PROJECT,
                'Content-Range':f'bytes {offset}-{offset+len(block)-1}/{size}'}
            req=urllib.request.Request(session,method='PUT',data=block,headers=request_headers)
            try:
                with open_upload(req) as r:
                    output=json.load(r); offset=size
            except urllib.error.HTTPError as error:
                if error.code==308:
                    confirmed=error.headers.get('Range','')
                    if confirmed!=f'bytes=0-{offset+len(block)-1}':
                        raise RuntimeError('Unexpected resumable acknowledgment; retain source') from None
                    offset+=len(block)
                else:
                    raise RuntimeError(f'Drive upload chunk HTTP {error.code}; source retained') from None
            if offset==size or offset%(256<<20)==0:
                print(json.dumps({'upload_bytes':offset,'total':size,'name':name}),flush=True)
    if not output or not output.get('id'): raise ValueError('No final upload ID')
    write(receipt,{'id':output['id'],'name':name,'bytes':size,'folder_id':FOLDER})
    return json.loads(receipt.read_text())


def relay_raw():
    root=BASE/'raw-fit-and-metric-relay'; root.mkdir(exist_ok=True)
    paths=sorted((PROJECT/'artifacts/offline_study/fit-capture-storage-relocation-20260907-v1').glob('*.json'))
    paths+=sorted((PROJECT/'artifacts/offline_study/window-metrics-storage-relocation-20260907-v1').glob('*.relocation.json'))
    queue=[]
    for path in paths:
        row=json.loads(path.read_text())
        if not row.get('backup_path') or 'sha256' not in row: continue
        if row.get('backup_instance',50189244)!=50189244: raise ValueError('Unexpected source instance')
        queue.append({**row,'source_receipt':str(path.relative_to(PROJECT))})
    if len(queue)!=48: raise ValueError(f'Expected 48 raw objects, found {len(queue)}')
    if not (root/'QUEUE.json').exists(): write(root/'QUEUE.json',queue)
    for index,row in enumerate(queue):
        receipt=root/f'{index:03d}-DRIVE_VERIFIED.json'
        if receipt.exists(): continue
        name=f'raw-{index:03d}-{row["sha256"][:16]}-{Path(row["backup_path"]).name}'
        target=root/name
        if target.exists():
            if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:
                raise ValueError('Existing staged file differs; retained for diagnosis')
        else:
            if shutil.disk_usage(root).free < row['bytes']+(1<<30):
                raise ValueError('Need one GiB free reserve beyond bounded source file')
            cmd,source=command(50189244,row['backup_path'])
            print(json.dumps({'stage':'stopped_disk_download','index':index,'bytes':row['bytes']}),flush=True)
            with (root/f'{index:03d}-rsync.log').open('a') as log:
                subprocess.run(cmd+['-t',source,str(target)],stdout=log,stderr=log,check=True,timeout=1800)
            if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:
                raise ValueError('Stopped source differs from historical verified hash')
        print(json.dumps({'stage':'drive_upload','index':index,'bytes':row['bytes']}),flush=True)
        obj=upload_direct(target,name,root/f'{index:03d}-DRIVE_UPLOADED.json')
        result=verify(obj['id'],name,row['bytes'],row['sha256'])
        write(receipt,{**row,**result,'drive_file_id':obj['id'],'drive_name':name,
                       'folder_id':FOLDER,'instance':50189244})
        # Exact script-created, hash-verified relay file only. Both remote and
        # Google Drive copies remain; receipt identifies recovery by file ID.
        if target.parent!=root or target.is_symlink() or digest(target)!=row['sha256']:
            raise ValueError('Scratch changed before removal')
        target.unlink()
        print(json.dumps({'stage':'drive_fully_verified','index':index,
                          'scratch_removed_bytes':row['bytes']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',nargs=2,metavar=('INSTANCE','FILE_ID'))
    p.add_argument('--relay-raw',action='store_true')
    a=p.parse_args()
    if a.bootstrap: bootstrap(int(a.bootstrap[0]),a.bootstrap[1])
    elif a.relay_raw: relay_raw()
    else: p.error('Choose one operation')
