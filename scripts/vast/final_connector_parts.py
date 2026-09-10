"""Prepare/verify bounded Drive connector uploads, binding exact ordered source bytes."""
import argparse
import hashlib
import json
from pathlib import Path
from final_preservation_drive import BASE, FOLDER, digest, verify, write

CHUNK=96<<20


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--receipt-dir',type=Path,required=True)
    p.add_argument('--index',type=int);p.add_argument('--file-id');p.add_argument('--finalize',action='store_true')
    a=p.parse_args();source=a.source;root=a.receipt_dir
    if source.is_symlink() or not source.resolve().is_relative_to(BASE.resolve()) or not root.resolve().is_relative_to(BASE.resolve()):
        raise ValueError('Require task-owned source and receipt directory')
    size=source.stat().st_size
    if a.file_id and a.index is None:
        result=verify(a.file_id,a.name,size,digest(source))
        write(root/'DRIVE_UPLOADED.json',{'id':a.file_id,'name':a.name,'bytes':size,'folder_id':FOLDER})
        print(json.dumps({'whole_file_verified':True,'id':a.file_id}),flush=True);return
    parts=root/'connector-parts';parts.mkdir(exist_ok=True)
    if a.finalize:
        rows=[json.loads((parts/f'{i:04d}-VERIFIED.json').read_text()) for i in range((size+CHUNK-1)//CHUNK)]
        for i,row in enumerate(rows):
            if row['offset']!=i*CHUNK or row['bytes']!=min(CHUNK,size-i*CHUNK) or row['name']!=a.name+f'.part-{i:04d}' or not row['full_byte_readback']:
                raise ValueError('Incomplete or incorrectly ordered parts')
        # Rehash current source slices to bind the independently read-back
        # Drive parts to the complete original file, including ordering.
        full=hashlib.sha256()
        with source.open('rb') as f:
            for row in rows:
                block=f.read(row['bytes']);full.update(block)
                if hashlib.sha256(block).hexdigest()!=row['sha256']:raise ValueError('Source part differs')
        value={'name':a.name,'bytes':size,'sha256':full.hexdigest(),'parts':rows,'folder_id':FOLDER,
               'full_byte_readback':True,'verified':True,'restore':'Concatenate parts in listed order; SHA256 must match sha256.'}
        write(root/'DRIVE_PARTS_VERIFIED.json',value)
        print(json.dumps({'multipart_complete':True,'parts':len(rows),'bytes':size}),flush=True);return
    i=a.index
    if i is None or i<0 or i*CHUNK>=size:raise ValueError('Invalid part index')
    name=a.name+f'.part-{i:04d}';path=parts/name;receipt=parts/f'{i:04d}.json'
    if a.file_id:
        row=json.loads(receipt.read_text())
        if digest(path)!=row['sha256']:raise ValueError('Part changed')
        result=verify(a.file_id,name,row['bytes'],row['sha256'])
        write(parts/f'{i:04d}-VERIFIED.json',{**row,**result,'drive_file_id':a.file_id})
        path.unlink()
        print(json.dumps({'part_verified':i,'drive_file_id':a.file_id}),flush=True);return
    if (parts/f'{i:04d}-VERIFIED.json').exists():
        print(json.dumps({'already_verified':i}),flush=True);return
    if not receipt.exists():
        with source.open('rb') as f,path.open('xb') as o:
            f.seek(i*CHUNK);block=f.read(CHUNK);o.write(block)
        write(receipt,{'name':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest(),
                       'offset':i*CHUNK,'source':str(source)})
    print(json.dumps({'path':str(path),**json.loads(receipt.read_text())}),flush=True)


if __name__=='__main__': main()
