"""Copy explicit remaining result files in bounded batches, verify Drive, retain receipts.

No GPU start or rental deletion. No public dataset or Python-env recopy. Small
files are archived together; large unique captures/checkpoints stay byte-exact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from final_storage_release import BASE, command, write
from final_preservation_drive import digest, upload_direct, verify
from final_preservation_plan import excluded
from upload_pause_archive import verify_stream


def existing_content(size, sha):
    """Exact content dedup, never filename/size-only dedup."""
    for receipt in BASE.glob('*/batch-*/DRIVE_VERIFIED.json'):
        try:value=json.loads(receipt.read_text())
        except json.JSONDecodeError:
            # Another bounded batch may still be writing its receipt. Do not
            # reuse it until a future read observes the complete JSON.
            continue
        if (value.get('kind')=='file' and value.get('bytes')==size and value.get('sha256')==sha
                and value.get('full_byte_readback') and value.get('drive_file_id')):
            meta=value['metadata']
            verify(meta['id'],meta['name'],int(meta['size']),meta['sha256Checksum'],meta['parents'][0],stream=False)
            return {'verified':True,'full_byte_readback':True,'restored_from_exact_duplicate':str(receipt),
                    'metadata':meta,'drive_file_id':meta['id']}
    registry=BASE/'PRIOR_DRIVE_ARCHIVES_REVERIFIED.json'
    if registry.exists():
        from final_storage_release import PROJECT
        for entry in json.loads(registry.read_text()):
            proof=PROJECT/entry['proof']; manifest=proof.parent/'FILES.json'
            if not manifest.exists():continue
            for name,item in json.loads(manifest.read_text()).items():
                if isinstance(item,dict) and item.get('bytes')==size and item.get('sha256')==sha:
                    meta=entry['metadata']
                    verify(meta['id'],meta['name'],int(meta['size']),meta['sha256Checksum'],meta['parents'][0],stream=False)
                    return {'verified':True,'full_byte_readback':True,
                        'verification_method':'historical full member readback plus fresh archive hash and current source hash',
                        'restored_from_archive':meta,'archive_member':name,'prior_member_proof':entry['proof']}
    return None


def prepare(instance):
    root=BASE/str(instance)
    p=root/'BATCH_PLAN.json'
    if p.exists():return json.loads(p.read_text())
    plan=json.loads((root/'PRESERVATION_CANDIDATES.json').read_text())
    rows=plan['candidates_needing_backup_or_exact_hash_reconciliation']
    # Recapture historical matches too: path/size alone cannot prove unchanged
    # source content. Large files still use exact SHA-based Drive dedup AFTER
    # capture, not an assumed old copy. Existing immutable plans are not edited.
    rows+= plan['historical_same_path_size_matches_require_final_review']
    selected={x['path']:x for x in rows if not excluded(x['path'])}
    raw=set()
    if instance==50189244:
        rawplan=BASE/'raw-fit-and-metric-relay/QUEUE.json'
        raw={x['backup_path'].removeprefix('/workspace/') for x in json.loads(rawplan.read_text())}
    selected={k:v for k,v in selected.items() if k not in raw}
    batches=[];small=[];total=0
    for name,row in sorted(selected.items()):
        if '\n' in name or '\0' in name or '..' in Path(name).parts:
            raise ValueError('Unsafe source path')
        if row['bytes']>=128<<20:
            batches.append({'kind':'file','files':[row]});continue
        if total+row['bytes']>256<<20 and small:
            batches.append({'kind':'archive','files':small});small=[];total=0
        small.append(row);total+=row['bytes']
    if small:batches.append({'kind':'archive','files':small})
    batches.sort(key=lambda b: (b['kind']=='file',sum(x['bytes'] for x in b['files'])))
    value={'instance':instance,'batches':batches,'raw_relay_elsewhere':sorted(raw),
           'files':len(selected),'bytes':sum(x['bytes'] for x in selected.values())}
    write(p,value);return value


def run(instance, prepare_only=False, indices=None, finalize=True):
    root=BASE/str(instance); plan=prepare(instance)
    for index,batch in enumerate(plan['batches']):
        if indices is not None and index not in indices:continue
        work=root/f'batch-{index:03d}';work.mkdir(exist_ok=True)
        done=work/'DRIVE_VERIFIED.json'
        if done.exists():continue
        stage=work/'stage';stage.mkdir(exist_ok=True)
        archive=work/f'instance-{instance}-batch-{index:03d}.tar.gz'
        final=work/'FILES.json'
        entries=batch['files']; total=sum(x['bytes'] for x in entries)
        if not final.exists():
            reserve=total*(2 if batch['kind']=='archive' else 1)+(1<<30)
            if shutil.disk_usage(work).free < reserve:
                raise ValueError(f'Insufficient scratch reserve for instance {instance} batch {index}')
            names=b'\0'.join(x['path'].encode() for x in entries)+b'\0'
            cmd,source=command(instance,'/workspace/')
            print(json.dumps({'instance':instance,'batch':index,'stage':'download','bytes':total}),flush=True)
            with (work/'rsync.log').open('a') as log:
                subprocess.run(cmd+['-rt','--from0','--files-from=-',source,str(stage)+'/'],
                    input=names,stdout=log,stderr=log,check=True,timeout=1800)
            manifest={}
            for row in entries:
                path=stage/row['path']
                if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes']:
                    raise ValueError('Transferred inventory size/type differs')
                manifest[row['path']]={'bytes':path.stat().st_size,'sha256':digest(path)}
            write(final,manifest)
        else:manifest=json.loads(final.read_text())
        if batch['kind']=='archive':
            if not archive.exists():
                with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
                    for name in sorted(manifest):tar.add(stage/name,arcname=name,recursive=False)
                    tar.add(final,arcname='FINAL_PRESERVATION_FILES.json',recursive=False)
            sourcefile=archive; name=archive.name
        else:
            sourcefile=stage/entries[0]['path']
            name=f'instance-{instance}-batch-{index:03d}-'+sourcefile.name
        size=sourcefile.stat().st_size; sha=digest(sourcefile)
        if batch['kind']=='archive':
            expected={**manifest,'FINAL_PRESERVATION_FILES.json':{'bytes':final.stat().st_size,'sha256':digest(final)}}
            with archive.open('rb') as f:verify_stream(f,sha,size,expected)
        print(json.dumps({'instance':instance,'batch':index,'stage':'upload','bytes':size}),flush=True)
        multipart=work/'DRIVE_PARTS_VERIFIED.json'
        if prepare_only and not (work/'DRIVE_UPLOADED.json').exists() and not multipart.exists():
            print(json.dumps({'needs_connector':True,'path':str(sourcefile),'name':name,
                  'bytes':size,'sha256':sha,'receipt_directory':str(work)}),flush=True)
            return
        duplicate=existing_content(size,sha) if batch['kind']=='file' else None
        if duplicate:
            result=duplicate;uploaded={'id':duplicate.get('drive_file_id')}
        elif multipart.exists():
            result=json.loads(multipart.read_text())
            if result['bytes']!=size or result['sha256']!=sha or not result['full_byte_readback']:
                raise ValueError('Multipart source binding differs')
            uploaded={'id':None}
        else:
            uploaded=upload_direct(sourcefile,name,work/'DRIVE_UPLOADED.json')
            result=verify(uploaded['id'],name,size,sha)
        write(done,{'instance':instance,'batch':index,'kind':batch['kind'],'manifest':manifest,
                    'drive_file_id':uploaded['id'],'drive_name':name,'bytes':size,'sha256':sha,**result})
        # Only exact script-created staging paths and archive after full Drive
        # readback. Original disks and every manifest/receipt remain.
        for name,item in manifest.items():
            path=stage/name
            # macOS case-insensitive scratch may alias two Linux source paths.
            # Final release requires a separate, numbered-source case audit;
            # matching archived local hashes alone cannot prove source equality.
            if not path.exists() and not path.is_symlink():continue
            if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()) or digest(path)!=item['sha256']:
                raise ValueError('Staging changed before verified cleanup')
            path.unlink()
        if archive.exists():
            if digest(archive)!=sha: raise ValueError('Archive changed before cleanup')
            archive.unlink()
        print(json.dumps({'instance':instance,'batch':index,'stage':'drive_full_readback_verified',
                          'source_files':len(manifest),'scratch_removed':True}),flush=True)
    if not finalize:return
    for index in range(len(plan['batches'])):
        if not (root/f'batch-{index:03d}'/'DRIVE_VERIFIED.json').exists():
            raise ValueError('Cannot finalize incomplete batch set')
    if (root/'ALL_BATCHES_VERIFIED.json').exists():return
    write(root/'ALL_BATCHES_VERIFIED.json',{'instance':instance,'batches':len(plan['batches']),
          'files':plan['files'],'bytes':plan['bytes'],'source_disks_untouched':True,
          'prior_archives_raw_relay_and_symlink_review_still_required':True})


def run_parallel(instance, workers):
    """Disjoint stopped-disk batches; no concurrent producer of the same batch."""
    if not 1<=workers<=3:raise ValueError('Require one to three bounded transfers')
    plan=prepare(instance);root=BASE/str(instance)
    pending=[i for i in range(len(plan['batches']))
             if not (root/f'batch-{i:03d}'/'DRIVE_VERIFIED.json').exists()]
    reserves=sorted((sum(x['bytes'] for x in plan['batches'][i]['files'])
                     *(2 if plan['batches'][i]['kind']=='archive' else 1)
                     for i in pending),reverse=True)
    if shutil.disk_usage(root).free<sum(reserves[:workers])+(1<<30):
        raise ValueError('Insufficient aggregate reserve for parallel transfers')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,instance,False,{i},False) for i in pending]
        for future in futures:future.result()
    run(instance,indices=set())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('instances',type=int,nargs='+')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    for instance in a.instances:
        if a.plan_only:
            x=prepare(instance);print(json.dumps({k:x[k] for k in ('instance','files','bytes')}|{'batches':len(x['batches'])}),flush=True)
        elif a.workers!=1:
            if a.prepare_only:p.error('Parallel mode cannot prepare connector scratch')
            run_parallel(instance,a.workers)
        else:run(instance,a.prepare_only)
