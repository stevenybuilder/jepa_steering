"""Reconcile a stopped workspace to verified Drive copies and archive restore receipts.

NO lifecycle calls. Any uncovered result, changed size, unknown archive, or
unresolved scientific symlink prevents a release-ready receipt.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import posixpath
import tarfile
from final_storage_release import BASE, PROJECT, write
from final_preservation_drive import digest, upload_direct, verify
from final_preservation_plan import excluded

ALLOWED={50159352,50195621,50231985,50259194,50189244,50239185,50245262}


def prepare(instance):
    if instance not in ALLOWED:raise ValueError('Not an eligible stopped-study instance')
    root=BASE/str(instance)
    if not (root/'ALL_BATCHES_VERIFIED.json').exists():raise ValueError('Backup batches incomplete')
    invpath=sorted(root.glob('*-inventory.json'))[-1]
    inventory=json.loads(invpath.read_text())
    if inventory['returncode']!=0:raise ValueError('Source inventory incomplete')
    files={};references=[]
    receipts=list(root.glob('batch-*/DRIVE_VERIFIED.json'))+list(root.glob('supplemental-*/DRIVE_VERIFIED.json'))
    for receipt in sorted(receipts):
        row=json.loads(receipt.read_text())
        if not row.get('verified') or not row.get('full_byte_readback'):raise ValueError('Unverified backup')
        references.append(str(receipt.relative_to(PROJECT)))
        for name,item in row['manifest'].items():files[name]={**item,'receipt':str(receipt.relative_to(PROJECT))}
    case_receipt=root/'case-alias-corrections/DRIVE_VERIFIED.json'
    if instance in (50189244,50239185,50245262):
        if not case_receipt.exists():raise ValueError('Case-sensitive source audit missing')
        case=json.loads(case_receipt.read_text())
        if not case.get('verified') or not case.get('full_byte_readback'):
            raise ValueError('Case-sensitive source audit incomplete')
        references.append(str(case_receipt.relative_to(PROJECT)))
        for name,item in case['manifest'].items():
            files[name]={**item,'receipt':str(case_receipt.relative_to(PROJECT)),
                         'overrides_case_aliased_batch_copy':True}
    # Historical archives remain in the recovery index, but name/size/mtime
    # agreement cannot establish current source content. Only freshly captured,
    # hashed and Drive-verified batches/supplements/raw receipts above or below
    # may satisfy coverage. A same-size replacement must not earn release-ready.
    if instance==50189244:
        for receipt in (BASE/'raw-fit-and-metric-relay').glob('*-DRIVE_VERIFIED.json'):
            row=json.loads(receipt.read_text())
            if not row.get('verified') or not row.get('full_byte_readback'):continue
            files[row['backup_path'].removeprefix('/workspace/')]={
                'bytes':row['bytes'],'sha256':row['sha256'],'receipt':str(receipt.relative_to(PROJECT))}
    covered={};omitted=[];gaps=[]
    for row in inventory['entries']:
        if not row['mode'].startswith('-'):continue
        name=row['path'];reason=excluded(name)
        if reason:omitted.append({**row,'reason':reason});continue
        if name not in files or files[name]['bytes']!=row['bytes']:gaps.append(row);continue
        covered[name]={**files[name],'inventory_mtime':row['mtime']}
    if gaps:
        print(json.dumps({'instance':instance,'uncovered_count':len(gaps),'uncovered':gaps[:12]}),flush=True)
        raise ValueError('Do not release: uncovered source files')
    links=json.loads((root/'SYMLINK_TARGETS.json').read_text())
    entries={x['path'] for x in inventory['entries']}
    for row in links:
        if excluded(row['path']):continue
        target=row['target']
        if target.startswith('/workspace/'):
            resolved=posixpath.normpath(target.removeprefix('/workspace/'))
        elif target.startswith('/'):
            raise ValueError('Review scientific symlink outside workspace')
        else:resolved=posixpath.normpath(posixpath.join(posixpath.dirname(row['path']),target))
        if resolved not in entries:raise ValueError('Scientific symlink target not inventoried: '+resolved)
    value={'instance':instance,'status':'all_required_workspace_content_accounted_for',
           'covered_files':covered,'covered_count':len(covered),'rebuildable_public_input_exclusions':omitted,
           'symlinks':links,'source_inventory':str(invpath.relative_to(PROJECT)),
           'prior_archive_registry':str((BASE/'PRIOR_DRIVE_ARCHIVES_REVERIFIED.json').relative_to(PROJECT)),
           'new_batch_receipts':references,'no_new_experiments':True,'ready_for_user_authorized_destruction_after_receipt_drive_verification':True}
    coverage=root/'FINAL_RELEASE_COVERAGE.json'
    if coverage.exists():
        if json.loads(coverage.read_text()) != value:
            raise ValueError('Existing coverage differs; retain immutable historical receipt')
    else:write(coverage,value)
    archive=root/f'instance-{instance}-final-restore-index.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
            for path in sorted(root.rglob('*')):
                if path.is_symlink() or not path.is_file() or path==archive:continue
                if any(x in path.relative_to(root).parts for x in ('stage','link-inventory','connector-parts')):continue
                if path.suffix not in ('.json','.txt','.log'):continue
                tar.add(path,arcname=f'instance-{instance}/'+path.relative_to(root).as_posix(),recursive=False)
            tar.add(BASE/'PRIOR_DRIVE_ARCHIVES_REVERIFIED.json',arcname='PRIOR_DRIVE_ARCHIVES_REVERIFIED.json')
            if instance==50189244:
                for path in sorted((BASE/'raw-fit-and-metric-relay').glob('*')):
                    if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.log'):
                        tar.add(path,arcname='raw-fit-and-metric-relay/'+path.name,recursive=False)
            tar.add(PROJECT/'reports/VAST_FINAL_STORAGE_RELEASE_20260910.md',arcname='RESTORE_AND_STATUS.md')
            for path in sorted((PROJECT/'scripts/vast').glob('final_*.py')):tar.add(path,arcname='preservation-code/'+path.name)
    size=archive.stat().st_size;sha=digest(archive)
    uploaded=upload_direct(archive,archive.name,root/'FINAL_INDEX_DRIVE_UPLOADED.json')
    result=verify(uploaded['id'],archive.name,size,sha)
    write(root/'RELEASE_READY.json',{'instance':instance,'files':len(covered),'index_file_id':uploaded['id'],
          'index_name':archive.name,'bytes':size,'sha256':sha,'coverage_sha256':digest(coverage),**result})
    print(json.dumps({'instance':instance,'release_ready':True,'covered_files':len(covered),
                      'drive_restore_index_id':uploaded['id']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('instance',type=int)
    a=p.parse_args();prepare(a.instance)
