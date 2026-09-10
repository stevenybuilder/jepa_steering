"""Generate auditable preservation candidates from stopped-disk inventory.

No deletion/transfer. Only explicitly verified historical Drive snapshots are
used for coverage, bound by instance, exact source-relative path and byte size.
"""
import json
from pathlib import Path
import re
from final_storage_release import BASE, PROJECT, write


def excluded(path):
    parts=Path(path).parts
    if any(re.fullmatch(r'jepa-(?:[a-z0-9-]*python|py311)',x) for x in parts):
        return 'rebuildable_python_environment'
    if 'maze-python' in parts:
        return 'rebuildable_relocated_python_environment'
    if any(x in parts for x in ('.git','__pycache__','.cache')) or path.endswith('.pyc'):
        return 'rebuildable_cache'
    if any(x in parts for x in ('.env','rclone.conf')) or path.endswith(('.key','.pem')):
        return 'credential_not_research'
    if re.search(r'(navigation|droid|pusht-planning)-assets-[^/]+/(downloads|extracted)/',path):
        return 'released_public_dataset_or_checkpoint_cache_preserve_download_receipts'
    if '/fixed-response-assets-' in path and '/metaworld/data/' in path and path.endswith('.parquet'):
        return 'released_metaworld_parquet_cache'
    if '/official_hf/' in path and path.endswith('.safetensors'):
        return 'released_dinov3_checkpoint_preserve_conversion_and_hash_receipts'
    if re.fullmatch(r'jepa_wm_(metaworld|wall|pointmaze|pusht|droid)\.pth\.tar',parts[-1]):
        return 'released_official_checkpoint_preserve_hash_and_download_receipts'
    return None


def normalize(name):
    name=name.removeprefix('/workspace/').removeprefix('workspace/')
    return name if name.startswith(('jepa-runtime/','jepa_steering/')) else 'jepa-runtime/'+name


def plan(instance):
    root=BASE/str(instance)
    inventories=sorted(root.glob('*-inventory.json'))
    if not inventories: raise ValueError('No inventory')
    inv=json.loads(inventories[-1].read_text())
    if inv['returncode']!=0: raise ValueError('Inventory incomplete')
    known={}; proofs=[]
    for proofpath in sorted((PROJECT/'artifacts/offline_study').glob('**/VERIFIED.json')):
        proof=json.loads(proofpath.read_text())
        if (proof.get('instance')!=instance or not str(proof.get('archive','')).startswith('gdrive:')
            or proof.get('status') not in ('immutable_live_snapshot_all_members_verified','drive_archive_all_members_verified')):
            continue
        manifest=proofpath.parent/'FILES.json'
        if not manifest.exists(): continue
        proofs.append(str(proofpath.relative_to(PROJECT)))
        for name,value in json.loads(manifest.read_text()).items():
            if isinstance(value,dict) and 'sha256' in value and 'bytes' in value:
                known.setdefault(normalize(name),[]).append({**value,'proof':str(proofpath.relative_to(PROJECT)),
                    'archive':proof['archive'],'archive_bytes':proof['archive_bytes'],
                    'archive_sha256':proof['archive_sha256']})
    covered=[]; omitted=[]; candidates=[]; links=[]
    for row in inv['entries']:
        if row['mode'].startswith('l'): links.append(row);continue
        if not row['mode'].startswith('-'):continue
        reason=excluded(row['path'])
        if reason: omitted.append({**row,'reason':reason});continue
        previous=[x for x in known.get(row['path'],[]) if x['bytes']==row['bytes']]
        if previous:
            # Candidate coverage only: must reconcile later timestamps/producer
            # states and refresh the exact archive objects before release.
            covered.append({**row,'historical_archive':previous[-1]})
        else:candidates.append(row)
    result={'instance':instance,'fresh_inventory':str(inventories[-1].relative_to(PROJECT)),
            'candidates_needing_backup_or_exact_hash_reconciliation':candidates,
            'historical_same_path_size_matches_require_final_review':covered,
            'excluded_rebuildable_or_public_inputs':omitted,'symlinks_require_target_review':links,
            'prior_drive_proofs':proofs,'authorizes_deletion':False}
    write(root/'PRESERVATION_CANDIDATES.json',result)
    print(json.dumps({'instance':instance,'candidate_files':len(candidates),
         'candidate_bytes':sum(x['bytes'] for x in candidates),'historical_matches':len(covered),
         'links':len(links),'largest_candidates':sorted(candidates,key=lambda x:-x['bytes'])[:8]}),flush=True)


if __name__=='__main__':
    import sys
    for arg in sys.argv[1:]: plan(int(arg))
