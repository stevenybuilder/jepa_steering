"""Private Drive snapshot of explicit new source/fit/preparation evidence only."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from backup_results_to_google import verify_archive

ROOT=Path(__file__).resolve().parents[2]
DRIVE='gdrive:Research-Archives/JEPA-WM/live-20260908T085400Z/preparation'


def main():
    global DRIVE
    parser=argparse.ArgumentParser();parser.add_argument('--extended',action='store_true');args=parser.parse_args()
    parents=['hmm-fixed-response-20260908-v1','hmm-fixed-response-preparation-20260908-v1',
        'hmm-fixed-response-preparation-20260908-v2','hmm-reference-source-20260908-v1',
        'pusht-coupling-preparation-20260908-v1','pusht-coupling-preparation-20260908-v2']
    if args.extended:
        DRIVE='gdrive:Research-Archives/JEPA-WM/live-20260908T103000Z/preparation'
        parents += ['hmm-fixed-response-preparation-20260908-v3',
            *['droid-fit-inputs-20260908-v'+str(i) for i in range(1,5)],
            'droid-fit-eligibility-20260908-v1','droid-fit-input-stage-20260908-v1',
            'droid-fit-audit-stage-20260908-v1','droid-fit-audit-stage-20260908-v2',
            'droid-fit-completed-inputs-20260908-v1','droid-coupling-preparation-20260908-v1',
            'droid-coupling-preparation-20260908-v2','pusht-reference-delivery-20260908-v1']
    files=set()
    for name in parents:
        root=ROOT/'artifacts/offline_study'/name
        if not root.is_dir():raise ValueError('Missing explicit preparation bundle: '+name)
        files.update(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.'))
    files.update((ROOT/'src/offline_study').glob('*.py'))
    files.update((ROOT/'tests').glob('test_*.py'))
    files.update((ROOT/'scripts/vast').glob('*.py'))
    files.update(ROOT/p for p in ('docs/EXPERIMENT_PLAN.md','docs/HMM_FIXED_RESPONSE_BEHAVIOR.md',
        'docs/PUSHT_COUPLING_BEHAVIOR.md','docs/TEMPORAL_ROUTING_PREPARATION.md',
        'docs/BEHAVIORAL_EVALUATION_AMENDMENT.md','configs/study.json','configs/hmm_fixed_response.json','jax_scaling_notes.md'))
    if args.extended:
        files.update(ROOT/p for p in ('docs/DROID_METHOD_ALIGNMENT.md','docs/DROID_COUPLING_BEHAVIOR.md',
            'docs/DROID_INTERVENTION_FIT_INPUTS.md','configs/droid_assets.json','reports/EXECUTION_STATUS.md'))
    native_csv=ROOT/'artifacts/offline_study/droid-fit-completed-inputs-20260908-v1/droid-fit-audit-20260908-v2/native_paths.csv'
    if any(p.is_symlink() or (p.suffix not in ('.py','.json','.jsonl','.pt','.md','.log') and p!=native_csv) for p in files):
        raise ValueError('Unexpected preparation member; no credentials permitted')
    output=ROOT/('artifacts/offline_study/preparation-drive-backup-20260908-v2' if args.extended else
                 'artifacts/offline_study/preparation-drive-backup-20260908-v1');output.mkdir(exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
    (output/'FILES.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    archive=output/'source-fits-preparation.tar.gz'
    with tarfile.open(archive,'x:gz',format=tarfile.USTAR_FORMAT) as stream:
        for name in manifest:stream.add(ROOT/name,arcname=name,recursive=False)
    digest=sha(archive);size=archive.stat().st_size
    verify_archive(['cat',str(archive)],digest,size,manifest)
    print(json.dumps({'status':'local_snapshot_member_verified','files':len(manifest),'archive_bytes':size}),flush=True)
    subprocess.run(['rclone','mkdir',DRIVE],check=True)
    destination=DRIVE+'/'+archive.name
    subprocess.run(['rclone','copyto',str(archive),destination,'--immutable','--drive-chunk-size','16M',
        '--tpslimit','2','--retries','5','--low-level-retries','10'],check=True)
    verify_archive(['rclone','cat',destination],digest,size,manifest)
    (output/'VERIFIED.json').write_text(json.dumps({'status':'complete_private_drive_archive_member_readback_verified',
        'archive':destination,'archive_sha256':digest,'archive_bytes':size,'files':len(manifest),
        'manifest_sha256':sha(output/'FILES.json'),'full_study_complete':False,'local_files_deleted':False},indent=2)+'\n')
    for name in ('FILES.json','VERIFIED.json'):
        subprocess.run(['rclone','copyto',str(output/name),DRIVE+'/'+name,'--immutable'],check=True)
        subprocess.run(['rclone','check',str(output),DRIVE,'--include',name,'--one-way','--download'],check=True)
    print(json.dumps({'status':'preparation_drive_verified','files':len(manifest)}),flush=True)


if __name__=='__main__':main()
