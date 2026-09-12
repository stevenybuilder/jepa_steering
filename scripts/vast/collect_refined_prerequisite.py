"""Preserve a separate prerequisite root without stopping the busy host."""
import json
import argparse
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

import component_extension_fleet as fleet
from stage_refined_pusht_boundary import LOCAL, REMOTE, INSTANCE, LABEL


def compact_collect(row):
    """Keep every result/source/log; omit reproducible raw download duplicates."""
    from final_preservation_drive import request, FOLDER, upload_direct, verify
    with request(FOLDER, '?fields=id,mimeType,trashed') as stream:
        folder = json.load(stream)
    if folder['mimeType'] != 'application/vnd.google-apps.folder' or folder.get('trashed'):
        raise ValueError('Private destination folder changed')
    ssh = fleet.connection(row)
    inventory = '''import json,hashlib,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]);files=set()
for key in ('code','ops','fit-v1','rank-source-v1','original','cohort','inputs','audit','fit','freeze','conditions','engineering','reference','reference-source','runtime-packages'):
 files.update(p for p in (r/key).rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)
files.update(p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.log'))
files.update((r/'encoder').glob('*.json'))
if (r/'assets/VERIFIED.json').exists():files.add(r/'assets/VERIFIED.json')
files.discard(r/'PRESERVATION_FILES.json')
manifest={}
for p in sorted(files):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4<<20),b''):h.update(c)
 manifest[str(p.relative_to(r))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
proof=r/'PRESERVATION_FILES.json'
if proof.exists():
 if json.loads(proof.read_text())!=manifest:raise ValueError('Terminal source changed')
else:
 with proof.open('x') as f:json.dump(manifest,f,indent=2)
names=list(manifest)+['PRESERVATION_FILES.json']
subprocess.run(['tar','--no-recursion','--null','-czf','-','-C',str(r),'-T','-'],input=b'\\0'.join(n.encode() for n in names)+b'\\0',check=True)
'''
    target = LOCAL / 'complete-worker.tar.gz'
    if not target.exists():
        partial = LOCAL / 'complete-worker.tar.gz.partial'
        with partial.open('wb') as output:
            subprocess.run(ssh + [shlex.join(['python3', '-c', inventory, REMOTE])], stdout=output, check=True, timeout=600)
        with tarfile.open(partial, 'r:gz') as archive:
            proof = json.load(archive.extractfile('PRESERVATION_FILES.json'))
            for member in archive:
                if member.name == 'PRESERVATION_FILES.json': continue
                import hashlib
                h = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(4 << 20), b''): h.update(chunk)
                if member.size != proof[member.name]['bytes'] or h.hexdigest() != proof[member.name]['sha256']:
                    raise ValueError('Collected member differs')
        partial.rename(target)
    digest, size = fleet.sha(target), target.stat().st_size
    metadata = upload_direct(target, Path(REMOTE).name + '.tar.gz', LOCAL / 'UPLOAD.json')
    verified = verify(metadata['id'], metadata['name'], size, digest)
    if not (LOCAL / 'DRIVE_VERIFIED.json').exists():
        fleet.write(LOCAL / 'DRIVE_VERIFIED.json', {'sha256': digest, 'bytes': size, **verified,
            'public_raw_download_duplicates_omitted': True, 'all_result_source_and_log_files_preserved': True})


def main():
    deadline = time.time() + 12 * 3600
    while time.time() < deadline:
        try:
            row = next(r for r in fleet.provider('show', 'instances') if r['id'] == INSTANCE)
            if row['label'] != LABEL or not row['geolocation'].endswith(', US'):
                raise ValueError('Prerequisite host ownership changed')
            ssh = fleet.connection(row)
            ready = subprocess.run(ssh + ['( test -f ' + REMOTE + '/TERMINAL.json && test -f ' +
                REMOTE + '/COMPONENTS_RESUMED.json ) || test -f ' + REMOTE + '/PREPARATION_FAILED.json'], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=30).returncode == 0
            if ready:
                # Existing tested collector requires complete remote output and
                # verifies full Drive readback; this helper never calls stop.
                compact_collect(row)
                root = LOCAL / 'restored'
                root.mkdir(exist_ok=True)
                with tarfile.open(LOCAL / 'complete-worker.tar.gz', 'r:gz') as archive:
                    for member in archive:
                        rel = Path(member.name)
                        if rel.is_absolute() or '..' in rel.parts:
                            raise ValueError('Unsafe archive member')
                        if member.isfile() and (member.name.startswith('fit-v1/') or '/fit-v1/' in member.name):
                            target = root / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists():
                                import shutil
                                with archive.extractfile(member) as source, target.open('xb') as dest:
                                    shutil.copyfileobj(source, dest)
                fleet.write(LOCAL / 'COLLECTED.json', {'time': time.time(),
                    'drive_verified': True, 'host_stopped': False,
                    'behavioral_panel_completed': False})
                print(json.dumps({'prerequisite_preserved': True, 'host_stopped': False}), flush=True)
                return
        except Exception as error:
            print(json.dumps({'collection_retry': type(error).__name__}), flush=True)
        time.sleep(30)
    fleet.write(LOCAL / 'COLLECTION_TIMEOUT.json', {'source_retained': True, 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('pusht', 'droid', 'droid-v2', 'pointmaze', 'wall', 'pusht-behavior',
        'pointmaze-behavior', 'wall-behavior', 'droid-behavior','wall-v2'), default='pusht')
    parser.add_argument('--attempt',type=int,choices=(1,2,3),default=1)
    arguments = parser.parse_args()
    task = arguments.task
    if arguments.attempt!=1 and task not in ('pointmaze-behavior','wall-behavior'):raise ValueError('Unregistered collection attempt')
    if task != 'pusht':
        if task == 'pusht-behavior':
            from stage_refined_pusht_behavior import LOCAL, REMOTE, INSTANCE, LABEL
        elif task.endswith('-behavior'):
            from stage_remaining_refined_behavior import ASSIGNMENTS
            kind = task.removesuffix('-behavior')
            INSTANCE, LABEL, _, _ = ASSIGNMENTS[kind]
            LOCAL = fleet.PROJECT / f'artifacts/offline_study/table-completion-20260911-v1/refined-{kind}-behavior-worker-v{arguments.attempt}'
            REMOTE = f'/workspace/refined-{kind}-behavior-20260911-v{arguments.attempt}'
        elif task in ('droid', 'droid-v2'):
            from stage_refined_droid_boundary import LOCAL, REMOTE, INSTANCE, LABEL
            if task == 'droid-v2':
                LOCAL = LOCAL.with_name('refined-droid-worker-v2')
                REMOTE = '/workspace/refined-droid-prerequisite-20260911-v2'
        else:
            from stage_navigation_refined import ASSIGNMENTS
            kind='wall' if task=='wall-v2' else task
            attempt=2 if task=='wall-v2' else 1
            INSTANCE, LABEL, _, _ = ASSIGNMENTS[kind]
            LOCAL = fleet.PROJECT / f'artifacts/offline_study/table-completion-20260911-v1/refined-{kind}-worker-v{attempt}'
            REMOTE = f'/workspace/refined-nav-{kind}-prerequisite-20260911-v{attempt}'
    main()
