"""Copy paused output records and latest resumable checkpoints; retain source disks."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from backup_results_to_google import verify_archive
from snapshot_live_results import HASH_FILES

PROJECT = Path(__file__).resolve().parents[2]
HOSTS = {'tx': (50259194, 44474, '45.23.135.240'), 'ne': (50231985, 32259, '38.65.239.11'),
         'in': (50205763, 17755, '99.22.20.240'), 'nj': (50239185, 50578, '71.104.167.38')}


def select(worker):
    from pathlib import Path
    root = Path('/workspace/jepa-runtime')
    selected = set()
    retained = []
    runtime_links = {}

    def add(p):
        if p.is_symlink() or not p.is_file() or '..' in p.parts:
            raise ValueError('Unsafe member ' + str(p))
        if p.name in ('rclone.conf', '.env') or p.suffix in ('.key', '.pem'):
            raise ValueError('Credential-like member')
        selected.add(str(p.relative_to(root)))

    def tree(label):
        path = root / label
        if not path.exists():
            return
        paths = list(path.rglob('*')) if path.is_dir() else [path]
        for p in paths:
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            if p.is_symlink() and p.parent == root / 'combined-full-planner-tx3-20260908-v2/lib' and p.name in ('libEGL.so', 'libOpenGL.so'):
                if p.resolve().parent != p.parent:
                    raise ValueError('Runtime link escaped private library directory')
                runtime_links[str(p.relative_to(root))] = str(p.resolve().relative_to(root))
                continue
            if p.name.startswith('jepa-') and (p.name.endswith('.pth.tar') or p.suffix == '.tmp'):
                retained.append(str(p.relative_to(root)))
            else:
                add(p)

    tree('core-priority-pause-20260908-v1')
    if worker in ('tx', 'ne', 'in'):
        base = 'navigation-redistribution-20260908-v2'
        for p in (root / base).iterdir():
            if p.is_file(): add(p)
        for name in ('results', 'workers', 'early-wall-engineering'):
            tree(base + '/' + name)
    if worker == 'in':
        tree('navigation-recovery-in0-20260908-v1')
        tree('combined-fit-stimulus-preparation-20260908-v1')
    if worker == 'ne':
        tree('navigation-coupling-behavior-20260908-v1')
    if worker == 'tx':
        for label in ('pointmaze-corrected-queue-20260908-v1', 'combined-full-planner-tx3-20260908-v1',
                      'combined-full-planner-tx3-20260908-v2'):
            tree(label)
        tree('combined-engineering-tx3-20260908-v1/check')
        tree('combined-engineering-tx3-20260908-v1/run')
    if worker == 'nj':
        for label in ('wall-training-history-resume-20260908-v1', 'wall-training-history-20260907-v1',
                      'wall-history-code-v1', 'wall-training-inputs-20260907-v1',
                      'wall-training-accumulation-pilot-20260907-v2', 'pusht-planning-native-20260908-v1'):
            tree(label)
        for p in root.glob('wall-history*log'):
            add(p)
    # Each interrupted training stream keeps a complete resume checkpoint off-worker;
    # every older checkpoint stays on the retained lease volume, never destroyed.
    checkpoint_dirs = {str((root / name).parent) for name in retained if name.endswith('.pth.tar')}
    for directory in checkpoint_dirs:
        paths = list(Path(directory).glob('jepa-e*.pth.tar'))
        if paths:
            latest = max(paths, key=lambda p: int(p.name.split('jepa-e')[1].split('.')[0]))
            add(latest)
            retained.remove(str(latest.relative_to(root)))
    return {'selected': sorted(selected), 'runtime_link_reconstruction': runtime_links,
            'older_checkpoints_and_temporary_files_retained_on_source': sorted(retained)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=HOSTS, required=True)
    args = parser.parse_args()
    instance, port, address = HOSTS[args.worker]
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-p', str(port), 'root@' + address]
    output = PROJECT / 'artifacts/offline_study/core-priority-pause-20260908-v1' / args.worker
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Never replace prior preservation receipts')
    def write(name, value):
        with (output / name).open('x') as f: json.dump(value, f, indent=2, sort_keys=True)
    code = inspect.getsource(select) + '\nimport json; print(json.dumps(select(' + repr(args.worker) + ')))'
    scope = json.loads(subprocess.check_output(ssh + [shlex.join(['/usr/bin/python3', '-c', code])], text=True))
    write('SCOPE.json', scope)
    names = scope['selected']
    def inventory():
        return json.loads(subprocess.check_output(ssh + [shlex.join(['/usr/bin/python3', '-c', HASH_FILES])],
                          input=json.dumps(names), text=True, timeout=180))
    manifest = inventory()
    write('FILES.json', manifest)
    size = sum(r['bytes'] for r in manifest.values())
    if shutil.disk_usage(output).free < size + 4 * 1024**3:
        raise ValueError('Local four-GiB reserve would be breached; do not transfer')
    archive = output / (args.worker + '-paused-evidence.tar.gz')
    print(json.dumps({'worker': args.worker, 'files': len(names), 'uncompressed_bytes': size}), flush=True)
    with tempfile.TemporaryFile() as listing, archive.open('xb') as stream:
        listing.write(b'\0'.join(n.encode() for n in names) + b'\0'); listing.seek(0)
        subprocess.run(ssh + ['tar -C /workspace/jepa-runtime --hard-dereference --no-recursion --null -czf - -T -'],
                       stdin=listing, stdout=stream, check=True, timeout=1200)
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    archive_hash = h.hexdigest()
    verify_archive(['cat', str(archive)], archive_hash, archive.stat().st_size, manifest)
    if inventory() != manifest:
        raise ValueError('Paused source changed during backup')
    write('LOCAL_VERIFIED.json', {'instance': instance, 'files': len(names), 'source_rechecked': True,
        'archive_sha256': archive_hash, 'archive_bytes': archive.stat().st_size,
        'older_training_checkpoints_retained_on_source': len(scope['older_checkpoints_and_temporary_files_retained_on_source']),
        'source_disk_must_not_be_destroyed': True, 'drive_verified': False})
    print(json.dumps({'status': 'local_full_member_verification_passed', 'archive': str(archive),
                      'bytes': archive.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
