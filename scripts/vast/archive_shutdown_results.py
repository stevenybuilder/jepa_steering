"""Archive explicit stopped-workload roots to private Drive, with full readback.

Leaves source disks untouched. No credentials, datasets or runtime-wide wildcard
copy; callers supply each result/proof/code root to preserve before instance stop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from backup_results_to_google import verify_archive


INVENTORY = r'''
import hashlib,json,os,pathlib,sys
root=pathlib.Path('/workspace/jepa-runtime')
files={}
for label in json.loads(sys.argv[1]):
    target=root/label
    if not target.is_dir(): raise ValueError('Missing explicit result root: '+label)
    for folder,dirs,names in os.walk(target):
        dirs[:]=[d for d in dirs if d != '__pycache__']
        for name in names:
            p=pathlib.Path(folder)/name
            if name.endswith('.pyc'): continue
            if p.is_symlink() or name in ('.env','rclone.conf') or p.suffix in ('.key','.pem'):
                raise ValueError('Unsafe result member: '+str(p))
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
            files[str(p.relative_to(root))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
print(json.dumps(files,sort_keys=True))
'''


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--instance', required=True, type=int)
    p.add_argument('--key', required=True)
    p.add_argument('--drive', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--roots', nargs='+', required=True)
    args = p.parse_args()
    if not args.drive.startswith('gdrive:Research-Archives/JEPA-WM/'):
        raise ValueError('Require the grounded private research archive folder')
    if any('/' in r or r in ('.', '..', 'data', 'checkpoints') for r in args.roots):
        raise ValueError('Explicit top-level result/proof roots only')
    args.output.mkdir(parents=True, exist_ok=False)
    ssh = ['ssh', '-i', args.key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ServerAliveInterval=15', '-p', str(args.port), 'root@' + args.host]
    command = shlex.join(['python', '-c', INVENTORY, json.dumps(args.roots)])
    manifest = json.loads(subprocess.check_output(ssh + [command], text=True))
    write(args.output / 'FILES.json', manifest)
    write(args.output / 'PLAN.json', {'instance': args.instance, 'roots': args.roots,
        'drive': args.drive, 'source_files': len(manifest),
        'source_bytes': sum(r['bytes'] for r in manifest.values()),
        'source_disks_retained': True, 'partial_results_are_not_complete_experiments': True})
    print(json.dumps({'status': 'inventory_complete', 'instance': args.instance,
        'files': len(manifest), 'bytes': sum(r['bytes'] for r in manifest.values())}), flush=True)
    subprocess.run(['rclone', 'mkdir', args.drive], check=True)
    archive = args.drive + '/instance-' + str(args.instance) + '-results.tar.gz'
    tar_cmd = shlex.join(['tar', '--exclude=__pycache__', '--exclude=*.pyc',
        '-C', '/workspace/jepa-runtime', '-czf', '-'] + args.roots)
    source = subprocess.Popen(ssh + [tar_cmd], stdout=subprocess.PIPE)
    upload = subprocess.Popen(['rclone', 'rcat', archive, '--drive-chunk-size', '16M'], stdin=subprocess.PIPE)
    h, size, last = hashlib.sha256(), 0, time.monotonic()
    try:
        for chunk in iter(lambda: source.stdout.read(1024**2), b''):
            h.update(chunk); size += len(chunk); upload.stdin.write(chunk)
            if time.monotonic() - last > 15:
                print(json.dumps({'status': 'uploading', 'instance': args.instance, 'archive_bytes': size}), flush=True)
                last = time.monotonic()
        upload.stdin.close()
        if source.wait() or upload.wait():
            raise RuntimeError('Source transfer or Drive upload failed; do not release instance')
    finally:
        source.stdout.close()
        for proc in (source, upload):
            if proc.poll() is None:
                proc.terminate(); proc.wait()
    print(json.dumps({'status': 'readback_verification', 'instance': args.instance, 'archive_bytes': size}), flush=True)
    verify_archive(['rclone', 'cat', archive], h.hexdigest(), size, manifest)
    # No producer may have changed a source result during this snapshot.
    after = json.loads(subprocess.check_output(ssh + [command], text=True))
    if after != manifest:
        raise ValueError('Source changed during archive; preserve both and retry a stable snapshot')
    write(args.output / 'VERIFIED.json', {'status': 'drive_archive_all_members_verified',
        'instance': args.instance, 'archive': archive, 'archive_sha256': h.hexdigest(),
        'archive_bytes': size, 'files': len(manifest), 'source_unchanged_during_snapshot': True,
        'source_disks_retained': True, 'public_sharing_enabled': False})
    subprocess.run(['rclone', 'copy', str(args.output), args.drive + '/receipt-' + str(args.instance), '--immutable'], check=True)
    subprocess.run(['rclone', 'check', str(args.output), args.drive + '/receipt-' + str(args.instance), '--one-way'], check=True)
    print(json.dumps({'status': 'safe_to_stop_instance', 'instance': args.instance,
        'receipt': str(args.output), 'archive': archive}), flush=True)


if __name__ == '__main__':
    main()
