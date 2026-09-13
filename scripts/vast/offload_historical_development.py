"""Authorized, manifested Drive offload of three exact historical artifact roots.

Never follows symlinks, selects by result value, or removes an unverified file.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / 'artifacts/geometry_map', ROOT / 'artifacts/cgs_pilot',
           ROOT / 'artifacts/offline_study/table-completion-20260911-v1']
REMOTE = 'gdrive:Research-Archives/JEPA-WM/Laptop-Offload-20260912/historical-development'
RECEIPTS = ROOT / 'artifacts/offline_study/laptop-offload-20260912'
LIMIT = 512 * 1024**2
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"(?:private_key|refresh_token|access_token)"\s*:\s*"[^"\s]{16,}')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def event(**values):
    print(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}), flush=True)


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)


def tracked():
    output = subprocess.check_output(['git', 'ls-files', '-z', '--', *map(str, SOURCES)], cwd=ROOT)
    return {ROOT / os.fsdecode(name) for name in output.split(b'\0') if name}


def unsafe(path, source):
    relative = path.relative_to(source)
    if path.is_symlink() or any(ord(c) < 32 for c in str(relative)):
        return 'symlink or control-character filename'
    names = {name.lower() for name in relative.parts}
    if (names & {'.ssh', '.aws', '.env', 'rclone.conf', 'credentials', 'credentials.json',
                 'application_default_credentials.json', 'id_rsa', 'id_ed25519'}
            or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}):
        return 'credential-like name'
    if path.suffix.lower() in {'.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.txt', '.log', '.env'}:
        with path.open('rb') as stream:
            tail = b''
            for block in iter(lambda: stream.read(1024**2), b''):
                if SECRET.search(tail + block):
                    return 'credential-like content'
                tail = block[-512:]
    return None


def run(command, log):
    result = subprocess.run(command, stdout=log, stderr=log)
    log.flush()
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command[0:2]}')


def main():
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    session = RECEIPTS / time.strftime('session-%Y%m%dT%H%M%SZ', time.gmtime())
    session.mkdir(exist_ok=False)
    tracked_paths = tracked()
    inventory, skipped = [], []
    for source in SOURCES:
        if not source.is_dir() or source.is_symlink():
            raise ValueError('Missing or symlinked authorized source: ' + str(source))
        for directory, folders, files in os.walk(source, followlinks=False):
            for name in list(folders):
                path = Path(directory) / name
                if path.is_symlink():
                    skipped.append({'path': str(path), 'reason': 'symlink directory'})
                    folders.remove(name)
            for name in files:
                path = Path(directory) / name
                if path in tracked_paths:
                    skipped.append({'path': str(path), 'reason': 'tracked file'})
                    continue
                reason = unsafe(path, source)
                if reason:
                    skipped.append({'path': str(path), 'reason': reason})
                    continue
                stat = path.stat()
                inventory.append((source, path, stat.st_size))
    write(session / 'SKIPPED.json', skipped)
    # Medium-sized folders produce useful freed space early; retain original paths.
    groups = {}
    for source, path, size in inventory:
        relative = path.relative_to(source)
        group = (source, relative.parts[0] if len(relative.parts) > 1 else '_top_level')
        groups.setdefault(group, []).append((path, size))
    ordered = sorted(groups.items(), key=lambda pair: (sum(n for _, n in pair[1]) > LIMIT,
                                                      -min(sum(n for _, n in pair[1]), LIMIT)))
    total_bytes = sum(size for _, _, size in inventory)
    event(status='inventory', files=len(inventory), bytes=total_bytes, skipped=len(skipped), session=str(session))
    total_freed, chunk_number = 0, 0
    for (source, group), paths in ordered:
        batches, batch, batch_size = [], [], 0
        for path, size in sorted(paths):
            if batch and (batch_size + size > LIMIT or len(batch) >= 256):
                batches.append(batch); batch, batch_size = [], 0
            batch.append(path); batch_size += size
        if batch:
            batches.append(batch)
        for batch in batches:
            chunk_number += 1
            chunk = session / f'chunk-{chunk_number:04d}'
            chunk.mkdir()
            members = []
            for path in batch:
                before = path.stat()
                digest = sha(path)
                after = path.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    skipped.append({'path': str(path), 'reason': 'changed during manifest'})
                    continue
                members.append({'path': str(path.relative_to(source)), 'bytes': after.st_size,
                                'sha256': digest, 'inode': after.st_ino, 'mtime_ns': after.st_mtime_ns})
            if not members:
                continue
            manifest = {'source': str(source), 'remote': REMOTE + '/' + source.name,
                        'historical_development_not_fresh_confirmation': True, 'members': members}
            write(chunk / 'MANIFEST.json', manifest)
            with (chunk / 'files.txt').open('x') as stream:
                stream.write(''.join(row['path'] + '\n' for row in members))
            byte_count = sum(row['bytes'] for row in members)
            event(status='copying', chunk=chunk_number, group=group, bytes=byte_count, total_freed=total_freed)
            with (chunk / 'rclone.log').open('x') as log:
                metadata_remote = REMOTE + '/_manifests/' + session.name + '/' + chunk.name
                run(['rclone', 'copyto', str(chunk / 'MANIFEST.json'), metadata_remote + '/MANIFEST.json', '--immutable'], log)
                common = ['--files-from-raw', str(chunk / 'files.txt')]
                run(['rclone', 'copy', str(source), manifest['remote'], *common, '--immutable', '--transfers', '8', '--checkers', '8', '--stats', '15s'], log)
                event(status='uploaded', chunk=chunk_number, bytes=byte_count, total_freed=total_freed)
                run(['rclone', 'check', str(source), manifest['remote'], *common, '--download', '--one-way', '--checkers', '8'], log)
                now_tracked = tracked()
                removable, changed = [], []
                for row in members:
                    path = source / row['path']
                    stat = path.lstat()
                    if (path.is_symlink() or path in now_tracked
                            or (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (row['inode'], row['bytes'], row['mtime_ns'])
                            or sha(path) != row['sha256']):
                        changed.append(row['path'])
                    else:
                        removable.append(row)
                verified = {'download_check_passed': True, 'verified_bytes': byte_count,
                            'eligible_for_removal': [r['path'] for r in removable], 'changed_preserved': changed}
                write(chunk / 'VERIFIED.json', verified)
                run(['rclone', 'copyto', str(chunk / 'VERIFIED.json'), metadata_remote + '/VERIFIED.json', '--immutable'], log)
                freed = 0
                for row in removable:
                    path = source / row['path']
                    # Final identity check immediately before unlink, never recursive deletion.
                    stat = path.lstat()
                    if (not path.is_symlink() and (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (row['inode'], row['bytes'], row['mtime_ns'])):
                        path.unlink()
                        freed += row['bytes']
                total_freed += freed
                write(chunk / 'REMOVED.json', {'removed_bytes': freed, 'total_removed_bytes': total_freed,
                                               'recovery': manifest['remote'], 'manifest': metadata_remote + '/MANIFEST.json'})
                run(['rclone', 'copyto', str(chunk / 'REMOVED.json'), metadata_remote + '/REMOVED.json', '--immutable'], log)
            event(status='verified_and_freed', chunk=chunk_number, uploaded_bytes=byte_count,
                  verified_bytes=byte_count, freed_bytes=freed, total_freed=total_freed)
    write(session / 'FINAL_SKIPPED.json', skipped)
    event(status='complete', total_freed=total_freed, skipped=len(skipped), session=str(session))


if __name__ == '__main__':
    main()
