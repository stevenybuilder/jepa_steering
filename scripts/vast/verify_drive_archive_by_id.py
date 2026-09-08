"""Read back one grounded private Drive archive without path-discovery queries.

Uses the existing local rclone access token, without printing or changing it.
Never uploads, changes permissions, refreshes credentials, or deletes anything.
Provider SHA256 and full local compressed/member readback must both match.
"""
import argparse
import configparser
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

from backup_results_to_google import verify_archive
from navigation_redistribution_common import digest, write


def verify(file_id, folder_id, archive, manifest_path, output):
    if any(not re.fullmatch(r'[A-Za-z0-9_-]{10,100}', x) for x in (file_id, folder_id)):
        raise ValueError('Require grounded Drive file and parent IDs')
    archive, manifest_path, output = map(Path, (archive, manifest_path, output))
    if not archive.is_file() or archive.is_symlink() or not manifest_path.is_file():
        raise ValueError('Require existing source archive and member manifest')
    manifest = json.loads(manifest_path.read_text())
    expected, size = digest(archive), archive.stat().st_size
    if size > 2 << 30 or shutil.disk_usage(output.parent).free < size + (4 << 30):
        raise ValueError('Two GiB archive bound and four GiB free reserve required')
    verify_archive(['cat', str(archive)], expected, size, manifest)
    output.mkdir(exist_ok=False)
    write(output / 'INTENT.json', {'file_id': file_id, 'parent_folder_id': folder_id,
        'local_archive_sha256': expected, 'archive_bytes': size,
        'manifest_sha256': digest(manifest_path), 'source_sha256': digest(Path(__file__)),
        'read_only_google_api': True, 'no_deletion': True})
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read('/Users/stevenyang/.config/rclone/rclone.conf')
    access = json.loads(cfg['gdrive']['token'])['access_token']
    base = 'https://www.googleapis.com/drive/v3/files/' + file_id
    def request(suffix):
        return urllib.request.urlopen(urllib.request.Request(base + suffix,
            headers={'Authorization': 'Bearer ' + access}), timeout=60)
    try:
        with request('?fields=id,name,size,sha256Checksum,parents') as source:
            metadata = json.load(source)
        if (metadata['id'] != file_id or metadata['parents'] != [folder_id] or
                metadata['name'] != archive.name or int(metadata['size']) != size or
                metadata['sha256Checksum'] != expected):
            raise ValueError('Provider archive metadata/checksum differs')
        write(output / 'PROVIDER_METADATA.json', metadata)
        received = output / archive.name
        count, hashed = 0, hashlib.sha256()
        with request('?alt=media') as source, received.open('xb') as destination:
            for block in iter(lambda: source.read(4 << 20), b''):
                count += len(block)
                if count > size:
                    raise ValueError('Unexpected extra archive bytes')
                hashed.update(block); destination.write(block)
        if count != size or hashed.hexdigest() != expected:
            raise ValueError('Downloaded archive differs')
        verify_archive(['cat', str(received)], expected, size, manifest)
        write(output / 'VERIFIED.json', {'status': 'provider_sha256_and_full_member_readback_verified',
            'file_id': file_id, 'parent_folder_id': folder_id, 'archive_bytes': size,
            'archive_sha256': expected, 'manifest_sha256': digest(manifest_path),
            'files': len(manifest), 'no_local_or_worker_deletions': True})
        print(json.dumps({'status': 'full_drive_archive_readback_verified',
            'bytes': size, 'files': len(manifest), 'sha256': expected}), flush=True)
    except Exception as error:
        # Never serialize request headers or config contents into failure output.
        write(output / 'FAILED.json', {'exception_type': type(error).__name__,
            'http_status': getattr(error, 'code', None), 'local_copy_and_sources_preserved': True})
        raise RuntimeError('Readback failed; existing sources and bounded failure evidence retained') from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('file-id', 'folder-id', 'archive', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    verify(args.file_id, args.folder_id, args.archive, args.manifest, args.output)
