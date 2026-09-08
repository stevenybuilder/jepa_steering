"""Read back bounded, explicitly uploaded restore files using existing local auth.

No discovery, upload, permission mutation, deletion or credential output.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from navigation_redistribution_common import write

PROJECT = Path(__file__).resolve().parents[2]


def verify(mapping, folder_id, output):
    rows = json.loads(Path(mapping).read_text())
    ids = [row['drive_file_id'] for row in rows]
    if (not rows or len(rows) > 64 or len(ids) != len(set(ids))
            or any(not re.fullmatch(r'[A-Za-z0-9_-]{10,100}', value) for value in [folder_id, *ids])):
        raise ValueError('Require bounded grounded unique file IDs')
    expected = []
    for row in rows:
        path = PROJECT / row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(PROJECT) or path.stat().st_size > 10 << 20:
            raise ValueError('Require bounded regular in-project metadata')
        body = path.read_bytes()
        expected.append({**row, 'bytes':len(body), 'sha256':hashlib.sha256(body).hexdigest()})
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read('/Users/stevenyang/.config/rclone/rclone.conf')
    access = json.loads(cfg['gdrive']['token'])['access_token']
    def request(file_id, suffix):
        for attempt in range(5):
            try:
                return urllib.request.urlopen(urllib.request.Request(
                    'https://www.googleapis.com/drive/v3/files/' + file_id + suffix,
                    headers={'Authorization':'Bearer ' + access}), timeout=60)
            except urllib.error.HTTPError as error:
                if error.code == 403:
                    detail = json.loads(error.read())
                    transient = any(x.get('reason') in ('rateLimitExceeded','userRateLimitExceeded')
                                    for x in detail.get('error',{}).get('errors',[]))
                else: transient = error.code in (429,503)
                if not transient or attempt == 4:
                    raise RuntimeError('Metadata read unavailable; no upload retry or success claim') from None
                time.sleep(2 ** (attempt + 1))
    verified = []
    for row in expected:
        with request(row['drive_file_id'], '?fields=id,name,size,sha256Checksum,parents') as source:
            meta = json.load(source)
        if (meta['id'] != row['drive_file_id'] or meta['name'] != row['name']
                or meta['parents'] != [folder_id] or int(meta['size']) != row['bytes']
                or meta['sha256Checksum'] != row['sha256']):
            raise ValueError('Restore metadata provider binding differs')
        with request(row['drive_file_id'], '?alt=media') as source:
            body = source.read(row['bytes'] + 1)
        if len(body) != row['bytes'] or hashlib.sha256(body).hexdigest() != row['sha256']:
            raise ValueError('Restore metadata full byte readback differs')
        verified.append(row)
    write(Path(output), {'status':'all_restore_files_provider_and_full_byte_readback_verified',
        'folder_id':folder_id, 'files':verified, 'file_count':len(verified), 'no_deletions':True})
    print(json.dumps({'status':'restore_metadata_full_readback_verified','files':len(verified)}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mapping','folder-id','output'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); verify(args.mapping,args.folder_id,args.output)
