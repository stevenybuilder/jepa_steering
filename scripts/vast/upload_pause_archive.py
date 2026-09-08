"""Direct resumable Drive upload and streamed full readback without path queries."""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request

from backup_results_to_google import HashReader, digest

FOLDER = '12r-UzKMTuyYm4wXKl9xPb3r5dcjfwsYr'
BASE = Path(__file__).resolve().parents[2] / 'artifacts/offline_study/core-priority-pause-20260908-v1'


def connector_part(root, archive, proof, index, file_id=None):
    """Prepare one <=96MiB connector upload, or verify its ID and remove that scratch copy."""
    chunk = 96 << 20
    offset = index * chunk
    if index < 0 or offset >= proof['archive_bytes']:
        raise ValueError('Part index outside source')
    output = root / 'connector-parts-v1'; output.mkdir(exist_ok=True)
    name = archive.name + f'.part-{index:04d}'
    path, receipt = output / name, output / f'part-{index:04d}.json'
    if file_id is None:
        if shutil.disk_usage(root).free < chunk + (1 << 30):
            raise ValueError('Insufficient reserve for one bounded scratch part')
        with archive.open('rb') as source, path.open('xb') as destination:
            source.seek(offset); data = source.read(chunk); destination.write(data)
        value = {'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                 'offset':offset,'archive_sha256':proof['archive_sha256']}
        with receipt.open('x') as f: json.dump(value, f, indent=2)
        print(json.dumps({**value,'path':str(path)}),flush=True)
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]{10,100}', file_id):
        raise ValueError('Require grounded uploaded Drive ID')
    value = json.loads(receipt.read_text())
    if (value['name'] != name or value['offset'] != offset or value['archive_sha256'] != proof['archive_sha256']
            or path.is_symlink() or digest(path) != value['sha256'] or path.stat().st_size != value['bytes']):
        raise ValueError('Scratch part changed')
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read('/Users/stevenyang/.config/rclone/rclone.conf')
    access = json.loads(cfg['gdrive']['token'])['access_token']
    request = urllib.request.Request('https://www.googleapis.com/drive/v3/files/' + file_id
        + '?fields=id,name,size,sha256Checksum,parents', headers={'Authorization':'Bearer '+access})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request,timeout=60) as response: metadata=json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code == 403:
                detail=json.loads(error.read())
                transient=any(x.get('reason') in ('rateLimitExceeded','userRateLimitExceeded')
                              for x in detail.get('error',{}).get('errors',[]))
            else:
                transient=error.code in (429,503)
            if not transient or attempt == 4:
                raise RuntimeError('Part metadata unavailable; retain scratch and retry only this read') from None
            time.sleep(2 ** (attempt + 1))
    if (metadata['id'] != file_id or metadata['name'] != name or metadata['parents'] != [FOLDER]
            or int(metadata['size']) != value['bytes'] or metadata['sha256Checksum'] != value['sha256']):
        raise ValueError('Uploaded part metadata/hash mismatch; scratch retained')
    with (output / f'part-{index:04d}-uploaded.json').open('x') as f:
        json.dump({**value,'drive_file_id':file_id,'provider_metadata':metadata},f,indent=2)
    # Only this invocation's derived scratch part; the verified source archive,
    # source worker volume, and GCS copy are untouched and reconstruct it exactly.
    path.unlink()
    print(json.dumps({'part':index,'file_id':file_id,'provider_hash_verified':True,
                     'removed_scratch_bytes':value['bytes'],'original_archive_retained':True}),flush=True)


def verify_stream(source, expected_hash, expected_size, manifest):
    reader, seen = HashReader(source), set()
    with tarfile.open(fileobj=reader, mode='r|gz') as tar:
        for member in tar:
            name = member.name.removeprefix('./')
            if not member.isfile() or name not in manifest or name in seen:
                raise ValueError('Unexpected archive member')
            hashed = hashlib.sha256()
            with tar.extractfile(member) as content:
                for block in iter(lambda: content.read(4 << 20), b''): hashed.update(block)
            if member.size != manifest[name]['bytes'] or hashed.hexdigest() != manifest[name]['sha256']:
                raise ValueError('Archive member mismatch')
            seen.add(name)
    while reader.read(4 << 20): pass
    if seen != set(manifest) or reader.size != expected_size or reader.hash.hexdigest() != expected_hash:
        raise ValueError('Full compressed/member readback mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('tx', 'nj'), required=True)
    parser.add_argument('--attempt', default='direct-drive')
    parser.add_argument('--prepare-part', type=int)
    parser.add_argument('--record-part-id')
    parser.add_argument('--finalize-parts', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'direct-drive(?:-v[2-9][0-9]*)?', args.attempt):
        raise ValueError('Require a new bounded attempt directory, not a path')
    root = BASE / args.worker
    archive = root / (args.worker + '-paused-evidence.tar.gz')
    proof = json.loads((root / 'LOCAL_VERIFIED.json').read_text())
    expected, size = digest(archive), archive.stat().st_size
    if proof['archive_sha256'] != expected or proof['archive_bytes'] != size:
        raise ValueError('Local verified archive changed')
    if args.finalize_parts:
        if args.prepare_part is not None or args.record_part_id is not None:
            raise ValueError('Finalize is a separate operation')
        output=root/'connector-parts-v1'; chunk=96 << 20
        parts=[json.loads((output/f'part-{i:04d}-uploaded.json').read_text())
               for i in range((size+chunk-1)//chunk)]
        for i,part in enumerate(parts):
            if (part['offset'] != i*chunk or part['bytes'] != min(chunk,size-i*chunk)
                    or part['archive_sha256'] != expected or part['name'] != archive.name+f'.part-{i:04d}'):
                raise ValueError('Incomplete or reordered part registry')
        spec={'archive_name':archive.name,'archive_bytes':size,'archive_sha256':expected,
              'member_manifest_sha256':digest(root/'FILES.json'),'parts_in_join_order':parts}
        with (output/'PARTS.json').open('x') as f: json.dump(spec,f,indent=2)
        print(json.dumps({'parts':len(parts),'spec':str(output/'PARTS.json'),'full_readback_pending':True}),flush=True)
        return
    if args.prepare_part is not None:
        connector_part(root, archive, proof, args.prepare_part, args.record_part_id)
        return
    if args.record_part_id is not None:
        raise ValueError('Part index required')
    output = root / args.attempt
    output.mkdir(exist_ok=False)
    def write(name, value):
        with (output / name).open('x') as f: json.dump(value, f, indent=2, sort_keys=True)
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read('/Users/stevenyang/.config/rclone/rclone.conf')
    access = json.loads(cfg['gdrive']['token'])['access_token']
    def request(url, *, data=None, method=None, headers=None):
        for attempt in range(5):
            try:
                return urllib.request.urlopen(urllib.request.Request(url, data=data, method=method,
                    headers={'Authorization': 'Bearer ' + access, **(headers or {})}), timeout=90)
            except urllib.error.HTTPError as error:
                # Chunk PUTs address the same byte range. Retry transient quota
                # failures only; never silently create a second upload session.
                if error.code not in (403, 429, 503) or attempt == 4 or method == 'POST':
                    raise
                if error.code == 403:
                    detail = json.loads(error.read())
                    reasons = {x.get('reason') for x in detail.get('error', {}).get('errors', [])}
                    if 'rateLimitExceeded' not in reasons and 'userRateLimitExceeded' not in reasons:
                        raise
                print(json.dumps({'worker': args.worker, 'transient_http_status': error.code,
                                  'retry_same_request': attempt + 1}), flush=True)
                time.sleep(2 ** (attempt + 1))
    write('INTENT.json', {'folder_id': FOLDER, 'archive_sha256': expected, 'archive_bytes': size,
                         'direct_resumable_upload_no_path_queries': True})
    try:
        with request('https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,sha256Checksum,parents',
                     data=json.dumps({'name': archive.name, 'parents': [FOLDER]}).encode(), method='POST',
                     headers={'Content-Type': 'application/json; charset=UTF-8',
                              'X-Upload-Content-Type': 'application/gzip', 'X-Upload-Content-Length': str(size)}) as response:
            location = response.headers['Location']
        offset, result = 0, None
        with archive.open('rb') as f:
            while offset < size:
                block = f.read(16 << 20)
                try:
                    with request(location, data=block, method='PUT', headers={
                        'Content-Type': 'application/gzip', 'Content-Range': f'bytes {offset}-{offset+len(block)-1}/{size}'}) as response:
                        result = json.load(response)
                except urllib.error.HTTPError as error:
                    if error.code != 308: raise
                    if error.headers.get('Range') != f'bytes=0-{offset+len(block)-1}':
                        raise ValueError('Unexpected upload acknowledgement') from None
                offset += len(block)
                if offset == size or offset % (128 << 20) == 0:
                    print(json.dumps({'worker': args.worker, 'uploaded_bytes': offset, 'total': size}), flush=True)
        if result is None: raise ValueError('No completed upload ID')
        write('UPLOADED.json', result)
        file_id = result['id']
        endpoint = 'https://www.googleapis.com/drive/v3/files/' + file_id
        with request(endpoint + '?fields=id,name,size,sha256Checksum,parents') as response:
            metadata = json.load(response)
        if (metadata['name'] != archive.name or metadata['parents'] != [FOLDER] or
                int(metadata['size']) != size or metadata['sha256Checksum'] != expected):
            raise ValueError('Provider metadata mismatch')
        write('PROVIDER_METADATA.json', metadata)
        with request(endpoint + '?alt=media') as source:
            verify_stream(source, expected, size, json.loads((root / 'FILES.json').read_text()))
        write('VERIFIED.json', {'status': 'provider_sha256_and_full_member_stream_readback_verified',
             'file_id': file_id, 'parent_folder_id': FOLDER, 'archive_sha256': expected,
             'archive_bytes': size, 'files': proof['files'], 'source_disks_retained': True})
        print(json.dumps({'worker': args.worker, 'status': 'drive_full_readback_verified', 'file_id': file_id}), flush=True)
    except Exception as error:
        write('FAILED.json', {'exception_type': type(error).__name__, 'http_status': getattr(error, 'code', None),
                             'local_and_worker_copies_retained': True})
        raise RuntimeError('Archive upload/readback incomplete; original copies preserved') from None


if __name__ == '__main__':
    main()
