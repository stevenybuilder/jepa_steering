"""Object-scoped direct upload; keep account credentials off rented workers.

The upload session is a secret capability. Send it over SSH stdin only, never
argv/logs. No overwrite, IAM change, public sharing, or automatic upload retry.
Cloud readback and source revalidation are the caller's required next steps.
"""
import json
import subprocess
import urllib.parse
import urllib.request


def validate_session(session, bucket):
    from urllib.parse import urlsplit, parse_qs
    url = urlsplit(session)
    query = parse_qs(url.query)
    if (url.scheme != 'https' or url.netloc != 'storage.googleapis.com'
            or url.path != '/upload/storage/v1/b/' + bucket + '/o'
            or url.fragment or query.get('uploadType') != ['resumable']
            or len(query.get('upload_id', [])) != 1):
        raise ValueError('Unexpected upload-session destination')


def create_session(cloud_uri, size):
    url = urllib.parse.urlsplit(cloud_uri)
    if (url.scheme != 'gs' or url.netloc != 'rgt-jepa-archive-2026'
            or not url.path.startswith('/rep_geometry_transcoder/')
            or url.query or url.fragment or size <= 0):
        raise ValueError('Unexpected archive destination or size')
    bucket, name = url.netloc, url.path.lstrip('/')
    token = subprocess.check_output(['gcloud', 'auth', 'print-access-token'], text=True).strip()
    endpoint = 'https://storage.googleapis.com/upload/storage/v1/b/' + bucket + '/o?'
    endpoint += urllib.parse.urlencode({'uploadType': 'resumable', 'name': name, 'ifGenerationMatch': 0})
    request = urllib.request.Request(endpoint, data=b'{}', method='POST', headers={
        'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
        'X-Upload-Content-Type': 'application/gzip', 'X-Upload-Content-Length': str(size)})
    with urllib.request.urlopen(request, timeout=60) as response:
        session = response.headers['Location']
    validate_session(session, bucket)
    return {'session': session, 'bucket': bucket, 'name': name, 'size': size}


def build_archive(root, output, names, source_bytes):
    import hashlib
    from pathlib import Path
    import shutil
    import tarfile
    root, output = Path(root), Path(output)
    if (output.parent != root or not output.name.startswith('core-completion-preservation-')
            or not names or len(names) != len(set(names))):
        raise ValueError('Unexpected archive scope')
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\0' in name:
            raise ValueError('Unsafe archive member')
        if (root / path).is_symlink() or not (root / path).is_file():
            raise ValueError('Archive source is not a regular file')
    if shutil.disk_usage(root).free < source_bytes * 1.05 + 5 * 1024**3:
        raise ValueError('Insufficient worker disk reserve')
    output.mkdir(exist_ok=False)
    archive = output / 'core-and-paused-hmm.tar.gz'
    with archive.open('xb') as target:
        with tarfile.open(fileobj=target, mode='w:gz', dereference=True, compresslevel=6) as stream:
            for name in names:
                stream.add(root / name, arcname=name, recursive=False)
    hashed = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): hashed.update(block)
    return {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': hashed.hexdigest()}


def upload_file(payload):
    import hashlib
    import json
    from pathlib import Path
    import urllib.request
    # validate_session is included in the receiving source alongside this function.
    validate_session(payload['session'], payload['bucket'])
    path = Path(payload['path'])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != payload['size']:
        raise ValueError('Upload source size/type differs')
    hashed = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): hashed.update(block)
    if hashed.hexdigest() != payload['sha256']:
        raise ValueError('Upload source hash differs')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError('Upload redirect rejected')
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with path.open('rb') as stream:
            request = urllib.request.Request(payload['session'], data=stream, method='PUT',
                headers={'Content-Length': str(payload['size']), 'Content-Type': 'application/gzip'})
            with opener.open(request, timeout=180) as response:
                if response.status not in (200, 201):
                    raise ValueError('Upload not finalized')
                metadata = json.load(response)
    except Exception as error:
        # HTTP exceptions may include the capability URI. Do not print them or
        # chain their traceback into the SSH output.
        raise RuntimeError('Direct upload failed; retained source requires review ('
                           + type(error).__name__ + ')') from None
    if (metadata.get('bucket') != payload['bucket'] or metadata.get('name') != payload['name']
            or int(metadata.get('size', -1)) != payload['size']):
        raise ValueError('Uploaded object binding differs')
    return {key: metadata[key] for key in ('bucket', 'name', 'size', 'generation', 'md5Hash') if key in metadata}
