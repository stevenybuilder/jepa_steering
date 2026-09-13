"""Bounded four-range GCS staging; tokens travel on stdin, never in argv/logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

SIZE = 1471380149


def segments(size, count=4):
    if size < count or count not in (4, 16, 32):
        raise ValueError('Bounded nonempty ranges required')
    return [(size * i // count, size * (i + 1) // count - 1) for i in range(count)]


def fetch_range(payload, start, end, deadline, consume, opener=urllib.request.urlopen):
    req = urllib.request.Request(payload['url'], headers={
        'Authorization': 'Bearer ' + payload['token'], 'Range': f'bytes={start}-{end}',
        'Accept-Encoding': 'identity'})
    expected = f'bytes {start}-{end}/{payload["size"]}'
    count = 0
    with opener(req, timeout=15) as response:
        if response.status != 206 or response.headers.get('Content-Range') != expected:
            raise ValueError('Pinned object byte range was not honored')
        while count < end - start + 1:
            if time.monotonic() >= deadline:
                raise TimeoutError('Bounded range download expired')
            block = response.read(min(262144, end - start + 1 - count))
            if not block:
                raise ValueError('Truncated object range')
            consume(start + count, block)
            count += len(block)
    return count


def write_at(fd, offset, block):
    while block:
        written = os.pwrite(fd, block, offset)
        if written <= 0:
            raise OSError('Incomplete positional write')
        offset += written
        block = block[written:]


def remote(payload, probe):
    if payload['size'] != SIZE:
        raise ValueError('Unexpected runtime size')
    started = time.monotonic()
    deadline = started + (30 if probe else 540)
    connections = payload.get('connections', 4)
    ranges = segments(SIZE, connections)
    if probe:
        ranges = [(a, min(a + 1048576 - 1, b)) for a, b in ranges]
        fd = None
        consume = lambda offset, block: None
    else:
        scratch = Path('/workspace/prepared-runtime-inputs-v2.parallel.partial')
        target = Path('/workspace/prepared-runtime-inputs-v2.parallel.tgz')
        if target.exists():
            raise ValueError('Verified-name target exists; reconcile instead of overwrite')
        fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(fd, SIZE)
        consume = lambda offset, block: write_at(fd, offset, block)
    try:
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [pool.submit(fetch_range, payload, a, b, deadline, consume) for a, b in ranges]
            counts = [future.result() for future in futures]
        if fd is not None:
            os.fsync(fd)
    finally:
        if fd is not None:
            os.close(fd)
    result = {'probe': probe, 'bytes': sum(counts), 'seconds': time.monotonic() - started,
              'ranges': ranges, 'local_laptop_payload_created': False}
    result['megabytes_per_second'] = result['bytes'] / result['seconds'] / 1000000
    if not probe:
        digest = hashlib.sha256()
        with scratch.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                digest.update(block)
        if scratch.stat().st_size != SIZE or digest.hexdigest() != payload['sha256']:
            raise ValueError('Assembled runtime SHA256/size differs; partial retained')
        # Never replace the old single-stream partial or an existing verified file.
        os.link(scratch, target)
        result.update(archive=str(target), sha256=digest.hexdigest(), download_verified=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--connections', type=int, choices=(4, 16, 32), default=4)
    parser.add_argument('--host')
    parser.add_argument('--port')
    parser.add_argument('--known-hosts', type=Path)
    args = parser.parse_args()
    if args.remote:
        try:
            print(json.dumps(remote(json.load(sys.stdin), args.probe)), flush=True)
        except Exception as error:
            # Exception text from HTTP libraries may expose a scoped URL; do not print it.
            print(json.dumps({'failed': True, 'error_type': type(error).__name__}), flush=True)
            raise SystemExit(1)
        return
    if not all((args.host, args.port, args.known_hosts)):
        parser.error('Exact receiver endpoint and known-hosts required')
    from fresh_gcs_download import token, URL, SHA
    scoped = token()
    payload = {'url': URL, 'sha256': SHA, 'size': SIZE, 'token': scoped,
               'connections': args.connections}
    remote_argv = ['python', '-c', Path(__file__).read_text(), '--remote']
    if args.probe:
        remote_argv.append('--probe')
    ssh = ['ssh', '-i', '/Users/stevenyang/.ssh/id_ed25519', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UserKnownHostsFile=' + str(args.known_hosts), '-p', args.port,
           'root@' + args.host, shlex.join(remote_argv)]
    try:
        subprocess.run(ssh, input=json.dumps(payload), text=True, check=True,
                       timeout=60 if args.probe else 600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'transport_failed': True, 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
