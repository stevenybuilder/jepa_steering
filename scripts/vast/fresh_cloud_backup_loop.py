"""Bounded cloud-first backup of one owned fresh receiver; no efficacy analysis."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def run(command):
    return subprocess.check_output(command, text=True, timeout=900)


def last_json(output):
    return json.loads(next(line for line in reversed(output.splitlines()) if line.startswith('{')))


def download_hash(uri):
    digest = hashlib.sha256()
    with subprocess.Popen(['gcloud', 'storage', 'cat', uri], stdout=subprocess.PIPE) as process:
        for block in iter(lambda: process.stdout.read(1048576), b''):
            digest.update(block)
        if process.wait(timeout=900) != 0:
            raise RuntimeError('GCS verification download failed')
    return digest.hexdigest()


def relay_upload(ssh, archive, uri):
    """Relay through a pipe, never a laptop payload file or remote account key."""
    with subprocess.Popen(ssh + [shlex.join(['cat', archive])], stdout=subprocess.PIPE) as source:
        with subprocess.Popen(['gcloud', 'storage', 'cp', '--if-generation-match=0', '-', uri],
                              stdin=source.stdout, stdout=subprocess.PIPE, text=True) as target:
            source.stdout.close()
            target.communicate(timeout=900)
            if target.returncode != 0 or source.wait(timeout=900) != 0:
                raise RuntimeError('SSH-to-GCS streaming relay failed')
    return json.loads(run(['gcloud', 'storage', 'objects', 'describe', uri, '--format=json']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    parser.add_argument('--host-key-alias', help='Previously verified identity for an alternate provider proxy route')
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--transport', choices=('direct', 'ssh-relay'), default='ssh-relay')
    args = parser.parse_args()
    if args.rounds < 1 or args.interval < 60:
        raise ValueError('Use a positive bounded round count and interval >=60 seconds')
    args.receipts.mkdir(parents=True, exist_ok=True)
    root = '/workspace/fresh-four-20260912-v1'
    ssh = ['ssh', '-i', '/Users/stevenyang/.ssh/id_ed25519', '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20',
           '-o', 'UserKnownHostsFile=' + str(args.known_hosts.resolve()),
           '-p', args.port, 'root@' + args.host]
    if args.host_key_alias:
        ssh[1:1] = ['-o', 'HostKeyAlias=' + args.host_key_alias]
    uploader = Path(__file__).with_name('fresh_gcs_upload.py')
    for iteration in range(args.rounds):
        started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        hardware = 'hardware-' + stamp + '.txt'
        run(ssh + [shlex.join(['nvidia-smi', '-q', '-f', root + '/' + hardware])])
        verified = []
        for kind in ('published', 'diagnostics'):
            name = kind + '-' + stamp + '.tgz'
            archive = '/workspace/fresh-backups-' + str(args.instance) + '/' + name
            command = ['python', '/workspace/fresh_stream_backup.py', '--root', root, '--archive', archive]
            if kind == 'diagnostics':
                command += ['--diagnostic']
                for item in ('worker-logs-v2', 'fresh-freeze-v2', 'receiving-restore.log',
                             'receiving-os-dependencies.log', 'receiving-launcher.py', hardware):
                    command += ['--include', item]
            else:
                command += ['--allow-empty']
            snapshot = last_json(run(ssh + [shlex.join(command)]))
            obj = 'fresh-campaign-20260912-v2/' + str(args.instance) + '/' + name
            uri = 'gs://rgt-jepa-archive-2026/' + obj
            if args.transport == 'ssh-relay':
                upload = relay_upload(ssh, archive, uri)
            else:
                upload = last_json(run([sys.executable, str(uploader), '--host', args.host,
                    '--port', args.port, '--known-hosts', str(args.known_hosts),
                    '--remote-file', archive, '--sha256', snapshot['sha256'], '--object', obj]))
            if download_hash(uri) != snapshot['sha256']:
                raise ValueError('GCS downloaded archive SHA256 differs from verified receiver archive')
            verified.append({'kind': kind, 'object': uri, 'generation': upload['generation'],
                             **snapshot, 'gcs_download_sha256_verified': True})
        receipt = {'verified_utc': datetime.now(timezone.utc).isoformat(),
                   'instance_id': args.instance, 'archives': verified,
                   'transport': args.transport, 'local_payload_files_created': False,
                   'requires_local_controller_network_uptime': True,
                   'efficacy_values_not_reported': True}
        path = args.receipts / ('BACKUP_' + stamp + '_VERIFIED.json')
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        run(['gcloud', 'storage', 'cp', '--if-generation-match=0', str(path),
             'gs://rgt-jepa-archive-2026/fresh-campaign-20260912-v2/' + str(args.instance) + '/' + path.name])
        print(json.dumps(receipt), flush=True)
        if iteration + 1 < args.rounds:
            time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
