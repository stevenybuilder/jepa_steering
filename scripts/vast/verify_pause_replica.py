"""Verify byte-identical pause archives on the already retained LA volume."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from backup_results_to_google import digest, verify_archive
from preserve_core_pause import PROJECT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('tx', 'nj'), required=True)
    parser.add_argument('--cloud', action='store_true')
    args = parser.parse_args()
    root = PROJECT / 'artifacts/offline_study/core-priority-pause-20260908-v1' / args.worker
    proof = json.loads((root / 'LOCAL_VERIFIED.json').read_text())
    remote = '/workspace/jepa-runtime/core-priority-archive-20260908-v1/' + args.worker
    names = [args.worker + '-paused-evidence.tar.gz', 'FILES.json', 'SCOPE.json', 'LOCAL_VERIFIED.json']
    expected = {name: digest(root / name) for name in names}
    if expected[names[0]] != proof['archive_sha256']:
        raise ValueError('Local archive differs from member-verified proof')
    if args.cloud:
        uri = 'gs://rgt-jepa-archive-2026/rep_geometry_transcoder/core-priority-pause-20260908-v1/' + names[0]
        verify_archive(['gcloud', 'storage', 'cat', uri], proof['archive_sha256'], proof['archive_bytes'],
                       json.loads((root / 'FILES.json').read_text()))
        value = {'status': 'google_cloud_compressed_and_member_readback_verified',
                 'source_instance': proof['instance'], 'cloud_uri': uri,
                 'archive_sha256': proof['archive_sha256'], 'archive_bytes': proof['archive_bytes'],
                 'files': proof['files'], 'source_disks_retained': True, 'drive_verified': False, 'time': time.time()}
        with (root / 'CLOUD_VERIFIED.json').open('x') as f: json.dump(value, f, indent=2)
        print(json.dumps({'worker': args.worker, 'status': value['status']})); return
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes',
           '-p', '21938', 'root@98.142.241.142']
    result = subprocess.check_output(ssh + [shlex.join(['sha256sum', *[remote + '/' + n for n in names]])], text=True)
    observed = {Path(line.split(maxsplit=1)[1]).name: line.split(maxsplit=1)[0] for line in result.splitlines()}
    if observed != expected:
        raise ValueError('Retained LA copy differs')
    value = {'status': 'retained_la_replica_full_byte_hash_verified', 'source_instance': proof['instance'],
             'replica_instance': 50233992, 'replica_root': remote, 'archive_sha256': proof['archive_sha256'],
             'archive_bytes': proof['archive_bytes'], 'files': proof['files'], 'members_sha256': expected,
             'all_source_disks_must_be_retained': True, 'drive_full_readback_complete': False, 'time': time.time()}
    with (root / 'REPLICA_VERIFIED.json').open('x') as f: json.dump(value, f, indent=2)
    print(json.dumps({'worker': args.worker, 'status': value['status']}))


if __name__ == '__main__':
    main()
