"""Retry an intact, verified archive over IPv4 without discarding failed receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from backup_results_to_google import verify_archive
from snapshot_live_results import HASH_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--drive', required=True)
    args = parser.parse_args()
    if not args.drive.startswith('gdrive:Research-Archives/JEPA-WM/live-'):
        raise ValueError('Require grounded private archive parent')
    manifest = json.loads((args.previous / 'FILES.json').read_text())
    plan = json.loads((args.previous / 'PLAN.json').read_text())
    spool = args.previous.parent / (args.previous.name + '.tar.gz')
    sha = hashlib.sha256()
    with spool.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            sha.update(block)
    digest, size = sha.hexdigest(), spool.stat().st_size
    verify_archive(['cat', str(spool)], digest, size, manifest)
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    row = next(x for x in rows if x['id'] == plan['instance'])
    if row['actual_status'] != 'running' or not row['geolocation'].endswith(', US'):
        raise ValueError('Original source must still be available for verification')
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=15', '-p', str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    def verify_source():
        result = json.loads(subprocess.check_output(ssh + [shlex.join(['python', '-c', HASH_FILES])],
            input=json.dumps(list(manifest)), text=True))
        if result != manifest:
            raise ValueError('Original immutable source changed')
    verify_source()
    args.output.mkdir(exist_ok=False)
    (args.output / 'FILES.json').write_bytes((args.previous / 'FILES.json').read_bytes())
    (args.output / 'PLAN.json').write_text(json.dumps({**plan, 'drive': args.drive,
        'retry_of': str(args.previous), 'source_spool': str(spool), 'ipv4': True}, indent=2) + '\n')
    subprocess.run(['rclone', 'mkdir', args.drive, '--bind', '0.0.0.0'], check=True, timeout=120)
    target = args.drive + f"/instance-{plan['instance']}-results.tar.gz"
    subprocess.run(['rclone', 'copyto', str(spool), target, '--immutable', '--bind', '0.0.0.0',
        '--drive-chunk-size', '8M', '--retries', '3', '--low-level-retries', '3', '--timeout', '90s',
        '--contimeout', '15s', '--stats', '15s', '--stats-one-line', '--stats-log-level', 'NOTICE'], check=True)
    verify_archive(['rclone', 'cat', target, '--bind', '0.0.0.0'], digest, size, manifest)
    verify_source()
    (args.output / 'VERIFIED.json').write_text(json.dumps({'status': 'immutable_live_snapshot_all_members_verified',
        'archive': target, 'archive_sha256': digest, 'archive_bytes': size, 'files': len(manifest),
        'instance': plan['instance'], 'jobs_untouched': True, 'all_study_complete': False,
        'source_disks_retained': True, 'original_failed_receipts_preserved': True}, indent=2) + '\n')
    subprocess.run(['rclone', 'copy', str(args.output), args.drive + '/receipts', '--immutable', '--bind', '0.0.0.0'], check=True)
    subprocess.run(['rclone', 'check', str(args.output), args.drive + '/receipts', '--download', '--one-way', '--bind', '0.0.0.0'], check=True)
    print(json.dumps({'status': 'spool_retry_verified', 'files': len(manifest), 'bytes': size}), flush=True)


if __name__ == '__main__':
    main()
