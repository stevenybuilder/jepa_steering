"""Finish the explicitly bound completed-DROID snapshot after Drive lookup quota.

No GPU work or source mutation. The original failed attempt and archive chain
remain intact; a known private folder ID avoids repeated path discovery calls.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from backup_results_to_google import verify_archive
from navigation_redistribution_common import digest, write
from navigation_redistribution_stage import connections
from snapshot_live_results import HASH_FILES


PROJECT = Path(__file__).resolve().parents[2]
RECEIPTS = PROJECT / 'artifacts/offline_study/live-20260908T151000Z/instance-50259194-droid-parallel'
FOLDER = '1heyTSOeTPCXD78LwqrEeK3hnBdpGSwM-'
EXPECTED_REPORT = 'b998218db8f1b87d18150bed04b9eec29731ef50af5e8c203f7ba028b9f83de8'


def main():
    manifest = json.loads((RECEIPTS / 'FILES.json').read_text())
    key = 'droid-coupling-behavior-20260908-v3/analysis/report.json'
    if (len(manifest) != 714 or manifest[key]['sha256'] != EXPECTED_REPORT or
            not (RECEIPTS / 'INHERITED.json').is_file() or
            (RECEIPTS / 'RECOVERY_VERIFIED.json').exists()):
        raise ValueError('Exact completed-panel snapshot required; never replace verified evidence')
    plan = json.loads((RECEIPTS / 'PLAN.json').read_text())
    if plan['instance'] != 50259194 or plan['bytes'] != 4848755:
        raise ValueError('Original snapshot scope changed')
    names = sorted(manifest)
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
        raise ValueError('Unsafe snapshot path')
    if shutil.disk_usage(RECEIPTS).free < plan['bytes'] + 4 * 1024**3:
        raise ValueError('Keep four GiB local reserve')
    ssh = connections()[0][50259194]
    def inventory():
        return json.loads(subprocess.check_output(ssh + [shlex.join([
            'python', '-c', HASH_FILES])], input=json.dumps(names), text=True, timeout=90))
    if inventory() != manifest:
        raise ValueError('Selected source records changed')
    write(RECEIPTS / 'RECOVERY_INTENT.json', {'private_folder_id': FOLDER,
        'prior_path_lookup_failed': True, 'source_manifest_sha256': digest(RECEIPTS / 'FILES.json'),
        'inherited_manifest_sha256': digest(RECEIPTS / 'INHERITED.json'),
        'script_sha256': digest(Path(__file__)), 'gpu_calls': 0, 'source_jobs_untouched': True})
    archive = RECEIPTS / 'instance-50259194-results.tar.gz'
    with tempfile.TemporaryFile() as listing, archive.open('xb') as output:
        listing.write(b'\0'.join(n.encode() for n in names) + b'\0'); listing.seek(0)
        subprocess.run(ssh + ['tar -C /workspace/jepa-runtime --hard-dereference --no-recursion --null -czf - -T -'],
            stdin=listing, stdout=output, check=True, timeout=120)
    archive_hash, size = digest(archive), archive.stat().st_size
    verify_archive(['cat', str(archive)], archive_hash, size, manifest)
    if inventory() != manifest:
        raise ValueError('Source changed during preservation')
    write(RECEIPTS / 'LOCAL_ARCHIVE_VERIFIED.json', {'archive_sha256': archive_hash,
        'archive_bytes': size, 'files': len(manifest), 'source_rechecked': True})
    env = dict(os.environ, RCLONE_BIND='0.0.0.0', RCLONE_TPSLIMIT='1',
        RCLONE_RETRIES='1', RCLONE_LOW_LEVEL_RETRIES='1', RCLONE_TIMEOUT='60s',
        RCLONE_CONTIMEOUT='15s', RCLONE_DRIVE_ROOT_FOLDER_ID=FOLDER)
    destination = 'gdrive:instance-50259194-results.tar.gz'
    try:
        subprocess.run(['rclone', 'copyto', str(archive), destination,
            '--immutable', '--drive-chunk-size', '8M'], env=env, check=True, timeout=180)
        # Explicit per-command root ID, including the verification subprocess.
        verify_archive(['rclone', 'cat', destination, '--drive-root-folder-id', FOLDER,
            '--bind', '0.0.0.0', '--retries', '1', '--low-level-retries', '1',
            '--timeout', '60s', '--contimeout', '15s', '--tpslimit', '1'],
            archive_hash, size, manifest)
        write(RECEIPTS / 'RECOVERY_VERIFIED.json', {'private_folder_id': FOLDER,
            'archive_name': archive.name, 'archive_sha256': archive_hash,
            'archive_bytes': size, 'files': len(manifest),
            'status': 'completed_droid_increment_full_readback_verified',
            'inherited_manifest_sha256': digest(RECEIPTS / 'INHERITED.json'),
            'source_manifest_sha256': digest(RECEIPTS / 'FILES.json'),
            'source_jobs_untouched': True, 'all_study_complete': False})
    except Exception as error:
        write(RECEIPTS / 'RECOVERY_FAILED.json', {'error': str(error),
            'local_archive_sha256': archive_hash, 'all_sources_preserved': True,
            'gpu_calls': 0})
        raise
    print(json.dumps({'status': 'completed_droid_archive_verified',
        'files': len(manifest), 'archive_bytes': size, 'archive_sha256': archive_hash}), flush=True)


if __name__ == '__main__':
    main()
