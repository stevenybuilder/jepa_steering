"""Bounded Drive connector staging for the already verified completed core archive.

No scientific execution or rental lifecycle. Reuse the tested part verifier;
the full source archive and worker/GCS originals are never deleted here.
"""
import argparse
import json
from pathlib import Path

import upload_pause_archive as parts_helper
from backup_results_to_google import verify_archive
from navigation_redistribution_common import digest, write

ROOT = Path(__file__).resolve().parents[2] / 'artifacts/offline_study/core-completion-preservation-20260908-v1'
FOLDER = '1SoXkmEJmrCZXBnKk3re6mX8WwJiWBokY'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-local', action='store_true')
    group.add_argument('--part', type=int)
    group.add_argument('--finalize', action='store_true')
    parser.add_argument('--record-id')
    args = parser.parse_args()
    if args.record_id is not None and args.part is None:
        raise ValueError('Recording requires an exact part index')
    proof = json.loads((ROOT / 'VERIFIED.json').read_text())
    if proof['status'] != 'completed_core_and_paused_hmm_full_cloud_member_readback_verified':
        raise ValueError('Verified cloud source required')
    archive = ROOT / 'core-and-paused-hmm.tar.gz'
    if archive.is_symlink() or archive.stat().st_size != proof['archive_bytes'] or digest(archive) != proof['archive_sha256']:
        raise ValueError('Downloaded source archive differs')
    if args.verify_local:
        verify_archive(['cat', str(archive)], proof['archive_sha256'], proof['archive_bytes'],
                       json.loads((ROOT / 'FILES.json').read_text()))
        write(ROOT / 'LOCAL_VERIFIED.json', {**proof, 'local_source_and_all_members_verified': True})
        print(json.dumps({'local_verified': True, 'bytes': proof['archive_bytes']}), flush=True)
        return
    local = json.loads((ROOT / 'LOCAL_VERIFIED.json').read_text())
    if local['archive_sha256'] != proof['archive_sha256'] or not local['local_source_and_all_members_verified']:
        raise ValueError('Require complete local readback before staging')
    if args.part is not None:
        parts_helper.FOLDER = FOLDER
        parts_helper.connector_part(ROOT, archive, proof, args.part, args.record_id)
        return
    output = ROOT / 'connector-parts-v1'; size = proof['archive_bytes']; chunk = 96 << 20
    parts = [json.loads((output / f'part-{i:04d}-uploaded.json').read_text())
             for i in range((size + chunk - 1) // chunk)]
    if len({p['drive_file_id'] for p in parts}) != len(parts):
        raise ValueError('Duplicate Drive parts')
    for i, part in enumerate(parts):
        if (part['offset'] != i * chunk or part['bytes'] != min(chunk, size - i * chunk)
                or part['archive_sha256'] != proof['archive_sha256']
                or part['name'] != archive.name + f'.part-{i:04d}'
                or part['provider_metadata']['parents'] != [FOLDER]):
            raise ValueError('Incomplete or reordered part registry')
    write(output / 'PARTS.json', {'archive_name': archive.name, 'archive_bytes': size,
        'archive_sha256': proof['archive_sha256'], 'member_manifest_sha256': digest(ROOT / 'FILES.json'),
        'parts_in_join_order': parts, 'folder_id': FOLDER})
    print(json.dumps({'parts': len(parts), 'full_readback_pending': True}), flush=True)


if __name__ == '__main__':
    main()
