"""Recover existing navigation fitting banks, streaming a verified Drive archive.

No evaluation, refit, overwrites, protected exposure or GPU work. Only the
hash-bound archived files required by the registered successor fit are retained.
"""
import hashlib
import json
from pathlib import Path
import tarfile

from component_extension_fleet import PROJECT, sha, write
from final_preservation_drive import request
from restore_table_evidence import HashedReader, safe_path

FILE = '1-0hJolJwqryD6YiNQlTxVyGuC-bc36vq'
EXPECTED = '8fac5678983ec69883ecafdd29ac7ae4df507c58c259e94259f3e2f7952ce44e'
SIZE = 928552960
ROOT = PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/refined-source-recovery-v1'


def main():
    ROOT.mkdir(exist_ok=False)
    with request(FILE, '?fields=id,name,size,sha256Checksum,parents,trashed,mimeType') as stream:
        metadata = json.load(stream)
    assert metadata['name'] == 'primary-durable-results.tar.gz'
    assert metadata['parents'] == ['1D325lCMFph9es8IvCs-7lVR-dz4n_DdK']
    assert metadata['sha256Checksum'] == EXPECTED and int(metadata['size']) == SIZE
    assert not metadata.get('trashed') and not metadata['mimeType'].startswith('application/vnd.google-apps.')
    write(ROOT / 'SOURCE.json', metadata)
    kept, relevant_names = {}, []
    with request(FILE, '?alt=media') as response:
        reader = HashedReader(response, SIZE)
        with tarfile.open(fileobj=reader, mode='r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if 'navigation' not in member.name:
                    continue
                relevant_names.append({'name': member.name, 'bytes': member.size})
                selected = ('navigation-fits-20260907-v1/' in member.name and
                    (member.name.endswith('.json') or
                     '/operator_rank/operator_bank.pt' in member.name))
                selected |= ('navigation-offline-cohorts-20260907-v1/' in member.name and member.name.endswith('.json'))
                if not selected:
                    continue
                relative = safe_path(member.name)
                target = ROOT / 'restored' / relative
                if member.size > 64 << 20:
                    raise ValueError('Unexpected large fitting member; preserve partial recovery')
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.extractfile(member) as source, target.open('xb') as out:
                    while block := source.read(4 << 20):
                        digest.update(block); out.write(block)
                assert target.stat().st_size == member.size
                kept[member.name] = {'sha256': digest.hexdigest(), 'bytes': member.size}
                print(json.dumps({'restored': member.name, 'bytes': member.size}), flush=True)
        while reader.read(4 << 20):
            pass
        assert reader.count == SIZE and reader.hashed.hexdigest() == EXPECTED
    write(ROOT / 'NAVIGATION_MEMBERS.json', relevant_names)
    write(ROOT / 'RESTORED_VERIFIED.json', {'files': kept, 'full_archive_sha256': EXPECTED,
        'full_archive_readback': True, 'source_drive_id': FILE, 'source_unchanged': True})
    print(json.dumps({'verified_members': len(kept), 'source_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
