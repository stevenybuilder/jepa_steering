"""Restore the existing byte-identical native encoder, not a new conversion."""
import hashlib
import json
from pathlib import Path
import shutil

from component_extension_fleet import PROJECT, sha, write
from final_preservation_drive import FOLDER, request

FILE = '1PEj3wA0J2Z2SheKa9VEqoszqaUJ9TAgU'
SIZE = 1213030147
SHA = 'fc933429ede48304e832fa11bbcc5d838eb539ca83b1746b93c8625526407b9a'
ROOT = PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/refined-droid-encoder-v1'


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    with request(FILE, '?fields=id,name,size,sha256Checksum,mimeType,parents,trashed') as response:
        metadata = json.load(response)
    if (metadata['id'] != FILE or metadata['parents'] != [FOLDER]
            or metadata['sha256Checksum'] != SHA or int(metadata['size']) != SIZE
            or metadata.get('trashed') or metadata['mimeType'].startswith('application/vnd.google-apps.')):
        raise ValueError('Archived encoder metadata differs')
    if shutil.disk_usage(ROOT).free < SIZE + 8 * 1024**3:
        raise ValueError('Keep eight GiB local reserve')
    write(ROOT / 'SOURCE.json', metadata)
    target = ROOT / 'native_state_from_official_hf.pth'
    partial = target.with_suffix('.partial')
    h, size = hashlib.sha256(), 0
    with request(FILE, '?alt=media') as source, partial.open('xb') as output:
        while chunk := source.read(4 << 20):
            h.update(chunk); size += len(chunk); output.write(chunk)
    if size != SIZE or h.hexdigest() != SHA:
        raise ValueError('Full encoder bytes differ; retain partial')
    partial.rename(target)
    prior = PROJECT / 'artifacts/offline_study/primary-durable-20260907/dinov3-native-20260907-v2'
    for name in ('protocol.json', 'report.json', 'DONE.json'):
        with (prior / name).open('rb') as source, (ROOT / name).open('xb') as output:
            shutil.copyfileobj(source, output)
    write(ROOT / 'RESTORED_VERIFIED.json', {'source_drive_id': FILE, 'bytes': size,
        'sha256': SHA, 'full_byte_readback': True, 'new_conversion': False,
        'files': {p.name: sha(p) for p in ROOT.iterdir() if p.is_file()}})
    print(json.dumps({'encoder_restored_exactly': True, 'bytes': size}), flush=True)


if __name__ == '__main__':
    main()
