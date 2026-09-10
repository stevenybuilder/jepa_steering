"""Bind complete existing MetaWorld/Push-T raw inputs for the required histories.

CPU byte verification only: no deserialization, video decoding, model or loader
construction, training, validation, or access authorization. In particular, this
does not open the protected MetaWorld validation rows. Native reader/numerical
parity and an explicit permitted execution contract remain separate prerequisites.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import time
import zipfile

from .protocol import sha256, write_json


DATA_REVISION = '6116f042ae7ae4c8e3f1fd2f194f432615664182'
MANIFEST_HASHES = {
    'metaworld': 'b1f7ba9b0999ae88dcb8553592c393f3110a9522a7c9d23a6737a2b04518f89e',
    'pusht': '49dd903b4056d7ff65a3f61523a859038b277fdc639d2838868fdad29f796911',
}
PUSHT_ARCHIVE_HASH = '442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08'
COUNTS = {'metaworld': (11340, 1260), 'pusht': (18685, 21)}
SIDECARS = ('abs_actions.pth', 'rel_actions.pth', 'states.pth', 'velocities.pth',
            'seq_lengths.pkl', 'tokens.pth')


def required_files(task):
    if task == 'metaworld':
        return {f'train-{index:05d}-of-00126.parquet' for index in range(126)}
    if task == 'pusht':
        result = set()
        for pool, count in zip(('train', 'val'), COUNTS[task]):
            result.update(f'{pool}/{name}' for name in SIDECARS)
            result.update(f'{pool}/obses/episode_{index:03d}.mp4' for index in range(count))
        return result
    raise ValueError('Only the two native 32-rank tasks are in scope')


def safe_member(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or str(relative) != name:
        raise ValueError('Unsafe raw-input relative path')
    path = root / name
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Do not follow a raw-input member symlink: ' + name)
    if not path.is_file():
        raise ValueError('Missing raw input: ' + name)
    return path


def bound_manifest(task, provenance):
    """Use known source-download manifests, not a freshly self-hashed inventory."""
    if task not in MANIFEST_HASHES or sha256(provenance) != MANIFEST_HASHES[task]:
        raise ValueError('Original pinned download/extraction manifest changed')
    original = json.loads(provenance.read_text())
    if task == 'metaworld':
        if original['data_revision'] != DATA_REVISION:
            raise ValueError('Official data revision changed')
        prefix = 'metaworld/data/'
        selected = [row for row in original['files'] if row['path'].startswith(prefix)]
        names = [row['path'][len(prefix):] for row in selected]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate official parquet entry')
        manifest = {name: {'bytes': row['bytes'], 'sha256': row['sha256']}
                    for name, row in zip(names, selected)}
        for row in selected:
            expected_url = ('https://huggingface.co/datasets/facebook/jepa-wms/resolve/' +
                            DATA_REVISION + '/' + row['path'])
            if row['url'] != expected_url:
                raise ValueError('Official source URL/revision differs')
    else:
        prefix = 'pusht_noise/'
        if any(not name.startswith(prefix) for name in original):
            raise ValueError('Unexpected Push-T extraction population')
        manifest = {name[len(prefix):]: row for name, row in original.items()}
    if set(manifest) != required_files(task):
        expected = required_files(task)
        raise ValueError('Official raw-input population differs for ' + task +
            '; missing=' + repr(sorted(expected - set(manifest))) +
            '; extra=' + repr(sorted(set(manifest) - expected)))
    return manifest


def verify_archive_members(archive, manifest):
    """Recheck each decompressed member against the original extraction manifest."""
    if sha256(archive) != PUSHT_ARCHIVE_HASH:
        raise ValueError('Push-T ZIP differs from pinned official asset')
    seen = set()
    with zipfile.ZipFile(archive) as zipped:
        all_names = set()
        for member in zipped.infolist():
            name = member.filename
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name or
                    stat.S_ISLNK(member.external_attr >> 16) or name in all_names):
                raise ValueError('Unsafe or duplicate official ZIP member')
            all_names.add(name)
            if member.is_dir():
                continue
            if not name.startswith('pusht_noise/'):
                raise ValueError('Unexpected ZIP population')
            name = name[len('pusht_noise/'):]
            if name not in manifest:
                raise ValueError('Unregistered official ZIP member')
            digest = hashlib.sha256()
            with zipped.open(member) as source:
                for block in iter(lambda: source.read(4 << 20), b''):
                    digest.update(block)
            if member.file_size != manifest[name]['bytes'] or digest.hexdigest() != manifest[name]['sha256']:
                raise ValueError('Original extraction manifest differs from ZIP: ' + name)
            seen.add(name)
    if seen != set(manifest):
        raise ValueError('Official ZIP population incomplete')


def verify_inputs(task, root, *, provenance, output=None, archive=None):
    """Return a raw-byte binding only; caller must separately enforce access/parity."""
    root, provenance = Path(root), Path(provenance)
    if (not root.is_absolute() or root == Path(root.anchor) or '..' in root.parts or
            not root.is_dir() or root.is_symlink()):
        raise ValueError('Explicit existing nonsymlink data root required')
    manifest = bound_manifest(task, provenance)
    if task == 'pusht' and archive is None:
        raise ValueError('Full pinned ZIP is required, not just the planning subset')
    if task == 'metaworld' and {str(p.relative_to(root)) for p in root.rglob('*.parquet')} != set(manifest):
        raise ValueError('Native HF loader would see a different parquet population')
    if task == 'pusht':
        for pool in ('train', 'val'):
            actual_sidecars = {p.name for p in (root / pool).iterdir()
                               if p.suffix in ('.pth', '.pkl')}
            if actual_sidecars != set(SIDECARS):
                raise ValueError('Native Push-T sidecars differ; optional shapes must remain absent')
        actual_videos = {str(p.relative_to(root)) for pool in ('train', 'val')
                         for p in (root / pool / 'obses').glob('*.mp4')}
        if actual_videos != {n for n in manifest if n.endswith('.mp4')}:
            raise ValueError('Native Push-T video population differs')
    started = time.monotonic()
    protocol = {'role': 'raw_byte_input_verification_only', 'task': task,
        'data_revision': DATA_REVISION, 'data_root': str(root),
        'raw_manifest_sha256': MANIFEST_HASHES[task],
        'source_sha256': sha256(Path(__file__)), 'training_or_validation_authorized': False,
        'native_reader_parity_established': False, 'model_or_data_loader_initialized': False}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'protocol.json', protocol)
    count = 0
    try:
        if task == 'pusht':
            verify_archive_members(Path(archive), manifest)
        for name, expected in sorted(manifest.items()):
            path = safe_member(root, name)
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError('Existing raw input changed: ' + name)
            count += 1
        # Both supplied manifest and archive are immutable provenance, not outputs.
        if sha256(provenance) != MANIFEST_HASHES[task]:
            raise ValueError('Provenance changed during verification')
        report = {**protocol, 'status': 'complete_robotics_raw_inputs_verified',
            'files': count, 'bytes': sum(row['bytes'] for row in manifest.values()),
            'training_rows': COUNTS[task][0], 'validation_rows': COUNTS[task][1],
            'row_counts_basis': 'prior_verified_native_metadata_not_new_video_decoding',
            'archive_sha256': PUSHT_ARCHIVE_HASH if task == 'pusht' else None,
            'seconds': time.monotonic() - started, 'gpu_calls': 0}
        if output is not None:
            report['protocol_sha256'] = sha256(output / 'protocol.json')
            write_json(output / 'report.json', report)
            write_json(output / 'DONE.json', {'report_sha256': sha256(output / 'report.json')})
        return report
    except Exception as error:
        if output is not None:
            write_json(output / 'FAILED.json', {'error': str(error), 'verified_files': count,
                'training_or_validation_authorized': False, 'gpu_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=tuple(COUNTS))
    for name in ('data-root', 'provenance', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_inputs(args.task, args.data_root, provenance=args.provenance,
        output=args.output, archive=args.archive)))


if __name__ == '__main__':
    main()
