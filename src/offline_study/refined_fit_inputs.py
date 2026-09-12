"""Exact official-byte inputs for the bounded successor response fit.

Push-T extracts only preselected fitting videos and loader metadata from the
already hash-pinned release. No validation clip is read by this fitting helper.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

from .protocol import sha256, write_json

PUSHT_ARCHIVE_SHA = '442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08'
PUSHT_ARCHIVE_BYTES = 2785304515


def selected_files(cohort):
    from .refined_task_fit import selection
    if cohort['task'] != 'pusht':
        raise ValueError('Push-T extraction cannot be reused for another dataset')
    selected = selection(cohort)
    rows = [row for role in selected.values() for row in role]
    # Include original excluded receiving-fit stimulus, not any behavioral row.
    indices = {row['index'] for row in rows} | {cohort['fit'][0]['index']}
    names = {f'train/obses/episode_{i:03d}.mp4' for i in indices}
    names.update('train/' + name for name in
        ('states.pth', 'rel_actions.pth', 'velocities.pth', 'seq_lengths.pkl'))
    return names


def extract(archive, cohort_path, output):
    cohort = json.loads(cohort_path.read_text())
    names = selected_files(cohort)
    if archive.stat().st_size != PUSHT_ARCHIVE_BYTES or sha256(archive) != PUSHT_ARCHIVE_SHA:
        raise ValueError('Wrong official Push-T archive')
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        if 'pusht_noise/train/shapes.pkl' in source.namelist():
            names.add('train/shapes.pkl')
        for name in sorted(names):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('pusht_noise/' + name) as stream, target.open('xb') as dest:
                shutil.copyfileobj(stream, dest, 4 << 20)
    proof = {'task': 'pusht', 'cohort_sha256': sha256(cohort_path),
        'archive_sha256': PUSHT_ARCHIVE_SHA, 'archive_bytes': PUSHT_ARCHIVE_BYTES,
        'files': {name: sha256(output / name) for name in sorted(names)},
        'validation_clips_extracted': False}
    write_json(output / 'FIT_INPUTS_VERIFIED.json', proof)
    return proof


def verify(cohort_path, root):
    cohort = json.loads(cohort_path.read_text())
    if cohort['task'] in ('wall', 'pointmaze'):
        from .navigation_cohort import verify_inputs
        verify_inputs(cohort_path, root)
        return {'cohort_sha256': sha256(cohort_path),
            'input_files_sha256': sha256(cohort_path.parent / 'input_files.json')}
    proof_path = root / 'FIT_INPUTS_VERIFIED.json'
    proof = json.loads(proof_path.read_text())
    names = selected_files(cohort)
    if (root / 'train/shapes.pkl').exists():
        names.add('train/shapes.pkl')
    if (proof.get('task') != 'pusht' or proof.get('cohort_sha256') != sha256(cohort_path)
            or proof.get('archive_sha256') != PUSHT_ARCHIVE_SHA
            or proof.get('archive_bytes') != PUSHT_ARCHIVE_BYTES
            or proof.get('validation_clips_extracted') is not False
            or set(proof['files']) != names):
        raise ValueError('Changed fitting input population or archive binding')
    for name in names:
        if sha256(root / name) != proof['files'][name]:
            raise ValueError('Fitting bytes changed: ' + name)
    return {'cohort_sha256': sha256(cohort_path), 'input_proof_sha256': sha256(proof_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'cohort', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.cohort, args.output)))
