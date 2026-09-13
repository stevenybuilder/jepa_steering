"""Snapshot published fresh-arm records on a receiver; never reads efficacy values."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def progress_counts(root, allowed=None):
    """Count published arms and hash-complete scenarios; never summarize outcomes."""
    names = {'native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only',
             'matched_random_coupling', 'joint', 'visual_only', 'action_condition_only'}
    frozen = json.loads((root / 'fresh-freeze-v2/FROZEN.json').read_text())['protocol_sha256']
    scientific, engineering, complete = 0, 0, 0
    output = root / 'results-v2'
    def included(path):
        return path.is_file() and (allowed is None or str(path.relative_to(root)) in allowed)
    for task in ('reach', 'reach-wall', 'pointmaze', 'wall'):
        for scenario in (output / task).glob('scenario-*'):
            published = {}
            for arm in names:
                path = scenario / (arm + '.json')
                if not included(path):
                    continue
                record = json.loads(path.read_text())
                if (record.get('freeze_sha256') != frozen or record.get('arm') != arm
                        or record.get('scientific_efficacy_measurement') is not True):
                    raise ValueError('Published science count identity mismatch')
                published[arm] = digest(path)
            scientific += len(published)
            if included(scenario / 'DONE.json') and included(scenario / 'report.json'):
                report = json.loads((scenario / 'report.json').read_text())
                done = json.loads((scenario / 'DONE.json').read_text())
                if (done['report_sha256'] != digest(scenario / 'report.json')
                        or report['freeze_sha256'] != frozen or report['engineering']
                        or set(published) != names or report['records_sha256'] != published):
                    raise ValueError('Complete scenario hash verification failed')
                complete += 1
        for path in (output / 'engineering' / task).glob('*/*.json'):
            if path.stem not in ('native', 'native_repeat') or not included(path):
                continue
            record = json.loads(path.read_text())
            if record.get('freeze_sha256') == frozen and record.get('scientific_efficacy_measurement') is False:
                engineering += 1
    return {'published_scientific_arms': scientific, 'published_engineering_arms': engineering,
            'hash_verified_complete_eight_arm_scenarios': complete,
            'planned_scientific_arms': 3072, 'planned_scenarios': 384,
            'efficacy_values_not_reported': True}


def snapshot(root, archive, allow_empty=False):
    root = root.resolve()
    selected, records = set(), []
    freeze = json.loads((root / 'fresh-freeze-v2/FROZEN.json').read_text())
    freeze_hash = freeze['protocol_sha256']
    for path in (root / 'results-v2').rglob('*.json'):
        if path.is_symlink():
            raise ValueError('Symlink in receiving results')
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if 'scientific_efficacy_measurement' not in record:
            continue
        if record['freeze_sha256'] != freeze_hash or path.stem != record['arm']:
            raise ValueError('Published record identity mismatch')
        selected.add(path)
        records.append({'path': str(path.relative_to(root)),
                        'scientific': record['scientific_efficacy_measurement']})
        # An atomically published arm record marks these auxiliary files complete.
        auxiliary = path.parent / record['arm']
        if auxiliary.is_dir():
            selected.update(p for p in auxiliary.rglob('*') if p.is_file())
        for name in ('INTENT.json', 'STARTED.json', 'report.json', 'DONE.json'):
            candidate = path.parent / name
            if candidate.exists():
                selected.add(candidate)
    selected.update(p for p in (root / 'fresh-freeze-v2').rglob('*') if p.is_file())
    if not records and not allow_empty:
        raise ValueError('No published records to preserve')
    members = []
    for path in sorted(selected):
        if path.is_symlink() or not path.is_relative_to(root):
            raise ValueError('Unsafe snapshot member')
        members.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                        'sha256': digest(path)})
    manifest = {'created_unix': time.time(), 'freeze_sha256': freeze_hash,
                'records': records, 'members': members,
                'incomplete_arm_files_excluded': True,
                'whole_scenario_completion_not_implied': True}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'x:gz') as out:
        for member in members:
            out.add(root / member['path'], arcname=member['path'], recursive=False)
        encoded = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('SNAPSHOT_MANIFEST.json')
        info.size = len(encoded)
        out.addfile(info, io.BytesIO(encoded))
    for member in members:
        if digest(root / member['path']) != member['sha256']:
            raise ValueError('Published file changed during snapshot')
    # Verify the actual archived payload, not just the inputs.
    with tarfile.open(archive) as src:
        for member in members:
            if hashlib.sha256(src.extractfile(member['path']).read()).hexdigest() != member['sha256']:
                raise ValueError('Archive member hash mismatch')
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
               'sha256': digest(archive), 'records': len(records),
               'scientific_records': sum(r['scientific'] for r in records),
               'all_members_verified': True,
               'progress': progress_counts(root, {m['path'] for m in members})}
    print(json.dumps(receipt), flush=True)


def diagnostic_snapshot(root, archive, names):
    """Preserve startup/partial bytes without calling them complete science."""
    root = root.resolve()
    selected = set()
    for name in names:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Diagnostic roots must be exact immediate children')
        source = root / name
        if source.is_symlink():
            raise ValueError('Symlinked diagnostic root')
        if source.is_dir():
            selected.update(p for p in source.rglob('*') if p.is_file() and not p.is_symlink())
        elif source.is_file():
            selected.add(source)
    if not selected:
        raise ValueError('No diagnostic files exist yet')
    archive.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with tarfile.open(archive, 'x:gz') as out:
        for path in sorted(selected):
            if path.name in ('rclone.conf', 'credentials.json') or path.suffix in ('.key', '.pem'):
                raise ValueError('Credential-like diagnostic file')
            content = path.read_bytes()
            name = str(path.relative_to(root))
            info = tarfile.TarInfo(name)
            info.size = len(content)
            out.addfile(info, io.BytesIO(content))
            members.append({'path': name, 'bytes': len(content),
                            'sha256': hashlib.sha256(content).hexdigest()})
        manifest = {'diagnostic_snapshot': True, 'scientific_completion_not_implied': True,
                    'files_may_have_been_active_at_capture': True, 'members': members}
        content = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('SNAPSHOT_MANIFEST.json')
        info.size = len(content)
        out.addfile(info, io.BytesIO(content))
    with tarfile.open(archive) as src:
        for member in members:
            if hashlib.sha256(src.extractfile(member['path']).read()).hexdigest() != member['sha256']:
                raise ValueError('Diagnostic archive member hash mismatch')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': digest(archive), 'diagnostic_files': len(members),
                      'all_members_verified': True, 'scientific_completion_not_implied': True,
                      'progress': progress_counts(root, {m['path'] for m in members})}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--allow-empty', action='store_true', help='Permit explicitly counted zero-record startup archives')
    parser.add_argument('--include', action='append', default=[])
    args = parser.parse_args()
    if args.diagnostic:
        diagnostic_snapshot(args.root, args.archive, args.include)
    else:
        snapshot(args.root, args.archive, args.allow_empty)
