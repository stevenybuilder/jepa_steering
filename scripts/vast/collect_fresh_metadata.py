"""Cloud-first immutable fresh-panel metadata assembly; no efficacy selection."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import urllib.parse
import urllib.request

FREEZE = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
TASKS = ('reach', 'reach-wall', 'pointmaze', 'wall')
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only',
        'matched_random_coupling', 'joint', 'visual_only', 'action_condition_only')
HOSTS = (50806821, 50819364, 50827072, 50828584, 50828583, 50833029, 50836730, 50839961)
BUCKET = 'rgt-jepa-archive-2026'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def immutable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Symlink destination')
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Conflicting immutable file: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(content)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def selected(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe archive path')
    parts = p.parts
    return ((len(parts) == 4 and parts[0] == 'results-v2' and parts[1] in TASKS
             and parts[2].startswith('scenario-') and parts[3].endswith('.json'))
            or (len(parts) == 5 and parts[:2] == ('results-v2', 'engineering')
                and parts[2] in TASKS and parts[4].endswith('.json'))
            or name in ('fresh-freeze-v2/protocol.json', 'fresh-freeze-v2/FROZEN.json'))


def download(source, scratch):
    obj = source['object'].removeprefix('gs://' + BUCKET + '/')
    if (source['instance_id'] not in HOSTS or not obj.startswith(
            'fresh-campaign-20260912-v2/' + str(source['instance_id']) + '/published-')
            or not source['gcs_download_sha256_verified']):
        raise ValueError('Unapproved source archive')
    path = scratch / (source['sha256'] + '.tgz')
    if path.exists():
        if file_sha(path) != source['sha256']:
            raise ValueError('Existing archive changed')
        return path
    url = ('https://storage.googleapis.com/download/storage/v1/b/' + BUCKET + '/o/'
           + urllib.parse.quote(obj, safe='') + '?alt=media&generation=' + str(source['generation']))
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + source['token']})
    partial = path.with_suffix('.partial')
    with urllib.request.urlopen(req, timeout=90) as response, partial.open('xb') as out:
        for block in iter(lambda: response.read(1048576), b''):
            out.write(block)
    if partial.stat().st_size != source['bytes'] or file_sha(partial) != source['sha256']:
        raise ValueError('Downloaded archive hash/size mismatch')
    partial.rename(path)
    return path


def read_archive(path):
    data = {}
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        for member in members:
            wanted = selected(member.name)
            if not member.isfile():
                raise ValueError('Non-regular archive member')
            if wanted or member.name == 'SNAPSHOT_MANIFEST.json':
                data[member.name] = archive.extractfile(member).read()
    manifest = json.loads(data.pop('SNAPSHOT_MANIFEST.json'))
    if manifest['freeze_sha256'] != FREEZE or sha(data['fresh-freeze-v2/protocol.json']) != FREEZE:
        raise ValueError('Wrong frozen protocol')
    index = {m['path']: m for m in manifest['members']}
    if len(index) != len(manifest['members']):
        raise ValueError('Duplicate manifest member')
    for name, content in data.items():
        m = index[name]
        if len(content) != m['bytes'] or sha(content) != m['sha256']:
            raise ValueError('Metadata member hash mismatch')
    return data


def bundle(data, prefix, task, row, engineering, uuid=None):
    report = json.loads(data[prefix + '/report.json'])
    done = json.loads(data[prefix + '/DONE.json'])
    arms = ('native', 'native_repeat') if engineering else ARMS
    if (done['report_sha256'] != sha(data[prefix + '/report.json'])
            or report['freeze_sha256'] != FREEZE or report['task'] != task
            or report['scenario'] != row or report['engineering'] != engineering
            or set(report['records_sha256']) != set(arms)
            or (uuid is not None and report['device_uuid'] != uuid)):
        raise ValueError('Wrong bundle role/input/UUID/report hash')
    names = [prefix + '/' + a + '.json' for a in arms]
    for arm, name in zip(arms, names):
        if sha(data[name]) != report['records_sha256'][arm]:
            raise ValueError('Arm hash mismatch')
    started = prefix + '/STARTED.json'
    if json.loads(data[started]) != {'task': task, 'scenario': row,
            'device_uuid': report['device_uuid'], 'freeze_sha256': FREEZE, 'engineering': engineering}:
        raise ValueError('STARTED binding mismatch')
    return report, names + [prefix + '/report.json', prefix + '/DONE.json', started]


def ingest(root, source, archive):
    data = read_archive(archive)
    protocol = json.loads(data['fresh-freeze-v2/protocol.json'])
    expected = {(task, row['episode']): row for task in TASKS
                for row in protocol['tasks'][task]['records'] if row['role'] == 'scientific_candidates'}
    if set(expected) != {(t, e) for t in TASKS for e in range(96)}:
        raise ValueError('Frozen panel mismatch')
    # Reject even unexpected incomplete scientific entries; do not silently filter them.
    prefixes = set()
    for name in data:
        parts = PurePosixPath(name).parts
        if len(parts) == 4 and parts[0] == 'results-v2' and parts[1] in TASKS:
            if parts[2] not in {f'scenario-{e:03d}' for e in range(96)}:
                raise ValueError('Unexpected scenario entry')
            prefixes.add('/'.join(parts[:3]))
    imported = 0
    for prefix in sorted(prefixes):
        if prefix + '/DONE.json' not in data:
            continue
        task, scenario = prefix.split('/')[1:]
        episode = int(scenario.split('-')[1])
        report, names = bundle(data, prefix, task, expected[task, episode], False)
        ep = 'results-v2/engineering/' + task + '/' + report['device_uuid']
        er = json.loads(data[ep + '/report.json'])
        if er['scenario'] not in [r for r in protocol['tasks'][task]['records'] if r['role'] == 'excluded_engineering']:
            raise ValueError('Engineering is not excluded input')
        _, enames = bundle(data, ep, task, er['scenario'], True, report['device_uuid'])
        if report['engineering_report_sha256'] != sha(data[ep + '/report.json']):
            raise ValueError('Engineering provenance chain mismatch')
        owner = {'instance_id': source['instance_id'], 'report_sha256': sha(data[prefix + '/report.json'])}
        immutable(root / 'owners' / task / (scenario + '.json'), encoded(owner))
        for name in names + enames:
            immutable(root / name, data[name])
        imported += 1
    return {'instance_id': source['instance_id'], 'object': source['object'],
            'generation': source['generation'], 'sha256': source['sha256'], 'complete_cases_in_archive': imported}


def inventory(root):
    pairs = []
    for task in TASKS:
        for p in (root / 'results-v2' / task).glob('*'):
            if p.name not in {f'scenario-{e:03d}' for e in range(96)}:
                raise ValueError('Unexpected collected scenario')
            if not (p / 'DONE.json').is_file():
                raise ValueError('Partially committed collection bundle')
            pairs.append((task, int(p.name.split('-')[1])))
    if len(pairs) != len(set(pairs)):
        raise ValueError('Duplicate collected scenario')
    return {'complete_cases': len(pairs), 'scientific_arms': len(pairs) * 8,
            'per_task_cases': {t: sum(x[0] == t for x in pairs) for t in TASKS},
            'full_exact_panel': set(pairs) == {(t, e) for t in TASKS for e in range(96)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if root != Path('/workspace/fresh-four-20260912-v1/collection-v2') or root.is_symlink():
        raise ValueError('Unexpected collection destination')
    request = json.load(sys.stdin)
    root.mkdir(exist_ok=True)
    scratch = root / 'source-archives'
    scratch.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda s: download(s, scratch), request['sources']))
    receipts = [ingest(root, s, p) for s, p in zip(request['sources'], paths)]
    result = {'utc': time.time(), 'freeze_sha256': FREEZE, 'sources': receipts,
              **inventory(root), 'efficacy_not_summarized': True, 'analysis_not_run': True}
    immutable(root / ('MANIFEST_' + request['generation'] + '.json'), encoded(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
