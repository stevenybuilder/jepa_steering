"""Read-only source comparisons; no simulator, model, or scientific launch.

Download only hash-bound source bytes. A source-data pass is not a complete
historical-exposure or receiving-runtime pass. Preserve every audit attempt.
"""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import sys
import time
import zipfile

import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/offline_study'
BANK = BASE / 'fresh-simulator-banks-20260912-v1'
OUT = BASE / 'four-task-source-exposure-20260912-v1'
REVISION = '6116f042ae7ae4c8e3f1fd2f194f432615664182'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def credential():
    # Parse only relevant entries; never execute or print .env contents.
    for line in (ROOT / '.env').read_text().splitlines():
        match = re.match(r'^\s*(?:export\s+)?(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN)\s*=\s*(.*)$', line)
        if match:
            parts = shlex.split(match.group(2), comments=True)
            if len(parts) == 1 and parts[0]:
                return parts[0]
    raise ValueError('No configured source-download credential')


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, token):
        self.url, self.size, self.token = url, size, token
        self.position = 0
        self.received = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= position <= self.size:
            raise ValueError('Out-of-bounds archive seek')
        self.position = position
        return position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count == 0:
            return b''
        if count > 16 * 1024**2 or self.received + count > 64 * 1024**2:
            raise ValueError('Bounded selective archive-read budget exceeded')
        end = self.position + count - 1
        headers = {'Authorization': 'Bearer ' + self.token, 'Range': f'bytes={self.position}-{end}'}
        url = self.url + f'?download=true&audit_range={self.position}-{end}'
        with requests.get(url, headers=headers, stream=True, timeout=30) as response:
            expected = f'bytes {self.position}-{end}/{self.size}'
            if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError(f'Source did not honor exact range: HTTP {response.status_code}')
            raw = response.raw.read(count + 1)
        if len(raw) != count:
            raise ValueError('Partial or oversized range response')
        self.position += count
        self.received += count
        return raw


def candidates(task):
    rows = []
    protocol_hash = digest((BANK / 'protocol.json').read_bytes())
    assert json.loads((BANK / 'FROZEN.json').read_text())['protocol_sha256'] == protocol_hash
    for rank in range(8):
        folder = BANK / task / f'rank-{rank:02d}'
        report = json.loads((folder / 'report.json').read_text())
        assert report['protocol_sha256'] == protocol_hash
        assert report['records_sha256'] == digest((folder / 'records.json').read_bytes())
        these = json.loads((folder / 'records.json').read_text())
        assert len(these) == 13
        for row in these:
            assert digest((folder / row['tensor_file']).read_bytes()) == row['tensor_sha256']
        rows.extend(these)
    assert sum(r['role'] == 'scientific_candidates' for r in rows) == 96
    assert sum(r['role'] == 'excluded_engineering' for r in rows) == 8
    return rows


def array_hash(value):
    value = np.ascontiguousarray(value)
    return digest(str((value.shape, value.dtype.str)).encode() + value.tobytes())


def navigation(task, token):
    spec = json.loads((ROOT / 'configs/navigation_assets.json').read_text())
    asset = next(x for x in spec['assets'] if x['task'] == task and x['kind'] == 'dataset_archive')
    assert asset['revision'] == REVISION
    cohort_root = BASE / 'primary-durable-20260907/navigation-offline-cohorts-20260907-v1' / task
    cohort = json.loads((cohort_root / 'cohort.json').read_text())
    manifest_raw = (cohort_root / 'source_manifest.json').read_bytes()
    assert digest(manifest_raw) == cohort['source_manifest_sha256']
    file_raw = (cohort_root / 'input_files.json').read_bytes()
    assert digest(file_raw) == cohort['source_input_files_sha256']
    hashes = json.loads(file_raw)
    names = ['states.pth', 'seq_lengths.pth'] if task == 'pointmaze' else ['states.pth', 'door_locations.pth', 'wall_locations.pth']
    url = f'https://huggingface.co/datasets/{asset["repo_id"]}/resolve/{REVISION}/{asset["filename"]}'
    reader = RangeReader(url, asset['size'], token)
    tensors, receipts = {}, []
    target = OUT / task
    target.mkdir()
    with zipfile.ZipFile(reader) as archive:
        for name in names:
            matches = [p for p in archive.namelist() if Path(p).name == name and not p.startswith('__MACOSX/')]
            if len(matches) != 1:
                raise ValueError('Nonunique source member: ' + name)
            raw = archive.read(matches[0])
            if digest(raw) != hashes[name]:
                raise ValueError('Source-array hash differs: ' + name)
            with (target / name).open('xb') as stream:
                stream.write(raw)
            tensors[name] = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            receipts.append({'member': matches[0], 'bytes': len(raw), 'sha256': digest(raw)})
    states = tensors['states.pth'].float() if task == 'pointmaze' else tensors['states.pth']
    manifest = json.loads(manifest_raw)
    assert len(manifest) == len(states)
    fingerprints = set()
    for row in manifest:
        i = row['index']
        initial = states[i, 0].reshape(-1)
        if task == 'wall':
            initial = torch.cat([initial, tensors['door_locations.pth'][i, 0].reshape(-1), tensors['wall_locations.pth'][i, 0].reshape(-1)])
        group = task + '/initial/' + array_hash(initial.numpy())
        assert group == row['lineage_group'], 'Raw source family differs from historical manifest'
        fingerprints.add(group)
    # Compare all valid recorded states, stronger than only initial-state families.
    if task == 'pointmaze':
        old_states = np.concatenate([states[i, :int(tensors['seq_lengths.pth'][i])].numpy() for i in range(len(states))])
    else:
        old_states = states.reshape(-1, states.shape[-1]).numpy()
    old_bytes = {np.ascontiguousarray(row).tobytes() for row in old_states}
    collisions = []
    for row in candidates(task):
        for key in ('initial_state', 'goal_state'):
            value = np.asarray(row[key], dtype=old_states.dtype)
            if value.tobytes() in old_bytes:
                collisions.append({'episode': row['episode'], 'role': row['role'], 'field': key})
    result = {'task': task, 'source_files': receipts, 'downloaded_range_bytes': reader.received,
              'raw_family_fingerprints_reproduced': len(manifest), 'recorded_states_compared': len(old_states),
              'candidate_endpoint_checks': 208, 'exact_state_matches': collisions,
              'comparison': 'All candidate initial/goal states at source dtype against all released valid states; Wall comparison conservatively ignores geometry',
              'passed_source_comparison': not collisions, 'historical_behavioral_audit_complete': False,
              'scientific_launch_ready': False}
    write(target / 'report.json', result)
    return result


def metaworld(token):
    import pyarrow.parquet as pq
    raw_manifest = BASE / 'combined-fit-stimulus-preparation-20260908-v1/stimulus/official_manifest.json'
    inventory_path = BASE / '2026-09-07-metaworld-dedup-correction/metaworld-inventory-v4-deduplicated-exposure-preserving/trajectories.jsonl'
    inventory = {}
    with inventory_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['task'] in ('mw-reach', 'mw-reach-wall'):
                inventory[row['index']] = row
    spec = json.loads(raw_manifest.read_text())
    assert spec['data_revision'] == REVISION
    files = [x for x in spec['files'] if re.search(r'train-0010[2-7]-of-00126\.parquet$', x['path'])]
    assert len(files) == 6
    target = OUT / 'metaworld'; target.mkdir()
    def download(entry):
        with requests.get(entry['url'], headers={'Authorization': 'Bearer ' + token}, stream=True, timeout=30) as response:
            if response.status_code != 200:
                raise ValueError(f'Pinned source HTTP {response.status_code}')
            raw = response.raw.read(entry['bytes'] + 1)
        if len(raw) != entry['bytes'] or digest(raw) != entry['sha256']:
            raise ValueError('Parquet source size/hash mismatch')
        with (target / Path(entry['path']).name).open('xb') as stream:
            stream.write(raw)
        rows = pq.read_table(io.BytesIO(raw), columns=['task', 'seed', 'episode', 'states', 'actions']).to_pylist()
        return entry, rows
    sys.path.insert(0, str(ROOT / 'src'))
    from offline_study.data.inventory import _metaworld_trajectory_group
    all_rows, receipts = {'mw-reach': [], 'mw-reach-wall': []}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for entry, rows in pool.map(download, files):
            shard = int(re.search(r'train-(\d+)-', entry['path']).group(1))
            assert len(rows) == 100
            for local, row in enumerate(rows):
                source = inventory[100 * shard + local]
                states = torch.tensor(np.asarray(row['states'])[:-1], dtype=torch.float32)
                actions = torch.tensor(np.asarray(row['actions']), dtype=torch.float32)
                assert row['task'] == source['task'] and row['seed'] == source['source_seed'] and row['episode'] == source['source_episode']
                assert _metaworld_trajectory_group(row['task'], states, actions) == source['lineage_group']
                all_rows[row['task']].append(states.numpy())
            receipts.append({k: entry[k] for k in ('path', 'bytes', 'sha256')})
    results = []
    for task in ('reach', 'reach-wall'):
        states = np.concatenate(all_rows['mw-' + task])
        goals = states[:, -3:]
        # Do not silently assume the final coordinates expose the goal.
        goal_distinct = len({x.tobytes() for x in np.ascontiguousarray(goals)})
        nonzero = bool(np.any(goals != 0))
        old = {x.tobytes() for x in np.ascontiguousarray(goals)}
        matches = []
        for row in candidates(task):
            goal = np.asarray(row['rand_vec'][-3:], dtype=np.float32)
            if goal.tobytes() in old:
                matches.append({'episode': row['episode'], 'role': row['role']})
        results.append({'task': task, 'raw_rows_reconciled': len(all_rows['mw-' + task]),
                        'recorded_states_compared': len(states), 'distinct_recorded_target_vectors': goal_distinct,
                        'recorded_targets_nonzero': nonzero, 'candidate_target_matches': matches,
                        'target_mapping_semantics_verified': False,
                        'interpretation': 'Numeric target-coordinate comparison only; verify raw-state last-three semantics against upstream environment before certifying scenario separation',
                        'scientific_launch_ready': False})
    result = {'source_files': receipts, 'tasks': results, 'historical_behavioral_audit_complete': False,
              'scientific_launch_ready': False}
    write(target / 'report.json', result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(OUT / 'protocol.json', {'role': 'source_only_exposure_reconciliation',
          'source_sha256': digest(Path(__file__).read_bytes()), 'input_bank_sha256': digest((BANK / 'protocol.json').read_bytes()),
          'model_calls': 0, 'simulator_calls': 0, 'scientific_launch_ready': False,
          'source_revision': REVISION, 'all_history_coverage_claimed': False})
    results, failures = {}, {}
    token = credential()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(navigation, t, token): t for t in ('pointmaze', 'wall')}
        jobs[pool.submit(metaworld, token)] = 'metaworld'
        for future in concurrent.futures.as_completed(jobs):
            task = jobs[future]
            try:
                results[task] = future.result()
                print(json.dumps({'completed_source_audit': task}), flush=True)
            except Exception as exc:
                failures[task] = type(exc).__name__ + ': ' + str(exc)
                print(json.dumps({'source_audit_failure': task, 'error': failures[task]}), flush=True)
    report = {'results': results, 'failures': failures, 'seconds': time.monotonic() - started,
              'scientific_launch_ready': False, 'model_calls': 0, 'simulator_calls': 0}
    write(OUT / 'report.json', report)
    print(json.dumps(report), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
