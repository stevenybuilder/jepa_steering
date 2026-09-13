"""Extend the preserved JSON exposure audit; never clears scientific launch."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'artifacts/offline_study/fresh-simulator-banks-20260912-v1'
OUT = ROOT / 'artifacts/offline_study/exposure-omission-audit-20260912-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identities(value):
    seeds, hashes, vectors = set(), set(), set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for k, v in item.items():
                if k == 'environment_seed' and type(v) is int:
                    seeds.add(v)
                if k in ('initial_sha256', 'goal_sha256') and isinstance(v, str):
                    hashes.add(v)
                if k in ('rand_vec', 'initial_state', 'goal_state', 'sampled_initial_state'):
                    if isinstance(v, list) and v and all(type(x) in (int, float) for x in v):
                        vectors.add(tuple(float(x) for x in v))
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(item, list):
            stack.extend(v for v in item if isinstance(v, (dict, list)))
    return seeds, hashes, vectors


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    old = json.loads((BANK / 'exclusions.json').read_text())
    protocol = {'role': 'historical_input_identity_audit_no_model_evaluation',
                'source_sha256': sha(Path(__file__)),
                'exclusions_sha256': sha(BANK / 'exclusions.json'),
                'files': old['oversized_json_not_scanned'],
                'scope': 'all 101 files omitted by the existing 8-MiB JSON cutoff; not raw/off-host completeness',
                'scientific_launch_ready': False}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    started = time.monotonic()
    seeds, hashes, vectors, records, failures = set(), set(), set(), [], []
    for name in protocol['files']:
        p = ROOT / name
        try:
            before = sha(p)
            with p.open() as f:
                value = json.load(f)
            s, h, v = identities(value)
            del value
            if sha(p) != before:
                raise ValueError('Source changed during audit')
            seeds.update(s); hashes.update(h); vectors.update(v)
            records.append({'path': name, 'sha256': before, 'bytes': p.stat().st_size,
                            'seeds': len(s), 'observation_hashes': len(h), 'vectors': len(v)})
        except Exception as exc:
            failures.append({'path': name, 'error': repr(exc)})
        print(json.dumps({'files_scanned': len(records), 'failures': len(failures)}), flush=True)
    collisions = []
    for task in ('reach', 'reach-wall', 'pointmaze', 'wall'):
        for p in sorted((BANK / task).glob('rank-*/records.json')):
            for row in json.loads(p.read_text()):
                s, h, v = identities(row)
                if s & seeds or h & hashes or v & vectors:
                    collisions.append({'task': task, 'rank_file': str(p.relative_to(ROOT)),
                                       'episode': row['episode'], 'role': row['role'],
                                       'seed_matches': sorted(s & seeds),
                                       'hash_matches': sorted(h & hashes), 'state_matches': len(v & vectors)})
    result = {'protocol_sha256': sha(OUT / 'protocol.json'), 'files': records, 'failures': failures,
              'unique_environment_seeds': len(seeds), 'unique_observation_hashes': len(hashes),
              'unique_state_vectors': len(vectors), 'candidate_collisions': collisions,
              'all_omitted_files_scanned': len(records) == len(protocol['files']) and not failures,
              'remaining_gaps': ['raw/off-host trajectory coverage', 'JSONL source-family reconciliation',
                                 'receiving-runtime parity', 'prospective scientific execution/analysis freeze'],
              'model_calls': 0, 'scientific_launch_ready': False, 'seconds': time.monotonic() - started}
    (OUT / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'failures')}), flush=True)
    if failures or collisions:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
