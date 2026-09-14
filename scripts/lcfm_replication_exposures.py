"""Collect retained project input identities before the new LCFM replication."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def identities(value):
    seeds, hashes, vectors = set(), set(), set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == 'environment_seed' and type(value) is int:
                    seeds.add(value)
                if key in ('initial_sha256', 'goal_sha256') and isinstance(value, str):
                    hashes.add(value)
                if key in ('rand_vec', 'initial_state', 'goal_state', 'sampled_initial_state', 'initial_state_vector'):
                    if isinstance(value, list) and value and all(type(x) in (int, float) for x in value):
                        vectors.add(tuple(float(x) for x in value))
                # Previously compiled exclusions include records whose sources
                # may now be archived. Retain their identities as well.
                if key == 'environment_seeds' and isinstance(value, list):
                    seeds.update(x for x in value if type(x) is int)
                if key == 'observation_hashes' and isinstance(value, list):
                    hashes.update(x for x in value if isinstance(x, str))
                if key == 'state_vectors' and isinstance(value, list):
                    vectors.update(tuple(float(x) for x in row) for row in value
                                   if isinstance(row, list) and row and all(type(x) in (int, float) for x in row))
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(x for x in item if isinstance(x, (dict, list)))
    return seeds, hashes, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new exposure snapshot path')
    root = Path(__file__).resolve().parents[1]
    pattern = '"initial_sha256"|"goal_sha256"|"environment_seed"|"rand_vec"|"observation_hashes"'
    names = subprocess.check_output(['rg', '-l', '--hidden', '--no-ignore',
        '-g', '*.json', '-g', '*.jsonl', pattern, 'artifacts', 'archive', 'reports', 'paper/data'],
        cwd=root, text=True).splitlines()
    seeds, hashes, vectors, sources = set(), set(), set(), {}
    for index, name in enumerate(sorted(names)):
        path = root/name
        raw = path.read_bytes()
        values = [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(raw)]
        for value in values:
            a, b, c = identities(value)
            seeds.update(a); hashes.update(b); vectors.update(c)
        sources[name] = hashlib.sha256(raw).hexdigest()
        if index % 2000 == 0:
            print(json.dumps({'files_scanned': index+1, 'total': len(names)}), flush=True)
    receipt = {'role': 'retained_project_exposure_snapshot_before_new_inputs',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_files_sha256': sources, 'environment_seeds': sorted(seeds),
        'observation_hashes': sorted(hashes), 'state_vectors': sorted(vectors),
        'limitations': 'All matching retained project JSON/JSONL, including archives and compiled exclusions; no claim about unlogged external work or model pretraining.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'files': len(sources), 'seeds': len(seeds), 'hashes': len(hashes),
                      'vectors': len(vectors), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
