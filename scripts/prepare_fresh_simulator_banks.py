"""Fresh input banks for the four tasks with unchanged native goal generators.

Expert rollouts only define MetaWorld goals. Never calls a learned policy/planner.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from offline_study.planning.planning_contract import prepare, seed_schedule
from offline_study.core.protocol import sha256, write_json

VENDOR = ROOT / 'vendor/jepa-wms'
OUT = ROOT / 'artifacts/offline_study/fresh-simulator-banks-20260912-v1'
TASKS = ('reach', 'reach-wall', 'pointmaze', 'wall')
BASE = 2026091217


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    seeds, hashes, vectors, sources, skipped = set(), set(), set(), {}, []
    def scan(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'environment_seed' and isinstance(item, int):
                    seeds.add(item)
                if key in ('initial_sha256', 'goal_sha256') and isinstance(item, str):
                    hashes.add(item)
                if key in ('rand_vec', 'initial_state', 'goal_state') and isinstance(item, list):
                    if item and all(type(v) in (int, float) for v in item):
                        vectors.add(tuple(float(v) for v in item))
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
    for path in (ROOT / 'artifacts/offline_study').rglob('*.json'):
        if OUT in path.parents or path.name.startswith('._'):
            continue
        if path.stat().st_size > 8 * 1024**2:
            skipped.append(str(path.relative_to(ROOT))); continue
        try:
            value = json.loads(path.read_text())
        except (UnicodeError, json.JSONDecodeError):
            continue
        before = (len(seeds), len(hashes), len(vectors))
        scan(value)
        if before != (len(seeds), len(hashes), len(vectors)):
            sources[str(path.relative_to(ROOT))] = sha256(path)
    if not seeds or not hashes or not vectors:
        raise ValueError('Historical input audit found incomplete evidence')
    write_json(OUT / 'exclusions.json', {'environment_seeds': sorted(seeds),
        'observation_hashes': sorted(hashes), 'state_vectors': sorted(vectors),
        'source_hashes': sources, 'oversized_json_not_scanned': skipped,
        'coverage': 'retained local JSON input identities, not exhaustive raw/off-host trajectory audit'})
    contracts = {}
    for index, task in enumerate(TASKS):
        contract = prepare(VENDOR, task)
        schedule = seed_schedule(BASE + index * 100000)
        engineering = seed_schedule(BASE + index * 100000 + 50000, episodes=8)
        if {r['environment_seed'] for r in schedule + engineering} & seeds:
            raise ValueError('New schedule overlaps historical environment seeds')
        contract['config']['meta']['seed'] = BASE + index * 100000
        contract['config']['device'] = 'cpu'
        contracts[task] = {'planning_contract': contract, 'scientific_candidates': schedule,
                           'excluded_engineering': engineering}
    write_json(OUT / 'protocol.json', {'role': 'fresh_input_preparation_not_evaluation',
        'script_sha256': sha256(Path(__file__)), 'tasks': contracts,
        'exclusions_sha256': sha256(OUT / 'exclusions.json'),
        'existing_generator_sha256': sha256(ROOT / 'src/offline_study/planning/planning_scenarios.py'),
        'vendor_commit': '13cf1d9c7e476f53c17714d2e0f1dc239a883ce0',
        'outcome_filtering': False, 'learned_policy_calls': 0,
        'scientific_launch_ready': False, 'source_family_audit_complete': False})
    write_json(OUT / 'FROZEN.json', {'protocol_sha256': sha256(OUT / 'protocol.json')})


def run(task, rank):
    import numpy as np
    import torch
    from offline_study.models.vendor import use_vendor
    use_vendor(VENDOR)
    from offline_study.planning.planning_scenarios import prepare_episode
    from offline_study.planning.planning_env_smoke import observation_digest
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.utils import make_td
    p = json.loads((OUT / 'protocol.json').read_text())
    if sha256(OUT / 'protocol.json') != json.loads((OUT / 'FROZEN.json').read_text())['protocol_sha256']:
        raise ValueError('Input protocol changed')
    if p['script_sha256'] != sha256(Path(__file__)) or p['existing_generator_sha256'] != sha256(ROOT / 'src/offline_study/planning/planning_scenarios.py'):
        raise ValueError('Input generation source changed')
    if sha256(OUT / 'exclusions.json') != p['exclusions_sha256']:
        raise ValueError('Exposure exclusions changed')
    audit = json.loads((OUT / 'exclusions.json').read_text())
    old_vectors = {tuple(v) for v in audit['state_vectors']}
    old_hashes = set(audit['observation_hashes'])
    target = OUT / task / f'rank-{rank:02d}'; target.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); records = []
    try:
        for role in ('excluded_engineering', 'scientific_candidates'):
            rows = [r for r in p['tasks'][task][role] if r['logical_rank'] == rank]
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            cfg = OmegaConf.create(copy.deepcopy(p['tasks'][task]['planning_contract']['config']))
            cfg.local_seed = rows[0]['local_seed']
            agent = SimpleNamespace(local_generator=torch.Generator().manual_seed(cfg.local_seed))
            env = make_env(cfg)
            try:
                for row in rows:
                    if task in ('reach', 'reach-wall'):
                        record, tensors = prepare_episode(cfg, agent, env, row)
                        vecs = [record['rand_vec'], record['goal_state']]
                    else:
                        initial_state, goal_state = env.sample_random_init_goal_states(row['environment_seed'])
                        goal, goal_info = env.prepare(row['environment_seed'], goal_state)
                        initial, initial_info = env.prepare(row['environment_seed'], initial_state)
                        init_td, goal_td = make_td(initial, initial_info), make_td(goal, goal_info)
                        record = {**row, 'initial_state': initial_state.tolist(), 'goal_state': goal_state.tolist(),
                            'initial_sha256': observation_digest(init_td), 'goal_sha256': observation_digest(goal_td)}
                        tensors = {'initial': {k: init_td[k].detach().cpu() for k in ('visual','proprio')},
                                   'goal': {k: goal_td[k].detach().cpu() for k in ('visual','proprio')}}
                        vecs = [record['initial_state'], record['goal_state']]
                    if any(tuple(v) in old_vectors for v in vecs) or {record['initial_sha256'], record['goal_sha256']} & old_hashes:
                        raise ValueError('Historical input collision; retain and stop without replacement')
                    filename = f'{role}-{row["episode"]:03d}.pt'
                    torch.save(tensors, target / filename)
                    records.append({**record, 'role': role, 'tensor_file': filename,
                                    'tensor_sha256': sha256(target / filename)})
                    write_json(target / 'progress.json', {'prepared': len(records), 'expected': len(rows) + 1,
                        'learned_policy_calls': 0, 'seconds': time.monotonic() - start})
            finally:
                env.close()
        write_json(target / 'records.json', records)
        write_json(target / 'report.json', {'status': 'native_input_shard_prepared', 'task': task, 'rank': rank,
            'scientific_candidates': sum(r['role'] == 'scientific_candidates' for r in records),
            'excluded_engineering': sum(r['role'] == 'excluded_engineering' for r in records),
            'records_sha256': sha256(target / 'records.json'), 'protocol_sha256': sha256(OUT / 'protocol.json'),
            'seconds': time.monotonic() - start, 'learned_policy_calls': 0,
            'full_exposure_audit_complete': False, 'receiving_device_parity': False, 'scientific_launch_ready': False})
    except Exception as exc:
        write_json(target / 'FAILED.json', {'error': str(exc), 'partial_inputs_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'run'))
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--rank', type=int, choices=range(8))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze()
    else:
        if args.task is None or args.rank is None:
            parser.error('run requires --task and --rank')
        run(args.task, args.rank)
