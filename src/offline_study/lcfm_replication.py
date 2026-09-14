"""Fresh-input replication of the frozen action-history diagnostic.

Reuse the original 35-arm computation without changing its kernel or audits.
Input generation and whole-case execution have separate, hashed contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from .action_counterfactual_pilot import ARMS, CHECKPOINT_SHA, run_case
from .action_condition_specificity import initial_population, verify_native_sampler
from .planning_contract import prepare as planning_config
from .planning_env_smoke import observation_digest
from .planning_scenarios import prepare_episode
from .protocol import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def validate_protocol(protocol):
    if (protocol['role'] != 'fresh_action_history_replication'
            or protocol['arms'] != list(ARMS)
            or protocol['checkpoint_sha256'] != CHECKPOINT_SHA
            or protocol['precision'] != 'float32_strict_no_tf32'
            or protocol['candidate_shape'] != [6, 300, 20]
            or protocol['banks'] != ['original', 'fresh']):
        raise ValueError('Replication computation differs from the registered method')
    keys = [(r['task'], r['episode']) for r in protocol['scenarios']]
    if len(keys) != 200 or set(keys) != {(t, e) for t in ('reach', 'reach-wall') for e in range(100)}:
        raise ValueError('Require exactly 100 independent scenarios per task')
    all_rows = protocol['scenarios'] + protocol['engineering']
    if len({r['environment_seed'] for r in all_rows}) != len(all_rows):
        raise ValueError('Repeated environment seed')
    if len(protocol['engineering']) != 2 or {r['task'] for r in protocol['engineering']} != {'reach', 'reach-wall'}:
        raise ValueError('Require excluded engineering scenarios for both tasks')


def verify_contract(args):
    protocol = read(args.protocol)
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError('Protocol bytes changed')
    validate_protocol(protocol)
    source = Path(__file__).resolve().parents[2]
    for name, digest in protocol['source_sha256'].items():
        if sha256(source/name) != digest:
            raise ValueError('Execution source changed: '+name)
    from .vendor import use_vendor
    use_vendor(args.vendor)
    for name, digest in protocol['vendor_sha256'].items():
        if sha256(args.vendor/name) != digest:
            raise ValueError('Pinned vendor changed: '+name)
    return protocol


def initialize(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare_inputs(args, protocol):
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    if sha256(args.exposures) != protocol['exposures_sha256']:
        raise ValueError('Exposure snapshot differs')
    exposed = read(args.exposures)
    old_seeds = set(exposed['environment_seeds'])
    old_hashes = set(exposed['observation_hashes'])
    old_vectors = {tuple(v) for v in exposed['state_vectors']}
    selected = [r for r in protocol['engineering']+protocol['scenarios'] if r['task'] == args.task]
    args.output.mkdir(parents=True, exist_ok=False)
    initialize()
    cfg = OmegaConf.create(planning_config(args.vendor, args.task)['config'])
    env = make_env(cfg)
    agent = SimpleNamespace(local_generator=torch.Generator())
    records = []
    seen = set()
    started = time.monotonic()
    try:
        for row in selected:
            if row['environment_seed'] in old_seeds:
                raise ValueError('Previously exposed seed; no automatic replacement')
            cfg.local_seed = row['local_seed']
            agent.local_generator.manual_seed(row['local_seed'])
            metadata, values = prepare_episode(cfg, agent, env, row)
            ids = {metadata['initial_sha256'], metadata['goal_sha256']}
            if ids & old_hashes or tuple(metadata['rand_vec']) in old_vectors:
                raise ValueError('Previously exposed input identity; retain diagnostics')
            if metadata['initial_sha256'] in seen:
                raise ValueError('Repeated initial input')
            seen.add(metadata['initial_sha256'])
            actions, rng_hash = initial_population(cfg, row['candidate_seed'], 'cuda:0')
            verify_native_sampler(cfg, row['candidate_seed'], actions, rng_hash, CEMPlanner)
            payload = {k: values[k] for k in ('initial', 'goal')}
            payload['actions'] = actions.cpu()
            filename = f"episode-{row['episode']:03d}.pt"
            torch.save(payload, args.output/filename)
            record = {**metadata, 'tensor_file': filename,
                      'inputs_sha256': sha256(args.output/filename), 'sampler_verified': True}
            records.append(record)
            write_json(args.output/'progress.json', {'prepared': len(records), 'total': len(selected),
                       'task': args.task, 'seconds': time.monotonic()-started})
    finally:
        env.close()
        write_json(args.output/'records.json', records)
    write_json(args.output/'PREPARED.json', {'protocol_sha256': args.protocol_sha256,
               'records_sha256': sha256(args.output/'records.json'), 'records': len(records),
               'task': args.task, 'learned_model_calls': 0,
               'all_expert_outcomes_retained': True, 'seconds': time.monotonic()-started})


def execute(args, protocol):
    from .backends import JepaBackend
    manifest = read(args.manifest)
    if sha256(args.manifest) != args.manifest_sha256 or manifest['protocol_sha256'] != args.protocol_sha256:
        raise ValueError('Execution manifest differs')
    if (manifest['exposure_audit']['passed'] is not True
            or manifest['exposure_audit']['scientific_scenarios'] != 200
            or sha256(args.checkpoint) != CHECKPOINT_SHA):
        raise ValueError('Fresh inputs or checkpoint are unverified')
    if torch.cuda.device_count() != 1 or os.environ.get('JEPA_VERIFIED_LOCAL_DINO') != '1':
        raise ValueError('Require one explicitly assigned GPU and verified local encoder')
    gpu = str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower()
    if gpu != args.gpu_uuid.removeprefix('GPU-').lower() or gpu not in [u.removeprefix('GPU-').lower() for u in manifest['gpu_uuids']]:
        raise ValueError('GPU ownership differs')
    initialize()
    backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINT_SHA, 'metaworld', 'cuda:0', 'float32', allow_tf32=False)
    runtime = {'gpu_uuid': gpu, 'backend_provenance': backend.provenance,
               'torch_version': torch.__version__, 'tf32_matmul': False, 'tf32_cudnn': False}
    rows = [r for r in manifest['records'] if r['task'] == args.task and r['episode'] in args.episodes]
    if len(rows) != len(args.episodes) or len(set(args.episodes)) != len(args.episodes):
        raise ValueError('Missing or duplicate assigned work')
    if not args.engineering:
        receiving = read(args.receiving)
        if (receiving['passed'] is not True or receiving['manifest_sha256'] != args.manifest_sha256
                or gpu not in receiving['gpu_uuids']):
            raise ValueError('Representative receiving validation missing')
    for row in rows:
        if (row['role'] == 'excluded_engineering') != args.engineering:
            raise ValueError('Engineering/scientific assignment crossed')
        if time.time() >= args.deadline_epoch:
            raise TimeoutError('Registered execution deadline; preserve partial results')
        source = args.inputs/row['task']/row['tensor_file']
        if sha256(source) != row['inputs_sha256']:
            raise ValueError('Input tensor changed')
        values = torch.load(source, map_location='cpu', weights_only=True)
        if any(observation_digest(values[k]) != row[k+'_sha256'] for k in ('initial', 'goal')):
            raise ValueError('Delivered initial/goal differs')
        output = args.output/row['task']/f"episode-{row['episode']:03d}"
        # The original kernel's diagnostic label remains intact. The separate
        # receipt records this replication's population and exclusion status.
        with torch.no_grad():
            report = run_case(backend, planning_config(args.vendor, args.task)['config'],
                values, row, output, args.manifest_sha256, args.protocol_sha256, runtime)
        write_json(output/'REPLICATION_CASE.json', {'role': row['role'],
            'fresh_input_replication': True, 'protected_behavioral_panel': False,
            'manifest_sha256': args.manifest_sha256, 'protocol_sha256': args.protocol_sha256,
            'kernel_done_sha256': sha256(output/'DONE.json'), 'gpu_uuid': gpu,
            'complete': True, 'forward_count': report['total_forward_count']})
        print(json.dumps({'task': row['task'], 'episode': row['episode'],
              'role': row['role'], 'complete': True, 'seconds': report['total_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute'])
    for name in ('vendor', 'protocol', 'output', 'exposures', 'manifest', 'checkpoint', 'inputs', 'receiving'):
        parser.add_argument('--'+name, type=Path, required=name in ('vendor', 'protocol', 'output'))
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--task', choices=['reach', 'reach-wall'], required=True)
    parser.add_argument('--episodes', nargs='+', type=int)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--deadline-epoch', type=float, default=0)
    args = parser.parse_args()
    protocol = verify_contract(args)
    if args.mode == 'prepare':
        prepare_inputs(args, protocol)
    else:
        execute(args, protocol)


if __name__ == '__main__':
    main()
