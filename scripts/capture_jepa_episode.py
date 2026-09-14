"""Capture one complete unsteered JEPA-WM episode for qualitative media.

Run on a qualified worker with the existing checkpoint and environment. This
does not rent hardware or add a result to the scientific evaluation panel.
The task and seed are explicit; a failed episode is retained as faithfully as
a successful one. Render the captured physics with render_jepa_episode.py.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np

PHYSICS = ('qpos', 'qvel', 'act', 'mocap_pos', 'mocap_quat', 'ctrl', 'qacc_warmstart')


def snapshot(env, observation=None):
    if observation is None:
        # MetaWorld's observation getter advances its frame-history cache.
        # Reading for media must not change the next observation seen by JEPA.
        previous = env._prev_obs.copy()
        try:
            observation = env._get_obs().copy()
        finally:
            env._prev_obs = previous
    return {'physics': {**{k: getattr(env.data, k).copy().tolist() for k in PHYSICS},
                        'time': float(env.data.time)},
            'state': np.asarray(observation, dtype=np.float32).tolist()}


@contextlib.contextmanager
def capture_unroll(evaluator_class, record):
    """Capture only agent execution, after expert goal construction finishes."""
    original = evaluator_class.unroll_agent

    def unroll(evaluator, env, *args, **kwargs):
        if record:
            raise ValueError('Only one complete episode may be captured')
        raw = env.proprio_env.unwrapped
        record.update(dt=float(raw.dt), rand_vec=np.asarray(raw._last_rand_vec).tolist(),
                      target=np.asarray(raw._target_pos).tolist(), frames=[snapshot(raw)],
                      actions=[], rewards=[], successes=[])
        step = raw.step
        def recorded_step(action):
            action_copy = np.asarray(action).copy().tolist()
            result = step(action)
            record['actions'].append(action_copy)
            record['frames'].append(snapshot(raw, result[0]))
            record['rewards'].append(float(result[1]))
            record['successes'].append(float(result[-1]['success']))
            return result
        raw.step = recorded_step
        try:
            return original(evaluator, env, *args, **kwargs)
        finally:
            raw.step = step

    evaluator_class.unroll_agent = unroll
    try:
        yield
    finally:
        evaluator_class.unroll_agent = original


def validate_capture(record, expected_steps=100):
    if len(record['actions']) != expected_steps or len(record['frames']) != expected_steps + 1:
        raise ValueError('Incomplete episode: retain diagnostics, do not publish a full-episode clip')
    times = np.array([f['physics']['time'] for f in record['frames']])
    if not np.allclose(np.diff(times), record['dt'], atol=1e-10, rtol=0):
        raise ValueError('Captured frame timing does not match simulator steps')
    if not np.isfinite(np.asarray(record['actions'])).all():
        raise ValueError('Nonfinite captured action')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vendor', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--task', choices=['reach', 'reach-wall'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    import torch
    from offline_study.backends import JepaBackend
    from offline_study.planning_contract import prepare
    from offline_study.planning_native_smoke import CHECKPOINTS, run_episode
    from offline_study.planning_scenarios import close_expert_environments
    if not torch.cuda.is_available():
        raise RuntimeError('Capture needs a qualified CUDA worker; no hardware is rented automatically')
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS['metaworld'],
                          'metaworld', 'cuda:0', 'float32', False)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    planning = prepare(args.vendor, args.task)
    cfg = OmegaConf.create(copy.deepcopy(planning['config']))
    cfg.local_seed = args.seed
    agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
    env = make_env(cfg)
    record = {}
    args.output.mkdir(parents=True)
    try:
        with close_expert_environments(plan_evaluator), capture_unroll(plan_evaluator.PlanEvaluator, record):
            result = run_episode(cfg, backend, agent, env, args.seed)
        validate_capture(record)
        receipt = {'role': 'new_qualitative_model_run_not_archived_evaluation_replay',
                   'task': args.task, 'seed': args.seed, 'arm': 'native',
                   'selection': 'Task and seed chosen before capture; retain entire episode regardless of outcome',
                   'checkpoint_sha256': CHECKPOINTS['metaworld'], 'planning': planning,
                   'precision': 'strict FP32, TF32 off', 'result': result, 'capture': record,
                   'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'versions': {p: importlib.metadata.version(p) for p in ['mujoco', 'metaworld', 'gymnasium', 'numpy']}}
        (args.output/'episode.json').write_text(json.dumps(receipt, indent=2)+'\n')
    finally:
        env.close()
        if not (args.output/'episode.json').exists():
            (args.output/'incomplete_capture.json').write_text(json.dumps(record)+'\n')


if __name__ == '__main__':
    main()
