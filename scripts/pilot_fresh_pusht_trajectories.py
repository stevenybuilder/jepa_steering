"""Excluded CPU engineering pilot, not an author-distribution or efficacy claim.

Frozen straight-push controller; retain every case. No model or old demonstrations.
Only four engineering seeds; scientific population generation is a separate decision.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/jepa-wms'
sys.path.insert(0, str(VENDOR))
OUT = ROOT / 'artifacts/offline_study/pusht-scripted-trajectory-engineering-20260912-v1'
SEEDS = (2026091250, 2026091251, 2026091252, 2026091253)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controller(state, direction):
    import numpy as np
    target = np.clip(state[2:4] + 90.0 * direction, 35.0, 477.0)
    delta = target - state[:2]
    return delta * min(1.0, 20.0 / max(float(np.linalg.norm(delta)), 1e-12)) / 100.0


def rollout(seed, actions=None):
    import numpy as np
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    env = PushTEnv(with_velocity=True, with_target=False, render_size=224, relative=True)
    try:
        env.seed(seed)
        obs, state = env.reset()
        direction = state[2:4] - state[:2]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        states, pixels, commands = [state.copy()], [obs['visual'].copy()], []
        for i in range(30):
            action = controller(state, direction) if actions is None else actions[i]
            obs, _, _, info = env.step(action)
            state = np.asarray(info['state']).copy()
            commands.append(np.asarray(action).copy())
            states.append(state); pixels.append(obs['visual'].copy())
        arrays = {'states': np.asarray(states), 'pixels': np.asarray(pixels), 'actions': np.asarray(commands)}
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Nonfinite input; preserve failure without replacement')
        return arrays
    finally:
        env.close()


def run():
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=False)
    env_source = VENDOR / 'evals/simu_env_planning/envs/pusht_env/pusht_env.py'
    protocol = {'role': 'excluded_input_engineering_not_scientific_evaluation', 'seeds': SEEDS,
                'source_sha256': sha(Path(__file__)), 'environment_sha256': sha(env_source),
                'controller': 'initial agent-to-object unit direction; target object+90px; bounded 20px commands',
                'steps': 30, 'goal': 'actual final state/frame', 'success_filtering': False,
                'packages': {k: importlib.metadata.version(k) for k in ('numpy', 'pymunk', 'pygame', 'gym')},
                'population_matches_author_demonstrations': False, 'scientific_launch_ready': False,
                'scientific_scenarios_generated': 0, 'model_calls': 0}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    started = time.monotonic(); records = []; failures = []
    for seed in SEEDS:
        try:
            a = rollout(seed)
            target = OUT / f'engineering-{seed}.npz'
            np.savez_compressed(target, **a)
            replay = rollout(seed, actions=a['actions'])
            exact = {key: bool(np.array_equal(a[key], replay[key])) for key in a}
            record = {'seed': seed, 'file': target.name, 'sha256': sha(target),
                      'shapes': {k: list(v.shape) for k, v in a.items()}, 'replay_exact': exact,
                      'block_displacement_pixels': float(np.linalg.norm(a['states'][-1, 2:4] - a['states'][0, 2:4])),
                      'initial_state': a['states'][0].tolist(), 'goal_state': a['states'][-1].tolist()}
            records.append(record)
            if not all(exact.values()):
                failures.append({'seed': seed, 'error': 'Exact replay mismatch'})
        except Exception as exc:
            failures.append({'seed': seed, 'error': repr(exc)})
        print(json.dumps({'cases_complete': len(records), 'failures': failures}), flush=True)
    result = {'protocol_sha256': sha(OUT / 'protocol.json'), 'cases': records, 'failures': failures,
              'all_engineering_cases_passed': len(records) == len(SEEDS) and not failures,
              'seconds': time.monotonic() - started, 'scientific_scenarios_generated': 0,
              'model_calls': 0, 'scientific_launch_ready': False,
              'remaining_gates': ['collection-policy decision', 'fresh source-family audit',
                                  'native evaluator reset/loader parity', 'frozen scientific manifest']}
    (OUT / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
