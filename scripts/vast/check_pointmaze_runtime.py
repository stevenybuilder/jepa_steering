"""CPU-only legacy runtime/reset/render parity against every old paired input."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path('/workspace/table-completion-20260911-v1')
REPAIR = ROOT / 'pointmaze-runtime-repair-v1'
SOURCE = ROOT / 'restored/50231985/batch-001/jepa-runtime/navigation-coupling-code-20260908-v2/src'
REFERENCE = ROOT / ('restored/50231985/batch-001/jepa-runtime/navigation-coupling-evidence-20260908-v2/'
                    'artifacts/offline_study/navigation-coupling-reference-20260908-v1/pointmaze')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from offline_study.vendor import use_vendor
    from offline_study.planning_contract import prepare
    from offline_study.planning_env_smoke import observation_digest
    from offline_study.planning_native_smoke import SMOKE_SEED
    from offline_study.behavioral_development import DEVELOPMENT_SEED
    from offline_study.protocol import sha256, write_json
    use_vendor(ROOT/'code/vendor/jepa-wms')
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    installed = {n: importlib.metadata.version(n) for n in
                 ('torch', 'numpy', 'gym', 'mujoco', 'mujoco-py', 'd4rl')}
    old = json.loads((REFERENCE/'smoke/report.json').read_text())
    assert installed == old['installed_versions'], (installed, old['installed_versions'])
    records = [(SMOKE_SEED, json.loads((REFERENCE/'smoke/repetition-0.json').read_text()))]
    for p in sorted((REFERENCE/'native').rglob('episode-*.json')):
        row = json.loads(p.read_text()); records.append((row['environment_seed'], row))
    assert len(records) == 97
    checks = []
    for seed, original in records:
        # Environment construction has global-RNG effects (including rendering).
        # An episode seed is NOT the seed used to construct the native runtime.
        construction_seed = SMOKE_SEED if seed == SMOKE_SEED else DEVELOPMENT_SEED
        random.seed(construction_seed); np.random.seed(construction_seed); torch.manual_seed(construction_seed)
        cfg = OmegaConf.create(prepare(ROOT/'code/vendor/jepa-wms', 'pointmaze')['config'])
        cfg.meta.seed = construction_seed
        cfg.local_seed = original.get('local_seed', SMOKE_SEED)
        env = make_env(cfg)
        try:
            env.reset(seed=seed, task_idx=0)
            env.proprio_env.unwrapped._freeze_rand_vec = False
            env.proprio_env.unwrapped.seeded_rand_vec = True
            env.seed(seed)
            initial, goal, _, _ = PlanEvaluator(cfg, None).set_episode(cfg, None, env, seed, task_idx=0)
            actual = {'initial_sha256': observation_digest(initial), 'goal_sha256': observation_digest(goal)}
            assert all(actual[k] == original['result'][k] for k in actual), (seed, actual, original['result'])
            checks.append({'environment_seed': seed, 'construction_seed': construction_seed, **actual})
        finally:
            env.close()
    report = {'status': 'all_97_original_pointmaze_inputs_exact_cpu_only', 'gpu_calls': 0,
              'scientific_episodes': 0, 'receiving_planner_still_required': True,
              'installed_versions': installed, 'checks': checks,
              'd4rl_commit': subprocess.check_output(['git', '-C', str(REPAIR/'D4RL'), 'rev-parse', 'HEAD'], text=True).strip(),
              'mujoco_archive_sha256': sha256(REPAIR/'mujoco210-linux-x86_64.tar.gz'),
              'check_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(REPAIR/'report.json', report)
    write_json(REPAIR/'READY.json', {'report_sha256': sha256(REPAIR/'report.json'), 'not_planner_clearance': True})
    print(json.dumps({'status': report['status'], 'checks': len(checks)}))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        with (REPAIR/f'CHECK_FAILED-{time.time_ns()}.json').open('x') as f:
            json.dump({'error_type': type(exc).__name__, 'detail': str(exc)[:1500], 'gpu_calls': 0}, f)
        raise
