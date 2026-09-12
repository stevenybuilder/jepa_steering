"""Prepare independent task runtimes on CPU, freeze verified fits, then queue GPU work."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PYTHON = '/workspace/component-python/bin/python'
COMPONENT = Path('/workspace/metaworld-components-20260911-v1')


def prepare_pointmaze(root, env):
    runtime = root / 'runtime'
    retry = root.name in ('refined-pointmaze-behavior-20260911-v2','refined-pointmaze-behavior-20260911-v3')
    old = root.with_name('refined-pointmaze-behavior-20260911-v2' if root.name.endswith('-v3') else 'refined-pointmaze-behavior-20260911-v1')
    if retry:
        if not (old / 'PREPARATION_FAILED.json').exists():
            raise ValueError('Retry requires preserved original CPU preparation failure')
        shutil.copytree(old / 'runtime', runtime, symlinks=True)
    else:
        runtime.mkdir()
    def run(command, timeout=600):
        with (runtime / 'installation.log').open('a') as log:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True, timeout=timeout)
    run(['apt-get', '-o', 'Acquire::ForceIPv4=true', 'install', '-y', '--no-upgrade', '--no-install-recommends',
         'build-essential', 'libosmesa6-dev', 'patchelf'])
    if not retry:run([PYTHON, '-m', 'venv', '--system-site-packages', str(runtime / 'python')])
    python = str(runtime / 'python/bin/python')
    site = subprocess.check_output([python, '-c', 'import sysconfig;print(sysconfig.get_path("purelib"))'], text=True).strip()
    parent = subprocess.check_output([PYTHON, '-c', 'import sysconfig;print(sysconfig.get_path("purelib"))'], text=True).strip()
    (Path(site) / 'component_parent.pth').write_text(parent + '\n')
    run([python, '-m', 'pip', 'install', '--no-cache-dir', '--index-url', 'https://pypi.org/simple',
         'cython==0.29.37', 'mujoco-py==2.1.2.14', 'gym==0.23.1', 'numpy==1.26.4', 'fasteners==0.20', 'patchelf==0.17.2.4',
         'opencv-python-headless==4.11.0.86'])
    if not (runtime / 'D4RL').exists():run(['git', 'clone', '--no-checkout', 'https://github.com/Farama-Foundation/D4RL.git', str(runtime / 'D4RL')])
    run(['git', '-C', str(runtime / 'D4RL'), 'checkout', '--detach', '89141a689b0353b0dac3da5cba60da4b1b16254d'])
    # The historical verified runtime used Python3.11 with this exact source.
    # Its packaging upper bound is stale; retain source and verify all97 resets.
    run([python, '-m', 'pip', 'install', '--no-deps', '--ignore-requires-python', '--index-url', 'https://pypi.org/simple', str(runtime / 'D4RL')])
    run(['curl', '--fail', '--location', '--retry', '2', '--max-time', '300',
         'https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz', '-o', str(runtime / 'mujoco.tar.gz')])
    run(['tar', '-xzf', str(runtime / 'mujoco.tar.gz'), '-C', str(runtime)])
    env.update(MUJOCO_PY_FORCE_CPU='1', D4RL_SUPPRESS_IMPORT_ERROR='1',
        MUJOCO_PY_MUJOCO_PATH=str(runtime / 'mujoco210'),
        LD_LIBRARY_PATH=str(runtime / 'mujoco210/bin') + ':/usr/lib/x86_64-linux-gnu:/opt/conda/lib')
    run([python, '-c', 'import mujoco_py;from d4rl import offline_env;import torch;assert not torch.cuda.is_initialized()'])
    return python


def reset_check(root, spec, python, env):
    code = '''import json,random,sys,importlib.metadata as m
from pathlib import Path
import numpy as np,torch
from omegaconf import OmegaConf
from offline_study.vendor import use_vendor
from offline_study.planning_contract import prepare
from offline_study.planning_env_smoke import observation_digest
from offline_study.planning_native_smoke import SMOKE_SEED
from offline_study.behavioral_development import DEVELOPMENT_SEED
from offline_study.protocol import write_json
r=Path(sys.argv[1]);s=json.loads((r/'PANEL.json').read_text());reference=r/'reference'
old=json.loads((reference/'smoke/report.json').read_text())['installed_versions']
actual={n:m.version(n) if old[n] is not None else None for n in old}
assert actual==old,(actual,old)
use_vendor(Path(s['vendor']))
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
records=[(SMOKE_SEED,json.loads((reference/'smoke/repetition-0.json').read_text()))]
records += [(q['environment_seed'],q) for p in sorted((reference/'native').rglob('episode-*.json')) for q in [json.loads(p.read_text())]]
assert len(records)==97
for seed,row in records:
 construction=SMOKE_SEED if seed==SMOKE_SEED else DEVELOPMENT_SEED
 random.seed(construction);np.random.seed(construction);torch.manual_seed(construction)
 cfg=OmegaConf.create(prepare(Path(s['vendor']),s['task'])['config']);cfg.meta.seed=construction;cfg.local_seed=row.get('local_seed',SMOKE_SEED)
 env=make_env(cfg)
 try:
  env.reset(seed=seed,task_idx=0);env.proprio_env.unwrapped._freeze_rand_vec=False;env.proprio_env.unwrapped.seeded_rand_vec=True;env.seed(seed)
  initial,goal,_,_=PlanEvaluator(cfg,None).set_episode(cfg,None,env,seed,task_idx=0)
  assert observation_digest(initial)==row['result']['initial_sha256']
  assert observation_digest(goal)==row['result']['goal_sha256']
 finally:env.close()
assert not torch.cuda.is_initialized()
write_json(r/'RESET_PARITY.json',{'all97_original_inputs_exact':True,'versions':actual,'model_calls':0})
'''
    with (root / 'reset-check.log').open('x') as log:
        subprocess.run([python, '-c', code, str(root)], env=env, stdout=log, stderr=log, check=True, timeout=600)


def main(root):
    sys.path.insert(0, str(root / 'code/src'))
    sys.path.insert(0, str(root / 'ops'))
    from offline_study.protocol import sha256, write_json
    from component_queue_handoff import process
    from component_boundary_job import validate_queue
    spec = json.loads((root / 'PANEL.json').read_text())
    task = spec['task']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root / 'code/src'),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', SDL_VIDEODRIVER='dummy', DEBIAN_FRONTEND='noninteractive',
        PIP_INDEX_URL='https://pypi.org/simple', PIP_EXTRA_INDEX_URL='')
    try:
        python = PYTHON
        if task == 'pointmaze': python = prepare_pointmaze(root, env)
        if task in ('pointmaze', 'wall'): reset_check(root, spec, python, env)
        if task == 'droid':
            old = Path('/workspace/refined-droid-prerequisite-20260911-v1')
            canonical = Path(spec['assets'])
            canonical.mkdir(parents=True, exist_ok=True)
            for name in ('protocol.json', 'report.json', 'DONE.json'):
                shutil.copy2(root / 'asset-receipt' / name, canonical / name)
            # Hard links preserve exact historical path strings without extra data copies.
            for p in (old / 'assets/downloads').rglob('*'):
                if p.is_file() and '.cache' not in p.parts:
                    target = canonical / 'downloads' / p.relative_to(old / 'assets/downloads')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists(): os.link(p, target)
            shutil.copytree('/workspace/refined-droid-prerequisite-20260911-v2/runtime-packages', root / 'runtime-packages',
                            ignore=shutil.ignore_patterns('__pycache__'))
            env['PYTHONPATH'] += os.pathsep + str(root / 'runtime-packages')
        write_json(root / 'RUNTIME_READY.json', {'time': time.time(), 'task': task, 'python': python, 'gpu_calls': 0})
        prerequisite = Path(spec['prerequisite'])
        deadline = time.time() + 7200
        while not (prerequisite / 'TERMINAL.json').exists():
            if time.time() >= deadline: raise TimeoutError('Fit prerequisite did not finish')
            time.sleep(3)
        if not (prerequisite / 'fit-v1/DONE.json').exists() or (prerequisite / 'fit-v1/FAILED.json').exists():
            raise ValueError('Prerequisite failed; no behavioral launch')
        shutil.copytree(prerequisite / 'fit-v1', root / 'fit')
        from refined_panel_queue import jobs
        first = jobs(root, spec)[0][2]
        command = first[:];command[4] = 'freeze'
        if task != 'droid': command[command.index('--output') + 1] = str(root / 'freeze')
        with (root / 'freeze.log').open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True, timeout=300)
        write_json(root / 'INPUTS_READY.json', {'time': time.time(), 'task': task, 'gpu_calls': 0})
        files = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                 for name in ('code','ops','fit','freeze','reference','reference-source','cohort','native-engineering','runtime-packages')
                 for p in (root / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        files['PANEL.json'] = {'bytes': (root / 'PANEL.json').stat().st_size, 'sha256': sha256(root / 'PANEL.json')}
        job = {'kind':'refined_panel','task':task,'timeout_seconds':21600,'cwd':str(root/'code'),
               'gpu_uuid':spec['physical_gpu_uuid'],'files':files,
               'command':[PYTHON,'-u',str(root/'ops/refined_panel_queue.py'),'--root',str(root)]}
        if task == 'droid':job['pythonpath_additions']=[str(root/'runtime-packages')]
        write_json(root / 'JOB.json', job)
        while time.time() < deadline:
            if not (prerequisite / 'COMPONENTS_RESUMED.json').exists():
                time.sleep(1);continue
            queues=[]
            for p in Path('/proc').iterdir():
                if not p.name.isdigit(): continue
                try:validate_queue(process(int(p.name)), COMPONENT)
                except (ValueError,IndexError):continue
                queues.append(int(p.name))
            if len(queues)==1:break
            time.sleep(1)
        else:raise TimeoutError('No available owned coordinator after fit')
        with (root/'boundary.log').open('x') as log:
            child=subprocess.Popen([PYTHON,'-u',str(root/'ops/component_boundary_job.py'),'--job',str(root/'JOB.json'),
                '--component-root',str(COMPONENT),'--queue-pid',str(queues[0])],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        write_json(root/'BOUNDARY_LAUNCH.json',{'time':time.time(),'pid':child.pid})
    except Exception as error:
        write_json(root/'PREPARATION_FAILED.json',{'error_type':type(error).__name__,'time':time.time()})
        raise


if __name__ == '__main__': main(Path(sys.argv[1]))
