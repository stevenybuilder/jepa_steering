"""Finish only timeout-margin leftovers from the two existing PointMaze queues.

Same frozen science and receiving proofs; never overlaps an occupied device,
reruns a complete stream, changes the deadline, or changes the sample schedule.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from table_resume_streams_v3 import command

ROOT = Path('/workspace/table-completion-20260911-v1')
REPAIR = ROOT/'pointmaze-runtime-repair-v1'
DEADLINE = 1789150635.690804


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2)


def complete(p):
    if not (p/'DONE.json').exists():
        return False
    assert sha(p/'report.json') == read(p/'DONE.json')['report_sha256']
    r = read(p/'report.json')
    assert r['episodes'] == 12 and r['parameters_unchanged'] and not r['fresh_confirmation']
    assert len(r['episode_files_sha256']) == 12
    for name, digest in r['episode_files_sha256'].items():
        assert Path(name).name == name and sha(p/name) == digest
    return True


def group(letter):
    original = ROOT/'expansion-20260911-v1'/('pointmaze-'+letter)
    out = ROOT/'expansion-20260911-v1'/('pointmaze-tail-'+letter)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = read(original/'PLAN.json')
        assert plan['task'] == 'pointmaze' and str(plan['instance']) == '50588893'
        write(out/'PLAN.json', {'original_plan_sha256': sha(original/'PLAN.json'),
            'reuse_receiving_proofs': True, 'deadline': DEADLINE,
            'per_stream_timeout_seconds': 1800, 'new_conditions': False,
            'reason': 'Original40-minute launch margin exceeds measured~21-minute stream time.'})
        while not any((original/n).exists() for n in ('QUEUE_DONE.json','QUEUE_FAILED.json')):
            if time.time()+60 > DEADLINE:
                raise RuntimeError('Original queue not terminal before guarded deadline')
            time.sleep(5)
        if (original/'QUEUE_FAILED.json').exists():
            failures = list(original.glob('GPU_*_FAILED.json'))
            assert failures and all(read(p)['error'] in (
                'Stop before intact job', 'Owned job interrupted; partial outputs retained') for p in failures), 'Unexpected scientific/runtime failure; do not retry'

        def worker(gpu):
            pending = [(arm,rank) for arm,rank in plan['queues'][str(gpu)]
                       if not complete(original/'conditions'/arm/f'shard-{rank}')]
            if not pending:
                return []
            physical = plan['physical_gpus'][gpu]
            while subprocess.check_output(['nvidia-smi','--id='+str(physical),
                    '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                if time.time()+1800 > DEADLINE:
                    raise RuntimeError('Device not free with sufficient margin')
                time.sleep(3)
            base, kwargs, source = command(ROOT, 'pointmaze')
            base[0] = str(REPAIR/'python/bin/python')
            engineering = original/'engineering'/f'gpu-{gpu}'
            assert sha(engineering/'report.json') == read(engineering/'DONE.json')['report_sha256']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical), PYTHONPATH=str(source),
                PYTHONDONTWRITEBYTECODE='1', JEPA_VERIFIED_LOCAL_DINO='1',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                SDL_VIDEODRIVER='dummy', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                MUJOCO_PY_FORCE_CPU='1', D4RL_SUPPRESS_IMPORT_ERROR='1',
                MUJOCO_PY_MUJOCO_PATH=str(REPAIR/'mujoco210'),
                LD_LIBRARY_PATH=str(REPAIR/'mujoco210/bin')+':/usr/lib/x86_64-linux-gnu:/opt/conda/lib')
            finished = []
            for arm, rank in pending:
                assert time.time()+1800 < DEADLINE, 'Insufficient margin for intact stream'
                target = out/'conditions'/arm/f'shard-{rank}'
                cmd = base+['run']+kwargs+['--engineering',str(engineering),'--arm',arm,
                    '--logical-ranks',str(rank),'--output',str(target)]
                with (out/f'gpu-{gpu}-{arm}-{rank}.log').open('x') as log:
                    p = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        assert p.wait(timeout=1800) == 0, 'Frozen science failed'
                    except BaseException:
                        if p.poll() is None:
                            os.killpg(p.pid,signal.SIGTERM)
                            try: p.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(p.pid,signal.SIGKILL);p.wait()
                        raise
                assert complete(target)
                finished.append([arm,rank])
            return finished

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker,range(2)))
        write(out/'QUEUE_DONE.json', {'completed': results, 'new_episodes': 12*sum(map(len,results)),
            'original_source_and_freeze_unchanged': True, 'full_study_complete': False})
    except Exception as e:
        write(out/'QUEUE_FAILED.json', {'error':str(e),'all_partial_evidence_retained':True})
        raise


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(group,('a','b')))
