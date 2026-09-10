"""Finish only already-authorized non-rank navigation comparisons after dispatch stop."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace/jepa-runtime')
PYTHON = '/workspace/jepa-planning-python/bin/python'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    # Only this stopped shell can dispatch the removed support categories.
    dispatcher = Path('/proc/2783/cmdline')
    if dispatcher.exists():
        if b'run_navigation_comparisons_v1.sh' not in dispatcher.read_bytes():
            raise ValueError('Prior dispatcher identity changed')
        status = Path('/proc/2783/status').read_text()
        if 'T (stopped)' not in status:
            raise ValueError('Old dispatcher must be stopped before replacing its queue')
    child = Path('/proc/3569/cmdline')
    while child.exists():
        # A stopped parent cannot reap its completed child. That zombie has an
        # empty cmdline but is terminal, not a new workload or identity change.
        stat = Path('/proc/3569/stat').read_text().split()
        if stat[2] == 'Z' and int(stat[3]) == 2783:
            break
        cmd = child.read_bytes()
        if b'offline_study.navigation_evaluate' not in cmd:
            raise ValueError('Previous child identity changed')
        time.sleep(3)
    if dispatcher.exists():
        os.kill(2783, signal.SIGKILL)  # stopped dispatcher only; completed child is gone
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
        raise ValueError('GPU is not free')
    os.environ.update(CUDA_VISIBLE_DEVICES='0', JEPA_VERIFIED_LOCAL_DINO='1',
        PYTHONPATH=str(ROOT/'navigation-evaluation-code-v1/src')+':/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/opt/conda/lib', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for category in ('vision_action_coupling', 'action_response_geometry'):
        for precision in ('bfloat16', 'float32'):
            for task, folder in (('wall', 'wall_single'), ('pointmaze', 'point_maze')):
                for index in range(8):
                    out = ROOT/f'navigation-comparisons-20260907-v1/{precision}/{task}/{category}/shard-{index}'
                    if (out/'DONE.json').exists():
                        done = json.loads((out/'DONE.json').read_text())
                        for name in ('report', 'window_metrics', 'mechanism_diagnostics', 'selection', 'protocol'):
                            if digest(out/(name+'.json')) != done[name+'_sha256']:
                                raise ValueError('Completed shard changed: '+str(out))
                        report = json.loads((out/'report.json').read_text())
                        if (report['task'], report['category'], report['precision'], report['shard_index'], report['shard_count']) != (task, category, precision, index, 8):
                            raise ValueError('Wrong completed scope')
                        continue
                    if out.exists():
                        raise ValueError('Preserve incomplete shard; no automatic overwrite: '+str(out))
                    print(json.dumps({'launch':str(out),'rank_support_categories_disabled_by_user':True}), flush=True)
                    subprocess.run([PYTHON, '-u', '-m', 'offline_study.navigation_evaluate',
                        '--vendor', '/workspace/jepa_steering/vendor/jepa-wms',
                        '--checkpoint', str(ROOT/f'navigation-assets-20260907-v1/downloads/model/jepa_wm_{task}.pth.tar'),
                        '--cohort', str(ROOT/f'navigation-offline-cohorts-20260907-v1/{task}/cohort.json'),
                        '--data-root', str(ROOT/f'navigation-assets-20260907-v1/extracted/{task}/{folder}'),
                        '--baseline', str(ROOT/f'navigation-offline-20260907-v1/{task}/{precision}'),
                        '--fit', str(ROOT/f'navigation-fits-20260907-v1/{precision}/{task}/{category}'),
                        '--precision', precision, '--shard-count', '8', '--shard-index', str(index),
                        '--output', str(out)], check=True)
    print(json.dumps({'status':'nonrank_navigation_queue_complete','frozen_scientific_protocols_unchanged':True}), flush=True)


if __name__ == '__main__':
    main()
