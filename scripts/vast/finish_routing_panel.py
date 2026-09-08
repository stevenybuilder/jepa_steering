"""CPU-only coordinator for the fixed32-shard HMM successor comparison."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/jepa-runtime')
PANEL=ROOT/'hmm-fixed-response-behavior-20260908-v3'


def main():
    code=ROOT/'hmm-fixed-response-code-20260908-v3'
    source=ROOT/'run_routing_queue_v3.py'
    spec=importlib.util.spec_from_file_location('queue',source);queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    started=time.monotonic();previous=None
    while True:
        statuses=[PANEL/f'queue-gpu{i}' for i in range(8)]
        if any((p/'FAILED.json').exists() for p in statuses):raise ValueError('Required routed queue failed; no partial analysis')
        complete=sum((p/'DONE.json').exists() for p in statuses)
        if complete!=previous:print(json.dumps({'completed_gpu_assignments':complete,'required':8,'partial_selection':False}),flush=True);previous=complete
        if complete==8:break
        if time.monotonic()-started>3*86400:raise TimeoutError('Coordinator wait only; scientific jobs are not stopped')
        time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        LD_LIBRARY_PATH='/opt/conda/lib',PYTHONPATH=str(code/'src')+':/workspace/jepa-python/lib/python3.10/site-packages')
    subprocess.run(['/workspace/jepa-planning-python/bin/python','-u','-m','offline_study.routing_analysis']+
        queue.arguments()+['--panel',str(PANEL),'--output',str(PANEL/'analysis')],env=env,check=True,timeout=3600)
    print(json.dumps({'status':'full_hmm_development_panel_analyzed','confirmation_launched':False,'full_study_complete':False}),flush=True)


if __name__=='__main__':main()
