"""Explicit bootstrap-to-science handoff, with terminal markers on every failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/table-completion-20260911-v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--instance',type=int,required=True)
    p.add_argument('--gpus',type=int,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--gpu-offset',type=int,required=True);p.add_argument('--total-gpus',type=int,required=True)
    a=p.parse_args();jobs=[]
    try:
        with (ROOT/'expansion-bootstrap.log').open('x') as f:
            jobs.append(subprocess.Popen(['bash',str(ROOT/'code/scripts/vast/bootstrap_table_completion.sh')],stdout=f,stderr=subprocess.STDOUT))
        if a.task=='pusht':
            with (ROOT/'expansion-pusht-inputs.log').open('x') as f:
                jobs.append(subprocess.Popen(['python3','-u',str(ROOT/'code/scripts/vast/table_pusht_inputs.py')],stdout=f,stderr=subprocess.STDOUT))
        for child in jobs:
            if child.wait(timeout=1800):raise RuntimeError('Receiving bootstrap/input job failed')
        cmd=['/workspace/table-python-inherited/bin/python','-u',str(ROOT/'code/scripts/vast/table_resume_streams.py')]+sys.argv[1:]
        with (ROOT/'expansion-queue.log').open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);jobs.append(child)
            if child.wait(timeout=max(1,a.deadline-time.time())):raise RuntimeError('Receiving/scientific queue failed')
        with (ROOT/'EXPANSION_DONE.json').open('x') as f:json.dump({'queue_terminal':True,'full_study_complete':False},f)
    except Exception as e:
        for child in jobs:
            if child.poll() is None:child.terminate()
        with (ROOT/'EXPANSION_FAILED.json').open('x') as f:json.dump({'error':str(e),'partial_outputs_retained':True},f)
        raise


if __name__=='__main__':main()
