"""Bounded remote bootstrap/queue supervisor; records every terminal exit."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--logical-ranks',nargs='+',required=True)
    args=p.parse_args()
    root=Path('/workspace/confirmation')
    status={'task':args.task,'deadline':args.deadline,'stage':'bootstrap','returncode':None}
    try:
        with Path('/workspace/confirmation-bootstrap.log').open('x') as log:
            subprocess.run(['bash',str(root/'scripts/vast/bootstrap_confirmation.sh')],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        status['stage']='receiving_engineering_then_confirmation'
        with Path('/workspace/confirmation-run.log').open('x') as log:
            subprocess.run(['/workspace/decision-python/bin/python','-u',str(root/'scripts/vast/run_confirmation_worker.py'),
                '--task',args.task,'--deadline',str(args.deadline),'--logical-ranks',*args.logical_ranks],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,args.deadline-time.time()))
        status['returncode']=0
    except Exception as error:
        status.update(returncode=1,error=str(error))
        raise
    finally:
        Path('/workspace/confirmation-launch-exit.json').write_text(json.dumps(status))


if __name__=='__main__':main()
