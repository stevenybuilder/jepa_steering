"""One owned MetaWorld GPU: await existing five-arm stream, then fixed HMM jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/workspace/jepa-runtime')
PYTHON='/workspace/jepa-planning-python/bin/python'
CODE=ROOT/'hmm-fixed-response-code-20260908-v3'
PANEL=ROOT/'hmm-fixed-response-behavior-20260908-v3'


def arguments():
    evidence=ROOT/'fixed-response-behavior-evidence-20260908-v1/artifacts/offline_study'
    return ['--vendor','/workspace/jepa_steering/vendor/jepa-wms',
        '--fits',str(evidence/'fixed-response-20260908-v1/fits'),
        '--routing',str(ROOT/'hmm-fixed-response-evidence-20260908-v3/fits'),
        '--stimuli',str(evidence/'restored-behavioral-inputs-20260908-v1'),
        '--reference',str(ROOT/'fixed-response-behavior-20260908-v1'),
        '--reference-freeze',str(evidence/'fixed-response-20260908-v1/behavioral-freeze-v1'),
        '--reference-code',str(ROOT/'fixed-response-code-20260908-v4/src/offline_study'),
        '--reference-workers',str(ROOT/'fixed-response-worker-checks-20260908-v1'),
        '--freeze',str(PANEL/'freeze')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=range(8),required=True)
    args=parser.parse_args();gpu=args.gpu;task='reach' if gpu<4 else 'reach-wall';rank=gpu%4
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',JEPA_VERIFIED_LOCAL_DINO='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/opt/conda/lib',
        PYTHONPATH=str(CODE/'src')+':/workspace/jepa-python/lib/python3.10/site-packages')
    sys.path.insert(0,str(CODE/'src'))
    from offline_study.protocol import sha256,write_json
    from offline_study.author_fit import source_hash
    from offline_study.behavioral_development import verified_report
    from offline_study.fixed_response_behavior import device_uuid
    from offline_study.routing_behavior import NEW_ARMS
    freeze=json.loads((PANEL/'freeze/protocol.json').read_text())
    if source_hash()!=freeze['source_sha256'] or sha256(PANEL/'freeze/protocol.json')!=json.loads((PANEL/'freeze/FROZEN.json').read_text())['protocol_sha256']:
        raise ValueError('Routed frozen source changed')
    status=PANEL/f'queue-gpu{gpu}';status.mkdir(exist_ok=False)
    write_json(status/'LAUNCH.json',{'pid':os.getpid(),'predecessor_pid':5202+gpu,'gpu':gpu,'task':task,
        'logical_ranks':[rank,rank+4],'source_sha256':source_hash(),'freeze_sha256':sha256(PANEL/'freeze/protocol.json'),
        'no_independent_gpu_overlap':True,'fresh_confirmation':False})
    predecessor=5202+gpu
    def alive(pid):
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    child=None
    def interrupted(signum,frame):raise RuntimeError('Queue interrupted; preserve partial results')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def execute(module,extra,label,cap):
        nonlocal child
        print(json.dumps({'stage':label,'gpu':gpu,'task':task}),flush=True)
        with (status/(label+'.log')).open('x') as log:
            child=subprocess.Popen([PYTHON,'-u','-m',module]+extra,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(timeout=cap):raise ValueError('Required '+label+' failed; no automatic retry')
        child=None
    try:
        deadline=time.monotonic()+24*3600
        while alive(predecessor):
            if time.monotonic()>deadline:raise TimeoutError('Original five-arm producer not complete')
            cmd=Path(f'/proc/{predecessor}/cmdline').read_bytes()
            if b'run_fixed_behavior_queue' not in cmd:raise ValueError('Predecessor PID was reused')
            time.sleep(30)
        bindings={}
        for arm in ('native','fixed_rank4','matched_random_fixed_rank4','coupling_only','matched_random_coupling'):
            root=ROOT/f'fixed-response-behavior-20260908-v1/{task}/{arm}/shard-gpu{gpu}'
            report,digest=verified_report(root)
            if (root/'FAILED.json').exists() or report['episodes']!=24 or report['status']!='fixed_response_behavioral_shard_complete':
                raise ValueError('Original assigned panel incomplete; no GPU handoff')
            bindings[arm]=digest
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader']).strip():
            raise ValueError('GPU remains occupied after original queue completion')
        write_json(status/'PREDECESSOR_VERIFIED.json',bindings)
        worker=device_uuid();engineering=PANEL/'engineering'/task/worker
        common=arguments()+['--checkpoint',str(ROOT/'fixed-response-assets-20260908-v1/jepa_wm_metaworld.pth.tar'),
            '--task',task,'--worker-reference',str(ROOT/f'fixed-response-worker-checks-20260908-v1/gpu-{gpu}')]
        execute('offline_study.routing_behavior',['engineer']+common+['--output',str(engineering)],'engineering',4*3600)
        verified_report(engineering)
        for arm in NEW_ARMS:
            target=PANEL/task/arm/f'shard-gpu{gpu}'
            execute('offline_study.routing_behavior',['run']+common+['--engineering',str(engineering),'--arm',arm,
                '--logical-ranks',str(rank),str(rank+4),'--output',str(target)],arm,8*3600)
            verified_report(target)
        write_json(status/'DONE.json',{'status':'assigned_routed_streams_complete','new_episodes':96,
            'full_panel_analysis_complete':False,'fresh_confirmation':False})
    except Exception as exc:
        write_json(status/'FAILED.json',{'error':str(exc),'partial_not_complete':True});raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
