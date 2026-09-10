"""Owned California GPU0: receiving validation, fixed engineering, paired panel.

Every shard is an unchanged complete12-episode stream. No partial retries,
outcome selection, independent GPU sharing or confirmation reveal.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/jepa-runtime')
CODE = ROOT / 'pusht-coupling-code-20260908-v2'
PANEL = ROOT / 'pusht-coupling-behavior-20260908-v2'
PYTHON = '/workspace/jepa-planning-python/bin/python'
SOURCE = '42cb7df90b82a71996687ac719c22a0ab9e4fd4f2dccf4ebd19c6847a841a27c'
FREEZE = '2ddf2e1457a093ac48288842fd6ea81bb021a276317993fa22d570cb809a04d0'
DEVICE = 'GPU-6c6aa1cb-e5f6-cede-089f-b76c4bc8638c'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.check_only else '0',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', JEPA_VERIFIED_LOCAL_DINO='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(CODE/'src')+':/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/opt/conda/lib')
    sys.path.insert(0, str(CODE/'src'))
    from offline_study.author_fit import source_hash
    from offline_study.protocol import sha256, write_json
    from offline_study.pusht_coupling_behavior import read_contract, load_native, ARMS
    from offline_study.behavioral_development import verified_report
    evidence = ROOT/'pusht-coupling-evidence-20260908-v2'
    kwargs = dict(vendor=Path('/workspace/jepa_steering/vendor/jepa-wms'),
        reference=ROOT/'pusht-planning-native-20260908-v1',
        reference_code=ROOT/'pusht-planning-code-20260908-v1',
        fit=evidence/'fits/vision_action_coupling', cohort=evidence/'cohort/cohort.json',
        data_root=ROOT/'pusht-planning-assets-20260908-v3/source/data/pusht_noise',
        freeze=PANEL/'freeze',
        checkpoint=ROOT/'pusht-planning-assets-20260908-v2/downloads/models/jepa_wm_pusht.pth.tar')
    if source_hash()!=SOURCE or sha256(kwargs['freeze']/'protocol.json')!=FREEZE:
        raise ValueError('Immutable source/freeze changed')
    protocol, _, cohort, native, _ = read_contract(argparse.Namespace(**kwargs))
    if sha256(kwargs['checkpoint']) != protocol['bindings']['checkpoint_sha256']:
        raise ValueError('Released checkpoint changed')
    # Verify every transferred source and fit member against the original sender.
    members=json.loads((PANEL/'PREPARATION_FILES.json').read_text())
    for name, expected in members.items():
        path=ROOT/name
        if path.is_symlink() or path.stat().st_size!=expected['bytes'] or sha256(path)!=expected['sha256']:
            raise ValueError('Changed receiving member: '+name)
    import cv2, pymunk, pygame
    from evals.simu_env_planning.envs.init import make_env
    for rank in range(3):
        load_native(kwargs['reference'],native,cohort,[rank])
    devices=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if devices!=DEVICE:
        raise ValueError('Different receiving GPU')
    if args.check_only:
        print(json.dumps({'status':'receiving_source_fit_inputs_checkpoint_and_three_native_streams_verified',
            'source_sha256':SOURCE,'freeze_sha256':FREEZE,'members':len(members),
            'device_uuid':devices,'gpu_job_launched':False}),flush=True)
        return
    if subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader']).strip():
        raise ValueError('Reserved GPU0 is not empty')
    launch=PANEL/'QUEUE_LAUNCH.json'
    with launch.open('x') as stream:
        json.dump({'pid':os.getpid(),'instance':50245262,'device_uuid':DEVICE,
            'source_sha256':SOURCE,'freeze_sha256':FREEZE,'episodes_per_condition_total':96,
            'native_reference_streams_verified_before_use':True,'fresh_confirmation':False},stream,indent=2)
    common=[]
    for name,value in kwargs.items():
        common+=['--'+name.replace('_','-'),str(value)]
    logs=PANEL/'queue-logs'; logs.mkdir(exist_ok=False)
    child=None
    def interrupt(signum, frame):
        raise RuntimeError('Queue interrupted; preserve partial evidence')
    signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
    def execute(command, extra, label, cap):
        nonlocal child
        print(json.dumps({'stage':label,'gpu':0}),flush=True)
        with (logs/(label+'.log')).open('x') as log:
            child=subprocess.Popen([PYTHON,'-u','-m','offline_study.pusht_coupling_behavior',command]+common+extra,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(timeout=cap):
                raise ValueError('Required '+label+' failed; no retry or advancement')
        child=None
    try:
        engineering=PANEL/'engineering'/DEVICE
        execute('engineer',['--output',str(engineering)],'engineering',4*3600)
        verified_report(engineering)
        # Rank-first lets completed reference streams release all fixed arms
        # while the independent native producer finishes later streams.
        for rank in range(8):
            stream=kwargs['reference']/'native'/f'shard-{rank}'
            deadline=time.monotonic()+8*3600
            while not (stream/'DONE.json').exists():
                if time.monotonic()>deadline:
                    raise TimeoutError('Missing complete reference stream; no scientific retry')
                write_json(PANEL/'QUEUE_WAIT.json',{'reference_rank':rank,'gpu_work_active':False})
                time.sleep(30)
            load_native(kwargs['reference'],native,cohort,[rank])
            for arm in ARMS[1:]:
                target=PANEL/'conditions'/arm/f'shard-{rank}'
                execute('run',['--engineering',str(engineering),'--arm',arm,'--logical-ranks',str(rank),
                    '--output',str(target)],arm+f'-shard-{rank}',4*3600)
                verified_report(target)
        os.environ['CUDA_VISIBLE_DEVICES']=''
        execute('analyze',['--panel',str(PANEL),'--output',str(PANEL/'analysis')],'analysis',1800)
        report,digest=verified_report(PANEL/'analysis')
        write_json(PANEL/'QUEUE_DONE.json',{'status':'complete_pusht_static_coupling_development_panel',
            'analysis_report_sha256':digest,'fresh_confirmation':False,'full_study_complete':False})
    except Exception as exc:
        write_json(PANEL/'QUEUE_FAILED.json',{'error':str(exc),'partial_not_complete':True})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':
    main()
