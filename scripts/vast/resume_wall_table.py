"""Finish only missing, frozen Wall streams on owned50561030; no refit/retry.

This is orchestration around the unchanged archived scientific module. All
complete old stream files are verified before deciding which streams are absent.
Two disjoint queues use two GPUs; aggregate sample counts remain96 per arm.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path('/workspace/table-completion-20260911-v1')
OUT = ROOT / 'wall-resume-20260911-v1'
FREEZE = 'd41381c52313e536e53ca80ccbf8324756bf665d248f436282622f9085fc7af3'
ARMS = ('visual_only','action_condition_only','joint','joint_equal_standardized_energy',
        'permuted_visual','permuted_joint','matched_random','matched_random_equal_standardized_energy')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2)


def inventory():
    found={}
    for path in sorted((ROOT/'restored').rglob('report.json')):
        if 'navigation' not in str(path) or 'conditions' not in str(path):continue
        try:
            report=read(path); launch=read(path.parent/'protocol.json'); done=read(path.parent/'DONE.json')
        except (OSError,ValueError):continue
        if report.get('task')!='wall' or launch.get('freeze_sha256')!=FREEZE:continue
        assert report['status']=='complete_coupling_navigation_shard'
        assert sha(path)==done['report_sha256']
        assert sha(path.parent/'protocol.json')==report['protocol_sha256']
        assert report['episodes']==12 and report['arm'] in ARMS
        assert len(launch['logical_ranks'])==1
        assert len(report['episode_files_sha256'])==12
        for name,digest in report['episode_files_sha256'].items():
            assert Path(name).name==name and sha(path.parent/name)==digest
        key=(report['arm'],launch['logical_ranks'][0])
        entry={'root':str(path.parent),'report_sha256':sha(path),'episodes':report['episode_files_sha256']}
        if key in found:
            assert found[key]['report_sha256']==entry['report_sha256'], 'Conflicting complete duplicate'
        else:found[key]=entry
    return found


def main():
    found=inventory()
    missing=[(a,r) for a in ARMS for r in range(8) if (a,r) not in found]
    assert set(missing)=={('permuted_visual',r) for r in range(1,8)} | {('permuted_joint',r) for r in range(3)}
    receiving=ROOT/'wall-receiving-engineering-v2'
    assert sha(receiving/'report.json')==read(receiving/'DONE.json')['report_sha256']
    assert read(receiving/'report.json')['status']=='complete_coupling_navigation_engineering'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
    OUT.mkdir(exist_ok=False)
    old=read(ROOT/'WALL_ENGINEERING_LAUNCH_V2.json')
    base=old['command'][:-2]
    assert base[4]=='engineer'
    base[4]='run'
    queues={gpu:missing[gpu::2] for gpu in (0,1)}
    plan={'instance':50561030,'task':'wall','unchanged_freeze_sha256':FREEZE,
          'receiving_report_sha256':sha(receiving/'report.json'),'missing_streams':missing,
          'queues':queues,'episodes_to_collect':12*len(missing),'source_sha256':sha(Path(__file__)),
          'old_streams':{a+':'+str(r):v for (a,r),v in found.items()},
          'sample_size_per_arm_total':96,'new_random_directions':False,'fresh_confirmation':False,
          'deadline_timestamp':1789128000.0,'per_stream_timeout_seconds':1800}
    write(OUT/'PLAN.json',plan)
    stop=threading.Event()
    def interrupted(signum,frame):stop.set()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def worker(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),JEPA_VERIFIED_LOCAL_DINO='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',
            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'restored/50231985/batch-001/jepa-runtime/navigation-coupling-code-20260908-v2/src'))
        completed=[]
        try:
            for arm,rank in queues[gpu]:
                if stop.is_set() or time.time()+1800>plan['deadline_timestamp']:
                    raise RuntimeError('Stop/deadline before another intact stream')
                output=OUT/'conditions'/arm/f'shard-{rank}'
                assert not output.exists()
                cmd=base+['--engineering',str(receiving),'--arm',arm,'--logical-ranks',str(rank),'--output',str(output)]
                with (OUT/f'gpu-{gpu}-{arm}-{rank}.log').open('x') as log:
                    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    print(json.dumps({'gpu':gpu,'arm':arm,'stream':rank,'pid':child.pid,'scientific_development':True}),flush=True)
                    start=time.monotonic()
                    while child.poll() is None:
                        if stop.is_set() or time.monotonic()-start>1800:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid,signal.SIGKILL);child.wait()
                            raise RuntimeError('Owned stream interrupted; partial records retained')
                        time.sleep(2)
                    assert child.returncode==0, 'Frozen scientific stream failed'
                report=read(output/'report.json')
                assert sha(output/'report.json')==read(output/'DONE.json')['report_sha256']
                assert report['episodes']==12 and report['arm']==arm
                for name,digest in report['episode_files_sha256'].items():assert sha(output/name)==digest
                completed.append((arm,rank))
            write(OUT/f'GPU_{gpu}_DONE.json',{'completed':completed})
        except Exception as exc:
            stop.set();write(OUT/f'GPU_{gpu}_FAILED.json',{'error':str(exc),'completed':completed});raise
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,(0,1)))
        write(OUT/'QUEUE_DONE.json',{'streams':len(missing),'episodes':12*len(missing),'full_six_task_study_complete':False})
    except Exception as exc:
        write(OUT/'QUEUE_FAILED.json',{'error':str(exc),'partial_records_retained':True})
        raise


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # A preflight failure must also wake the collector, not leave an idle
        # paid worker waiting for a terminal marker that can never arrive.
        OUT.mkdir(parents=True,exist_ok=True)
        if not (OUT/'QUEUE_FAILED.json').exists() and not (OUT/'QUEUE_DONE.json').exists():
            write(OUT/'QUEUE_FAILED.json',{'error':str(exc),'preflight_or_queue_failure':True})
        raise
