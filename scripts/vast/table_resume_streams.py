"""Disjoint whole-stream continuation around immutable archived science.

Per-GPU engineering must finish before that GPU starts its fixed queue.
No rerun of a complete, checksum-verified stream; no treatment selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time

ARMS=('visual_only','action_condition_only','joint','joint_equal_standardized_energy',
      'permuted_visual','permuted_joint','matched_random','matched_random_equal_standardized_energy')
FREEZES={'wall':'d41381c52313e536e53ca80ccbf8324756bf665d248f436282622f9085fc7af3',
         'pointmaze':'2d94a21b4554670dfb289baf7018291be7764b01a00be093a98ac19a3afe1fbf',
         'pusht':'2ddf2e1457a093ac48288842fd6ea81bb021a276317993fa22d570cb809a04d0'}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)


def inventory(root,task):
    found={}; incomplete=[]
    for path in sorted((root/'restored').rglob('report.json')):
        if '/conditions/' not in str(path):continue
        if ('/'+task+'/' if task in ('pointmaze','wall') else '/pusht-coupling-behavior-20260908-v2/') not in str(path):continue
        report=read(path)
        if report.get('freeze_sha256')!=FREEZES[task]:continue
        # Some preservation batches split one directory: its report is in
        # batch000 and later episode members in batch001. Rejoin only exact
        # hash-bound members; never call that an unrun stream or select outcomes.
        relative=path.parent.relative_to(next(p for p in path.parents if p.name=='jepa-runtime'))
        candidates=[p/relative for p in (root/'restored').glob('*/*/jepa-runtime')]
        def locate(name,digest):
            matches=[p/name for p in candidates if (p/name).is_file() and sha(p/name)==digest]
            return matches[0] if matches else None
        protocol_path=locate('protocol.json',report['protocol_sha256'])
        dones=[p/'DONE.json' for p in candidates if (p/'DONE.json').is_file() and
               read(p/'DONE.json').get('report_sha256')==sha(path)]
        members={n:locate(n,h) for n,h in report.get('episode_files_sha256',{}).items()}
        if protocol_path is None or not dones or any(p is None for p in members.values()):
            incomplete.append(str(path));continue
        launch=read(protocol_path)
        assert report['episodes']==12 and len(report['episode_files_sha256'])==12
        assert launch['freeze_sha256']==FREEZES[task] and len(launch['logical_ranks'])==1
        assert report['arm'] in ARMS and report['parameters_unchanged'] and not report['fresh_confirmation']
        assert report['status']==('complete_coupling_navigation_shard' if task in ('pointmaze','wall') else 'complete_pusht_coupling_shard')
        for name,digest in report['episode_files_sha256'].items():
            assert Path(name).name==name and sha(members[name])==digest
        key=(report['arm'],launch['logical_ranks'][0])
        target=root/'reconciled'/task/'conditions'/report['arm']/f'shard-{key[1]}'
        target.mkdir(parents=True,exist_ok=True)
        for name,p in {**members,'report.json':path,'protocol.json':protocol_path,'DONE.json':dones[0]}.items():
            q=target/name
            if q.exists():assert sha(q)==sha(p),'Conflicting reconstructed stream'
            else:shutil.copyfile(p,q)
        value={'root':str(target),'report_sha256':sha(path)}
        if key in found:assert found[key]['report_sha256']==value['report_sha256'],'Conflicting complete duplicate'
        else:found[key]=value
    return found,incomplete


def command(root,task):
    n=root/'restored/50231985/batch-001/jepa-runtime'
    c=root/'restored/50245262/batch-000/jepa-runtime'
    nj=root/'restored/50239185/batch-000/jepa-runtime'
    if task=='pointmaze':
        evidence=n/'navigation-coupling-evidence-20260908-v2/artifacts/offline_study'
        fits=evidence/'primary-durable-20260907/navigation-fits-20260907-v1/bfloat16/pointmaze'
        kwargs={'vendor':root/'code/vendor/jepa-wms',
            'reference':evidence/'navigation-coupling-reference-20260908-v1',
            'fit':fits/'vision_action_coupling','cohort':fits/'cohort.json',
            'freeze':evidence/'navigation-coupling-behavior-20260908-v1/pointmaze/freeze',
            'checkpoint':root/'checkpoints/jepa_wm_pointmaze.pth.tar','task':task}
        source=n/'navigation-coupling-code-20260908-v2/src'
        module='offline_study.navigation_coupling_behavior'
    else:
        evidence=c/'pusht-coupling-evidence-20260908-v2'
        kwargs={'vendor':root/'code/vendor/jepa-wms',
            'reference':nj/'pusht-planning-native-20260908-v1',
            'reference-code':nj/'pusht-planning-code-20260908-v1',
            'fit':evidence/'fits/vision_action_coupling','cohort':evidence/'cohort/cohort.json',
            'freeze':c/'pusht-coupling-behavior-20260908-v2/freeze',
            'checkpoint':root/'checkpoints/jepa_wm_pusht.pth.tar',
            'data-root':root/'data/pusht_noise'}
        source=c/'pusht-coupling-code-20260908-v2/src'
        module='offline_study.pusht_coupling_behavior'
    assert sha(kwargs['freeze']/'protocol.json')==FREEZES[task]
    args=[]
    for key,val in kwargs.items():args+=['--'+key,str(val)]
    return ['/workspace/table-python-inherited/bin/python','-u','-m',module],args,source


def main(a):
    root=a.root; out=a.output or root/'expansion-20260911-v1'/a.task
    out.mkdir(parents=True,exist_ok=False)
    try:
        found,incomplete=inventory(root,a.task)
        missing=[(arm,r) for arm in ARMS for r in range(8) if (arm,r) not in found]
        expected=({('permuted_visual',r) for r in range(8)}|{('permuted_joint',r) for r in range(3)}
                  if a.task=='pointmaze' else
                  {(arm,r) for arm in ARMS for r in range(8)}-{(arm,0) for arm in ARMS[:6]})
        assert set(missing)==expected, 'Observed missing streams differ; reconcile before launch'
        physical=a.physical_gpus or list(range(a.gpus))
        assert len(physical)==a.gpus and len(set(physical))==a.gpus
        assert not subprocess.check_output(['nvidia-smi','--id='+','.join(map(str,physical)),
            '--query-compute-apps=pid','--format=csv,noheader']).strip()
        base,kwargs,source=command(root,a.task)
        if a.python: base[0]=str(a.python)
        total=a.total_gpus or a.gpus
        assert 0<=a.gpu_offset and a.gpu_offset+a.gpus<=total
        queues={g:missing[(a.gpu_offset+g)::total] for g in range(a.gpus)}
        write(out/'PLAN.json',{'task':a.task,'instance':a.instance,'source_sha256':sha(Path(__file__)),
            'freeze_sha256':FREEZES[a.task],'missing_streams':missing,'queues':queues,
            'old_streams':{arm+':'+str(r):v for (arm,r),v in found.items()},
            'incomplete_archive_copies_preserved':incomplete,'episodes_to_collect':12*sum(map(len,queues.values())),
            'gpu_offset':a.gpu_offset,'total_task_gpus':total,
            'physical_gpus':physical,'python':base[0],
            'episodes_per_arm_total':96,'fresh_confirmation':False,'deadline':a.deadline})
        stop=threading.Event()
        signal.signal(signal.SIGTERM,lambda *_:stop.set())
        signal.signal(signal.SIGINT,lambda *_:stop.set())
        def worker(gpu):
            completed=[]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(physical[gpu]),PYTHONPATH=str(source),
                JEPA_VERIFIED_LOCAL_DINO='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',
                MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
            def run(cmd,label,cap):
                if stop.is_set() or time.time()+cap>a.deadline:raise RuntimeError('Stop before intact job')
                with (out/f'gpu-{gpu}-{label}.log').open('x') as log:
                    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    print(json.dumps({'task':a.task,'gpu':gpu,'job':label,'pid':child.pid}),flush=True)
                    started=time.monotonic()
                    while child.poll() is None:
                        if stop.is_set() or time.monotonic()-started>cap:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=20)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                            raise RuntimeError('Owned job interrupted; partial outputs retained')
                        time.sleep(2)
                    if child.returncode:raise RuntimeError('Frozen worker failed: '+label)
            try:
                eng=out/'engineering'/f'gpu-{gpu}'
                run(base+['engineer']+kwargs+['--output',str(eng)],'engineering',2700)
                assert sha(eng/'report.json')==read(eng/'DONE.json')['report_sha256']
                for arm,rank in queues[gpu]:
                    target=out/'conditions'/arm/f'shard-{rank}'
                    run(base+['run']+kwargs+['--engineering',str(eng),'--arm',arm,
                        '--logical-ranks',str(rank),'--output',str(target)],f'{arm}-{rank}',2400)
                    report=read(target/'report.json')
                    assert sha(target/'report.json')==read(target/'DONE.json')['report_sha256']
                    assert report['episodes']==12 and report['arm']==arm
                    for name,digest in report['episode_files_sha256'].items():assert sha(target/name)==digest
                    completed.append((arm,rank))
                write(out/f'GPU_{gpu}_DONE.json',{'completed':completed})
            except Exception as e:
                stop.set();write(out/f'GPU_{gpu}_FAILED.json',{'error':str(e),'completed':completed});raise
        with ThreadPoolExecutor(max_workers=a.gpus) as pool:list(pool.map(worker,range(a.gpus)))
        assigned=sum(map(len,queues.values()))
        write(out/'QUEUE_DONE.json',{'task':a.task,'streams':assigned,'episodes':12*assigned,
                                   'full_six_task_study_complete':False})
    except Exception as e:
        write(out/'QUEUE_FAILED.json',{'error':str(e),'partial_results_retained':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=('pointmaze','pusht'),required=True)
    p.add_argument('--root',type=Path,default=Path('/workspace/table-completion-20260911-v1'))
    p.add_argument('--gpus',type=int,choices=(2,4,8),required=True)
    p.add_argument('--instance',type=int,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--gpu-offset',type=int,default=0);p.add_argument('--total-gpus',type=int)
    p.add_argument('--physical-gpus',nargs='+',type=int)
    p.add_argument('--output',type=Path);p.add_argument('--python',type=Path)
    main(p.parse_args())
