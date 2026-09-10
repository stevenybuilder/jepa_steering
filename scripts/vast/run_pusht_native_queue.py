"""Owned NJ GPU0: receiving checks,96 native replication episodes,then Wall235.

No candidate selection, confirmation reveal, incomplete-run retry or other GPU use.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace/jepa-runtime')
PYTHON = '/workspace/jepa-planning-python/bin/python'
SOURCE_SHA = '6528bdc788756d1c34e6cd955e3ed64996e6192ad60e5edb560b431f81dedc59'


def main():
    code = ROOT / 'pusht-planning-code-20260908-v1'
    assets = ROOT / 'pusht-planning-assets-20260908-v3'
    evidence = ROOT / 'pusht-planning-evidence-20260908-v1'
    output = ROOT / 'pusht-planning-native-20260908-v1'
    cohort = evidence / 'cohorts/pusht/cohort.json'
    checkpoint = ROOT / 'pusht-planning-assets-20260908-v2/downloads/models/jepa_wm_pusht.pth.tar'
    os.environ.update(CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        JEPA_VERIFIED_LOCAL_DINO='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(code/'src')+':/workspace/jepa-python/lib/python3.10/site-packages', LD_LIBRARY_PATH='/opt/conda/lib')
    import sys
    sys.path.insert(0, str(code / 'src'))
    from offline_study.author_fit import source_hash
    from offline_study.behavioral_development import verified_report, schedule, assigned_rows
    from offline_study.navigation_replication import validate_records
    from offline_study.protocol import sha256, write_json
    if source_hash() != SOURCE_SHA:
        raise ValueError('Wrong immutable Push-T execution source')
    if subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader']).strip():
        raise ValueError('Reserved GPU0 is not empty')
    if sha256(assets/'receiving_report.json') != json.loads((assets/'RECEIVING_DONE.json').read_text())['report_sha256']:
        raise ValueError('Missing receiving input verification')
    if sha256(checkpoint) != '9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb':
        raise ValueError('Wrong released Push-T model')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'LAUNCH.json', {'queue_pid':os.getpid(),'gpu':0,'source_sha256':SOURCE_SHA,
        'input_report_sha256':sha256(assets/'receiving_report.json'),'episodes_per_condition_total':96,
        'role':'native_replication_then_existing_wall235_history','fresh_confirmation':False})
    child = None
    def interrupt(signum, frame):
        raise RuntimeError('Owned queue interrupted; preserve partial output')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    def execute(module, args, label, cap):
        nonlocal child
        print(json.dumps({'stage':label,'gpu':0}),flush=True)
        with (ROOT/f'pusht-native-{label}-20260908-v1.log').open('x') as log:
            child = subprocess.Popen([PYTHON,'-u','-m',module]+args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(timeout=cap):
                raise ValueError('Required '+label+' failed; no automatic retry/advance')
        child = None
    vendor = ['--vendor','/workspace/jepa_steering/vendor/jepa-wms']
    try:
        simulator = output/'simulator-check'
        execute('offline_study.planning_env_smoke',vendor+['--original-root',str(evidence),
            '--data-root',str(assets/'source/data'),'--tasks','pusht','--output',str(simulator)],'simulator',1200)
        sim,_ = verified_report(simulator)
        for key,expected in {'initial_sha256':'538a6b89f257eebd7c9d085cafca0c6a6c8b3253b12f0b7252f3965dad5b037b',
            'goal_sha256':'cde7f5c4b6ec177ba2bbf32b89cc57df9a0f66339caf1ba5aa82c9bfe9ceb427',
            'fit_trajectory_id':'pusht:train:10810'}.items():
            if sim['tasks']['pusht'][key] != expected:
                raise ValueError('Receiving stimulus differs from original engineering: '+key)
        engineering = output/'engineering'
        common = vendor+['--cohort',str(cohort),'--data-root',str(assets/'source/data/pusht_noise')]
        execute('offline_study.pusht_planning_check',common+['--checkpoint',str(checkpoint),
            '--simulator-smoke',str(simulator),'--output',str(engineering)],'engineering',1800)
        freeze = output/'freeze'
        execute('offline_study.pusht_planning_replication',['freeze']+common+['--engineering',str(engineering),
            '--baseline-code',str(code),'--output',str(freeze)],'freeze',600)
        records, bindings = [], {}
        for rank in range(8):
            target = output/f'native/shard-{rank}'
            execute('offline_study.pusht_planning_replication',['native']+common+['--checkpoint',str(checkpoint),
                '--freeze',str(freeze),'--logical-ranks',str(rank),'--output',str(target)],f'shard-{rank}',7200)
            report,digest = verified_report(target)
            protocol = json.loads((target/'protocol.json').read_text())
            if (report['status']!='native_pusht_planning_replication_shard_complete' or report['episodes']!=12 or
                    report['protocol_sha256']!=sha256(target/'protocol.json') or
                    protocol['freeze_sha256']!=sha256(freeze/'protocol.json') or protocol['logical_ranks']!=[rank]):
                raise ValueError('Wrong completed native scope')
            piece=[]
            for row in assigned_rows(schedule(),[rank]):
                name=f"episode-{row['episode']:03d}.json"
                if sha256(target/name)!=report['episode_files_sha256'][name]:
                    raise ValueError('Completed native episode changed')
                piece.append(json.loads((target/name).read_text()))
            validate_records(piece,assigned_rows(schedule(),[rank]));records.extend(piece)
            bindings[str(rank)]=digest
        validate_records(records,schedule())
        families={r['source_segment']['lineage_group'] for r in records}
        write_json(output/'report.json',{'status':'all96_native_pusht_replication_episodes_verified',
            'episodes':96,'source_families_sampled':len(families),'released_source_families':21,
            'successes':sum(r['result']['native_success'] for r in records),'shard_reports_sha256':bindings,
            'freeze_sha256':sha256(freeze/'protocol.json'),'paired_candidate_comparisons_complete':False,
            'fresh_confirmation':False,'training_seed_history_complete':False,'full_study_complete':False})
        write_json(output/'DONE.json',{'report_sha256':sha256(output/'report.json')})
        # Outcome-independent, previously reserved next job on the same GPU.
        os.environ['PYTHONPATH']=str(ROOT/'wall-history-code-v1/src')+':/workspace/jepa-python/lib/python3.10/site-packages'
        execute('offline_study.training_history',vendor+[
            '--data-root',str(ROOT/'navigation-assets-20260907-v1/extracted/wall/wall_single'),
            '--input-check',str(ROOT/'navigation-input-check-20260907-v1'),
            '--input-receipt',str(ROOT/'wall-training-inputs-20260907-v1'),
            '--pilot',str(ROOT/'wall-training-accumulation-pilot-20260907-v2'),'--seed','235',
            '--resume-from',str(ROOT/'wall-training-history-20260907-v1/seed-235/remaining-epochs/jepa-e1.pth.tar'),
            '--engineering-proof',str(ROOT/'wall-training-history-20260907-v1/seed-235/epoch-one-engineering'),
            '--output',str(ROOT/'wall-training-history-resume-20260908-v1/seed-235')],'wall235-history',48*3600)
    except Exception as exc:
        write_json(output/'QUEUE_FAILED.json',{'error':str(exc),'partial_not_complete':True})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':
    main()
