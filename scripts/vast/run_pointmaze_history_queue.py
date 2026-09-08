"""Owned GPU0 handoffs: Virginia Wall234 to Maze234, Indiana offline to Maze235.

No interruption, rental, output-based selection or confirmation access. Other
PointMaze seeds remain separately schedulable; they are not silently omitted.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace/jepa-runtime')
PYTHON = '/workspace/jepa-planning-python/bin/python'
SOURCE_SHA = 'd4bd561e1d1074059dd3ead28fc5509ec7c3a5f6a854dea3f78856b762c0ccab'
UUID = 'GPU-5bbfd503-dbb3-c9ad-7b41-2a4f985b18fc'


def verify_wall_history(output, verified_report, sha256):
    report, digest = verified_report(output)
    if (report['status'] != 'all50_wall_training_epochs_complete' or report['seed'] != 234 or
            not report['training_history_complete'] or report['checkpoints'] != 50 or
            report['validation_events'] != 100 or report['protocol_sha256'] != sha256(output/'protocol.json') or
            report['checkpoint_manifest_sha256'] != sha256(output/'CHECKPOINTS.json')):
        raise ValueError('Incomplete or changed Wall234 history')
    history = json.loads((output/'CHECKPOINTS.json').read_text())['history']
    if [r['epoch'] for r in history] != list(range(1,51)):
        raise ValueError('Missing a required Wall checkpoint')
    for row in history:
        if sha256(Path(row['path'])) != row['sha256']:
            raise ValueError('A completed Wall checkpoint changed')
    return digest


def verify_navigation_completion(sha256):
    bindings = {}
    for category in ('vision_action_coupling', 'action_response_geometry'):
        for precision in ('bfloat16', 'float32'):
            for task in ('wall', 'pointmaze'):
                for shard in range(8):
                    output=ROOT/f'navigation-comparisons-20260907-v1/{precision}/{task}/{category}/shard-{shard}'
                    if (output/'FAILED.json').exists():
                        raise ValueError('Predecessor has a failed required navigation shard')
                    if not (output/'DONE.json').exists():
                        return None
                    done=json.loads((output/'DONE.json').read_text())
                    for name in ('report','window_metrics','mechanism_diagnostics','selection','protocol'):
                        if sha256(output/(name+'.json'))!=done[name+'_sha256']:
                            raise ValueError('Completed navigation shard changed')
                    report=json.loads((output/'report.json').read_text())
                    if (report['task'],report['category'],report['precision'],report['shard_index'],report['shard_count']) != (task,category,precision,shard,8):
                        raise ValueError('Wrong completed navigation scope')
                    bindings[str(output.relative_to(ROOT))]=sha256(output/'DONE.json')
    return bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance',type=int,choices=(50189244,50205763),default=50189244)
    args=parser.parse_args()
    is_wall=args.instance==50189244
    seed=234 if is_wall else 235
    predecessor=7255 if is_wall else 3722
    expected_command=b'offline_study.training_history' if is_wall else b'run_navigation_nonrank_remaining.py'
    expected_uuid=UUID if is_wall else 'GPU-1541ee72-c8fc-ca75-7fea-470ae793c81e'
    import sys
    code = ROOT/'pointmaze-history-code-20260908-v1'
    sys.path.insert(0, str(code/'src'))
    from offline_study.author_fit import source_hash
    from offline_study.behavioral_development import verified_report
    from offline_study.protocol import sha256, write_json
    if source_hash() != SOURCE_SHA:
        raise ValueError('Unexpected PointMaze immutable source')
    actual_uuid = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if actual_uuid != expected_uuid:
        raise ValueError('Wrong leased device')
    output = ROOT/f'pointmaze-training-history-20260908-v1/seed-{seed}'
    output.mkdir(parents=True, exist_ok=False)
    wall = ROOT/'wall-training-history-20260907-v1/seed-234/remaining-epochs'
    write_json(output/'QUEUE_LAUNCH.json',{'instance':args.instance,'gpu':0,'device_uuid':expected_uuid,
        'pid':os.getpid(),'source_sha256':SOURCE_SHA,'predecessor_pid':predecessor,
        'predecessor_output':str(wall) if is_wall else str(ROOT/'navigation-comparisons-20260907-v1'),
        'seed':seed,'full_required_seed_triplet':[234,235,236],'other_seeds_not_complete':True,
        'outcome_based_decision':False,'confirmation_launched':False})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0',JEPA_VERIFIED_LOCAL_DINO='1',
        PYTHONPATH=str(code/'src')+':/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/opt/conda/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    child = None
    def interrupt(signum, frame):
        raise RuntimeError('Queue interrupted; preserve all partial outputs')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def execute(module,args,label,cap):
        nonlocal child
        print(json.dumps({'stage':label,'gpu':0,'seed':seed}),flush=True)
        with (ROOT/f'pointmaze-history-{label}-20260908-v1.log').open('x') as log:
            child = subprocess.Popen([PYTHON,'-u','-m',module]+args,env=env,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(timeout=cap):
                raise ValueError('Required '+label+' failed; preserve and stop, no retry')
        child = None
    try:
        print(json.dumps({'stage':'waiting_for_complete_predecessor','predecessor_pid':predecessor}),flush=True)
        while True:
            if is_wall and (wall/'DONE.json').exists():
                break
            if not is_wall and (ROOT/'navigation-comparisons-20260907-v1/float32/pointmaze/action_response_geometry/shard-7/DONE.json').exists():
                break
            if is_wall and (wall/'FAILED.json').exists():
                raise ValueError('Predecessor Wall run failed')
            cmd = Path(f'/proc/{predecessor}/cmdline')
            if not cmd.exists():
                raise ValueError('Predecessor process terminal without completion')
            command = cmd.read_bytes()
            if expected_command not in command or (is_wall and str(wall).encode() not in command):
                raise ValueError('Predecessor identity changed or became terminal')
            time.sleep(20)
        previous_proof={'wall_report_sha256':verify_wall_history(wall,verified_report,sha256),'checkpoints_verified':50} if is_wall else {
            'navigation_shard_receipts':verify_navigation_completion(sha256),'required_shards':64}
        if not is_wall and (previous_proof['navigation_shard_receipts'] is None or len(previous_proof['navigation_shard_receipts'])!=64):
            raise ValueError('Not all registered navigation predecessor shards completed')
        # A DONE record may be published just before Python releases CUDA memory.
        for _ in range(12):
            occupied = subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
            if not occupied:
                break
            if is_wall and occupied != ['7255']:
                raise ValueError('Reserved GPU is occupied by a different job; do not multiplex')
            if not is_wall:
                for pid in occupied:
                    command=Path(f'/proc/{pid}/cmdline').read_bytes()
                    parent=int(Path(f'/proc/{pid}/stat').read_text().split()[3])
                    if b'offline_study.navigation_evaluate' not in command or parent!=predecessor:
                        raise ValueError('A different job owns the intended GPU; do not multiplex')
            time.sleep(10)
        else:
            raise ValueError('Completed predecessor did not release GPU; do not terminate it')
        write_json(output/'PREDECESSOR_VERIFIED.json',{**previous_proof,
            'prior_results_preserved':True,'prior_process_not_interrupted':True})
        vendor=['--vendor','/workspace/jepa_steering/vendor/jepa-wms']
        pilot=ROOT/'pointmaze-training-accumulation-pilot-20260908-v1'
        execute('offline_study.training_pilot',vendor+['--assets',str(ROOT/'navigation-assets-20260907-v1'),
            '--input-check',str(ROOT/'navigation-input-check-20260907-v1'),'--task','pointmaze',
            '--output',str(pilot)],'pilot',1800)
        common=vendor+['--data-root',str(ROOT/'navigation-assets-20260907-v1/extracted/pointmaze/point_maze'),
            '--input-check',str(ROOT/'navigation-input-check-20260907-v1'),
            '--input-receipt',str(ROOT/'pointmaze-training-inputs-20260908-v1'),'--pilot',str(pilot),'--seed',str(seed)]
        engineering=output/'epoch-one-engineering'
        execute('offline_study.pointmaze_training_history',common+['--engineering-only','--output',str(engineering)],'engineering',7200)
        execute('offline_study.pointmaze_training_history',common+['--resume-from',str(engineering/'jepa-e0.pth.tar'),
            '--engineering-proof',str(engineering),'--output',str(output/'remaining-epochs')],'remaining',72*3600)
        print(json.dumps({'status':'pointmaze_training_queue_finished','seed':seed,'other_seeds_complete':False,
            'planning_history_evaluations_complete':False,'full_study_complete':False}),flush=True)
    except Exception as exc:
        write_json(output/'QUEUE_FAILED.json',{'error':str(exc),'full_study_complete':False})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':
    main()
