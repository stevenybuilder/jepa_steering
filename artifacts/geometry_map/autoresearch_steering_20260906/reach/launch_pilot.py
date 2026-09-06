from dispatch import *
import time
p=json.loads((LOCAL/'protocol-v1.json').read_text());p['runner_sha256']=hashlib.sha256((LOCAL/'pilot.py').read_bytes()).hexdigest();p['per_worker_process_seconds_limit']=1200;p['test_input_manifest_sha256']=hashlib.sha256((LOCAL/'test-inputs.json').read_bytes()).hexdigest();p['frozen_correction_sha256']=hashlib.sha256((LOCAL/'fit-correction.npz').read_bytes()).hexdigest();(LOCAL/'protocol-v1.json').write_text(json.dumps(p,indent=2)+'\n')
def launch(w):
 r=root(w)
 for f,d in [('pilot.py','code-v1/pilot.py'),('test-inputs.json','test-inputs.json'),('protocol-v1.json','protocol-v1.json')]:upload(w,LOCAL/f,r+'/'+d)
 # No alternate experiment permitted on reserved worker.
 before=run(w,'nvidia-smi --query-compute-apps=pid --format=csv,noheader');assert not before.strip(),before
 repo=BASE+'/vendor/jepa-wms';config=repo+'/configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml'
 episodes={49155754:[66,67],49902461:[68,69],49982193:[70,71,72,73]}[w]
 cmd=f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH={r}/code-v1:{repo} /opt/conda/bin/python -u {r}/code-v1/pilot.py --root {BASE} --repo {repo} --config {config} --correction {r}/correction.npz --protocol {r}/protocol-v1.json --worker {w} --episodes "+' '.join(map(str,episodes))+f' --output {r}/pilot-v1'
 pid=run(w,f'nohup bash -c '+shlex.quote(cmd)+f' > {r}/pilot-v1.log 2>&1 < /dev/null & echo $!').strip()
 receipt=dict(worker=w,pid=pid,episodes=episodes,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),output=r+'/pilot-v1',command=cmd)
 (LOCAL/f'launch-{w}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
with ThreadPoolExecutor(3)as ex:list(ex.map(launch,HOSTS))
