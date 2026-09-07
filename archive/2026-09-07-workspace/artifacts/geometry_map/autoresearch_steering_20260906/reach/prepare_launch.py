from dispatch import *
import time,tarfile,io
P=dict(schema=1,task='Reach-Wall exploratory frozen output forecast-error correction',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training_episodes=[4,7,9,10],training_labels='Only actual executed H1..H3 native full99 prefixes; no future after replanning',test_episodes=list(range(66,74)),original_held_access=False,operator='Pooled400channel correction broadcast uniformly over spatial tokens at terminalH6; native fullgrid spatial complement retained. Explicit OUTPUT mechanistic comparison, not an activation-site intervention.',model='Dual ridge with fixed lambda10*n; standardized observable context400/nativeforecast400/normalized proposedaction mean4,last4,sum4/15 and horizon clipped3',dose=.5,sham='Seed20260906 signed channel permutation, independently within visual384 and proprio16, exact per-channel norm on each candidate',horizon_caveat='Trained physicaltruth H1..3; application to terminalH6 is unvalidated extrapolation. Report H1..3 actual selected native prefix forecast error separately.',bank='Saved last native CEM300proposals plus original native selected mean; all301 candidates frozen before correctedscore; native planner/weights/objective fixed',arms=['native_planner_selected','native_bank_best','correction_best','matched_sham_best'],primary='Episode-equal physically executed15rawstep hand-goal progress, correction minus eachcontrol',promotion='Full99 followup ONLY if >.005m mean over nativeplanner, banknative and sham, positive vsnativeplanner and sham on at least5/8starts; descriptive exploratory evidence only',statistical_unit='8development initialstates; plans/horizons not independent; no pvalues or confirmationclaim',process_gpu_seconds_limit=5400,hardware={49155754:[66,67],49902461:[68,69],49982193:[70,71,72,73]},source_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [LOCAL/'pilot.py',LOCAL/'extract_fit.py']})
(LOCAL/'protocol-v1.json').write_text(json.dumps(P,indent=2)+'\n')
# Fit once on491, from compact already verified train extracts.
w=49155754;r=root(w)
for who in [49155754,49902461]:upload(w,LOCAL/f'train-{who}-rows.npz',f'{r}/train-{who}.npz')
print(run(w,f"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/conda/bin/python {r}/code-v1/extract_fit.py fit --sources {r}/train-49155754.npz {r}/train-49902461.npz --output {r}/fit-v1",120),flush=True)
for n in ['correction.npz','DONE.json']:(LOCAL/('fit-'+n)).write_bytes(subprocess.check_output(ssh(w)+[f'cat {r}/fit-v1/{n}'],timeout=60))
# Immutable runtime source snapshots. Data stay on GPU hosts; relay is a stream.
scripts=LOCAL.parents[3]/'scripts/geometry_map'
assert scripts.is_dir(),scripts
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz')as tf:
 for p in scripts.glob('*.py'):tf.add(p,arcname=p.name)
raw=buf.getvalue()
def prepare(w):
 r=root(w);run(w,f'mkdir -p {r}/code-v1 {r}/input-v1')
 subprocess.run(ssh(w)+[f'tar -xzf - -C {r}/code-v1'],input=raw,check=True,capture_output=True)
 for p,dest in [(LOCAL/'pilot.py','code-v1/pilot.py'),(LOCAL/'extract_fit.py','code-v1/extract_fit.py'),(LOCAL/'fit-correction.npz','correction.npz'),(LOCAL/'protocol-v1.json','protocol-v1.json')]:upload(w,p,r+'/'+dest)
 print('PREPARED',w,flush=True)
with ThreadPoolExecutor(3)as p:list(p.map(prepare,HOSTS))
# Copy only two selected test inputs and complete initial-replan snapshots to491.
src=f'{BASE}/development-expansion-66-73-v1/worker-49902461'
files=[f'inputs/input-{e:03d}.pt'for e in [66,67]]+[f'results-v2/episode-{e:03d}-unsteered-replan-00.pt'for e in [66,67]]
prod=subprocess.Popen(ssh(49902461)+[f'tar -cf - -C {src} '+shlex.join(files)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
cons=subprocess.Popen(ssh(49155754)+[f'tar -xf - --strip-components=1 -C {root(49155754)}/input-v1'],stdin=prod.stdout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);prod.stdout.close();_,err=cons.communicate(timeout=180);assert not cons.returncode and not prod.wait(timeout=30),err
print('READY',flush=True)
