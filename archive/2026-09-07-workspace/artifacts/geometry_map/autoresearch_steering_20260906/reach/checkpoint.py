from dispatch import *
import time

def finish(w):
 r=root(w);assert json.loads(run(w,'cat '+r+'/pilot-fullgrid-v1/DONE.json'))['complete']
 assert not run(w,'nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip()
 upload(w,LOCAL/'margin_diagnostic.py',r+'/code-v2/margin_diagnostic.py')
 text=run(w,f"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/conda/bin/python {r}/code-v2/margin_diagnostic.py --root {r} --output {r}/margin-diagnostic-v1",120)
 for sub in ['pilot-fullgrid-v1/summary.json','margin-diagnostic-v1/summary.json']:
  raw=subprocess.check_output(ssh(w)+['cat '+r+'/'+sub],timeout=60);p=LOCAL/f'worker-{w}'/sub;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 print('checkpoint',w,text,flush=True)
with ThreadPoolExecutor(3)as ex:list(ex.map(finish,HOSTS))
