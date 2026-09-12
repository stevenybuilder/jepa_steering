"""Resolve official asset redirects with local auth, then launch safe input repair."""
from concurrent.futures import ThreadPoolExecutor
import configparser
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import urllib.request

PROJECT=Path(__file__).resolve().parents[2]
BASE=PROJECT/'artifacts/offline_study/table-completion-20260911-v1/expansion-v2'
REMOTE='/workspace/table-completion-20260911-v1'


def resolve():
    values={}
    for line in (PROJECT/'.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);values[k.strip()]=v.strip().strip('"').strip("'")
    token=next(values[k] for k in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN') if values.get(k))
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            result=super().redirect_request(req,fp,code,msg,headers,newurl)
            if result is not None:result.remove_header('Authorization')
            return result
    opener=urllib.request.build_opener(Redirect)
    assets={}
    for task,digest in [('pointmaze','a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc'),
                        ('pusht','9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb')]:
        url='https://huggingface.co/facebook/jepa-wms/resolve/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_'+task+'.pth.tar?download=true&check='+str(int(time.time()))
        with opener.open(urllib.request.Request(url,method='HEAD',headers={'Authorization':'Bearer '+token}),timeout=30) as r:
            assets[task]={'label':task+'_checkpoint','url':r.url,'sha256':digest,
                          'target':REMOTE+'/checkpoints/jepa_wm_'+task+'.pth.tar'}
    url='https://huggingface.co/datasets/facebook/jepa-wms/resolve/6116f042ae7ae4c8e3f1fd2f194f432615664182/pusht/pusht_noise.zip?download=true&check='+str(int(time.time()))
    with opener.open(urllib.request.Request(url,method='HEAD',headers={'Authorization':'Bearer '+token}),timeout=30) as r:
        assets['pusht_data']={'label':'pusht_pinned_zip','url':r.url,
            'sha256':'442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08','target':REMOTE+'/downloads/pusht_noise.zip'}
    assets['dino']={'label':'dino_encoder_weights','url':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'target':'/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth',
        'sha256':'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'}
    return assets


def main():
    assets=resolve();lease=json.loads((BASE/'LEASE.json').read_text())
    def launch(item):
        i,spec=item;local=BASE/i
        if (local/'RECOVERY_LAUNCH.json').exists():return
        if not (local/'LAUNCH.json').exists() and not (local/'READY_FOR_RECOVERY.json').exists():
            print(json.dumps({'instance':i,'awaiting_initial_staging':True}),flush=True);return
        ssh=json.loads((local/'CONNECTION.json').read_text())['ssh']
        with subprocess.Popen(ssh+['tar --no-same-owner --keep-old-files -xf - -C '+REMOTE+'/code/scripts/vast'],stdin=subprocess.PIPE) as p:
            with tarfile.open(fileobj=p.stdin,mode='w|') as t:
                for name in ['bootstrap_table_runtime_v2.sh','table_bootstrap_recovery.py']:
                    t.add(PROJECT/'scripts/vast'/name,arcname=name,recursive=False)
            p.stdin.close();assert p.wait()==0
        queue=['--task',spec['task'],'--instance',i,'--gpus',str(spec['gpus']),
            '--deadline',str(lease['deadline_timestamp']),'--gpu-offset',str(spec.get('gpu_offset',0)),
            '--total-gpus',str(spec.get('total_gpus',spec['gpus']))]
        selected=[assets[spec['task']],assets['dino']]
        if spec['task']=='pusht':selected.append(assets['pusht_data'])
        payload={'task':spec['task'],'queue_args':queue,'assets':selected}
        code="""import json,os,sys,hashlib
from pathlib import Path
r=Path('/workspace/table-completion-20260911-v1');payload=json.load(sys.stdin)
pid=os.fork()
if pid:
 record={'pid':pid,'stage':'authenticated_official_CDN_and_dependency_repair','credentials_saved':False}
 with (r/'EXPANSION_RECOVERY_LAUNCH.json').open('x') as f:json.dump(record,f)
 print(json.dumps(record),flush=True)
else:
 os.setsid()
 with (r/'expansion-recovery-supervisor.log').open('x') as f:
  os.dup2(f.fileno(),1);os.dup2(f.fileno(),2)
 sys.path.insert(0,str(r/'code/scripts/vast'))
 from table_bootstrap_recovery import main
 main(payload)
"""
        result=json.loads(subprocess.check_output(ssh+[shlex.join(['python3','-u','-c',code])],input=json.dumps(payload),text=True,timeout=45))
        with (local/'RECOVERY_LAUNCH.json').open('x') as f:json.dump(result,f,indent=2)
        if not (local/'LAUNCH.json').exists():
            with (local/'LAUNCH.json').open('x') as f:json.dump({**result,'corrected_bootstrap_direct':True},f,indent=2)
        print(json.dumps({'instance':i,'recovery_pid':result['pid']}),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,lease['workers'].items()))


if __name__=='__main__':main()
