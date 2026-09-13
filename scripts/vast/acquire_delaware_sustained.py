"""Exact authorized replacement; 40-second all-GPU load, no hardware changes."""
import json
from pathlib import Path
import fresh_fleet_acquire as fleet
from smoke_eight_gpu_hosts import api, now

SMOKE = r'''
import concurrent.futures as cf, json, subprocess, threading, time
import xml.etree.ElementTree as ET
start=time.monotonic()
def hardware(stage):
 p=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,timeout=5)
 assert p.returncode==0, p.stderr
 nodes=ET.fromstring(p.stdout).findall('gpu')
 print(json.dumps({'hardware_stage':stage,'visible_gpu_count':len(nodes),'visible_uuids':[n.findtext('uuid') for n in nodes],'nvidia_stderr':p.stderr}),flush=True)
 assert len(nodes)==8, len(nodes)
 rows=[]
 for node in nodes:
  values={child.tag:child.text for child in node.iter() if not list(child)}
  for key,value in values.items():
   if ('hw_thermal_slowdown' in key or 'sw_thermal_slowdown' in key) and value=='Active':
    raise RuntimeError('Thermal safeguard active: '+str(values))
  rows.append({'uuid':node.findtext('uuid'),'name':node.findtext('product_name'),
   'temperature':node.findtext('temperature/gpu_temp'),'sm_clock':node.findtext('clocks/sm_clock'),
   'power_draw':node.findtext('gpu_power_readings/power_draw') or node.findtext('power_readings/power_draw'),
   'power_limit':node.findtext('gpu_power_readings/power_limit') or node.findtext('power_readings/power_limit'),
   'thermal_and_power_flags':{k:v for k,v in values.items() if 'slowdown' in k or 'power_cap' in k}})
 print(json.dumps({'hardware':stage,'devices':rows}),flush=True)
hardware('before')
import torch
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
assert torch.cuda.device_count()==8
barrier=threading.Barrier(8,timeout=25)
def run(i):
 with torch.cuda.device(i),torch.inference_mode():
  scratch=torch.full((19*1024**3,),17,dtype=torch.uint8,device=f'cuda:{i}')
  a=torch.ones((4096,4096),dtype=torch.float32,device=f'cuda:{i}')
  out=torch.empty_like(a)
  torch.cuda.synchronize(i)
  barrier.wait()
  begin=time.monotonic(); calls=0
  while time.monotonic()-begin<40:
   torch.mm(a,a,out=out);torch.cuda.synchronize(i);calls+=1
  assert bool((out==4096).all().item())
  assert bool((scratch[::1024]==17).all().item())
  return {'gpu':i,'sustained_seconds':time.monotonic()-begin,'fp32_matmuls':calls,'allocated_bytes':torch.cuda.memory_allocated(i)}
with cf.ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(run,i) for i in range(8)]
 while not all(f.done() for f in futures):
  hardware('during');time.sleep(5)
 results=[f.result() for f in futures]
hardware('after')
assert time.monotonic()-start<115
print(json.dumps({'infrastructure_smoke_pass':True,'devices':results,'elapsed_s':time.monotonic()-start}),flush=True)
'''

def main():
    fleet.OUT=fleet.ROOT/'artifacts/offline_study/fresh-fleet-20260912-v9-delaware-sustained'
    if fleet.OUT.exists():raise SystemExit('Existing receipt: no duplicate rental')
    offers=api('search','offers','num_gpus=8 rentable=true','-n','--storage',80,'--limit',1000,'-o','dph_total',timeout=25)
    assert isinstance(offers,list)
    selected=[r for r in offers if r.get('id')==46814927 and r.get('machine_id')==140118
              and r.get('gpu_name')=='RTX 5090' and r.get('num_gpus')==8
              and r.get('dph_total',999)<=8.60 and r.get('reliability',0)>=.96
              and str(r.get('geolocation','')).endswith('US')]
    if len(selected)!=1:raise SystemExit('Exact authorized offer unavailable; no substitution')
    rows=api('show','instances',timeout=15)
    assert isinstance(rows,list)
    assert not any(r.get('machine_id')==140118 for r in rows), 'Already leased'
    active=[r for r in rows if r.get('intended_status')=='running']
    rate=sum(float(r.get('dph_total',999)) for r in active)
    assert rate+selected[0]['dph_total']+.03<=12
    fleet.OUT.mkdir(parents=True,exist_ok=False)
    fleet.save('authorization.json',{'utc':now(),'selected':selected,'existing_running_rate':rate,
        'hourly_cap':12,'campaign_total_cap':220,'sustained_load_seconds':40,'smoke_max_seconds':120,
        'provisioning_max_seconds':900,'hardware_settings_unchanged':True})
    fleet.SMOKE=SMOKE
    result=fleet.acquire(selected[0])
    fleet.save('summary.json',result)
    print(json.dumps({'receipt':str(fleet.OUT),'status':result['status']}),flush=True)

if __name__=='__main__':main()
