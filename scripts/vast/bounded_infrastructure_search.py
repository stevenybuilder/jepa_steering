"""User-authorized infrastructure-only search; one diagnostic rental at a time."""
import datetime
import hashlib
import json
import time
from pathlib import Path

import fresh_fleet_acquire as fleet
from acquire_delaware_sustained import SMOKE
from smoke_eight_gpu_hosts import api, now

EXCLUDED={68094,145795,147839,149537,141210,142274,140835,140118,142429}
DEADLINE=datetime.datetime(2026,9,12,22,2,tzinfo=datetime.timezone.utc).timestamp()
ROOT=fleet.ROOT/'artifacts/offline_study/fresh-fleet-20260912-v10-bounded'

def eligible(row):
    return (row.get('num_gpus')==8 and row.get('gpu_ram',0)>=23000
        and row.get('compute_cap',0)>=750 and row.get('reliability',0)>=.96
        and row.get('disk_space',0)>=80 and str(row.get('geolocation','')).endswith('US')
        and row.get('machine_id') not in EXCLUDED and row.get('dph_total',999)<=11.97)

def save(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2)+'\n')

def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    save('authorization.json',{'started':now(),'deadline_utc':'2026-09-12T22:02:00Z',
        'diagnostic_allowance':5,'fleet_hourly_cap':12,'campaign_cap':220,
        'one_unqualified_rental_at_a_time':True,'minimum_reliability':.96,
        'excluded_machines':sorted(EXCLUDED),'smoke_sha256':hashlib.sha256(SMOKE.encode()).hexdigest(),
        'scope':'infrastructure only; retain first pass for frozen actual receiving evaluation'})
    spent=0.;attempts=[];poll=0
    while time.time()<DEADLINE:
        start=time.monotonic();poll+=1
        try:
            offers=api('search','offers','num_gpus=8 rentable=true','-n','--storage',80,'--limit',1000,'-o','dph_total',timeout=25)
            live=api('show','instances',timeout=15)
            if not isinstance(offers,list) or not isinstance(live,list):
                raise ValueError('Cannot verify market and current budget')
            if any(r.get('intended_status')=='running' for r in live):
                raise SystemExit('Existing running lease; stop controller and reconcile ownership')
            active_rate=sum(float(r.get('dph_total',999)) for r in live if r.get('intended_status')=='running')
            candidates=sorted([r for r in offers if eligible(r) and not any(x.get('machine_id')==r['machine_id'] for x in live)],key=lambda r:r['dph_total'])
            # Reserve full provisioning+smoke+cleanup quote before any paid action.
            candidates=[r for r in candidates if r['dph_total']*1050/3600+spent<=5
                        and r['dph_total']+active_rate+.03<=12 and time.time()+1050<=DEADLINE]
            save(f'market-{poll:03d}.json',{'utc':now(),'quotes':offers,'selected_eligible':[r['id'] for r in candidates],
                'diagnostic_quote_time_spend':spent,'provider_invoice_not_claimed':True})
            print(json.dumps({'event':'search','utc':now(),'eligible':[{k:r.get(k) for k in ('id','machine_id','gpu_name','dph_total','reliability')} for r in candidates],'spent_quote_proxy':spent}),flush=True)
            if candidates:
                candidate=candidates[0]
                fleet.OUT=ROOT/f"attempt-{len(attempts)+1:02d}-{candidate['machine_id']}"
                fleet.OUT.mkdir(exist_ok=False)
                fleet.SMOKE=SMOKE
                result=fleet.acquire(candidate)
                attempts.append(result)
                spent+=candidate['dph_total']*result.get('elapsed_s',1050)/3600
                save('summary.json',{'attempts':attempts,'diagnostic_quote_time_spend':spent,'finished':now()})
                if result['status']=='smoke_pass_retained_for_staging':
                    print(json.dumps({'event':'PASS_RETAINED','instance':result['instance_id'],'ssh':result.get('ssh'),'receipt':str(fleet.OUT)}),flush=True)
                    return
                EXCLUDED.add(candidate['machine_id'])
        except (ValueError,TimeoutError) as error:
            print(json.dumps({'event':'read_only_check_failed','error':str(error)}),flush=True)
        time.sleep(max(0,min(30-(time.monotonic()-start),DEADLINE-time.time())))
    save('summary.json',{'attempts':attempts,'diagnostic_quote_time_spend':spent,'finished':now(),'reason':'bounded deadline reached; no qualified host'})

if __name__=='__main__':main()
