#!/usr/bin/env python3
"""Small exact initial-stimulus extraction, no model or simulator calls."""
import argparse
import json
from pathlib import Path
import torch
from cem_search_audit_v1 import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);rows=json.loads(a.inputs.read_text());a.output.mkdir(parents=True,exist_ok=False);outputs=[]
    for r in rows:
        path=Path(r['bank']);receipt=json.loads(Path(r['receipt']).read_text());entry=next(x for x in receipt['outputs'] if x['path']==path.name)
        if not receipt['complete'] or not receipt['original9_actions_goal_physics_pixels_exact'] or entry['split']!='development_external' or entry['source_id'] not in range(4):raise ValueError('Source DEV receipt')
        if sha(path)!=entry['sha256']:raise ValueError('Source SHA before load')
        bank=torch.load(path,map_location='cpu',weights_only=False)
        small={k:bank[k] for k in ('row','context','raw_goal','goal_encoded','goal_state')}
        small['candidates']=[{'states':bank['candidates'][0]['states']}]
        small['source_full64_sha256']=entry['sha256'];out=a.output/f"stimulus-{r['episode']:03d}.pt";torch.save(small,out)
        outputs.append(dict(path=out.name,source_id=r['episode'],split='development_external',sha256=sha(out),bytes=out.stat().st_size,source_full64_sha256=entry['sha256']))
    write(a.output/'DONE.json',dict(complete=True,outputs=outputs,original9_actions_goal_physics_pixels_exact=True,
        inherited_guard='Source full64 adapters verified original9 actions/goals/physics/pixels; this pack preserves only exact initial-state/goal inputs needed for NEW CEM',model_calls=0,simulator_calls=0,held_access=False))
    print(json.dumps(dict(outputs=outputs)),flush=True)


if __name__=='__main__':main()
