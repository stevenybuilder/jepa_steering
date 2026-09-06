#!/usr/bin/env python3
"""Equal-state descriptive summaries only; no candidate selection or new model call."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def mean(x):return sum(x)/len(x)
def norm(x):return math.sqrt(sum(v*v for v in x))
def ratio(a,b):return a/b if b else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    workers=[49987413,49987402,49987405,49987414];reports=[];sources=[]
    for e,w in enumerate(workers):
        folder=a.root/f'worker-{w}';path=folder/f'expanded-dev-{e:03d}.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if (folder/'CPU_FINALIZED.json').exists():
            proof=json.loads((folder/'CPU_FINALIZED.json').read_text());expected=proof['output_sha256']
        else:
            proof=json.loads((folder/'DONE.json').read_text());expected=next(r['sha256'] for r in proof['outputs'] if r['path']==path.name)
        if digest!=expected:raise ValueError('Compact source SHA')
        d=json.loads(path.read_text());reports.append(d);sources.append(dict(episode=e,path=str(path),sha256=digest,final_parameter_guard_reached=e in (1,2)))
    arms=[]
    for i,s in enumerate(reports[0]['score']):
        row=dict(arm=s['arm'],per_state=[],equal_state_h6_relative_gain_percent={})
        for e,d in enumerate(reports):
            base=d['score'][0];v=d['score'][i]
            row['per_state'].append(dict(episode=e,choice=v['choice'],native_choice=base['choice'],goal_coverage_delta=v['goal_coverage_delta'],xy_delta_px=v['xy_delta_px'],
                h6_forecast_gain_percent={k:100*(1-v['equal_candidate_mse_by_horizon'][k][-1]/base['equal_candidate_mse_by_horizon'][k][-1]) for k in ('visual','proprio')},
                radius_strata=v['radius_strata']))
        for k in ('visual','proprio'):
            gains=[r['h6_forecast_gain_percent'][k] for r in row['per_state']]
            row['equal_state_h6_relative_gain_percent'][k]=mean(gains)
        arms.append(row)
    coupling=[]
    for e,d in enumerate(reports):
        for contrast in d['interactions']:
            result=dict(episode=e,sign=contrast['sign'],visual_kind=contrast['visual_kind'],per_horizon={})
            for k,stats in contrast['output'].items():
                result['per_horizon'][k]=[]
                for h in range(6):
                    r=norm(stats['raw_l2'][h][1:]);v=norm(stats['individual_visual_l2'][h][1:]);z=norm(stats['individual_action_l2'][h][1:]);floor=norm(stats['additive_float32_rounding_l2'][h][1:])
                    result['per_horizon'][k].append(dict(horizon=h+1,interaction_l2=r,visual_effect_l2=v,action_effect_l2=z,interaction_over_action=ratio(r,z),interaction_over_visual=ratio(r,v),interaction_over_sum=ratio(r,v+z),additive_rounding_l2=floor,interaction_over_rounding=ratio(r,floor)))
            coupling.append(result)
    payload=dict(complete=True,independent_states=4,candidates_per_state=64,arms=arms,coupling=coupling,sources=sources,
        all11_choices_unchanged=all(r['choice']==r['native_choice'] for a in arms for r in a['per_state']),
        process_seconds_upper_bound=60.7241981131956+3*75+63.382730705081485,
        gpu_model_calls='Four banks, eleven arms; state2 repeated solely after timer lost unsaved result. No new physics/CEM/held inputs.',
        limitations=['States0/3 CPU-finalized complete tensors; final live-parameter hash not reached.','Broader rank10 current actions plus5–20x added radii and64-candidate bank; not an isolated breadth comparison.','Joint uses both component budgets, not matched total intervention effect.','Squared metric interaction is decomposed separately from vector interaction; no global linearity or efficacy claim.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(payload,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(complete=True,unchanged=payload['all11_choices_unchanged'],process_seconds=payload['process_seconds_upper_bound'],joint_plus_h6_visual_gain=[r['h6_forecast_gain_percent']['visual'] for r in next(x for x in arms if x['arm']=='joint_plus')['per_state']]),indent=2))


if __name__=='__main__':main()
