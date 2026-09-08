"""Complete paired HMM development analysis, preserving original reference labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from .behavioral_analysis import scenario_key,shard_paths
from .behavioral_development import schedule,validate_coverage,verified_report
from .protocol import sha256,write_json
from .routing_behavior import ARMS,CONTRASTS,NEW_ARMS,REUSED,TASKS,read_protocol,reference_records,verify_engineering,verify_routed_calls


def load_panel(args):
    protocol=read_protocol(args);panels={};bindings={};engineering_cache={}
    freeze_hash=sha256(args.freeze/'protocol.json')
    for task in TASKS:
        panels[task]={}
        for arm in ARMS:
            if arm in REUSED:
                rows,source=reference_records(args,protocol,task,arm);bindings.update(source)
            else:
                rows=[]
                for root in shard_paths(args.panel/task/arm):
                    report,digest=verified_report(root);launch=json.loads((root/'protocol.json').read_text())
                    if ((root/'FAILED.json').exists() or report['status']!='complete_routed_fixed_response_shard' or
                            report['task']!=task or report['arm']!=arm or
                            report['protocol_sha256']!=sha256(root/'protocol.json') or
                            launch['freeze_sha256']!=freeze_hash or launch['source_sha256']!=protocol['source_sha256'] or
                            launch['task']!=task or launch['arm']!=arm or launch['engineering_only'] is not False or
                            launch['device_uuid']!=report['device_uuid'] or report['parameters_unchanged'] is not True or
                            report['scientific_efficacy_measurement'] is not True or report['fresh_confirmation'] is not False):
                        raise ValueError('Unbound or failed routed shard')
                    engineering=args.panel/'engineering'/task/report['device_uuid']
                    if str(engineering) not in engineering_cache:
                        engineering_cache[str(engineering)]=verify_engineering(engineering,protocol,task,freeze_hash)
                    proof,proof_hash=engineering_cache[str(engineering)]
                    if proof_hash!=launch['engineering_report_sha256'] or proof['device_uuid']!=report['device_uuid']:
                        raise ValueError('Unbound routed receiving-device proof')
                    bindings[str(engineering/'report.json')]=proof_hash
                    expected=launch['expected_episodes'];piece=[]
                    names=[f"episode-{r['episode']:03d}.json" for r in expected]
                    if set(report['episode_files_sha256'])!=set(names) or len(names)!=report['episodes']:
                        raise ValueError('Missing or extra routed episode')
                    for name in names:
                        if sha256(root/name)!=report['episode_files_sha256'][name]:raise ValueError('Routed episode changed')
                        row=json.loads((root/name).read_text());calls=root/f"calls-{row['episode']:03d}"
                        if row['arm']!=arm or row['device_uuid']!=report['device_uuid']:raise ValueError('Wrong arm/device')
                        for filename,key in (('unroll_calls.json','unroll_calls_sha256'),('action_trace.json','action_trace_sha256')):
                            if sha256(calls/filename)!=row[key]:raise ValueError('Routed raw trace changed')
                        verify_routed_calls(row['result'],json.loads((calls/'unroll_calls.json').read_text()),arm)
                        piece.append(row)
                    validate_coverage(piece,expected);rows.extend(piece);bindings[str(root/'report.json')]=digest
            panels[task][arm]=sorted(rows,key=lambda r:r['episode'])
        # Reused constant/random endpoints need the same GPU's routed-equivalence
        # proof even though their original records remain entirely unchanged.
        for arm in REUSED:
            for row in panels[task][arm]:
                root=args.panel/'engineering'/task/row['device_uuid']
                if str(root) not in engineering_cache:
                    engineering_cache[str(root)]=verify_engineering(root,protocol,task,freeze_hash)
                _,digest=engineering_cache[str(root)]
                bindings[str(root/'report.json')]=digest
    validate_panel(panels)
    return panels,protocol,bindings


def validate_panel(panels):
    if tuple(panels)!=TASKS:raise ValueError('Both tasks required')
    for task,arms in panels.items():
        if tuple(arms)!=ARMS:raise ValueError('All seven arms required')
        for arm,rows in arms.items():
            validate_coverage(rows,schedule())
            for native,row in zip(arms['native'],rows,strict=True):
                if (row['arm']!=arm or row['local_seed']!=native['local_seed'] or
                        row['device_uuid']!=native['device_uuid'] or
                        row['initial_state_vector']!=native['initial_state_vector'] or
                        any(row['result'][k]!=native['result'][k] for k in ('initial_sha256','goal_sha256')) or
                        not np.isfinite(row['seconds']) or row['seconds']<=0):
                    raise ValueError('Unpaired scenario/RNG/device or invalid duration')


def analyze(panels):
    validate_panel(panels);rng=np.random.default_rng(2026090803);tasks={};contrasts=[]
    for task,arms in panels.items():
        groups={}
        for i,row in enumerate(arms['native']):groups.setdefault(scenario_key(row),[]).append(i)
        clusters=list(groups.values());n=len(clusters)
        if n<2:raise ValueError('Need independent scenario clusters')
        sizes=np.array([len(c) for c in clusters]);draws=rng.integers(n,size=(20000,n));denominator=sizes[draws].sum(1)
        success={a:np.array([r['result']['native_success'] for r in rows],float) for a,rows in arms.items()}
        tasks[task]={'scenario_clusters':n,'arms':{a:{'episodes':96,'successes':int(success[a].sum()),
            'success_percent':float(success[a].mean()*100),'mean_episode_seconds':float(np.mean([r['seconds'] for r in rows])),
            'runtime_source':'inherited one-pass reference' if a in REUSED else 'causal same-pass routing; exact two-pass equivalence required in engineering'} for a,rows in arms.items()}}
        for candidate,control in CONTRASTS:
            delta=success[candidate]-success[control];totals=np.array([delta[c].sum() for c in clusters])
            samples=totals[draws].sum(1)/denominator*100;lo,hi=np.quantile(samples,[.05/32,1-.05/32])
            contrasts.append({'task':task,'candidate':candidate,'control':control,
                'gain_percentage_points':float(delta.mean()*100),'simultaneous_95_interval_percentage_points':[float(lo),float(hi)],
                'paired_positive_episodes':int((delta>0).sum()),'paired_negative_episodes':int((delta<0).sum()),
                'scenario_clusters':n,'zero_discordance_does_not_establish_equivalence':True})
    return {'status':'complete_paired_hmm_fixed_response_development_analysis','tasks':tasks,'contrasts':contrasts,
        'bootstrap_draws':20000,'simultaneous_interval_family_size':16,'requested_tasks':6,'measured_tasks':2,
        'fresh_confirmation':False,'confirmation_access_authorized':False,'full_study_complete':False,
        'conditional_history_eligibility_claimed':False,'reference_records_relabelled_in_place':False,
        'selection_from_partial_results':False,'task_success_distinct_from_fit_history_diagnostic':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('vendor','fits','routing','stimuli','reference','reference-freeze','reference-code','reference-workers','freeze','panel','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();panels,protocol,bindings=load_panel(args);report=analyze(panels)
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'bindings.json',bindings)
    report.update(bindings_sha256=sha256(args.output/'bindings.json'),freeze_sha256=sha256(args.freeze/'protocol.json'))
    write_json(args.output/'report.json',report);write_json(args.output/'DONE.json',{'report_sha256':sha256(args.output/'report.json')})


if __name__=='__main__':main()
