"""Complete56, source-bound H3 physical-prefix analysis; never partial means."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'paper/data'
BASE = ROOT/'artifacts/offline_study/layer-pilot-20260913-v1'
PROTOCOL_SHA = '8d80f98a557c79b6dedad27bb3ef63c2ede4979150ba8373c893da8882a232b3'
RUNNER_SHA = '95f2afe3c0ae4d6a91327f3f4232d78f302484a36be2033c4fa43aea87e626ce'
PRIOR_ANALYSIS_FREEZE_SHA = 'c9ed5ee5cf109fc21d09aa65d1dd68ce2265a73db9c972960cbf40835d629080'
PARENT_HELPER_SHA = 'dcccc73c69a9248e83482376a6a1c50424b80db4bbbd6d2eb5517300e3868bc3'
PARENT_MANIFEST_SHA = 'da2a512a4d70bee10051ad1c152467d95d397ed8829cd4d9eb1357d861a8ca92'
ORIGINAL_PROTOCOL_SHA = 'ca0061a6861a9f0ddb2694d9277cffff142909db69eb8b75a6316ab85bb328dd'
TASKS = ('reach','reach-wall')
ARMS = ('native','fixed_rank4','matched_random_fixed_rank4')
EDITS = ARMS[1:]
EPISODES = tuple(range(4,32))
EXPECTED = {(task,episode) for task in TASKS for episode in EPISODES}
MODALITIES = ('visual','proprio','weighted')
PRIMARY = ('terminal_ee_distance','actual_weighted_goal_cost','same_native_prefix_forecast_weighted_mse')
PACKAGES = {'torch':'2.7.1+cu128','numpy':'2.2.6','mujoco':'3.3.0','metaworld':'3.1.1',
            'gym':'0.23.1','gymnasium':'1.3.0','tensordict':'0.9.1'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def parent_helper():
    path=Path(__file__).with_name('cem_expansion_summary.py')
    require(sha(path)==PARENT_HELPER_SHA,'Frozen parent validation helper changed')
    spec=importlib.util.spec_from_file_location('planned_prefix_parent_helper',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def protocol():
    path=DATA/'planned_prefix_protocol.json'
    require(sha(path)==PROTOCOL_SHA,'Physical protocol changed')
    value=read(path)
    require(value['episode_ids']==list(EPISODES) and value['cases']==56 and
        value['model_arms']==list(ARMS) and value['plan_sources']==list(ARMS),'Registered cohort/grid changed')
    require(value['bootstrap_seed']==20260913 and value['bootstrap_replicates']==20000,'Statistical registry changed')
    return value


def freeze(path,prior_freeze=None):
    protocol();parent_helper()
    require(sha(ROOT/'src/offline_study/planned_prefix_replay.py')==RUNNER_SHA,'Physical runner changed')
    receipt={'status':'analysis_frozen_before_physical_prefix_outcomes',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'analysis_source_sha256':sha(__file__),
        'execution_source_sha256':RUNNER_SHA,'parent_helper_sha256':PARENT_HELPER_SHA,
        'protocol_sha256':PROTOCOL_SHA,'parent_execution_manifest_sha256':PARENT_MANIFEST_SHA,
        'cases':56,'n_per_task':28,'primary_metrics':list(PRIMARY),'primary_family':12,
        'bootstrap_replicates':20000,'bootstrap_seed':20260913,
        'quantiles':{'marginal':[.025,.975],'family12':[.05/24,1-.05/24]}}
    if prior_freeze is not None:
        require(sha(prior_freeze)==PRIOR_ANALYSIS_FREEZE_SHA,'Prior physical analysis freeze changed')
        prior=read(prior_freeze)
        require(prior['primary_metrics']==receipt['primary_metrics'] and prior['primary_family']==12 and
            prior['protocol_sha256']==PROTOCOL_SHA,'Amendment cannot change registered science')
        receipt.update(status='provenance_refrozen_before_successful_physical_case_or_outcome_inspection',
            amendment_of_analysis_freeze_sha256=PRIOR_ANALYSIS_FREEZE_SHA,
            amendment='Private RNG plumbing for exact degenerate upstream preprocessing; original failed v1 retained',
            statistical_implementation_and_primary_family_unchanged=True)
    with Path(path).open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    return receipt


def validate_freeze(path):
    value=read(path)
    for name,expected in [('analysis_source_sha256',sha(__file__)),('execution_source_sha256',RUNNER_SHA),
        ('parent_helper_sha256',PARENT_HELPER_SHA),('protocol_sha256',PROTOCOL_SHA),
        ('parent_execution_manifest_sha256',PARENT_MANIFEST_SHA)]:
        require(value[name]==expected,'Analysis freeze mismatch: '+name)
    require(value['primary_metrics']==list(PRIMARY) and value['primary_family']==12,'Primary family changed')
    return value


def scalar(value,nonnegative=True):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value),'Nonfinite/nonnumeric metric')
    require(not nonnegative or value>=0,'Negative squared error/distance')
    return float(value)


def triplet(value):
    require(set(value)==set(MODALITIES),'Incomplete modality metric')
    out={key:scalar(value[key]) for key in MODALITIES}
    require(out['weighted']==out['visual']+.1*out['proprio'],'Weighted metric is not visual +0.1 proprio')
    return out


def prefix_sha(value):
    array=np.asarray(value,dtype=np.float32)
    require(array.shape==(3,20) and np.isfinite(array).all(),'Invalid parent selected prefix')
    # Exact decision_runtime.tensor_hash convention, independently reproduced.
    return hashlib.sha256(b'torch.float32'+b'torch.Size([3, 20])'+array.tobytes(order='C')).hexdigest()


def extract_case(payload,key,parent_payload):
    task,episode=key
    require(key in EXPECTED,'Unregistered physical case')
    require(payload['forecast_horizon_scored']==3 and payload['CEM_optimized_horizon']==6 and
            payload['executed_elementary_steps_per_prefix']==15,'Horizon/physical prefix contract changed')
    metrics=payload['metrics']
    require(set(metrics['physical'])==set(ARMS) and set(metrics['forecast'])==set(ARMS) and
        set(metrics['preferences'])==set(EDITS),'Incomplete model/prefix grid')
    initial=triplet(metrics['initial_goal_cost'])
    physical=[];forecasts=[];preferences=[];primary=[]
    for arm in ARMS:
        row=metrics['physical'][arm]
        require(row['elementary_steps']==15 and type(row['prefix_success']) is bool,'Invalid physical prefix length/success flag')
        require(row['selected_prefix_sha256']==prefix_sha(parent_payload['traces'][arm]['selected_plan']),
                'Executed normalized prefix differs from actual parent CEM plan')
        for name in ('executed_actions_sha256','endpoint_observation_sha256','endpoint_physics_sha256'):
            require(re.fullmatch('[0-9a-f]{64}',row[name]) is not None,'Missing physical identity: '+name)
        cost=triplet(row['actual_goal_cost'])
        physical.append(dict(task=task,episode=episode,plan_arm=arm,
            terminal_ee_distance=scalar(row['terminal_ee_distance']),initial_ee_distance=scalar(row['initial_ee_distance']),
            physical_ee_progress=row['initial_ee_distance']-row['terminal_ee_distance'],
            **{'actual_goal_cost_'+k:v for k,v in cost.items()},
            actual_weighted_goal_progress=initial['weighted']-cost['weighted'],
            prefix_success=row['prefix_success'],selected_prefix_sha256=row['selected_prefix_sha256'],
            executed_actions_sha256=row['executed_actions_sha256'],
            endpoint_observation_sha256=row['endpoint_observation_sha256'],endpoint_physics_sha256=row['endpoint_physics_sha256']))
    require(len({row['initial_ee_distance'] for row in physical})==1,'Physical plans started at different distances')
    for model in ARMS:
        require(set(metrics['forecast'][model])==set(ARMS),'Incomplete three-prefix forecast row')
        for plan in ARMS:
            row=metrics['forecast'][model][plan]
            error=triplet(row['actual_endpoint_mse']);cost=triplet(row['predicted_goal_cost'])
            actual_cost=metrics['physical'][plan]['actual_goal_cost']
            forecasts.append(dict(task=task,episode=episode,model_arm=model,plan_arm=plan,
                **{'actual_endpoint_mse_'+k:v for k,v in error.items()},
                **{'predicted_goal_cost_'+k:v for k,v in cost.items()},
                **{'actual_goal_cost_'+k:actual_cost[k] for k in MODALITIES},
                **{'predicted_minus_actual_goal_cost_'+k:cost[k]-actual_cost[k] for k in MODALITIES}))
    for arm in EDITS:
        declared=metrics['preferences'][arm]
        pred=metrics['forecast'][arm][arm]['predicted_goal_cost']['weighted']-metrics['forecast'][arm]['native']['predicted_goal_cost']['weighted']
        actual=metrics['physical'][arm]['actual_goal_cost']['weighted']-metrics['physical']['native']['actual_goal_cost']['weighted']
        require(declared=={'predicted_H3_own_minus_native':pred,'actual_H3_own_minus_native':actual,
            'preference_reversal':pred<0 and actual>0,'predicted_tie':pred==0,'actual_tie':actual==0},
            'Declared preference/tie differs from complete crossed scores')
        preferences.append(dict(task=task,episode=episode,arm=arm,**declared,
            predicted_preference_sign=int(np.sign(pred)),actual_preference_sign=int(np.sign(actual)),
            reverse_direction_disagreement=pred>0 and actual<0,
            non_tied_agreement=pred*actual>0,either_tie=pred==0 or actual==0))
        edited_values=[metrics['physical'][arm]['terminal_ee_distance'],metrics['physical'][arm]['actual_goal_cost']['weighted'],
            metrics['forecast'][arm]['native']['actual_endpoint_mse']['weighted']]
        native_values=[metrics['physical']['native']['terminal_ee_distance'],metrics['physical']['native']['actual_goal_cost']['weighted'],
            metrics['forecast']['native']['native']['actual_endpoint_mse']['weighted']]
        primary.extend(dict(task=task,episode=episode,arm=arm,metric=name,edited=edit,native=native,effect=edit-native)
                       for name,edit,native in zip(PRIMARY,edited_values,native_values))
    return dict(physical_cases=physical,forecast_cases=forecasts,preference_cases=preferences,primary_cases=primary)


def validate_coverage(frames):
    designs={'physical_cases':(['plan_arm'],[(a,) for a in ARMS]),
        'forecast_cases':(['model_arm','plan_arm'],[(a,b) for a in ARMS for b in ARMS]),
        'preference_cases':(['arm'],[(a,) for a in EDITS]),
        'primary_cases':(['arm','metric'],[(a,m) for a in EDITS for m in PRIMARY])}
    for name,(columns,crosses) in designs.items():
        frame=frames[name];keys=['task','episode']+columns
        expected={(t,e,*cross) for t,e in EXPECTED for cross in crosses}
        require(not frame.duplicated(keys).any() and set(frame[keys].itertuples(index=False,name=None))==expected,
                'Complete56 exact-case coverage required: '+name)


def weights(task,replicates=20000,seed=20260913):
    rng=np.random.default_rng(np.random.SeedSequence([seed,TASKS.index(task)]))
    return rng.multinomial(28,np.full(28,1/28),size=replicates)/28


def statistics(values,bootstrap_weights,primary=False):
    values=np.asarray(values,dtype=float)
    require(values.shape==(28,) and np.isfinite(values).all() and bootstrap_weights.shape[1]==28,'Scenario units must be n28')
    draws=bootstrap_weights@values
    low,high=np.quantile(draws,[.025,.975]);sd=float(values.std(ddof=1))
    corrected=np.quantile(draws,[.05/24,1-.05/24]) if primary else (None,None)
    return dict(n=28,mean=float(values.mean()),scenario_sd=sd,scenario_se=sd/np.sqrt(28),
        marginal_95_low=float(low),marginal_95_high=float(high),bonferroni_family=12 if primary else None,
        bonferroni_95_low=None if corrected[0] is None else float(corrected[0]),
        bonferroni_95_high=None if corrected[1] is None else float(corrected[1]),
        inference_scope='prespecified_primary_family12' if primary else 'descriptive_only')


def summarize(frames,replicates=20000,seed=20260913):
    validate_coverage(frames)
    w={task:weights(task,replicates,seed) for task in TASKS}
    def aggregate(frame,groups,metrics,primary=False):
        rows=[]
        for group,block in frame.groupby(groups,sort=True):
            if not isinstance(group,tuple):group=(group,)
            identity=dict(zip(groups,group));block=block.sort_values('episode')
            require(tuple(block.episode)==EPISODES,'Duplicate/missing scenario within metric cell')
            for metric in metrics:
                rows.append(dict(**identity,**({} if 'metric' in identity else {'metric':metric}),
                    **statistics(block[metric].to_numpy(dtype=float),w[identity['task']],primary)))
        return pd.DataFrame(rows)
    physical_metrics=['terminal_ee_distance','initial_ee_distance','physical_ee_progress',
        *('actual_goal_cost_'+k for k in MODALITIES),'actual_weighted_goal_progress','prefix_success']
    forecast_metrics=[prefix+k for prefix in ('actual_endpoint_mse_','predicted_goal_cost_',
        'actual_goal_cost_','predicted_minus_actual_goal_cost_') for k in MODALITIES]
    preference_metrics=['predicted_H3_own_minus_native','actual_H3_own_minus_native','preference_reversal',
        'predicted_tie','actual_tie','either_tie','reverse_direction_disagreement','non_tied_agreement']
    result=dict(frames)
    result['primary_summary']=aggregate(frames['primary_cases'],['task','arm','metric'],['effect'],True)
    result['physical_summary']=aggregate(frames['physical_cases'],['task','plan_arm'],physical_metrics)
    result['forecast_summary']=aggregate(frames['forecast_cases'],['task','model_arm','plan_arm'],forecast_metrics)
    result['preference_summary']=aggregate(frames['preference_cases'],['task','arm'],preference_metrics)
    contingency=[]
    for (task,arm),block in frames['preference_cases'].groupby(['task','arm']):
        for pred in (-1,0,1):
            for actual in (-1,0,1):
                count=int(((block.predicted_preference_sign==pred)&(block.actual_preference_sign==actual)).sum())
                contingency.append(dict(task=task,arm=arm,predicted_preference_sign=pred,
                    actual_preference_sign=actual,count=count,n=28,fraction=count/28))
    result['preference_contingency']=pd.DataFrame(contingency)
    return result


def complete_paths(root):
    paths={(t,e):Path(root)/t/f'episode-{e}' for t,e in EXPECTED}
    required=('report.json','DONE.json','CLOUD_VERIFIED.json','prefix-summary.json')
    require(all((path/name).is_file() for path in paths.values() for name in required),
            'Require all56 complete source/DONE/cloud files before any physical outcome read')
    found={(p.parent.parent.name,int(p.parent.name.split('-')[-1])) for p in Path(root).glob('*/episode-*/report.json')}
    require(found==EXPECTED,'Unexpected physical case directories')
    return paths


def member(cloud,key,name,digest):
    matches=[value for path,value in cloud['verified_archive_members'].items()
        if path==name or path.endswith(f'/{key[0]}/episode-{key[1]}/{name}')]
    require(matches==[digest],'Missing unique original archive member: '+name)


def validate_metadata(directory,key,manifest,manifest_sha,binding,gpu,parent_report,parent_cloud,parent_directory,original_root,helper):
    report,done,cloud=(read(directory/name) for name in ('report.json','DONE.json','CLOUD_VERIFIED.json'))
    require(report['status']=='complete_development_physical_prefix_case' and report['input_binding']==binding,'Physical identity/status changed')
    require(report['execution_manifest_sha256']==manifest_sha and report['protocol_sha256']==PROTOCOL_SHA,'Physical source freeze changed')
    require(done['report_sha256']==sha(directory/'report.json') and done['files']==report['files'],'Physical DONE/report changed')
    require(cloud.get('gcs_download_sha256_verified') is True and cloud.get('all_report_files_hash_verified') is True
        and cloud.get('raw_preservation_pending') is False,'Complete physical raw preservation required')
    require(str(cloud['cloud_uri']).startswith('gs://') and bool(cloud['generation']) and
        re.fullmatch('[0-9a-f]{64}',cloud['sha256']) is not None,'Missing generation-pinned SHA archive proof')
    for name in ('report.json','DONE.json','prefix-summary.json'):
        require(cloud['compact_sha256'][name]==sha(directory/name),'Physical compact hash changed: '+name)
    require(set(report['files'])=={'STARTED.json','physical-prefix.pt','prefix-summary.json'},'Missing/extra original physical payload')
    require(report['files']['prefix-summary.json']==sha(directory/'prefix-summary.json'),'Unbound physical compact')
    for name,digest in report['files'].items():member(cloud,key,name,digest)
    qualification={'task':key[0],'episode':key[1]} in manifest['tail_qualification_cases']
    parity={'exact_original_reset_all_four_trajectories':True,'native_physical_repeat_byte_equal':True,
        'forecast_phase_global_rng_unchanged':True,'simulator_phase_global_rng_guarded':False,
        'inputs_unchanged':True,'parameters_unchanged':True,'future_tail_H3_byte_equal':True if qualification else None,
        'future_tail_qualification_case':qualification}
    require(report['parities']==parity,'Missing exact reset/repeat/causal-address/RNG parity')
    require(report['preprocessing_rng_audit']=={
        'method':'exact_upstream_degenerate_crop_private_function_globals','full_frame_reference_byte_equal':True,
        'global_rng_unchanged':True,'calls':5,'global_rng_restored':False,'original_spatial_transform_restored':True},
        'Missing exact-reference private preprocessing RNG receipt')
    for name,value in [('physical_trajectories',4),('total_elementary_actions',60),('scientific_forecast_calls',9),
        ('engineering_forecast_calls',3*int(qualification)),('forecast_horizon_scored',3),('CEM_optimized_horizon',6)]:
        require(report[name]==value,'Physical forecast/execution count changed: '+name)
    require(report['physical_outcomes_measured'] is True and report['full_task_success_measured'] is False
        and report['fresh_confirmation'] is False,'Physical evidence scope mismatch')
    require(report['gpu_uuid']==gpu and report['package_versions']==PACKAGES,'Physical receiver/package mismatch')
    require(report['tf32_matmul'] is False and report['tf32_cudnn'] is False,'Physical TF32 forbidden')
    backend=report['backend_provenance']
    for name,value in [('checkpoint_sha256',helper.CHECKPOINT_SHA),('dino_source_sha256',helper.DINO_SOURCE_SHA),
        ('dino_weights_sha256',helper.DINO_WEIGHTS_SHA),('precision','float32'),('allow_tf32',False),
        ('dino_loader_source','verified_local_cache_no_network_branch_resolution')]:
        require(backend[name]==value,'Physical frozen encoder changed: '+name)
    require(report['fit_bank_sha256']==helper.FIT_SHA[key[0]],'Physical task fit changed')
    parent=report['cem_parent']
    require(parent['execution_manifest_sha256']==PARENT_MANIFEST_SHA and parent['cloud_receipt']==parent_cloud,'Physical parent archive changed')
    for keyname,filename in [('report_sha256','report.json'),('done_sha256','DONE.json'),
        ('summary_sha256','steered-cem-summary.json'),('cloud_receipt_sha256','CLOUD_VERIFIED.json')]:
        require(parent[keyname]==sha(parent_directory/filename),'Physical CEM parent hash mismatch: '+filename)
    require(parent['source_trace_sha256']=={arm:parent_report['files'][arm+'-cem-trace.pt'] for arm in ARMS},'Parent trace hashes changed')
    rb=manifest['original_records'][f'{key[0]}/{key[1]}']
    require(report['original_record_binding']==rb,'Original reset-record manifest binding changed')
    original=Path(original_root)/key[0]/f'episode-{key[1]:03d}'
    require(sha(original/'record.json')==rb['record_sha256'] and sha(original/'DONE.json')==rb['done_sha256'],'Original reset/goal source changed')
    original_done=read(original/'DONE.json')
    require(original_done['files']['record.json']==rb['record_sha256'] and
        original_done['files']['inputs.pt']==binding['inputs_sha256'],'Original reset/input DONE mismatch')
    return report,cloud


def run(args):
    protocol();freeze_receipt=validate_freeze(args.analysis_freeze);helper=parent_helper()
    require(sha(args.execution_manifest)==args.execution_manifest_sha256,'Physical execution manifest hash changed')
    manifest=read(args.execution_manifest)
    require(manifest['protocol_sha256']==PROTOCOL_SHA and manifest['scenarios']=={t:list(EPISODES) for t in TASKS},'Physical registry changed')
    require(manifest['model_arms']==list(ARMS) and manifest['plan_sources']==list(ARMS),'Physical arm registry changed')
    require(manifest['checkpoint_sha256']==helper.CHECKPOINT_SHA and manifest['fit_bank_sha256']==helper.FIT_SHA
        and manifest['input_manifest_sha256']==helper.INPUT_MANIFEST_SHA==sha(args.input_manifest),'Physical frozen assets changed')
    require(manifest['cem_execution_manifest_sha256']==PARENT_MANIFEST_SHA and
        manifest['original_protocol_sha256']==ORIGINAL_PROTOCOL_SHA and manifest['package_versions']==PACKAGES,'Physical parents/runtime changed')
    require(manifest['source_sha256']['planned_prefix_replay.py']==RUNNER_SHA,'Physical runner differs from analysis freeze')
    require(sha(args.parent_manifest)==PARENT_MANIFEST_SHA,'Parent CEM execution freeze changed')
    parent_manifest=read(args.parent_manifest)
    for registry in (manifest,parent_manifest):
        require(registry['source_sha256'] and registry['vendor_source_sha256'],'Missing exact local/vendor source bindings')
        for name,digest in registry['source_sha256'].items():
            require(Path(name).name==name and sha(ROOT/'src/offline_study'/name)==digest,'Pinned local source changed: '+name)
        for name,digest in registry['vendor_source_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                sha(ROOT/'vendor/jepa-wms'/name)==digest,'Pinned vendor source changed: '+name)
    bindings={(r['task'],r['episode']):r for r in read(args.input_manifest)['records']}
    gpus=helper.assigned_gpus(manifest);parent_gpus=helper.assigned_gpus(parent_manifest)
    require(set(manifest['original_records'])=={f'{t}/{e}' for t,e in EXPECTED},'Incomplete original reset binding registry')
    qualifiers={(r['task'],r['episode']) for r in manifest['tail_qualification_cases']}
    require(qualifiers and qualifiers<=EXPECTED and len(qualifiers)==len(manifest['tail_qualification_cases']),'Invalid receiving qualification registry')
    paths=complete_paths(args.compact_root)
    parent_paths=helper.case_paths(args.parent_compact_root,'extension56')
    helper.require_complete_files(parent_paths,'steered-cem-summary.json')
    metadata={}
    # Every physical and parent source/receipt passes before any physical scalar payload opens.
    for key,path in sorted(paths.items()):
        parent_report,parent_cloud,_=helper.read_bound_case(parent_paths[key],'steered-cem-summary.json',True)
        helper.validate_extension(parent_report,parent_cloud,None,key,parent_manifest,PARENT_MANIFEST_SHA,bindings[key],parent_gpus[key])
        metadata[key]=validate_metadata(path,key,manifest,args.execution_manifest_sha256,bindings[key],gpus[key],
            parent_report,parent_cloud,parent_paths[key],args.original_records,helper)
    rows={name:[] for name in ('physical_cases','forecast_cases','preference_cases','primary_cases')};sources=[]
    for key,path in sorted(paths.items()):
        report,cloud=metadata[key];payload=read(path/'prefix-summary.json')
        require(payload['input_binding']==report['input_binding'] and payload['execution_manifest_sha256']==args.execution_manifest_sha256
            and payload['protocol_sha256']==PROTOCOL_SHA and payload['parities']==report['parities'],'Physical compact identity/parity changed')
        parent_payload=read(parent_paths[key]/'steered-cem-summary.json')
        extracted=extract_case(payload,key,parent_payload)
        for name,values in extracted.items():rows[name].extend(values)
        sources.append(dict(task=key[0],episode=key[1],report_sha256=sha(path/'report.json'),done_sha256=sha(path/'DONE.json'),
            compact_sha256=sha(path/'prefix-summary.json'),cloud_receipt_sha256=sha(path/'CLOUD_VERIFIED.json'),
            cloud_uri=cloud['cloud_uri'],cloud_generation=cloud['generation'],cloud_archive_sha256=cloud['sha256'],
            original_files_sha256=report['files'],input_binding=report['input_binding'],gpu_uuid=report['gpu_uuid'],
            cem_parent_report_sha256=report['cem_parent']['report_sha256'],original_record_binding=report['original_record_binding'],
            parities=report['parities']))
    frames=summarize({name:pd.DataFrame(values) for name,values in rows.items()})
    outputs={}
    for name,frame in frames.items():
        path=DATA/f'planned_prefix_{name}.csv';frame.to_csv(path,index=False)
        outputs[str(path.relative_to(ROOT))]={'sha256':sha(path),'rows':len(frame)}
    result={'status':'complete56_development_physical_prefix_analysis','cases':56,'n_per_task':28,
        'analysis_source_sha256':sha(__file__),'analysis_freeze_sha256':sha(args.analysis_freeze),
        'analysis_frozen_at_utc':freeze_receipt['frozen_at_utc'],'execution_source_sha256':RUNNER_SHA,
        'execution_manifest_sha256':args.execution_manifest_sha256,'protocol_sha256':PROTOCOL_SHA,
        'parent_execution_manifest_sha256':PARENT_MANIFEST_SHA,'parent_helper_sha256':PARENT_HELPER_SHA,
        'primary_family':12,'bootstrap_replicates':20000,'bootstrap_seed':20260913,
        'physical_trajectories':224,'scientific_forecast_calls':504,'engineering_forecast_calls':3*len(qualifiers),
        'forecast_horizon_scored':3,'CEM_optimized_horizon':6,'sources':sources,'outputs':outputs,
        'all_full_raw_preservation_verified':True,'physical_outcomes_measured':True,'full_task_success_measured':False,
        'fresh_confirmation':False,'cpu_recomputed':['paired contrasts','preference signs/ties','scenario statistics and intervals'],
        'execution_attested':['exact reset and full physical native repeat','H3 encoder/forecast errors','frozen-model and forecast-phase RNG parity'],
        'caveats':['Development states, not protected confirmation','H3 prefix objective differs from optimized H6 objective',
            'Prefix success is not full100-step task success','Preference disagreement alone is not proof of planner model exploitation',
            'Simulation resets intentionally change RNG; strict RNG guard covers encoding/forecasts only',
            'Large physical arrays remain in generation-pinned verified cloud archives; CPU aggregates compact scalar metrics']}
    (DATA/'planned_prefix_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--amend-freeze',type=Path)
    parser.add_argument('--analysis-freeze',type=Path,default=DATA/'planned_prefix_analysis_freeze.json')
    parser.add_argument('--execution-manifest',type=Path)
    parser.add_argument('--execution-manifest-sha256')
    parser.add_argument('--compact-root',type=Path)
    parser.add_argument('--parent-manifest',type=Path,default=BASE/'cem-expansion-execution-manifest-v1.json')
    parser.add_argument('--parent-compact-root',type=Path,default=BASE/'compact/cem-expansion-v1')
    parser.add_argument('--input-manifest',type=Path,default=BASE/'development-inputs-64/INPUT_MANIFEST.json')
    parser.add_argument('--original-records',type=Path,default=ROOT/'artifacts/offline_study/decision-diagnostic-20260910-v2-retry1')
    args=parser.parse_args()
    result=freeze(args.freeze,args.amend_freeze) if args.freeze else run(args)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','outputs')},indent=2))
