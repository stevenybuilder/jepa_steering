"""Paired, finite HMM behavioral panel; no old solver or confirmation access."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .author_fit import source_hash
from .backends import JepaBackend
from .behavioral_analysis import scenario_key,shard_paths
from .behavioral_development import DEVELOPMENT_SEED,assigned_rows,schedule,validate_coverage,verified_report
from .fixed_response import load_fitted_bank
from .fixed_response_behavior import checked_source,device_uuid,stimulus_contract,validate_protocol,verify_cem
from .fixed_response_check import CountedBackend,reference_fields
from .fixed_response_smoke import trace_actor,verify_episode,verify_pair
from .intervention_runner import _model_versions
from .interventions import PredictorIntervention
from .planning_contract import prepare
from .planning_native_smoke import CHECKPOINTS,SMOKE_SEED,run_episode
from .planning_scenarios import close_expert_environments
from .protocol import sha256,write_json
from .routing_fit import METHOD,load_routing_fit
from .routing_hmm import route_prefix
from .routing_intervention import ARMS,GATE_KEYS,RoutedFixedResponse
from .routing_one_pass import RoutedFixedResponseOnePass
from .vendor import use_vendor

TASKS=('reach','reach-wall')
REUSED={'native':'native','constant_gate':'fixed_rank4','matched_random_constant_gate':'matched_random_fixed_rank4'}
NEW_ARMS=tuple(a for a in ARMS if a not in REUSED)
GATES=('constant_gate','memoryless_gate','hmm_filtered_gate')
CONTRASTS=(('hmm_filtered_gate','memoryless_gate'),('hmm_filtered_gate','constant_gate'),
    *((g,'native') for g in GATES),*((g,'matched_random_'+g) for g in GATES))
ENGINEERING=('native','native_repeat','zero_dose')+ARMS[1:]
ROLE='paired_native_history_routing_fixed_response_development'


def prepared(args):
    baseline=validate_protocol(args.reference_freeze)
    checked_source(args.reference_code,baseline['source_sha256'])
    tasks={}
    for task in TASKS:
        fixed=args.fits/task; root=args.routing/task
        load_fitted_bank(fixed,task='mw-'+task,checkpoint_sha256=CHECKPOINTS['metaworld'])
        _,fit=load_routing_fit(root,fixed)
        _,_,stimuli=stimulus_contract(args.stimuli,task)
        planning=prepare(args.vendor,task);planning['config']['meta']['seed']=DEVELOPMENT_SEED
        old=baseline['tasks'][task]
        if (fit['task']!='mw-'+task or planning!=old['planning'] or stimuli!=old['stimuli'] or
                sha256(fixed/'DONE.json')!=old['fit_done_sha256'] or
                sha256(fixed/'operator_bank.pt')!=old['fit_bank_sha256']):
            raise ValueError('HMM/refined reference task,bank or canonical stimuli changed')
        tasks[task]={'planning':planning,'stimuli':stimuli,'fixed_done_sha256':sha256(fixed/'DONE.json'),
            'fixed_bank_sha256':sha256(fixed/'operator_bank.pt'),'routing_done_sha256':sha256(root/'DONE.json'),
            'routing_model_sha256':sha256(root/'model.pt'),'routing_protocol_sha256':sha256(root/'protocol.json')}
    return {'role':ROLE,'method':METHOD,'source_sha256':source_hash(),'tasks':tasks,'arms':list(ARMS),
        'reference_freeze_sha256':sha256(args.reference_freeze/'protocol.json'),
        'reference_source_sha256':baseline['source_sha256'],'reused_arm_mapping':REUSED,
        'reused_records_keep_original_labels_and_runtime':True,'new_arms':list(NEW_ARMS),
        'episodes':schedule(),'episodes_per_task_condition_total':96,'new_episodes_total':768,
        'reference_episodes_reused_total':576,'checkpoint_sha256':CHECKPOINTS['metaworld'],
        'precision':'float32_strict_no_tf32','development_base_seed':DEVELOPMENT_SEED,
        'primary_contrasts_per_task':[list(c) for c in CONTRASTS],
        'analysis':{'complete_two_task_seven_arm_panel_required':True,'family':16,'replicates':20000,
            'seed':2026090803,'intervals':'paired whole-scenario cluster bootstrap Bonferroni95%',
            'minimum_useful_success_gain_percentage_points':5,'selection_from_partial_results':False},
        'engineering_episode_order':list(ENGINEERING),'native_shadow_counted':True,
        'runtime_implementation':'causal_same_pass_v1',
        'scientific_native_shadow_rollouts':0,'scientific_backend_calls_per_forecast':1,
        'engineering_requires_exact_two_pass_prediction_gate_parity_on_every_call':True,
        'fit_history_diagnostic_was_negative_not_a_behavioral_veto':True,
        'conditional_history_eligibility_claimed':False,'fresh_confirmation':False,
        'confirmation_access_authorized':False,'training_histories_complete':False}


def read_protocol(args):
    expected=prepared(args); path=args.freeze/'protocol.json'
    if json.loads(path.read_text())!=expected or json.loads((args.freeze/'FROZEN.json').read_text())['protocol_sha256']!=sha256(path):
        raise ValueError('Routed behavioral freeze/source/bindings changed')
    return expected


class FieldReference:
    """Independent passive capture; only engineering can also read native H3."""
    def __init__(self,predictor):
        self.predictor=predictor;self.horizon=0;self.values={}
    def __enter__(self):
        self.handles=[self.predictor.register_forward_pre_hook(self._input),
            self.predictor.predictor_blocks[3].register_forward_hook(self._output)]
        return self
    def _input(self,module,args):self.horizon+=1
    def _output(self,module,args,output):
        if self.horizon in (1,2,3):self.values[self.horizon]=output[:,-256:].detach().clone()
    def __exit__(self,kind,exc,tb):
        for handle in self.handles:handle.remove()
        if kind is None and (self.horizon!=6 or set(self.values)!={1,2,3}):
            raise ValueError('Incomplete independent reference field capture')


class ObservedRouting:
    def __init__(self,adapter,counted,output,engineering=False):
        self.adapter,self.counted,self.output=adapter,counted,output
        self.engineering=engineering;self.checked=set();self.calls=[]
        self.one_pass=isinstance(adapter,RoutedFixedResponseOnePass)
        self.shadow_reference=RoutedFixedResponse(counted,adapter.bank,adapter.routing,adapter.arm) if engineering and self.one_pass else None
    def __call__(self,context,act_suffix=None,**kwargs):
        horizon,count,_=act_suffix.shape;started=time.monotonic()
        write_json(self.output/'progress.json',{'completed_calls':len(self.calls),
            'running_horizon':horizon,'running_candidates':count,'engineering_only':self.engineering})
        before=self.counted.calls
        result=self.adapter(context,act_suffix,**kwargs)
        actual_calls=self.counted.calls-before
        active=horizon==6 and self.adapter.arm not in ('native','zero_dose')
        expected_calls=1 if self.one_pass else (2 if active else 1)
        if actual_calls!=expected_calls:raise ValueError('Wrong native-shadow/forecast budget')
        shadow_parity=False
        if self.shadow_reference is not None:
            expected=self.shadow_reference(context,act_suffix,**kwargs)
            if any(not torch.equal(result[k],expected[k]) for k in ('visual','proprio')):
                raise ValueError('Same-pass forecast differs from frozen two-pass implementation')
            if active:
                for key in ('normalized_gate','raw_gate','posterior','gate_normalizer','coefficients','requested_l2','realized_l2','active'):
                    if not torch.equal(self.adapter.last_record[key],self.shadow_reference.last_record[key]):
                        raise ValueError('Same-pass routing/field differs from frozen two-pass implementation: '+key)
            shadow_parity=True
        parity=False
        if self.engineering and horizon==6 and count not in self.checked:
            with FieldReference(self.counted.predictor) as captured:
                native=self.counted.predict(context,act_suffix)
            if active:
                route=route_prefix(torch.stack([captured.values[h].float().mean(1) for h in (1,2)],1),self.adapter.routing['model'])
                key=GATE_KEYS[self.adapter.arm.removeprefix('matched_random_')]
                gate=(route[key]/self.adapter.routing['normalizers'][key]).float()
                bank={**self.adapter.bank,'dose':self.adapter.bank['dose']*gate}
                arm='matched_random_fixed_rank4' if self.adapter.arm.startswith('matched_random_') else 'fixed_rank4'
                fields=reference_fields(captured.values[3],bank,arm)
                with PredictorIntervention(self.counted.predictor,fields):expected=self.counted.predict(context,act_suffix)
            else:expected=native
            if any(not torch.equal(result[k],expected[k]) for k in ('visual','proprio')):
                raise ValueError('Independent static compiler/zero parity failed')
            self.checked.add(count);parity=True
        torch.cuda.synchronize()
        energy={k:v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v for k,v in self.adapter.last_record.items()}
        if energy['backend_calls']!=expected_calls or energy['future_routing_features_read'] or energy['response_probe_rollouts']:
            raise ValueError('Routing causality or deployed probe budget changed')
        self.calls.append({'horizon':horizon,'candidates':count,'backend_calls':actual_calls,
            'seconds':time.monotonic()-started,'energy':energy,'independent_reference_check':parity,
            'two_pass_equivalence_checked':shadow_parity,
            'engineering_extra_reference_calls':self.counted.calls-before-actual_calls})
        return result


def verify_routed_calls(result,calls,arm,engineering=False):
    verify_episode(result,calls)
    for call in calls:
        active=call['horizon']==6 and arm not in ('native','zero_dose')
        energy=call['energy']
        if (call['backend_calls']!=1 or energy.get('runtime_implementation')!='causal_same_pass_v1' or
                energy['native_shadow_rollouts']!=0 or energy['response_probe_rollouts']!=0 or
                energy['future_routing_features_read'] is not False or
                energy['routing_horizons']!=([1,2] if active else []) or
                call.get('two_pass_equivalence_checked') is not engineering or
                (not engineering and call['engineering_extra_reference_calls']!=0)):
            raise ValueError('Changed routed model work/causality trace')
    if engineering and {c['candidates'] for c in calls if c['independent_reference_check']}!={1,300}:
        raise ValueError('Missing single-mean/full-population independent parity')


def verify_engineering(root,protocol,task,freeze_hash):
    report,digest=verified_report(root);launch=json.loads((root/'protocol.json').read_text())
    if (report['status']!='complete_routed_fixed_response_full_cem_engineering' or
            report['task']!=task or report['protocol_sha256']!=sha256(root/'protocol.json') or
            launch['freeze_sha256']!=freeze_hash or launch['source_sha256']!=protocol['source_sha256'] or
            report['device_uuid']!=launch['device_uuid'] or report['episodes']!=len(ENGINEERING) or
            report['parameters_unchanged'] is not True or report['scientific_efficacy_measurement'] is not False or
            set(report['episode_files_sha256'])!=set(ENGINEERING)):
        raise ValueError('Missing bound full routed engineering')
    rows={}
    for name in ENGINEERING:
        path=root/(name+'.json')
        if sha256(path)!=report['episode_files_sha256'][name]:raise ValueError('Engineering record changed')
        row=json.loads(path.read_text());arm='native' if name=='native_repeat' else name
        if row['arm']!=arm:raise ValueError('Engineering label changed')
        calls=root/name/'unroll_calls.json';trace=root/name/'action_trace.json'
        if sha256(calls)!=row['unroll_calls_sha256'] or sha256(trace)!=row['action_trace_sha256']:
            raise ValueError('Engineering raw trace changed')
        verify_routed_calls(row['result'],json.loads(calls.read_text()),arm,True)
        rows[name]={'result':row['result'],'action_trace':json.loads(trace.read_text())}
        if name!='native':verify_pair(rows['native'],rows[name],native_repeat=name in ('native_repeat','zero_dose'))
    return report,digest


def reference_records(args,protocol,task,arm,ranks=None):
    original_arm=REUSED[arm]; rows=[];bindings={}
    expected=assigned_rows(schedule(),list(range(8)) if ranks is None else ranks)
    wanted={r['episode'] for r in expected}
    for root in shard_paths(args.reference/task/original_arm):
        launch=json.loads((root/'protocol.json').read_text())
        assigned={r['episode'] for r in launch['expected_episodes']}
        if not assigned&wanted:continue
        report,digest=verified_report(root)
        if (report['status']!='fixed_response_behavioral_shard_complete' or report['task']!=task or
                report['arm']!=original_arm or report['protocol_sha256']!=sha256(root/'protocol.json') or
                launch['freeze_sha256']!=protocol['reference_freeze_sha256'] or
                launch['source_sha256']!=protocol['reference_source_sha256'] or
                report['parameters_unchanged'] is not True or report['fresh_confirmation'] is not False or
                report['episodes']!=len(assigned)):
            raise ValueError('Invalid completed original reference shard')
        worker=args.reference_workers/('gpu-'+root.name.removeprefix('shard-gpu'))
        proof=verify_cem(worker,args.fits/task,args.reference_code,task)
        worker_binding=json.loads((worker/'WORKER.json').read_text())
        if (proof!=launch['engineering_report_sha256'] or
                sha256(worker/'WORKER.json')!=launch['receiving_worker_sha256'] or
                worker_binding['device_uuid']!=launch['device_uuid']):
            raise ValueError('Original worker proof changed')
        for name,digest_file in report['episode_files_sha256'].items():
            if Path(name).name!=name or sha256(root/name)!=digest_file:raise ValueError('Reference episode changed')
            row=json.loads((root/name).read_text())
            calls_root=root/f"calls-{row['episode']:03d}"
            for filename,key in (('unroll_calls.json','unroll_calls_sha256'),('action_trace.json','action_trace_sha256')):
                if sha256(calls_root/filename)!=row[key]:raise ValueError('Reference raw trace changed')
            calls=json.loads((calls_root/'unroll_calls.json').read_text());verify_episode(row['result'],calls)
            if row['arm']!=original_arm or any(c['backend_calls']!=1 for c in calls):raise ValueError('Reference arm/work mismatch')
            if row['episode'] in wanted:
                rows.append({**row,'source_arm':original_arm,'arm':arm,'reference_record':str(root/name),
                    'reference_record_sha256':digest_file,'device_uuid':launch['device_uuid'],
                    'runtime_from_original_one_pass_not_routed_shadow':True})
        bindings[str(root/'report.json')]=digest
    rows.sort(key=lambda row:row['episode']);validate_coverage(rows,expected)
    return rows,bindings


@torch.no_grad()
def execute(args):
    protocol=read_protocol(args); engineering=args.mode=='engineer'; task=args.task
    bank=load_fitted_bank(args.fits/task,task='mw-'+task,checkpoint_sha256=CHECKPOINTS['metaworld'])
    routing,_=load_routing_fit(args.routing/task,args.fits/task)
    _,goals,_=stimulus_contract(args.stimuli,task)
    worker_id=device_uuid();worker=args.worker_reference
    old_proof=verify_cem(worker,args.fits/task,args.reference_code,task)
    worker_meta=json.loads((worker/'WORKER.json').read_text())
    if worker_meta['device_uuid']!=worker_id or worker_meta['cem_report_sha256']!=old_proof:
        raise ValueError('Current GPU differs from original reference receiver')
    proof_hash=None;native=None
    if not engineering:
        proof,proof_hash=verify_engineering(args.engineering,protocol,task,sha256(args.freeze/'protocol.json'))
        if proof['device_uuid']!=worker_id:raise ValueError('Missing routed proof on this GPU')
        rows,_=reference_records(args,protocol,task,'native',args.logical_ranks)
        native={row['episode']:row for row in rows}
        if any(row['device_uuid']!=worker_id for row in rows):raise ValueError('Cross-device reference reuse not authorized')
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();records={}
    expected=None if engineering else assigned_rows(schedule(),args.logical_ranks)
    write_json(args.output/'protocol.json',{'freeze_sha256':sha256(args.freeze/'protocol.json'),
        'source_sha256':source_hash(),'task':task,'arm':None if engineering else args.arm,
        'device_uuid':worker_id,'engineering_only':engineering,'engineering_report_sha256':proof_hash,
        'reference_worker_report_sha256':old_proof,'expected_episodes':expected,
        'logical_ranks':None if engineering else args.logical_ranks,'fresh_confirmation':False})
    try:
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        backend=JepaBackend(args.vendor,args.checkpoint,CHECKPOINTS['metaworld'],'metaworld','cuda:0','float32')
        if backend.model.ctxt_window!=2:raise ValueError('Wrong planning context')
        counted=CountedBackend(backend);versions=_model_versions(backend.model)
        states=random.getstate(),np.random.get_state(),torch.get_rng_state()
        reference_engineering={}
        for name in (ENGINEERING if engineering else (args.arm,)):
            arm='native' if name=='native_repeat' else name
            for rank in ([None] if engineering else sorted(args.logical_ranks)):
                if engineering:
                    random.seed(SMOKE_SEED);np.random.seed(SMOKE_SEED);torch.manual_seed(SMOKE_SEED)
                    rows=[{'episode':0,'logical_rank':0,'local_seed':SMOKE_SEED,'environment_seed':SMOKE_SEED}]
                else:
                    random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
                    rows=[r for r in expected if r['logical_rank']==rank]
                cfg=OmegaConf.create(protocol['tasks'][task]['planning']['config']);cfg.local_seed=rows[0]['local_seed']
                if engineering:cfg.meta.seed=SMOKE_SEED
                agent=GC_Agent(cfg,backend.model,preprocessor=backend.preprocessor);env=make_env(cfg)
                try:
                    for row in rows:
                        before=time.monotonic();calls=args.output/(name if engineering else f"calls-{row['episode']:03d}");calls.mkdir()
                        adapter=RoutedFixedResponseOnePass(counted,bank,routing,arm);observed=ObservedRouting(adapter,counted,calls,engineering)
                        agent.planner.unroll=observed;old_act=agent.act;trace=trace_actor(agent)
                        try:
                            with close_expert_environments(plan_evaluator):
                                if engineering:result=run_episode(cfg,backend,agent,env,SMOKE_SEED)
                                else:
                                    with goals.deliver(row):result=run_episode(cfg,backend,agent,env,row['environment_seed'])
                        finally:
                            agent.act=old_act;write_json(calls/'unroll_calls.json',observed.calls);write_json(calls/'action_trace.json',trace)
                        verify_routed_calls(result,observed.calls,arm,engineering)
                        initial=np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist()
                        current={'result':result,'action_trace':trace}
                        if engineering:
                            reference_engineering[name]=current
                            if name!='native':verify_pair(reference_engineering['native'],current,native_repeat=name in ('native_repeat','zero_dose'))
                            if arm in REUSED or arm=='zero_dose':
                                old_arm='native' if arm=='zero_dose' else REUSED[arm]
                                old_report,_=verified_report(worker/old_arm)
                                old_current={'result':old_report['result'],'action_trace':json.loads((worker/old_arm/'action_trace.json').read_text())}
                                verify_pair(old_current,current,native_repeat=True)
                        else:
                            metadata,_=goals.load(row)
                            baseline=native[row['episode']]
                            if (initial!=metadata['rand_vec'] or initial!=baseline['initial_state_vector'] or
                                    any(result[k]!=metadata[k] or result[k]!=baseline['result'][k] for k in ('initial_sha256','goal_sha256'))):
                                raise ValueError('Unpaired canonical goal/initial state')
                        record={**row,'arm':arm,'result':result,'initial_state_vector':initial,'device_uuid':worker_id,
                            'seconds':time.monotonic()-before,'unroll_calls_sha256':sha256(calls/'unroll_calls.json'),
                            'action_trace_sha256':sha256(calls/'action_trace.json')}
                        if not engineering:validate_coverage([record],[row])
                        filename=name+'.json' if engineering else f"episode-{row['episode']:03d}.json"
                        write_json(args.output/filename,record);records[name if engineering else filename]=record
                        progress={'task':task,'arm':arm,'completed':len(records),'target':len(ENGINEERING) if engineering else len(expected),
                            'engineering_only':engineering,'seconds':time.monotonic()-started}
                        write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
                finally:env.close()
        if versions!=_model_versions(backend.model) or source_hash()!=protocol['source_sha256']:raise ValueError('Frozen model/source changed')
        write_json(args.output/'report.json',{'status':'complete_routed_fixed_response_full_cem_engineering' if engineering else 'complete_routed_fixed_response_shard',
            'task':task,'arm':None if engineering else args.arm,'device_uuid':worker_id,
            'protocol_sha256':sha256(args.output/'protocol.json'),'episodes':len(records),
            'episode_files_sha256':{key:sha256(args.output/(key+'.json' if engineering else key)) for key in records},
            'parameters_unchanged':True,'scientific_efficacy_measurement':not engineering,'fresh_confirmation':False,
            'seconds':time.monotonic()-started})
        write_json(args.output/'DONE.json',{'report_sha256':sha256(args.output/'report.json')})
        if engineering:verify_engineering(args.output,protocol,task,sha256(args.freeze/'protocol.json'))
    except Exception as exc:
        write_json(args.output/'FAILED.json',{'error':str(exc),'partial_not_complete':True});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','engineer','run'))
    for name in ('vendor','fits','routing','stimuli','reference','reference-freeze','reference-code','reference-workers','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('freeze','checkpoint','engineering','worker-reference'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--task',choices=TASKS);parser.add_argument('--arm',choices=NEW_ARMS)
    parser.add_argument('--logical-ranks',type=int,nargs='+')
    args=parser.parse_args()
    if args.mode=='freeze':
        protocol=prepared(args);args.output.mkdir(parents=True,exist_ok=False)
        write_json(args.output/'protocol.json',protocol)
        write_json(args.output/'FROZEN.json',{'protocol_sha256':sha256(args.output/'protocol.json'),
            'routed_behavioral_outcomes_observed_before_freeze':False,'fit_only_diagnostics_already_observed':True})
    else:
        if any(getattr(args,k) is None for k in ('freeze','checkpoint','worker_reference','task')):
            parser.error('Execution requires source/fit/freeze/worker bindings')
        if args.mode=='run' and any(getattr(args,k) is None for k in ('engineering','arm','logical_ranks')):
            parser.error('Scientific run needs full engineering, arm and streams')
        execute(args)


if __name__=='__main__':main()
