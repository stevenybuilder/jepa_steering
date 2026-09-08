"""Fit the fixed archived HMM mathematics to verified native TRAIN histories.

No validation targets, simulator outcomes or candidate selection are loaded.
Family-disjoint diagnostics assess history information, not behavioral efficacy.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from .author_fit import source_hash
from .author_runtime import examples, validate_cohort
from .behavioral_development import verified_report
from .fixed_response import load_fitted_bank
from .planning_native_smoke import CHECKPOINTS
from .protocol import sha256, write_json
from .routing_hmm import emission, filter_beliefs, fit_hmm, route_prefix, validate_model

METHOD = 'native_history_hmm_fixed_response_v1'
CAPTURE_SOURCE = 'cb72cd7963d2a56e538af641588a194e310d670d728c79fbbc5cb6c7d323437d'


def load_histories(root, fit):
    cohort=json.loads((fit/'cohort.json').read_text()); validate_cohort(cohort)
    bank=load_fitted_bank(fit,task=cohort['task'],checkpoint_sha256=CHECKPOINTS['metaworld'])
    histories,metadata,bindings=[],[],{}
    for index in range(4):
        shard=root/f'shard-{index:03d}'
        report,digest=verified_report(shard)
        protocol=json.loads((shard/'protocol.json').read_text())
        rows=cohort['fit'][index::4]
        expected=examples(rows,cohort['reference_config'],fitting=True)
        if (report['status']!='native_fit_history_captured_not_hmm_eligibility_or_efficacy' or
                report['protocol_sha256']!=sha256(shard/'protocol.json') or
                report['history_sha256']!=sha256(shard/'history.pt') or
                report['task']!=cohort['task'] or report['shard_index']!=index or
                report['fit_families']!=32 or report['fit_prefixes']!=128 or report['identity_checks']<1 or
                any(report[k] is not True for k in ('model_parameters_unchanged','passive_capture_bitwise_identity')) or
                any(report[k] is not False for k in ('evaluation_outcomes_accessed','scientific_efficacy_measurement')) or
                protocol['source_sha256']!=CAPTURE_SOURCE or protocol['precision']!='bfloat16' or
                protocol['checkpoint_sha256']!=CHECKPOINTS['metaworld'] or
                protocol['cohort_sha256']!=sha256(fit/'cohort.json') or protocol['selection']!=rows or
                protocol['examples']!=expected or protocol['horizons']!=[1,2,3,4,5,6] or
                protocol['routing_decision_may_use_only']!=[1,2] or protocol['hidden_features']!=400 or
                protocol['source_rank_protocol_sha256']!=bank['binding']['source_protocol_sha256'] or
                protocol['source_fit_receipt_sha256']!=bank['binding']['source_fit_receipt_sha256']):
            raise ValueError('Changed native fitting history/cohort/identity receipt')
        payload=torch.load(shard/'history.pt',map_location='cpu',weights_only=True)
        actual=payload['metadata']; history=payload['history']
        key=lambda row:(row['trajectory_id'],row['start'])
        by_key={key(row):row for row in expected}
        if (len(actual)!=128 or len({key(row) for row in actual})!=128 or
                any(row!=by_key.get(key(row)) for row in actual) or
                history.shape!=(128,6,400) or history.dtype!=torch.float32 or not torch.isfinite(history).all()):
            raise ValueError('Missing, duplicated or changed fitting prefixes')
        histories.append(history);metadata.extend(actual)
        bindings[str(index)]={'report_sha256':digest,'protocol_sha256':sha256(shard/'protocol.json'),
            'history_sha256':sha256(shard/'history.pt')}
    if len(Counter(r['lineage_group'] for r in metadata))!=128 or set(Counter(r['lineage_group'] for r in metadata).values())!={4}:
        raise ValueError('Require128 equally weighted fitting families')
    return torch.cat(histories), metadata, cohort, bank, bindings


def family_folds(metadata):
    families=sorted({r['lineage_group'] for r in metadata},key=lambda name:hashlib.sha256(
        ('2026090802:'+name).encode()).hexdigest())
    assignment={family:i%4 for i,family in enumerate(families)}
    return torch.tensor([assignment[r['lineage_group']] for r in metadata]),assignment


@torch.no_grad()
def history_value(features,metadata):
    folds,assignment=family_folds(metadata)
    scores=torch.empty(len(features),dtype=torch.float64)
    receipts=[]
    for fold in range(4):
        train,test=folds!=fold,folds==fold
        model=fit_hmm(features[train])
        x=((features[test].double()-model['center'])@model['basis'].T)/model['scale']
        hmm=filter_beliefs(x[:,:2],model)[:,-1]
        memoryless=(emission(x[:,1],model['means'],model['variance'])+model['occupancy'].log()).softmax(-1)
        next_emission=emission(x[:,2],model['means'],model['variance'])
        score=lambda belief:torch.logsumexp((belief@model['transition']).log()+next_emission,-1)
        scores[test]=score(hmm)-score(memoryless)
        receipts.append({'fold':fold,'train_families':len({r['lineage_group'] for i,r in enumerate(metadata) if train[i]}),
            'test_families':len({r['lineage_group'] for i,r in enumerate(metadata) if test[i]}),
            'test_prefixes':int(test.sum()),'mean_h3_log_density_gain_nats':float(scores[test].mean())})
    family_scores={name:float(scores[torch.tensor([r['lineage_group']==name for r in metadata])].mean()) for name in assignment}
    return {'folds':receipts,'family_folds':assignment,'family_log_density_gain_nats':family_scores,
        'mean_h3_log_density_gain_nats':float(scores.mean()),'positive_gain_families':sum(v>0 for v in family_scores.values()),
        'independent_of_base_checkpoint_and_operator_training':False,'task_success_evidence':False}


def load_routing_fit(root, fixed_fit):
    report,_=verified_report(root)
    protocol=json.loads((root/'protocol.json').read_text())
    if (report['status']!='hmm_fit_and_native_history_diagnostics_complete_not_behavioral_clearance' or
            report['protocol_sha256']!=sha256(root/'protocol.json') or
            json.loads((root/'FROZEN.json').read_text())['protocol_sha256']!=sha256(root/'protocol.json') or
            report['model_sha256']!=sha256(root/'model.pt') or
            report['diagnostics_sha256']!=sha256(root/'diagnostics.json') or
            protocol['fixed_fit_done_sha256']!=sha256(fixed_fit/'DONE.json') or
            protocol['fixed_bank_sha256']!=sha256(fixed_fit/'operator_bank.pt') or
            report['evaluation_outcomes_accessed'] is not False or report['behavioral_launch_ready'] is not False):
        raise ValueError('Unbound HMM fit')
    payload=torch.load(root/'model.pt',map_location='cpu',weights_only=True)
    validate_model(payload['model'])
    if payload['protocol_sha256']!=sha256(root/'protocol.json'):
        raise ValueError('HMM model/protocol mismatch')
    return payload,protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('history','fixed-fit','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(2)
    config=json.loads(args.config.read_text())
    if config['method']!=METHOD or config['hidden_states']!=2 or config['em_iterations']!=20 or config['pca_dimensions']!=3:
        raise ValueError('Unregistered HMM specification')
    features,metadata,cohort,bank,bindings=load_histories(args.history,args.fixed_fit)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    protocol={'method':METHOD,'config':config,'config_sha256':sha256(args.config),'task':cohort['task'],
        'fixed_fit_done_sha256':sha256(args.fixed_fit/'DONE.json'),'fixed_bank_sha256':sha256(args.fixed_fit/'operator_bank.pt'),
        'checkpoint_sha256':CHECKPOINTS['metaworld'],'cohort_sha256':sha256(args.fixed_fit/'cohort.json'),
        'history_bindings':bindings,'fit_families':128,'fit_prefixes':512,'source_sha256':source_hash(),
        'development_and_protected_outcomes_accessed':False,'behavioral_launch_authorized_by_this_fit':False}
    write_json(args.output/'protocol.json',protocol)
    write_json(args.output/'FROZEN.json',{'protocol_sha256':sha256(args.output/'protocol.json'),
        'hmm_estimator_outputs_observed_before_freeze':False})
    try:
        diagnostic=history_value(features,metadata)
        model=fit_hmm(features)
        gates=route_prefix(features[:,:2],model)
        normalizers={name:gates[name].square().mean().sqrt() for name in ('static','memoryless','hmm')}
        normalized={name:gates[name]/normalizers[name] for name in normalizers}
        separation=float((normalized['hmm']-normalized['memoryless']).square().mean().sqrt())
        torch.save({'model':model,'normalizers':normalizers,'protocol_sha256':sha256(args.output/'protocol.json')},args.output/'model.pt')
        write_json(args.output/'diagnostics.json',{'history_value':diagnostic,
            'normalized_hmm_vs_memoryless_gate_rms_difference':separation,
            'fit_normalizers':{k:float(v) for k,v in normalizers.items()},
            'fit_gate_mean_squared':{k:float(v.square().mean()) for k,v in normalized.items()},
            'comparator_energy_equality_is_fit_expectation_not_guaranteed_on_planner_actions':True,
            'treatment_separation_is_not_behavioral_benefit':True})
        if source_hash()!=protocol['source_sha256']:
            raise ValueError('Fitting source changed during execution')
        write_json(args.output/'report.json',{'status':'hmm_fit_and_native_history_diagnostics_complete_not_behavioral_clearance',
            'task':cohort['task'],'protocol_sha256':sha256(args.output/'protocol.json'),'model_sha256':sha256(args.output/'model.pt'),
            'diagnostics_sha256':sha256(args.output/'diagnostics.json'),'fit_families':128,'fit_prefixes':512,
            'evaluation_outcomes_accessed':False,'behavioral_launch_ready':False,'fresh_confirmation':False,
            'seconds':time.monotonic()-started})
        write_json(args.output/'DONE.json',{'report_sha256':sha256(args.output/'report.json')})
        load_routing_fit(args.output,args.fixed_fit)
        print(json.dumps({'task':cohort['task'],'status':'fit_complete_not_behavioral_clearance','seconds':time.monotonic()-started}),flush=True)
    except Exception as exc:
        write_json(args.output/'FAILED.json',{'error':str(exc),'partial_not_complete':True});raise


if __name__=='__main__':
    main()
