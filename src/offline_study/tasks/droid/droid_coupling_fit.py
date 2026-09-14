"""Fit the predeclared DROID coupling factorial on128separate raw recordings."""
from offline_study._paths import source_path
import argparse
import json
from pathlib import Path
import time

import torch
import yaml

from offline_study.tasks.droid.droid_contract import prepare, TRAIN_CONFIG
from offline_study.tasks.droid.droid_coupling import METHOD, ARMS, Capture, fit_bank, arm_fields
from offline_study.tasks.droid.droid_fit_audit import make_raw_dataset, reset
from offline_study.tasks.droid.droid_native import array_hash, load_model, verified_report
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


@torch.no_grad()
def run(args):
    use_vendor(args.vendor)
    audit,audit_hash=verified_report(args.audit)
    protocol=json.loads((args.audit/'protocol.json').read_text())
    if (audit['status']!='native_droid_fit_inputs_parity_and_exact_separation_passed' or
            audit['recordings']!=128 or audit['prefixes']!=512 or not audit['native_pixels_actions_states_rng_exact'] or
            any(audit['exact_evaluation_state_overlap'].values()) or any(audit['duplicate_fit_groups'].values()) or
            protocol['source_sha256']!=sha256(source_path('droid_fit_audit.py'))):
        raise ValueError('Require complete matching fit-input audit without exact overlaps/duplicates')
    for name,digest in audit['files_sha256'].items():
        if sha256(args.audit/name)!=digest:raise ValueError('Input audit member changed')
    inputs,input_hash=verified_report(args.inputs)
    if input_hash!=protocol['inputs_report_sha256'] or inputs['files_sha256']!=sha256(args.inputs/'FILES.json'):
        raise ValueError('Wrong raw fitting inputs')
    raw_files=json.loads((args.inputs/'FILES.json').read_text())
    for name,want in raw_files.items():
        source=args.inputs/'raw'/name
        if source.stat().st_size!=want['bytes'] or sha256(source)!=want['sha256']:
            raise ValueError('Raw fitting object changed')
    contract=prepare(args.vendor,json.loads(args.manifest.read_text()))
    prefixes=json.loads((args.audit/'prefixes.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    binding={'method':METHOD,'task':'droid','checkpoint_sha256':contract['checkpoint_sha256'],
        'audit_report_sha256':audit_hash,'input_report_sha256':input_hash,
        'native_training_config_sha256':sha256(args.vendor/TRAIN_CONFIG),
        'source_files_sha256':{name:sha256(source_path(name)) for name in
            ('droid_coupling.py','droid_coupling_fit.py','droid_fit_audit.py','droid_native.py','operator_fit.py')},
        'fitting_recordings':128,'prefixes_per_recording':4,'fit_precision':'bfloat16',
        'model_architecture':'12blocks_1024features_no_proprio','context_capacity':2,'horizon':3,
        'condition_block':6,'visual_site':'native_predictor_input_H3','dose':'0.1_fit_robust_score_sigma_per_component',
        'solver':'native12iteration_cross_covariance_power_iteration','runtime_arms':list(ARMS),
        'future_frames_encoded':False,'evaluation_observations_used_for_fit':False,
        'runtime_online_probes':0,'runtime_native_shadows':0,'max_fit_seconds':3600,
        'behavioral_launch_ready':False,'fresh_confirmation':False}
    write_json(args.output/'protocol.json',binding)
    started=time.monotonic()
    try:
        torch.cuda.set_device(0);torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        reset();dset=make_raw_dataset(args.audit/'native_paths.csv',args.vendor,True)
        from app.plan_common.datasets.preprocessor import Preprocessor
        from app.plan_common.datasets.transforms import make_inverse_transforms
        train=yaml.safe_load((args.vendor/TRAIN_CONFIG).read_text())
        preprocessor=Preprocessor(**{name:getattr(dset,name) for name in
            ('action_mean','action_std','state_mean','state_std','proprio_mean','proprio_std')},
            transform=dset.transform,inverse_transform=make_inverse_transforms(img_size=256,**train['data_aug']))
        model,provenance=load_model(args.vendor,args.assets,args.encoder_source,args.encoder_root,contract,preprocessor)
        versions=[(name,p._version) for name,p in model.named_parameters()]
        # Model initialization consumes RNG. Recreate native clip stream exactly as audited.
        reset();dset=make_raw_dataset(args.audit/'native_paths.csv',args.vendor,True)
        visual,condition,seen=[],[],[]
        for item in prefixes:
            obs,actions,_,_=dset[item['recording_index']]
            if ({**dset.last_sample,'visual_sha256':array_hash(obs['visual'].numpy()),
                    'actions_sha256':array_hash(actions)}!={key:item[key] for key in
                        ('path','indices','state_sha256','visual_sha256','actions_sha256')}):
                raise ValueError('Fixed native fit prefix changed; do not resample')
            acts=torch.as_tensor(actions[:3],device='cuda:0',dtype=torch.float32)[:,None]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                context=model.model.encode_obs({'visual':obs['visual'][None,:1].to('cuda:0')})['visual']
                with Capture(model.model.predictor) as capture:
                    observed=model.unroll(context,act_suffix=acts)
                if not seen:
                    native=model.unroll(context,act_suffix=acts)
                    if not torch.equal(native,observed):raise ValueError('Passive native capture changed forecast')
            visual.append(capture.visual);condition.append(capture.condition);seen.append(item)
            if time.monotonic()-started>3600:raise TimeoutError('Bounded DROID fitting cap exceeded')
            if len(seen)%4==0:
                write_json(args.output/'progress.json',{'fit_recordings':len(seen)//4,'prefixes':len(seen),
                    'seconds':time.monotonic()-started,'behavioral_outcomes':0})
        captures={'visual':torch.cat(visual),'condition':torch.cat(condition)}
        torch.save(captures,args.output/'native_captures.pt')
        bank=fit_bank(captures['visual'],captures['condition'])
        for arm in ARMS:arm_fields(bank,arm,'cpu')
        bank['protocol_sha256']=sha256(args.output/'protocol.json')
        torch.save(bank,args.output/'operator_bank.pt');write_json(args.output/'prefixes.json',seen)
        if versions!=[(name,p._version) for name,p in model.named_parameters()]:raise ValueError('Frozen model changed')
        for name,digest in binding['source_files_sha256'].items():
            if sha256(source_path(name))!=digest:raise ValueError('Fitting source changed')
        write_json(args.output/'report.json',{'status':'droid_coupling_fit_complete_not_behavioral_clearance',
            'protocol_sha256':sha256(args.output/'protocol.json'),'model_provenance':provenance,
            'fit_recordings':128,'fit_prefixes':512,'passive_capture_exact':True,
            'files_sha256':{name:sha256(args.output/name) for name in ('native_captures.pt','operator_bank.pt','prefixes.json')},
            'fit_diagnostics':bank['fit_diagnostics'],'seconds':time.monotonic()-started,
            'parameters_unchanged':True,'behavioral_launch_ready':False,'fresh_confirmation':False})
        write_json(args.output/'DONE.json',{'report_sha256':sha256(args.output/'report.json')})
    except Exception as exc:
        write_json(args.output/'FAILED.json',{'error':str(exc),'partial_fit_not_eligible':True});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('vendor','assets','manifest','encoder-source','encoder-root','inputs','audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
