"""CPU native-loader parity and exact-identity separation for DROID fit inputs.

No model calls, fitted operator, behavioral outcomes or confirmation access.
Absence of exact duplicate fingerprints is not proof of population independence.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from offline_study.tasks.droid.droid_native import array_hash, assert_same, verified_report
from offline_study.tasks.droid.droid_contract import TRAIN_CONFIG, prepare
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


def strict_raw_class():
    from app.plan_common.datasets.droid_dset import DROIDVideoDataset
    class StrictRaw(DROIDVideoDataset):
        def __getitem__(self,idx,debug=False,**kwargs):
            return super().__getitem__(idx,debug=True,**kwargs)
        def loadvideo_decord(self,path):
            result=super().loadvideo_decord(path)
            self.last_sample={'path':str(path),'indices':result[-1].tolist(),
                'state_sha256':array_hash(result[2])}
            return result
    return StrictRaw


def make_raw_dataset(csv,vendor,strict=True):
    from app.plan_common.datasets.droid_dset import DROIDVideoDataset
    from app.plan_common.datasets.transforms import make_transforms
    train=yaml.safe_load((vendor/TRAIN_CONFIG).read_text())
    data=train['data']
    if (data['droid']['camera_views']!=['left_mp4_path'] or data['droid']['dataset_fpcs']!=[4]
            or data['droid']['fps']!=4 or data['custom']['frameskip']!=1 or data['custom']['action_skip']!=1
            or data['custom']['normalize_action'] is not False or data['droid']['droid_to_rcasa_action_format']!=1):
        raise ValueError('Native raw-DROID training inputs changed')
    return (strict_raw_class() if strict else DROIDVideoDataset)(data_path=str(csv),camera_views=['left_mp4_path'],
        frameskip=1,action_skip=1,frames_per_clip=4,fps=4,transform=make_transforms(img_size=256,**train['data_aug']),
        camera_frame=data['droid']['camera_frame'],normalize_action=False,mpk_dset=False,
        droid_to_rcasa_action_format=1,local=True,seed=234)


def fingerprints(states):
    values=np.asarray(states,dtype=np.float32)
    if values.ndim!=2 or values.shape[1]!=7 or not np.isfinite(values).all():
        raise ValueError('Malformed or nonfinite seven-dimensional state')
    return {'state_float32_sha256':array_hash(values),'initial_float32_sha256':array_hash(values[0])}


def compare_groups(fit,evaluation):
    groups={};overlap={}
    for field in ('state_float32_sha256','initial_float32_sha256'):
        by_hash={}
        for row in fit:by_hash.setdefault(row[field],[]).append(row['priority_rank'])
        groups[field]=[rows for rows in by_hash.values() if len(rows)>1]
        overlap[field]=[row['priority_rank'] for row in fit if row[field] in {x[field] for x in evaluation}]
    if any(overlap.values()):raise ValueError('Exact fitting/evaluation state overlap')
    return {'duplicate_fit_groups':groups,'exact_evaluation_state_overlap':overlap,
        'absence_of_exact_overlap_is_not_proof_of_statistical_independence':True}


def reset():
    random.seed(2026090804);np.random.seed(2026090804);torch.manual_seed(2026090804)


def run(args):
    import h5py
    from decord import VideoReader,cpu
    use_vendor(args.vendor)
    source=json.loads((args.inputs/'protocol.json').read_text())
    report,digest=verified_report(args.inputs)
    if (report['status']!='all128_native_recording_objects_downloaded_and_verified_not_fit' or
            report['files_sha256']!=sha256(args.inputs/'FILES.json') or report['recordings']!=128):
        raise ValueError('Raw input completion missing')
    files=json.loads((args.inputs/'FILES.json').read_text())
    for name,want in files.items():
        path=args.inputs/'raw'/name
        if path.is_symlink() or '..' in Path(name).parts or path.stat().st_size!=want['bytes'] or sha256(path)!=want['sha256']:
            raise ValueError('Raw source bytes changed')
    contract=prepare(args.vendor,json.loads(args.manifest.read_text()))
    assets,asset_hash=verified_report(args.assets)
    if assets['status']!='official_droid_inputs_staged_and_verified':raise ValueError('Evaluation asset identity missing')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',{'role':'input_parity_and_exact_identity_audit_not_model_fit',
        'inputs_report_sha256':digest,'assets_report_sha256':asset_hash,'source_sha256':sha256(Path(__file__)),
        'native_loader_sha256':sha256(args.vendor/'app/plan_common/datasets/droid_dset.py'),
        'native_training_config_sha256':sha256(args.vendor/TRAIN_CONFIG),'recordings':128,
        'prefixes_per_recording':4,'native_constructor_rng_preserved':True,'model_calls':0,'gpu_calls':0})
    try:
        rows=[]
        for row in source['recordings']:
            root=args.inputs/'raw'/row['directory']
            camera=args.inputs/'raw'/row['left_camera_name']
            name=camera.stem+'_left'
            with h5py.File(root/'trajectory.h5','r') as handle:
                obs=handle['observation'];poses=np.asarray(obs['robot_state/cartesian_position'])
                grip=np.asarray(obs['robot_state/gripper_position'])
                if grip.shape!=(len(poses),):raise ValueError('Native gripper alignment changed')
                states=np.c_[poses,grip];fp=fingerprints(states)
                ext=np.asarray(obs['camera_extrinsics'][name])
            vr=VideoReader(str(camera),num_threads=1,ctx=cpu(0));fps=float(vr.get_avg_fps())
            if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid native video fps')
            stride=math.ceil(fps/4);frames=len(vr)
            if frames<=4*stride or frames>len(states) or len(ext)<frames or not np.isfinite(ext).all():
                raise ValueError('Native camera/state window alignment or finite data failed')
            first=vr[0].asnumpy()
            if first.ndim!=3 or first.shape[-1]!=3 or min(first.shape[:2])<256 or first.max()==first.min():
                raise ValueError('Invalid native camera resolution/content')
            rows.append({**row,**fp,'video_frames':frames,'state_frames':len(states),'source_fps':fps,
                'frame_stride':stride,'camera_shape':list(first.shape),'first_frame_sha256':array_hash(first)})
            del vr
        evaluation=[]
        expected={x['filename']:x for x in assets['assets']}
        for name in contract['released_config_recordings']:
            path=args.assets/'downloads/dataset'/name
            if sha256(path)!=expected[name]['verified_content_sha256']:raise ValueError('Evaluation source changed')
            with h5py.File(path,'r') as handle:
                obs=handle['episode_data/observation']
                states=np.c_[np.asarray(obs['cartesian_position']),np.asarray(obs['gripper_position'])]
                evaluation.append({'source':name,**fingerprints(states)})
        separation=compare_groups(rows,evaluation)
        write_json(args.output/'identity.json',{'fit':rows,'evaluation_input_fingerprints':evaluation,**separation,
            'evaluation_model_outcomes_accessed':False})
        csv=args.output/'native_paths.csv'
        paths=[str(args.inputs/'raw'/row['directory']) for row in rows]
        if any(' ' in path or '\n' in path for path in paths):raise ValueError('Native CSV path separator ambiguity')
        csv.write_text('\n'.join(paths)+'\n')
        reset();native=make_raw_dataset(csv,args.vendor,False)
        native_records=[(native[i],copy.deepcopy(native.rng.get_state())) for i in (0,63,127)]
        native_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        reset();strict=make_raw_dataset(csv,args.vendor,True)
        strict_records=[(strict[i],copy.deepcopy(strict.rng.get_state())) for i in (0,63,127)]
        assert_same(native_records,strict_records)
        assert_same(native_rng,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
        # Fixed native-RNG clip sequence for future fitting; no model called here.
        reset();dset=make_raw_dataset(csv,args.vendor,True);prefixes=[]
        for i,row in enumerate(rows):
            for prefix in range(4):
                obs,actions,state,_=dset[i]
                if (obs['visual'].shape!=(4,3,256,256) or actions.shape!=(4,7) or
                        not torch.isfinite(obs['visual']).all() or not np.isfinite(actions).all()):
                    raise ValueError('Native prepared clip shape/finite parity failed')
                prefixes.append({'recording_index':i,'priority_rank':row['priority_rank'],'prefix':prefix,
                    **dset.last_sample,'visual_sha256':array_hash(obs['visual'].numpy()),'actions_sha256':array_hash(actions)})
            print(json.dumps({'audited_recordings':i+1,'fit_prefixes':len(prefixes),'model_calls':0}),flush=True)
        write_json(args.output/'prefixes.json',prefixes)
        write_json(args.output/'report.json',{'status':'native_droid_fit_inputs_parity_and_exact_separation_passed',
            'protocol_sha256':sha256(args.output/'protocol.json'),'recordings':128,'prefixes':512,
            'files_sha256':{name:sha256(args.output/name) for name in ('identity.json','prefixes.json','native_paths.csv')},
            'native_pixels_actions_states_rng_exact':True,**separation,'fit_ready_for_separate_frozen_operator_protocol':True,
            'model_calls':0,'statistical_independence_proven':False,'fresh_confirmation':False})
        write_json(args.output/'DONE.json',{'report_sha256':sha256(args.output/'report.json')})
    except Exception as exc:
        write_json(args.output/'FAILED.json',{'error':str(exc),'partial_inputs_retained':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','assets','manifest','vendor','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
