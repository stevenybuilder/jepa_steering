#!/usr/bin/env python3
"""Read four frozen official TRAIN clips into a hash-receipted tar stream.

No source writes, image decoding, simulator, model, or outcome calculation.
The receiver must use a NEW empty root; stdout contains tar bytes only.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import tarfile
import time


def digest(value):
    return hashlib.sha256(value).hexdigest()


def validate_manifest(manifest):
    if manifest['panel'] != 'pusht_new_official_train_development_v1':
        raise ValueError('Wrong panel')
    if [r['source_id'] for r in manifest['rows']] != [2222,2323,2424,2525]:
        raise ValueError('Frozen sources changed')
    if Path(manifest['source_root']).name != 'train' or manifest['clip_offset'] != 0:
        raise ValueError('Official TRAIN offset0 required')
    if len({r['initial_state_sha256'] for r in manifest['rows']}) != 4:
        raise ValueError('Initial groups not distinct')
    if not manifest['no_simulator_or_model_execution_authorized']:
        raise ValueError('This stage only exports raw source inputs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-b64',required=True)
    args=p.parse_args()
    manifest_bytes=base64.b64decode(args.manifest_b64)
    manifest=json.loads(manifest_bytes)
    validate_manifest(manifest) # Before opening source tensor storage.
    import torch
    torch.set_num_threads(1)
    started=time.process_time()
    root=Path(manifest['source_root'])
    with (root/'seq_lengths.pkl').open('rb') as stream:
        lengths=pickle.load(stream)
    raw=torch.load(root/'states.pth',mmap=True,map_location='cpu',weights_only=True)
    velocity=torch.load(root/'velocities.pth',mmap=True,map_location='cpu',weights_only=True)
    actions=torch.load(root/'rel_actions.pth',mmap=True,map_location='cpu',weights_only=True)
    assets_bytes=(root.parents[2]/'DONE.json').read_bytes()
    assets=json.loads(assets_bytes)
    archive=next(r for r in assets['outputs'] if r['file']==manifest['archive'])
    if archive['sha256']!=manifest['archive_sha256'] or archive['revision']!=manifest['dataset_revision']:
        raise ValueError('Pinned official asset receipt mismatch')
    outputs=[]
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
        def add(name,data):
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644
            tar.addfile(item,io.BytesIO(data))
            outputs.append({'path':name,'bytes':len(data),'sha256':digest(data)})
        add('manifest.json',manifest_bytes)
        add('SOURCE_ASSETS_DONE.json',assets_bytes)
        for row in manifest['rows']:
            i=row['source_id']
            if int(lengths[i])!=row['source_length'] or lengths[i]<31:
                raise ValueError('Frozen source clip unavailable; no replacement')
            states=torch.cat([raw[i,:31].float(),velocity[i,:31].float()],dim=-1).contiguous()
            if digest(states[0].numpy().tobytes())!=row['initial_state_sha256']:
                raise ValueError('Frozen raw initializer mismatch')
            commands=actions[i,:30].float().contiguous()/100.
            if commands.shape!=(30,2) or not torch.isfinite(commands).all():
                raise ValueError('Invalid native raw commands')
            value={'panel':manifest['panel'],'split':manifest['split'],**row,
                'source_offset':0,'environment_seed':2026090600+i,'planner_seed':91600+i,
                'initial_state':states[0].clone(),'env_info':{'shape':'T'},
                'source_states':states,'expert_actions_raw':commands,
                'goal_construction_status':'pending_native_simulator_replay',
                'source_manifest_sha256':digest(manifest_bytes)}
            buffer=io.BytesIO();torch.save(value,buffer)
            add(f'input-{i}.pt',buffer.getvalue())
            add(f'videos/episode_{i:03d}.mp4',(root/'obses'/f'episode_{i:03d}.mp4').read_bytes())
        done={'complete':True,'stage':'raw_source_staging_only','outputs':list(outputs),
            'source_root':str(root),'manifest_sha256':digest(manifest_bytes),
            'source_dataset_revision':manifest['dataset_revision'],
            'archive_sha256_from_existing_verified_receipt':archive['sha256'],
            'archive_rehashed_this_stage':False,'source_write':False,
            'model_execution':False,'simulator_execution':False,'video_decoding':False,
            'outcome_analysis':False,'confirmation_opened':False,
            'cpu_seconds_excluding_import':time.process_time()-started}
        add('DONE.json',(json.dumps(done,indent=2,sort_keys=True)+'\n').encode())


if __name__=='__main__':main()
