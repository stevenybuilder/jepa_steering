"""Bounded, pre-outcome source selection/metadata for separate DROID operator fit.

Not the authors'8k base-training subset, a model fit or behavioral evaluation.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path,PurePosixPath
import urllib.parse
import urllib.request

from .droid_catalogue import API,PREFIX,validate_object
from .protocol import sha256,write_json

CATALOGUE='e247ebb89a069b775514af415e8de9a7ff720f0ef16327aa5b9e5610cc970150'
SEED=2026090804


def select(rows):
    if len(rows)!=74970 or len({r['name'] for r in rows})!=len(rows):raise ValueError('Changed full source catalogue')
    for row in rows:validate_object(row)
    return sorted(rows,key=lambda row:hashlib.sha256((str(SEED)+':'+row['name']).encode()).hexdigest())[:128]


def freeze(catalogue,output,vendor):
    if sha256(catalogue)!=CATALOGUE:raise ValueError('Catalogue bytes differ from completed provenance audit')
    rows=select(json.loads(catalogue.read_text()))
    config=vendor/'configs/vjepa_wm/droid_final_sweep/droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_noprop_repro_2roll_4n.yaml'
    output.mkdir(parents=True,exist_ok=False)
    protocol={'role':'separate_droid_intervention_fit_source_preparation','catalogue_sha256':CATALOGUE,
        'source_catalogue_trajectories':74970,'selection_seed':SEED,'selected_source_recordings':128,
        'selection':'ascending SHA256(seed:full_source_object_name), no success/lab filter','rows':rows,
        'source_generator_sha256':sha256(vendor/'src/scripts/generate_droid_paths.py'),
        'native_loader_sha256':sha256(vendor/'app/plan_common/datasets/droid_dset.py'),
        'native_training_config_sha256':sha256(config),'camera_view':'left_mp4_path',
        'all_selected_inputs_required_no_silent_substitution':True,'source_sha256':sha256(Path(__file__)),
        'author8000_base_training_subset_reproduced':False,'recording_independence_audited':False,
        'evaluation_family_disjointness_audited':False,'model_or_intervention_fitting_authorized_by_this_receipt':False,
        'franka_evaluation_recordings_used_for_fit':False,'behavioral_outcomes_accessed':False,
        'metadata_workers':4,'maximum_directory_objects':1000,'maximum_metadata_json_bytes':2*1024**2}
    write_json(output/'protocol.json',protocol)
    write_json(output/'FROZEN.json',{'protocol_sha256':sha256(output/'protocol.json'),
        'selected_observations_read_before_freeze':False})


def request_json(url,cap=2*1024**2):
    with urllib.request.urlopen(url,timeout=30) as response:data=response.read(cap+1)
    if len(data)>cap:raise ValueError('Metadata response exceeds bound')
    return json.loads(data)


def validate_directory(items,row):
    directory=str(PurePosixPath(row['name']).parent)+'/'
    by_name={item['name']:item for item in items}
    if len(by_name)!=len(items) or row['name'] not in by_name:raise ValueError('Missing/duplicate source trajectory')
    for key in ('generation','size','md5Hash'):
        if by_name[row['name']][key]!=row[key]:raise ValueError('Pinned trajectory object changed')
    for item in items:
        if (not item['name'].startswith(directory) or '..' in PurePosixPath(item['name']).parts or
                not str(item['generation']).isdigit() or int(item['size'])<0):raise ValueError('Unsafe or unversioned metadata object')
    jsons=[item for item in items if str(PurePosixPath(item['name']).parent)+'/'==directory and item['name'].endswith('.json')]
    if len(jsons)!=1:raise ValueError('Native metadata choice is ambiguous or missing; do not invent ordering')
    return directory,by_name,jsons[0]


def one_metadata(index,row,output):
    target=output/f'recording-{index:03d}';target.mkdir(exist_ok=False)
    directory=str(PurePosixPath(row['name']).parent)+'/'
    url=API+'?'+urllib.parse.urlencode({'prefix':directory,'maxResults':1000,
        'fields':'items(name,generation,size,md5Hash),nextPageToken'})
    response=request_json(url)
    if response.get('nextPageToken'):raise ValueError('Unexpected recording directory expansion')
    items=response.get('items',[]);directory,by_name,metadata=validate_directory(items,row)
    write_json(target/'directory.json',response)
    if int(metadata['size'])>2*1024**2:raise ValueError('Oversized source metadata JSON')
    media=API+'/'+urllib.parse.quote(metadata['name'],safe='')+'?'+urllib.parse.urlencode({'alt':'media','generation':metadata['generation']})
    with urllib.request.urlopen(media,timeout=30) as stream:data=stream.read(2*1024**2+1)
    if len(data)!=int(metadata['size']) or base64.b64encode(hashlib.md5(data).digest()).decode()!=metadata['md5Hash']:
        raise ValueError('Source metadata bytes differ from pinned object')
    value=json.loads(data);path=value['left_mp4_path']
    pieces=path.split('recordings/MP4/')
    if len(pieces)!=2 or PurePosixPath(pieces[-1]).name!=pieces[-1] or not pieces[-1].endswith('.mp4'):
        raise ValueError('Native left-camera path cannot be resolved safely')
    camera=directory+'recordings/MP4/'+pieces[-1]
    if camera not in by_name or int(by_name[camera]['size'])<=0:raise ValueError('Selected native camera missing')
    (target/'metadata.json').write_bytes(data)
    report={'role':'source_metadata_only_not_fit_eligibility','source_trajectory':row,'metadata_object':metadata,
        'native_left_video_object':by_name[camera],'metadata_sha256':sha256(target/'metadata.json'),
        'directory_sha256':sha256(target/'directory.json'),'trajectory_or_video_downloaded':False,
        'evaluation_or_confirmation_outcomes_accessed':False}
    write_json(target/'report.json',report);write_json(target/'DONE.json',{'report_sha256':sha256(target/'report.json')})
    return {'recording':index,'report_sha256':sha256(target/'report.json')}


def metadata(frozen,output):
    protocol=json.loads((frozen/'protocol.json').read_text())
    if (json.loads((frozen/'FROZEN.json').read_text())['protocol_sha256']!=sha256(frozen/'protocol.json') or
            protocol['source_sha256']!=sha256(Path(__file__)) or protocol['catalogue_sha256']!=CATALOGUE or
            protocol['selected_source_recordings']!=128 or len(protocol['rows'])!=128):raise ValueError('Input freeze changed')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'protocol.json',{'input_freeze_sha256':sha256(frozen/'protocol.json'),
        'mode':'metadata_only','fit_or_gpu_job_launched':False})
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(one_metadata,i,row,output) for i,row in enumerate(protocol['rows'])]
            records=[]
            for future in futures:
                records.append(future.result());print(json.dumps({'metadata_complete':len(records),'required':128}),flush=True)
        write_json(output/'report.json',{'status':'all128_selected_source_metadata_resolved_not_fit_eligibility',
            'protocol_sha256':sha256(output/'protocol.json'),'recordings':records,'model_calls':0,
            'author8000_manifest_reproduced':False,'evaluation_family_disjointness_audited':False})
        write_json(output/'DONE.json',{'report_sha256':sha256(output/'report.json')})
    except Exception as exc:
        write_json(output/'FAILED.json',{'error':str(exc),'partial_metadata_retained':True});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('freeze','metadata'))
    parser.add_argument('--output',type=Path,required=True)
    for name in ('catalogue','vendor','freeze'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.mode=='freeze':freeze(args.catalogue,args.output,args.vendor)
    else:metadata(args.freeze,args.output)


if __name__=='__main__':main()
