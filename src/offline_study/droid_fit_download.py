"""Download only frozen raw DROID fit objects; no model, GPU or evaluation calls."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.error
import urllib.request

from .droid_catalogue import PREFIX
from .droid_fit_availability import media_url
from .protocol import sha256, write_json


def selected_objects(root):
    report=json.loads((root/'report.json').read_text())
    if (json.loads((root/'DONE.json').read_text())['report_sha256']!=sha256(root/'report.json') or
            report['protocol_sha256']!=sha256(root/'protocol.json') or
            report['cohort_sha256']!=sha256(root/'cohort.json') or
            report['status']!='128_native_input_object_sets_resolved_not_model_fit'):
        raise ValueError('Require complete source-object cohort')
    cohort=json.loads((root/'cohort.json').read_text())
    if len(cohort['selected'])!=128:raise ValueError('Incomplete fixed fitting pool')
    objects={};recordings=[]
    for row in cohort['selected']:
        proof=root/f"priority-{row['priority_rank']:03d}"
        if (sha256(proof/'report.json')!=row['report_sha256'] or
                json.loads((proof/'DONE.json').read_text())['report_sha256']!=row['report_sha256'] or
                sha256(proof/'directory.json')!=row['directory_sha256'] or
                any(sha256(proof/name)!=digest for name,digest in row['all_metadata_alias_sha256'].items())):
            raise ValueError('Selected input metadata proof changed')
        required=[row['source_trajectory'],row['native_left_video_object'],*row['all_metadata_alias_objects']]
        for obj in required:
            path=PurePosixPath(obj['name'])
            if (not obj['name'].startswith(PREFIX) or path.is_absolute() or '..' in path.parts or
                    '\\' in obj['name'] or int(obj['size'])<=0 or not obj['generation'].isdigit() or
                    len(base64.b64decode(obj['md5Hash'],validate=True))!=16):
                raise ValueError('Unsafe or unversioned source object')
            if obj['name'] in objects and objects[obj['name']]!=obj:raise ValueError('Object identity conflict')
            objects[obj['name']]=obj
        recordings.append({'priority_rank':row['priority_rank'],
            'directory':str(PurePosixPath(row['source_trajectory']['name']).parent),
            'metadata_names':[obj['name'] for obj in row['all_metadata_alias_objects']],
            'left_camera_name':row['native_left_video_object']['name']})
    if len({r['directory'] for r in recordings})!=128:raise ValueError('Duplicated source recording')
    return sorted(objects.values(),key=lambda obj:obj['name']),recordings


def download_object(obj, root):
    target=root/obj['name'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError('Do not overwrite prior object')
    for attempt in range(4):
        partial=target.with_name(target.name+f'.partial-attempt-{attempt}')
        digest=hashlib.sha256();md5=hashlib.md5();size=0
        try:
            with urllib.request.urlopen(media_url(obj),timeout=60) as response, partial.open('xb') as sink:
                for chunk in iter(lambda:response.read(4*1024**2),b''):
                    size+=len(chunk)
                    if size>int(obj['size']):raise ValueError('Download exceeds pinned size')
                    sink.write(chunk);digest.update(chunk);md5.update(chunk)
            if size!=int(obj['size']) or base64.b64encode(md5.digest()).decode()!=obj['md5Hash']:
                raise ValueError('Object bytes do not match pinned identity; do not replace source')
            os.rename(partial,target)
            return {'object':obj,'relative_path':obj['name'],'sha256':digest.hexdigest(),
                'bytes':size,'attempts':attempt+1,'earlier_partial_attempts_retained':attempt>0}
        except (TimeoutError,urllib.error.URLError) as exc:
            if isinstance(exc,urllib.error.HTTPError) and exc.code not in (408,429,500,502,503,504):raise
            if attempt==3:raise
            time.sleep(2**attempt)


def run(metadata,output):
    objects,recordings=selected_objects(metadata)
    size=sum(int(obj['size']) for obj in objects)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',{'role':'frozen_native_droid_fit_inputs_download_not_model_fit',
        'cohort_sha256':sha256(metadata/'cohort.json'),'metadata_report_sha256':sha256(metadata/'report.json'),
        'source_sha256':sha256(Path(__file__)),'objects':objects,'recordings':recordings,
        'expected_bytes':size,'download_workers':2,'raw_training_count_not_author8000':128,
        'native_loader_parity_passed':False,'family_disjointness_verified':False,'gpu_calls':0})
    try:
        if shutil.disk_usage(output).free < size+8*1024**3:
            raise ValueError('Require full source bytes plus8GiB reserve')
        with ThreadPoolExecutor(max_workers=2) as pool:
            records=[]
            for result in pool.map(lambda obj:download_object(obj,output/'raw'),objects):
                records.append(result)
                write_json(output/'progress.json',{'verified_objects':len(records),'required_objects':len(objects)})
                print(json.dumps({'verified_objects':len(records),'required_objects':len(objects)}),flush=True)
        write_json(output/'FILES.json',{r['relative_path']:{'sha256':r['sha256'],'bytes':r['bytes']} for r in records})
        write_json(output/'report.json',{'status':'all128_native_recording_objects_downloaded_and_verified_not_fit',
            'protocol_sha256':sha256(output/'protocol.json'),'files_sha256':sha256(output/'FILES.json'),
            'recordings':128,'objects':len(records),'bytes':size,'downloads':records,
            'native_loader_parity_passed':False,'family_disjointness_verified':False,'model_calls':0})
        write_json(output/'DONE.json',{'report_sha256':sha256(output/'report.json')})
    except Exception as exc:
        write_json(output/'FAILED.json',{'error':str(exc),'partial_inputs_retained':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.metadata,args.output)
