"""Pre-outcome DROID fit-input availability amendment; no model or outcome reads.

Retain the failed first proposed pool and its exact deterministic priority order.
Only demonstrably missing native left-camera objects are excluded. Network,
identity, metadata or decode errors are never reasons to choose another row.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.parse
import urllib.request

from offline_study.tasks.droid.droid_catalogue import API
from offline_study.tasks.droid.droid_fit_inputs import CATALOGUE, SEED, select, validate_directory
from offline_study.core.protocol import sha256, write_json


def order(rows):
    select(rows)  # Full catalogue cardinality, uniqueness and object validation.
    return sorted(rows, key=lambda row: hashlib.sha256((str(SEED)+':'+row['name']).encode()).hexdigest())


def fetch(url, cap):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as stream:
                data = stream.read(cap+1)
            if len(data) > cap:
                raise ValueError('Source response exceeds frozen bound')
            return data
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408,429,500,502,503,504):
                raise
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def media_url(obj):
    return API+'/'+urllib.parse.quote(obj['name'],safe='')+'?'+urllib.parse.urlencode(
        {'alt':'media','generation':obj['generation']})


def camera_object(directory, items, metadata):
    path = metadata['left_mp4_path']
    pieces = path.split('recordings/MP4/')
    if len(pieces)!=2 or PurePosixPath(pieces[-1]).name!=pieces[-1] or not pieces[-1].endswith('.mp4'):
        raise ValueError('Ambiguous native camera path; cannot skip recording')
    name = directory+'recordings/MP4/'+pieces[-1]
    if name not in items:
        return None, name
    obj = items[name]
    if int(obj['size']) <= 0 or len(base64.b64decode(obj['md5Hash'],validate=True)) != 16:
        raise ValueError('Malformed camera object; cannot skip recording')
    return obj, name


def equivalent_metadata(values):
    """Only aliases differing in collector identity are input-equivalent."""
    if not values or len(values)>4:
        raise ValueError('Missing or unbounded metadata aliases')
    projection=lambda value:{k:(v.casefold() if k=='lab' and isinstance(v,str) else v)
        for k,v in value.items() if k not in ('uuid','user_id')}
    if any(projection(value)!=projection(values[0]) for value in values[1:]):
        raise ValueError('Metadata aliases disagree on native inputs; cannot skip recording')
    return values[0]


def inspect_record(row, target):
    target.mkdir(exist_ok=False)
    directory = str(PurePosixPath(row['name']).parent)+'/'
    url = API+'?'+urllib.parse.urlencode({'prefix':directory,'maxResults':1000,
        'fields':'items(name,generation,size,md5Hash),nextPageToken'})
    listing = json.loads(fetch(url, 2*1024**2))
    if listing.get('nextPageToken'):
        raise ValueError('Expanded directory cannot be skipped')
    write_json(target/'directory.json',listing)
    raw_items=listing.get('items',[])
    aliases=sorted([obj for obj in raw_items if str(PurePosixPath(obj['name']).parent)+'/'==directory
        and obj['name'].endswith('.json')],key=lambda obj:obj['name'])
    if not aliases or len(aliases)>4:raise ValueError('Missing or unbounded metadata aliases')
    alias_names={obj['name'] for obj in aliases}
    for alias in aliases:
        validate_directory([obj for obj in raw_items if obj['name'] not in alias_names]+[alias],row)
    items={obj['name']:obj for obj in raw_items}
    values=[];metadata_hashes={}
    for i,meta in enumerate(aliases):
        if int(meta['size']) > 2*1024**2:raise ValueError('Oversized native metadata cannot be skipped')
        data=fetch(media_url(meta),2*1024**2)
        if len(data)!=int(meta['size']) or base64.b64encode(hashlib.md5(data).digest()).decode()!=meta['md5Hash']:
            raise ValueError('Metadata identity differs from pinned source')
        path=target/('metadata.json' if i==0 else f'metadata-alias-{i}.json')
        path.write_bytes(data);metadata_hashes[path.name]=sha256(path);values.append(json.loads(data))
    value=equivalent_metadata(values)
    camera, missing = camera_object(directory,items,value)
    result = {'source_trajectory':row,'metadata_object':aliases[0],'native_left_video_object':camera,
        'all_metadata_alias_objects':aliases,'all_metadata_alias_sha256':metadata_hashes,
        'metadata_aliases_input_equivalent':True,
        'metadata_sha256':sha256(target/'metadata.json'),'directory_sha256':sha256(target/'directory.json'),
        'eligibility':'native_camera_object_present' if camera else 'excluded_native_camera_object_absent',
        'missing_camera_name':missing if camera is None else None,
        'observations_read':False,'model_calls':0,'evaluation_outcomes_accessed':False}
    write_json(target/'report.json',result)
    write_json(target/'DONE.json',{'report_sha256':sha256(target/'report.json')})
    return result


def freeze(catalogue, original, output):
    if sha256(catalogue)!=CATALOGUE:
        raise ValueError('Unverified source catalogue')
    rows = order(json.loads(catalogue.read_text()))
    protocol = json.loads((original/'freeze/protocol.json').read_text())
    if (json.loads((original/'freeze/FROZEN.json').read_text())['protocol_sha256']!=sha256(original/'freeze/protocol.json')
            or protocol['rows']!=rows[:128] or not (original/'metadata/FAILED.json').is_file()):
        raise ValueError('Original failed proposed cohort identity changed')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',{'role':'availability_only_intervention_fit_input_amendment',
        'source_sha256':sha256(Path(__file__)),'catalogue_sha256':CATALOGUE,
        'original_proposal_sha256':sha256(original/'freeze/protocol.json'),
        'original_failure_sha256':sha256(original/'metadata/FAILED.json'),
        'original_inputs_preserved':True,'priority_seed':SEED,'required_recordings':128,
        'maximum_inspected_recordings':160,'priority_rows':rows[:160],
        'sole_exclusion':'native_left_mp4_object_absent_in_complete_published_directory',
        'metadata_alias_resolution':'at_most4_all_fields_equal_except_uuid_user_id_and_lab_capitalization_then_lexical_first',
        'network_identity_metadata_decode_failures_excludable':False,
        'success_lab_quality_or_model_outcome_filter':False,'raw_video_or_states_read_before_freeze':False,
        'fit_or_gpu_execution_authorized_by_this_receipt':False,'authors8000_subset_reproduced':False})
    write_json(output/'FROZEN.json',{'protocol_sha256':sha256(output/'protocol.json')})


def resolve(frozen, output):
    protocol=json.loads((frozen/'protocol.json').read_text())
    if (sha256(frozen/'protocol.json')!=json.loads((frozen/'FROZEN.json').read_text())['protocol_sha256'] or
            protocol['source_sha256']!=sha256(Path(__file__))):
        raise ValueError('Availability contract changed')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',{'freeze_sha256':sha256(frozen/'protocol.json'),'model_calls':0})
    selected,excluded=[],[]
    try:
        for rank,row in enumerate(protocol['priority_rows']):
            target=output/f'priority-{rank:03d}'
            record=inspect_record(row,target)
            entry={'priority_rank':rank,'report_sha256':sha256(target/'report.json'),**record}
            (selected if record['native_left_video_object'] else excluded).append(entry)
            print(json.dumps({'examined':rank+1,'resolved':len(selected),'missing_camera_exclusions':len(excluded)}),flush=True)
            if len(selected)==128:break
        if len(selected)!=128:raise ValueError('Frozen bounded availability search exhausted')
        write_json(output/'cohort.json',{'role':'input_objects_only_not_fit_eligibility','selected':selected,
            'excluded':excluded,'freeze_sha256':sha256(frozen/'protocol.json'),
            'native_loader_parity_passed':False,'evaluation_disjointness_audited':False,
            'independent_families_audited':False,'authors8000_subset_reproduced':False})
        write_json(output/'report.json',{'status':'128_native_input_object_sets_resolved_not_model_fit',
            'protocol_sha256':sha256(output/'protocol.json'),'cohort_sha256':sha256(output/'cohort.json'),
            'selected_recordings':128,'excluded_missing_camera':len(excluded),'model_calls':0})
        write_json(output/'DONE.json',{'report_sha256':sha256(output/'report.json')})
    except Exception as exc:
        write_json(output/'FAILED.json',{'error':str(exc),'partial_inputs_preserved':True});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','resolve'))
    p.add_argument('--output',type=Path,required=True)
    for name in ('catalogue','original','freeze'):p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.mode=='freeze':freeze(args.catalogue,args.original,args.output)
    else:resolve(args.freeze,args.output)


if __name__=='__main__':main()
