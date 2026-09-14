"""Native clip eligibility before fitting; preserve failed input proposals.

Never exclude on labels, model outcomes, decode failures or malformed data.
Only missing native MP4 and mathematically impossible native four-frame windows
may be skipped in the original deterministic priority order.
"""
from offline_study._paths import source_path
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from offline_study.tasks.droid.droid_fit_availability import inspect_record, order, CATALOGUE
from offline_study.tasks.droid.droid_fit_download import download_object
from offline_study.core.protocol import sha256, write_json


def native_clip_possible(frames,fps):
    if (not isinstance(frames,int) or frames<1 or not math.isfinite(fps) or fps<=0):
        raise ValueError('Malformed video cannot be silently excluded')
    # Native ef=randint(fpc*ceil(vfps/fps),vlen) needs a nonempty interval.
    return frames>4*math.ceil(fps/4)


def freeze(catalogue,proposal,output):
    if sha256(catalogue)!=CATALOGUE:raise ValueError('Wrong source catalogue')
    proposed=json.loads((proposal/'cohort.json').read_text())
    if (len(proposed['selected'])!=128 or sha256(proposal/'report.json')!=json.loads((proposal/'DONE.json').read_text())['report_sha256']
            or json.loads((proposal/'report.json').read_text())['cohort_sha256']!=sha256(proposal/'cohort.json')):
        raise ValueError('Original metadata proposal changed')
    rows=order(json.loads(catalogue.read_text()))
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',{'role':'native_clip_eligibility_before_intervention_fit',
        'source_sha256':{name:sha256(source_path(name)) for name in
            ('droid_fit_eligible.py','droid_fit_availability.py','droid_fit_download.py','droid_fit_inputs.py')},
        'catalogue_sha256':CATALOGUE,'original_metadata_cohort_sha256':sha256(proposal/'cohort.json'),
        'priority_rows':rows[:160],'required_recordings':128,'maximum_examined':160,
        'allowed_exclusions':['native_left_camera_object_absent','video_frames_le_4_times_ceil_actual_fps_over4'],
        'all_other_input_failures_abort':True,'success_lab_quality_or_model_outcome_filter':False,
        'previous_failed_input_proposals_retained':True,'evaluation_population_unchanged':True,
        'author8000_training_manifest_reproduced':False,'intervention_fitting_or_evaluation_launched':False})
    write_json(output/'FROZEN.json',{'protocol_sha256':sha256(output/'protocol.json')})


def run(args):
    from decord import VideoReader,cpu
    p=json.loads((args.freeze/'protocol.json').read_text())
    if sha256(args.freeze/'protocol.json')!=json.loads((args.freeze/'FROZEN.json').read_text())['protocol_sha256']:
        raise ValueError('Eligibility freeze changed')
    for name,digest in p['source_sha256'].items():
        if sha256(source_path(name))!=digest:raise ValueError('Frozen input source changed')
    if sha256(args.metadata/'cohort.json')!=p['original_metadata_cohort_sha256']:raise ValueError('Metadata cohort changed')
    previous=json.loads((args.inputs/'report.json').read_text())
    if (sha256(args.inputs/'report.json')!=json.loads((args.inputs/'DONE.json').read_text())['report_sha256'] or
            previous['files_sha256']!=sha256(args.inputs/'FILES.json')):raise ValueError('Original raw inputs changed')
    original_files=json.loads((args.inputs/'FILES.json').read_text())
    original_objects={row['name']:row for row in json.loads((args.inputs/'protocol.json').read_text())['objects']}
    cohort=json.loads((args.metadata/'cohort.json').read_text())
    existing={row['priority_rank']:row for row in cohort['selected']+cohort['excluded']}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'eligibility_protocol.json',{'freeze_sha256':sha256(args.freeze/'protocol.json'),
        'input_report_sha256':sha256(args.inputs/'report.json'),'model_calls':0})
    selected,excluded,files,objects=[],[],{},{}
    try:
        for priority,row in enumerate(p['priority_rows']):
            if priority in existing:
                record=existing[priority]
                proof=args.metadata/f'priority-{priority:03d}'
                if sha256(proof/'report.json')!=record['report_sha256'] or record['source_trajectory']!=row:
                    raise ValueError('Existing metadata priority identity changed')
            else:
                proof=args.output/f'priority-{priority:03d}'
                record=inspect_record(row,proof)
            if record['native_left_video_object'] is None:
                excluded.append({'priority_rank':priority,'reason':'native_left_camera_object_absent',
                    'metadata_report_sha256':sha256(proof/'report.json')});continue
            required=[row,record['native_left_video_object'],*record['all_metadata_alias_objects']]
            candidate_files={}
            for obj in required:
                target=args.output/'raw'/obj['name']
                if obj['name'] in original_files:
                    source=args.inputs/'raw'/obj['name'];want=original_files[obj['name']]
                    if (original_objects[obj['name']]!=obj or source.is_symlink() or source.stat().st_size!=want['bytes']
                            or sha256(source)!=want['sha256']):raise ValueError('Existing immutable source changed')
                    target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
                    verified=want
                else:
                    downloaded=download_object(obj,args.output/'raw')
                    verified={'sha256':downloaded['sha256'],'bytes':downloaded['bytes']}
                candidate_files[obj['name']]=verified
            camera=args.output/'raw'/record['native_left_video_object']['name']
            vr=VideoReader(str(camera),num_threads=1,ctx=cpu(0));frames=len(vr);fps=float(vr.get_avg_fps());del vr
            if not native_clip_possible(frames,fps):
                excluded.append({'priority_rank':priority,'reason':'video_frames_le_4_times_ceil_actual_fps_over4',
                    'frames':frames,'fps':fps,'source_files':candidate_files,
                    'metadata_report_sha256':sha256(proof/'report.json')});continue
            selected.append({'priority_rank':priority,'directory':str(Path(row['name']).parent),
                'metadata_names':[obj['name'] for obj in record['all_metadata_alias_objects']],
                'left_camera_name':record['native_left_video_object']['name'],
                'video_frames':frames,'source_fps':fps,'metadata_report_sha256':sha256(proof/'report.json')})
            files.update(candidate_files);objects.update({obj['name']:obj for obj in required})
            print(json.dumps({'examined':priority+1,'usable_recordings':len(selected),'exclusions':len(excluded),'model_calls':0}),flush=True)
            if len(selected)==128:break
        if len(selected)!=128:raise ValueError('Frozen native-input priority bound exhausted')
        write_json(args.output/'eligibility.json',{'selected':selected,'excluded':excluded,
            'native_camera_and_clip_length_only':True,'decode_failures_skipped':False,'model_calls':0})
        write_json(args.output/'protocol.json',{'role':'explicit_native_eligible_input_amendment_not_model_fit',
            'eligibility_protocol_sha256':sha256(args.output/'eligibility_protocol.json'),
            'eligibility_sha256':sha256(args.output/'eligibility.json'),'objects':list(objects.values()),
            'recordings':selected,'inherited_raw_bytes_not_copied_twice':True,'model_calls':0})
        write_json(args.output/'FILES.json',files)
        write_json(args.output/'report.json',{'status':'all128_native_recording_objects_downloaded_and_verified_not_fit',
            'protocol_sha256':sha256(args.output/'protocol.json'),'files_sha256':sha256(args.output/'FILES.json'),
            'recordings':128,'objects':len(files),'bytes':sum(item['bytes'] for item in files.values()),
            'excluded_inputs':len(excluded),'native_loader_parity_passed':False,
            'family_disjointness_verified':False,'model_calls':0,'author8000_subset_reproduced':False})
        write_json(args.output/'DONE.json',{'report_sha256':sha256(args.output/'report.json')})
    except Exception as exc:
        write_json(args.output/'FAILED.json',{'error':str(exc),'partial_inputs_retained':True});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('freeze','run'))
    for name in ('catalogue','proposal','freeze','metadata','inputs'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.mode=='freeze':freeze(args.catalogue,args.proposal,args.output)
    else:run(args)
