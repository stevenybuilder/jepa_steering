"""Stop only owned LA50233992 after verified core and Google Drive preservation.

Never destroy storage, restart a lease, or repeat a possibly accepted stop.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from navigation_redistribution_common import digest, write
from preserve_completed_core import PROJECT, STATUS, remote

ROOT = PROJECT / 'artifacts/offline_study/core-completion-preservation-20260908-v1'
CLI = '/Users/stevenyang/.local/bin/vastai'
FOLDER = '1SoXkmEJmrCZXBnKk3re6mX8WwJiWBokY'


def verify_preservation(root):
    read = lambda name: json.loads((root / name).read_text())
    cloud = read('VERIFIED.json'); science = read('SCIENCE_VERIFIED.json')
    drive = read('connector-parts-v1/stream-readback-v1/VERIFIED.json')
    spec = read('connector-parts-v1/PARTS.json')
    supplemental = read('la-supplement-v2/drive-readback-v1/VERIFIED.json')
    metadata = read('DRIVE_METADATA_VERIFIED.json')
    if (science['status'] != 'all960_core_records_and_frozen_analysis_recomputed_exact'
            or science['report_sha256'] != cloud['report_sha256']
            or drive['status'] != 'all_parts_and_rejoined_archive_members_verified'
            or drive['archive_sha256'] != cloud['archive_sha256']
            or drive['archive_bytes'] != cloud['archive_bytes'] or drive['files'] != 6872
            or drive['manifest_sha256'] != digest(root/'FILES.json')
            or len(drive['parts']) != 14 or spec['folder_id'] != FOLDER
            or [p['id'] for p in drive['parts']] != [p['drive_file_id'] for p in spec['parts_in_join_order']]
            or any(p['parents'] != [FOLDER] for p in drive['parts'])
            or supplemental['status'] != 'all_parts_and_rejoined_archive_members_verified'
            or supplemental['files'] != 661
            or supplemental['archive_sha256'] != 'a730c5cd34fa282458a59f1fc32a54d0482160aae979640abb391a05c56872bd'
            or supplemental['manifest_sha256'] != digest(root/'la-supplement-v2/FILES.json')
            or any(p['parents'] != [FOLDER] for p in supplemental['parts'])
            or metadata['status'] != 'all_restore_files_provider_and_full_byte_readback_verified'
            or metadata['file_count'] != 13 or metadata['folder_id'] != FOLDER):
        raise ValueError('Complete science, Drive archives and restore metadata required')
    required = {'core-PARTS.json','core-FILES.json','core-SCIENCE_VERIFIED.json','ANALYSIS_REPORT.json',
        'core-GCS_VERIFIED.json','supplement-FILES.json','supplement-INVENTORY.json',
        'supplement-DRIVE_VERIFIED.json','ENVIRONMENT.json','RESTORE.md',
        'CORE_METAWORLD_BEHAVIORAL_RESULTS.md','EXPERIMENT_PLAN.md','study.json'}
    if {x['name'] for x in metadata['files']} != required:
        raise ValueError('Incomplete restore metadata scope')
    if any(digest(PROJECT / x['path']) != x['sha256'] for x in metadata['files']):
        raise ValueError('Restore metadata changed after verification')
    return {'core_archive_sha256':drive['archive_sha256'], 'core_files':drive['files'],
            'supplement_files':supplemental['files'], 'drive_folder_id':FOLDER,
            'metadata_receipt_sha256':digest(root/'DRIVE_METADATA_VERIFIED.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(); proof = verify_preservation(ROOT)
    if not args.execute:
        print(json.dumps({'preservation_gates_pass':True, **proof})); return
    rows = json.loads(subprocess.check_output([CLI,'show','instances','--raw'],text=True,timeout=60))
    current = next(x for x in rows if x['id'] == 50233992)
    if (current['label'] != 'jepa-fixed-behavior-us-v1' or current['cur_state'] != 'running'
            or current['actual_status'] != 'running' or current['num_gpus'] != 8
            or not current['geolocation'].endswith(', US') or current['public_ipaddr'] != '98.142.241.142'):
        raise ValueError('Live owned US rental identity differs')
    status = remote(STATUS,timeout=45)
    if (status['complete_shards'] != 40 or status['published_files'] != 960
            or not status['analysis_done'] or status['live'] or status['failed']):
        raise ValueError('Core completion or terminal-process gate differs')
    device = remote('import json,subprocess\np=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True)\nif p.strip(): raise ValueError("GPU still in use")\nprint(json.dumps({"all_gpus_empty":True}))',timeout=45)
    intent = {'instance':50233992,'operation':'stop_not_destroy','time':time.time(),
        'label':current['label'],'geolocation':current['geolocation'],'proof':proof,
        'core_status':status,'gpu_status':device,'source_volume_retained':True}
    write(ROOT/'STOP_INTENT.json',intent)
    response = subprocess.check_output([CLI,'stop','instance','50233992'],text=True,timeout=90)
    write(ROOT/'STOP_RESPONSE.json',{'response':response,'time':time.time(),'instance':50233992})
    print(response,flush=True)


if __name__ == '__main__': main()
