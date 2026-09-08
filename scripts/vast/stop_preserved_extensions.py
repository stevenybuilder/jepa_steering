"""Stop, never destroy, explicitly paused owned rentals after verified Drive readback."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from preserve_core_pause import HOSTS, PROJECT

LABELS = {'tx': 'jepa-droid-parallel-us-v3', 'ne': 'jepa-fixed-offline-us-v1',
          'in': 'jepa-navigation-offline-indiana', 'nj': 'jepa-pusht-wall-history-us-v1'}
FOLDER = '12r-UzKMTuyYm4wXKl9xPb3r5dcjfwsYr'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=HOSTS, required=True)
    args = parser.parse_args()
    root = PROJECT / 'artifacts/offline_study/core-priority-pause-20260908-v1' / args.worker
    local = json.loads((root / 'LOCAL_VERIFIED.json').read_text())
    path = root / ('direct-drive' if args.worker in ('tx', 'nj') else 'drive-readback') / 'VERIFIED.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['parent_folder_id'] != FOLDER:
            raise ValueError('Unexpected Drive destination')
    elif args.worker in ('tx', 'nj') and (root / 'REPLICA_VERIFIED.json').exists():
        saved = json.loads((root / 'REPLICA_VERIFIED.json').read_text())
        if saved['replica_instance'] != 50233992 or saved['all_source_disks_must_be_retained'] is not True:
            raise ValueError('Unexpected retained storage replica')
    elif args.worker in ('tx', 'nj') and (root / 'CLOUD_VERIFIED.json').exists():
        saved = json.loads((root / 'CLOUD_VERIFIED.json').read_text())
        if saved['cloud_uri'] != 'gs://rgt-jepa-archive-2026/rep_geometry_transcoder/core-priority-pause-20260908-v1/' + args.worker + '-paused-evidence.tar.gz':
            raise ValueError('Unexpected Google Cloud archive')
    else:
        raise ValueError('No verified durable preservation')
    if (saved['archive_sha256'] != local['archive_sha256'] or
            saved['archive_bytes'] != local['archive_bytes'] or saved['files'] != local['files'] or
            saved['status'] not in ('provider_sha256_and_full_member_readback_verified',
                                   'provider_sha256_and_full_member_stream_readback_verified',
                                   'retained_la_replica_full_byte_hash_verified',
                                   'google_cloud_compressed_and_member_readback_verified')):
        raise ValueError('Verified durable archive binding required')
    instance, port, address = HOSTS[args.worker]
    cli = '/Users/stevenyang/.local/bin/vastai'
    rows = json.loads(subprocess.check_output([cli, 'show', 'instances', '--raw'], text=True))
    current = next(r for r in rows if r['id'] == instance)
    if (current['label'] != LABELS[args.worker] or current['cur_state'] != 'running' or
            current['actual_status'] != 'running' or not current['geolocation'].endswith(', US') or
            current['public_ipaddr'] != address):
        raise ValueError('Owned live US identity changed; no lifecycle operation')
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-p', str(port), 'root@' + address]
    check = '''import sys,json,pathlib,subprocess
sys.path.insert(0,'/workspace/jepa-runtime')
import pause_deferred_workers as p
root=pathlib.Path('/workspace/jepa-runtime/core-priority-pause-20260908-v1')
bound=json.loads((root/'BOUND_PROCESSES.json').read_text())
if any(p.live(x) for x in bound): raise ValueError('Bound processes still live')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(): raise ValueError('GPU still in use')
print(json.dumps({'all_bound_processes_terminal':True,'all_gpus_empty':True}))
'''
    result = json.loads(subprocess.check_output(ssh + [shlex.join(['/usr/bin/python3', '-c', check])], text=True))
    target = root / 'STOP_INTENT.json'
    with target.open('x') as f:
        json.dump({'instance': instance, 'label': current['label'], 'geolocation': current['geolocation'],
                   'drive_file_id': saved.get('file_id'), 'replica_root': saved.get('replica_root'),
                   'cloud_uri': saved.get('cloud_uri'),
                   'preservation_status': saved['status'], 'archive_sha256': saved['archive_sha256'],
                   'process_check': result, 'operation': 'stop_not_destroy', 'source_volume_retained': True,
                   'checked_unix': time.time()}, f, indent=2)
    response = subprocess.check_output([cli, 'stop', 'instance', str(instance)], text=True)
    with (root / 'STOP_RESPONSE.json').open('x') as f:
        json.dump({'instance': instance, 'response': response, 'time': time.time()}, f, indent=2)
    print(response, flush=True)


if __name__ == '__main__':
    main()
