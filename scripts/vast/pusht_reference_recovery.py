"""Read-only NJ recovery of immutable Push-T source and all native references.

No California connection, candidate assignment, GPU calls or remote writes.
Large public inputs are hash-verified on NJ; only compact evidence is recovered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
PROOF = PROJECT / 'artifacts/offline_study/pusht-native-recovery-20260908-v1'
REMOTE_ROOT = '/workspace/jepa-runtime'
SOURCE = '42cb7df90b82a71996687ac719c22a0ab9e4fd4f2dccf4ebd19c6847a841a27c'
FREEZE = '2ddf2e1457a093ac48288842fd6ea81bb021a276317993fa22d570cb809a04d0'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def safe_member(name):
    path = Path(name)
    if (not name or path.is_absolute() or '..' in path.parts or str(path) != name or
            path.parts[0] not in ALLOWED_ROOTS):
        raise ValueError('Unregistered recovery path: ' + name)
    return path


ALLOWED_ROOTS = ('pusht-coupling-code-20260908-v2', 'pusht-coupling-evidence-20260908-v2',
    'pusht-coupling-behavior-20260908-v2', 'pusht-planning-code-20260908-v1',
    'pusht-planning-evidence-20260908-v1', 'pusht-planning-native-20260908-v1',
    'pusht-planning-assets-20260908-v3')


REMOTE = r'''
import argparse,contextlib,hashlib,importlib.metadata,json,pathlib,subprocess,sys
root=pathlib.Path('/workspace/jepa-runtime'); request=json.load(sys.stdin)
code=root/'pusht-coupling-code-20260908-v2'
sys.path.insert(0,str(code/'src'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def check(p,want):
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=want['bytes'] or sha(p)!=want['sha256']:
  raise ValueError('Frozen member changed: '+str(p))
for name,want in request['preparation'].items():check(root/name,want)
native=root/'pusht-planning-native-20260908-v1'
for name,want in request['native_delivery'].items():check(native/name,want)
freeze=root/'pusht-coupling-behavior-20260908-v2/freeze'
assert sha(freeze/'protocol.json')==request['freeze_sha256']
assert read(freeze/'FROZEN.json')['protocol_sha256']==request['freeze_sha256']
evidence=root/'pusht-coupling-evidence-20260908-v2'
assets=root/'pusht-planning-assets-20260908-v3'
checkpoint=root/'pusht-planning-assets-20260908-v2/downloads/models/jepa_wm_pusht.pth.tar'
kwargs=dict(vendor=pathlib.Path('/workspace/jepa_steering/vendor/jepa-wms'),reference=native,
 reference_code=root/'pusht-planning-code-20260908-v1',fit=evidence/'fits/vision_action_coupling',
 cohort=evidence/'cohort/cohort.json',data_root=assets/'source/data/pusht_noise',freeze=freeze,checkpoint=checkpoint)
with contextlib.redirect_stdout(sys.stderr):
 from offline_study.author_fit import source_hash
 from offline_study.pusht_coupling_behavior import read_contract,load_native
 assert source_hash()==request['source_sha256']
 protocol,fit,cohort,native_protocol,smoke=read_contract(argparse.Namespace(**kwargs))
 rows,bindings=load_native(native,native_protocol,cohort)
assert len(rows)==96 and len(bindings)==8
report=read(native/'report.json')
assert read(native/'DONE.json')['report_sha256']==sha(native/'report.json')
assert report['status']=='all96_native_pusht_replication_episodes_verified' and report['episodes']==96
assert report['freeze_sha256']==protocol['bindings']['native_freeze_sha256']
for rank in range(8):assert report['shard_reports_sha256'][str(rank)]==sha(native/f'native/shard-{rank}/report.json')
assert sha(checkpoint)==protocol['bindings']['checkpoint_sha256']
receipt=read(assets/'receiving_report.json')
assert read(assets/'RECEIVING_DONE.json')['report_sha256']==sha(assets/'receiving_report.json')
assert receipt['status']=='all_native_planning_and_fit_check_input_bytes_verified'
full=read(assets/'source/report.json')
assert read(assets/'source/DONE.json')['report_sha256']==sha(assets/'source/report.json')
assert full['files_sha256']==sha(assets/'source/files.json')
assert full['archive_sha256']=='442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08'
all_inputs=read(assets/'source/files.json')
public={}
for name,want in receipt['files'].items():
 assert all_inputs[name]==want
 p=assets/'source/data'/name;check(p,want);public[str(p.relative_to(root))]=want
public[str(checkpoint.relative_to(root))]={'bytes':checkpoint.stat().st_size,'sha256':sha(checkpoint)}
members=dict(request['preparation'])
def add(p):
 if p.is_symlink() or not p.is_file():raise ValueError('Only regular compact evidence allowed')
 name=str(p.relative_to(root))
 if p.name in ('.env','rclone.conf','credentials.json') or p.suffix in ('.pem','.key'):
  raise ValueError('Credential-like member refused')
 members[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
for base in (root/'pusht-planning-code-20260908-v1',root/'pusht-planning-evidence-20260908-v1',native,freeze):
 for p in sorted(base.rglob('*')):
  if '__pycache__' in p.parts or p.name.startswith('._') or p.name=='progress.json':continue
  if p.is_file():add(p)
for name in ('receiving_report.json','RECEIVING_DONE.json','source/protocol.json','source/files.json','source/report.json','source/DONE.json'):add(assets/name)
versions={}
for package in ('torch','numpy','opencv-python','pygame','pymunk','shapely','omegaconf'):
 try:versions[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError:versions[package]=None
print(json.dumps({'status':'all96_native_and_frozen_pusht_inputs_verified_cpu_only',
 'source_sha256':request['source_sha256'],'freeze_sha256':request['freeze_sha256'],
 'native_report_sha256':sha(native/'report.json'),'native_freeze_sha256':protocol['bindings']['native_freeze_sha256'],
 'native_engineering_report_sha256':protocol['bindings']['native_engineering_report_sha256'],
 'native_streams':list(range(8)),'native_episodes':96,'native_report_bindings':bindings,
 'unchanged_scientific_validators':['read_contract','load_native'],
 'members':members,'public_inputs_verified_not_copied':public,'runtime_versions':versions,
 'gpu_calls':0,'candidate_outcomes_accessed':False,'remote_writes':0,'california_contact':False},sort_keys=True))
'''


def verify_archive(path, members):
    seen = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            safe_member(member.name)
            if member.name in seen or member.name not in members or not member.isfile():
                raise ValueError('Unexpected/duplicate/nonregular archive member')
            wanted = members[member.name]
            if member.size != wanted['bytes']:
                raise ValueError('Archive member size changed')
            h = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(4 << 20), b''):
                    h.update(block)
            if h.hexdigest() != wanted['sha256']:
                raise ValueError('Archive member hash changed')
            seen.add(member.name)
    if seen != set(members):
        raise ValueError('Incomplete archive')
    return {'files': len(seen), 'archive_bytes': path.stat().st_size, 'archive_sha256': sha(path)}


def recover():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    if ('13:53Z Push-T native preservation / CPU preparation' not in board or
            'geometry_reference_check' not in board):
        raise ValueError('Explicit preservation reservation required')
    # Query only NJ, never the quarantined California resource.
    row = json.loads(subprocess.check_output(['/Users/stevenyang/.local/bin/vastai',
        'show', 'instance', '50239185', '--raw'], text=True))
    if (row['id'] != 50239185 or row['label'] != 'jepa-pusht-wall-history-us-v1' or
            row['geolocation'] != 'New Jersey, US' or row['actual_status'] != 'running' or row['intended_status'] != 'running'):
        raise ValueError('Owned source lease changed')
    connection = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-p',
        str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    PROOF.mkdir(exist_ok=False)
    write(PROOF / 'AUTHORITY.json', {'owner': 'rep_geometry_transcoder/root', 'instance': 50239185,
        'source_label': row['label'], 'geography': row['geolocation'], 'checked_unix': time.time(),
        'board_sha256': hashlib.sha256(board.encode()).hexdigest(), 'script_sha256': sha(Path(__file__)),
        'provider_queries': [50239185], 'remote_writes': 0, 'gpu_calls': 0})
    request = {'source_sha256': SOURCE, 'freeze_sha256': FREEZE,
        'preparation': json.loads((PROJECT / 'artifacts/offline_study/pusht-coupling-preparation-20260908-v2/FILES.json').read_text()),
        'native_delivery': json.loads((PROJECT / 'artifacts/offline_study/pusht-reference-delivery-20260908-v1/FILES.json').read_text())}
    command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'LD_LIBRARY_PATH=/opt/conda/lib',
        'PYTHONPATH=/workspace/jepa-python/lib/python3.10/site-packages',
        '/workspace/jepa-planning-python/bin/python', '-c', REMOTE]
    print(json.dumps({'stage': 'NJ_read_only_frozen_source_native96_and_public_input_validation'}), flush=True)
    with (PROOF / 'CPU_VALIDATION.log').open('x') as log:
        result = subprocess.check_output(connection + [shlex.join(command)], input=json.dumps(request),
            text=True, stderr=log, timeout=300)
    verified = json.loads(result)
    write(PROOF / 'SOURCE_VERIFIED.json', verified)
    members = verified['members']
    if sum(x['bytes'] for x in members.values()) > 100_000_000:
        raise ValueError('Compact recovery unexpectedly exceeds 100MB; do not transfer public frames')
    for name in members:
        safe_member(name)
    write(PROOF / 'FILES.json', members)
    write(PROOF / 'PUBLIC_INPUTS.json', verified['public_inputs_verified_not_copied'])
    print(json.dumps({'stage': 'compact_native_source_evidence_recovery', 'members': len(members),
        'bytes': sum(x['bytes'] for x in members.values())}), flush=True)
    archive = PROOF / 'native-source-evidence.tar.gz'
    remote_tar = ['tar', '--no-recursion', '--null', '-C', REMOTE_ROOT, '-czf', '-', '-T', '-']
    with archive.open('xb') as stream:
        subprocess.run(connection + [shlex.join(remote_tar)], input=b'\0'.join(n.encode() for n in sorted(members)) + b'\0',
            stdout=stream, check=True, timeout=300)
    proof = verify_archive(archive, members)
    # Recheck source-selected bytes after transfer. Never overwrite local evidence.
    recheck = "import pathlib,hashlib,json,sys;root=pathlib.Path(sys.argv[1]);m=json.load(sys.stdin);assert all(not (root/n).is_symlink() and (root/n).stat().st_size==v['bytes'] and hashlib.sha256((root/n).read_bytes()).hexdigest()==v['sha256'] for n,v in m.items());print(json.dumps({'source_selected_members_unchanged':len(m)}))"
    after = json.loads(subprocess.check_output(connection + [shlex.join(['/usr/bin/python3', '-c', recheck, REMOTE_ROOT])],
        input=json.dumps(members), text=True, timeout=120))
    write(PROOF / 'DONE.json', {'status': 'all96_native_frozen_source_fit_input_evidence_recovered_verified',
        **proof, **after, 'source_sha256': SOURCE, 'freeze_sha256': FREEZE,
        'files_sha256': sha(PROOF / 'FILES.json'), 'source_receipt_sha256': sha(PROOF / 'SOURCE_VERIFIED.json'),
        'public_inputs_manifest_sha256': sha(PROOF / 'PUBLIC_INPUTS.json'), 'native_episodes': 96,
        'native_streams': list(range(8)), 'public_frames_and_checkpoint_reverified_on_NJ_not_relayed': True,
        'candidate_results_recovered': 0, 'candidate_partition_created': False,
        'california_contact': False, 'gpu_calls': 0, 'remote_writes': 0})
    print(json.dumps({'status': 'native96_compact_recovery_complete', **proof,
        'done_sha256': sha(PROOF / 'DONE.json')}), flush=True)


def self_test():
    import io
    import tempfile
    import unittest
    class RecoveryTests(unittest.TestCase):
        def test_paths(self):
            for name in ('/etc/passwd', '../escape', 'other/file', 'pusht-planning-native-20260908-v1/../x'):
                with self.assertRaises(ValueError):safe_member(name)
        def test_archive_exactness(self):
            name='pusht-planning-native-20260908-v1/native/shard-0/DONE.json'
            raw=b'{}'; members={name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'proof.tar.gz'
                with tarfile.open(path,'w:gz') as archive:
                    item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
                self.assertEqual(verify_archive(path,members)['files'],1)
                members[name]['sha256']='0'*64
                with self.assertRaises(ValueError):verify_archive(path,members)
        def test_duplicate_refused(self):
            name='pusht-planning-native-20260908-v1/DONE.json';raw=b'{}'
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'proof.tar.gz'
                with tarfile.open(path,'w:gz') as archive:
                    for _ in range(2):
                        item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
                with self.assertRaises(ValueError):verify_archive(path,{name:{'bytes':2,'sha256':hashlib.sha256(raw).hexdigest()}})
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('self-test','recover'))
    args=parser.parse_args()
    self_test() if args.command=='self-test' else recover()
