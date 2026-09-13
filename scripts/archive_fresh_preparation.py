"""Preserve the bounded preparation outputs in the existing private Drive archive."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/offline_study'
OUT = ART / 'protected-preparation-preservation-20260912-v1'
DRIVE = 'gdrive:Research-Archives/JEPA-WM/protected-preparation-20260912-v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    # Copy/verify local receiving checks before creating a preservation receipt.
    native = ART / 'protected-input-cpu-validation-20260912-v1/native-droid-diagnostic-v1'
    rows = json.loads((native / 'report.json').read_text())['records']
    assert len(rows) == 64 and len({r['directory'] for r in rows}) == 64
    for row in rows:
        assert digest(native / row['native_file']) == row['native_sha256']
    OUT.mkdir(exist_ok=False)
    archive = OUT / 'prepared-inputs-runtime-source.tgz'
    names = ['protected-preparation-20260912-v1', 'protected-input-cpu-validation-20260912-v1',
             'fresh-simulator-banks-20260912-v1', 'protected-preparation-worker-20260912-v1']
    scripts = ['prepare_protected_inputs.py', 'validate_protected_inputs_cpu.py',
               'prepare_fresh_simulator_banks.py', 'run_fresh_bank_preparation.py',
               'diagnose_droid_input_parity.py', 'audit_droid_native_inputs.py',
               'capture_preparation_runtime.py', 'archive_fresh_preparation.py',
               'vast/protected_preparation_lease.py', 'vast/bootstrap_protected_preparation.sh',
               'vast/bootstrap_protected_pointmaze.sh']
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for name in names:
            tar.add(ART / name, arcname='artifacts/offline_study/' + name)
        for name in scripts:
            tar.add(ROOT / 'scripts' / name, arcname='scripts/' + name)
        tar.add(ROOT / 'protected_evaluation_repair.md', arcname='protected_evaluation_repair.md')
    record = {'archive': archive.name, 'sha256': digest(archive), 'bytes': archive.stat().st_size,
              'drive': DRIVE, 'created_unix': time.time(), 'scientific_launch_ready': False}
    (OUT / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    subprocess.run(['rclone', 'copy', str(OUT), DRIVE, '--immutable', '--transfers', '2'], check=True)
    subprocess.run(['rclone', 'check', str(OUT), DRIVE, '--download', '--one-way'], check=True)
    (OUT / 'VERIFIED.json').write_text(json.dumps({**record, 'download_check_passed': True}, indent=2) + '\n')
    subprocess.run(['rclone', 'copyto', str(OUT / 'VERIFIED.json'), DRIVE + '/VERIFIED.json', '--immutable'], check=True)
    print(json.dumps(record), flush=True)
