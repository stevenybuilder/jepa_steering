"""Build a private, immutable confirmation bundle without touching original files."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/offline_study/confirmation-20260911-v1'


def main():
    stage = Path(tempfile.mkdtemp(prefix='jepa-confirmation-source-'))
    ignore = shutil.ignore_patterns('__pycache__', '._*', '.DS_Store')
    for name in ('src', 'vendor/jepa-wms'):
        shutil.copytree(PROJECT/name, stage/name, ignore=ignore)
    # Preserve the exact historical numerical snapshot's complete file contract.
    # These two paused-task modules are not imported/executed by MetaWorld.
    # Do NOT revert the user's current main versions.
    reference = PROJECT/'artifacts/offline_study/fixed-response-20260908-v1/numerical-source-v1/src/offline_study'
    pins = {}
    for name in ('training_pilot.py', 'pusht_planning_replication.py'):
        shutil.copy2(reference/name, stage/'src/offline_study'/name)
        pins[name] = hashlib.sha256((reference/name).read_bytes()).hexdigest()
    fixed = 'artifacts/offline_study/fixed-response-20260908-v1/'
    paths = [fixed+name for name in ('fits','numerical-checks','numerical-source-v1')]
    paths += ['artifacts/offline_study/restored-behavioral-inputs-20260908-v1']
    primary = 'artifacts/offline_study/primary-durable-20260907/'
    paths += [primary+'fits-v1/bfloat16/'+t+'/vision_action_coupling' for t in ('reach','reach-wall')]
    paths += [primary+'planning-scenarios-20260907']
    for name in paths:
        shutil.copytree(PROJECT/name, stage/name, ignore=ignore)
    for task in ('reach', 'reach-wall'):
        name = primary+'fits-v1/bfloat16/'+task+'/PARITY.json'
        shutil.copy2(PROJECT/name, stage/name)
    freeze = stage/'artifacts/offline_study/confirmation-20260911-v1/freeze-v2'
    freeze.mkdir(parents=True)
    shutil.copy2(ROOT/'EXPOSURE_AUDIT.json', freeze/'EXPOSURE_AUDIT.json')
    env = dict(os.environ, PYTHONPATH=str(stage/'src'))
    subprocess.run([str(PROJECT/'.venv/bin/python'), '-m', 'offline_study.confirmation', 'freeze',
        '--project', str(stage), '--vendor', str(stage/'vendor/jepa-wms'), '--output', str(freeze)],
        cwd=stage, env=env, check=True)
    subprocess.run([str(PROJECT/'.venv/bin/python'), '-c',
        'from pathlib import Path; from offline_study.fixed_response_smoke import verify_numerical; '
        'from offline_study.planning_native_smoke import CHECKPOINTS; '
        'b=Path("artifacts/offline_study/fixed-response-20260908-v1"); '
        'print([verify_numerical(b/"numerical-checks"/t,b/"fits"/t,b/"numerical-source-v1/src/offline_study",'
        't,CHECKPOINTS["metaworld"])["report_sha256"] for t in ("reach","reach-wall")])'],
        cwd=stage, env=env, check=True)
    shutil.copytree(freeze, ROOT/'freeze-v2')
    for name in ('run_confirmation_worker.py','bootstrap_confirmation.sh','preserve_confirmation_worker.py','launch_confirmation.py'):
        (stage/'scripts/vast').mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT/'scripts/vast'/name, stage/'scripts/vast'/name)
    archive = ROOT/'confirmation-inputs.tar.gz'
    with tarfile.open(archive, 'x:gz') as out:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                out.add(path, arcname=str(path.relative_to(stage)), recursive=False)
    receipt = {'stage': str(stage), 'archive': str(archive), 'bytes': archive.stat().st_size,
        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'isolated_historical_runtime_pins': pins, 'main_source_untouched': True,
        'supersedes_prelaunch_top_level_freeze': True, 'no_confirmation_policy_outcomes_observed': True}
    with (ROOT/'PACKAGE.json').open('x') as output:
        json.dump(receipt, output, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
