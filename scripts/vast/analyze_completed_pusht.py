"""Assemble checksum-bound completed streams and invoke the frozen CPU analysis.

No new model calls, treatment selection, or changed scientific analysis.
Archives must be accompanied by the collector's full Drive verification receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from table_resume_streams import ARMS, command, inventory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), 'Conflicting immutable evidence'
    else:
        shutil.copyfile(source, target)


def copy_tree(source, target):
    for p in source.rglob('*'):
        if p.is_file():
            assert not p.is_symlink()
            copy(p, target / p.relative_to(source))


def main(a):
    out = a.output or a.root/'expansion-20260911-v1/pusht-complete-audit'
    out.mkdir(parents=True, exist_ok=False)
    try:
        panel = out/'panel'
        old, incomplete = inventory(a.root, 'pusht')
        assert len(old) == 6
        for (arm, rank), item in old.items():
            copy_tree(Path(item['root']), panel/'conditions'/arm/f'shard-{rank}')
        original = a.root/'restored/50245262/batch-000/jepa-runtime/pusht-coupling-behavior-20260908-v2/engineering'
        # Historical directories used GPU-<UUID>, but the immutable analyzer
        # resolves the UUID recorded inside each report. Copy bytes unchanged.
        for p in original.iterdir():
            if p.is_dir() and (p/'report.json').exists():
                proof = read(p/'report.json')
                assert sha(p/'report.json') == read(p/'DONE.json')['report_sha256']
                copy_tree(p, panel/'engineering'/proof['device_uuid'])
        archives = {}
        for arc in a.archives:
            receipt = read(arc.with_suffix('.verified.json'))
            assert sha(arc) == receipt['archive_sha256']
            assert receipt['state']['done'] and not receipt['state']['failed']
            manifest = receipt['manifest']
            extracted = out/'restored-snapshots'/arc.parent.name
            with tarfile.open(arc) as t:
                assert len(t.getnames()) == len(set(t.getnames())) == len(manifest)
                for member in t:
                    p = PurePosixPath(member.name)
                    assert member.isfile() and not p.is_absolute() and '..' not in p.parts
                    raw = t.extractfile(member).read()
                    assert len(raw) == manifest[member.name]['bytes']
                    assert hashlib.sha256(raw).hexdigest() == manifest[member.name]['sha256']
                    prefix = 'expansion-20260911-v1/pusht/'
                    if member.name.startswith(prefix):
                        target = extracted/member.name[len(prefix):]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as f:
                            f.write(raw)
            copy_tree(extracted/'conditions', panel/'conditions')
            for p in (extracted/'engineering').glob('gpu-*'):
                proof = read(p/'report.json')
                assert sha(p/'report.json') == read(p/'DONE.json')['report_sha256']
                copy_tree(p, panel/'engineering'/proof['device_uuid'])
            archives[str(arc)] = receipt['archive_sha256']
        shards = list((panel/'conditions').glob('*/shard-*'))
        assert len(shards) == 64
        assert {(p.parent.name, int(p.name.split('-')[-1])) for p in shards} == {(arm,r) for arm in ARMS for r in range(8)}
        write(out/'ASSEMBLY.json', {'archive_sha256': archives, 'old_streams': 6,
            'candidate_streams': 64, 'candidate_episodes': 768, 'native_episodes': 96,
            'incomplete_archive_copies_retained': incomplete, 'new_gpu_calls': 0})
        base, kwargs, source = command(a.root, 'pusht')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source),
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (out/'analysis.log').open('x') as log:
            subprocess.run(base+['analyze']+kwargs+['--panel', str(panel), '--output', str(out/'analysis')],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        assert sha(out/'analysis/report.json') == read(out/'analysis/DONE.json')['report_sha256']
        write(out/'QUEUE_DONE.json', {'analysis_report_sha256': sha(out/'analysis/report.json'),
            'gpu_calls': 0, 'full_six_task_study_complete': False})
        print(json.dumps(read(out/'analysis/report.json')), flush=True)
    except Exception as e:
        write(out/'QUEUE_FAILED.json', {'error': str(e), 'preserve_all_inputs': True})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/workspace/table-completion-20260911-v1'))
    p.add_argument('--archives', type=Path, nargs=3, required=True)
    p.add_argument('--output', type=Path)
    main(p.parse_args())
