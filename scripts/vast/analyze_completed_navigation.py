"""Assemble preserved navigation records and run the unchanged frozen analysis.

CPU only, no new episodes or models. Reject missing/conflicting evidence, verify
receiving proofs separately, and retain the full two-task, 32-contrast family.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

from table_resume_streams import FREEZES, inventory

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/offline_study/table-completion-20260911-v1'
HISTORICAL = ROOT/'restored/50231985/batch-001/jepa-runtime'
EVIDENCE = HISTORICAL/'navigation-coupling-evidence-20260908-v2/artifacts/offline_study'
SOURCE = HISTORICAL/'navigation-coupling-code-20260908-v2/src'
SOURCE_HASH = 'fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)


def copy_tree(source, target):
    for p in source.rglob('*'):
        if not p.is_file():
            continue
        assert not p.is_symlink()
        q = target/p.relative_to(source)
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            assert sha(p) == sha(q), 'Conflicting immutable evidence'
        else:
            shutil.copyfile(p, q)


def extract_verified(archive, receipt, output):
    digest = receipt.get('archive_sha256', receipt.get('sha256'))
    assert receipt['verified'] and receipt['full_byte_readback']
    assert sha(archive) == digest == receipt['metadata']['sha256Checksum']
    assert archive.stat().st_size == int(receipt['metadata']['size'])
    manifest = receipt['manifest']
    assert sum(m['bytes'] for m in manifest.values()) < 300 << 20
    with tarfile.open(archive) as stream:
        assert len(stream.getnames()) == len(set(stream.getnames())) == len(manifest)
        for member in stream:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe preserved member')
            assert member.name in manifest
            raw = stream.extractfile(member).read()
            assert len(raw) == manifest[member.name]['bytes']
            assert hashlib.sha256(raw).hexdigest() == manifest[member.name]['sha256']
            path = output/member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(raw)
    return digest


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    try:
        # Bind the entire immutable scientific Python package, not just an
        # apparently similar current checkout of the analysis function.
        h = hashlib.sha256()
        for p in sorted((SOURCE/'offline_study').glob('*.py')):
            h.update(p.name.encode()+b'\0'+p.read_bytes())
        assert h.hexdigest() == SOURCE_HASH
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.path.insert(0, str(SOURCE))
        from offline_study.author_evaluate import verify_fit
        from offline_study.navigation_coupling_analysis import load_panel, analyze
        from offline_study.navigation_coupling_behavior import validate_engineering
        from offline_study.behavioral_development import assigned_rows, schedule

        archives = {}
        wall_receipt = ROOT/'wall-closeout/DRIVE_VERIFIED.json'
        wall_archive = ROOT/'wall-closeout/wall-table-resume-20260911-v1.tar.gz'
        archives[str(wall_archive)] = extract_verified(wall_archive, read(wall_receipt), out/'inputs/wall')
        pm_receipt = ROOT/'expansion-v2/50588893/FINAL_DRIVE_VERIFIED.json'
        proof = read(pm_receipt)
        assert all(q['done'] and not q['failed'] for q in proof['state']['additional_queues'].values())
        name = proof['metadata']['name']
        assert name.startswith('jepa-table-50588893-snapshot-')
        pm_archive = pm_receipt.parent/name.removeprefix('jepa-table-50588893-')
        archives[str(pm_archive)] = extract_verified(pm_archive, proof, out/'inputs/pointmaze')
        copy_tree(SOURCE, out/'frozen-source')

        engineering = {}
        for base in (ROOT/'restored', out/'inputs'):
            for p in base.rglob('report.json'):
                r = read(p)
                if r.get('status') == 'complete_coupling_navigation_engineering':
                    engineering.setdefault(sha(p), []).append(p.parent)
        receiving = {}
        contract_bindings = {}
        for task in ('wall', 'pointmaze'):
            freeze = EVIDENCE/f'navigation-coupling-behavior-20260908-v1/{task}/freeze'
            assert sha(freeze/'protocol.json') == FREEZES[task]
            protocol = read(freeze/'protocol.json')
            assert protocol['source_sha256'] == SOURCE_HASH
            fits = EVIDENCE/f'primary-durable-20260907/navigation-fits-20260907-v1/bfloat16/{task}'
            cohort = read(fits/'cohort.json')
            receipt, _ = verify_fit(fits/'vision_action_coupling', sha(fits/'cohort.json'),
                                   protocol['bindings']['checkpoint_sha256'], 'bfloat16')
            assert receipt['development_outcomes_accessed'] is False
            assert cohort['task'] == task
            assert not ({r['lineage_group'] for r in cohort['fit']} &
                        {r['lineage_group'] for r in cohort['evaluation']})
            for name, digest in protocol['bindings']['fit'].items():
                assert sha(fits/'vision_action_coupling'/name) == digest
            assert sha(fits/'cohort.json') == protocol['bindings']['cohort_sha256']
            copy_tree(freeze, out/'panel'/task/'freeze')
            copy_tree(EVIDENCE/f'navigation-coupling-reference-20260908-v1/{task}', out/'reference'/task)
            copy_tree(fits/'vision_action_coupling', out/'fit-evidence'/task/'vision_action_coupling')
            for name in ('cohort.json', 'PARITY.json'):
                shutil.copyfile(fits/name, out/'fit-evidence'/task/name)
            contract_bindings[task] = protocol['bindings']
            old, incomplete = inventory(ROOT, task)
            assert not incomplete
            assert len(old) == (54 if task == 'wall' else 53)
            for item in old.values():
                p = Path(item['root'])
                copy_tree(p, out/'panel'/task/'conditions'/p.parent.name/p.name)
            new_roots = ([out/'inputs/wall/wall-resume-20260911-v1'] if task == 'wall' else
                         [out/'inputs/pointmaze/expansion-20260911-v1'/n for n in
                          ('pointmaze-a', 'pointmaze-b', 'pointmaze-tail-a', 'pointmaze-tail-b')])
            for p in new_roots:
                assert (p/'QUEUE_DONE.json').exists() and not (p/'QUEUE_FAILED.json').exists()
                for shard in (p/'conditions').glob('*/shard-*'):
                    assert (shard/'DONE.json').exists()
                    copy_tree(shard, out/'panel'/task/'conditions'/shard.parent.name/shard.name)

            shards = list((out/'panel'/task/'conditions').glob('*/shard-*'))
            assert len(shards) == 64
            for shard in shards:
                launch = read(shard/'protocol.json')
                assert launch['expected_episodes'] == assigned_rows(schedule(), launch['logical_ranks'])
                proof_hash = launch['engineering_report_sha256']
                key = (task, proof_hash)
                if key not in receiving:
                    assert proof_hash in engineering, 'Missing bound receiving proof'
                    candidates = engineering[proof_hash]
                    # Duplicate preserved copies must agree; prefer a complete
                    # copy without altering or synthesizing an engineering record.
                    failures = []
                    for source in candidates:
                        try:
                            assert validate_engineering(source, protocol, FREEZES[task]) == proof_hash
                            break
                        except (ValueError, OSError) as e:
                            failures.append(str(e))
                    else:
                        raise ValueError('No valid receiving proof: '+repr(failures))
                    target = out/'receiving-proofs'/task/proof_hash
                    copy_tree(source, target)
                    receiving[key] = str(target)

        panels, protocols, bindings = load_panel(out/'panel', out/'reference')
        report = analyze(panels, protocols)
        write(out/'analysis/bindings.json', bindings)
        report['bindings_sha256'] = sha(out/'analysis/bindings.json')
        write(out/'analysis/report.json', report)
        write(out/'analysis/DONE.json', {'report_sha256': sha(out/'analysis/report.json')})
        write(out/'ASSEMBLY.json', {'source_package_sha256': SOURCE_HASH,
            'archive_sha256': archives, 'receipt_sha256': {str(p): sha(p) for p in (wall_receipt, pm_receipt)},
            'frozen_contract_bindings': contract_bindings,
            'receiving_proofs': {t+':'+h: p for (t,h),p in receiving.items()},
            'episodes': 1728, 'tasks': 2, 'arms_per_task': 9, 'episodes_per_arm': 96,
            'original_analysis_unchanged': True, 'new_gpu_calls': 0, 'new_simulator_episodes': 0})
        write(out/'DONE.json', {'report_sha256': sha(out/'analysis/report.json'),
            'assembly_sha256': sha(out/'ASSEMBLY.json'), 'full_six_task_matrix_complete': False})
        print(json.dumps(report, indent=2))
    except Exception as exc:
        write(out/'FAILED.json', {'error': str(exc), 'original_evidence_retained': True})
        raise


def preserve(out):
    """Archive this completed analysis and its table in the existing private Drive."""
    from final_preservation_drive import FOLDER, request, upload_direct, verify
    from backup_results_to_google import verify_archive
    assert sha(out/'analysis/report.json') == read(out/'DONE.json')['report_sha256']
    assert sha(out/'ASSEMBLY.json') == read(out/'DONE.json')['assembly_sha256']
    destination = out.parent/(out.name+'-preservation')
    destination.mkdir(exist_ok=False)
    with request(FOLDER, '?fields=id,name,mimeType,trashed') as response:
        metadata = json.load(response)
    assert metadata['id'] == FOLDER and not metadata.get('trashed')
    assert metadata['mimeType'] == 'application/vnd.google-apps.folder'
    files = [p for p in out.rglob('*') if p.is_file()]
    files += [PROJECT/n for n in (
        'reports/SIX_TASK_RESULTS_20260911.md',
        'reports/SIX_TASK_TABLE_PROGRESS_20260911.md',
        'scripts/vast/analyze_completed_navigation.py',
        'scripts/vast/table_resume_streams.py',
        'tests/test_navigation_table_analysis.py',
        'artifacts/offline_study/core-completion-preservation-20260908-v1/ANALYSIS_REPORT.json',
        'artifacts/offline_study/table-completion-20260911-v1/pusht-final-analysis-v2/report.json',
        'artifacts/offline_study/table-completion-20260911-v1/pusht-final-analysis-v2/DONE.json',
        'artifacts/offline_study/table-completion-20260911-v1/restored/50259194/batch-000/jepa-runtime/droid-coupling-behavior-20260908-v3/analysis/report.json',
        'artifacts/offline_study/table-completion-20260911-v1/restored/50259194/batch-000/jepa-runtime/droid-coupling-behavior-20260908-v3/analysis/DONE.json')]
    assert all(p.is_file() and not p.is_symlink() and p.name not in ('.env','rclone.conf') for p in files)
    manifest = {str(p.relative_to(PROJECT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}
    assert len(manifest) == len(files)
    write(destination/'FILES.json', manifest)
    archive = destination/'jepa-navigation-analysis-and-six-task-table-20260911-v1.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for p in sorted(files):
            stream.add(p, arcname=str(p.relative_to(PROJECT)), recursive=False)
    digest, size = sha(archive), archive.stat().st_size
    verify_archive(['cat', str(archive)], digest, size, manifest)
    subprocess.run(['/usr/local/bin/rclone','about','gdrive:','--json'],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
    obj = upload_direct(archive, archive.name, destination/'UPLOADED.json')
    checked = verify(obj['id'], archive.name, size, digest)
    write(destination/'DRIVE_VERIFIED.json', {**checked, 'drive_file_id': obj['id'],
        'archive_sha256': digest, 'archive_bytes': size,
        'members_verified': len(manifest), 'manifest_sha256': sha(destination/'FILES.json'),
        'all_originals_retained': True, 'full_six_task_matrix_complete': False})
    print(json.dumps({'drive_verified': True, 'id': obj['id'], 'bytes': size,
                      'members_verified': len(manifest)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'navigation-final-analysis-v1')
    parser.add_argument('--preserve', action='store_true')
    args = parser.parse_args()
    (preserve if args.preserve else main)(args.output)
