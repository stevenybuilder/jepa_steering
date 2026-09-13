"""CPU/input-only preparation. Does not load a model or launch paid compute.

Candidate populations are explicitly different from the historical dataset-goal
Push-T and custom-Franka evaluations. No scientific readiness is implied.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from offline_study.protocol import sha256, write_json

OUT = ROOT / 'artifacts/offline_study/protected-preparation-20260912-v1'
ART = ROOT / 'artifacts/offline_study'
CAT = ART / 'primary-durable-20260907/droid-public-catalogue-20260907-v1/catalogue.json'
IDENTITY = ART / 'droid-fit-completed-inputs-20260908-v1/droid-fit-audit-20260908-v2/identity.json'
PUSH_MANIFEST = ART / '2026-09-07-pusht-family-correction/pusht-inventory-v3-development-exposed/trajectories.jsonl'
SEED = 2026091203


def fresh_dir(path):
    path.mkdir(parents=True, exist_ok=False)


def freeze():
    from offline_study.droid_fit_inputs import CATALOGUE
    if sha256(CAT) != CATALOGUE:
        raise ValueError('Historical catalogue checksum differs')
    fresh_dir(OUT)
    source = ROOT / 'vendor/jepa-wms/evals/simu_env_planning/envs/pusht_gym_wrap.py'
    write_json(OUT / 'protocol.json', {
        'role': 'candidate_input_preparation_no_model_outcomes',
        'script_sha256': sha256(Path(__file__)), 'seed': SEED,
        'source_inputs': {str(p.relative_to(ROOT)): sha256(p) for p in (CAT, IDENTITY, PUSH_MANIFEST, source)},
        'pusht': {'protected_pairs': 96, 'engineering_pairs': 8,
                  'sampler': 'exact upstream sample_random_init_goal_states AST',
                  'population_change': 'random-state candidates, not historical dataset-segment goals'},
        'droid': {'required_candidates': 64, 'metadata_priority_bound': 80,
                  'selection': 'ascending SHA256(seed:object_name) after historical exclusions',
                  'exclude_original_fit_priority_count': 160,
                  'metadata_only_exclusions': ['missing_native_left_mp4'],
                  'raw_eligibility_exclusions': ['insufficient_native_five_frame_window'],
                  'other_errors': 'retain failure and stop; never silently substitute',
                  'maximum_download_bytes': 2 * 1024**3, 'minimum_free_bytes': 8 * 1024**3,
                  'population_change': 'public raw-DROID, not custom Franka evaluation'},
        'model_calls': 0, 'gpu_calls': 0, 'paid_instances_created': 0,
        'scientific_launch_ready': False})
    write_json(OUT / 'FROZEN.json', {'protocol_sha256': sha256(OUT / 'protocol.json')})


def check():
    p = json.loads((OUT / 'protocol.json').read_text())
    if sha256(OUT / 'protocol.json') != json.loads((OUT / 'FROZEN.json').read_text())['protocol_sha256']:
        raise ValueError('Input protocol changed')
    if p['script_sha256'] != sha256(Path(__file__)):
        raise ValueError('Preparation source changed; use a separate version')
    for name, digest in p['source_inputs'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Bound source changed: ' + name)
    return p


def array_hash(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def family_hashes(state):
    import numpy as np
    return {'pusht:initial-state:' + hashlib.sha256(np.asarray(state[:n], dtype=d).tobytes()).hexdigest()
            for n in (5, 7) for d in ('float32', 'float64')}


def pusht():
    import numpy as np
    p = check(); target = OUT / 'pusht'; fresh_dir(target)
    source = ROOT / 'vendor/jepa-wms/evals/simu_env_planning/envs/pusht_gym_wrap.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PushTWrapper')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_random_init_goal_states')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    sample = namespace[method.name]; receiver = SimpleNamespace(env=SimpleNamespace(with_velocity=True))
    old = set()
    for line in PUSH_MANIFEST.read_text().splitlines():
        row = json.loads(line); old.add(row['lineage_group'])
    # The original corrected train inventory may omit the separate val pool.
    val = ART / 'primary-durable-20260907/pusht-author-replication-20260907/cohorts/pusht/cohort.json'
    cohort = json.loads(val.read_text())
    for row in cohort['fit'] + cohort['evaluation']:
        old.add(row['lineage_group'])
    if len(cohort['evaluation']) != 21:
        raise ValueError('Released validation exclusion incomplete')
    protected, engineering, seen = [], [], set()
    for index in range(104):
        seed = int.from_bytes(hashlib.sha256(f'{SEED}:pusht:{index}'.encode()).digest()[:4], 'big')
        initial, goal = sample(receiver, seed)
        initial2, goal2 = sample(receiver, seed)
        if not np.array_equal(initial, initial2) or not np.array_equal(goal, goal2):
            raise ValueError('Native sampling is not repeatable')
        keys = family_hashes(initial)
        if keys & (old | seen) or family_hashes(goal) & old:
            raise ValueError('Input collides with old/source candidate state; no automatic replacement')
        seen.update(keys)
        row = {'candidate': index, 'seed': seed, 'initial_state': initial.tolist(), 'goal_state': goal.tolist(),
               'initial_sha256': array_hash(initial.astype('float32')), 'goal_sha256': array_hash(goal.astype('float32'))}
        (protected if index < 96 else engineering).append(row)
    write_json(target / 'protected_candidates.json', protected)
    write_json(target / 'engineering_candidates.json', engineering)
    write_json(target / 'report.json', {
        'status': 'native_sampler_candidate_states_prepared_not_dataset_goal_confirmation',
        'protected_candidates': 96, 'engineering_candidates': 8,
        'native_sampler_source_sha256': sha256(source), 'repeat_exact': True,
        'historical_initial_family_hashes': len(old), 'exact_initial_family_matches': 0,
        'validation_cohort_sha256': sha256(val), 'manifest_sha256': sha256(PUSH_MANIFEST),
        'protected_sha256': sha256(target / 'protected_candidates.json'),
        'engineering_sha256': sha256(target / 'engineering_candidates.json'),
        'scope_of_overlap_check': 'retained released source initial-state families; not every historical trajectory frame',
        'physics_and_rendering_parity_passed': False, 'historical_dataset_goal_population_preserved': False,
        'fresh_source_trajectories_collected': 0, 'model_calls': 0,
        'scientific_launch_ready': False,
        'remaining': ['decide fresh source-trajectory goal collection versus disclosed random-state population',
                      'validate simulator and native input adapter on excluded engineering cases']})
    print(json.dumps({'task': 'pusht', 'candidate_pairs': 104, 'old_family_matches': 0, 'scientific_launch_ready': False}), flush=True)


def droid_exclusions():
    from offline_study.droid_fit_availability import order
    catalogue = json.loads(CAT.read_text())
    excluded = {str(Path(r['name']).parent) for r in order(catalogue)[:160]}
    pattern = re.compile(r'robotics/droid_raw/1\.0\.1/[^"\n\\]+')
    evidence, skipped, scanned = {}, [], 0
    for path in ART.rglob('*.json'):
        if OUT in path.parents or 'droid-public-catalogue' in str(path) or path.name.startswith('._'):
            continue
        if path.stat().st_size > 8 * 1024**2:
            skipped.append(str(path.relative_to(ROOT))); continue
        content = path.read_text(errors='replace'); scanned += 1
        dirs = {'/'.join(match.group().split('/')[:7]) for match in pattern.finditer(content)
                if len(match.group().split('/')) >= 7}
        if dirs:
            excluded.update(dirs); evidence[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUT / 'droid_exclusions.json', {'excluded_directories': sorted(excluded),
        'evidence_sha256': evidence, 'json_files_scanned': scanned, 'oversized_files_not_scanned': skipped,
        'catalogue_enumeration_is_not_observation_exposure': True,
        'scope': 'retained local JSON source references; remote archives and unrecorded work not exhaustively audited',
        'base_model_training_exposure': 'unknown'})
    return catalogue, excluded


def droid_metadata():
    from offline_study.droid_fit_availability import inspect_record
    check(); target = OUT / 'droid'; fresh_dir(target)
    catalogue, excluded = droid_exclusions()
    ordered = sorted((r for r in catalogue if str(Path(r['name']).parent) not in excluded),
                     key=lambda r: hashlib.sha256(f'{SEED}:{r["name"]}'.encode()).digest())[:80]
    write_json(target / 'priority.json', ordered)
    metadata = target / 'metadata'; metadata.mkdir()
    def one(pair):
        i, row = pair
        result = inspect_record(row, metadata / f'priority-{i:03d}')
        return {'priority': i, **result}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = []
        for result in pool.map(one, enumerate(ordered)):
            records.append(result)
            if len(records) % 10 == 0:
                print(json.dumps({'droid_metadata_complete': len(records), 'required': 80}), flush=True)
    selected = [r for r in records if r['native_left_video_object'] is not None]
    write_json(target / 'metadata_candidates.json', records)
    write_json(target / 'metadata_report.json', {'status': 'candidate_metadata_resolved',
        'examined': len(records), 'camera_present': len(selected),
        'excluded_missing_camera': len(records) - len(selected),
        'exclusions_sha256': sha256(OUT / 'droid_exclusions.json'),
        'priority_sha256': sha256(target / 'priority.json'),
        'candidates_sha256': sha256(target / 'metadata_candidates.json'),
        'population': 'public_raw_DROID_not_custom_Franka', 'scientific_launch_ready': False, 'model_calls': 0})
    print(json.dumps({'droid_camera_candidates': len(selected), 'historical_directories_excluded': len(excluded)}), flush=True)


def droid_download():
    import numpy as np
    import h5py
    from offline_study.droid_fit_download import download_object
    p = check(); target = OUT / 'droid'; report = json.loads((target / 'metadata_report.json').read_text())
    if sha256(target / 'metadata_candidates.json') != report['candidates_sha256']:
        raise ValueError('Metadata candidates changed')
    candidates = json.loads((target / 'metadata_candidates.json').read_text())
    raw = target / 'raw'; fresh_dir(raw)
    old = json.loads(IDENTITY.read_text())
    old_records = old['fit'] + old['evaluation_input_fingerprints']
    existing = {key: {r[key] for r in old_records} for key in ('initial_float32_sha256', 'state_float32_sha256')}
    seen = {k: set() for k in existing}; selected = []; excluded = []; files = {}; reserved = 0
    write_json(target / 'download_protocol.json', {'metadata_report_sha256': sha256(target / 'metadata_report.json'),
        'candidate_order_fixed': True, 'required_eligible': 64, 'maximum_bytes': p['droid']['maximum_download_bytes'],
        'camera_and_length_only_exclusions': True, 'native_evaluation_frames_per_clip': 5,
        'endpoint_selection_and_scoring_not_yet_frozen': True, 'model_calls': 0})
    for row in candidates:
        if row['native_left_video_object'] is None:
            excluded.append({'priority': row['priority'], 'reason': 'missing_native_left_mp4'}); continue
        objects = [row['source_trajectory'], row['native_left_video_object'], *row['all_metadata_alias_objects']]
        size = sum(int(o['size']) for o in objects)
        if reserved + size > p['droid']['maximum_download_bytes'] or shutil.disk_usage(raw).free < size + p['droid']['minimum_free_bytes']:
            raise ValueError('Bounded input download capacity exhausted; retain partial and stop')
        reserved += size
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            downloads = list(pool.map(lambda o: download_object(o, raw), objects))
        for item in downloads:
            files[item['relative_path']] = {'sha256': item['sha256'], 'bytes': item['bytes']}
        camera = raw / row['native_left_video_object']['name']
        proc = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=nb_frames,avg_frame_rate,width,height','-of','json',str(camera)], check=True, capture_output=True, text=True)
        stream = json.loads(proc.stdout)['streams'][0]
        frames = int(stream['nb_frames']); num, den = map(int, stream['avg_frame_rate'].split('/')); fps = num / den
        if frames < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Malformed video metadata; cannot replace candidate')
        if frames <= 5 * math.ceil(fps / 4):
            excluded.append({'priority': row['priority'], 'reason': 'insufficient_native_five_frame_window', 'frames': frames, 'fps': fps}); continue
        with h5py.File(raw / row['source_trajectory']['name'], 'r') as f:
            poses = np.asarray(f['observation/robot_state/cartesian_position'])
            gripper = np.asarray(f['observation/robot_state/gripper_position'])
            if poses.ndim != 2 or poses.shape[1] != 6 or gripper.shape != (len(poses),):
                raise ValueError('Malformed state schema; cannot replace candidate')
            states = np.c_[poses, gripper].astype('float32')
            if not np.isfinite(states).all() or len(states) < frames:
                raise ValueError('State/video alignment or finite-state failure')
        fingerprints = {'initial_float32_sha256': array_hash(states[0]), 'state_float32_sha256': array_hash(states)}
        for key, value in fingerprints.items():
            if value in existing[key] or value in seen[key]:
                raise ValueError('Historical or candidate state overlap; cannot silently replace')
            seen[key].add(value)
        selected.append({'priority': row['priority'], 'directory': str(Path(row['source_trajectory']['name']).parent),
            'video_frames': frames, 'state_frames': len(states), 'fps': fps, 'width': stream['width'], 'height': stream['height'],
            'left_video': row['native_left_video_object']['name'], **fingerprints})
        write_json(target / 'download_progress.json', {'eligible_candidates': len(selected), 'bytes': reserved, 'model_calls': 0})
        print(json.dumps({'eligible_droid_candidates': len(selected), 'bytes': reserved}), flush=True)
        if len(selected) == 64: break
    write_json(target / 'FILES.json', files)
    write_json(target / 'eligible_candidates.json', {'selected': selected, 'excluded': excluded})
    write_json(target / 'report.json', {'status': 'candidate_raw_inputs_prepared' if len(selected) == 64 else 'incomplete',
        'recordings': len(selected), 'objects': len(files), 'bytes': reserved,
        'files_sha256': sha256(target / 'FILES.json'), 'candidates_sha256': sha256(target / 'eligible_candidates.json'),
        'metadata_report_sha256': sha256(target / 'metadata_report.json'),
        'historical_state_fingerprints_compared': len(old_records), 'exact_state_overlaps': 0,
        'native_loader_parity_passed': False, 'frame_decode_all_verified': False,
        'historical_custom_Franka_population_preserved': False,
        'exposure_audit_scope': 'retained local source identities and 128 fit plus 15 evaluation state fingerprints',
        'base_model_training_exposure': 'unknown', 'scientific_launch_ready': False, 'model_calls': 0,
        'remaining': ['population decision', 'full native five-frame camera/action conversion parity',
                      'endpoint freeze', 'receiving-device engineering']})
    if len(selected) != 64: raise ValueError('Priority bound exhausted before 64 eligible recordings')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'pusht', 'droid-metadata', 'droid-download'))
    args = parser.parse_args()
    try:
        {'freeze': freeze, 'pusht': pusht, 'droid-metadata': droid_metadata, 'droid-download': droid_download}[args.mode]()
    except Exception as exc:
        if OUT.exists():
            write_json(OUT / ('FAILED-' + args.mode + '.json'), {'error_type': type(exc).__name__,
                'message': str(exc), 'partial_inputs_preserved': True, 'scientific_launch_ready': False})
        raise
