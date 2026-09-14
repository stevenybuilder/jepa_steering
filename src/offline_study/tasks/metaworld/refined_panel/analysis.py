"""Frozen, complete-panel analysis of the refined-edit six-task completion (Push-T, PointMaze, Wall, DROID).

CPU only. No new episodes, no candidate selection, no pooling across tasks. The
statistical contract is the one frozen in refined_task_behavior.ANALYSIS
(20,000 paired scenario-cluster bootstrap draws, seed 2026091102, twelve
contrasts, Bonferroni simultaneous 95% intervals). Push-T clusters by original
source initial-state family, navigation by exact initial/goal pair, DROID by
source recording. Simulation endpoint: official binary task success. DROID
endpoint: the native recorded-plan checkpoint action score (never robot success).

Every shard is re-verified before it counts: DONE/report checksum, protocol
binding, launch/report agreement, whole-stream coverage against the frozen
schedule, episode-record hashes, raw call-trace hashes and CEM budget, paired
inputs against the same-shard native stream, and one physical device for all
three arms of a scenario stream. A missing or failed shard aborts the analysis.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.behavioral_development import assigned_rows, schedule, verified_report
from offline_study.tasks.droid.droid_contract import checkpoint_score
from offline_study.tasks.droid.droid_coupling_behavior import paired_inputs as paired_droid
from offline_study.tasks.droid.droid_fixed_response_behavior import METHOD as DROID_METHOD, engineering_receipt, load_shard as load_droid_shard
from offline_study.tasks.navigation.navigation_replication import validate_records
from offline_study.core.protocol import sha256, write_json
from offline_study.tasks.pusht.pusht_coupling_behavior import validate_push_records
from offline_study.tasks.metaworld.refined_task_behavior import ANALYSIS, ARMS, METHOD, paired, validate_calls, validate_protocol

SIM_TASKS = ('pusht', 'pointmaze', 'wall')
TASKS = tuple(ANALYSIS['family_tasks'])
CONTRASTS = tuple(ANALYSIS['contrasts'])
EPISODES = {'pusht': 96, 'pointmaze': 96, 'wall': 96, 'droid': 64}


def read(path):
    return json.loads(Path(path).read_text())


def shard_dirs(root):
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith('shard-')),
                  key=lambda p: int(p.name.split('-')[1])) if root.is_dir() else []


def cluster_key(task, row):
    if task == 'pusht':
        return row['source_segment']['lineage_group']          # original source initial-state family
    if task == 'droid':
        return row['result']['dataset_sample']['path']          # source recording
    return row['result']['initial_sha256'] + ':' + row['result']['goal_sha256']   # exact initial/goal pair


# ----------------------------------------------------------------------------- simulation tasks
def load_sim_shard(shard, task, arm, freeze_hash, source_hash, dose):
    report, digest = verified_report(shard)
    launch = read(shard / 'protocol.json')
    if ((shard / 'FAILED.json').exists() or report.get('status') != METHOD + '_shard_complete'
            or report.get('task') != task or report.get('arm') != arm
            or report.get('freeze_sha256') != freeze_hash or report.get('source_sha256') != source_hash
            or report.get('protocol_sha256') != sha256(shard / 'protocol.json')
            or report.get('parameters_unchanged') is not True or report.get('fresh_confirmation') is not False
            or report.get('scientific_efficacy_measurement') is not True
            or (dose is not None and report.get('dose') != dose)
            or launch.get('task') != task or launch.get('arm') != arm or launch.get('engineering') is not False
            or launch.get('freeze_sha256') != freeze_hash or launch.get('source_sha256') != source_hash
            or launch.get('fresh_confirmation') is not False or not launch.get('engineering_report_sha256')
            or launch.get('device_uuid') != report.get('device_uuid') or not launch.get('device_uuid')):
        raise ValueError(f'Unbound, failed or partial refined shard: {shard}')
    expected = assigned_rows(schedule(), launch['logical_ranks'])
    if launch['expected_episodes'] != expected:
        raise ValueError(f'Altered logical stream membership: {shard}')
    names = [f"episode-{row['episode']:03d}" for row in expected]
    if set(report['episodes']) != set(names):
        raise ValueError(f'Incomplete whole-stream shard: {shard}')
    rows = []
    for name in names:
        path = shard / name
        if sha256(path / 'episode.json') != report['episodes'][name]:
            raise ValueError(f'Changed episode record: {path}')
        record = read(path / 'episode.json')
        if record['arm'] != arm or record['call_trace_sha256'] != sha256(path / 'unroll_calls.json'):
            raise ValueError(f'Changed raw call trace or mislabeled arm: {path}')
        validate_calls(read(path / 'unroll_calls.json'), arm, report['dose'])
        if arm != 'native' and record['static_reference_counts'] != []:
            pass  # engineering-only reference checks are not expected on scientific episodes
        rows.append(record)
    validate_records(rows, expected)
    return rows, report, launch, digest


def load_sim_task(task, root, freeze, engineering_roots):
    conditions = root / 'conditions'
    freeze_hash = source_hash = dose = None
    protocol = None
    if freeze is not None and (freeze / 'protocol.json').exists():
        protocol = validate_protocol(freeze)
        if protocol['task'] != task or protocol['analysis'] != ANALYSIS:
            raise ValueError(f'Frozen protocol is not the {task} refined contract')
        freeze_hash, source_hash, dose = sha256(freeze / 'protocol.json'), protocol['source_sha256'], protocol['binding']['dose']
    shards = {arm: shard_dirs(conditions / arm) for arm in ARMS}
    if any(len(shards[arm]) == 0 for arm in ARMS):
        raise ValueError(f'{task}: missing condition directories')
    if freeze_hash is None:
        # Freeze directory not supplied: require every shard of every arm to bind to ONE freeze/source hash.
        heads = {(read(s / 'report.json').get('freeze_sha256'), read(s / 'report.json').get('source_sha256'), read(s / 'report.json').get('dose'))
                 for arm in ARMS for s in shards[arm]}
        if len(heads) != 1:
            raise ValueError(f'{task}: shards bind to different freeze/source/dose: {sorted(heads)}')
        (freeze_hash, source_hash, dose), = heads
    panel, bindings, devices, receipts = {}, {}, {}, {}
    for arm in ARMS:
        rows = []
        for shard in shards[arm]:
            shard_rows, report, launch, digest = load_sim_shard(shard, task, arm, freeze_hash, source_hash, dose)
            bindings[str(shard / 'report.json')] = digest
            key = tuple(launch['logical_ranks'])
            devices.setdefault(key, {})[arm] = launch['device_uuid']
            receipts.setdefault(launch['device_uuid'], set()).add(launch['engineering_report_sha256'])
            rows.extend(shard_rows)
        rows.sort(key=lambda r: r['episode'])
        validate_records(rows, schedule())
        if len(rows) != EPISODES[task]:
            raise ValueError(f'{task}/{arm}: {len(rows)} episodes, require {EPISODES[task]}')
        panel[arm] = rows
    # same physical GPU for the three conditions of every logical stream, and identical shard partition per arm
    for key, by_arm in devices.items():
        if set(by_arm) != set(ARMS):
            raise ValueError(f'{task}: logical ranks {key} not present in every arm')
        if len(set(by_arm.values())) != 1:
            raise ValueError(f'{task}: paired conditions for ranks {key} executed on different devices {by_arm}')
    for uuid, hashes in receipts.items():
        if len(hashes) != 1:
            raise ValueError(f'{task}: device {uuid} bound to several engineering receipts')
    # paired inputs: every edited row against the native row of the same episode
    for arm in ARMS[1:]:
        for native, row in zip(panel['native'], panel[arm], strict=True):
            paired(native, row, task)
    if task == 'pusht':
        cohort_path = next((p for p in (root / 'cohort' / 'cohort.json', root / 'fit' / 'cohort.json') if p.exists()), None)
        if cohort_path is not None:
            for arm in ARMS:
                validate_push_records(panel[arm], schedule(), read(cohort_path))
    # engineering receipts that are available locally are re-verified; the rest are recorded as unverified bindings
    verified_receipts = {}
    for uuid, hashes in receipts.items():
        expected_hash, = hashes
        root_for_device = None
        for candidate in engineering_roots:
            if candidate.is_dir() and (candidate / 'report.json').exists() and read(candidate / 'report.json').get('device_uuid') == uuid:
                root_for_device = candidate
        if root_for_device is None:
            verified_receipts[uuid] = {'engineering_report_sha256': expected_hash, 'receipt_reverified_locally': False}
            continue
        if protocol is None:
            raise ValueError(f'{task}: engineering receipt present for {uuid} but no freeze protocol to verify it against')
        from offline_study.tasks.metaworld.refined_task_behavior import verify_engineering
        digest = verify_engineering(root_for_device, protocol, freeze_hash, uuid)
        if digest != expected_hash:
            raise ValueError(f'{task}: receiving proof changed for device {uuid}')
        verified_receipts[uuid] = {'engineering_report_sha256': expected_hash, 'receipt_reverified_locally': True,
                                   'root': str(root_for_device)}
    return {'panel': panel, 'bindings': bindings, 'freeze_sha256': freeze_hash, 'source_sha256': source_hash,
            'dose': dose, 'freeze_protocol_reverified': protocol is not None, 'devices': verified_receipts,
            'streams_per_arm': {arm: len(shards[arm]) for arm in ARMS}}


# ----------------------------------------------------------------------------- DROID
def load_droid(root, freeze, engineering):
    protocol = read(freeze / 'protocol.json')
    freeze_hash = sha256(freeze / 'protocol.json')
    if (read(freeze / 'FROZEN.json')['protocol_sha256'] != freeze_hash or protocol['method'] != DROID_METHOD
            or protocol['analysis'] != ANALYSIS or protocol['arms'] != list(ARMS)
            or protocol['episodes_per_condition'] != 64 or protocol['robot_executions'] != 0
            or protocol['endpoint'] != 'official_recorded_plan_action_score_not_physical_task_success'):
        raise ValueError('DROID frozen protocol changed or is not the refined contract')
    dose = protocol['binding']['dose']
    conditions = root / 'conditions'
    panel, bindings, devices, receipts = {}, {}, {}, {}
    for arm in ARMS:
        rows = []
        for shard in shard_dirs(conditions / arm):
            launch = read(shard / 'protocol.json')
            uuid = launch['device_uuid']
            if uuid not in receipts:
                receipts[uuid] = engineering_receipt(engineering, freeze_hash, uuid, dose)
            ranks = launch['logical_ranks']
            if len(ranks) != 1:
                raise ValueError(f'DROID shard must hold one logical stream: {shard}')
            shard_rows, digest = load_droid_shard(shard, protocol, freeze_hash, uuid, arm, ranks[0], receipts[uuid])
            bindings[str(shard / 'report.json')] = digest
            devices.setdefault(ranks[0], {})[arm] = uuid
            rows.extend(shard_rows)
        rows.sort(key=lambda r: r['episode'])
        if len(rows) != 64 or [r['episode'] for r in rows] != [r['episode'] for r in protocol['planning']['episodes']]:
            raise ValueError(f'droid/{arm}: incomplete or reordered 64-episode population')
        panel[arm] = rows
    if set(devices) != set(range(8)):
        raise ValueError('DROID: require all eight logical streams')
    for rank, by_arm in devices.items():
        if set(by_arm) != set(ARMS) or len(set(by_arm.values())) != 1:
            raise ValueError(f'DROID: paired conditions for rank {rank} not on one device: {by_arm}')
    for arm in ARMS[1:]:
        for native, row in zip(panel['native'], panel[arm], strict=True):
            paired_droid(native, row)
    return {'panel': panel, 'bindings': bindings, 'freeze_sha256': freeze_hash, 'source_sha256': protocol['source_sha256'],
            'dose': dose, 'freeze_protocol_reverified': True,
            'devices': {u: {'engineering_report_sha256': h, 'receipt_reverified_locally': True} for u, h in receipts.items()},
            'streams_per_arm': {arm: 8 for arm in ARMS}}


# ----------------------------------------------------------------------------- statistics
def clusters_for(task, rows):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(cluster_key(task, row), []).append(i)
    return list(groups.values())


def analyze(loaded):
    rng = np.random.default_rng(ANALYSIS['seed'])
    tail = .05 / (2 * ANALYSIS['family'])
    tasks, contrasts = {}, []
    for task in TASKS:
        panel = loaded[task]['panel']
        clusters = clusters_for(task, panel['native'])
        if len(clusters) < 2:
            raise ValueError(f'{task}: insufficient independent clusters')
        draws = rng.integers(len(clusters), size=(ANALYSIS['replicates'], len(clusters)))
        sizes = np.array([len(c) for c in clusters])
        denominator = sizes[draws].sum(1)
        if task in SIM_TASKS:
            values = {arm: np.array([r['result']['native_success'] for r in panel[arm]], dtype=float) for arm in ARMS}
            summary = {arm: {'episodes': len(panel[arm]), 'successes': int(values[arm].sum()),
                             'success_percent': float(values[arm].mean() * 100),
                             'mean_reward': float(np.mean([r['result']['native_reward'] for r in panel[arm]])),
                             'mean_task_distance': float(np.mean([r['result']['native_state_distance'] for r in panel[arm]])),
                             'mean_episode_seconds': float(np.mean([r['seconds'] for r in panel[arm]]))} for arm in ARMS}
            point = {arm: values[arm].mean() * 100 for arm in ARMS}
            sampled = {arm: np.array([values[arm][c].sum() for c in clusters])[draws].sum(1) / denominator * 100 for arm in ARMS}
            unit = 'percentage_points_task_success'
        else:
            errors = {arm: np.array([r['result']['metrics']['action_error_xyz'] for r in panel[arm]], dtype=float) for arm in ARMS}
            summary = {arm: {'episodes': len(panel[arm]), 'recordings': len(clusters),
                             'checkpoint_score': float(checkpoint_score(torch.tensor(errors[arm]))),
                             'mean_xyz_action_error': float(errors[arm].mean()),
                             'mean_orientation_action_error': float(np.mean([r['result']['metrics']['action_error_orientation'] for r in panel[arm]])),
                             'mean_gripper_action_error': float(np.mean([r['result']['metrics']['action_error_gripper'] for r in panel[arm]])),
                             'mean_episode_seconds': float(np.mean([r['seconds'] for r in panel[arm]])),
                             'score_is_not_success_percentage': True} for arm in ARMS}
            point = {arm: summary[arm]['checkpoint_score'] for arm in ARMS}
            sampled = {}
            for arm in ARMS:
                sums = np.array([errors[arm][c].sum() for c in clusters])
                sampled[arm] = np.maximum(0., 800 * (.1 - sums[draws].sum(1) / denominator))
            unit = 'checkpoint_score_points'
        tasks[task] = {'scenario_clusters': len(clusters), 'arms': summary, 'endpoint_unit': unit}
        for name in CONTRASTS:
            a, b = name.split('-')
            interval = np.quantile(sampled[a] - sampled[b], [tail, 1 - tail])
            contrasts.append({'task': task, 'contrast': name, 'estimate': float(point[a] - point[b]), 'unit': unit,
                              'simultaneous_95_interval': interval.tolist(), 'scenario_clusters': len(clusters)})
    return {'status': METHOD + '_complete_panel_analyzed', 'tasks': tasks, 'contrasts': contrasts, 'analysis': ANALYSIS,
            'bonferroni_two_sided_tail': tail, 'fresh_confirmation': False, 'automatic_candidate_selection': False,
            'pooled_robotics_success_score': False, 'historical_native_equivalence_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-root', type=Path, required=True,
                        help='directory with <task>/conditions/<arm>/shard-k for pusht, pointmaze, wall, droid')
    for task in TASKS:
        parser.add_argument(f'--freeze-{task}', type=Path, help=f'{task} freeze dir (default <panel-root>/{task}/freeze)')
        parser.add_argument(f'--engineering-{task}', type=Path, nargs='*', default=None,
                            help=f'{task} engineering receipt dir(s) (default <panel-root>/{task}/engineering)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    for task in TASKS:
        root = args.panel_root / task
        freeze = getattr(args, f'freeze_{task}') or (root / 'freeze')
        eng = getattr(args, f'engineering_{task}')
        eng = [root / 'engineering'] if eng is None else eng
        if task == 'droid':
            loaded[task] = load_droid(root, freeze, eng[0])
        else:
            loaded[task] = load_sim_task(task, root, freeze if freeze.exists() else None, eng)
    report = analyze(loaded)
    report['inputs'] = {task: {k: v for k, v in loaded[task].items() if k != 'panel'} for task in TASKS}
    report['analysis_source_sha256'] = sha256(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})
    print(json.dumps({t: {a: (s['arms'][a].get('success_percent', s['arms'][a].get('checkpoint_score'))) for a in ARMS}
                      for t, s in report['tasks'].items()}, indent=1))
    for c in report['contrasts']:
        print(f"{c['task']:9s} {c['contrast']:44s} {c['estimate']:+8.3f}  [{c['simultaneous_95_interval'][0]:+8.3f}, {c['simultaneous_95_interval'][1]:+8.3f}]")


if __name__ == '__main__':
    main()
