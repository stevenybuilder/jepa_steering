import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('split_mw', SCRIPTS / 'split_fresh_mw_third_host.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_three_partitions_exactly_cover_original_96_without_overlap():
    sets = [{(task, e) for task in ('reach', 'reach-wall') for w in range(start, start + 4)
             for e in range(w, 96, 24)} for start in (3, 11, 19)]
    assert all(len(s) == 32 for s in sets)
    assert len(set.union(*sets)) == 96
    assert set.union(*sets) == {(task, e) for task in ('reach', 'reach-wall')
                                for e in range(96) if e % 8 in (3, 4, 5, 6)}
    assert all(a[:3] == b[:3] and b[3] == 24 for a, b in zip(m.OLD, m.NEW))


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    protocol = tmp_path / 'fresh-freeze-v2/protocol.json'
    save(protocol, {'tasks': {t: {'records': [{'episode': e, 'role': 'scientific_candidates'}
                      for e in range(96)]} for t in ('reach', 'reach-wall')}})
    monkeypatch.setattr(m, 'FREEZE', m.sha(protocol))
    logs = tmp_path / 'worker-logs-v2'
    save(logs / 'ASSIGNMENT.json', {'instance': m.INSTANCE, 'freeze_sha256': m.FREEZE, 'mapping': m.OLD})
    workers = []
    for gpu, task, worker, count in m.OLD:
        path = logs / f'{task}-worker{worker}-LAUNCHED.json'
        save(path, {'gpu': gpu, 'task': task, 'worker_id': worker, 'workers': count,
                    'pid': 100 + gpu, 'command': m.command(tmp_path, task, worker, count)})
        workers.append({'launch_receipt': str(path), 'launch_sha256': m.sha(path)})
    budget = {'as_of_utc': '2026-09-13T01:00:00Z', 'accrued_usd': 50,
              'fleet_cap_usd_hour': 35, 'campaign_cap_usd': 220, 'closeout_reserve_usd': 5}
    until = m.guard.deadline(m.datetime.fromisoformat('2026-09-13T01:00:00+00:00').timestamp(), 50, 35)
    path = logs / 'APPROVED35.json'
    save(path, {'instance': m.INSTANCE, 'assignment_sha256': m.sha(logs / 'ASSIGNMENT.json'),
                'budget': budget, 'deadline_unix': until, 'workers': workers})
    monkeypatch.setattr(m, 'verify_bundle', lambda *a: 'verified-engineering-sha')
    return tmp_path, {g: f'uuid-{g}' for g in range(8)}, path


def inspect(receiver, **kwargs):
    root, uuids, budget = receiver
    return m.plan(root, uuids, budget.name, m.sha(budget), inspect=kwargs.get('identity', lambda pid: None))


def test_plan_is_readonly_and_exact_eight_guard_receipts(receiver):
    root, _, _ = receiver
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    proposal, _ = inspect(receiver)
    assert len(proposal['jobs']) == len(set(proposal['launch_receipts'])) == 8
    assert all(len(j['episodes']) == 4 and j['workers'] == 24 for j in proposal['jobs'])
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_live_old_worker_rejected_without_signals(receiver):
    with pytest.raises(ValueError, match='already be dead'):
        inspect(receiver, identity=lambda pid: {'command': ['old'], 'start_tick': 1})


def test_engineering_done_required(receiver, monkeypatch):
    monkeypatch.setattr(m, 'verify_bundle', lambda *a: (_ for _ in ()).throw(ValueError('Missing DONE')))
    with pytest.raises(ValueError, match='Missing DONE'):
        inspect(receiver)


@pytest.mark.parametrize('episode', [11, 27])
def test_any_later_coarse_output_rejected_including_transferred(receiver, episode):
    root, _, _ = receiver
    (root / f'results-v2/reach/scenario-{episode:03d}').mkdir(parents=True)
    with pytest.raises(ValueError, match='Later coarse-stripe'):
        inspect(receiver)


def test_initial_partial_stays_on_exact_gpu_and_inputs(receiver):
    root, _, _ = receiver
    initial = root / 'results-v2/reach/scenario-003'
    identity = {'task': 'reach', 'scenario': {'episode': 3, 'role': 'scientific_candidates'},
                'freeze_sha256': m.FREEZE, 'device_uuid': 'uuid-0', 'engineering': False}
    save(initial / 'STARTED.json', identity)
    p, _ = inspect(receiver)
    assert p['engineering_and_resume_proofs'][0]['initial_files_sha256']['STARTED.json'] == m.sha(initial / 'STARTED.json')
    identity['device_uuid'] = 'another-gpu'
    save(initial / 'STARTED.json', identity)
    with pytest.raises(ValueError, match='cannot move GPU'):
        inspect(receiver)


def test_budget_hash_and_cap_overrides_rejected(receiver):
    root, uuids, path = receiver
    with pytest.raises(ValueError, match='SHA'):
        m.plan(root, uuids, path.name, '0' * 64)
    content = m.read(path)
    content['budget']['campaign_cap_usd'] = 250
    save(path, content)
    with pytest.raises(ValueError, match='Unapproved budget'):
        inspect(receiver)


def test_approval_exact_and_old_launcher_disabled(receiver):
    root, _, _ = receiver
    p, _ = inspect(receiver)
    path = root / 'approval.json'
    record = {'approved': True, 'proposal': p, 'old_coarse_launchers_disabled': True,
              'global_split_ownership_verified': True}
    save(path, record)
    assert m.approval(path, p) == m.sha(path)
    record['old_coarse_launchers_disabled'] = False
    save(path, record)
    with pytest.raises(ValueError, match='Exact global'):
        m.approval(path, p)


def test_frozen_cli_retains_engineering_and_workers24():
    argv = m.command(Path('/frozen'), 'reach', 3, 24)
    assert argv[3] == 'worker'
    assert argv[argv.index('--workers') + 1] == '24'
    assert '--skip-engineering' not in argv


def test_spawn_identity_failure_cleans_owned_child_and_never_old_workers(receiver, monkeypatch):
    root, _, path = receiver
    prior = m.read(path)
    prior['watchdog_pid'] = 999
    save(path, prior)
    p, prior = inspect(receiver)
    manifest = root / 'approval.json'
    save(manifest, {'approved': True, 'proposal': p, 'old_coarse_launchers_disabled': True,
                    'global_split_ownership_verified': True})
    monkeypatch.setattr(m.guard, 'identity', lambda pid: {'command': ['python', 'fresh_budget_watchdog.py',
                        str(root)], 'start_tick': 1} if pid == 999 else None)
    monkeypatch.setattr(m.time, 'time', lambda: prior['deadline_unix'] - 10000)
    monkeypatch.setattr(m, 'plan', lambda *a: (p, prior))
    events = []
    class Child:
        pid = 2000
        def poll(self): return None
        def terminate(self): events.append('own-terminate')
        def wait(self, timeout): return -15
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: Child())
    import start_fresh_v2_navigation_handoff as nav
    monkeypatch.setattr(nav, 'wait_exact_identity', lambda *a: (_ for _ in ()).throw(ValueError('identity')))
    monkeypatch.setattr(m.guard, 'quiesce', lambda owned: events.extend(owned) or {})
    with pytest.raises(ValueError, match='identity'):
        m.launch(root, p, prior, manifest)
    assert events == ['own-terminate']
    failed = m.read(root / 'worker-logs-v2' / f'HANDOFF_FAILED.{m.GENERATION}.json')
    assert failed['old_workers_signaled'] is False
    assert failed['unregistered_children'][0]['owned_popen_child_reaped'] is True
