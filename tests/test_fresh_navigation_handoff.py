import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('navigation_handoff', SCRIPTS / 'start_fresh_v2_navigation_handoff.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    logs = tmp_path / 'worker-logs-v2'
    logs.mkdir()
    freeze = tmp_path / 'fresh-freeze-v2'
    freeze.mkdir()
    (freeze / 'protocol.json').write_bytes(b'{}')
    digest = hashlib.sha256(b'{}').hexdigest()
    monkeypatch.setattr(m, 'FREEZE', digest)
    mapping = [[i, 'reach', i + 3, 8] for i in range(3)]
    mapping += [[i + 3, 'reach-wall', i + 3, 8] for i in range(3)]
    mapping += [[6, 'pointmaze', 1, 4], [7, 'wall', 1, 4]]
    save(logs / 'ASSIGNMENT.json', {'instance': m.INSTANCE, 'freeze_sha256': digest, 'mapping': mapping})
    for gpu, task, w, n in mapping:
        save(logs / f'{task}-worker{w}-LAUNCHED.json',
             {'gpu': gpu, 'task': task, 'worker_id': w, 'workers': n,
              'pid': 900000 + gpu, 'command': m.command(tmp_path, task, w, n)})
    return tmp_path, {i: f'uuid-{i}' for i in range(8)}


def test_plan_is_read_only_exact_and_disjoint(receiver):
    root, uuids = receiver
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan = m.plan(root, uuids, inspect=lambda _: None)
    assert plan['new_scenarios'] == 96
    assert [j['gpu'] for j in plan['jobs']] == [1, 3, 4, 5]
    assert len({(j['task'], e) for j in plan['jobs'] for e in j['episodes']}) == 96
    assert all(j['episodes'] == list(range(j['worker_id'], 96, 4)) for j in plan['jobs'])
    assert len(plan['launch_receipts']) == 8
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_live_old_process_rejected(receiver):
    with pytest.raises(ValueError, match='still alive'):
        m.plan(*receiver, inspect=lambda _: {'command': ['python'], 'start_tick': 1})


def test_even_empty_incoming_scenario_directory_rejected(receiver):
    root, uuids = receiver
    (root / 'results-v2/pointmaze/scenario-002').mkdir(parents=True)
    with pytest.raises(ValueError, match='local output'):
        m.plan(root, uuids, inspect=lambda _: None)


def test_existing_handoff_and_duplicate_devices_rejected(receiver):
    root, uuids = receiver
    with pytest.raises(ValueError, match='distinct'):
        m.plan(root, dict.fromkeys(range(8), 'same'), inspect=lambda _: None)
    (root / 'worker-logs-v2' / m.HANDOFF).write_text('{}')
    with pytest.raises(ValueError, match='Existing handoff'):
        m.plan(root, uuids, inspect=lambda _: None)


def test_ownership_requires_explicit_slot2_disable(receiver):
    root, uuids = receiver
    p = root / 'approval.json'
    save(p, {'approved': True, 'instance': m.INSTANCE, 'freeze_sha256': m.FREEZE,
             'mapping': [list(row) for row in m.NEW], 'globally_unstarted_and_unclaimed': True})
    with pytest.raises(ValueError, match='global ownership'):
        m.ownership(p, {})
    value = json.loads(p.read_text())
    value['stock_slot2_launcher_disabled'] = True
    save(p, value)
    assert m.ownership(p, {})['sha256'] == m.sha(p)


def test_explicit_active_receipts_ignore_preserved_originals(receiver, monkeypatch):
    root, uuids = receiver
    plan = m.plan(root, uuids, inspect=lambda _: None)
    logs = root / 'worker-logs-v2'
    for job in plan['jobs']:
        save(logs / job['launch_receipt'], {**job, 'pid': 910000 + job['gpu']})
    save(logs / m.ACTIVE, plan)
    monkeypatch.setattr(m.guard, 'identity', lambda _: None)
    workers, digest = m.guard.load_workers(root, m.INSTANCE, m.FREEZE, m.ACTIVE)
    assert len(workers) == 8 and len(list(logs.glob('*-LAUNCHED.json'))) == 12
    assert digest == m.sha(logs / m.ACTIVE)
    assert {w['pid'] for w in workers} == {900000, 900002, 900006, 900007, 910001, 910003, 910004, 910005}


def test_duplicate_receipt_selection_rejected(receiver, monkeypatch):
    root, uuids = receiver
    plan = m.plan(root, uuids, inspect=lambda _: None)
    plan['launch_receipts'][-1] = plan['launch_receipts'][0]
    save(root / 'worker-logs-v2' / m.ACTIVE, plan)
    with pytest.raises(ValueError, match='duplicate receipt'):
        m.guard.load_workers(root, m.INSTANCE, m.FREEZE, m.ACTIVE)


def test_guard_generation_preserves_default_paths():
    assert m.guard.guard_names()[1] == 'BUDGET_WATCHDOG_ARMED.json'
    assert m.guard.guard_names(m.GENERATION)[1] != m.guard.guard_names()[1]
    for name in ('../bad', '', 'a/b'):
        with pytest.raises(ValueError):
            m.guard.guard_names(name)


def test_launch_failure_quiesces_only_new_workers(receiver, monkeypatch):
    root, uuids = receiver
    plan = m.plan(root, uuids, inspect=lambda _: None)
    logs = root / 'worker-logs-v2'
    save(logs / 'BUDGET_WATCHDOG_ARMED.json', {'watchdog_pid': 100,
         'instance': m.INSTANCE, 'assignment_sha256': plan['original_assignment_sha256'],
         'deadline_unix': m.time.time() + 1000})
    current = {100: {'command': ['python', 'fresh_budget_watchdog.py', '--root', str(root)], 'start_tick': 1}}
    monkeypatch.setattr(m.guard, 'identity', current.get)
    class Child:
        pid = 101
    calls = []
    def spawn(command, **kwargs):
        if calls:
            raise OSError('second launch failed')
        calls.append(command)
        current[101] = {'command': command, 'start_tick': 2}
        return Child()
    quiesced = []
    monkeypatch.setattr(m.subprocess, 'Popen', spawn)
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: quiesced.extend(workers) or {'quiesced': True})
    with pytest.raises(OSError, match='second launch'):
        m.launch(root, plan, {'sha256': 'reviewed'})
    assert [w['pid'] for w in quiesced] == [101]
    assert (logs / 'BUDGET_WATCHDOG_ARMED.json').exists()
    assert not (logs / m.ACTIVE).exists()


def test_unarmed_replacement_guard_quiesces_all_four_new_workers(receiver, monkeypatch):
    root, uuids = receiver
    proposal = m.plan(root, uuids, inspect=lambda _: None)
    logs = root / 'worker-logs-v2'
    save(logs / 'BUDGET_WATCHDOG_ARMED.json', {'watchdog_pid': 100,
         'instance': m.INSTANCE, 'assignment_sha256': proposal['original_assignment_sha256'],
         'deadline_unix': m.time.time() + 1000,
         'budget': {'as_of_utc': '2026-09-12T23:00:00Z', 'accrued_usd': 30,
                    'fleet_cap_usd_hour': 20, 'campaign_cap_usd': 220,
                    'closeout_reserve_usd': 5}})
    current = {100: {'command': ['python', 'fresh_budget_watchdog.py', '--root', str(root)], 'start_tick': 1}}
    monkeypatch.setattr(m.guard, 'identity', current.get)
    spawned = []
    class Child:
        def __init__(self, pid): self.pid = pid
        def poll(self): return 1
    def spawn(command, **kwargs):
        pid = 101 + len(spawned)
        spawned.append(command)
        current[pid] = {'command': command, 'start_tick': pid}
        return Child(pid)
    quiesced = []
    monkeypatch.setattr(m.subprocess, 'Popen', spawn)
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: quiesced.extend(workers) or {'quiesced': True})
    with pytest.raises(FileNotFoundError):
        m.launch(root, proposal, {'sha256': 'reviewed'})
    assert [w['pid'] for w in quiesced] == [101, 102, 103, 104]
    assert (logs / m.ACTIVE).exists()
    assert not (logs / ('HANDOFF_MANAGED.' + m.GENERATION + '.json')).exists()


@pytest.mark.parametrize('child_identity', [None, {'command': ['wrong'], 'start_tick': 2}])
def test_identity_failure_reaps_just_spawned_popen_child(receiver, monkeypatch, child_identity):
    root, uuids = receiver
    proposal = m.plan(root, uuids, inspect=lambda _: None)
    logs = root / 'worker-logs-v2'
    save(logs / 'BUDGET_WATCHDOG_ARMED.json', {'watchdog_pid': 100,
         'instance': m.INSTANCE, 'assignment_sha256': proposal['original_assignment_sha256'],
         'deadline_unix': m.time.time() + 1000})
    current = {100: {'command': ['python', 'fresh_budget_watchdog.py', '--root', str(root)], 'start_tick': 1}}
    current[101] = child_identity
    monkeypatch.setattr(m.guard, 'identity', current.get)
    events = []
    class Child:
        pid = 101
        def poll(self): events.append('poll'); return None
        def terminate(self): events.append('terminate')
        def wait(self, timeout): events.append(('wait', timeout)); return -15
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: Child())
    quiesced = []
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: quiesced.extend(workers) or {'quiesced': True})
    with pytest.raises(ValueError, match='exact frozen command'):
        m.launch(root, proposal, {'sha256': 'reviewed'})
    assert events[-3:] == ['poll', 'terminate', ('wait', 30)]
    assert quiesced == []
    failure = json.loads((logs / ('HANDOFF_FAILED.' + m.GENERATION + '.json')).read_text())
    assert failure['unregistered_children'] == [{'pid': 101, 'returncode': -15, 'owned_popen_child_reaped': True}]
    assert failure['identity_observations'][0]['samples'][-1]['exact_command_match'] is False


def test_already_exited_unregistered_child_is_not_signaled():
    class Child:
        pid = 101
        def poll(self): return 1
        def wait(self, timeout): return 1
        def terminate(self): raise AssertionError('Must not signal exited child')
    assert m.reap_unregistered(Child())['returncode'] == 1


def test_transient_identity_then_exact_command_with_same_start_tick():
    class Child:
        pid = 101
        def poll(self): return None
    expected = ['python', '-u', 'run_fresh_confirmation.py']
    sequence = iter([None, {'command': ['python', 'old.py'], 'start_tick': 99},
                     {'command': expected, 'start_tick': 99}])
    samples = []
    current = m.wait_exact_identity(Child(), expected, samples,
                                   inspect=lambda _: next(sequence), sleep=lambda _: None)
    assert current == {'command': expected, 'start_tick': 99}
    assert [s['exact_command_match'] for s in samples] == [False, False, True]


def test_persistent_wrong_command_hits_bound_without_exposing_arguments():
    class Child:
        pid = 101
        def poll(self): return None
    ticks = iter([0., 0., 1., 2.])
    samples = []
    with pytest.raises(ValueError, match='startup bound'):
        m.wait_exact_identity(Child(), ['python', 'expected.py'], samples,
            inspect=lambda _: {'command': ['python', 'secret-value'], 'start_tick': 99},
            clock=lambda: next(ticks), sleep=lambda _: None)
    assert samples[-1]['elapsed_seconds'] == 2.
    assert 'secret-value' not in json.dumps(samples)


def test_start_tick_change_never_accepted_as_settling():
    class Child:
        pid = 101
        def poll(self): return None
    sequence = iter([{'command': ['old'], 'start_tick': 98},
                     {'command': ['expected'], 'start_tick': 99}])
    with pytest.raises(ValueError, match='start tick changed'):
        m.wait_exact_identity(Child(), ['expected'], [], inspect=lambda _: next(sequence), sleep=lambda _: None)


def test_v2_preserves_v1_failure_records(receiver):
    root, uuids = receiver
    logs = root / 'worker-logs-v2'
    for name in ('HANDOFF', 'HANDOFF_FAILED'):
        (logs / f'{name}.navigation-handoff-v1.json').write_text('{"preserve":true}')
    proposal = m.plan(root, uuids, inspect=lambda _: None)
    assert proposal['generation'] == 'navigation-handoff-v2'
    assert (logs / 'HANDOFF_FAILED.navigation-handoff-v1.json').read_text() == '{"preserve":true}'
