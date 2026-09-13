import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('static_queue', SCRIPTS / 'run_fresh_static_queue.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    protocol = {'source_sha256': m.SOURCE, 'tasks': {t: {'records': [
        {'episode': e, 'role': 'scientific_candidates'} for e in range(96)]} for t in ('pointmaze', 'wall')}}
    save(tmp_path / 'fresh-freeze-v2/protocol.json', protocol)
    monkeypatch.setattr(m, 'FREEZE', m.sha(tmp_path / 'fresh-freeze-v2/protocol.json'))
    (tmp_path / 'worker-logs-v2').mkdir()
    uuids = {i: f'uuid-{i}' for i in range(8)}
    request = {'instance': 50806821, 'generation': 'queue-tail-v1', 'queues': [
        {'gpu': 6, 'task': 'pointmaze', 'episodes': [89, 93], 'device_uuid': 'uuid-6'},
        {'gpu': 7, 'task': 'wall', 'episodes': [90, 94], 'device_uuid': 'uuid-7'}]}
    return tmp_path, request, uuids


def plan(receiver, **kwargs):
    root, request, uuids = receiver
    return m.plan(root, request, uuids, 'same-boot', validate=kwargs.get('validate', lambda root: None),
                  busy=kwargs.get('busy', lambda *a: False))


def test_exact_order_and_single_child_guard_without_dummy_rows(receiver):
    p = plan(receiver)
    assert [[j['episode'] for j in q['jobs']] for q in p['queues']] == [[89, 93], [90, 94]]
    for q in p['queues']:
        for job in q['jobs']:
            a = m.assignment(p, job)
            assert a['expected_gpu_count'] == len(a['mapping']) == len(a['launch_receipts']) == 1
            assert a['mapping'][0] == [job['gpu'], job['task'], job['episode'], 96]


def test_plan_readonly_and_new_source_validation_required(receiver):
    root = receiver[0]
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan(receiver)
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='input modified'):
        plan(receiver, validate=lambda root: (_ for _ in ()).throw(ValueError('input modified')))


def test_busy_selected_gpu_denied(receiver):
    with pytest.raises(ValueError, match='Selected GPU still busy'):
        plan(receiver, busy=lambda *a: True)


def test_duplicate_task_episode_across_lanes_denied(receiver):
    request = receiver[1]
    request['queues'][1].update(task='pointmaze', episodes=[89])
    with pytest.raises(ValueError, match='duplicate scientific'):
        plan(receiver)


def test_duplicate_gpu_or_wrong_uuid_denied(receiver):
    request = receiver[1]
    request['queues'][1]['gpu'] = 6
    with pytest.raises(ValueError, match='Distinct existing'):
        plan(receiver)
    request['queues'][1]['gpu'] = 7
    request['queues'][1]['device_uuid'] = 'changed-hardware'
    with pytest.raises(ValueError, match='Task/UUID'):
        plan(receiver)


def test_existing_scenario_requires_exact_same_uuid_partial(receiver):
    root, request, _ = receiver
    path = root / 'results-v2/pointmaze/scenario-089/STARTED.json'
    intent = {'task': 'pointmaze', 'scenario': {'episode': 89, 'role': 'scientific_candidates'},
              'device_uuid': 'uuid-6', 'freeze_sha256': m.FREEZE, 'engineering': False}
    save(path, intent)
    with pytest.raises(ValueError, match='already exists'):
        plan(receiver)
    request['queues'][0]['same_uuid_partials'] = {'89': m.sha(path)}
    plan(receiver)
    intent['device_uuid'] = 'uuid-7'
    save(path, intent)
    request['queues'][0]['same_uuid_partials']['89'] = m.sha(path)
    with pytest.raises(ValueError, match='Partial scenario input/UUID'):
        plan(receiver)


def test_existing_local_claim_denied(receiver):
    save(receiver[0] / 'worker-logs-v2/STATIC_CLAIM.pointmaze.089.other.json', {})
    with pytest.raises(ValueError, match='already claimed'):
        plan(receiver)


def test_global_approval_binds_donor_quiescence_and_exact_claims(receiver):
    p = plan(receiver)
    path = receiver[0] / 'approval.json'
    record = {'approved': True, 'proposal': p, 'donors_quiesced_and_started_reconciled': True,
              'global_claims': p['global_claims'], 'competing_launchers_disabled_for_claims': True}
    save(path, record)
    m.approve(path, p)
    record['global_claims'] = record['global_claims'][:-1]
    save(path, record)
    with pytest.raises(ValueError, match='Exact root approval'):
        m.approve(path, p)


def test_gpu_lanes_advance_without_cross_gpu_barrier():
    queues = [{'gpu': 6, 'jobs': ['fast1', 'fast2']}, {'gpu': 7, 'jobs': ['slow1']}]
    events, ticks = [], [0]
    class Child:
        def __init__(self, due): self.due = due
        def poll(self): return 0 if ticks[0] >= self.due else None
    def start(job):
        events.append(('start', job, ticks[0]))
        return {'child': Child(ticks[0] + (4 if job == 'slow1' else 1)), 'job': job}
    def finish(active): events.append(('finish', active['job'], ticks[0]))
    m.run_lanes(queues, start, finish, lambda: None, sleep=lambda _: ticks.__setitem__(0, ticks[0] + 1))
    assert next(t for e, j, t in events if e == 'start' and j == 'fast2') < next(
        t for e, j, t in events if e == 'finish' and j == 'slow1')


def test_nonzero_child_stops_queue_without_retry():
    class Child:
        def poll(self): return 1
    started = []
    with pytest.raises(ValueError, match='no automatic retry'):
        m.run_lanes([{'gpu': 1, 'jobs': ['first', 'second']}],
                    lambda j: started.append(j) or {'child': Child()}, lambda a: None,
                    lambda: None, sleep=lambda _: None)
    assert started == ['first']


def test_stop_checked_before_any_child_start():
    with pytest.raises(ValueError, match='STOP'):
        m.run_lanes([{'gpu': 1, 'jobs': ['first']}], lambda j: pytest.fail('No spawn'),
            lambda a: None, lambda: (_ for _ in ()).throw(ValueError('STOP')))


def test_unregistered_child_cleanup_after_identity_failure(receiver, monkeypatch):
    p = plan(receiver)
    job = p['queues'][0]['jobs'][0]
    events = []
    class Child:
        pid = 889
        def poll(self): return None
        def terminate(self): events.append('own-terminate')
        def wait(self, timeout): return -15
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: Child())
    monkeypatch.setattr(m, 'wait_exact_identity', lambda *a: (_ for _ in ()).throw(ValueError('identity')))
    owned, pending, observed = [], [], []
    with pytest.raises(ValueError, match='identity'):
        m.start_job(receiver[0], p, job, owned, pending, observed, lambda **kw: None)
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: events.extend(workers) or {})
    report = m.cleanup(owned, pending)
    assert events == ['own-terminate'] and report['unregistered_children'][0]['owned_popen_child_reaped']


def test_engineering_headroom_added_for_unqualified_task(receiver, monkeypatch):
    p = plan(receiver)
    job = p['queues'][0]['jobs'][0]
    allowances = []
    def check(**kw):
        allowances.append(kw['allowance'])
        raise ValueError('budget stop')
    with pytest.raises(ValueError, match='budget stop'):
        m.start_job(receiver[0], p, job, [], [], [], check)
    assert allowances == [1860]


def test_guard_failure_leaves_only_owned_child_for_cleanup(receiver, monkeypatch):
    p = plan(receiver)
    job = p['queues'][0]['jobs'][0]
    class Child:
        def __init__(self, pid, code): self.pid, self.code = pid, code
        def poll(self): return self.code
    children = iter([Child(900, None), Child(901, 1)])
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **kw: next(children))
    monkeypatch.setattr(m, 'wait_exact_identity', lambda child, argv, samples: {'command': argv, 'start_tick': 42})
    owned, pending = [], []
    with pytest.raises(ValueError, match='guard failed to arm'):
        m.start_job(receiver[0], p, job, owned, pending, [], lambda **kw: None)
    assert [w['pid'] for w in owned] == [900] and pending == []
    seen = []
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: seen.extend(workers) or {})
    m.cleanup(owned, pending)
    assert [w['pid'] for w in seen] == [900]


def test_one_lane_assignment_is_accepted_by_actual_guard(receiver, monkeypatch):
    p = plan(receiver)
    root = receiver[0]
    job = p['queues'][0]['jobs'][0]
    a = m.assignment(p, job)
    aname = 'ASSIGNMENT.single-test.json'
    save(root / 'worker-logs-v2' / aname, a)
    save(root / 'worker-logs-v2' / job['launch_receipt'], {**job, 'pid': 999,
         'command': m.command(root, job['task'], job['episode'], 96), 'start_tick': 1})
    monkeypatch.setattr(m.guard, 'identity', lambda pid: None)
    workers, digest = m.guard.load_workers(root, p['instance'], m.FREEZE, aname)
    assert len(workers) == 1
    assert workers[0]['pid'] == 999


def test_default_budget_proposal_unchanged_without_optional_receipt(receiver):
    p = plan(receiver)
    assert p['budget'] == m.BUDGET and p['deadline_unix'] == m.DEADLINE
    assert 'budget_binding' not in p


def test_explicit_budget_receipt_pins_new_caps_for_future_jobs(receiver):
    root, request, uuids = receiver
    path = root / 'approved-budget.json'
    budget = {'as_of_utc': '2026-09-13T03:00:00Z', 'accrued_usd': 115,
              'fleet_cap_usd_hour': 39, 'campaign_cap_usd': 320, 'closeout_reserve_usd': 7}
    save(path, {'approved': True, 'source_snapshot_sha256': 'a' * 64, 'budget': budget})
    p = m.plan(root, request, uuids, 'boot', validate=lambda root: None, busy=lambda *a: False,
               budget_receipt=path, budget_sha=m.sha(path))
    assert p['budget'] == budget
    assert p['budget_binding'] == {'path': str(path), 'sha256': m.sha(path)}
    assert p['deadline_unix'] > m.DEADLINE
    with pytest.raises(ValueError, match='provided together'):
        m.plan(root, request, uuids, 'boot', budget_receipt=path)


def test_new_job_guard_uses_proposal_budget_not_legacy_globals(receiver, monkeypatch):
    root = receiver[0]
    p = plan(receiver)
    p['budget'] = {**p['budget'], 'fleet_cap_usd_hour': 39, 'campaign_cap_usd': 320}
    p['deadline_unix'] += 1000
    job = p['queues'][0]['jobs'][0]
    launches = []
    class Child:
        def __init__(self, pid): self.pid = pid
        def poll(self): return None
    def spawn(argv, **kwargs):
        launches.append(argv)
        if len(launches) == 1:
            return Child(800)
        path = root / 'worker-logs-v2' / m.guard.guard_names(job['generation'])[1]
        save(path, {'watchdog_pid': 801, 'assignment_sha256': m.sha(root / 'worker-logs-v2' / f'ASSIGNMENT.{job["generation"]}.json'),
             'deadline_unix': p['deadline_unix'], 'workers': [{'pid': 800, 'command': launches[0], 'start_tick': 42}]})
        return Child(801)
    monkeypatch.setattr(m.subprocess, 'Popen', spawn)
    monkeypatch.setattr(m, 'wait_exact_identity', lambda child, argv, samples: {'command': argv, 'start_tick': 42})
    monkeypatch.setattr(m.guard, 'identity', lambda pid: {'command': ['guard'], 'start_tick': 43})
    active = m.start_job(root, p, job, [], [], [], lambda **kw: None)
    argv = launches[1]
    assert argv[argv.index('--fleet-cap-usd-hour') + 1] == '39'
    assert argv[argv.index('--campaign-cap-usd') + 1] == '320'
    assert active['guard_sha256']
