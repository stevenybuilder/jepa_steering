import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('mw_waves', SCRIPTS / 'start_fresh_v2_mw_waves.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def proposal():
    return {'device_uuids': {str(i): f'uuid-{i}' for i in range(8)}, 'waves': m.wave_specs()}


def test_exact_24_scenarios_are_disjoint_from_original_and_reserved_stripes():
    waves = m.wave_specs()
    assert len(waves) == 6
    assert [len(w['jobs']) for w in waves] == [2] * 4 + [8] * 2
    ids = [(j['task'], j['episode']) for w in waves for j in w['jobs']]
    assert len(ids) == len(set(ids)) == 24
    expected = {(t, e) for t in ('reach', 'reach-wall') for e in range(96) if e % 8 == 7}
    assert set(ids) == expected
    counts = {gpu: sum(j['gpu'] == gpu for w in waves for j in w['jobs']) for gpu in range(8)}
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 6, 7: 6}
    assert not any(e % 8 in (3, 4, 5, 6) for _, e in ids)


def test_each_wave_has_exact_eight_guard_receipts():
    p = proposal()
    for wave in p['waves']:
        generation, jobs, assignment = m.wave_assignment(p, wave)
        assert len(assignment['mapping']) == len(assignment['launch_receipts']) == 8
        assert len(set(assignment['launch_receipts'])) == 8
        assert {row[0] for row in assignment['mapping']} == set(range(8))
        if wave['number'] <= 4:
            assert assignment['mapping'][:6] == m.ORIGINAL[:6]
        assert all(job['workers'] == 96 and job['worker_id'] == job['episode'] for job in jobs)


def test_frozen_worker_cli_retains_receiving_gate(tmp_path):
    argv = m.command(tmp_path, 'reach', 31, 96)
    assert argv[3] == 'worker'
    assert argv[argv.index('--workers') + 1] == '96'
    assert argv[argv.index('--worker-id') + 1] == '31'
    assert '--skip-engineering' not in argv and '--episode' not in argv


def test_exact_approval_no_implicit_mapping():
    p = proposal()
    approved = {'approved': True, 'globally_unstarted_and_unclaimed': True,
                'other_residual24_launchers_disabled': True, 'proposal': p}
    m.validate_approval(approved, p)
    altered = json.loads(json.dumps(p))
    altered['waves'][0]['jobs'][0]['episode'] = 3
    with pytest.raises(ValueError, match='Exact reviewed'):
        m.validate_approval(approved, altered)


@pytest.mark.parametrize('existing', ['directory', 'claim'])
def test_existing_incoming_output_or_claim_fails_closed(tmp_path, existing):
    job = {'gpu': 6, 'task': 'reach', 'episode': 7}
    if existing == 'directory':
        (tmp_path / 'results-v2/reach/scenario-007').mkdir(parents=True)
    else:
        save(tmp_path / 'worker-logs-v2/MW_CLAIM.reach.007.other.json', {})
    with pytest.raises(ValueError):
        m.check_unstarted(tmp_path, [job])


def test_deadline_reserves_whole_scenario_and_closeout(tmp_path):
    with pytest.raises(m.Paused, match='headroom'):
        m.budget_gate(tmp_path, 2500, 2100, now=101)
    m.budget_gate(tmp_path, 2500, 2100, now=100)


@pytest.mark.parametrize('cause', ['cancel', 'stop', 'budget'])
def test_stop_prevents_launches(tmp_path, cause):
    if cause == 'stop':
        save(tmp_path / 'worker-logs-v2/MW_WAVES_STOP.json', {})
    if cause == 'budget':
        save(tmp_path / 'worker-logs-v2/BUDGET_QUIESCED.some-generation.json', {})
    with pytest.raises(m.Paused, match='STOP'):
        m.budget_gate(tmp_path, 10000, 0, now=0, stopped=cause == 'cancel')


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    logs = tmp_path / 'worker-logs-v2'
    logs.mkdir()
    protocol = tmp_path / 'fresh-freeze-v2/protocol.json'
    save(protocol, {})
    digest = m.sha(protocol)
    monkeypatch.setattr(m, 'FREEZE', digest)
    assignment = logs / 'ASSIGNMENT.json'
    save(assignment, {'instance': m.INSTANCE, 'freeze_sha256': digest, 'mapping': m.ORIGINAL})
    workers = []
    for gpu, task, w, n in m.ORIGINAL:
        path = logs / f'{task}-worker{w}-LAUNCHED.json'
        argv = m.command(tmp_path, task, w, n)
        save(path, {'gpu': gpu, 'task': task, 'worker_id': w, 'workers': n,
                    'pid': 1000 + gpu, 'command': argv})
        workers.append({'pid': 1000 + gpu, 'command': argv, 'start_tick': gpu + 1,
                        'launch_receipt': str(path), 'launch_sha256': m.sha(path)})
    stamp = __import__('datetime').datetime.fromisoformat('2026-09-13T00:30:00+00:00').timestamp()
    until = __import__('datetime').datetime.fromisoformat(m.DEADLINE_UTC).timestamp()
    accrued = 215 - (until - stamp) / 3600 * 22
    budget = {'as_of_utc': '2026-09-13T00:30:00Z', 'accrued_usd': accrued,
              'fleet_cap_usd_hour': 22, 'campaign_cap_usd': 220, 'closeout_reserve_usd': 5}
    save(logs / 'BUDGET_WATCHDOG_ARMED.current22.json', {'instance': m.INSTANCE,
         'assignment_sha256': m.sha(assignment), 'workers': workers, 'budget': budget,
         'deadline_utc': m.DEADLINE_UTC, 'deadline_unix': m.guard.deadline(stamp, accrued, 22)})
    return tmp_path, {i: f'uuid-{i}' for i in range(8)}


def inspect_receiver(root, uuids):
    name = 'BUDGET_WATCHDOG_ARMED.current22.json'
    return m.inspect_root(root, uuids, name, m.sha(root / 'worker-logs-v2' / name))


def test_root_readonly_validation_and_no_budget_override(receiver):
    root, uuids = receiver
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    p, _, _ = inspect_receiver(root, uuids)
    assert p['campaign_cap_usd'] == 220 and p['deadline_utc'] == m.DEADLINE_UTC
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    path = root / 'worker-logs-v2/BUDGET_WATCHDOG_ARMED.current22.json'
    changed = json.loads(path.read_text())
    changed['budget']['campaign_cap_usd'] = 250
    save(path, changed)
    with pytest.raises(ValueError, match='override'):
        inspect_receiver(root, uuids)


def test_guard_arithmetic_tamper_rejected(receiver):
    root, uuids = receiver
    path = root / 'worker-logs-v2/BUDGET_WATCHDOG_ARMED.current22.json'
    changed = json.loads(path.read_text())
    changed['deadline_unix'] += 100
    save(path, changed)
    with pytest.raises(ValueError, match='arithmetic'):
        inspect_receiver(root, uuids)


def test_exact_approved_budget_receipt_hash_required(receiver):
    root, uuids = receiver
    with pytest.raises(ValueError, match='receipt SHA256'):
        m.inspect_root(root, uuids, 'BUDGET_WATCHDOG_ARMED.current22.json', '0' * 64)


def test_run_stays_disabled_before_hardware_or_filesystem_access(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['waves', '--root', '/not-present', '--run',
        '--budget-receipt-name', 'approved.json', '--budget-receipt-sha256', '0' * 64])
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: pytest.fail('No hardware access'))
    with pytest.raises(SystemExit) as error:
        m.main()
    assert error.value.code == 2
    assert 'Execution remains disabled pending root review' in capsys.readouterr().err


def test_failed_spawn_cleanup_covers_owned_child_not_originals(tmp_path, monkeypatch):
    (tmp_path / 'worker-logs-v2').mkdir()
    _, jobs, _ = m.wave_assignment(proposal(), m.wave_specs()[0])
    events = []
    class Child:
        pid = 99
        def poll(self): return None
        def terminate(self): events.append('terminate-owned')
        def wait(self, timeout): events.append('reaped-owned'); return -15
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: Child())
    monkeypatch.setattr(m, 'wait_exact_identity', lambda *a: (_ for _ in ()).throw(ValueError('identity')))
    owned, pending = [], []
    with pytest.raises(ValueError, match='identity'):
        m.spawn_jobs(tmp_path, jobs, owned, pending, lambda: None, [])
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: {'quiesced': not workers})
    result = m.cleanup(owned, pending)
    assert events == ['terminate-owned', 'reaped-owned']
    assert result['quiesced'] and result['unregistered_children'][0]['pid'] == 99


def test_stop_check_precedes_any_popen(tmp_path, monkeypatch):
    _, jobs, _ = m.wave_assignment(proposal(), m.wave_specs()[0])
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: pytest.fail('No launch allowed'))
    with pytest.raises(m.Paused):
        m.spawn_jobs(tmp_path, jobs, [], [], lambda: (_ for _ in ()).throw(m.Paused('STOP')), [])


def test_guard_failure_is_fail_closed(tmp_path, monkeypatch):
    logs = tmp_path / 'worker-logs-v2'
    logs.mkdir()
    generation, _, assignment = m.wave_assignment(proposal(), m.wave_specs()[0])
    class Child:
        pid = 44
        def poll(self): return 1
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: Child())
    prior = {'budget': {k: 0 for k in ('as_of_utc', 'accrued_usd', 'fleet_cap_usd_hour',
                                      'campaign_cap_usd', 'closeout_reserve_usd')}}
    with pytest.raises(RuntimeError, match='guard failed'):
        m.arm_wave(tmp_path, generation, assignment, prior, [], lambda: None)
    assert (logs / ('ACTIVE_ASSIGNMENT.' + generation + '.json')).exists()


def test_cleanup_registered_children_never_includes_originals(monkeypatch):
    new = {'pid': 9001, 'command': ['python', 'frozen-worker'], 'start_tick': 8}
    seen = []
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: seen.extend(workers) or {'quiesced': True})
    assert m.cleanup([new], [])['quiesced']
    assert seen == [new]


def test_controller_guard_failure_cleans_only_current_wave(receiver, monkeypatch):
    root, uuids = receiver
    p, prior, originals = inspect_receiver(root, uuids)
    prior['watchdog_pid'] = 500
    manifest = root / 'approved.json'
    save(manifest, {'approved': True, 'globally_unstarted_and_unclaimed': True,
                    'other_residual24_launchers_disabled': True, 'proposal': p})
    original_guard = {'command': ['python', 'fresh_budget_watchdog.py', '--root', str(root)], 'start_tick': 88}
    monkeypatch.setattr(m.guard, 'identity', lambda pid: original_guard if pid == 500 else None)
    monkeypatch.setattr(m, 'budget_gate', lambda *a, **k: None)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m, 'verify_bundle', lambda *a: 'verified')
    def spawn(root, jobs, owned, pending, check, observations):
        owned.extend([{'pid': 6000 + j['gpu'], 'command': ['frozen'], 'start_tick': 1} for j in jobs])
        return []
    monkeypatch.setattr(m, 'spawn_jobs', spawn)
    monkeypatch.setattr(m, 'arm_wave', lambda *a: (_ for _ in ()).throw(RuntimeError('guard failed')))
    seen = []
    monkeypatch.setattr(m.guard, 'quiesce', lambda workers: seen.extend(workers) or {'quiesced': True})
    with pytest.raises(RuntimeError, match='guard failed'):
        m.run(root, p, prior, originals, manifest)
    assert {w['pid'] for w in seen} == {6006, 6007}
    paused = json.loads((root / 'worker-logs-v2' / ('WAVES_PAUSED.' + m.GENERATION + '.json')).read_text())
    assert paused['original_workers_signaled_by_controller'] is False
