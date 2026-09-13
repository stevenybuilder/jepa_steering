import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('rebase_budget', SCRIPTS / 'rebase_fresh_budget.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def budget_receipt(tmp_path):
    path = tmp_path / 'approved-budget.json'
    save(path, {'approved': True, 'source_snapshot_sha256': 'a' * 64, 'budget': {
        'as_of_utc': '2026-09-13T03:00:00Z', 'accrued_usd': 115,
        'fleet_cap_usd_hour': 39, 'campaign_cap_usd': 320, 'closeout_reserve_usd': 7}})
    return path


def test_approved_budget_exact_caps_and_arithmetic(budget_receipt):
    budget, until, binding = m.load_budget(budget_receipt, m.sha(budget_receipt))
    assert until == m.guard.deadline(m.datetime.fromisoformat('2026-09-13T03:00:00+00:00').timestamp(), 115, 39, 320, 7)
    assert binding == {'path': str(budget_receipt), 'sha256': m.sha(budget_receipt)}


@pytest.mark.parametrize('field,value', [('fleet_cap_usd_hour', 40), ('campaign_cap_usd', 321),
                                        ('closeout_reserve_usd', 4), ('accrued_usd', float('nan'))])
def test_budget_not_inferred_or_relaxed(budget_receipt, field, value):
    record = m.read(budget_receipt)
    record['budget'][field] = value
    save(budget_receipt, record)
    with pytest.raises(ValueError):
        m.load_budget(budget_receipt, m.sha(budget_receipt))


def test_budget_sha_and_explicit_approval_required(budget_receipt):
    with pytest.raises(ValueError, match='SHA256'):
        m.load_budget(budget_receipt, 'b' * 64)
    record = m.read(budget_receipt)
    record['approved'] = False
    save(budget_receipt, record)
    with pytest.raises(ValueError, match='Explicit approved'):
        m.load_budget(budget_receipt, m.sha(budget_receipt))


@pytest.fixture
def identities(budget_receipt):
    budget, until, binding = m.load_budget(budget_receipt, m.sha(budget_receipt))
    worker = {'pid': 10, 'command': ['python', 'run_fresh_confirmation.py'], 'start_tick': 100,
              'launch_receipt': '/frozen/worker-logs-v2/worker.json', 'launch_sha256': 'c' * 64}
    old = {'pid': 20, 'command': ['python', 'fresh_budget_watchdog.py'], 'start_tick': 200, 'boot_id': 'boot'}
    new = {'command': ['python', 'fresh_budget_watchdog.py', '--generation', 'rebase-v1'], 'start_tick': 300}
    p = {'instance': 50806821, 'assignment_sha256': 'a' * 64, 'budget': budget, 'deadline_unix': until,
         'workers': [worker], 'old_guards': [old], 'budget_binding': binding}
    receipt = {'watchdog_pid': 30, 'instance': p['instance'], 'assignment_sha256': p['assignment_sha256'],
               'deadline_unix': until, 'workers': [worker], 'budget': budget}
    def inspect(pid):
        return {10: {k: worker[k] for k in ('command', 'start_tick')},
                20: {k: old[k] for k in ('command', 'start_tick')}, 30: new}.get(pid)
    return p, receipt, inspect, new


def test_new_guard_must_cover_exact_live_worker_before_retirement(identities):
    p, receipt, inspect, new = identities
    watcher = type('Child', (), {'pid': 30})()
    m.verify_new_guard(Path('/frozen'), p, watcher, receipt, inspect=inspect, expected_watchdog_identity=new)
    receipt['workers'] = []
    with pytest.raises(ValueError, match='coverage'):
        m.verify_new_guard(Path('/frozen'), p, watcher, receipt, inspect=inspect)


def test_new_guard_pid_identity_or_budget_mismatch_rejected(identities):
    p, receipt, inspect, new = identities
    watcher = type('Child', (), {'pid': 30})()
    with pytest.raises(ValueError, match='identity/assignment'):
        m.verify_new_guard(Path('/frozen'), p, watcher, receipt, inspect=inspect,
                           expected_watchdog_identity={**new, 'start_tick': 999})
    receipt['budget'] = {**receipt['budget'], 'campaign_cap_usd': 999}
    with pytest.raises(ValueError, match='identity/assignment'):
        m.verify_new_guard(Path('/frozen'), p, watcher, receipt, inspect=inspect)


def test_retirement_signals_only_watchdogs_never_science(identities):
    p, _, inspect, _ = identities
    seen = []
    result = m.retire_old_watchdogs(p, inspect=inspect,
        quiesce=lambda rows, **kw: seen.extend(rows) or {'quiesced': True})
    assert result['quiesced'] and [r['pid'] for r in seen] == [20]
    assert 10 not in [r['pid'] for r in seen]


def test_changed_old_guard_refuses_all_retirement(identities):
    p, _, inspect, _ = identities
    with pytest.raises(ValueError, match='nothing retired'):
        m.retire_old_watchdogs(p, inspect=lambda pid: {'command': ['other'], 'start_tick': 200},
                              quiesce=lambda *a, **kw: pytest.fail('No signal permitted'))


def test_guard_command_must_match_receipt_generation():
    root = Path('/frozen')
    r = {'instance': 1, 'budget': {'assignment_name': 'ACTIVE.json', 'generation': 'old-v1'}}
    argv = ['python', '/ops/fresh_budget_watchdog.py', '--root', str(root), '--instance', '1',
            '--freeze-sha256', m.FREEZE, '--assignment-name', 'ACTIVE.json', '--generation', 'old-v1']
    assert m.guard_command_matches(argv, r, root)
    assert not m.guard_command_matches([*argv[:-1], 'unrelated-v2'], r, root)


def test_failed_arming_preserves_old_guards_and_reaps_only_new_child(tmp_path, budget_receipt, monkeypatch):
    logs = tmp_path / 'worker-logs-v2'
    logs.mkdir()
    budget, until, binding = m.load_budget(budget_receipt, m.sha(budget_receipt))
    proposal = {'generation': 'rebase-fail-v1', 'instance': 50806821, 'request': {}, 'boot_id': 'boot',
        'assignment_name': 'ACTIVE.json', 'budget': budget, 'deadline_unix': until, 'budget_binding': binding}
    approval = tmp_path / 'approval.json'
    save(approval, {'approved': True, 'proposal': proposal, 'only_named_watchdogs_may_be_retired': True})
    monkeypatch.setattr(m, 'plan', lambda *a: proposal)
    original_read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **kw: 'boot' if str(p) == '/proc/sys/kernel/random/boot_id'
                        else original_read_text(p, *a, **kw))
    monkeypatch.setattr(m.time, 'time', lambda: until - 1000)
    events = []
    class Child:
        pid = 30
        def poll(self): return 1
        def wait(self, timeout): events.append('reaped-new'); return 1
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **kw: Child())
    monkeypatch.setattr(m, 'wait_exact_identity', lambda *a: {'command': ['guard'], 'start_tick': 300})
    monkeypatch.setattr(m, 'retire_old_watchdogs', lambda *a: pytest.fail('Old guard must stay live'))
    with pytest.raises(ValueError, match='failed to arm'):
        m.launch(tmp_path, proposal, approval)
    assert events == ['reaped-new']
    assert m.read(logs / 'REBASE_FAILED.rebase-fail-v1.json')['scientific_workers_signaled'] is False
