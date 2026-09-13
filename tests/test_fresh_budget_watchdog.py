import importlib.util
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/vast/fresh_budget_watchdog.py'
spec = importlib.util.spec_from_file_location('fresh_budget_watchdog', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_deadline_reserves_closeout_and_all_future_fleet_capacity():
    assert m.deadline(1000, 18, 20) == 1000 + 197 / 20 * 3600


@pytest.mark.parametrize('values', [(1000, 18, 0, 220, 5), (1000, -1, 20, 220, 5),
                                    (1000, 18, 20, 220, 4), (1000, float('nan'), 20, 220, 5)])
def test_invalid_budgets_fail_closed(values):
    with pytest.raises(ValueError):
        m.deadline(*values)


def test_exhausted_budget_is_immediate():
    assert m.deadline(1000, 216, 20) == 1000


def test_only_matching_live_process_is_signaled():
    worker = {'pid': 123, 'command': ['python', 'worker'], 'start_tick': 99}
    live = {123: {'command': worker['command'], 'start_tick': 99}}
    calls = []
    def send(pid, sig):
        calls.append((pid, sig))
        live.pop(pid, None)
    result = m.quiesce([worker], inspect=live.get, send=send, sleep=lambda _: None)
    assert calls == [(123, signal.SIGTERM)]
    assert result['quiesced'] and result['billing_continues']
    assert not result['provider_stopped'] and not result['cloud_verified']


def test_reused_pid_is_never_signaled():
    worker = {'pid': 123, 'command': ['python', 'worker'], 'start_tick': 99}
    calls = []
    result = m.quiesce([worker], inspect=lambda _: {'command': worker['command'], 'start_tick': 100},
                       send=lambda *x: calls.append(x), sleep=lambda _: None)
    assert not calls and not result['quiesced'] and result['identity_mismatches'] == [123]


def test_term_resistant_matching_worker_gets_kill_only_after_grace():
    worker = {'pid': 123, 'command': ['python', 'worker'], 'start_tick': 99}
    live = {'command': worker['command'], 'start_tick': 99}
    calls = []
    def send(pid, sig):
        nonlocal live
        calls.append(sig)
        if sig == signal.SIGKILL:
            live = None
    result = m.quiesce([worker], inspect=lambda _: live, send=send, sleep=lambda t: calls.append(t))
    assert calls == [signal.SIGTERM, 30, signal.SIGKILL, 1]
    assert result['quiesced']


def test_proc_identity_includes_start_tick(tmp_path):
    proc = tmp_path / '123'
    proc.mkdir()
    (proc / 'cmdline').write_bytes(b'python\0worker\0')
    (proc / 'stat').write_text('123 (python with spaces) S ' + ' '.join(['0'] * 18 + ['987']))
    assert m.identity(123, tmp_path) == {'command': ['python', 'worker'], 'start_tick': 987}


@pytest.mark.parametrize('count', [8, 4, 6])
def test_dry_run_neither_signals_nor_writes_guard(tmp_path, count):
    root = tmp_path
    logs = root / 'worker-logs-v2'
    logs.mkdir()
    freeze = root / 'fresh-freeze-v2'
    freeze.mkdir()
    protocol = b'{}'
    (freeze / 'protocol.json').write_bytes(protocol)
    sha = hashlib.sha256(protocol).hexdigest()
    mapping = [[i, 'reach', i, 8] for i in range(count)]
    assignment = {'instance': 42, 'freeze_sha256': sha, 'mapping': mapping}
    if count != 8:
        assignment['expected_gpu_count'] = count
    (logs / 'ASSIGNMENT.json').write_text(json.dumps(assignment))
    for i in range(count):
        command = [sys.executable, '-u', str(root / 'scripts/run_fresh_confirmation.py'),
                   'worker', '--project', str(root), '--freeze', str(freeze), '--task', 'reach',
                   '--worker-id', str(i), '--workers', '8', '--output', str(root / 'results-v2')]
        (logs / f'reach-worker{i}-LAUNCHED.json').write_text(json.dumps({
            'gpu': i, 'task': 'reach', 'worker_id': i, 'workers': 8, 'pid': 999999 + i,
            'command': command}))
    before = set(logs.iterdir())
    run = subprocess.run([sys.executable, str(path), '--root', str(root), '--instance', '42',
                          '--freeze-sha256', sha, '--as-of-utc', '2026-09-12T22:44:58Z',
                          '--accrued-usd', '18', '--dry-run'], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result['dry_run'] and result['signals_sent'] is False
    assert result['deadline_utc'] == '2026-09-13T08:35:58+00:00'
    assert set(logs.iterdir()) == before


@pytest.mark.parametrize('count,mapping', [
    (4, [[i, 'reach', i, 8] for i in range(3)]),
    (True, [[0, 'reach', 0, 8]]),
    (4, [[0, 'reach', i, 8] for i in range(4)]),
    (9, [[i, 'reach', i, 16] for i in range(9)]),
])
def test_explicit_gpu_count_rejects_missing_or_duplicate_lanes(tmp_path, count, mapping):
    logs = tmp_path / 'worker-logs-v2'
    logs.mkdir()
    freeze = tmp_path / 'fresh-freeze-v2'
    freeze.mkdir()
    (freeze / 'protocol.json').write_bytes(b'{}')
    digest = hashlib.sha256(b'{}').hexdigest()
    (logs / 'ASSIGNMENT.json').write_text(json.dumps({
        'instance': 42, 'freeze_sha256': digest, 'mapping': mapping,
        'expected_gpu_count': count}))
    with pytest.raises(ValueError, match='distinct explicitly assigned'):
        m.load_workers(tmp_path, 42, digest)
