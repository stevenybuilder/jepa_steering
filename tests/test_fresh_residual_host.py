import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('residual_host', SCRIPTS / 'start_fresh_v2_residual_host.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    for name in ('RUNTIME_READY.json', 'ASSETS_READY.json'):
        save(tmp_path / name, {})
    save(tmp_path / 'fresh-freeze-v2/protocol.json', {'source_sha256': m.SOURCE})
    monkeypatch.setattr(m, 'FREEZE', m.sha(tmp_path / 'fresh-freeze-v2/protocol.json'))
    return tmp_path, {i: f'uuid-{i}' for i in range(4)}


def plan(receiver, task='reach', **kwargs):
    return m.plan(receiver[0], 50833029, task, receiver[1],
                  validate=kwargs.get('validate', lambda root: None), scan=lambda root: [])


def test_exact_two_host_residual24_coverage_and_four_guard_rows(receiver):
    cases = []
    for task in ('reach', 'reach-wall'):
        p = plan(receiver, task)
        assert p['expected_gpu_count'] == len(p['mapping']) == len(set(p['launch_receipts'])) == 4
        assert [j['worker_id'] for j in p['jobs']] == [7, 15, 23, 31]
        assert all(j['workers'] == 32 and len(j['episodes']) == 3 for j in p['jobs'])
        cases += [(task, e) for j in p['jobs'] for e in j['episodes']]
    assert len(cases) == len(set(cases)) == 24
    assert set(cases) == {(t, e) for t in ('reach', 'reach-wall') for e in range(96) if e % 8 == 7}


def test_plan_readonly_and_frozen_worker_receiving_retained(receiver):
    root = receiver[0]
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    p = plan(receiver)
    assert all(j['command'][3] == 'worker' and '--skip-engineering' not in j['command'] for j in p['jobs'])
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_physical_gpu_count_required(receiver):
    receiver[1][4] = 'extra'
    with pytest.raises(ValueError, match='Four distinct'):
        plan(receiver)


@pytest.mark.parametrize('kind', ['results', 'launch'])
def test_existing_work_rejected(receiver, kind):
    root = receiver[0]
    if kind == 'results':
        (root / 'results-v2').mkdir()
    else:
        save(root / 'worker-logs-v2/old-LAUNCHED.json', {})
    with pytest.raises(ValueError, match='Existing results'):
        plan(receiver)


def test_frozen_input_validator_failure_rejected(receiver):
    with pytest.raises(ValueError, match='bad asset'):
        plan(receiver, validate=lambda root: (_ for _ in ()).throw(ValueError('bad asset')))


def test_approval_binds_task_instance_and_budget(receiver):
    p = plan(receiver)
    path = receiver[0] / 'approved.json'
    save(path, {'approved': True, 'proposal': p, 'globally_unstarted_and_unclaimed': True,
                'other_residual_launchers_disabled': True, 'provider_and_budget_authorized': True})
    m.approve(path, p)
    altered = json.loads(json.dumps(p))
    altered['budget']['campaign_cap_usd'] = 250
    with pytest.raises(ValueError, match='Exact instance'):
        m.approve(path, altered)


def test_new_child_identity_failure_reaped_without_unowned_signals(receiver, monkeypatch):
    root = receiver[0]
    p = plan(receiver)
    path = root / 'approved.json'
    save(path, {'approved': True, 'proposal': p, 'globally_unstarted_and_unclaimed': True,
                'other_residual_launchers_disabled': True, 'provider_and_budget_authorized': True})
    monkeypatch.setattr(m, 'plan', lambda *a, **k: p)
    monkeypatch.setattr(m.time, 'time', lambda: m.DEADLINE - 10000)
    events = []
    class Child:
        pid = 888
        def poll(self): return None
        def terminate(self): events.append('own-terminate')
        def wait(self, timeout): return -15
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: Child())
    monkeypatch.setattr(m, 'wait_exact_identity', lambda *a: (_ for _ in ()).throw(ValueError('identity')))
    monkeypatch.setattr(m.guard, 'quiesce', lambda owned: events.extend(owned) or {})
    with pytest.raises(ValueError, match='identity'):
        m.launch(root, p, path)
    assert events == ['own-terminate']
    failed = m.read(root / 'worker-logs-v2' / f'RESIDUAL_FAILED.{m.GENERATION}.json')
    assert failed['unregistered_children'][0]['owned_popen_child_reaped'] is True
