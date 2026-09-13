import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('resume_host', SCRIPTS / 'resume_fresh_v2_host.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    instance = 50819364
    mapping = [[0, 'reach', 3, 8], [1, 'pointmaze', 2, 4], [2, 'reach', 5, 8],
               [3, 'wall', 2, 4], [4, 'pointmaze', 3, 4], [5, 'wall', 3, 4],
               [6, 'pointmaze', 1, 4], [7, 'wall', 1, 4]]
    protocol = {'source_sha256': m.SOURCE, 'tasks': {t: {'records': [
        {'episode': 0, 'role': 'excluded_engineering'}, *[{'episode': e, 'role': 'scientific_candidates'}
        for e in range(96)]]} for t in ('reach', 'pointmaze', 'wall')}}
    save(tmp_path / 'fresh-freeze-v2/protocol.json', protocol)
    monkeypatch.setattr(m, 'FREEZE', m.sha(tmp_path / 'fresh-freeze-v2/protocol.json'))
    logs = tmp_path / 'worker-logs-v2'
    names, workers = [], []
    for gpu, task, worker, count in mapping:
        name = f'old-gpu{gpu}-LAUNCHED.json'
        names.append(name)
        save(logs / name, {'gpu': gpu, 'task': task, 'worker_id': worker, 'workers': count,
              'pid': 100 + gpu, 'start_tick': 500 + gpu, 'command': m.command(tmp_path, task, worker, count)})
        workers.append({'launch_receipt': str(logs / name), 'launch_sha256': m.sha(logs / name)})
        save(tmp_path / 'results-v2/engineering' / task / f'uuid-{gpu}' / 'STARTED.json',
             {'task': task, 'scenario': protocol['tasks'][task]['records'][0], 'device_uuid': f'uuid-{gpu}',
              'freeze_sha256': m.FREEZE, 'engineering': True})
    aname = m.HOSTS[instance][0]
    save(logs / aname, {'instance': instance, 'freeze_sha256': m.FREEZE, 'mapping': mapping,
                        'launch_receipts': names, 'device_uuids': {str(i): f'uuid-{i}' for i in range(8)}})
    stamp = m.datetime.fromisoformat('2026-09-13T01:00:00+00:00').timestamp()
    budget = {'as_of_utc': '2026-09-13T01:00:00Z', 'accrued_usd': 60,
              'fleet_cap_usd_hour': 35, 'campaign_cap_usd': 220, 'closeout_reserve_usd': 5}
    bname = 'OLD_GUARD.json'
    save(logs / bname, {'instance': instance, 'assignment_sha256': m.sha(logs / aname),
                       'workers': workers, 'budget': budget, 'deadline_unix': m.guard.deadline(stamp, 60, 35)})
    return tmp_path, instance, aname, bname, {i: f'uuid-{i}' for i in range(8)}


def plan(receiver, **overrides):
    root, instance, aname, bname, uuids = receiver
    opts = {'inspect': lambda pid: None, 'scan': lambda root: [], 'validate': lambda root: None}
    opts.update(overrides)
    return m.plan(root, instance, aname, m.sha(root / 'worker-logs-v2' / aname), bname,
        m.sha(root / 'worker-logs-v2' / bname), uuids, 'resume-provider-v1', 'new-boot', **opts)


def test_exact_same_assignment_six_nav_and_eight_guard_receipts(receiver):
    root, _, aname, _, _ = receiver
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    proposal, _ = plan(receiver)
    assert proposal['mapping'] == m.read(root / 'worker-logs-v2' / aname)['mapping']
    assert {j['gpu'] for j in proposal['jobs']} == {1, 3, 4, 5, 6, 7}
    assert len(set(proposal['launch_receipts'])) == 8
    assert proposal['launch_receipts'][0] == 'old-gpu0-LAUNCHED.json'
    assert proposal['boot_id'] == 'new-boot'
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_live_or_reused_old_pid_denied(receiver):
    with pytest.raises(ValueError, match='Old PID live or reused'):
        plan(receiver, inspect=lambda pid: {'command': ['different'], 'start_tick': 1})


def test_scoped_duplicate_worker_denied(receiver):
    with pytest.raises(ValueError, match='scoped worker'):
        plan(receiver, scan=lambda root: [{'pid': 888}])


def test_hardware_changed_denied(receiver):
    receiver[-1][1] = 'another-card'
    with pytest.raises(ValueError, match='hardware UUID'):
        plan(receiver)


def test_source_changed_denied(receiver, monkeypatch):
    root = receiver[0]
    path = root / 'fresh-freeze-v2/protocol.json'
    value = m.read(path)
    value['source_sha256'] = 'bad'
    save(path, value)
    monkeypatch.setattr(m, 'FREEZE', m.sha(path))
    with pytest.raises(ValueError, match='freeze/source'):
        plan(receiver)


def test_full_frozen_input_validation_required(receiver):
    with pytest.raises(ValueError, match='changed tensor'):
        plan(receiver, validate=lambda root: (_ for _ in ()).throw(ValueError('changed tensor')))


def test_partial_scenario_identity_and_receipt_preserved(receiver):
    root = receiver[0]
    folder = root / 'results-v2/pointmaze/scenario-002'
    identity = {'task': 'pointmaze', 'scenario': {'episode': 2, 'role': 'scientific_candidates'},
                'device_uuid': 'uuid-1', 'freeze_sha256': m.FREEZE, 'engineering': False}
    save(folder / 'STARTED.json', identity)
    save(folder / 'native.json', {'opaque_atomic_record': True})
    before = (folder / 'native.json').read_bytes()
    proposal, _ = plan(receiver)
    assert proposal['history'][1]['started_sha256']['results-v2/pointmaze/scenario-002'] == m.sha(folder / 'STARTED.json')
    assert (folder / 'native.json').read_bytes() == before
    identity['device_uuid'] = 'wrong'
    save(folder / 'STARTED.json', identity)
    with pytest.raises(ValueError, match='partial/physical'):
        plan(receiver)


def test_budget_deadline_override_denied(receiver):
    root, _, _, name, _ = receiver
    path = root / 'worker-logs-v2' / name
    value = m.read(path)
    value['deadline_unix'] += 100
    save(path, value)
    with pytest.raises(ValueError, match='budget/deadline'):
        plan(receiver)


def test_reviewed_approval_binds_mapping_and_provider_permission(receiver):
    root = receiver[0]
    proposal, _ = plan(receiver)
    path = root / 'approval.json'
    record = {'approved': True, 'proposal': proposal, 'global_same_assignment_resume_verified': True,
              'provider_resume_authorized': True}
    save(path, record)
    assert m.approve(path, proposal) == m.sha(path)
    record['provider_resume_authorized'] = False
    save(path, record)
    with pytest.raises(ValueError, match='reviewed same-assignment'):
        m.approve(path, proposal)


def test_frozen_validator_readonly_command(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(m.subprocess, 'run', lambda argv, **kw: seen.append((argv, kw)))
    m.validate_files(tmp_path)
    assert seen[0][0][2] == 'validate'
    assert seen[0][1]['check'] is True
    assert 'worker' not in seen[0][0]


@pytest.fixture
def four_receiver(tmp_path, monkeypatch):
    instance, aname, bname = 50836730, 'ASSIGNMENT.json', 'GUARD4.json'
    mapping = [[gpu, 'reach-wall', worker, 32] for gpu, worker in enumerate((7, 15, 23, 31))]
    protocol = {'source_sha256': m.SOURCE, 'tasks': {'reach-wall': {'records': [
        {'episode': 0, 'role': 'excluded_engineering'}, *[{'episode': e, 'role': 'scientific_candidates'}
        for e in range(96)]]}}}
    save(tmp_path / 'fresh-freeze-v2/protocol.json', protocol)
    monkeypatch.setattr(m, 'FREEZE', m.sha(tmp_path / 'fresh-freeze-v2/protocol.json'))
    logs, names, workers = tmp_path / 'worker-logs-v2', [], []
    for gpu, task, worker, count in mapping:
        name = f'old-gpu{gpu}-LAUNCHED.json'
        names.append(name)
        save(logs / name, {'gpu': gpu, 'task': task, 'worker_id': worker, 'workers': count,
              'pid': 100 + gpu, 'start_tick': 500 + gpu, 'command': m.command(tmp_path, task, worker, count)})
        workers.append({'launch_receipt': str(logs / name), 'launch_sha256': m.sha(logs / name)})
        folder = tmp_path / 'results-v2/engineering/reach-wall' / f'uuid-{gpu}'
        save(folder / 'STARTED.json', {'task': task, 'scenario': protocol['tasks'][task]['records'][0],
             'device_uuid': f'uuid-{gpu}', 'freeze_sha256': m.FREEZE, 'engineering': True})
        save(folder / 'native.json', {'preserved_excluded_native': True})
    save(logs / aname, {'instance': instance, 'freeze_sha256': m.FREEZE, 'mapping': mapping,
          'expected_gpu_count': 4, 'launch_receipts': names, 'device_uuids': {str(i): f'uuid-{i}' for i in range(4)}})
    stamp = m.datetime.fromisoformat('2026-09-13T01:00:00+00:00').timestamp()
    budget = {'as_of_utc': '2026-09-13T01:00:00Z', 'accrued_usd': 60,
              'fleet_cap_usd_hour': 35, 'campaign_cap_usd': 220, 'closeout_reserve_usd': 5}
    save(logs / bname, {'instance': instance, 'assignment_sha256': m.sha(logs / aname),
         'workers': workers, 'budget': budget, 'deadline_unix': m.guard.deadline(stamp, 60, 35)})
    return tmp_path, instance, aname, bname, {i: f'uuid-{i}' for i in range(4)}


def test_four_gpu_replacement_resumes_exact_assignment_and_preserves_natives(four_receiver):
    root = four_receiver[0]
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    p, prior = plan(four_receiver)
    assert p['expected_gpu_count'] == len(p['jobs']) == len(p['launch_receipts']) == 4
    assert p['mapping'] == [[g, 'reach-wall', w, 32] for g, w in enumerate((7, 15, 23, 31))]
    assert p['deadline_unix'] == prior['deadline_unix'] <= m.LATEST
    assert all(j['command'][3] == 'worker' for j in p['jobs'])
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_four_gpu_replacement_cannot_use_other_task_or_count(four_receiver):
    root, _, aname, bname, _ = four_receiver
    path = root / 'worker-logs-v2' / aname
    original = m.read(path)
    for change in ('task', 'count'):
        edited = json.loads(json.dumps(original))
        if change == 'task':
            edited['mapping'][0][1] = 'reach'
        else:
            edited['expected_gpu_count'] = 8
        save(path, edited)
        prior_path = root / 'worker-logs-v2' / bname
        prior = m.read(prior_path)
        prior['assignment_sha256'] = m.sha(path)
        save(prior_path, prior)
        with pytest.raises(ValueError, match='exact residual|Expected unique'):
            plan(four_receiver)


def test_four_gpu_replacement_still_rejects_live_jobs_and_wrong_uuid(four_receiver):
    with pytest.raises(ValueError, match='Old PID live or reused'):
        plan(four_receiver, inspect=lambda pid: {'command': ['old'], 'start_tick': 1})
    four_receiver[-1][0] = 'wrong-uuid'
    with pytest.raises(ValueError, match='hardware UUID'):
        plan(four_receiver)
