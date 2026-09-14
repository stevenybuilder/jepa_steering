import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('collector', Path(__file__).parents[3] / 'scripts/vast/collect_fresh_metadata.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_immutable_repeated_source_and_conflict(tmp_path):
    p = tmp_path / 'x.json'
    c.immutable(p, b'one')
    c.immutable(p, b'one')
    with pytest.raises(ValueError, match='Conflicting'):
        c.immutable(p, b'two')
    assert p.read_bytes() == b'one'


def test_owner_rejects_identical_case_from_different_host(tmp_path):
    p = tmp_path / 'owners/reach/scenario-000.json'
    c.immutable(p, c.encoded({'instance_id': 50806821, 'report_sha256': 'same'}))
    with pytest.raises(ValueError):
        c.immutable(p, c.encoded({'instance_id': 50827072, 'report_sha256': 'same'}))


@pytest.mark.parametrize('name', ['../x', '/root/x', 'results-v2/../x'])
def test_unsafe_archive_path(name):
    with pytest.raises(ValueError):
        c.selected(name)


def test_only_top_level_metadata():
    assert c.selected('results-v2/reach/scenario-000/native.json')
    assert c.selected('results-v2/engineering/reach/uuid/native.json')
    assert not c.selected('results-v2/reach/scenario-000/native/frame.json')
    assert not c.selected('worker-logs-v2/example.json')


def test_exact_panel_inventory(tmp_path):
    for task in c.TASKS:
        for episode in range(96):
            p = tmp_path / 'results-v2' / task / f'scenario-{episode:03d}' / 'DONE.json'
            p.parent.mkdir(parents=True)
            p.write_text('{}')
    assert c.inventory(tmp_path)['full_exact_panel']
    assert c.inventory(tmp_path)['scientific_arms'] == 3072
    (tmp_path / 'results-v2/reach/scenario-096').mkdir()
    with pytest.raises(ValueError, match='Unexpected'):
        c.inventory(tmp_path)


def test_bundle_checks_binding_without_efficacy_selection():
    row = {'episode': 0, 'input': 'frozen'}
    prefix = 'results-v2/reach/scenario-000'
    data = {prefix + '/' + a + '.json': b'{"arbitrary_opaque_record":true}' for a in c.ARMS}
    report = {'freeze_sha256': c.FREEZE, 'task': 'reach', 'scenario': row, 'engineering': False,
              'device_uuid': 'uuid', 'records_sha256': {a: c.sha(data[prefix + '/' + a + '.json']) for a in c.ARMS}}
    data[prefix + '/report.json'] = c.encoded(report)
    data[prefix + '/DONE.json'] = c.encoded({'report_sha256': c.sha(data[prefix + '/report.json'])})
    data[prefix + '/STARTED.json'] = c.encoded({'task': 'reach', 'scenario': row, 'engineering': False,
        'device_uuid': 'uuid', 'freeze_sha256': c.FREEZE})
    assert len(c.bundle(data, prefix, 'reach', row, False)[1]) == 11
    with pytest.raises(ValueError, match='UUID'):
        c.bundle(data, prefix, 'reach', row, False, 'other')
    data[prefix + '/native.json'] = b'changed'
    with pytest.raises(ValueError, match='Arm hash'):
        c.bundle(data, prefix, 'reach', row, False)
