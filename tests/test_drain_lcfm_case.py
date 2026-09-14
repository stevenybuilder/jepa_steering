import hashlib
import json
from pathlib import Path

import pytest

from scripts.vast.drain_lcfm_case import assert_same_process, snapshot, validate_process


def identity():
    return {'args': ['python', '-m', 'offline_study.lcfm_replication', 'execute',
        '--task', 'reach', '--manifest-sha256', 'frozen', '--output', '/worker/cases']}


def test_only_exact_owned_science_executor_can_be_drained():
    validate_process(identity(), 'reach', 'frozen', Path('/worker/cases'))
    for task, manifest, output in [('reach-wall', 'frozen', '/worker/cases'),
        ('reach', 'other', '/worker/cases'), ('reach', 'frozen', '/other/cases')]:
        with pytest.raises(ValueError, match='ownership'):
            validate_process(identity(), task, manifest, Path(output))
    wrong = identity(); wrong['args'][2] = 'unrelated_module'
    with pytest.raises(ValueError, match='not the registered'):
        validate_process(wrong, 'reach', 'frozen', Path('/worker/cases'))
    wrong = identity(); wrong['args'].append('--engineering')
    with pytest.raises(ValueError, match='scientific work'):
        validate_process(wrong, 'reach', 'frozen', Path('/worker/cases'))


def record(root, episode, complete):
    directory = root/'reach'/f'episode-{episode:03d}'
    directory.mkdir(parents=True)
    (directory/'STARTED.json').write_text(json.dumps({'execution_manifest_sha256': 'frozen',
        'input_binding': {'task': 'reach', 'episode': episode, 'role': 'scientific_replication'}}))
    if complete:
        (directory/'DONE.json').write_text('{}')
        (directory/'REPLICATION_CASE.json').write_text(json.dumps({'manifest_sha256': 'frozen',
            'complete': True, 'forward_count': 70,
            'kernel_done_sha256': hashlib.sha256(b'{}').hexdigest()}))
    return directory


def test_next_started_case_is_retained_as_incomplete(tmp_path):
    record(tmp_path, 0, True); pending = record(tmp_path, 1, False)
    assert snapshot(tmp_path, 'reach', 'frozen') == {'complete': [0], 'incomplete': [1]}
    assert (pending/'STARTED.json').exists()
    assert not (pending/'REPLICATION_CASE.json').exists()


def test_changed_completion_receipt_cannot_trigger_a_boundary(tmp_path):
    directory = record(tmp_path, 0, True)
    (directory/'DONE.json').write_text('{"corrupted":true}')
    with pytest.raises(ValueError, match='receipt invalid'):
        snapshot(tmp_path, 'reach', 'frozen')


def test_partially_written_json_is_retried_without_counting_completion(tmp_path):
    directory = record(tmp_path, 0, False)
    (directory/'REPLICATION_CASE.json').write_text('{')
    assert snapshot(tmp_path, 'reach', 'frozen') == {'complete': [], 'incomplete': []}


def test_empty_command_during_linux_exit_is_not_pid_reuse():
    original = {'start_ticks': '100', 'args': ['python', '-m', 'owned']}
    assert_same_process({'start_ticks': '100', 'args': []}, original)
    with pytest.raises(ValueError, match='PID identity'):
        assert_same_process({'start_ticks': '200', 'args': []}, original)
    with pytest.raises(ValueError, match='command changed'):
        assert_same_process({'start_ticks': '100', 'args': ['unrelated']}, original)
