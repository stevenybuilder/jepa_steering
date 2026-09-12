"""Verify completed whole streams using the worker's unchanged scientific code."""
import argparse
from pathlib import Path

from offline_study.behavioral_development import assigned_rows, validate_coverage, verified_report
from offline_study.fixed_response_smoke import verify_episode
from offline_study.metaworld_component_behavior import METHOD, read, validate_protocol, verify_engineering
from offline_study.protocol import sha256


def validate(root, freeze, engineering, task, uuid, arm=None, rank=None):
    protocol = validate_protocol(freeze)
    identity = uuid.removeprefix('GPU-')
    engineering_hash = verify_engineering(engineering, freeze, task, identity)
    if arm is None:
        return engineering_hash
    report, digest = verified_report(root)
    launch = read(root / 'protocol.json')
    expected = assigned_rows(protocol['episodes'], [rank])
    if ((root / 'FAILED.json').exists() or report['status'] != METHOD + '_shard_complete'
            or report['task'] != task or report['arm'] != arm or report['episodes'] != 12
            or report['protocol_sha256'] != sha256(root / 'protocol.json')
            or launch['source_sha256'] != protocol['source_sha256']
            or launch['freeze_sha256'] != sha256(freeze / 'protocol.json')
            or launch['task'] != task or launch['arm'] != arm or launch['logical_ranks'] != [rank]
            or launch['device_uuid'] != identity or launch['engineering_report_sha256'] != engineering_hash
            or launch['expected_episodes'] != expected or report['parameters_unchanged'] is not True
            or report['scientific_efficacy_measurement'] is not True or report['fresh_confirmation'] is not False):
        raise ValueError('Completed stream cannot be reused: identity/completion differs')
    records = []
    for filename, checksum in report['episode_files_sha256'].items():
        if Path(filename).name != filename or not filename.startswith('episode-') or sha256(root / filename) != checksum:
            raise ValueError('Unsafe or altered episode')
        row = read(root / filename)
        traces = root / f"calls-{row['episode']:03d}"
        for name, key in (('unroll_calls.json', 'unroll_calls_sha256'), ('action_trace.json', 'action_trace_sha256')):
            if sha256(traces / name) != row[key]:
                raise ValueError('Raw evidence checksum differs')
        calls = read(traces / 'unroll_calls.json')
        verify_episode(row['result'], calls)
        if row['arm'] != arm or any(c['backend_calls'] != 1 for c in calls):
            raise ValueError('Wrong condition or forecast work')
        records.append(row)
    validate_coverage(sorted(records, key=lambda r: r['episode']), expected)
    return digest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'freeze', 'engineering'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('task', 'uuid'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--arm')
    p.add_argument('--rank', type=int)
    a = p.parse_args()
    print(validate(a.root, a.freeze, a.engineering, a.task, a.uuid, a.arm, a.rank))
