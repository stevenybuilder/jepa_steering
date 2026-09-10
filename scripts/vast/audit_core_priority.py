"""Read-only receiving-worker audit; no outcome analysis or scientific mutation."""
import argparse
import json
from pathlib import Path
import sys
import time

CONTROL = Path('/workspace/jepa-runtime/routing-priority-20260908-v3')
sys.path.insert(0, str(CONTROL))
import routing_priority_common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    args, protocol = c.frozen_dependencies()
    from offline_study.behavioral_development import assigned_rows, schedule
    from offline_study.fixed_response_smoke import verify_episode
    reports, records, workers = {}, {}, []
    for gpu in range(8):
        task = 'reach' if gpu < 4 else 'reach-wall'
        rank = gpu % 4
        expected = {row['episode']: row for row in assigned_rows(schedule(), [rank, rank + 4])}
        native = {}
        for arm in c.ORIGINAL_ARMS:
            shard = args.reference / task / arm / f'shard-gpu{gpu}'
            if (shard / 'FAILED.json').exists():
                raise ValueError('Required core shard failed')
            if (shard / 'DONE.json').exists():
                reports[str(shard)] = c.verify_boundary(shard, gpu, args, protocol)
            elif arm != 'matched_random_coupling':
                raise ValueError('A previously complete core reference is missing')
            piece = []
            for path in sorted(shard.glob('episode-*.json')):
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # Concurrent writer has not published this record yet.
                episode = row['episode']
                if (episode not in expected or row['arm'] != arm or
                        row['local_seed'] != expected[episode]['local_seed']):
                    raise ValueError('Unregistered episode, arm or seed')
                calls = shard / f'calls-{episode:03d}'
                for name, key in (('unroll_calls.json', 'unroll_calls_sha256'),
                                  ('action_trace.json', 'action_trace_sha256')):
                    if c.digest(calls / name) != row[key]:
                        raise ValueError('Published raw trace mismatch')
                trace = json.loads((calls / 'unroll_calls.json').read_text())
                verify_episode(row['result'], trace)
                if any(call['backend_calls'] != 1 for call in trace):
                    raise ValueError('Extra model forecasts in core intervention')
                if arm == 'native':
                    native[episode] = row
                else:
                    reference = native[episode]
                    if (row['initial_state_vector'] != reference['initial_state_vector'] or
                            any(row['result'][k] != reference['result'][k] for k in ('initial_sha256', 'goal_sha256'))):
                        raise ValueError('Unpaired initial or goal stimuli')
                piece.append(episode)
                records[str(path)] = c.digest(path)
            if len(set(piece)) != len(piece):
                raise ValueError('Duplicate episode')
            workers.append({'gpu': gpu, 'task': task, 'arm': arm, 'verified_records': len(piece),
                            'whole_shard_complete': (shard / 'DONE.json').exists()})
    options.output.mkdir(parents=True, exist_ok=False)
    value = {'status': 'published_core_records_raw_trace_and_pairing_verified', 'time': time.time(),
             'workers': workers, 'complete_shard_reports_sha256': reports, 'episode_files_sha256': records,
             'published_episodes_verified': len(records), 'complete_shards_verified': len(reports),
             'outcome_selection_or_effect_estimation': False, 'new_gpu_jobs': 0,
             'full_panel_complete': len(records) == 960 and len(reports) == 40,
             'analysis_complete': False, 'source_sha256': c.digest(__file__)}
    c.write(options.output / 'REPORT.json', value)
    print(json.dumps({k: value[k] for k in ('status', 'published_episodes_verified',
           'complete_shards_verified', 'full_panel_complete', 'analysis_complete')}))


if __name__ == '__main__':
    main()
