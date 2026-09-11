"""Read-only provenance audit; writes a new receipt, never edits historical data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    reserve = project / 'artifacts/offline_study/primary-durable-20260907/planning-scenarios-20260907'
    cohorts = {t: json.loads((reserve / (t+'-v1') / 'scenarios.json').read_text())
               for t in ('reach', 'reach-wall')}
    # Search all retained JSON, including ignored evidence, but not raw numerical
    # arrays. Match actual outcome records, never seed schedules inside freezes.
    candidates = subprocess.check_output(['rg', '-l', '--hidden', '--no-ignore',
        '--glob', '*.json', '"native_success"|"initial_state_vector"',
        'artifacts', 'archive/2026-09-07-workspace/artifacts'], cwd=project, text=True).splitlines()
    checked, matches, records = {}, [], 0
    for name in candidates:
        path = project / name
        raw = path.read_bytes()
        value = json.loads(raw)
        checked[name] = hashlib.sha256(raw).hexdigest()
        for row in walk(value):
            result = row.get('result', {})
            if not isinstance(result, dict) or not isinstance(result.get('native_success'), bool):
                continue
            if 'environment_seed' not in row:
                continue
            records += 1
            for task, scenarios in cohorts.items():
                for scenario in scenarios:
                    # Pixel and physical-state identities are stronger than an
                    # integer seed shared by different simulators/tasks.
                    pixel = result.get('initial_sha256') == scenario['initial_sha256']
                    vector = row.get('initial_state_vector') == scenario['rand_vec']
                    if pixel or vector:
                        matches.append({'file': name, 'task': task, 'episode': scenario['episode'],
                                        'initial_image_match': pixel, 'initial_vector_match': vector})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump({'role': 'retained_artifact_exposure_audit', 'checked_files_sha256': checked,
            'actual_outcome_records_examined': records, 'reserved_outcome_matches': matches,
            'reserved_reports_sha256': {t: hashlib.sha256((reserve/(t+'-v1')/'report.json').read_bytes()).hexdigest()
                                        for t in cohorts},
            'limitations': ['No claim about unrecorded work outside the project.',
                'Two inaccessible source disks were not read; retained preservation evidence is the audit source.'],
            'no_model_executed': True, 'passed': not matches and records >= 960}, out, indent=2)
    print(json.dumps({'files': len(checked), 'outcome_records': records, 'matches': matches,
                      'passed': not matches and records >= 960}))
    if matches or records < 960:
        raise SystemExit('Exposure audit failed; do not open confirmation')


if __name__ == '__main__':
    main()
