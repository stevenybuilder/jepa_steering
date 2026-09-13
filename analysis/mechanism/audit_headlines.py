"""Independent streaming recount of mechanism headline quantities from raw arms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    output = {}
    for task in ('reach', 'reach-wall', 'pointmaze', 'wall'):
        rows = {}
        for scenario in sorted((args.results / task).glob('scenario-*')):
            reports = {}
            for path in sorted(scenario.glob('*.json')):
                if path.stem in ('STARTED', 'DONE', 'report'):
                    continue
                data = json.loads(path.read_text())
                if data.get('scientific_efficacy_measurement') is True:
                    reports[path.stem] = data
            assert len(reports) == 8 and (scenario / 'DONE.json').exists()
            native = reports['native']
            assert len({d['device_uuid'] for d in reports.values()}) == 1
            for arm, record in reports.items():
                r = rows.setdefault(arm, dict(n=0, success=0, rescue=0, regress=0,
                    first_action_changed=0, energy_calls=0, realized_mean_sum=0.,
                    requested_mean_sum=0., coefficient_count=0, gram=np.zeros((4, 4))))
                n, s = bool(native['result']['native_success']), bool(record['result']['native_success'])
                r['n'] += 1; r['success'] += s
                r['rescue'] += s and not n; r['regress'] += n and not s
                r['first_action_changed'] += (record['action_trace'][0]['actions_sha256'] !=
                                               native['action_trace'][0]['actions_sha256'])
                for call in record['calls']:
                    if call['horizon'] != 6 or call['candidates'] <= 1:
                        continue
                    e = call.get('energy', {})
                    if 'coefficients' in e:
                        c = np.asarray(e['coefficients'], dtype=np.float64)
                        r['gram'] += c.T @ c
                        r['coefficient_count'] += len(c)
                    if 'realized_squared_l2_mean' in e:
                        r['energy_calls'] += 1
                        r['realized_mean_sum'] += e['realized_squared_l2_mean']
                        r['requested_mean_sum'] += e['requested_squared_l2_mean']
        for r in rows.values():
            assert r['n'] == 96
            gram = r.pop('gram')
            if r['coefficient_count']:
                r['uncentered_coefficient_PR'] = float(np.trace(gram)**2 / np.sum(gram*gram))
        output[task] = rows
        print(task, {a: {k:r[k] for k in ('success','rescue','regress','first_action_changed')}
                     for a,r in rows.items() if a in ('native','fixed_rank4','coupling_only')}, flush=True)
    output['_audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
