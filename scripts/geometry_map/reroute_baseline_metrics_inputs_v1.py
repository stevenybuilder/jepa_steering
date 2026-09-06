"""Only resolve initially absent sealed source copies; outcomes do not select rows."""
import json
from pathlib import Path
import re

root=Path('artifacts/geometry_map/unsteered_baseline_metrics_v1')
d=json.loads((root/'manifest.json').read_text());rows=[]
for source in d['source']:
    if not (source['panel']=='pusht_official' and source['episode']>=10 or source['panel']=='reach_wall' and 12<=source['episode']<24):continue
    r=dict(source)
    if r['panel']=='pusht_official':
        name=Path(r['local_path']).parent.name;host=int(re.search(r'worker-(\d+)',name)[1]);r['host']=host
        suffix='recovery-v2' if name.endswith('recovery-v2') else 'baseline-v1'
        r['remote_path']=f'/root/geometry-map-jepawm-pusht-v1/worker-{host}/{suffix}/episode-{r["episode"]:03d}.pt'
    elif r['episode']<16:
        r['host']=49987405;r['offline_original_source']=49982195
        r['remote_path']=f'/root/geometry-map-jepawm-pusht-v1/unsteered-baseline-metrics-v1/offline195-inputs/episode-{r["episode"]:03d}.pt'
    else:
        r['host']=49155754 if r['episode']<20 else 49766237
        r['remote_path']=f'/root/geometry-map-jepawm-reach-wall-v1/staging-{r["host"]}/on-policy-baseline/episode-{r["episode"]:03d}.pt'
    r['prior_missing_preparation_mirror']=source['remote_path'];rows.append(r)
assert len(rows)==23
for host in sorted({r['host'] for r in rows}):
    with (root/f'host-{host}-source-v2.json').open('x') as f:json.dump(dict(d,source=[r for r in rows if r['host']==host],path_resolution_only=True),f,indent=2)
print(json.dumps({'retry_rows':len(rows),'hosts':sorted({r['host'] for r in rows})}))
