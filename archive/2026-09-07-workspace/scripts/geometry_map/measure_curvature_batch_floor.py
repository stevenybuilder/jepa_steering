#!/usr/bin/env python3
"""Measure frozen batch7 versus fresh batch8 central activation/output drift."""
import argparse
import json
from pathlib import Path
import torch
from complete_cached_geometry import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--curvature',type=Path,required=True);p.add_argument('--interior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dc=json.loads((a.curvature/'DONE.json').read_text());di=json.loads((a.interior/'DONE.json').read_text())
    if not dc['complete'] or not di['complete']:raise ValueError('Both stages must be complete')
    oldrows={r['path']:r for r in dc['outputs']};newrows={r['path']:r for r in di['outputs']};rows=[]
    torch.set_num_threads(1)
    for episode in range(4):
        key=f'near-dev-{episode:03}';oldname=key+'-pair0-radius1.pt';newname=key+'-pair0-radius1-minus025.pt'
        paths=[a.curvature/oldname,a.interior/newname]
        for path,row in zip(paths,[oldrows[oldname],newrows[newname]]):
            if sha256(path)!=row['sha256']:raise ValueError('Source SHA mismatch before loading')
        old,new=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        report=json.loads((a.interior/newname).with_suffix('.json').read_text())
        for j,h in enumerate((1,3,6)):
            residual_delta=new['replacements'][j,0].double()-old['residual_samples'][3+j,2].double()
            old_pred=old['native_central_predicted_visual'][:,0].double();new_pred=new['native_central_visual'].double()
            output_delta=new_pred-old_pred
            cubic=next(m for m in report['rows'][j]['methods'] if m['method']=='cubic_equal_data')
            floor=float(residual_delta.norm());scale=cubic['midpoint_l2_error']
            rows.append(dict(key=key,horizon=h,residual_maxabs=float(residual_delta.abs().max()),residual_l2=floor,
                downstream_native_maxabs=float(output_delta[h-1:].abs().max()),downstream_native_mse=float(output_delta[h-1:].square().mean()),
                reference_interior_cubic_error_l2=scale,floor_over_cubic_error=floor/max(scale,1e-20),
                appreciable_by_fixed_relative_threshold=floor/max(scale,1e-20)>=.001))
    write_json(a.output,dict(complete=True,rows=rows,any_appreciable=any(r['appreciable_by_fixed_relative_threshold'] for r in rows),
        criterion='Measure12fixedstate/H floors; propagated midpoint-patch reference needed if raw norm floor≥.001 of fixed first interior cubic error',
        scope='Natural central batch7 versus batch8 drift; selfpatch exact0 alone does not bound crossbatch floor'))
    print(json.dumps(dict(complete=True,maximum_residual_floor=max(r['residual_l2'] for r in rows),
        maximum_floor_over_error=max(r['floor_over_cubic_error'] for r in rows),any_appreciable=any(r['appreciable_by_fixed_relative_threshold'] for r in rows))))


if __name__=='__main__':main()
