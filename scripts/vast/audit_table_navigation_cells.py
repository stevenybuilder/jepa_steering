"""Descriptive table-cell audit; does not run or replace the frozen32-contrast analysis."""
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from table_resume_streams import inventory,read,sha,FREEZES
from offline_study.navigation_coupling_behavior import load_native,paired_inputs
from offline_study.navigation_replication import validate_records
from offline_study.behavioral_development import schedule

ROOT=PROJECT/'artifacts/offline_study/table-completion-20260911-v1'
EVIDENCE=ROOT/'restored/50231985/batch-001/jepa-runtime/navigation-coupling-evidence-20260908-v2/artifacts/offline_study'


def main():
    summaries={};bindings={}
    for task in ('wall','pointmaze'):
        freeze=EVIDENCE/f'navigation-coupling-behavior-20260908-v1/{task}/freeze/protocol.json'
        protocol=read(freeze);assert sha(freeze)==FREEZES[task]
        reference=EVIDENCE/f'navigation-coupling-reference-20260908-v1/{task}'
        _,native,native_hash=load_native(reference/'native',reference/'freeze')
        assert native_hash==protocol['bindings']['native_report_sha256']
        bindings[str(freeze.relative_to(PROJECT))]=sha(freeze)
        native=sorted(native,key=lambda x:x['episode'])
        assert len(native)==96
        found,_=inventory(ROOT,task);panels={}
        for (arm,rank),value in found.items():
            p=Path(value['root']);report=read(p/'report.json');launch=read(p/'protocol.json')
            assert report['source_sha256']==protocol['source_sha256']
            assert report['freeze_sha256']==sha(freeze) and launch['freeze_sha256']==sha(freeze)
            records=sorted([read(p/n) for n in report['episode_files_sha256']],key=lambda x:x['episode'])
            validate_records(records,launch['expected_episodes'])
            for row in records:
                assert row['arm']==arm
                paired_inputs(native[row['episode']],row)
            panels.setdefault(arm,[]).extend(records)
            bindings[str((p/'report.json').relative_to(PROJECT))]=sha(p/'report.json')
        complete={a:sorted(rows,key=lambda x:x['episode']) for a,rows in panels.items() if len(rows)==96}
        complete['native']=native
        assert len(complete)==7
        summaries[task]={}
        for arm,rows in complete.items():
            validate_records(rows,schedule())
            values=[r['result']['native_success'] for r in rows]
            assert all(v in (0,1) for v in values)
            summaries[task][arm]={'episodes':96,'successes':int(sum(values)),
                'success_percent':100*sum(values)/96,'paired_inputs_verified':True}
    out=ROOT/'NAVIGATION_COMPLETE_CELLS.json'
    value={'status':'verified_completed_cells_descriptive_only','task_summaries':summaries,
           'bindings':bindings,'full_registered_analysis_complete':False,
           'new_table_cells':10,'remaining_displayed_table_cells':19,'fresh_confirmation':False}
    if out.exists():assert read(out)==value
    else:
        with out.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
