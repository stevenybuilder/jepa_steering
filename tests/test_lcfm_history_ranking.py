import copy
import json
import numpy as np
import pytest
from scipy.stats import spearmanr
from analysis.mechanism.lcfm_history_ranking import agreement,describe,load_inputs,DATA
import pandas as pd


def score(x,ids=None):
    return {'costs':np.asarray(x).tolist(),'elite_indices':np.argsort(x,kind='stable')[:10].tolist() if ids is None else ids}


def test_rank_reversal_and_disjoint_decisions():
    a=np.arange(300.)
    r=agreement(score(a),score(-a))
    assert r['spearman']==pytest.approx(-1)
    assert r['elite_overlap']==0 and r['winner_mismatch']==1 and r['minimum_sets_disjoint']==1


def test_ties_match_scipy_and_do_not_imply_forced_winner_change():
    a=np.arange(300.);a[:2]=0
    b=a.copy();b[0]=.1
    r=agreement(score(a),score(b))
    assert r['spearman']==pytest.approx(spearmanr(a,b).statistic)
    assert r['winner_mismatch']==1 and r['minimum_sets_disjoint']==0
    assert r['left_minimum_ties']==2


def test_actual_archived_elites_used_when_cutoff_tied():
    a=np.arange(300.);a[8:12]=8
    r=agreement(score(a,list(range(10))),score(a,list(range(8))+[10,11]))
    assert r['elite_overlap']==8 and r['left_cutoff_ambiguous'] and r['spearman']==pytest.approx(1)


def test_constant_cost_rank_is_undefined():
    r=agreement(score(np.ones(300)),score(np.arange(300)))
    assert r['spearman'] is None
    assert r['minimum_sets_disjoint']==0


def test_invalid_costs_and_elites_rejected():
    a=score(np.arange(300.));b=copy.deepcopy(a);b['costs'][15]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):agreement(a,b)
    b=copy.deepcopy(a);b['elite_indices'][0]=250
    with pytest.raises(ValueError,match='top10'):agreement(a,b)


def test_no_reduced_cohort_means():
    d=pd.DataFrame({'task':['reach']*8,'bank':['original']*8,'episode':range(8),'value':[1.]*7+[np.nan]})
    r=describe(d,['task','bank'],['value']).iloc[0]
    assert r.n_defined==7 and pd.isna(r['mean'])
    with pytest.raises(ValueError,match='eight'):describe(d.iloc[:7],['task','bank'],['value'])


def test_modified_public_input_rejected(tmp_path):
    for name in ('lcfm_history_ranking_inputs.json','lcfm_history_ranking_inputs_manifest.json'):
        (tmp_path/name).write_bytes((DATA/name).read_bytes())
    with (tmp_path/'lcfm_history_ranking_inputs.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='hash'):load_inputs(tmp_path)


def test_complete_public_reproduction(tmp_path):
    from analysis.mechanism.lcfm_history_ranking import run,read,sha
    run(DATA,tmp_path)
    expected=read(DATA/'lcfm_history_ranking_report.json')
    for name,metadata in expected['outputs'].items():
        assert sha(DATA/name)==metadata['sha256']
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path/name),pd.read_csv(DATA/name),check_exact=False,rtol=1e-12,atol=1e-12)
    case=pd.read_csv(tmp_path/'lcfm_history_ranking_cases.csv')
    selected=case[(case.site=='all_blocks')&(case.comparison=='h3_vs_counterfactual')]
    assert len(selected)==32 and selected.winner_mismatch.sum()==17
    assert len(case)==704 and not case.spearman.isna().any()
