import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.mechanism.controlled_geometry_summary import (
    CONDITIONS, TASKS, WEIGHTS, audit_row, discover, ratio_record, summarize, estimate, plots,
)


def fixture_row():
    # Synthetic field fixture is temporary/in-memory, never public research data.
    values=np.array([[.8,2.1],[.95,2.05],[1.,2.],[1.08,1.99],[1.2,2.02]],dtype=float)
    anchors=values[[0,1,3,4]];center=values[2];delta=anchors-center
    gram=delta@delta.T/2
    row=dict(field_numel=2,field_shape=[2],center_rms=float(np.sqrt(np.mean(center**2))),
        centered_anchor_gram_mean_inner_product=gram.tolist(),donor_response_rms=np.sqrt(gram.diagonal()).tolist(),
        exact_unchanged_fraction=(anchors==center).mean(1).tolist(),finite_differences=[])
    for name,w in WEIGHTS.items():
        direct=float(np.mean((w@anchors-center)**2));saved=float(w@gram@w)
        eps=np.finfo(float).eps;roundoff=32*eps*abs(values).max()
        row.update({name+'_center_mse':direct,name+'_gram_mse':saved,
            name+'_gram_absolute_error':abs(direct-saved),name+'_gram_tolerance':
            2*np.sqrt(max(direct,abs(saved)))*roundoff+roundoff**2+64*eps*abs(gram).max()})
    for radius,a,b in ((.05,1,3),(.1,0,4)):
        first=(values[b]-values[a])/2;second=values[b]-2*center+values[a]
        row['finite_differences'].append(dict(radius=radius,first_difference_mean=float(first.mean()),
            first_difference_rms=float(np.sqrt(np.mean(first**2))),second_difference_mean=float(second.mean()),
            second_difference_rms=float(np.sqrt(np.mean(second**2))),
            central_first_derivative_rms=float(np.sqrt(np.mean((first/radius)**2))),
            central_second_derivative_rms=float(np.sqrt(np.mean((second/radius**2)**2)))))
    return row


def test_direct_gram_and_finite_difference_audit():
    row=fixture_row();result=audit_row(row)
    assert result['ratio_status']=='both_positive'
    assert result['log_ratio']==pytest.approx(np.log(row['cubic_center_mse']/row['linear_center_mse']))


@pytest.mark.parametrize('field', ['linear_center_mse','linear_gram_mse','linear_gram_absolute_error'])
def test_corrupted_reconstruction_fails(field):
    row=fixture_row();row[field]+=1
    with pytest.raises(ValueError):audit_row(row)


def test_gram_corruption_and_tolerance_fails():
    row=fixture_row();row['centered_anchor_gram_mean_inner_product'][0][1]+=1
    with pytest.raises(ValueError,match='symmetric'):audit_row(row)
    row=fixture_row();row['cubic_gram_tolerance']=10
    with pytest.raises(ValueError,match='tolerance'):audit_row(row)


def test_derivative_corruption_fails():
    row=fixture_row();row['finite_differences'][0]['central_second_derivative_rms']+=10
    with pytest.raises(ValueError,match='difference'):audit_row(row)


@pytest.mark.parametrize('linear,cubic,status,ratio', [(0,0,'both_zero',None),(0,1,'linear_zero',None),(1,0,'cubic_zero',0)])
def test_zeros_explicit(linear,cubic,status,ratio):
    result=ratio_record(linear,cubic)
    assert result==dict(ratio_status=status,ratio=ratio,log_ratio=None)


def test_negative_mse_fails():
    with pytest.raises(ValueError):ratio_record(-1,2)


def test_partial_cohort_rejected_before_reading_metrics(tmp_path):
    directory=tmp_path/'reach'/'episode-0';directory.mkdir(parents=True)
    (directory/'report.json').write_text('THIS IS NOT JSON')
    with pytest.raises(ValueError,match='Complete64'):discover(tmp_path)


def synthetic_frame():
    rows=[]
    for task in TASKS:
        for condition_id,condition in enumerate(CONDITIONS):
            for layer in range(6):
                for episode in range(32):
                    rows.append(dict(task=task,condition=condition,layer=layer,episode=episode,
                        log_ratio=episode/32+condition_id*.3+layer*.1,linear_center_mse=1.,cubic_center_mse=2.,
                        donor_response_rms_mean=.1,exact_unchanged_fraction_mean=.2,ratio_status='both_positive'))
    return pd.DataFrame(rows)


def test_paired_resampling_preserves_constant_condition_difference():
    frame=synthetic_frame();summary,contrasts=summarize(frame,replicates=300)
    assert len(summary)==180 and len(contrasts)==36
    assert set(summary.n)=={32}
    for row in contrasts.itertuples():
        expected=(CONDITIONS.index(row.left_condition)-CONDITIONS.index(row.right_condition))*.3
        assert row.mean==pytest.approx(expected)
        assert row.marginal_95_low==pytest.approx(expected)
        assert row.marginal_95_high==pytest.approx(expected)
        assert row.simultaneous_six_layer_95_low==pytest.approx(expected)


def test_undefined_context_not_silently_excluded():
    frame=synthetic_frame()
    mask=(frame.task=='reach')&(frame.condition=='bfloat16')&(frame.layer==0)&(frame.episode==0)
    frame.loc[mask,'log_ratio']=np.nan;frame.loc[mask,'ratio_status']='both_zero'
    summary,contrasts=summarize(frame,replicates=100)
    row=summary[(summary.task=='reach')&(summary.condition=='bfloat16')&(summary.layer==0)&(summary.metric=='log_ratio')].iloc[0]
    assert row.n_defined==31 and np.isnan(row['mean']) and row.both_zero_count==1
    family=contrasts[(contrasts.task=='reach')&(contrasts.left_condition=='bfloat16')]
    assert family.simultaneous_six_layer_95_low.isna().all()


def test_plot_public_hash_corruption_rejected(tmp_path):
    path=tmp_path/'controlled_geometry_summary.json'
    csv=tmp_path/'controlled_geometry_summary.csv';csv.write_text('fixture\n')
    path.write_text(json.dumps(dict(status='complete64_controlled_geometry_development',cohort=64,
        outputs={csv.name:dict(sha256='0'*64,rows=0)})))
    with pytest.raises(ValueError,match='hash'):plots(path,tmp_path/'figures')
