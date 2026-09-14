import copy
import numpy as np
import pandas as pd
import pytest

from analysis.mechanism.action_counterfactual_summary import (
    BANKS,TASKS,PRIMARY,aggregate,discover,estimate,expected_addresses,norm_audit,reconstruction_audit,score_audit,
    validate_runtime,CHECKPOINT_SHA,
)


def test_runtime_frozen_receiver_and_precision():
    manifest={'receivers':{'0':{'gpu_uuid':'expected'}}}
    runtime=dict(gpu_uuid='GPU-expected',tf32_matmul=False,tf32_cudnn=False,
        backend_provenance=dict(checkpoint_sha256=CHECKPOINT_SHA,precision='float32',allow_tf32=False,autocast=False))
    validate_runtime(runtime,manifest)
    wrong=copy.deepcopy(runtime);wrong['gpu_uuid']='other'
    with pytest.raises(ValueError,match='receiver'):validate_runtime(wrong,manifest)
    wrong=copy.deepcopy(runtime);wrong['backend_provenance']['autocast']=True
    with pytest.raises(ValueError,match='FP32'):validate_runtime(wrong,manifest)


def test_ratio_of_means_not_mean_candidate_ratios():
    denominator=np.arange(1,301,dtype=float);numerator=np.ones(300)*2
    row=dict(numerator_mse=numerator.tolist(),denominator_mse=denominator.tolist(),
        reconstruction=(1-numerator/denominator).tolist(),n_defined=300,
        pooled_reconstruction=1-numerator.mean()/denominator.mean())
    result=reconstruction_audit(row)
    assert result['donor_reconstruction']==pytest.approx(1-2/150.5)
    row['pooled_reconstruction']=np.mean(row['reconstruction'])
    with pytest.raises(ValueError,match='ratio of means'):reconstruction_audit(row)


def test_negative_and_undefined_reconstruction_retained():
    row=dict(numerator_mse=[3.]*300,denominator_mse=[1.]*300,reconstruction=[-2.]*300,n_defined=300,pooled_reconstruction=-2.)
    assert reconstruction_audit(row)['donor_reconstruction']==-2
    row.update(denominator_mse=[0.]*300,reconstruction=[None]*300,n_defined=0,pooled_reconstruction=None)
    assert reconstruction_audit(row)['donor_reconstruction'] is None
    row['pooled_reconstruction']=0
    with pytest.raises(ValueError,match='Undefined'):reconstruction_audit(row)


def fixture_norm():
    return dict(requested_delta_l2=[1.]*300,delivered_delta_l2=[1.]*300,relative_norm_error=[0.]*300,
        max_relative_norm_error=0.,zero_target_count=0,in_range_energy_fraction=[.2]*300,off_range_energy_fraction=[.8]*300)


def test_norm_and_projection_identity():
    row=fixture_norm();assert norm_audit(row)['delivered_l2_mean']==1
    row['delivered_delta_l2'][0]=1.01
    with pytest.raises(ValueError,match='norm'):norm_audit(row)
    row=fixture_norm();row['off_range_energy_fraction'][0]=.7
    with pytest.raises(ValueError,match='energy'):norm_audit(row)


def test_zero_norm_requires_undefined_fraction():
    row=fixture_norm();row['requested_delta_l2'][0]=row['delivered_delta_l2'][0]=0;row['zero_target_count']=1
    with pytest.raises(ValueError,match='Undefined'):norm_audit(row)
    row['in_range_energy_fraction'][0]=row['off_range_energy_fraction'][0]=None
    assert norm_audit(row)['in_range_energy_fraction'] is None


def test_exact_history_addresses():
    assert expected_addresses('donor_h3_B2')=={(3,2,1)}
    assert expected_addresses('donor_persistent_B2')=={(3,2,1),(4,2,0)}
    assert len(expected_addresses('all_blocks_persistent_donor'))==12
    assert expected_addresses('coherent_raw_h3')==set()


def test_tie_aware_ranks_and_actual_elites():
    values=np.repeat(np.arange(150),2).astype(float)
    payload=dict(costs=values.tolist(),elite_indices=list(range(10)))
    result=score_audit(payload,payload)
    assert result['native_spearman']==pytest.approx(1) and result['native_top10_overlap']==10
    corrupt=copy.deepcopy(payload);corrupt['elite_indices'][0]=100
    with pytest.raises(ValueError,match='top10'):score_audit(payload,corrupt)


def test_partial_rejected_before_result_access(tmp_path):
    directory=tmp_path/'reach'/'episode-0';directory.mkdir(parents=True)
    (directory/'report.json').write_text('not json')
    with pytest.raises(ValueError,match='Complete16'):discover(tmp_path)


def synthetic_frame():
    rows=[];arms={arm for _,left,right,_ in PRIMARY for arm in (left,right)}
    for task in TASKS:
        for bank in BANKS:
            for layer in range(6):
                for arm in sorted(arms):
                    offset=sorted(arms).index(arm)/10
                    for episode in range(8):
                        rows.append(dict(task=task,bank=bank,layer=layer,arm=f'{arm}_B{layer}',episode=episode,modality='official',horizon=6,
                            donor_reconstruction=episode/8+offset,numerator_mse_mean=1,denominator_mse_mean=2,
                            native_spearman=.9-offset,spearman_loss=.1+offset,native_top10_overlap=9,centered_cost_change_rms=.1))
    return pd.DataFrame(rows)


def test_all72_primary_cells_joint_paired_band():
    summary,primary=aggregate(synthetic_frame(),draws=200)
    assert len(primary)==72 and set(primary.n)=={8}
    assert np.allclose(primary['mean'],primary.simultaneous_72_primary_95_low)
    assert np.allclose(primary['mean'],primary.simultaneous_72_primary_95_high)
    assert np.allclose(primary.family_quantile,1-.05/6)


def test_undefined_primary_not_dropped_or_replaced():
    frame=synthetic_frame();mask=(frame.task=='reach')&(frame.bank=='original')&(frame.arm=='donor_h3_B0')&(frame.episode==0)
    frame.loc[mask,'donor_reconstruction']=np.nan
    summary,primary=aggregate(frame,draws=100)
    family=primary[(primary.task=='reach')&(primary.contrast=='persistence_minus_h3')]
    assert family.simultaneous_72_primary_95_low.isna().all()
    assert (family.n_defined==7).sum()==1
