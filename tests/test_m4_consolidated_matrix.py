"""Synthetic/non-training matrix checks; all real forecasting fits forbidden."""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from src.decision import consolidated_matrix as m
from test_m4_p4_policy import sources,forbid_training


def annotations(families):
    return pd.DataFrame({'family':families,'econometric_signature':['family context']*len(families),'statistically_sensitive_fdr':[False]*len(families)})


def wide_fixture():
    records=[]
    for family in ['A','B']:
        for city in ['X','Y']:
            for i,fold in enumerate(m.FOLDS):
                records.append(dict(date=pd.Timestamp('2016-09-01')+pd.Timedelta(days=i),store_nbr='1',family=family,city=city,fold=fold,
                    scenario='S1',tau_key='tau_0.50',y_true=0. if family=='A' else 3.,P1=10.,P2=6.,P3=5. if city=='X' else (7. if i%2 else 5.),
                    P4_M0=5.,P4_M3=4. if family=='A' else 6.))
    return pd.DataFrame(records)


def test_cell_grain_indices_fold_consistency_and_distinct_weather_comparisons():
    wide=wide_fixture();matrix,folds=m.aggregate_matrix(wide,annotations(['A','B']),['A','B'],['X','Y'],['S1'])
    assert len(matrix)==4 and len(folds)==16 and not matrix.duplicated(m.CELL).any()
    row=matrix.loc[matrix.family.eq('A')&matrix.city.eq('X')].iloc[0]
    assert row.cost_index_P2==.6 and row.cost_index_P4_M3==.4
    assert row.weather_class=='directionally_favorable' and row.zero_rate==1.
    row=matrix.loc[matrix.family.eq('B')&matrix.city.eq('X')].iloc[0]
    assert row.direction_P3_minus_P2=='directionally_favorable'
    assert row.direction_P4_M3_minus_M0=='directionally_unfavorable' and row.weather_class=='mixed_or_inconsistent'
    m.validate_matrix(matrix,['A','B'],['X','Y'],['S1'])


@pytest.mark.parametrize('values,expected',[([-1]*4,'directionally_favorable'),([1]*4,'directionally_unfavorable'),
    ([0]*4,'mixed_or_inconsistent'),([-1,1,-1,1],'mixed_or_inconsistent'),([-1,-1,-1,np.nan],'insufficient_evidence')])
def test_sign_only_classification(values,expected):
    assert m.classify_fold_deltas(pd.DataFrame({'fold':m.FOLDS,'sample_size':[1]*4,'delta':values}),'delta')==expected


def test_missing_fold_or_entire_cell_flagged_without_invented_values():
    wide=wide_fixture().loc[lambda x:~(x.family.eq('A')&x.city.eq('X')&x.fold.eq('F4'))]
    matrix,_=m.aggregate_matrix(wide,annotations(['A','B']),['A','B'],['X','Y','Z'],['S1'])
    assert len(matrix)==6 and matrix.loc[matrix.city.eq('Z'),'weather_class'].eq('insufficient_evidence').all()
    assert matrix.loc[matrix.city.eq('Z'),'mean_cost_P1'].isna().all()
    assert matrix.loc[matrix.family.eq('A')&matrix.city.eq('X'),'weather_class'].iloc[0]=='insufficient_evidence'


def test_duplicate_observations_and_bad_annotation_grain_fail():
    wide=wide_fixture()
    with pytest.raises(ValueError):m.aggregate_matrix(pd.concat([wide,wide.iloc[[0]]]),annotations(['A','B']),['A','B'],['X','Y'],['S1'])
    for ann in [annotations(['A']),annotations(['A','A','B'])]:
        with pytest.raises(ValueError):m.aggregate_matrix(wide,ann,['A','B'],['X','Y'],['S1'])


def test_no_cell_ci_or_significance_and_bootstrap_grain_guard():
    matrix,_=m.aggregate_matrix(wide_fixture(),annotations(['A','B']),['A','B'],['X','Y'],['S1'])
    matrix['confidence_interval']='pooled CI copied to cell'
    with pytest.raises(ValueError):m.validate_matrix(matrix,['A','B'],['X','Y'],['S1'])
    intervals=pd.DataFrame([dict(scope='pooled_F1_F4',scenario='S1',variant='corrected',block_length=7,metric='relative_cost_delta_pct')])
    m.validate_supported_uncertainty(intervals)
    with pytest.raises(ValueError):m.validate_supported_uncertainty(intervals.assign(city='X'))
    with pytest.raises(ValueError):m.validate_supported_uncertainty(intervals.assign(scope='family_city'))


def test_zero_baseline_indices_and_descriptive_ranks():
    wide=wide_fixture();wide['P1']=0.
    matrix,_=m.aggregate_matrix(wide,annotations(['A','B']),['A','B'],['X','Y'],['S1'])
    assert matrix.filter(regex='^cost_index_').isna().all().all()
    ranked=m.policy_summary(wide)
    assert len(ranked)==5 and ranked.loc[ranked.policy.eq('P1'),'descriptive_rank'].iloc[0]==1
    assert ranked.rank_interpretation.str.contains('not validated deployment').all()


def test_policy_pairing_exact_keys_and_targets(sources):
    output=m.p4._evaluate(**sources)
    wide=m.paired_costs(sources['baseline'],sources['direct'],output['p4_rows'],sources['expected'].loc[lambda x:x.split.eq('test')],sources['config'])
    assert len(wide)==9 and wide[m.POLICIES].notna().all().all()
    bad=output['p4_rows'].drop(index=0)
    with pytest.raises(ValueError):m.paired_costs(sources['baseline'],sources['direct'],bad,sources['expected'].loc[lambda x:x.split.eq('test')],sources['config'])
    bad=output['p4_rows'].copy();bad.loc[0,'y_true']+=1
    with pytest.raises(ValueError):m.paired_costs(sources['baseline'],sources['direct'],bad,sources['expected'].loc[lambda x:x.split.eq('test')],sources['config'])


def test_annotation_is_family_only_and_summary_uniqueness():
    matrix,_=m.aggregate_matrix(wide_fixture(),annotations(['A','B']),['A','B'],['X','Y'],['S1'])
    assert matrix.m2_annotation_grain.str.startswith('family only').all()
    assert matrix.groupby('family').m2_rq1_econometric_signature.nunique().eq(1).all()
    summary=m.cell_summary(matrix,m.policy_summary(wide_fixture()))
    assert len(summary)==4 and not summary.duplicated(['family','city']).any()


def test_integrity_and_exclusive_writes(tmp_path):
    assert m.inspect_bundle(tmp_path)=='incomplete'
    m.p4.reproduction._json(tmp_path/'table.json',{'a':1})
    with pytest.raises(FileExistsError):m.p4.reproduction._json(tmp_path/'table.json',{})
    output={'table.json':m.file_hash(tmp_path/'table.json')}
    m.p4.reproduction._json(tmp_path/'manifest.json',{'output_sha256':output})
    m.p4.reproduction._json(tmp_path/'COMPLETE.json',{'files':{**output,'manifest.json':m.file_hash(tmp_path/'manifest.json')}})
    assert m.inspect_bundle(tmp_path)=='complete'
    (tmp_path/'table.json').write_text('tamper')
    assert m.inspect_bundle(tmp_path)=='corrupt'


def test_forecast_cost_link_cannot_silently_change(sources):
    records=[]
    for config,policy in [('M0','P2'),('M3','P3')]:
        rows=sources['direct'].loc[sources['direct'].policy.eq(policy)&sources['direct'].variant.eq('corrected')].copy()
        rows['config']=config;rows['split']='test';rows['corrected_prediction']=rows.quantity;rows['raw_prediction']=rows.quantity
        records.append(rows)
    panel=pd.concat(records,ignore_index=True)
    m.verify_quantile_link(panel,sources['direct'])
    panel.loc[0,'raw_prediction']+=.01
    with pytest.raises(ValueError):m.verify_quantile_link(panel,sources['direct'])
