"""Roles canonical non-training checks. No model/economic/bootstrap execution."""
import json
import numpy as np
import pandas as pd
import pytest
from src.decision import decision_matrix_canonical as c
from test_m4_p4_policy import sources,forbid_training


def quantile_fixture():
    config=c.matrix.p4.reproduction.read_yaml(c.ROOT/'configs/cost_scenarios.yaml');records=[]
    for cfg in ['M0','M3']:
        for split,date in [('test','2016-09-01'),('validation','2016-08-04')]:
            for i,s in enumerate(c.scenarios(config)):
                records.append(dict(date=pd.Timestamp(date),store_nbr='1',family='A',city='X',fold='F1',model='HistGradientBoostingQuantile',config=cfg,
                    tau_key=s['tau_key'],tau=s['tau'],y_true=1.25,corrected_prediction=float(i)+.25,split=split))
    source=pd.DataFrame(records)
    expected=source.loc[source.split.eq('test'),['date','store_nbr','family','city','fold','y_true']].drop_duplicates().assign(split='test')
    return source,expected,config


def test_canonical_quantile_schema_exact_tau_test_only_and_unchanged_values():
    source,expected,config=quantile_fixture();before=source.copy(deep=True)
    out=c.canonical_quantiles(source,config);c.validate_quantiles(out,expected,config)
    assert list(out)==c.QCOLS and len(out)==6 and set(out.tau)=={.5,.75,.8333333333}
    assert out.y_pred.tolist()==source.loc[source.split.eq('test'),'corrected_prediction'].tolist()
    pd.testing.assert_frame_equal(source,before)


@pytest.mark.parametrize('fault',['negative','crossing','missing','duplicate','truth','key','null'])
def test_invalid_canonical_quantiles_fail(fault):
    source,expected,config=quantile_fixture();out=c.canonical_quantiles(source,config).reset_index(drop=True)
    if fault=='negative':out.loc[0,'y_pred']=-1
    elif fault=='crossing':out.loc[0,'y_pred']=99
    elif fault=='missing':out=out.drop(index=0)
    elif fault=='duplicate':out=pd.concat([out,out.iloc[[0]]],ignore_index=True)
    elif fault=='truth':out.loc[0,'y_true']+=1
    elif fault=='key':out.loc[0,'city']='WRONG'
    else:out.loc[0,'model']=None
    with pytest.raises(ValueError):c.validate_quantiles(out,expected,config)


def test_matrix_154_style_schema_preserves_three_directions_and_conservative_recommendation():
    from test_m4_consolidated_matrix import wide_fixture,annotations
    base,_=c.matrix.aggregate_matrix(wide_fixture(),annotations(['A','B']),['A','B'],['X','Y'],['S1'])
    parts=[]
    for scenario in ['S1','S3','S5']:
        p=base.copy();p['scenario']=scenario;p['baseline_error']=2.;p['weather_error']=1.9;p['delta_error_pct']=-5.;parts.append(p)
    source=pd.concat(parts,ignore_index=True);before=source.copy(deep=True)
    config=c.matrix.p4.reproduction.read_yaml(c.ROOT/'configs/cost_scenarios.yaml')
    out=c.canonical_matrix(source,config);c.validate_matrix(out,['A','B'],['X','Y'])
    assert list(out)==c.MCOLS and len(out)==4 and out.recommendation.eq('Further validation').all()
    assert all(len(json.loads(s))==3 for s in out.cost_scenarios)
    assert all(json.loads(s)['lower'] is None for s in out.confidence_interval)
    pd.testing.assert_frame_equal(source,before)
    out.loc[0,'recommendation']='Prioritize pilot'
    with pytest.raises(ValueError):c.validate_matrix(out,['A','B'],['X','Y'])


def test_canonical_costs_only_copy_supported_ci_and_existing_metrics(sources):
    result=c.matrix.p4._evaluate(**sources);rows=result['comparisons'].copy();direct=[]
    for _,r in rows.loc[rows.config.eq('M0')].iterrows():
        direct.append({**{x:r[x] if x in r else np.nan for x in ['scenario','variant','report_level','fold','family','city']},
                       **{name:1. for name in ['mean_over_P2','mean_under_P2','mean_over_P3','mean_under_P3']}})
    for name in ['fold','family','city']:
        if name not in rows:rows[name]=np.nan
    direct=pd.DataFrame(direct)
    p1=sources['baseline'].assign(overstock_units=1.,understock_units=1.)
    p4=result['p4_rows'][['scenario','config','fold','family','city']].assign(overstock_units=1.,understock_units=1.)
    ci=pd.DataFrame([dict(scope=scope,scenario=scenario,variant=variant,block_length=7,metric='relative_cost_delta_pct',lower=-1.,upper=1.)
                     for scope in ['pooled_F1_F4','F1'] for scenario in ['S1','S3','S5'] for variant in ['corrected','raw_clipped_sensitivity']])
    out=c.canonical_costs(rows,direct,p1,p4,ci,sources['config']);c.validate_costs(out)
    assert out.loc[~out.report_level.isin(['pooled','fold']),'P2P3_relative_delta_CI95_lower_pct'].isna().all()
    assert out.P4_comparison_CI.eq('UNAVAILABLE').all()
    bad=out.copy();bad.loc[bad.report_level.eq('family_city'),'P2P3_relative_delta_CI95_lower_pct']=-.2
    with pytest.raises(ValueError):c.validate_costs(bad)


def test_existing_target_is_inspected_but_never_overwritten(tmp_path):
    a=tmp_path/'a';a.write_text('source');target=tmp_path/'target'
    assert c.publish_exclusive(a,target)=='CREATED_EXCLUSIVELY'
    assert c.publish_exclusive(a,target)=='REUSED_IDENTICAL_AFTER_INSPECTION'
    target.write_text('user content')
    assert c.publish_exclusive(a,target)=='BLOCKED_EXISTING_DIFFERENT_FILE' and target.read_text()=='user content'


def test_required_entry_point_and_integrity_detection(tmp_path,monkeypatch):
    import sys
    from src.decision import decision_matrix as entry
    p=c.ROOT/'src/decision/decision_matrix.py'
    before=p.read_bytes();monkeypatch.setattr(sys,'argv',['decision_matrix','--help'])
    with pytest.raises(SystemExit) as result:entry.main()
    assert result.value.code==0 and p.read_bytes()==before
    monkeypatch.setattr(sys,'argv',['decision_matrix'])
    with pytest.raises(SystemExit) as result:entry.main()
    assert result.value.code==2 and p.read_bytes()==before
    assert c.inspect_audit(tmp_path)=='incomplete'
