"""Synthetic/non-training checks for the NEW clipped-residual P4 proposal."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.decision import p4_policy_evaluation as policy

FOLDS=[dict(fold_id='F1',train=['2013-01-01','2016-08-03'],validation=['2016-08-04','2016-08-31'],test=['2016-09-01','2016-09-28'])]


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    for cls in [Ridge,ExtraTreesRegressor,HistGradientBoostingRegressor]:
        monkeypatch.setattr(cls,'fit',lambda *a,**k:pytest.fail('No forecasting fits permitted'))


@pytest.fixture
def sources():
    records=[]
    for config,model in [('M0','Ridge'),('M3','ExtraTrees')]:
        for i in range(32):
            records.append(dict(fold='F1',split='validation',date=pd.Timestamp('2016-08-04')+pd.Timedelta(days=i%20),
                store_nbr=str(i//20+1),family='A' if i<30 else 'B',city='X',y_true=2.25+i%3,
                config=config,model=model,raw_prediction=float(i%5-2),fit_stage='train_only',selected_parameters='{"test": 1}'))
        for i,family in enumerate(['A','B','UNSEEN']):
            records.append(dict(fold='F1',split='test',date=pd.Timestamp('2016-09-01')+pd.Timedelta(days=i),
                store_nbr='1',family=family,city='X',y_true=float(i)*2.25,config=config,model=model,
                raw_prediction=-1.5 if i==0 else 1.25,fit_stage='fresh_train_validation_refit',selected_parameters='{"test": 1}'))
    panel=pd.DataFrame(records);panel['clipped_prediction']=np.maximum(panel.raw_prediction,0)
    expected=panel[policy.reproduction.KEY+['y_true']].drop_duplicates()
    tuning=pd.DataFrame([dict(fold='F1',config=c,model=m,parameters='{"test": 1}',validation_MAE=1.,selected_candidate=True,selected_algorithm=True) for c,m in [('M0','Ridge'),('M3','ExtraTrees')]])
    audit=policy.selection_audit(panel,FOLDS,tuning)
    config=policy.reproduction.read_yaml(policy.ROOT/'configs/cost_scenarios.yaml')
    test=expected.loc[expected.split.eq('test')];baseline=[];direct=[]
    for s in policy.configured_scenarios(config):
        for _,obs in test.iterrows():
            key={c:obs[c] for c in policy.MATCH if c!='tau_key'};key['tau_key']=s['tau_key']
            base=dict(key,scenario=s['scenario'],y_true=obs.y_true,quantity=2.5,policy='P1')
            base['simulated_cost']=policy.newsvendor_cost([obs.y_true],[2.5],s['c_over'],s['c_under'])[0];baseline.append(base)
            for name,q in [('P2',1.5),('P3',2.25)]:
                for variant in ['corrected','raw_clipped_sensitivity']:
                    direct.append(dict(key,scenario=s['scenario'],y_true=obs.y_true,quantity=q,policy=name,variant=variant,
                        simulated_cost=policy.newsvendor_cost([obs.y_true],[q],s['c_over'],s['c_under'])[0]))
    return dict(panel=panel,expected=expected,folds=FOLDS,audit=audit,baseline=pd.DataFrame(baseline),direct=pd.DataFrame(direct),config=config)


def test_changed_proposal_and_clipping_consistency_group_sizes_fallback(sources):
    assert policy.PROPOSAL['residual']=='y_true - clipped_validation_prediction'
    assert policy.PROPOSAL['pooling']==['fold','config','family']
    q,cal=policy._calibrate(sources['panel'],sources['expected'],FOLDS,sources['audit'])
    assert q.methodology_status.eq('PENDING_APPROVAL').all()
    assert q.groupby('family').fallback_level.first().to_dict()=={'A':0,'B':1,'UNSEEN':1}
    assert q.groupby('family').calibration_samples.first().to_dict()=={'A':30,'B':32,'UNSEEN':32}
    assert q.loc[q.family.eq('A'),'point_prediction'].eq(0).all()
    v=sources['panel'].loc[lambda x:x.split.eq('validation')&x.config.eq('M0')&x.family.eq('A')]
    for tau,g in q.loc[q.family.eq('A')&q.config.eq('M0')].groupby('tau'):
        np.testing.assert_allclose(g.residual_quantile,np.quantile(v.y_true-v.clipped_prediction,tau,method='linear'))
    assert q.refit_cutoff.eq('2016-08-31').all() and q.training_cutoff.eq('2016-08-03').all()
    assert q.selected_parameters.eq('{"test": 1}').all() and q.quantity.mod(1).ne(0).any()


def test_test_targets_cannot_change_quantities(sources):
    a,_=policy._calibrate(sources['panel'],sources['expected'],FOLDS,sources['audit'])
    panel=sources['panel'].copy();panel.loc[panel.split.eq('test'),'y_true']=1e6
    expected=panel[policy.reproduction.KEY+['y_true']].drop_duplicates()
    b,_=policy._calibrate(panel,expected,FOLDS,sources['audit'])
    pd.testing.assert_frame_equal(a.drop(columns='y_true'),b.drop(columns='y_true'))


def test_scenarios_cost_identity_indices_and_separate_variants(sources):
    result=policy._evaluate(**sources);rows=result['p4_rows'];pairs=result['matched_costs'];summary=result['comparisons']
    assert len(rows)==18 and not rows.duplicated(policy.MATCH+['config']).any()
    assert len(pairs)==36 and set(summary.variant)=={'corrected','raw_clipped_sensitivity'}
    for _,g in rows.groupby('scenario'):
        tau=g.tau.iloc[0];scale=g.c_over.iloc[0]+g.c_under.iloc[0]
        np.testing.assert_allclose(g.simulated_cost,scale*policy.pinball_loss(g.y_true,g.quantity,tau),rtol=1e-12,atol=0)
    for _,r in summary.loc[summary.report_level.eq('pooled')].iterrows():
        g=pairs.loc[pairs.scenario.eq(r.scenario)&pairs.config.eq(r.config)&pairs.variant.eq(r.variant)]
        assert r.cost_index_P4==policy.cost_index(g.cost_P4,g.cost_P1)
    assert rows.quantity.ge(0).all()


@pytest.mark.parametrize('fault',['missing','duplicate','truth','tau','null','variant'])
def test_invalid_fixed_policy_rows_fail(sources,fault):
    direct=sources['direct'].copy()
    if fault=='missing':direct=direct.drop(index=0)
    elif fault=='duplicate':direct=pd.concat([direct,direct.iloc[[0]]],ignore_index=True)
    elif fault=='truth':direct.loc[0,'y_true']+=1
    elif fault=='tau':direct.loc[0,'tau_key']='tau_BAD'
    elif fault=='null':direct.loc[0,'city']=None
    else:direct.loc[0,'variant']='unapproved'
    with pytest.raises(ValueError):policy.validate_fixed_costs(sources['expected'].loc[lambda x:x.split.eq('test')],sources['baseline'],direct,sources['config'])


def test_zero_denominator_indices_are_nan(sources):
    sources['baseline']['quantity']=sources['baseline'].y_true;sources['baseline']['simulated_cost']=0.
    result=policy._evaluate(**sources)
    assert result['comparisons'].filter(regex='^cost_index_').isna().all().all()
    assert result['comparisons'].p1_denominator_zero.all()


def test_minimum_size_missing_fallback_and_stage_leakage_fail(sources):
    small=sources['panel'].loc[lambda x:x.split.eq('test')|x.family.eq('B')]
    expected=small[policy.reproduction.KEY+['y_true']].drop_duplicates()
    with pytest.raises(ValueError,match='minimum_samples'):policy._calibrate(small,expected,FOLDS,sources['audit'])
    wrong=sources['panel'].copy();wrong.loc[0,'fit_stage']='fresh_train_validation_refit'
    with pytest.raises(ValueError):policy._calibrate(wrong,sources['expected'],FOLDS,sources['audit'])


@pytest.mark.parametrize('field,value',[('model','HistGradientBoosting'),('selected_parameters','{}')])
def test_selected_mapping_cannot_change_during_calibration(sources,field,value):
    audit=sources['audit'].copy();audit.loc[0,field]=value
    with pytest.raises(ValueError,match='audit mapping'):
        policy._calibrate(sources['panel'],sources['expected'],FOLDS,audit)


def test_approved_receipt_requires_exact_clipped_method(tmp_path):
    request=dict(status=policy.STATUS,methodology=deepcopy(policy.PROPOSAL),file_sha256={})
    request['methodology']['residual']='y_true - raw_validation_prediction'
    receipt=dict(status='APPROVED',request=request,request_sha256=policy.semantic_hash(request))
    with pytest.raises(ValueError,match='method/request'):policy.validate_approval(receipt,tmp_path)


def test_old_and_pending_approvals_rejected_before_sources_or_arithmetic(tmp_path,monkeypatch):
    monkeypatch.setattr(policy,'load_sources',lambda *a:pytest.fail('Pending approval reached real sources'))
    for r in [dict(status='PENDING_APPROVAL'),dict(status='APPROVED',request={'status':'PROPOSED_NOT_APPROVED'})]:
        path=tmp_path/('receipt_'+str(len(json.dumps(r)))+'.json');path.write_text(json.dumps(r))
        with pytest.raises(ValueError):policy.run_approved(path,tmp_path)


def test_mock_approved_synthetic_bundle_nonoverwrite_integrity(tmp_path,monkeypatch,sources):
    evidence=tmp_path/'input.txt';evidence.write_text('frozen')
    request=dict(status=policy.STATUS,methodology=deepcopy(policy.PROPOSAL),file_sha256={str(evidence):policy.file_hash(evidence)})
    receipt=dict(status='APPROVED',request=request,request_sha256=policy.semantic_hash(request),approved_by='SYNTHETIC TEST ONLY',
                 approval_evidence='Not a real approval',approval_communicated_date='2026-11-09')
    path=tmp_path/'approved.json';path.write_text(json.dumps(receipt))
    monkeypatch.setattr(policy,'make_request',lambda root:request)
    monkeypatch.setattr(policy,'load_sources',lambda root:sources)
    first=policy.run_approved(path,tmp_path);second=policy.run_approved(path,tmp_path)
    assert first!=second and policy.inspect_bundle(first)=='complete' and policy.inspect_bundle(second)=='complete'
    (first/'p4_rows.csv').write_text('tamper')
    assert policy.inspect_bundle(first)=='corrupt'
    evidence.write_text('changed')
    with pytest.raises(ValueError):policy.validate_approval(receipt,tmp_path)


def test_failed_write_remains_incomplete_with_diagnostics(tmp_path,monkeypatch,sources):
    monkeypatch.setattr(policy,'validate_approval',lambda *a:None)
    monkeypatch.setattr(policy,'load_sources',lambda root:sources)
    path=tmp_path/'synthetic_receipt.json';path.write_text('{}')
    original=pd.DataFrame.to_csv
    def fail(self,*a,**k):raise OSError('synthetic write failure')
    monkeypatch.setattr(pd.DataFrame,'to_csv',fail)
    with pytest.raises(OSError):policy.run_approved(path,tmp_path)
    failed=list((tmp_path/'results/experiments').glob('.*.staging-*'))
    assert len(failed)==1 and policy.inspect_bundle(failed[0])=='failed'
    assert not (failed[0]/'COMPLETE.json').exists()
