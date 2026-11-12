"""Synthetic approval/comparison tests, no production arithmetic or model fitting."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from test_m4_p1_residual_policy import points, FOLDS, forbid_ml_fitting
from src.decision import p1_residual_policy as p1
from src.decision import p1_execution as execution


@pytest.fixture
def evaluated():
    cfg=p1.read_yaml(p1.ROOT/'configs/cost_scenarios.yaml')
    inputs=p1.P1Inputs(points(),points('test'),pd.DataFrame(),FOLDS,'synthetic',{})
    rows,_=p1.evaluate_p1_synthetic(inputs,p1.ResidualSpec((),'linear',2,()),cfg)
    direct=pd.concat([rows.assign(policy=pol,variant=var) for pol in ['P2','P3']
                      for var in ['corrected','raw_clipped_sensitivity']],ignore_index=True)
    return rows,direct,cfg


def test_proposal_is_explicit_and_not_approved():
    spec=execution.proposed_spec()
    assert execution.PROPOSED=='PROPOSED_NOT_APPROVED'
    assert spec.pooling==('family',) and spec.minimum_samples==30
    assert spec.quantile_method=='linear' and spec.fallback==((),)
    assert execution.PROPOSAL['tau_keys']==list(execution.TAU_KEYS)


def test_proposed_minimum_30_boundary_and_family_fallback():
    first=points(truth=[12.]*16,prediction=[10.]*16)
    second=points(truth=[12.]*14,prediction=[10.]*14);second['store_nbr']='2'
    sparse=points(truth=[14.],prediction=[10.],families=['B']);sparse['store_nbr']='3'
    validation=pd.concat([first,second,sparse],ignore_index=True)
    fit=p1.fit_residual_adjuster(validation,execution.proposed_spec(),FOLDS,(.5,.75,5/6))
    test=points('test',truth=[0.,0.],prediction=[1.,1.],families=['A','B'])
    q=fit.predict(test)
    assert q.groupby('family').fallback_level.first().to_dict()=={'A':0,'B':1}
    assert q.groupby('family').calibration_samples.first().to_dict()=={'A':30,'B':31}
    with pytest.raises(ValueError,match='minimum_samples'):
        p1.fit_residual_adjuster(validation.iloc[:29],execution.proposed_spec(),FOLDS,(.5,.75,5/6))


def test_empty_calibration_fails():
    with pytest.raises(ValueError):
        p1.fit_residual_adjuster(points().iloc[:0],execution.proposed_spec(),FOLDS,(.5,.75,5/6))


def test_exact_matched_indices_and_separate_variants(evaluated):
    rows,direct,cfg=evaluated
    paired,summary=execution.compare_with_p1(rows,direct,cfg)
    assert len(paired)==len(rows)*4
    assert summary.cost_index_policy.eq(1).all() and summary.cost_index_P1.eq(1).all()
    assert set(summary.variant)=={'corrected','raw_clipped_sensitivity'}
    assert set(summary.report_level)==set(execution.LEVELS)


@pytest.mark.parametrize('fault',['missing','duplicate','truth','city','cost','negative','null'])
def test_bad_direct_panels_fail(evaluated,fault):
    rows,direct,cfg=evaluated;direct=direct.copy()
    if fault=='missing':direct=direct.drop(index=0)
    elif fault=='duplicate':direct=pd.concat([direct,direct.iloc[[0]]],ignore_index=True)
    elif fault=='truth':direct.loc[0,'y_true']+=1
    elif fault=='city':direct.loc[0,'city']='Unknown'
    elif fault=='cost':direct.loc[0,'simulated_cost']+=1
    elif fault=='negative':direct.loc[0,'quantity']=-1
    else:direct.loc[0,'city']=None
    with pytest.raises(ValueError):execution.compare_with_p1(rows,direct,cfg)


def test_zero_p1_cost_returns_nan_not_infinity(evaluated):
    rows,direct,cfg=evaluated
    for frame in [rows,direct]:frame['quantity']=frame.y_true;frame['simulated_cost']=0.
    _,summary=execution.compare_with_p1(rows,direct,cfg)
    assert summary.cost_index_policy.isna().all() and summary.cost_index_P1.isna().all()
    assert summary.p1_denominator_zero.all()


def approval(tmp_path):
    evidence=tmp_path/'input.txt';evidence.write_text('frozen')
    request=dict(status=execution.PROPOSED,methodology=deepcopy(execution.PROPOSAL),
                 file_sha256={str(evidence):execution.file_hash(evidence)})
    return dict(status='APPROVED',request=request,request_sha256=execution.semantic_hash(request),
                approved_by='SYNTHETIC TEST ONLY',approval_evidence='Synthetic fixture, no real approval',
                approval_communicated_date='2026-11-09')


@pytest.mark.parametrize('fault',['pending','method','digest','evidence','hash','empty_hashes'])
def test_approval_rejects_pending_drift_and_missing_evidence(tmp_path,fault):
    record=approval(tmp_path)
    if fault=='pending':record['status']=execution.PROPOSED
    elif fault=='method':record['request']['methodology']['minimum_samples']=29
    elif fault=='digest':record['request_sha256']='wrong'
    elif fault=='evidence':record['approval_evidence']=''
    elif fault=='hash':(tmp_path/'input.txt').write_text('changed')
    else:
        record['request']['file_sha256']={}
        record['request_sha256']=execution.semantic_hash(record['request'])
    with pytest.raises(ValueError):execution.validate_approval(record)


def test_valid_synthetic_receipt_and_pending_runner_never_loads_real_data(tmp_path,monkeypatch):
    record=approval(tmp_path)
    assert execution.validate_approval(record)==record['request']
    path=tmp_path/'pending.json';path.write_text(json.dumps({'status':execution.PROPOSED}))
    monkeypatch.setattr(p1,'load_real_p1_inputs',lambda *a:pytest.fail('Pending request reached real loader'))
    with pytest.raises(p1.MethodologyPendingError):execution.run_approved(path,tmp_path)


def test_truncated_bindings_fail_before_loader(tmp_path,monkeypatch):
    record=approval(tmp_path);path=tmp_path/'approval.json';path.write_text(json.dumps(record))
    monkeypatch.setattr(execution,'make_request',lambda root:{'different':'required bindings'})
    monkeypatch.setattr(p1,'load_real_p1_inputs',lambda *a:pytest.fail('Truncated binding reached real loader'))
    with pytest.raises(ValueError,match='bindings'):execution.run_approved(path,tmp_path)


def test_exclusive_json_and_incomplete_bundle_detection(tmp_path):
    path=tmp_path/'request.json';execution._json(path,{'status':execution.PROPOSED})
    with pytest.raises(FileExistsError):execution._json(path,{})
    assert execution.inspect_bundle(tmp_path)=='incomplete'


def test_approved_runner_synthetic_bundle_roundtrip_and_unique_outputs(tmp_path,monkeypatch,evaluated):
    rows,direct,cfg=evaluated
    root=tmp_path;record=approval(root)
    path=root/'approved.json';path.write_text(json.dumps(record))
    source=root/execution.COST_RELATIVE;source.mkdir(parents=True)
    direct.to_csv(source/'row_costs.csv',index=False)
    (source/'manifest.json').write_text(json.dumps(dict(source_forecast_provenance={'file_sha256':{},'input':{}},file_sha256={})))
    (root/'configs').mkdir()
    (root/'configs/cost_scenarios.yaml').write_bytes((p1.ROOT/'configs/cost_scenarios.yaml').read_bytes())
    inputs=p1.P1Inputs(points(),points('test'),pd.DataFrame(),FOLDS,'synthetic',{'file_sha256':{},'input':{}})
    monkeypatch.setattr(execution,'make_request',lambda root:record['request'])
    monkeypatch.setattr(execution,'inspect_evaluation',lambda path:'complete')
    monkeypatch.setattr(p1,'load_real_p1_inputs',lambda root:inputs)
    monkeypatch.setattr(execution,'proposed_spec',lambda:p1.ResidualSpec((),'linear',2,()))
    first=execution.run_approved(path,root)
    second=execution.run_approved(path,root)
    assert first!=second and execution.inspect_bundle(first)=='complete' and execution.inspect_bundle(second)=='complete'
    saved=pd.read_csv(first/'p1_rows.csv')
    assert saved.methodology_status.eq('APPROVED').all()
    assert saved.approval_sha256.eq(execution.file_hash(path)).all()
    (first/'p1_rows.csv').write_text('corrupted')
    assert execution.inspect_bundle(first)=='corrupt'
