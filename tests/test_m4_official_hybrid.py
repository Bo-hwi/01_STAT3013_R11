"""Synthetic-only hybrid adapter tests; no real-data costs or calibration."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.decision import p4_official_hybrid as h
from test_m4_p4_policy import sources, forbid_training, FOLDS


@pytest.fixture
def hybrid(sources):
    _,cal=h.old._calibrate(sources['panel'],sources['expected'],sources['folds'],sources['audit'])
    official=sources['panel'].loc[sources['panel'].split.eq('test'),h.old.reproduction.KEY+['config','model','y_true','clipped_prediction']].rename(columns={'clipped_prediction':'y_pred'})
    # Deliberately change official bases to detect accidental M4 test substitution.
    official['y_pred']+=.1234
    selected=h.select_official(official,sources['expected'],sources['audit'])
    return dict(selected=selected,validation=sources['panel'].loc[sources['panel'].split.eq('validation')],
                calibration=cal,**{k:sources[k] for k in ['expected','folds','audit','baseline','direct','config']})


def test_hybrid_uses_official_base_and_unchanged_offsets(hybrid):
    q=h.quantities(hybrid['selected'],hybrid['validation'],hybrid['calibration'],hybrid['folds'],hybrid['audit'])
    point=hybrid['selected'][h.old.residual.POINT_KEY+['y_pred']]
    pair=q.merge(point,on=h.old.residual.POINT_KEY,validate='m:1')
    assert np.array_equal(pair.point_prediction,pair.y_pred)
    np.testing.assert_allclose(pair.quantity,np.maximum(0,pair.y_pred+pair.residual_quantile))
    assert not pair.official_raw_available.any()
    assert set(pair.fallback_level)=={0,1}
    assert pair.quantity.mod(1).ne(0).any()


def test_test_labels_never_change_quantities(hybrid):
    a=h.quantities(hybrid['selected'],hybrid['validation'],hybrid['calibration'],hybrid['folds'],hybrid['audit'])
    test=hybrid['selected'].copy();test['y_true']=1e9
    b=h.quantities(test,hybrid['validation'],hybrid['calibration'],hybrid['folds'],hybrid['audit'])
    pd.testing.assert_frame_equal(a.drop(columns='y_true'),b.drop(columns='y_true'))


@pytest.mark.parametrize('problem',['missing','duplicate','truth','model','negative','null'])
def test_official_alignment_guard(hybrid,problem):
    x=hybrid['selected'].copy()
    if problem=='missing':x=x.drop(index=x.index[0])
    elif problem=='duplicate':x=pd.concat([x,x.iloc[[0]]],ignore_index=True)
    elif problem=='truth':x.loc[x.index[0],'y_true']+=1.
    elif problem=='model':x.loc[x.index[0],'model']='unselected'
    elif problem=='negative':x.loc[x.index[0],'y_pred']=-1.
    else:x.loc[x.index[0],'city']=None
    with pytest.raises(ValueError):h.select_official(x,hybrid['expected'],hybrid['audit'])


@pytest.mark.parametrize('problem',['n','dates','tau','method','duplicate','stage'])
def test_frozen_offset_contract(hybrid,problem):
    cal=hybrid['calibration'].copy();val=hybrid['validation'].copy()
    if problem=='n':cal.loc[0,'n']+=1
    elif problem=='dates':cal.loc[0,'date_max']=pd.Timestamp('2016-09-01')
    elif problem=='tau':cal.loc[0,'tau_key']='bad'
    elif problem=='method':cal.loc[0,'quantile_method']='nearest'
    elif problem=='duplicate':cal=pd.concat([cal,cal.iloc[[0]]],ignore_index=True)
    else:val.loc[val.index[0],'fit_stage']='fresh_train_validation_refit'
    with pytest.raises(ValueError):h.frozen_adjuster(cal,val,hybrid['folds'])


def test_synthetic_costs_identity_fixed_policies_and_zero_denominator(hybrid):
    result=h.evaluate(hybrid);rows=result['p4_rows']
    assert len(rows)==18 and len(result['matched_costs'])==36
    for _,g in rows.groupby('scenario'):
        np.testing.assert_allclose(g.simulated_cost,(g.c_over+g.c_under)*h.old.pinball_loss(g.y_true,g.quantity,g.tau.iloc[0]))
    hybrid['baseline']=hybrid['baseline'].assign(simulated_cost=0.)
    result=h.evaluate(hybrid)
    assert result['comparisons'].filter(regex='^cost_index_').isna().all().all()


def test_pending_and_old_approvals_stop_before_real_sources(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Pending approval reached real sources/economics')
    monkeypatch.setattr(h,'load_sources',forbidden);monkeypatch.setattr(h,'evaluate',forbidden)
    for status in ['PENDING_APPROVAL','APPROVED']:
        path=tmp_path/(status+'.json');path.write_text(json.dumps(dict(status=status,request={'status':'old'})))
        with pytest.raises(ValueError,match='PENDING_APPROVAL'):h.run_approved(path,tmp_path)


def test_approval_digest_environment_and_hash_drift(monkeypatch,tmp_path):
    f=tmp_path/'source';f.write_text('frozen')
    request=dict(status=h.STATUS,specification=h.SPEC,file_sha256={str(f):h.old.file_hash(f)})
    receipt=dict(status='APPROVED',request=request,request_sha256=h.old.semantic_hash(request),approved_by='Synthetic',approval_evidence='Fixture only',approval_communicated_date='2026-11-11')
    monkeypatch.setattr(h,'make_request',lambda root:request)
    assert h.validate_approval(receipt,tmp_path)==request
    bad=deepcopy(receipt);bad['request_sha256']='wrong'
    with pytest.raises(ValueError,match='digest'):h.validate_approval(bad,tmp_path)
    f.write_text('changed')
    with pytest.raises(ValueError):h.validate_approval(receipt,tmp_path)


def test_synthetic_transaction_nonoverwrite_and_integrity(monkeypatch,tmp_path,hybrid):
    (tmp_path/'results/experiments').mkdir(parents=True)
    receipt=dict(status='APPROVED',request=dict(file_sha256={}))
    path=tmp_path/'approved.json';path.write_text(json.dumps(receipt))
    monkeypatch.setattr(h,'validate_approval',lambda *a:receipt['request'])
    monkeypatch.setattr(h,'load_sources',lambda *a:(hybrid,{}))
    a=h.run_approved(path,tmp_path);b=h.run_approved(path,tmp_path)
    assert a!=b and h.inspect_bundle(a)=='complete' and h.inspect_bundle(b)=='complete'
    (a/'comparisons.csv').write_text('corrupt')
    assert h.inspect_bundle(a)=='corrupt'
    def fail(*a,**k):raise OSError('simulated disk failure')
    monkeypatch.setattr(pd.DataFrame,'to_csv',fail)
    with pytest.raises(OSError):h.run_approved(path,tmp_path)
    failed=[p for p in (tmp_path/'results/experiments').iterdir() if (p/'FAILED.json').exists()]
    assert len(failed)==1 and h.inspect_bundle(failed[0])=='failed'
