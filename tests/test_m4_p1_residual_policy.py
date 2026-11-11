"""Synthetic P1 infrastructure tests only. No production policy evaluation/fitting."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from test_m4_quantile_pipeline import contracts, features, small_thread_budget
from src.decision import p1_residual_policy as p1
from src.decision.quantile_models import pinball_loss


FOLDS=[dict(fold_id='F1',train=['2013-01-01','2016-08-03'],validation=['2016-08-04','2016-08-31'],test=['2016-09-01','2016-09-28']),
       dict(fold_id='F2',train=['2013-01-01','2016-10-17'],validation=['2016-10-18','2016-11-14'],test=['2016-11-15','2016-12-12'])]
TAUS=(.5,.75,5/6)


@pytest.fixture(autouse=True)
def forbid_ml_fitting(monkeypatch):
    for estimator in [HistGradientBoostingRegressor,ExtraTreesRegressor,Ridge]:
        monkeypatch.setattr(estimator,'fit',lambda *a,**k:pytest.fail('P1 tests must not train forecasting models'))


def points(split='validation',fold='F1',truth=None,prediction=None,cities=None,families=None):
    prediction=prediction if prediction is not None else [10.]*4
    truth=truth if truth is not None else [8.,10.,12.,14.]
    spec=next(f for f in FOLDS if f['fold_id']==fold)
    n=len(prediction);dates=pd.date_range(spec[split][0],periods=n)
    return pd.DataFrame(dict(date=dates,store_nbr=['1']*n,family=families or ['A']*n,city=cities or ['X']*n,
        fold=fold,model='SeasonalNaive',config='M0',y_true=truth,y_pred=prediction,split=split,score_mask=True,
        forecast_origin=dates-pd.Timedelta(days=1),horizon_days=1))


@pytest.fixture
def spec():return p1.ResidualSpec(pooling=(),quantile_method='linear',minimum_samples=2,fallback=())


@pytest.fixture
def baseline_source(features,contracts):
    data=p1.prepare_features(features,ready=True);outputs=[]
    for fold in p1.iter_folds(data,*contracts,contracts[0]['scope']['deep_study_families']):
        rows=fold.test.loc[fold.test.score_mask]
        forecast,_=p1.m1.seasonal_naive(rows)
        outputs.append(p1.m1.to_output(rows,forecast,fold.fold_id,'SeasonalNaive'))
        outputs.append(p1.m1.to_output(rows,rows.sales_roll7_mean,fold.fold_id,'MovingAverage'))
    return pd.concat(outputs,ignore_index=True)


def test_adapter_reuses_m1_rule_and_exact_shared_scored_rows(features,contracts,baseline_source,monkeypatch):
    original=p1.m1.seasonal_naive;calls=[]
    def spy(rows):calls.append(rows.copy());return original(rows)
    monkeypatch.setattr(p1.m1,'seasonal_naive',spy)
    inputs=p1.build_p1_inputs(features,baseline_source,*contracts,data_kind='synthetic')
    assert len(calls)==2 and len(inputs.diagnostics)==2
    fold=next(p1.iter_folds(p1.prepare_features(features,ready=True),*contracts,contracts[0]['scope']['deep_study_families']))
    for split in ['validation','test']:
        out=getattr(inputs,split);expected=getattr(fold,split).loc[lambda x:x.score_mask]
        assert len(out)==len(expected) and out.score_mask.all()
        assert out.date.min()>=expected.date.min() and out.date.max()<=expected.date.max()
        assert out.forecast_origin.eq(out.date-pd.Timedelta(days=1)).all()
    source=baseline_source.loc[baseline_source.model.eq('SeasonalNaive')].copy();source['store_nbr']=source.store_nbr.astype(str)
    pd.testing.assert_frame_equal(inputs.test[p1.m1.OUTPUT_COLUMNS].sort_values(p1.POINT_KEY).reset_index(drop=True),
                                  source[p1.m1.OUTPUT_COLUMNS].sort_values(p1.POINT_KEY).reset_index(drop=True),check_dtype=False)


@pytest.mark.parametrize('fault',['duplicate','missing','truth','prediction','unknown_fold','unscored_extra'])
def test_adapter_rejects_bad_existing_test_predictions(features,contracts,baseline_source,fault):
    b=baseline_source.copy();sn=b.index[b.model.eq('SeasonalNaive')][0]
    if fault=='duplicate':b=pd.concat([b,b.loc[[sn]]],ignore_index=True)
    elif fault=='missing':b=b.drop(index=sn)
    elif fault=='truth':b.loc[sn,'y_true']+=1
    elif fault=='prediction':b.loc[sn,'y_pred']+=.001
    elif fault=='unknown_fold':b.loc[sn,'fold']='F9'
    else:
        f=next(p1.iter_folds(p1.prepare_features(features,ready=True),*contracts,contracts[0]['scope']['deep_study_families']))
        excluded=f.test.loc[~f.test.score_mask].head(1);assert len(excluded)==1
        pred,_=p1.m1.seasonal_naive(excluded)
        b=pd.concat([b,p1.m1.to_output(excluded,pred,f.fold_id,'SeasonalNaive')],ignore_index=True)
    with pytest.raises(ValueError):p1.build_p1_inputs(features,b,*contracts,data_kind='synthetic')


def test_m1_lag_fallback_path_is_counted():
    frame=pd.DataFrame(dict(date=pd.date_range('2016-08-04',periods=2),store_nbr=[1,1],family=['A','A'],city=['X','X'],
                            sales=[1.,2.],sales_lag7=[np.nan,np.nan],sales_lag14=[1.25,np.nan],sales_roll7_mean=[2.,2.75]))
    out,log=p1._point_output(frame,'F1','validation')
    np.testing.assert_array_equal(out.y_pred,[1.25,2.75])
    assert log['lag7_missing']==2 and log['lag14_used']==1 and log['rolling7_used']==1


def test_signed_residual_quantiles_fractional_and_pending(spec):
    model=p1.fit_residual_adjuster(points(),spec,FOLDS,TAUS)
    q=model.predict(points('test',truth=[0.],prediction=[1.25]))
    np.testing.assert_allclose(q.residual_quantile,np.quantile([-2,0,2,4],TAUS,method='linear'))
    np.testing.assert_allclose(q.quantity,[2.25,3.75,4.25])
    assert q.methodology_status.eq('PENDING_APPROVAL').all() and model.spec.metadata()['minimum_samples']==2
    assert q.quantity.mod(1).ne(0).any() and q.minimum_samples.eq(2).all()


def test_negative_signed_adjustment_clips_only_final_quantity(spec):
    model=p1.fit_residual_adjuster(points(truth=[.25]*4,prediction=[1.25]*4),spec,FOLDS,TAUS)
    q=model.predict(points('test',truth=[0.],prediction=[.25]))
    assert q.residual_quantile.eq(-1).all() and q.point_prediction.eq(.25).all() and q.quantity.eq(0).all()


def test_quantile_method_explicitly_changes_empirical_rule():
    spec=p1.ResidualSpec((),'higher',2,())
    q=p1.fit_residual_adjuster(points(),spec,FOLDS,TAUS).predict(points('test',truth=[0.],prediction=[1.25]))
    assert q.loc[q.tau_key.eq('tau_0.75'),'residual_quantile'].iloc[0]==4.


def test_declared_sparse_and_unseen_group_fallback():
    spec=p1.ResidualSpec(('family','city'),'linear',3,(('family',),()))
    model=p1.fit_residual_adjuster(points(cities=['X','X','X','Y']),spec,FOLDS,TAUS)
    q=model.predict(points('test',truth=[0.,0.,0.],prediction=[1.,1.,1.],cities=['X','Y','Z'],families=['A','A','B']))
    assert q.groupby('city').fallback_level.first().to_dict()=={'X':0,'Y':1,'Z':2}
    assert q.groupby('city').calibration_samples.first().to_dict()=={'X':3,'Y':4,'Z':4}
    assert q.groupby('city').fallback_level.nunique().eq(1).all()  # Same pool across taus.


def test_no_silent_fallback_or_group_dropping():
    model=p1.fit_residual_adjuster(points(),p1.ResidualSpec(('family','city'),'linear',2,()),FOLDS,TAUS)
    with pytest.raises(ValueError,match='predeclared'):model.predict(points('test',truth=[0.],prediction=[1.],cities=['UNKNOWN']))
    with pytest.raises(ValueError,match='minimum_samples'):
        p1.fit_residual_adjuster(points(),p1.ResidualSpec((),'linear',5,()),FOLDS,TAUS)


@pytest.mark.parametrize('fault',['test_split','future_date','wrong_fold','unscored','bad_origin','nan_prediction'])
def test_calibration_is_validation_only(spec,fault):
    frame=points()
    if fault=='test_split':frame['split']='test'
    elif fault=='future_date':
        frame['date']+=pd.Timedelta(days=60);frame['forecast_origin']=frame.date-pd.Timedelta(days=1)
    elif fault=='wrong_fold':frame['fold']='F9'
    elif fault=='unscored':frame.loc[0,'score_mask']=False
    elif fault=='bad_origin':frame['forecast_origin']=frame.date
    else:frame.loc[0,'y_pred']=np.nan
    with pytest.raises(ValueError):p1.fit_residual_adjuster(frame,spec,FOLDS,TAUS)


def test_future_fold_residuals_cannot_enter_earlier_fold(spec):
    first=points();future=points(fold='F2',truth=[1000.]*4)
    one=p1.fit_residual_adjuster(first,spec,FOLDS,TAUS)
    both=p1.fit_residual_adjuster(pd.concat([first,future],ignore_index=True),spec,FOLDS,TAUS)
    test=points('test')
    pd.testing.assert_frame_equal(one.predict(test),both.predict(test))
    # Caller mutation after fitting cannot change cached quantiles.
    first['y_true']=999999
    pd.testing.assert_frame_equal(one.predict(test),both.predict(test))


def test_policy_quantities_are_test_target_blind(spec):
    model=p1.fit_residual_adjuster(points(),spec,FOLDS,TAUS)
    test=points('test');expected=model.predict(test.drop(columns='y_true'))
    changed=test.copy();changed['y_true']=1e200
    pd.testing.assert_frame_equal(expected,model.predict(changed))
    early=test.copy();early['date']=pd.to_datetime('2016-08-31');early['forecast_origin']=early.date-pd.Timedelta(days=1)
    with pytest.raises(ValueError):model.predict(early)


def test_global_fallback_cannot_cross_model_or_config(spec):
    model=p1.fit_residual_adjuster(points(),spec,FOLDS,TAUS)
    for column,value in [('model','OtherPointModel'),('config','M3')]:
        test=points('test');test[column]=value
        with pytest.raises(ValueError,match='predeclared'):model.predict(test)


def test_duplicate_fold_specs_and_tau_order_fail(spec):
    with pytest.raises(ValueError,match='unique nonempty'):
        p1.fit_residual_adjuster(points(),spec,[FOLDS[0],FOLDS[0]],TAUS)
    with pytest.raises(ValueError,match='unique and increasing'):
        p1.fit_residual_adjuster(points(),spec,FOLDS,[.75,.5])


def test_synthetic_scenarios_exact_keys_and_cost_pinball_identity(spec):
    cfg=p1.read_yaml(p1.ROOT/'configs/cost_scenarios.yaml')
    inputs=p1.P1Inputs(points(),points('test'),pd.DataFrame(),FOLDS,'synthetic',{})
    rows,cal=p1.evaluate_p1_synthetic(inputs,spec,cfg)
    assert len(rows)==len(inputs.test)*3 and len(cal)==3
    assert not rows.duplicated(p1.POINT_KEY+['tau_key']).any()
    for (_,tau,co,cu),g in rows.groupby(['scenario','tau','c_over','c_under']):
        np.testing.assert_allclose(g.simulated_cost,(co+cu)*pinball_loss(g.y_true,g.quantity,tau),rtol=1e-12,atol=0)
        assert len(g)==len(inputs.test)
    assert rows.quantity.ge(0).all() and rows.methodology_status.eq(p1.STATUS).all()


@pytest.mark.parametrize('kind',['real','unverified'])
def test_pending_method_blocks_real_policy_evaluation_before_calibration(spec,monkeypatch,kind):
    inputs=p1.P1Inputs(points(),points('test'),pd.DataFrame(),FOLDS,kind,{})
    monkeypatch.setattr(p1,'fit_residual_adjuster',lambda *a:pytest.fail('Real pending method reached residual fitting'))
    with pytest.raises(p1.MethodologyPendingError,match='PENDING_APPROVAL'):
        p1.evaluate_p1_synthetic(inputs,spec,{})


@pytest.mark.parametrize('kwargs',[dict(pooling=('date',)),dict(fallback=[()]),dict(fallback=(('city',),)),
                                  dict(quantile_method='auto'),dict(minimum_samples=0),dict(minimum_samples=True),
                                  dict(fallback=((),()))])
def test_method_choices_must_be_explicit_and_valid(kwargs):
    args=dict(pooling=('family',),quantile_method='linear',minimum_samples=2,fallback=())
    args.update(kwargs)
    with pytest.raises(ValueError):p1.ResidualSpec(**args)
