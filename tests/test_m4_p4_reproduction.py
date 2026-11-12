"""Non-training tests: real estimator fitting forbidden; fakes audit orchestration."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from test_m4_quantile_pipeline import features, contracts, small_thread_budget
from src.decision import p4_reproduction as p4
from src.decision import p4_calibration as calibration
from src.decision import p1_residual_policy as p1
from src.forecasting.rolling_validation import add_scoring_mask


@pytest.fixture(autouse=True)
def no_forecaster_training(monkeypatch):
    for cls in [Ridge,ExtraTreesRegressor,HistGradientBoostingRegressor]:
        monkeypatch.setattr(cls,'fit',lambda *a,**k:pytest.fail('Real forecasting fit forbidden'))


@pytest.fixture
def reference():
    return p4.reference_functions()


def fake_reference(reference, log):
    reference=dict(reference)
    def factory(model,parameters,prep):
        class Fake:
            def fit(self,x,y):
                log.append(dict(model=model,parameters=deepcopy(parameters),prep=prep,x=x.copy(),y=y.copy(),instance=self))
                return self
            def predict(self,x):return np.full(x.shape[0],-1.)
        return Fake()
    reference['estimator']=factory
    return reference


def test_pinned_functions_and_exact_original_grids(reference,contracts):
    assert [len(reference['GRIDS'][m]) for m in p4.MODELS]==[3,2,2]
    assert reference['GRIDS']['Ridge']==[{'alpha':1.},{'alpha':100.},{'alpha':10000.}]
    assert [p['min_samples_leaf'] for p in reference['GRIDS']['ExtraTrees']]==[10,30]
    assert [p['max_leaf_nodes'] for p in reference['GRIDS']['HistGradientBoosting']]==[15,31]
    for config in p4.CONFIGS:
        numeric,categorical=p4.feature_columns(config,contracts[1])
        assert numeric==p4.m0_numeric()+([] if config=='M0' else p4.FEATURES['weather'][:2])
        assert categorical==['store_nbr','family','city']
        assert not set(numeric)&{'sales','onpromotion','transactions','precipitation_sum'}
    with pytest.raises(ValueError):p4.feature_columns('M4',contracts[1])


def test_reference_hash_mismatch_fails_before_source_execution(monkeypatch):
    monkeypatch.setattr(p4.subprocess,'check_output',lambda *a,**k:b'corrupt')
    with pytest.raises(ValueError,match='source mismatch'):p4.reference_functions()


def small_points(n=4):
    return pd.DataFrame(dict(x=[np.nan,2.,4.,6.][:n],store_nbr=['1']*n,family=['A']*n,city=['X']*n,sales=[3.5]*n))


@pytest.mark.parametrize('model',p4.MODELS)
def test_preprocessing_missing_unseen_categories_and_estimator_params(reference,model):
    train=small_points();test=small_points(2);test['store_nbr']='UNSEEN';test['x']=1e9
    prep=reference['preprocess'](model,['x']);prep.fit(train)
    before=prep.named_transformers_['numeric'].named_steps['imputer'].statistics_.copy()
    transformed=prep.transform(test)
    np.testing.assert_array_equal(before,[4.])
    np.testing.assert_array_equal(before,prep.named_transformers_['numeric'].named_steps['imputer'].statistics_)
    estimator=reference['estimator'](model,reference['GRIDS'][model][0],prep)
    if model=='Ridge':
        assert estimator.solver=='lsqr' and estimator.tol==1e-6 and estimator.max_iter==5000
        assert 'scaler' in prep.named_transformers_['numeric'].named_steps
    elif model=='ExtraTrees':
        assert estimator.n_jobs==2 and estimator.random_state==42
        assert (transformed[:,prep.output_indices_['categorical'].start]==-1).all()
    else:
        assert estimator.early_stopping is False and estimator.random_state==42
        cat=prep.output_indices_['categorical']
        assert estimator.categorical_features==list(range(cat.start,cat.stop))
        assert np.isnan(transformed[:,cat.start]).all()


def test_deterministic_ties_clip_before_selection_and_train_only(reference):
    log=[];train=small_points();validation=small_points(2)
    winner,records=p4.select_on_validation(train,validation,['x'],fake_reference(reference,log))
    assert winner['model']=='Ridge' and winner['candidate']==0 and winner['parameters']=='{"alpha": 1.0}'
    assert len(log)==7 and len(records)==7
    assert all(r['validation_MAE']==3.5 for r in records)
    assert sum(r['selected_candidate'] for r in records)==3
    assert winner['raw'].tolist()==[-1.,-1.]
    assert all(np.array_equal(r['y'],train.sales) for r in log)


@pytest.fixture
def fake_run(features,contracts,reference):
    log=[];result=p4.reproduce(features,*contracts,reference=fake_reference(reference,log))
    return result,log


def test_shared_masks_keys_fit_budget_fresh_refit_and_precision(fake_run,contracts):
    (panel,tuning,expected,d),log=fake_run
    assert d['fits']==16 and d['selection_fits']==14 and d['refits']==2
    assert len(log)==16 and len({id(r['instance']) for r in log})==16
    assert panel.result_label.eq(p4.LABEL).all() and panel.raw_prediction.eq(-1).all() and panel.clipped_prediction.eq(0).all()
    assert panel.groupby(['fold','config']).model.first().eq('Ridge').all()
    assert panel.groupby('config').size().nunique()==1
    assert tuning.groupby(['fold','config']).size().eq(7).all()
    p4.validate_forecasts(panel,expected,contracts[0]['rq2']['validation']['test_folds'])
    assert log[7]['prep'] is not log[0]['prep'] and len(log[7]['y'])>len(log[0]['y'])
    assert log[15]['prep'] is not log[8]['prep']
    assert all(r['x'].dtype==np.float32 for r in log)


@pytest.mark.parametrize('fault',['duplicate','missing','truth','city','model','clip','stage','null','date'])
def test_prediction_contract_failures(fake_run,contracts,fault):
    (panel,_,expected,_),_=fake_run;panel=panel.copy()
    if fault=='duplicate':panel=pd.concat([panel,panel.iloc[[0]]],ignore_index=True)
    elif fault=='missing':panel=panel.drop(index=0)
    elif fault=='truth':panel.loc[0,'y_true']+=1
    elif fault=='city':panel.loc[0,'city']='WRONG'
    elif fault=='model':panel.loc[0,'model']='ExtraTrees'
    elif fault=='clip':panel.loc[0,'clipped_prediction']=1
    elif fault=='stage':panel.loc[0,'fit_stage']='fresh_train_validation_refit'
    elif fault=='null':panel.loc[0,'city']=None
    else:panel.loc[0,'date']=pd.Timestamp('2099-01-01')
    with pytest.raises(ValueError):p4.validate_forecasts(panel,expected,contracts[0]['rq2']['validation']['test_folds'])


def test_test_targets_not_used_for_selection_or_refit(features,contracts,reference,monkeypatch):
    first=p4.reproduce(features,*contracts,reference=fake_reference(reference,[]))
    original=p4.iter_folds
    def changed(*args,**kwargs):
        for fold in original(*args,**kwargs):
            test=fold.test.copy();test['sales']=test.sales+1e6
            yield replace(fold,test=test)
    monkeypatch.setattr(p4,'iter_folds',changed)
    second=p4.reproduce(features,*contracts,reference=fake_reference(reference,[]))
    pd.testing.assert_frame_equal(first[1],second[1])
    np.testing.assert_array_equal(first[0].raw_prediction,second[0].raw_prediction)


def test_published_comparison_reports_discrepancies(fake_run):
    (panel,tuning,_,_),_=fake_run
    compared,tests=p4.published_comparison(tuning.copy(),panel)
    assert len(compared)==14 and len(tests)==2
    assert compared.delta_validation_MAE.ne(0).all() and not tests.parameters_match.all()


def test_pending_and_drifted_reproduction_gate_before_preflight(tmp_path,monkeypatch):
    path=tmp_path/'pending.json';path.write_text(json.dumps({'status':'PENDING_APPROVAL'}))
    monkeypatch.setattr(p4,'production_preflight',lambda *a:pytest.fail('Unapproved run reached preflight'))
    with pytest.raises(ValueError,match='approval required'):p4.run_approved(path,tmp_path)
    request={'status':'PENDING_APPROVAL','file_sha256':{}}
    receipt=dict(status='APPROVED',request=request,request_sha256=p4.semantic_hash(request))
    monkeypatch.setattr(p4,'approval_request',lambda root:{'status':'different'})
    with pytest.raises(ValueError,match='drifted'):p4.validate_approval(receipt,tmp_path)


def test_p4_gate_rejects_p1_receipt_and_pending_proposal():
    for record in [{'status':'PENDING_APPROVAL'},dict(status='APPROVED',request={'status':'PROPOSED_NOT_APPROVED','methodology':{'models':['SeasonalNaive']}})]:
        with pytest.raises(ValueError):calibration.validate_approval(record)


def test_raw_validation_residual_but_clipped_test_base_and_target_blindness(fake_run,contracts):
    (panel,_,_,_),_=fake_run
    q,_=calibration._calibrate(panel,contracts[0]['rq2']['validation']['test_folds'])
    assert q.point_prediction.eq(0).all() and q.residual_quantile.gt(1).all()
    assert q.methodology_status.eq('PENDING_APPROVAL').all()
    validation=panel.loc[panel.split.eq('validation')]
    for (fold,model,config,family),g in validation.groupby(['fold','model','config','family']):
        actual=q.loc[q.fold.eq(fold)&q.model.eq(model)&q.config.eq(config)&q.family.eq(family)]
        for tau,part in actual.groupby('tau'):
            pool=validation.loc[validation.fold.eq(fold)&validation.model.eq(model)&validation.config.eq(config)] if part.used_fallback.all() else g
            np.testing.assert_allclose(part.residual_quantile,np.quantile(pool.y_true-pool.raw_prediction,tau,method='linear'))
    changed=panel.copy();changed.loc[changed.split.eq('test'),'y_true']=1e9
    other,_=calibration._calibrate(changed,contracts[0]['rq2']['validation']['test_folds'])
    pd.testing.assert_frame_equal(q,other)


def test_exclusive_request_and_incomplete_detection(tmp_path):
    path=tmp_path/'request.json';p4._json(path,{'status':'PENDING_APPROVAL'})
    with pytest.raises(FileExistsError):p4._json(path,{})
    assert p4.inspect_bundle(tmp_path)=='incomplete'


def full_synthetic():
    protocol=p4.read_yaml(p4.ROOT/'configs/research_protocol.yaml');ablation=p4.read_yaml(p4.ROOT/'configs/ablation_M0_M5.yaml')
    data=pd.MultiIndex.from_product([pd.date_range('2016-06-01','2017-08-15'),[1,2],protocol['scope']['deep_study_families']],names=['date','store_nbr','family']).to_frame(index=False)
    data['city']='SYNTHETIC';data['sales']=10.5+data.date.dt.day;data['store_active']=1;data['store_zero_day']=0
    for column in p4.FEATURES['calendar']:data[column]=0
    data=p4.prepare_features(data,ready=False)
    data=add_scoring_mask(data,ablation['scoring_exclusions']).drop(columns=['excluded_closure','excluded_post_reopening'])
    for column in p4.FEATURES['weather']:data[column]=1.
    return data,protocol,ablation


def test_four_shared_folds_64_fake_fits(reference):
    data,protocol,ablation=full_synthetic()
    log=[];panel,tuning,expected,d=p4.reproduce(data,protocol,ablation,reference=fake_reference(reference,log))
    assert d['fits']==64 and d['selection_fits']==56 and d['refits']==8 and len(log)==64
    assert set(panel.fold)=={'F1','F2','F3','F4'} and len(tuning)==56
    assert tuning.groupby(['fold','config','model']).selected_candidate.sum().eq(1).all()
    assert not panel.duplicated(p4.KEY+['config']).any()


def test_non_tied_algorithm_and_candidate_selection(reference):
    ref=dict(reference)
    def factory(model,parameters,prep):
        # Winner depends exclusively on validation MAE; HGB is deliberately better.
        value={'Ridge':0.,'ExtraTrees':2.,'HistGradientBoosting':3.}[model]
        if model=='Ridge' and parameters['alpha']==100.:value=1.
        if model=='HistGradientBoosting' and parameters['max_leaf_nodes']==31:value=3.5
        class Fake:
            def fit(self,x,y):return self
            def predict(self,x):return np.full(x.shape[0],value)
        return Fake()
    ref['estimator']=factory
    winner,records=p4.select_on_validation(small_points(),small_points(2),['x'],ref)
    assert winner['model']=='HistGradientBoosting' and winner['candidate']==1 and winner['validation_MAE']==0
    assert next(r for r in records if r['model']=='Ridge' and r['selected_candidate'])['candidate']==1


@pytest.mark.parametrize('exercise_calibration',[False,True],ids=['bundle_only','calibration_gate'])
def test_approved_fake_runner_roundtrip(tmp_path,reference,monkeypatch,exercise_calibration):
    from scripts import m4_pilot
    monkeypatch.setattr(m4_pilot,'verify_candidate_bundle',lambda *a:{})
    data,protocol,ablation=full_synthetic()
    source_data=data.copy(deep=True);source_data['store_nbr']=source_data.store_nbr.astype('int16')
    _,_,expected,_=p4.reproduce(data,protocol,ablation,reference=fake_reference(reference,[]))
    configs=tmp_path/'configs';configs.mkdir()
    for name in ['research_protocol.yaml','ablation_M0_M5.yaml']:
        (configs/name).write_bytes((p4.ROOT/'configs'/name).read_bytes())
    bound=tmp_path/'binding.txt';bound.write_text('synthetic frozen evidence')
    request=dict(status='PENDING_APPROVAL',file_sha256={str(bound):p4.file_hash(bound),str(configs/'research_protocol.yaml'):p4.file_hash(configs/'research_protocol.yaml')})
    receipt=dict(status='APPROVED',request=request,request_sha256=p4.semantic_hash(request),
                 approved_by='SYNTHETIC TEST ONLY',approval_evidence='Not a real approval',approval_communicated_date='2026-11-09')
    path=tmp_path/'approved.json';path.write_text(json.dumps(receipt))
    refdir=tmp_path/'results/experiments'/('m4_quantiles_'+p4.FINAL);refdir.mkdir(parents=True)
    expected.to_csv(refdir/('m4_quantiles_'+p4.FINAL+'.csv'),index=False)
    records=[]
    for (fold,split),rows in expected.groupby(['fold','split']):
        records.append(dict(fold=fold,split=split,rows=len(rows),sha256=p4.scored_key_digest(rows)))
    (refdir/'manifest.json').write_text(json.dumps(dict(provenance={'file_sha256':{},'input':{}},scored_key_contract=records)))
    monkeypatch.setattr(p4,'approval_request',lambda root:request)
    monkeypatch.setattr(p4,'production_preflight',lambda *a:(source_data,{'hard_failures':[],'file_sha256':{}}))
    monkeypatch.setattr(p4,'inspect_run',lambda *a:{'status':'complete'})
    original=p4.reference_bytes
    monkeypatch.setattr(p4,'reference_bytes',lambda path,root=p4.ROOT:original(path,p4.ROOT))
    log=[];monkeypatch.setattr(p4,'reference_functions',lambda:fake_reference(reference,log))
    monkeypatch.setattr(p4,'production_preflight',lambda *a:(source_data,{'hard_failures':['synthetic failure'],'file_sha256':{}}))
    with pytest.raises(ValueError,match='Preflight hard failures'):p4.run_approved(path,tmp_path)
    assert not log
    monkeypatch.setattr(p4,'production_preflight',lambda *a:(source_data,{'hard_failures':[],'file_sha256':{}}))
    destination=p4.run_approved(path,tmp_path)
    assert p4.inspect_bundle(destination)=='complete' and len(log)==64
    manifest=json.loads((destination/'manifest.json').read_text())
    assert manifest['diagnostics']['fits']==64 and manifest['label']=='M4_REPRODUCED_M3'
    assert manifest['calibration_status']=='PENDING_APPROVAL'
    assert source_data.store_nbr.dtype==np.dtype('int16')
    if not exercise_calibration:
        (destination/'predictions.csv').write_text('tamper')
        assert p4.inspect_bundle(destination)=='corrupt'
        return
    pending=calibration.calibration_request(destination)
    with pytest.raises(ValueError):calibration.validate_approval({'status':'PENDING_APPROVAL','request':pending})
    cal_receipt=dict(status='APPROVED',request=pending,request_sha256=p4.semantic_hash(pending),
                     approved_by='SYNTHETIC TEST ONLY',approval_evidence='No real approval',approval_communicated_date='2026-11-09')
    cal_path=tmp_path/'calibration.approved.json';cal_path.write_text(json.dumps(cal_receipt))
    quantities,audit=calibration.apply_approved(cal_path)
    assert quantities.methodology_status.eq('APPROVED').all() and not audit.empty
    assert quantities.quantity.ge(0).all() and quantities.policy.eq('P4').all()
    (destination/'predictions.csv').write_text('tamper')
    assert p4.inspect_bundle(destination)=='corrupt'
    with pytest.raises(ValueError):calibration.validate_approval(cal_receipt)


def comparison_rows(ids):
    n=len(ids)
    return pd.DataFrame(dict(fold=['F1']*n,split=['test']*n,date=[pd.Timestamp('2016-09-01')]*n,
                             store_nbr=ids,family=['A']*n,city=['X']*n,y_true=np.arange(n)+1.25,
                             config=['M0']*n,raw_prediction=np.arange(n)+.123456789012345))


def test_mixed_object_int16_comparison_is_lossless_and_does_not_mutate():
    expected=comparison_rows(['1','54']);actual=comparison_rows(np.array([1,54],dtype=np.int16))
    before_expected=expected.copy(deep=True);before_actual=actual.copy(deep=True)
    with pytest.raises(ValueError,match='object and int16'):
        expected.merge(actual,on=p4.KEY,validate='1:1')
    p4.assert_matched_rows(expected,actual)
    normal=p4.comparison_copy(actual)
    pd.testing.assert_frame_equal(actual,before_actual);pd.testing.assert_frame_equal(expected,before_expected)
    pd.testing.assert_frame_equal(normal.drop(columns='store_nbr'),actual.drop(columns='store_nbr'))
    assert normal.store_nbr.tolist()==['1','54'] and actual.store_nbr.dtype==np.dtype('int16')


@pytest.mark.parametrize('value',[None,np.nan,pd.NA,np.inf,-np.inf,True,np.bool_(False),0,-1,1.5,
    '01',' 1','1 ','+1','1.0','1e0','١','',2**63,str(2**63),float(2**53+2),complex(1,0),['1']])
def test_invalid_ids_rejected_without_silent_coercion(value):
    frame=comparison_rows([value])
    with pytest.raises(ValueError):p4.comparison_copy(frame)


@pytest.mark.parametrize('value',[np.int16(1),1,'1',1.0,np.float32(1),np.int64(2**63-1),str(2**63-1)])
def test_exact_integer_identifiers_supported(value):
    result=p4.comparison_copy(comparison_rows([value]))
    assert result.store_nbr.iloc[0]==str(int(value))


@pytest.mark.parametrize('fault',['duplicate','alias_collision','missing','extra','target','null_city'])
def test_normalization_keeps_exact_one_to_one_contract(fault):
    expected=comparison_rows(['1','2']);actual=comparison_rows(np.array([1,2],dtype=np.int16))
    if fault=='duplicate':actual=pd.concat([actual,actual.iloc[[0]]],ignore_index=True)
    elif fault=='alias_collision':
        extra=actual.iloc[[0]].copy();extra['store_nbr']='1';actual=pd.concat([actual,extra],ignore_index=True)
    elif fault=='missing':actual=actual.iloc[:1]
    elif fault=='extra':actual=pd.concat([actual,comparison_rows([3])],ignore_index=True)
    elif fault=='target':actual.loc[0,'y_true']+=.0001
    else:actual.loc[0,'city']=None
    with pytest.raises(ValueError):p4.assert_matched_rows(expected,actual)


def test_postfit_validator_also_handles_numeric_expected_keys(fake_run,contracts):
    (panel,_,expected,_),_=fake_run
    numeric=expected.copy();numeric['store_nbr']=numeric.store_nbr.astype('int16')
    before=numeric.copy(deep=True)
    p4.validate_forecasts(panel,numeric,contracts[0]['rq2']['validation']['test_folds'])
    pd.testing.assert_frame_equal(numeric,before)
