"""Small synthetic economic panels only; no estimator fitting."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from src.decision import policy_evaluation as pe
from src.decision.quantile_pipeline import prediction_rows
from src.decision.quantile_models import pinball_loss
from src.forecasting.rolling_validation import read_yaml


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    monkeypatch.setattr(HistGradientBoostingRegressor, 'fit', lambda *a, **k: pytest.fail('Cost evaluation must never fit models'))


@pytest.fixture
def config():
    return read_yaml(pe.ROOT/'configs/cost_scenarios.yaml')


@pytest.fixture
def panel():
    rows = pd.DataFrame(dict(date=pd.date_range('2016-09-01', periods=3), store_nbr=['1','1','2'],
                             family=['A','A','B'], city=['X','X','Y'], sales=[0.,1.25,4.5]))
    return pd.concat([
        prediction_rows(rows, np.array([[-1.,.25,1.],[.5,2.,1.5],[4.,5.,6.]]), 'F1','M0','test','synthetic_costs'),
        prediction_rows(rows, np.array([[-.5,.5,.75],[.75,1.75,2.25],[3.75,4.5,5.5]]), 'F1','M3','test','synthetic_costs'),
    ], ignore_index=True)


def test_matched_fractional_costs_and_identity(panel, config):
    rows, summary = pe.evaluate_policies(panel, config, 'synthetic_eval')
    assert len(rows)==36 and set(rows.policy)=={'P2','P3'}
    assert set(rows.variant)=={'corrected','raw_clipped_sensitivity'}
    assert rows.quantity.ge(0).all() and rows.quantity.mod(1).ne(0).any()
    for (_,tau,co,cu),g in rows.groupby(['variant','tau','c_over','c_under']):
        np.testing.assert_allclose(g.simulated_cost, (co+cu)*pinball_loss(g.y_true,g.quantity,tau),rtol=1e-12,atol=0)
    raw=rows.loc[rows.source_prediction.lt(0)]
    assert raw.quantity.eq(0).all() and raw.clipped_at_zero.all()
    assert (summary.result_label==pe.LABEL).all()
    assert 'cost_index' not in summary


def test_raw_and_corrected_sensitivities_preserve_distinct_quantiles(panel, config):
    rows,_=pe.evaluate_policies(panel,config,'synthetic_variants')
    g=rows.loc[rows.policy.eq('P2') & rows.date.eq('2016-09-02') & rows.tau_key.eq('tau_0.75')]
    assert g.set_index('variant').quantity.to_dict()=={'corrected':1.5,'raw_clipped_sensitivity':2.}


def test_validation_rows_never_enter_economic_evaluation(panel, config):
    decoy=panel.copy();decoy['split']='validation';decoy['date']-=pd.Timedelta(days=28)
    # Keep raw/corrected/truth internally consistent; dates/origin must agree.
    decoy['forecast_origin']=decoy.date-pd.Timedelta(days=1)
    rows,_=pe.evaluate_policies(pd.concat([decoy,panel],ignore_index=True),config,'synthetic_test_only')
    assert len(rows)==36 and rows.date.min()==pd.Timestamp('2016-09-01')


@pytest.mark.parametrize('fault',['duplicate','missing_tau','missing_config','mismatched_truth','null_city'])
def test_bad_key_panels_fail(panel,config,fault):
    if fault=='duplicate':panel=pd.concat([panel,panel.iloc[:1]],ignore_index=True)
    elif fault=='missing_tau':panel=panel.iloc[1:]
    elif fault=='missing_config':panel=panel.loc[~(panel.config.eq('M3') & panel.date.eq('2016-09-01'))]
    elif fault=='mismatched_truth':panel.loc[panel.config.eq('M3'),'y_true']+=1
    else:panel.loc[0,'city']=None
    with pytest.raises(ValueError):pe.evaluate_policies(panel,config,'invalid')


def test_expected_rows_detect_observations_missing_from_both_configs(panel,config):
    expected=panel.loc[panel.config.eq('M0') & panel.tau_key.eq('tau_0.50'),['date','store_nbr','family','city','fold','split']]
    missing=panel.loc[panel.date.ne('2016-09-01')]
    with pytest.raises(ValueError,match='expected scored'):
        pe.evaluate_policies(missing,config,'missing_both',expected_rows=expected)


def test_zero_denominator_is_explicit_nan(config):
    frame=pd.DataFrame(dict(date=pd.to_datetime(['2016-09-01']),store_nbr=['1'],family=['A'],city=['X'],sales=[0.]))
    panel=pd.concat([prediction_rows(frame,np.zeros((1,3)),'F1','M0','test','zero'),
                     prediction_rows(frame,np.ones((1,3)),'F1','M3','test','zero')],ignore_index=True)
    _,summary=pe.evaluate_policies(panel,config,'zero_cost')
    assert summary.p2_denominator_zero.all()
    assert summary.relative_delta_pct.isna().all() and summary.cost_ratio_P3_over_P2.isna().all()
    assert summary.delta_mean_cost_P3_minus_P2.gt(0).all()


def test_config_nonunit_scales_are_used_not_hardcoded(panel,config):
    baseline,_=pe.evaluate_policies(panel,config,'base')
    changed=deepcopy(config)
    for scenario in changed['scenarios']:
        scenario['c_over']*=2;scenario['c_under']*=2
    scaled,_=pe.evaluate_policies(panel,changed,'scaled')
    np.testing.assert_allclose(scaled.simulated_cost,2*baseline.simulated_cost)


@pytest.mark.parametrize('fault',['duplicate_id','wrong_tau','negative_cost','missing_scenario'])
def test_inconsistent_scenario_contract_fails(config,fault):
    if fault=='duplicate_id':config['scenarios'][1]['id']=config['scenarios'][0]['id']
    elif fault=='wrong_tau':config['scenarios'][1]['tau']=.5
    elif fault=='negative_cost':config['scenarios'][0]['c_over']=-1
    else:config['scenarios'].pop()
    with pytest.raises(ValueError):pe.configured_scenarios(config)


def test_pooled_delta_uses_sums_on_matched_rows(panel,config):
    rows,summary=pe.evaluate_policies(panel,config,'synthetic_ratios')
    for r in summary.loc[summary.report_level.eq('pooled')].itertuples():
        g=rows.loc[rows.scenario.eq(r.scenario) & rows.variant.eq(r.variant)]
        a=g.loc[g.policy.eq('P2'),'simulated_cost'];b=g.loc[g.policy.eq('P3'),'simulated_cost']
        assert r.sample_size==3
        assert r.mean_cost_P2==pytest.approx(a.mean())
        assert r.relative_delta_pct==pytest.approx(100*(b.sum()/a.sum()-1))


def test_source_must_be_complete_before_reading_data(tmp_path,monkeypatch):
    monkeypatch.setattr(pe,'inspect_run',lambda *a:{'status':'incomplete'})
    monkeypatch.setattr(pe.pd,'read_csv',lambda *a,**k:pytest.fail('Incomplete source was read'))
    with pytest.raises(ValueError,match='incomplete'):pe.load_verified_source(tmp_path,'synthetic')


def setup_writer(root,panel,config,monkeypatch):
    (root/'configs').mkdir();(root/'docs').mkdir();(root/'results/experiments/m4_quantiles_synthetic').mkdir(parents=True)
    import yaml
    (root/'configs/cost_scenarios.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    (root/'docs/m4_phase3a_costs.md').write_text('synthetic protocol',encoding='utf-8')
    source_path=root/'results/experiments/m4_quantiles_synthetic/manifest.json'
    source_path.write_text('{}',encoding='utf-8')
    source={'provenance':{'file_sha256':{str(source_path):pe.file_hash(source_path)},'input':{}}}
    monkeypatch.setattr(pe,'load_verified_source',lambda *a:(panel,source,{str(source_path):pe.file_hash(source_path)}))


def test_transaction_roundtrip_checksums_nonoverwrite(tmp_path,panel,config,monkeypatch):
    setup_writer(tmp_path,panel,config,monkeypatch)
    directory=pe.run_evaluation(tmp_path,'synthetic','synthetic_eval')
    assert pe.inspect_evaluation(directory)=='complete'
    marker=json.loads((directory/'COMPLETE.json').read_text())
    assert all(pe.file_hash(directory/n)==h for n,h in marker['files'].items())
    with pytest.raises(FileExistsError):pe.run_evaluation(tmp_path,'synthetic','synthetic_eval')
    (directory/'comparisons.csv').write_text('tampered',encoding='utf-8')
    assert pe.inspect_evaluation(directory)=='corrupt'


def test_failed_evaluation_remains_incomplete_and_blocks_reuse(tmp_path,panel,config,monkeypatch):
    setup_writer(tmp_path,panel,config,monkeypatch)
    monkeypatch.setattr(pe,'evaluate_policies',lambda *a:(_ for _ in ()).throw(ValueError('synthetic failure')))
    with pytest.raises(ValueError,match='synthetic failure'):pe.run_evaluation(tmp_path,'synthetic','failed_eval')
    stage,=(tmp_path/'results/experiments').glob('.m4_costs_failed_eval.staging-*')
    assert pe.inspect_evaluation(stage)=='failed' and (stage/'FAILED.json').exists()
    assert not (tmp_path/'results/experiments/m4_costs_failed_eval').exists()
    with pytest.raises(FileExistsError):pe.run_evaluation(tmp_path,'synthetic','failed_eval')
