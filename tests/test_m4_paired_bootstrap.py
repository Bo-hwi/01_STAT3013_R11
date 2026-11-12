"""Synthetic fixed cost panels and resampling only. No estimator fitting."""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from src.decision import paired_bootstrap as bs
from src.decision.quantile_models import canonical_tau_key


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    monkeypatch.setattr(HistGradientBoostingRegressor,'fit',lambda *a,**k:pytest.fail('Bootstrap must never fit models'))


@pytest.fixture
def rows():
    data=[]
    for f,fold in enumerate(['F1','F2']):
        for d,date in enumerate(pd.date_range('2016-09-01',periods=14)[f*7:(f+1)*7]):
            for store in range(1,2+d%3):
                for scenario,tau,scale in [('S1',.5,1),('S3',.75,2),('S5',5/6,3)]:
                    for variant in ['corrected','raw_clipped_sensitivity']:
                        for policy in ['P2','P3']:
                            base=10*(d+1)+store+f
                            change=[-2,1,-3,2,0,-1,2][d] if policy=='P3' else 0
                            cost=scale*(base+change+(.5 if variant!='corrected' else 0))
                            data.append(dict(date=date,store_nbr=str(store),family='A',city='X',fold=fold,
                                             scenario=scenario,tau_key=canonical_tau_key(tau),variant=variant,
                                             policy=policy,y_true=1.5+store,simulated_cost=float(cost)))
    return pd.DataFrame(data)


@pytest.mark.parametrize('fault',['duplicate','missing_policy_key','demand','missing_series_key','null_key'])
def test_paired_alignment_rejects_bad_panels(rows,fault):
    if fault=='duplicate':rows=pd.concat([rows,rows.iloc[:1]],ignore_index=True)
    elif fault=='missing_policy_key':rows=rows.iloc[1:]
    elif fault=='demand':rows.loc[rows.policy.eq('P3'),'y_true']+=1
    elif fault=='missing_series_key':rows=rows.loc[~(rows.scenario.eq('S1')&rows.variant.eq('corrected')&rows.date.eq('2016-09-01'))]
    else:rows.loc[0,'city']=None
    with pytest.raises(ValueError):bs.pair_daily_costs(rows)


def test_same_observed_demand_required_across_scenarios(rows):
    rows.loc[rows.scenario.eq('S1'),'y_true']+=1
    with pytest.raises(ValueError,match='Demand differs'):bs.pair_daily_costs(rows)


def test_blocks_cannot_cross_calendar_gaps():
    dates=pd.to_datetime(['2016-01-01','2016-01-02','2016-01-03','2016-01-08','2016-01-09','2016-01-10'])
    index=bs.sample_date_indices(dates,2,50,np.random.default_rng(42))
    assert index.shape==(50,6)
    assert np.isin(index[:,:3],[0,1,2]).all() and np.isin(index[:,3:],[3,4,5]).all()
    for first in [0,3]:
        assert np.all(np.diff(index[:,first:first+2],axis=1)==1)
    with pytest.raises(ValueError,match='shorter'):bs.sample_date_indices(dates,4,5,np.random.default_rng(42))


@pytest.mark.parametrize('dates',[[pd.Timestamp('2016-01-01')]*2,
                                 pd.to_datetime(['2016-01-02','2016-01-01']),
                                 pd.to_datetime(['2016-01-01 12:00']),pd.to_datetime([None])])
def test_invalid_calendars_fail(dates):
    with pytest.raises(ValueError):bs.sample_date_indices(dates,1,5,np.random.default_rng(42))


def test_reproducible_and_sensitivity_order_invariant(rows):
    daily=bs.pair_daily_costs(rows)
    a=bs.bootstrap_costs(daily,[2,3],40,42)
    b=bs.bootstrap_costs(daily,[2,3],40,42)
    pd.testing.assert_frame_equal(a[0],b[0]);pd.testing.assert_frame_equal(a[1],b[1])
    c=bs.bootstrap_costs(daily,[3,2],40,42)
    for key in a[2]:np.testing.assert_array_equal(a[2][key],c[2][key])
    d=bs.bootstrap_costs(daily,[2],40,43)
    assert not np.array_equal(a[2]['L2_F1'],d[2]['L2_F1'])


def test_fold_boundaries_and_whole_day_pairing(rows):
    daily=bs.pair_daily_costs(rows)
    intervals,reps,indices,layout=bs.bootstrap_costs(daily,[2],40,42)
    assert len(layout)==14 and set(indices)=={'L2_F1','L2_F2'}
    for fold in ['F1','F2']:
        idx=indices['L2_'+fold];assert idx.shape==(40,7) and idx.min()>=0 and idx.max()<7
        # Both folds are adjacent in this synthetic calendar; the index arrays
        # still refer only to their own seven days, not the pooled 14-day path.
        g=reps.loc[reps.scope.eq(fold)&reps.scenario.eq('S1')&reps.variant.eq('corrected')]
        base=daily.loc[daily.fold.eq(fold)&daily.scenario.eq('S1')&daily.variant.eq('corrected')].sort_values('date')
        expected=(base.cost_P3.to_numpy()[idx].sum(axis=1)-base.cost_P2.to_numpy()[idx].sum(axis=1))/base.n.to_numpy()[idx].sum(axis=1)
        np.testing.assert_allclose(g.mean_cost_delta_P3_minus_P2,expected)
        # Same draws across scenarios preserve known proportional cost changes.
        s3=reps.loc[reps.scope.eq(fold)&reps.scenario.eq('S3')&reps.variant.eq('corrected')]
        np.testing.assert_allclose(s3.mean_cost_delta_P3_minus_P2,2*g.mean_cost_delta_P3_minus_P2)
    pooled=reps.loc[reps.scope.eq('pooled_F1_F4')&reps.scenario.eq('S1')&reps.variant.eq('corrected')]
    f1=reps.loc[reps.scope.eq('F1')&reps.scenario.eq('S1')&reps.variant.eq('corrected')]
    f2=reps.loc[reps.scope.eq('F2')&reps.scenario.eq('S1')&reps.variant.eq('corrected')]
    n=f1.sampled_observations.to_numpy()+f2.sampled_observations.to_numpy()
    expected=(f1.mean_cost_delta_P3_minus_P2.to_numpy()*f1.sampled_observations.to_numpy()+f2.mean_cost_delta_P3_minus_P2.to_numpy()*f2.sampled_observations.to_numpy())/n
    np.testing.assert_allclose(pooled.mean_cost_delta_P3_minus_P2,expected)


def test_unequal_day_counts_use_ratio_of_sums():
    daily=pd.DataFrame(dict(fold='F1',date=pd.date_range('2016-01-01',periods=3),scenario='S1',variant='corrected',
                            cost_P2=[10.,60.,60.],cost_P3=[12.,63.,66.],n=[1,3,2]))
    intervals,reps,idx,_=bs.bootstrap_costs(daily,[1],50,42)
    point=intervals.loc[intervals.scope.eq('F1')&intervals.metric.eq('mean_cost_delta_P3_minus_P2'),'point_estimate'].iloc[0]
    assert point==pytest.approx(11/6) and point!=pytest.approx(2.)
    index=idx['L1_F1'];expected=(daily.cost_P3.to_numpy()[index].sum(axis=1)-daily.cost_P2.to_numpy()[index].sum(axis=1))/daily.n.to_numpy()[index].sum(axis=1)
    np.testing.assert_allclose(reps.loc[reps.scope.eq('F1'),'mean_cost_delta_P3_minus_P2'],expected)
    assert reps.sampled_observations.nunique()>1


def test_identical_policies_zero_difference_not_spurious_sign_flip(rows):
    lookup=rows.loc[rows.policy.eq('P2')].set_index(bs.KEY).simulated_cost
    mask=rows.policy.eq('P3');rows.loc[mask,'simulated_cost']=lookup.reindex(pd.MultiIndex.from_frame(rows.loc[mask,bs.KEY])).to_numpy()
    result=bs.bootstrap_costs(bs.pair_daily_costs(rows),[2],30,42)[0]
    assert result.point_estimate.eq(0).all() and result.lower.eq(0).all() and result.upper.eq(0).all()
    assert result.includes_zero.all() and result.opposite_point_sign_fraction.isna().all()


def test_zero_denominators_are_undefined_and_counted(rows):
    rows.loc[rows.policy.eq('P2'),'simulated_cost']=0
    result=bs.bootstrap_costs(bs.pair_daily_costs(rows),[2],30,42)[0]
    relative=result.loc[result.metric.eq('relative_cost_delta_pct')]
    assert relative.valid_resamples.eq(0).all() and relative.undefined_resamples.eq(30).all()
    assert relative.lower.isna().all() and relative.includes_zero.isna().all()
    assert result.loc[result.metric.eq('mean_cost_delta_P3_minus_P2'),'point_estimate'].gt(0).all()


def test_sign_mass_is_not_a_pvalue():
    r=bs.summarize_interval([-2,-1,0,1,2,np.nan],1,.95)
    assert r['negative_fraction']==.4 and r['positive_fraction']==.4 and r['zero_fraction']==.2
    assert r['opposite_point_sign_fraction']==.4 and r['undefined_resamples']==1 and r['includes_zero']


@pytest.mark.parametrize('kwargs',[{'block_lengths':[2,2]},{'block_lengths':[]},{'resamples':0},{'seed':-1},{'confidence':1}])
def test_invalid_design_fails(rows,kwargs):
    with pytest.raises(ValueError):bs.bootstrap_costs(bs.pair_daily_costs(rows),**kwargs)


def test_source_integrity_gate_before_bootstrap(tmp_path,monkeypatch):
    monkeypatch.setattr(bs.costs,'inspect_evaluation',lambda *a:'corrupt')
    monkeypatch.setattr(bs,'bootstrap_costs',lambda *a:pytest.fail('Corrupt source reached bootstrap'))
    with pytest.raises(ValueError,match='not complete'):bs.run_bootstrap(tmp_path)


def test_bundle_checksums_and_failure_marker(tmp_path):
    required=['STARTED.json','daily_costs.csv','intervals.csv','replicates.csv','date_layout.csv','resampling_indices.npz']
    for name in required:(tmp_path/name).write_bytes(b'synthetic bytes')
    hashes={name:bs.file_hash(tmp_path/name) for name in required}
    (tmp_path/'manifest.json').write_text(json.dumps({'output_sha256':hashes}))
    (tmp_path/'COMPLETE.json').write_text(json.dumps({'files':{**hashes,'manifest.json':bs.file_hash(tmp_path/'manifest.json')}}))
    assert bs.inspect_bootstrap(tmp_path)=='complete'
    (tmp_path/'daily_costs.csv').write_text('changed')
    assert bs.inspect_bootstrap(tmp_path)=='corrupt'
    (tmp_path/'FAILED.json').write_text('{}')
    assert bs.inspect_bootstrap(tmp_path)=='failed'


def synthetic_source(tmp_path,rows,monkeypatch):
    from src.forecasting.rolling_validation import read_yaml
    directory=tmp_path/'results/experiments'/bs.SOURCE;directory.mkdir(parents=True)
    rows.to_csv(directory/'row_costs.csv',index=False)
    source_hash={str(directory/'row_costs.csv'):bs.file_hash(directory/'row_costs.csv')}
    manifest=dict(file_sha256=source_hash,source_forecast_provenance={'file_sha256':{},'input':{}},
                  row_cost_records=len(rows),scenarios=read_yaml(bs.ROOT/'configs/cost_scenarios.yaml'))
    (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (tmp_path/'docs').mkdir();(tmp_path/'docs/m4_phase3b_bootstrap.md').write_text('synthetic method',encoding='utf-8')
    monkeypatch.setattr(bs.costs,'inspect_evaluation',lambda *a:'complete')
    monkeypatch.setattr(bs,'DESIGN',{**bs.DESIGN,'block_lengths':[2],'primary_block_length':2,'resamples':20})
    return directory


def test_synthetic_roundtrip_source_preservation_and_no_overwrite(tmp_path,rows,monkeypatch):
    from types import SimpleNamespace
    from datetime import datetime,timezone
    source=synthetic_source(tmp_path,rows,monkeypatch)
    before={p:bs.file_hash(p) for p in source.iterdir()}
    class FixedClock:
        @staticmethod
        def now(tz):return datetime(2026,1,1,tzinfo=timezone.utc)
    monkeypatch.setattr(bs,'datetime',FixedClock)
    monkeypatch.setattr(bs,'uuid4',lambda:SimpleNamespace(hex='syntheticfixedid'))
    destination=bs.run_bootstrap(tmp_path)
    assert bs.inspect_bootstrap(destination)=='complete'
    saved={p:bs.file_hash(p) for p in destination.iterdir()}
    with pytest.raises(FileExistsError):bs.run_bootstrap(tmp_path)
    assert all(bs.file_hash(p)==h for p,h in saved.items())
    assert all(bs.file_hash(p)==h for p,h in before.items())
    assert not (destination/'FAILED.json').exists()


def test_failure_preserves_staging_evidence(tmp_path,rows,monkeypatch):
    synthetic_source(tmp_path,rows,monkeypatch)
    monkeypatch.setattr(bs,'bootstrap_costs',lambda *a:(_ for _ in ()).throw(ValueError('synthetic resampling failure')))
    with pytest.raises(ValueError,match='synthetic resampling failure'):bs.run_bootstrap(tmp_path)
    stage,=(tmp_path/'results/experiments').glob('.m4_bootstrap_p2_p3_*.staging-*')
    assert (stage/'STARTED.json').exists() and (stage/'FAILED.json').exists()
    assert bs.inspect_bootstrap(stage)=='failed'
