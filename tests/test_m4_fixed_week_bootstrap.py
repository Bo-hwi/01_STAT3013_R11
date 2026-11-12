"""Synthetic/non-training checks for the approval-gated fixed-week supplement."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.decision import fixed_week_bootstrap as w


@pytest.fixture(scope='module')
def fixture():
    folds = [dict(fold_id=f'F{i+1}', test=[str(start.date()), str((start+pd.Timedelta(days=27)).date())])
             for i, start in enumerate(pd.to_datetime(['2016-09-01','2016-11-15','2017-03-01','2017-07-19']))]
    expected = []
    for f in folds:
        for d in pd.date_range(*f['test']):
            for store, city in [(1,'Complete'),(2,'Partial')]:
                if city == 'Partial' and d == pd.Timestamp(folds[1]['test'][0]): continue
                expected.append(dict(fold=f['fold_id'],date=d,store_nbr=store,family='A',city=city,y_true=10.,split='test'))
    expected = pd.DataFrame(expected)
    parts = []
    for scenario in ['S1','S3','S5']:
        for variant in w.VARIANTS:
            for i, policy in enumerate(w.POLICIES):
                x = expected.drop(columns='split').copy()
                x['scenario'],x['variant'],x['policy'] = scenario,variant,policy
                x['simulated_cost'] = (i+1.) * (1+(x.date.dt.day%7))
                parts.append(x)
    rows = pd.concat(parts,ignore_index=True)
    panel = w.align_costs(rows,expected,folds)
    return rows,expected,folds,panel


def test_exact_m3_draw_tensor_and_fold_week_boundaries(fixture):
    _,_,folds,panel = fixture
    assert np.array_equal(w.fixed_draws(),np.random.default_rng(42).integers(0,4,size=(1000,4,4)))
    weekly = w.weekly_totals(panel,folds)
    complete = weekly.loc[weekly.city.eq('Complete')]
    assert complete.n_dates.eq(7).all() and set(complete.block)=={0,1,2,3}
    assert w.fold_calendar(folds)['F1'][7] == pd.Timestamp('2016-09-08')


@pytest.mark.parametrize('problem',['duplicate','missing','target','negative','infinite','null','unknown_policy','raw_p4'])
def test_alignment_rejects_invalid_sources(fixture,problem):
    rows,expected,folds,_ = fixture
    x = rows.copy()
    if problem=='duplicate': x=pd.concat([x,x.iloc[[0]]],ignore_index=True)
    elif problem=='missing': x=x.drop(index=0)
    elif problem=='target': x.loc[0,'y_true']=11.
    elif problem=='negative': x.loc[0,'simulated_cost']=-1.
    elif problem=='infinite': x.loc[0,'simulated_cost']=np.inf
    elif problem=='null': x.loc[0,'city']=None
    elif problem=='unknown_policy': x.loc[0,'policy']='P5'
    elif problem=='raw_p4':
        idx=x.index[x.variant.eq('raw_clipped_sensitivity') & x.policy.eq('P4_M0')][0]
        x.loc[idx,'simulated_cost']+=1
    with pytest.raises(ValueError): w.align_costs(x,expected,folds)


def test_common_missing_observation_is_detected_against_frozen_expected(fixture):
    rows,expected,folds,_=fixture
    x=rows.loc[~(rows.city.eq('Complete') & rows.date.eq(pd.Timestamp('2016-09-01')))]
    with pytest.raises(ValueError): w.align_costs(x,expected,folds)


def test_date_gaps_never_compressed_or_reindexed(fixture):
    _,_,folds,panel=fixture
    support=w.calendar_support(panel,folds).set_index('city')
    assert support.loc['Complete','eligible'] and not support.loc['Partial','eligible']
    assert support.loc['Partial','n_dates']==111
    weekly=w.weekly_totals(panel,folds)
    first=weekly.loc[weekly.city.eq('Partial') & weekly.fold.eq('F2') & weekly.block.eq(0)]
    assert first.n_dates.eq(6).all()
    bad=deepcopy(folds);bad[0]['test'][1]='2016-09-27'
    with pytest.raises(ValueError):w.fold_calendar(bad)


def test_paired_weighting_counts_and_resampled_denominator():
    # Unequal block counts: ratio of sums, not mean of block/row ratios.
    costs=np.array([[10,20,30,40,50],[90,90,180,180,270]],float)
    stats=w.statistics(costs,np.array([1,9]))
    assert np.array_equal(stats['P2:P1_cost_index'][0],[2,1])
    combined=w.statistics(costs.sum(axis=0),np.array(10))
    assert combined['P2:P1_cost_index'][0]==1.1
    assert combined['P3_minus_P2:mean_cost_difference'][0]==10
    assert combined['P3_minus_P2:P1_normalized_difference'][0]==1
    assert not np.isclose(combined['P2:P1_cost_index'][0],1.5)


def test_zero_denominators_and_sign_change_null():
    stats=w.statistics(np.zeros((3,5)),np.ones(3))
    assert np.isnan(stats['P2:P1_cost_index'][0]).all()
    result=w.interval_summary(stats['P2:P1_cost_index'][0],np.nan,1.)
    assert result['undefined_replicates']==3 and result['valid_replicates']==0
    assert result['ci_status']=='UNDEFINED_DENOMINATOR'
    result=w.interval_summary(np.array([.8,1.,1.2]),1.1,1.)
    assert result['sign_change_fraction']==pytest.approx(1/3)
    assert result['includes_null']
    assert np.isnan(w.interval_summary(np.array([-1,0,1]),0)['sign_change_fraction'])


@pytest.fixture(scope='module')
def result(fixture): return w.bootstrap_supplement(fixture[3],fixture[2])


def test_cell_ci_exclusion_point_retention_and_reproducibility(fixture,result):
    ci=result['intervals'];bad=ci.loc[ci.city.eq('Partial')]
    assert not bad.empty and bad.point_estimate.notna().all()
    assert bad[['lower','upper']].isna().all().all()
    assert bad.resamples_attempted.eq(0).all() and bad.ci_status.eq('INCOMPLETE_CALENDAR_CI_UNAVAILABLE').all()
    good=ci.loc[ci.city.eq('Complete')]
    assert good.valid_replicates.eq(1000).all()
    again=w.bootstrap_supplement(fixture[3],fixture[2])
    pd.testing.assert_frame_equal(ci,again['intervals'])
    # Paired proportional policies remain exact on every draw, every scope.
    indices=good.loc[good.metric.eq('P2:P1_cost_index')]
    assert np.allclose(indices.lower,2) and np.allclose(indices.upper,2)
    assert len(set(ci.metric))==22


def test_exact_sampled_mean_with_unequal_day_counts(fixture):
    panel=fixture[3].copy();folds=fixture[2]
    # Partial cell has one fewer day in week zero; pooled mean uses drawn n.
    out=w.bootstrap_supplement(panel,folds)
    weekly=out['weekly_totals'].query("scenario == 'S1' and variant == 'corrected' and fold == 'F2'")
    blocks=weekly.groupby('block')[list(w.POLICIES)+['n']].sum().sort_index()
    idx=out['draws'][:,1,:]
    totals=blocks[list(w.POLICIES)].to_numpy()[idx].sum(axis=1)
    counts=blocks.n.to_numpy()[idx].sum(axis=1)
    expected=np.quantile((totals[:,2]-totals[:,1])/counts,[.025,.975])
    row=out['intervals'].query("scope == 'F2' and scenario == 'S1' and variant == 'corrected' and metric == 'P3_minus_P2:mean_cost_difference'").iloc[0]
    assert np.allclose([row.lower,row.upper],expected)


def test_approval_failure_stops_before_loading_or_resampling(tmp_path,monkeypatch):
    receipt=tmp_path/'pending.json';receipt.write_text(json.dumps({'status':'PENDING_APPROVAL'}))
    def forbidden(*a,**k):raise AssertionError('Must stop before execution')
    monkeypatch.setattr(w,'make_request',forbidden);monkeypatch.setattr(w,'fixed_draws',forbidden)
    with pytest.raises(ValueError,match='PENDING_APPROVAL'):w.run_approved(receipt,tmp_path)


def test_receipt_digest_hash_environment_scope_guards(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_text('frozen')
    req=dict(status=w.STATUS,specification=w.SPEC,file_sha256={str(source):w.file_hash(source)},environment={'x':1})
    receipt=dict(status='APPROVED',request=req,request_sha256=w.semantic_hash(req),approved_by='User',approval_evidence='Explicit authorization',approval_communicated_date='2026-11-09')
    monkeypatch.setattr(w,'make_request',lambda root:req)
    assert w.validate_approval(receipt,tmp_path)==req
    bad=deepcopy(receipt);bad['request_sha256']='wrong'
    with pytest.raises(ValueError,match='digest'):w.validate_approval(bad,tmp_path)
    monkeypatch.setattr(w,'make_request',lambda root:{**req,'environment':{'x':2}})
    with pytest.raises(ValueError,match='drift'):w.validate_approval(receipt,tmp_path)
    source.write_text('changed')
    with pytest.raises(ValueError,match='Frozen file drift'):w.validate_approval(receipt,tmp_path)


def test_prepare_request_never_draws(fixture,tmp_path,monkeypatch):
    _,_,folds,panel=fixture
    for relative in ['src/decision/fixed_week_bootstrap.py','tests/test_m4_fixed_week_bootstrap.py','docs/m4_phase4e_fixed_week.md','configs/cost_scenarios.yaml','configs/research_protocol.yaml']:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    verified={'evidence':{}}
    monkeypatch.setattr(w,'load_verified_costs',lambda root:(panel,folds,w.calendar_support(panel,folds),verified))
    monkeypatch.setattr(w,'current_branch',lambda root:'feature/m4-quantile-decision')
    monkeypatch.setattr(w.points,'reference_bytes',lambda *a:b'pinned fixture')
    def forbidden():raise AssertionError('Request preparation must not sample')
    monkeypatch.setattr(w,'fixed_draws',forbidden)
    request=w.make_request(tmp_path)
    assert request['status']==w.STATUS and request['execution_authorized'] is False


def test_wrong_branch_stops_before_source_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'current_branch',lambda root:'main')
    def forbidden(*a):raise AssertionError('Do not load on main')
    monkeypatch.setattr(w,'load_verified_costs',forbidden)
    with pytest.raises(ValueError,match='feature/m4'):w.make_request(tmp_path)


def test_no_false_cell_interval_for_missing_scenarios_or_changed_truth(fixture):
    panel=fixture[3].copy();folds=fixture[2]
    with pytest.raises(ValueError,match='scope'):w.bootstrap_supplement(panel.loc[panel.scenario.ne('S5')],folds)
    panel.loc[panel.index[-1],'y_true']=99.
    with pytest.raises(ValueError,match='keys/targets'):w.bootstrap_supplement(panel,folds)


def test_transactional_bundle_nonoverwrite_integrity_failure(tmp_path,result,monkeypatch):
    dest=tmp_path/'new';req={'file_sha256':{}}
    w.save_bundle(result,{},req,dest,1.)
    assert w.inspect_bundle(dest)=='complete'
    with pytest.raises(FileExistsError):w.save_bundle(result,{},req,dest,1.)
    (dest/'point_totals.csv').write_text('corrupt')
    assert w.inspect_bundle(dest)=='corrupt'
    def fail(*a,**k):raise OSError('disk failure')
    monkeypatch.setattr(pd.DataFrame,'to_csv',fail)
    with pytest.raises(OSError,match='disk failure'):w.save_bundle(result,{},req,tmp_path/'failed',1.)
    stages=list(tmp_path.glob('.failed.staging-*'))
    assert len(stages)==1 and w.inspect_bundle(stages[0])=='failed'
    assert not (tmp_path/'failed').exists()
