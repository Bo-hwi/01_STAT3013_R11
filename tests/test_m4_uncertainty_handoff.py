"""Non-training integration checks; bootstrap fixture is entirely synthetic."""
import json
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from src.decision import uncertainty_handoff as u
from test_m4_fixed_week_bootstrap import fixture, result


@pytest.fixture
def inputs(result):
    intervals=result['intervals'];support=result['calendar_support'];totals=result['point_totals']
    matrix=[];costs=[]
    for s in support.itertuples():
        nested=[]
        for scenario in u.fixed.SPEC['scenarios']:
            row=totals.loc[totals.scope.eq('family_city_pooled')&totals.family.eq(s.family)&totals.city.eq(s.city)&totals.scenario.eq(scenario)&totals.variant.eq('corrected')].iloc[0]
            means={p:row[p+'_total']/row.n_observations for p in u.fixed.POLICIES}
            nested.append(dict(scenario=scenario,mean_cost=means,cost_index={p:means[p]/means['P1'] for p in means},combined_descriptive_direction='Mixed'))
        row={c:'unchanged' for c in u.canonical.MCOLS}
        row.update(family=s.family,city=s.city,sample_size=s.sample_size,zero_rate=.5,
                   cost_scenarios=u.dump(nested),confidence_interval=u.dump({'status':'old'}),recommendation='Further validation')
        matrix.append(row)
    for r in totals.itertuples():
        level='pooled' if r.scope=='pooled_F1_F4' else 'fold' if r.scope in u.fixed.SPEC['folds'] else 'family_city'
        for cfg in ['M0','M3']:
            means={p:getattr(r,p+'_total')/r.n_observations for p in u.fixed.POLICIES}
            means['P4']=means['P4_'+cfg]
            costs.append(dict(config=cfg,scenario=r.scenario,variant=r.variant,report_level=level,
                fold=r.scope if level=='fold' else '',family=r.family,city=r.city,
                **{'mean_cost_'+p:means[p] for p in ['P1','P2','P3','P4']},
                **{'cost_index_'+p:means[p]/means['P1'] for p in ['P1','P2','P3','P4']},
                P2P3_relative_delta_CI95_lower_pct=np.nan,P2P3_relative_delta_CI95_upper_pct=np.nan,
                CI_estimand='old sensitivity',P1_comparison_CI='UNAVAILABLE',P4_comparison_CI='UNAVAILABLE'))
    cost=pd.DataFrame(costs)
    extra=cost.loc[cost.report_level.eq('pooled')].copy();extra['report_level']='family';extra['family']='A'
    return pd.DataFrame(matrix,columns=u.canonical.MCOLS),pd.concat([cost,extra],ignore_index=True),intervals,support


def test_matrix_exact_schema_own_cell_ci_point_estimates_and_labels_preserved(inputs):
    base,_,ci,support=inputs;before=base.copy(deep=True)
    out=u.integrate_matrix(base,ci,support)
    assert list(out)==u.canonical.MCOLS and out.recommendation.eq('Further validation').all()
    for row in out.itertuples():
        evidence=json.loads(row.confidence_interval);assert len(evidence['estimates'])==66
        assert evidence['grain']=='family_city_pooled'
        for r in evidence['estimates']:
            assert r['scope']=='family_city_pooled' and r['confidence_level']==.95
            assert r['sampling_scheme']=='fixed_calendar_week_paired_within_fold'
            assert r['multiple_testing_control'] is False
            if row.city=='Complete': assert r['lower'] is not None and r['valid_replicates']==1000
            else: assert r['lower'] is None and r['upper'] is None and r['resamples_attempted']==0
    pd.testing.assert_frame_equal(base,before)


@pytest.mark.parametrize('problem',['duplicate','missing_metric','missing_variant','missing_cell','pooled_as_cell','wrong_level','bad_count','false_complete','null_support'])
def test_invalid_uncertainty_panels_rejected(inputs,problem):
    _,_,ci,support=inputs;ci=ci.copy();support=support.copy()
    if problem=='duplicate':ci=pd.concat([ci,ci.iloc[[0]]],ignore_index=True)
    elif problem=='missing_metric':ci=ci.drop(index=0)
    elif problem=='missing_variant':ci=ci.loc[ci.variant.eq('corrected')]
    elif problem=='missing_cell':ci=ci.loc[~ci.city.eq('Complete')]
    elif problem=='pooled_as_cell':ci.loc[ci.city.eq('Complete'),'scope']='pooled_F1_F4'
    elif problem=='wrong_level':ci.loc[0,'confidence_level']=.9
    elif problem=='bad_count':ci.loc[0,'valid_replicates']=999
    elif problem=='false_complete':support.loc[support.city.eq('Partial'),'eligible']=True
    else:support.loc[0,'calendar_flag']=None
    with pytest.raises(ValueError):u.validate_intervals(ci,support)


def test_incomplete_calendar_never_receives_numeric_ci(inputs):
    base,_,ci,support=inputs;ci=ci.copy()
    ci.loc[ci.city.eq('Partial'),['lower','upper']]=0.
    with pytest.raises(ValueError,match='Unsupported CI'):u.integrate_matrix(base,ci,support)


def test_source_points_and_exact_cell_keys_must_match(inputs):
    base,_,ci,support=inputs;bad=base.copy()
    values=json.loads(bad.loc[0,'cost_scenarios']);values[0]['mean_cost']['P3']+=1.
    bad.loc[0,'cost_scenarios']=u.dump(values)
    with pytest.raises(ValueError,match='point estimate'):u.integrate_matrix(bad,ci,support)
    bad=base.copy();bad.loc[0,'city']='Wrong'
    with pytest.raises(ValueError,match='key mismatch'):u.integrate_matrix(bad,ci,support)


def test_cost_integration_preserves_all_point_columns_separates_units_and_legacy(inputs):
    _,base,ci,support=inputs;before=base.copy(deep=True)
    out=u.integrate_costs(base,ci,support)
    assert len(out)==len(base)
    pd.testing.assert_frame_equal(out.filter(regex='^(mean_cost|cost_index)_'),base.filter(regex='^(mean_cost|cost_index)_'))
    assert 'legacy_moving_block_sensitivity__CI_estimand' in out
    unsupported=out.report_level.eq('family')
    assert all(json.loads(s)['status']=='UNAVAILABLE_AT_THIS_GRAIN' for s in out.loc[unsupported,'fixed_week_primary_uncertainty'])
    record=json.loads(out.loc[0,'fixed_week_primary_uncertainty']);by_metric={r['metric']:r for r in record['estimates']}
    a=by_metric['P3_minus_P2:P1_normalized_difference'];b=by_metric['P3_minus_P2:relative_difference_pct']
    assert a['denominator']=='sampled_P1_cost' and a['units']=='P1_normalized_index'
    assert b['denominator']=='sampled_comparison_policy_cost' and b['units']=='percent'
    assert a['point_estimate']!=b['point_estimate']
    pd.testing.assert_frame_equal(base,before)


def test_proposed_wrapper_api_and_help_no_export_or_source_rewrite(tmp_path,monkeypatch,inputs):
    source=u.ROOT/'src/decision/decision_matrix.py';before=source.read_bytes()
    wrapper=tmp_path/'decision_matrix.py';wrapper.write_text(u.WRAPPER)
    namespace=runpy.run_path(str(wrapper),run_name='proposed_compatibility')
    assert namespace['build_decision_matrix'] is u.integrate_matrix
    base,_,ci,support=inputs
    out=namespace['build_decision_matrix'](base,ci,support);assert list(out)==u.canonical.MCOLS
    def forbidden(*a,**k):raise AssertionError('Help must not write/verify sources')
    monkeypatch.setattr(u,'prepare',forbidden);monkeypatch.setattr(sys,'argv',[str(wrapper),'--help'])
    with pytest.raises(SystemExit) as e:runpy.run_path(str(wrapper),run_name='__main__')
    assert e.value.code==0
    assert b'final_handoff' in before
    assert source.read_bytes()==before


def test_compliance_keeps_methodology_and_acceptance_distinct():
    path=u.ROOT/u.PHASE4C/'compliance.csv'
    if not path.is_file():pytest.skip('Local historical Phase4C compliance bundle not hydrated')
    old=pd.read_csv(path);out=u.updated_compliance(old).set_index('id')
    assert out.loc['R11','status']=='PASS'
    assert out.loc['R12','status']=='PARTIAL' and '14 cells' in out.loc['R12','methodological_limitation']
    assert out.loc['R02','status']=='PARTIAL' and 'PENDING' in out.loc['R02','team_acceptance']
    assert out.loc['R21','technical_status']=='PASS_PROPOSED_WRAPPER_ONLY'
    assert 'No supported cell' not in out.loc['R12','evidence']


def test_manuscript_replaces_superseded_primary_uncertainty_without_changing_methods():
    path=u.ROOT/u.PHASE4C/'canonical/reports/manuscript/04_rq2_decision_value.md'
    if not path.is_file():pytest.skip('Local historical Phase4C manuscript not hydrated')
    old=path.read_text(encoding='utf-8')
    out=u.revise_historical_methods(old)
    assert 'không có cell/P1/P4 intervals' not in out
    assert 'four blocks/fold, 1,000 common paired draws' in out
    assert "HistGradientBoostingRegressor(loss='quantile')" in out
    assert '48 selection fits + 24 fresh' in out


def test_integrity_rejects_corruption_path_escape_and_failed_runs(tmp_path):
    with pytest.raises(FileNotFoundError):u.inspect_files(tmp_path)
    (tmp_path/'a').write_text('ok');u.fixed.write_json(tmp_path/'COMPLETE.json',{'files':{'a':u.fixed.file_hash(tmp_path/'a')}})
    assert u.inspect_files(tmp_path)
    (tmp_path/'a').write_text('bad')
    with pytest.raises(ValueError,match='corrupt'):u.inspect_files(tmp_path)
    u.fixed.write_json(tmp_path/'FAILED.json',{'message':'preserved'})
    assert u.inspect_revision(tmp_path)=='failed'
