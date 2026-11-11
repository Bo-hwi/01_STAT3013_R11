"""Versioned handoff integration of VERIFIED fixed-week intervals; no resampling."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from . import decision_matrix_canonical as canonical
from . import fixed_week_bootstrap as fixed

ROOT = fixed.ROOT
FIXED = 'results/experiments/m4_fixed_week_supplement_20261109T170740Z_96cd674e6b714482a62f46c8699c4417'
AUDIT = 'results/experiments/m4_fixed_week_execution_audit_20261109T171026Z_ec047f9afb1d42e09cbeb36c67f7af29'
PHASE4C = 'results/experiments/m4_phase4c_20261109T155831Z_1854a207899040fd88da572082e37252'
REQUEST_SHA = '81a6ebbe8dd0f9571702df04a7e8225f88ccd930af53cb4d774c1ad02c9d2ce5'
METRICS = {a+'_minus_'+b+':'+metric for a,b in fixed.CONTRASTS
           for metric in ['mean_cost_difference','P1_normalized_difference','relative_difference_pct']}
METRICS |= {p+':P1_cost_index' for p in fixed.POLICIES[1:]}
WRAPPER = '''"""PROPOSED installation only; not installed over frozen historical source."""
from src.decision.uncertainty_handoff import integrate_matrix as build_decision_matrix
from src.decision.uncertainty_handoff import main

if __name__ == '__main__':
    main()
'''


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def inspect_files(directory):
    directory = Path(directory).resolve()
    marker = json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    for name, digest in marker['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or fixed.file_hash(path) != digest:
            raise ValueError('Incomplete/corrupt evidence: '+str(path))
    return {str((directory/n).resolve()):d for n,d in marker['files'].items()}


def validate_intervals(intervals, support):
    required = ['scope','family','city','scenario','variant','metric','ci_eligible',
                'point_estimate','lower','upper','confidence_level','valid_replicates',
                'undefined_replicates','replicates_requested','resamples_attempted',
                'n_observations','denominator','null_reference','ci_status','sign_change_fraction']
    if intervals.empty or set(required)-set(intervals): raise ValueError('Missing fixed-week interval fields')
    if support.duplicated(['family','city']).any(): raise ValueError('Duplicate calendar support')
    if {'family','city','eligible','n_dates','fold_dates','calendar_flag'}-set(support.columns):
        raise ValueError('Missing calendar support fields')
    if support[['family','city','eligible','n_dates','fold_dates','calendar_flag']].isna().any().any():
        raise ValueError('Null calendar support')
    index = intervals.fillna({'family':'','city':''}).copy()
    if not index.ci_eligible.isin([True,False]).all() or not support.eligible.isin([True,False]).all():
        raise ValueError('Eligibility must be boolean')
    keys = ['scope','family','city','scenario','variant','metric']
    if index[keys].isna().any().any() or index.duplicated(keys).any(): raise ValueError('Duplicate/null CI keys')
    if set(index.metric) != METRICS: raise ValueError('Fixed-week metric/comparator scope differs')
    if not index.confidence_level.eq(.95).all() or not index.replicates_requested.eq(1000).all():
        raise ValueError('Wrong confidence/replicate specification')
    if not np.isfinite(index.point_estimate).all(): raise ValueError('Nonfinite fixed point estimate')
    expected_scopes = [('pooled_F1_F4','','')] + [(f,'','') for f in fixed.SPEC['folds']]
    expected_scopes += [('family_city_pooled',r.family,r.city) for r in support.itertuples()]
    actual = set(map(tuple,index[['scope','family','city']].to_numpy()))
    if actual != set(expected_scopes): raise ValueError('Unsupported or missing CI grain')
    lookup = support.set_index(['family','city'])
    for (scope,family,city,scenario,variant), g in index.groupby(keys[:-1],sort=False):
        if set(g.metric) != METRICS: raise ValueError('Incomplete contrast triplet/index metrics')
        if scenario not in fixed.SPEC['scenarios'] or variant not in fixed.VARIANTS: raise ValueError('Unknown scenario/variant')
        eligible = bool(lookup.loc[(family,city),'eligible']) if scope == 'family_city_pooled' else True
        if not g.ci_eligible.eq(eligible).all(): raise ValueError('CI/calendar eligibility mismatch')
        if eligible:
            if scope == 'family_city_pooled':
                row=lookup.loc[(family,city)]
                if row.n_dates != 112 or json.loads(row.fold_dates) != {f:28 for f in fixed.SPEC['folds']}:
                    raise ValueError('Eligible cell lacks complete four-fold calendar')
            if g[['lower','upper']].isna().any().any() or not np.isfinite(g[['lower','upper']]).all().all():
                raise ValueError('Missing/nonfinite supported CI')
            if g.lower.gt(g.upper).any() or not g.valid_replicates.eq(1000).all() or not g.undefined_replicates.eq(0).all():
                raise ValueError('Verified valid-replicate/bound contract changed')
            if not g.resamples_attempted.eq(1000).all() or not g.ci_status.eq('AVAILABLE').all(): raise ValueError('Invalid available CI metadata')
        else:
            if g[['lower','upper']].notna().any().any() or not g.valid_replicates.eq(0).all() or not g.resamples_attempted.eq(0).all():
                raise ValueError('Unsupported CI on incomplete-calendar cell')
            if not g.ci_status.eq('INCOMPLETE_CALENDAR_CI_UNAVAILABLE').all(): raise ValueError('Missing incomplete-calendar flag')
    for _, g in index.groupby(['scope','family','city','scenario','variant']):
        if len(g) != 22: raise ValueError('Incomplete fixed-week metric panel')
    if len(index) != len(expected_scopes)*3*2*22: raise ValueError('Missing scenario/variant panel')
    return index


def expected_point(metric, means):
    name, statistic = metric.split(':')
    if statistic == 'P1_cost_index':
        return means[name]/means['P1'] if means['P1'] else np.nan
    a,b = name.split('_minus_');difference = means[a]-means[b]
    if statistic == 'mean_cost_difference': return difference
    denominator = means['P1'] if statistic == 'P1_normalized_difference' else means[b]
    return (difference/denominator)*(100 if statistic == 'relative_difference_pct' else 1) if denominator else np.nan


def ci_record(row, means):
    expected = expected_point(row.metric, means)
    if not np.isclose(expected,row.point_estimate,rtol=1e-10,atol=1e-10,equal_nan=True):
        raise ValueError('Stored point estimate differs from canonical costs: '+row.metric)
    name, statistic = row.metric.split(':')
    comparator = 'P1' if statistic == 'P1_cost_index' else name.split('_minus_')[1]
    return dict(metric=row.metric, comparator=comparator, scope=row.scope, scenario=row.scenario,
        variant=row.variant, point_estimate=float(row.point_estimate),
        lower=float(row.lower) if pd.notna(row.lower) else None,
        upper=float(row.upper) if pd.notna(row.upper) else None,
        confidence_level=.95, sampling_scheme='fixed_calendar_week_paired_within_fold',
        block_days=7, blocks_per_fold=4, replicates_requested=1000, seed=42,
        valid_replicates=int(row.valid_replicates),undefined_replicates=int(row.undefined_replicates),
        resamples_attempted=int(row.resamples_attempted), denominator=row.denominator,
        null_reference=float(row.null_reference), status=row.ci_status,
        units='percent' if statistic == 'relative_difference_pct' else 'relative_cost_per_observation' if statistic == 'mean_cost_difference' else 'P1_normalized_index',
        source_bundle=FIXED, multiple_testing_control=False)


def ci_lookup(intervals):
    return {key:g for key,g in intervals.groupby(['scope','family','city','scenario','variant'],sort=False)}


def integrate_matrix(base, intervals, support):
    """Preserve exactly the 15 Roles fields and every economic/forecast point field."""
    if list(base) != canonical.MCOLS or base.duplicated(['family','city']).any() or base.isna().any().any():
        raise ValueError('Invalid canonical matrix schema')
    if set(map(tuple,base[['family','city']].to_numpy())) != set(map(tuple,support[['family','city']].to_numpy())):
        raise ValueError('Matrix/calendar key mismatch')
    validated=validate_intervals(intervals,support);lookup=ci_lookup(validated)
    out=base.copy(deep=True);calendar=support.set_index(['family','city'])
    for i,row in out.iterrows():
        record=calendar.loc[(row.family,row.city)];entries=[]
        for scenario in json.loads(row.cost_scenarios):
            key=('family_city_pooled',row.family,row.city,scenario['scenario'],'corrected')
            if key not in lookup: raise ValueError('Missing exact cell uncertainty')
            entries += [ci_record(r,scenario['mean_cost']) for r in lookup[key].itertuples()]
        out.at[i,'confidence_interval']=dump(dict(status='AVAILABLE_POINTWISE' if record.eligible else 'INCOMPLETE_CALENDAR_CI_UNAVAILABLE',
            grain='family_city_pooled',forecast_error_CI='UNAVAILABLE; economic cost CI is not pinball/MAE CI',
            calendar_flag=record.calendar_flag,n_dates=int(record.n_dates),fold_dates=json.loads(record.fold_dates),estimates=entries))
        out.at[i,'uncertainty_flag']=('pointwise_fixed_week_no_multiplicity_control' if record.eligible else 'incomplete_calendar_primary_CI_unavailable')+'; fixed_policy_only; four_blocks_per_fold; F1_exposed; F2_F4_exposure_unverified'
        out.at[i,'interpretation']='Further validation: research triage, not deployment. Fixed-week pointwise economic intervals do not change original descriptive directions. No forecast-error CI, preregistration or multiple-testing claim. Operational weather availability/prior exposure unresolved. M2 annotation is family context. P4=M4_REPRODUCED_M3.'
    protected=[c for c in canonical.MCOLS if c not in ['confidence_interval','uncertainty_flag','interpretation']]
    pd.testing.assert_frame_equal(out[protected],base[protected])
    if not out.recommendation.eq('Further validation').all(): raise ValueError('Do not retune recommendations')
    return out


def integrate_costs(base, intervals, support):
    canonical.validate_costs(base)
    validated=validate_intervals(intervals,support);lookup=ci_lookup(validated)
    # Retain old values under explicitly sensitivity-only names, never reinterpret them.
    legacy=[c for c in base if c.startswith(('P2P3_relative_delta_CI','CI_')) or c in ['P1_comparison_CI','P4_comparison_CI']]
    renames={c:'legacy_moving_block_sensitivity__'+c for c in legacy}
    out=base.rename(columns=renames).copy()
    keys=['scenario','variant','report_level','fold','family','city']
    paired=base.fillna({'fold':'','family':'','city':''}).set_index(keys+['config'])
    all_records=[]
    for row in base.fillna({'fold':'','family':'','city':''}).itertuples():
        scope='pooled_F1_F4' if row.report_level=='pooled' else row.fold if row.report_level=='fold' else 'family_city_pooled' if row.report_level=='family_city' else None
        if scope is None:
            all_records.append(dump(dict(status='UNAVAILABLE_AT_THIS_GRAIN',reason='No family-only/city-only/fold-cell CI approved; never borrow pooled intervals')));continue
        key=tuple(getattr(row,k) for k in keys)
        m0=paired.loc[key+('M0',)];m3=paired.loc[key+('M3',)]
        means=dict(P1=row.mean_cost_P1,P2=row.mean_cost_P2,P3=row.mean_cost_P3,P4_M0=m0.mean_cost_P4,P4_M3=m3.mean_cost_P4)
        group=lookup[(scope,row.family if scope=='family_city_pooled' else '',row.city if scope=='family_city_pooled' else '',row.scenario,row.variant)]
        all_records.append(dump(dict(status='AVAILABLE_POINTWISE' if group.ci_eligible.all() else 'INCOMPLETE_CALENDAR_CI_UNAVAILABLE',
            estimates=[ci_record(r,means) for r in group.itertuples()])))
    out['fixed_week_primary_uncertainty']=all_records
    pd.testing.assert_frame_equal(out[list(renames.get(c,c) for c in base)].rename(columns={v:k for k,v in renames.items()}),base)
    return out


def verify_evidence(root=ROOT):
    root=Path(root).resolve()
    if fixed.current_branch(root)!='feature/m4-quantile-decision': raise ValueError('Wrong branch')
    source=root/FIXED;audit=root/AUDIT;previous=root/PHASE4C
    if fixed.inspect_bundle(source)!='complete' or canonical.inspect_audit(previous)!='complete': raise ValueError('Incomplete source bundle')
    files={**inspect_files(source),**inspect_files(audit),**inspect_files(previous)}
    requirements=root.parent/'m4_requirements.md'
    if fixed.file_hash(requirements)!=fixed.file_hash(previous/'requirements_extract.md'):
        raise ValueError('Authoritative requirements changed; review before integration')
    files[str(requirements)]=fixed.file_hash(requirements)
    manifest=json.loads((source/'manifest.json').read_text());receipt=manifest['approval'];request=receipt['request']
    if receipt['status']!='APPROVED' or receipt['request_sha256']!=REQUEST_SHA or fixed.semantic_hash(request)!=REQUEST_SHA:
        raise ValueError('Fixed-week approval mismatch')
    if request['specification']!=fixed.SPEC or request['environment']!=fixed.environment(): raise ValueError('Specification/environment drift')
    for name,digest in request['file_sha256'].items():
        if fixed.file_hash(name)!=digest: raise ValueError('Frozen provenance drift: '+name)
    files.update(request['file_sha256'])
    summary=json.loads((audit/'verification.json').read_text())
    if summary['source_complete_sha256']!=fixed.file_hash(source/'COMPLETE.json') or summary['request_sha256']!=REQUEST_SHA:
        raise ValueError('Execution audit references a different supplement')
    ci=pd.read_csv(source/'intervals.csv',float_precision='round_trip')
    support=pd.read_csv(source/'calendar_support.csv');ci=validate_intervals(ci,support)
    audited=pd.read_csv(audit/'coarse_all_estimands.csv',float_precision='round_trip').fillna({'family':'','city':''})
    pd.testing.assert_frame_equal(ci.loc[ci.scope.ne('family_city_pooled')].reset_index(drop=True),audited.reset_index(drop=True),check_dtype=False)
    if len(support)!=154 or int(support.eligible.sum())!=140: raise ValueError('Frozen 140/14 calendar scope drift')
    with np.load(source/'draws.npz',allow_pickle=False) as archive:draws=archive['draws']
    # Compare exact expected tensor, not a new bootstrap analysis.
    if not np.array_equal(draws,np.random.default_rng(42).integers(0,4,size=(1000,4,4))): raise ValueError('Draw tensor differs from pinned M3')
    weekly=pd.read_csv(source/'weekly_totals.csv',float_precision='round_trip')
    for scope in ['pooled_F1_F4']+fixed.SPEC['folds']:
        g=weekly.loc[weekly.scenario.eq('S1')&weekly.variant.eq('corrected')];counts=np.zeros(1000)
        for fold in (fixed.SPEC['folds'] if scope=='pooled_F1_F4' else [scope]):
            b=g.loc[g.fold.eq(fold)].groupby('block').n.sum().sort_index()
            if list(b.index)!=[0,1,2,3]:raise ValueError('Wrong calendar block IDs')
            counts+=b.to_numpy()[draws[:,fixed.SPEC['folds'].index(fold),:]].sum(axis=1)
        if [int(counts.min()),int(counts.max())]!=summary['sampled_count_ranges'][scope]:raise ValueError('Sampled-count audit mismatch')
    # Canonical files must still match Phase4C exports before creating a revision.
    for relative in ['results/tables/cost_scenarios.csv','results/tables/decision_matrix.csv','results/predictions/quantiles.csv.gz','reports/manuscript/04_rq2_decision_value.md']:
        path=root/relative
        if fixed.file_hash(path)!=fixed.file_hash(previous/'canonical'/relative):raise ValueError('Unapproved canonical drift: '+relative)
        files[str(path)]=fixed.file_hash(path)
    return ci,support,files


def provenance_note(root):
    receipt=json.loads((root/'docs/m4_final_signoff.approved.json').read_text())
    return ('# Versioned quantile-model provenance note\n\nFrozen YAML remains `primary_quantile_model: TODO_choose_quantile_regressor`. This historical field requirement was not completed before execution and is not retroactively marked PASS.\n\n'
        +'Recorded pre-run evidence: `docs/m4_final_signoff.approved.json`; approval communicated '+receipt['approval_communicated_date']+'; receipt UTC '+receipt['approved_at_utc']+'.\n\n'
        +receipt['decisions']['estimator_grid']['resolution']+'\n\nActual implementation: HistGradientBoostingRegressor(loss="quantile"), recorded as HistGradientBoostingQuantile. This note documents existing approval, not a new approval or a model change. Team must accept the historical TODO discrepancy for Roles v4.2. No frozen config is changed.\n')


def updated_compliance(old):
    out=old.copy();out['phase4c_status']=out.status
    out['technical_status']=np.where(out.status.eq('PASS'),'PASS','DOCUMENTED')
    out['methodological_limitation']='';out['team_acceptance']='NOT_REQUIRED_FOR_TECHNICAL_CHECK'
    actions={
        'R02':('PARTIAL','PASS_PROVENANCE_NOTE','Historical YAML TODO; dated pre-run HGB approval exists','PENDING_HISTORICAL_EXCEPTION'),
        'R08':('PARTIAL','PASS_APPROVED_REPRODUCTION','P4 is M4_REPRODUCED_M3, not official M3','PENDING_ROLES_SUBSTITUTION_ACCEPTANCE'),
        'R10':('PASS','PASS','Pointwise fixed-policy intervals; four weekly blocks per fold','PENDING_REPORT_ACCEPTANCE'),
        'R11':('PASS','PASS','Pinned M3 sampling; cost/cell estimands explicitly extend MAE analysis','PENDING_REPORT_ACCEPTANCE'),
        'R12':('PARTIAL','PASS_SUPPORTED_140_CELLS','14 cells CI unavailable; no fold-cell, forecast-error or multiplicity-adjusted CI','PENDING_MISSING_EVIDENCE_ACCEPTANCE'),
        'R15':('PARTIAL','PASS_CONSERVATIVE_LABELS','Recommendation timing remains post-test; cannot repair preregistration','PENDING_CLAIMS_ACCEPTANCE'),
        'R21':('PARTIAL','PASS_PROPOSED_WRAPPER_ONLY','Frozen zero-byte entry point unchanged; tested wrapper proposal supplied','PENDING_ENTRY_POINT_MIGRATION_OR_ADAPTER_ACCEPTANCE'),
        'R22':('PARTIAL','PASS_MODULAR_IMPLEMENTATION','Training in quantile_pipeline.py; primitives in quantile_models.py','PENDING_LAYOUT_ACCEPTANCE'),
        'R23':('PARTIAL','PASS_MODULAR_IMPLEMENTATION','Economic orchestration outside cost_simulation.py','PENDING_LAYOUT_ACCEPTANCE'),
        'R28':('PARTIAL','PASS_M4_REPORT_PACKAGE','Joint report/IEEE/slides not integrated here','PENDING_TEAM_INTEGRATION'),
        'R29':('PARTIAL','PASS_LOCAL_ARTIFACTS','Drive delivery not evidenced; no upload or push performed','PENDING_GIT_DRIVE_DELIVERY'),
    }
    for id,(status,technical,limitation,acceptance) in actions.items():
        mask=out.id.eq(id);out.loc[mask,['status','technical_status','methodological_limitation','team_acceptance']]=[status,technical,limitation,acceptance]
        out.loc[mask,'corrective_action']=acceptance
    # Requirements now point at versioned revision paths, not unchanged Phase4C exports.
    revised=['R09','R10','R11','R12','R13','R14','R17','R18','R19','R20','R21','R25','R26']
    for id in revised:out.loc[out.id.eq(id),'evidence']='Phase4F manifest artifact map; '+out.loc[out.id.eq(id),'evidence'].iloc[0]
    new_evidence={
        'R10':FIXED+'/intervals.csv; canonical/results/tables/cost_scenarios.csv in this revision',
        'R11':FIXED+'/draws.npz; '+FIXED+'/manifest.json; pinned M3 b35f5cc cell 17',
        'R12':'uncertainty/calendar_support.csv; canonical/results/tables/decision_matrix.csv: 140 complete cells supported, 14 explicitly unavailable',
        'R02':'provenance/quantile_model_note.md; docs/m4_final_signoff.approved.json; unchanged configs/research_protocol.yaml:98',
        'R21':'compatibility/src/decision/decision_matrix.py (PROPOSAL ONLY); live frozen src/decision/decision_matrix.py unchanged',
    }
    for id,evidence in new_evidence.items():out.loc[out.id.eq(id),'evidence']=evidence
    out.loc[out.id.isin(['R18','R19','R20','R25']),'team_acceptance']='PENDING_VERSIONED_REVISION_ACCEPTANCE'
    return out


def revise_historical_methods(previous):
    """Replace superseded uncertainty statements, retaining frozen forecasting methods."""
    start=previous.index('P2/P3 uncertainty dùng')
    end=previous.index('[B]',start)+3
    replacement=('Primary economic uncertainty hiện dùng fixed non-overlapping seven-day calendar blocks, '
        'four blocks/fold, 1,000 common paired draws, seed 42, percentile 95% linear intervals. '
        'Sampled cost sums/counts and P1 denominators travel together. Six contrasts and P1 indices '
        'have pooled/per-fold CI; 140 complete-calendar family-city cells have their own pooled CI. '
        'Fourteen incomplete cells retain point estimates and CI unavailable. Forecast-error CI is '
        'not supplied. The older P2/P3 moving-block analysis (L7/5000, L3/14 sensitivity) remains '
        'a separate historical sensitivity analysis [B], never relabeled as fixed-week evidence. '
        'Fixed-week authorization was communicated by the user on 2026-11-10, after test exposure; '
        'not a preregistered forecasting design. [FW]')
    return previous[:start]+replacement+previous[end:]


def inspect_revision(directory):
    directory=Path(directory)
    if (directory/'FAILED.json').exists():return 'failed'
    try:
        inspect_files(directory)
        manifest=json.loads((directory/'manifest.json').read_text())
        marker=json.loads((directory/'COMPLETE.json').read_text())
        if manifest['output_sha256']!={n:h for n,h in marker['files'].items() if n!='manifest.json'}:return 'corrupt'
        return 'complete'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete_or_corrupt'


def prepare(root=ROOT,output_parent=None):
    root=Path(root).resolve();ci,support,evidence=verify_evidence(root)
    parent=Path(output_parent).resolve() if output_parent else root/'results/experiments'
    if not parent.is_relative_to(root):raise ValueError('Revision must remain inside workspace')
    dest=parent/('m4_phase4f_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+dest.name+'.staging-',dir=parent))
    for relative in ['src/decision/uncertainty_handoff.py','tests/test_m4_uncertainty_handoff.py','docs/m4_phase4f_integration.md']:
        evidence[str(root/relative)]=fixed.file_hash(root/relative)
    def text(relative,content):
        p=stage/relative;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:f.write(content)
    def csv(relative,frame):
        p=stage/relative;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
    try:
        base_matrix=pd.read_csv(root/'results/tables/decision_matrix.csv',keep_default_na=False,float_precision='round_trip')
        base_cost=pd.read_csv(root/'results/tables/cost_scenarios.csv',float_precision='round_trip')
        revised_matrix=integrate_matrix(base_matrix,ci,support);revised_cost=integrate_costs(base_cost,ci,support)
        csv('canonical/results/tables/decision_matrix.csv',revised_matrix)
        csv('canonical/results/tables/cost_scenarios.csv',revised_cost)
        m=pd.read_csv(stage/'canonical/results/tables/decision_matrix.csv',keep_default_na=False,float_precision='round_trip')
        c=pd.read_csv(stage/'canonical/results/tables/cost_scenarios.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(m,revised_matrix,check_dtype=False);pd.testing.assert_frame_equal(c,revised_cost,check_dtype=False)
        canonical.handoff.export_copy(root/'results/predictions/quantiles.csv.gz',stage,'canonical/results/predictions/quantiles.csv.gz')
        expected=pd.read_csv(root/canonical.handoff.SOURCES['point_reproduction']/'expected_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
        quantiles=pd.read_csv(stage/'canonical/results/predictions/quantiles.csv.gz',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
        canonical.validate_quantiles(quantiles,expected.loc[expected.split.eq('test')],fixed.points.read_yaml(root/'configs/cost_scenarios.yaml'))
        csv('uncertainty/fixed_week_intervals.csv',ci);csv('uncertainty/calendar_support.csv',support)
        canonical.handoff.export_copy(root/PHASE4C/'crossing_counts.csv',stage,'tables/crossing_counts.csv')
        text('provenance/quantile_model_note.md',provenance_note(root))
        text('compatibility/src/decision/decision_matrix.py',WRAPPER)
        text('compatibility/INSTALLATION_PENDING.md','Do NOT install over the frozen zero-byte source. Team may accept src.decision.uncertainty_handoff as the versioned CLI/API, or approve a future source migration with archival code snapshots and version-aware historical validation. Merely copying the wrapper into the live path would invalidate current frozen-file checks. No wrapper installed.\n')
        old=pd.read_csv(root/PHASE4C/'compliance.csv');compliance=updated_compliance(old)
        csv('compliance.csv',compliance);text('COMPLIANCE.md',canonical.matrix.markdown(compliance))
        primary=ci.loc[ci.scope.eq('pooled_F1_F4')&ci.variant.eq('corrected')]
        table=primary.loc[primary.metric.isin(['P3_minus_P2:P1_normalized_difference','P4_M3_minus_P4_M0:P1_normalized_difference']),['scenario','metric','point_estimate','lower','upper','valid_replicates']]
        previous=(root/'reports/manuscript/04_rq2_decision_value.md').read_text(encoding='utf-8')
        # Preserve methods/results but replace superseded absence/readiness statements explicitly.
        previous=previous.split('## Robustness\n',1)[0]
        previous=revise_historical_methods(previous)
        previous=previous.replace('Tất cả recommendation là **Further validation** vì thiếu cell-level paired CI,\noperational availability chưa xác minh và test exposure unresolved.',
            'Tất cả recommendation giữ **Further validation**. Fixed-week pointwise economic CI có ở 140 cells; 14 cells thiếu calendar giữ CI unavailable. Operational availability và test exposure vẫn unresolved.')
        manuscript=previous+'\n## Robustness — Phase4F fixed-week integration\n\n'+canonical.matrix.markdown(table)+'\n\nPrimary intervals: paired fixed seven-day calendar blocks, four blocks/fold, 1,000 shared draws, seed 42, 95% linear percentile. P1-normalized difference resamples denominator with numerator; it is not relative percentage change versus P2. P1 policy-index CI and all six contrasts are retained by pooled/fold and eligible cell. Moving-block L7/5000 stays sensitivity only. 140 complete-calendar cells receive their OWN intervals; 14 Esmeraldas/Salinas cells retain point estimates with CI unavailable. No pooled CI is borrowed. Test raw quantile crossing triplets: M0=3,827; M3=3,745; corrected=0 for both. Full validation/test counts retained in tables/crossing_counts.csv.\n\n## Limitations\n\nFour blocks/fold, fixed policies, no selection/calibration/refit uncertainty or multiplicity correction. No forecast-error CI is inferred from cost CI. F1 previously exposed; F2-F4 prior test exposure remains unverified. Retrospective ERA5 availability; observed sales is censored demand proxy. Simulated costs/indices are not money, profit, ROI or causal/operational/conformal guarantees. P4 remains M4_REPRODUCED_M3 with non-independent validation calibration and train-only versus fresh-refit mismatch. Descriptive labels unchanged; no retrospective preregistration. Frozen primary-model TODO documented by pre-run HGB receipt, not edited.\n\n## Conclusions\n\nREADY_FOR_TEAM_REVIEW as a VERSIONED package; not unconditional Roles acceptance or deployment approval. Team acceptance checklist retains historical TODO, entry-point layout, M4 reproduction substitution, exposure verification and Git/Drive/IEEE/slides integration. Original canonical files are unchanged; migration requires explicit acceptance.\n'
        text('canonical/reports/manuscript/04_rq2_decision_value.md',manuscript)
        text('source_index.md','\n'.join('['+tag+'] '+canonical.handoff.SOURCES[name]+'/manifest.json' for tag,name in [('Q','quantiles'),('P1','p1'),('D','p2_p3'),('B','bootstrap'),('R','point_reproduction'),('P4','p4'),('M','matrix')])+'\n[FW] '+FIXED+'/manifest.json\n')
        for relative in ['src/decision/uncertainty_handoff.py','tests/test_m4_uncertainty_handoff.py','docs/m4_phase4f_integration.md']:
            canonical.handoff.export_copy(root/relative,stage,'provenance/source_snapshot/'+relative)
        figures=[]
        for p in sorted((root/'reports/figures').glob('cost_*.png')):
            evidence[str(p)]=fixed.file_hash(p);canonical.handoff.export_copy(p,stage,'canonical/reports/figures/'+p.name)
            figures.append(dict(figure=p.name,caption='Stored simulated costs/descriptive directions; unchanged point estimates. These panels do not display CIs. Fixed-week pointwise CIs are in versioned tables (140 complete cells only); moving-block intervals are sensitivity. No multiplicity, causal or deployment claim.'))
        csv('figure_captions.csv',pd.DataFrame(figures))
        text('TEAM_ACCEPTANCE_CHECKLIST.md',(root/'docs/m4_phase4f_integration.md').read_text(encoding='utf-8'))
        text('README.md','READY_FOR_TEAM_REVIEW — versioned canonical revision; no live migration.\n\nAuthoritative fixed-week source: '+FIXED+'\nAudit: '+AUDIT+'\n\nUse canonical/ subtree only after team accepts this revision. Historical source bundles remain authoritative for predictions/costs. New CI evidence does not change quantities, means, indices or directions. Exact 15-column matrix/10-column quantile schemas preserved. Proposed compatibility wrapper is tested but NOT installed.\n\nReproduce without fitting/calibration/cost/bootstrap reruns: python -B -m src.decision.uncertainty_handoff\n')
        artifacts=[dict(path=str(p.relative_to(stage)).replace('\\','/'),sha256=fixed.file_hash(p)) for p in stage.rglob('*') if p.is_file()]
        csv('artifact_map.csv',pd.DataFrame(artifacts))
        for p,d in evidence.items():
            if fixed.file_hash(p)!=d:raise ValueError('Protected source changed: '+p)
        outputs={str(p.relative_to(stage)).replace('\\','/'):fixed.file_hash(p) for p in stage.rglob('*') if p.is_file()}
        fixed.write_json(stage/'manifest.json',dict(readiness='READY_FOR_TEAM_REVIEW',team_acceptance='PENDING',
            source_bundles=[FIXED,AUDIT,PHASE4C],file_sha256=evidence,output_sha256=outputs,matrix_rows=len(m),cost_rows=len(c),quantile_rows=len(quantiles),
            eligible_cells=140,incomplete_cells=14,recommendations_unchanged=True,point_estimates_unchanged=True,
            canonical_live_migration=False,wrapper_installed=False,new_bootstrap_performed=False,forecasting_fits=0,
            compliance_counts=compliance.status.value_counts().to_dict()))
        fixed.write_json(stage/'COMPLETE.json',dict(files={**outputs,'manifest.json':fixed.file_hash(stage/'manifest.json')}))
        if inspect_revision(stage)!='complete':raise ValueError('Revision bundle integrity failure')
        if dest.exists():raise FileExistsError('Never overwrite revision')
        stage.rename(dest);return dest
    except Exception as e:
        fixed.write_json(stage/'FAILED.json',dict(type=type(e).__name__,message=str(e)));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent',type=Path,help='Existing directory within workspace; always creates a unique revision')
    args=parser.parse_args();print(prepare(output_parent=args.output_parent))


if __name__=='__main__':main()
