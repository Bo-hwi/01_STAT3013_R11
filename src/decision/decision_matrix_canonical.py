"""Roles-extract canonical adapters only; no models or economic reevaluation."""
import argparse
from datetime import datetime,timezone
import gzip
import io
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from . import m4_handoff as handoff
from . import consolidated_matrix as matrix
from .quantile_models import canonical_tau_key

ROOT=handoff.ROOT
REQUIREMENTS=ROOT.parent/'m4_requirements.md'
QCOLS=['date','store_nbr','family','city','fold','model','config','tau','y_true','y_pred']
MCOLS=['family','city','sample_size','zero_rate','weather_signature','baseline_error','weather_error',
       'delta_error_pct','confidence_interval','fold_consistency','cost_scenarios','uncertainty_flag',
       'source_QA_flag','recommendation','interpretation']
LABELS={'Prioritize pilot','Further validation','Do not deploy'}


def scenarios(config):return matrix.direct.configured_scenarios(config)


def canonical_quantiles(source,config):
    source=source.loc[source.split.eq('test')].copy()
    mapping={s['tau_key']:float(next(x['tau'] for x in config['scenarios'] if x['id']==s['scenario'])) for s in scenarios(config)}
    out=source[['date','store_nbr','family','city','fold','model','config','tau_key','y_true','corrected_prediction']].copy()
    out['tau']=out.tau_key.map(mapping);out['y_pred']=out.corrected_prediction
    return out[QCOLS]


def validate_quantiles(frame,expected,config):
    if list(frame)!=QCOLS or frame.empty or frame.isna().any().any():raise ValueError('Canonical quantile schema/null failure')
    if not np.isfinite(frame[['tau','y_true','y_pred']].to_numpy(float)).all() or frame.y_pred.lt(0).any():raise ValueError('Nonfinite/negative quantile quantities')
    normalized=frame.assign(tau_key=frame.tau.map(canonical_tau_key))
    keys=['date','store_nbr','family','city','fold','model','config']
    if normalized.duplicated(keys+['tau_key']).any():raise ValueError('Duplicate quantile keys')
    allowed=[s['tau_key'] for s in scenarios(config)]
    if set(normalized.tau_key)!=set(allowed) or normalized.groupby(keys).size().ne(3).any():raise ValueError('Incomplete canonical quantile triplets')
    wide=normalized.pivot(index=keys,columns='tau_key',values='y_pred').reindex(columns=allowed)
    if (np.diff(wide.to_numpy(),axis=1)<0).any():raise ValueError('Canonical quantile crossing')
    if normalized.groupby(keys).y_true.nunique().ne(1).any():raise ValueError('Target mismatch across taus')
    if set(frame.config)!={'M0','M3'}:raise ValueError('Canonical feature scope mismatch')
    for _,g in normalized.groupby(['config','tau_key']):
        g=g.assign(split='test');matrix.p4.reproduction.assert_matched_rows(expected,g)


def canonical_matrix(source,config):
    result=[];lookup={s['scenario']:s for s in scenarios(config)}
    for (family,city),g in source.groupby(['family','city'],sort=True):
        if set(g.scenario)!=set(lookup) or g.scenario.duplicated().any():raise ValueError('Missing/duplicate cell scenarios')
        if any(g[c].nunique(dropna=False)!=1 for c in ['sample_size','zero_rate','fold_coverage']):raise ValueError('Cell sample/target scope differs across scenarios')
        first=g.iloc[0];costs=[];errors={};weather={};delta={};folds={}
        for _,r in g.iterrows():
            s=lookup[r.scenario];tau=str(next(x['tau'] for x in config['scenarios'] if x['id']==r.scenario))
            errors[tau]=float(r.baseline_error);weather[tau]=float(r.weather_error);delta[tau]=float(r.delta_error_pct)
            costs.append(dict(scenario=r.scenario,c_over=s['c_over'],c_under=s['c_under'],tau_config=float(tau),tau_key=s['tau_key'],
                mean_cost={p:float(r['mean_cost_'+p]) for p in matrix.POLICIES},cost_index={p:float(r['cost_index_'+p]) for p in matrix.POLICIES},
                direction_P3_P2=r.direction_P3_minus_P2,direction_P4_weather=r.direction_P4_M3_minus_M0,combined_descriptive_direction=r.weather_class))
            folds[r.scenario]={tag:dict(favorable=int(r['favorable_folds_'+tag]),unfavorable=int(r['unfavorable_folds_'+tag]),zero=int(r['zero_folds_'+tag])) for tag in ['P3_minus_P2','P4_M3_minus_M0']}
        dump=lambda value:json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True)
        result.append(dict(family=family,city=city,sample_size=int(first.sample_size),zero_rate=float(first.zero_rate),
            weather_signature=first.m2_rq1_econometric_signature,
            baseline_error=dump(dict(metric='corrected M0 quantile pinball loss',by_tau_config=errors)),
            weather_error=dump(dict(metric='corrected M3 quantile pinball loss',by_tau_config=weather)),
            delta_error_pct=dump(dict(definition='100*(M3/M0-1)',by_tau_config=delta)),
            confidence_interval=dump(dict(status='UNAVAILABLE_AT_CELL_GRAIN',lower=None,upper=None,scope_note='P2/P3 CIs only pooled/fold; P1/P4/cell uncertainty missing')),
            fold_consistency=dump(dict(coverage=first.fold_coverage.split(','),by_scenario=folds)),cost_scenarios=dump(costs),
            uncertainty_flag='cell_CI_missing; P1_P4_uncertainty_unavailable; no_significance_or_multiplicity_claim',
            source_QA_flag=first.source_QA_flag,recommendation='Further validation',
            interpretation='Research triage only; no deployment authorization. Cell paired uncertainty absent; operational availability/prior exposure unverified. Direction classes retained in cost_scenarios. M2 signature is family context, not city evidence. Recommendation mapping declared Phase4C, not claimed pre-test.'))
    return pd.DataFrame(result,columns=MCOLS)


def validate_matrix(frame,families,cities):
    if list(frame)!=MCOLS or frame.duplicated(['family','city']).any() or frame.isna().any().any():raise ValueError('Canonical matrix fields/uniqueness/null failure')
    if set(map(tuple,frame[['family','city']].to_numpy()))!=set((f,c) for f in families for c in cities):raise ValueError('Canonical 154-cell scope failure')
    if not set(frame.recommendation).issubset(LABELS):raise ValueError('Invalid Roles recommendation')
    for _,r in frame.iterrows():
        ci=json.loads(r.confidence_interval)
        if ci['status']!='UNAVAILABLE_AT_CELL_GRAIN' or ci['lower'] is not None or ci['upper'] is not None:raise ValueError('Unsupported cell CI')
        if r.recommendation!='Further validation':raise ValueError('No cell-level evidence authorizes pilot/deploy classification')
        if len(json.loads(r.cost_scenarios))!=3:raise ValueError('Incomplete nested cost scenarios')


def stored_unit_means(frame,group_base,over_name,under_name):
    records=[]
    if not np.isfinite(frame[['overstock_units','understock_units']].to_numpy(float)).all():raise ValueError('Nonfinite stored unit metrics')
    for level,grouping in matrix.direct.LEVELS.items():
        names=group_base+grouping
        g=frame.groupby(names,observed=True)[['overstock_units','understock_units']].mean().reset_index()
        g=g.rename(columns={'overstock_units':over_name,'understock_units':under_name});g['report_level']=level;records.append(g)
    return pd.concat(records,ignore_index=True)


def canonical_costs(source,direct_comparisons,p1_units,p4_units,intervals,config):
    out=source.copy();keys=['scenario','variant','report_level','fold','family','city']
    out=out.merge(direct_comparisons[keys+['mean_over_P2','mean_under_P2','mean_over_P3','mean_under_P3']],on=keys,how='left',validate='m:1')
    p1=stored_unit_means(p1_units,['scenario'],'mean_over_P1','mean_under_P1')
    p4=stored_unit_means(p4_units,['scenario','config'],'mean_over_P4','mean_under_P4')
    for metrics,join in [(p1,['scenario','report_level','fold','family','city']),(p4,['scenario','config','report_level','fold','family','city'])]:
        out=out.merge(metrics,on=join,how='left',validate='m:1')
    if out.filter(regex='^mean_(over|under)_').isna().any().any():raise ValueError('Missing stored unit aggregates')
    out['weather_cost_index_delta_P3_minus_P2']=out.cost_index_P3-out.cost_index_P2
    a=out.loc[out.config.eq('M0'),keys+['cost_index_P4']].rename(columns={'cost_index_P4':'index_P4_M0'})
    b=out.loc[out.config.eq('M3'),keys+['cost_index_P4']].rename(columns={'cost_index_P4':'index_P4_M3'})
    pair=a.merge(b,on=keys,validate='1:1');pair['weather_cost_index_delta_P4_M3_minus_M0']=pair.index_P4_M3-pair.index_P4_M0
    out=out.merge(pair[keys+['weather_cost_index_delta_P4_M3_minus_M0']],on=keys,validate='m:1')
    ci=intervals.loc[intervals.block_length.eq(7)&intervals.metric.eq('relative_cost_delta_pct'),['scope','scenario','variant','lower','upper']]
    ci=ci.rename(columns={'scope':'CI_scope','lower':'P2P3_relative_delta_CI95_lower_pct','upper':'P2P3_relative_delta_CI95_upper_pct'})
    out['CI_scope']=np.where(out.report_level.eq('pooled'),'pooled_F1_F4',np.where(out.report_level.eq('fold'),out.fold,'UNSUPPORTED_AT_THIS_GRAIN'))
    out=out.merge(ci,on=['CI_scope','scenario','variant'],how='left',validate='m:1')
    out['CI_estimand']='100*(sum(P3)/sum(P2)-1); NOT policy/P1 index CI'
    out['CI_design_caveat']='M4 paired moving blocks L7/5000; differs from M3 fixed weekly blocks/1000'
    out['P1_comparison_CI']='UNAVAILABLE';out['P4_comparison_CI']='UNAVAILABLE'
    out['caveats']='simulated relative costs; no ROI/profit/causality; F1 exposed; F2-F4 exposure unresolved; P4 M4_REPRODUCED_M3; non-independent calibration'
    mapping={s['scenario']:s for s in scenarios(config)}
    out['c_over']=out.scenario.map(lambda x:mapping[x]['c_over']);out['c_under']=out.scenario.map(lambda x:mapping[x]['c_under'])
    out['tau_key']=out.scenario.map(lambda x:mapping[x]['tau_key'])
    validate_costs(out)
    return out


def validate_costs(frame):
    columns=['config','scenario','variant','report_level','fold','family','city']
    if frame.duplicated(columns).any():raise ValueError('Duplicate canonical cost summary keys')
    unsupported=~frame.report_level.isin(['pooled','fold'])
    if frame.loc[unsupported,['P2P3_relative_delta_CI95_lower_pct','P2P3_relative_delta_CI95_upper_pct']].notna().any().any():raise ValueError('Invalid CI attached to cell/subgroup')
    for p in ['P1','P2','P3','P4']:
        expected=frame.mean_cost_P1.where(frame.mean_cost_P1.ne(0))
        if not np.allclose(frame['mean_cost_'+p]/expected,frame['cost_index_'+p],rtol=1e-12,equal_nan=True):raise ValueError('Canonical stored cost-index mismatch')


def publish_exclusive(source,target):
    target=Path(target)
    if target.exists():
        if not target.is_file() or handoff.file_hash(target)!=handoff.file_hash(source):return 'BLOCKED_EXISTING_DIFFERENT_FILE'
        return 'REUSED_IDENTICAL_AFTER_INSPECTION'
    target.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as src,target.open('xb') as dst:handoff.shutil.copyfileobj(src,dst,1024*1024)
    if handoff.file_hash(target)!=handoff.file_hash(source):raise ValueError('Published export checksum failed')
    return 'CREATED_EXCLUSIVELY'


def compliance_rows(root,published,audit_path=None):
    rows=[]
    def add(id,requirement,authority,evidence,status,action='None',science=False):
        rows.append(dict(id=id,requirement=requirement,authoritative_source=authority,evidence=evidence,status=status,corrective_action=action,scientific_methodology_decision=science))
    add('R01','Read authoritative extract/config precedence','extract intro',str(REQUIREMENTS),'PASS')
    add('R02','Main algorithm fixed in primary_quantile_model BEFORE full run','extract §1; research_protocol.quantile_and_decision.primary_quantile_model','configs/research_protocol.yaml; docs/m4_final_signoff.approved.json','PARTIAL','TODO remains; dated HGB sign-off exists. Versioned protocol resolution/acknowledgement, not retroactive config edit.',True)
    add('R03','Exact taus and M0/M3/seven-family/fold scope','extract §1; frozen configs','canonical quantile export and authoritative quantile manifest','PASS')
    add('R04','No test-target tuning; train-only preprocessing; fresh refit; shared masks/folds','extract §2','quantile/point manifests and staged prediction schemas','PASS')
    add('R05','Q>=0, no crossing; report correction counts','extract §2','canonical gzip validation; crossing_counts.csv; raw source preserved','PASS')
    add('R06','P1 residual policy/SeasonalNaive validation','extract §3','P1 completed bundle; actual verified validation lag7 fallback counts zero','PASS')
    add('R07','P2/P3/P4 same scored test rows','extract §3','21 matched source cost groups and reconciliation checks','PASS')
    add('R08','Official M3/leader validation source for P4','extract §3/§8','M4_REPRODUCED_M3 completed bundle; explicit reproduction approvals','PARTIAL','Team accept/document approved M4 reproduction substitute; do not call official M3.',True)
    add('R09','Pinball/coverage/cost/indices/over-under by required grain','extract §4','authoritative metrics and stored-unit canonical cost table','PASS')
    add('R10','95% paired weekly bootstrap','extract §4; research_protocol.rq2.uncertainty','existing P2/P3 L7 percentile intervals, 5000 paired moving-block draws','PARTIAL','Pooled/fold P2/P3 only; obtain approved missing-grain/comparison design.',True)
    add('R11','Same bootstrap design as M3','cost_scenarios.evaluation.uncertainty; extract §4','M3 pinned fixed weekly blocks/1000 vs M4 moving blocks/5000','PARTIAL','Harmonize sampling design or approve versioned methodological deviation before new analysis.',True)
    add('R12','Family-city paired CI and P1/P4 comparison uncertainty','extract §4/§5/G4','No supported cell/P1/P4 intervals; explicit unavailable fields','MISSING','Predeclare additional paired analyses and cell calendar feasibility; do not launch yet.',True)
    add('R13','One row per family-city, all 15 matrix fields','extract §5; research_protocol.outputs.decision_matrix_required_columns','results/tables/decision_matrix.csv:154 rows; three scenarios nested','PASS')
    add('R14','Roles recommendation labels separate from directions','extract §5','154 Further validation; 462 original directions preserved in nested scenario data','PASS')
    add('R15','Recommendation rule fixed before looking at test','extract §5','descriptive Phase4A rule and conservative Phase4C mapping declared after experiments','PARTIAL','Acknowledge post-test research triage; cannot retroactively certify preregistration.',True)
    add('R16','weather_signature from M2 informational, not p-value filter','extract §5','results/tables/weather_signatures_for_m4.csv; family-level annotation','PASS')
    for id,path in [('R17','results/predictions/quantiles.csv.gz'),('R18','results/tables/cost_scenarios.csv'),('R19','results/tables/decision_matrix.csv'),('R20','reports/manuscript/04_rq2_decision_value.md')]:
        ok=published.get(path,'').startswith(('CREATED','REUSED'));add(id,path,'extract §6',path,'PASS' if ok else 'MISSING','None' if ok else 'Inspect existing differing target; no overwrite performed')
    add('R21','src/decision/decision_matrix.py implements matrix/labels','extract §6','zero-byte file bound by completed reproduction hashes; adapter in decision_matrix_canonical.py','MISSING','Accept versioned entry-point alternative or archival source/provenance policy. Do not overwrite frozen source.',False)
    add('R22','quantile_models.py owns training/preprocessing API','extract §6','primitives there; actual training/preprocessing in quantile_pipeline.py','PARTIAL','Team accept modular layout or versioned API/provenance migration; do not invalidate frozen source.',False)
    add('R23','cost_simulation.py owns ratios/Q/fair comparison API','extract §6','primitives there; actual orchestration in policy_evaluation/p4_policy_evaluation','PARTIAL','Team accept modular layout or versioned API/provenance migration.',False)
    add('R24','tests/test_decision.py','extract §6','tests/test_decision.py','PASS')
    add('R25','reports/figures/cost_*.png sensitivity/matrix','extract §6','canonical figures copied from verified corrected supplement','PASS' if all(published.get(p,'').startswith(('CREATED','REUSED')) for p in ['reports/figures/cost_sensitivity.png','reports/figures/cost_decision_matrix_quantiles.png','reports/figures/cost_decision_matrix_point.png']) else 'PARTIAL')
    add('R26','MAE vs asymmetric cost; demand proxy; no money/ROI/causality; F1/reproduction caveats','extract §2/§6','canonical manuscript Methods/Results/Limitations','PASS')
    add('R27','Break-even normalized gain, no dollar guess','extract §2; cost_scenarios.rules.break_even','Phase4B tables/T5_break_even_headroom.csv (stored aggregate algebra)','PASS')
    add('R28','Coordinate full report/IEEE/slides structure','extract §6','M4 section + report integration outline in audit','PARTIAL','Team merge member-owned sections and approve final manuscript/IEEE/slides structure.',False)
    add('R29','Drive 4. Processed data handoff','extract §6','local gzip and handoff manifest ready; no upload authorization/destination verified','PARTIAL','Team arrange artifact delivery; no external upload executed.',False)
    add('R30','Meet G4/G5 deadlines','extract §7','current artifacts; future team gate/submission events not evidenced','NOT VERIFIABLE','Team gate acceptance and eventual submission evidence required.',False)
    exact={
        'R03':['results/predictions/quantiles.csv.gz',handoff.SOURCES['quantiles']+'/manifest.json'],
        'R04':[handoff.SOURCES['quantiles']+'/manifest.json',handoff.SOURCES['point_reproduction']+'/predictions.csv','src/forecasting/rolling_validation.py'],
        'R05':['results/predictions/quantiles.csv.gz',str(Path(audit_path)/'crossing_counts.csv') if audit_path else 'crossing_counts.csv'],
        'R06':[handoff.SOURCES['p1']+'/p1_rows.csv',handoff.SOURCES['p1']+'/calibration.csv'],
        'R07':['results/experiments/m4_handoff_20261109T065456Z_2ef54c17bd6745068a85964fc467f523/reconciliation_checks.csv',handoff.SOURCES['point_reproduction']+'/expected_rows.csv'],
        'R08':[handoff.SOURCES['point_reproduction']+'/manifest.json','docs/m4_p4_reproduction.dtypefix.approved.json'],
        'R09':['results/tables/cost_scenarios.csv',handoff.SOURCES['quantiles']+'/metrics.csv'],
        'R10':[handoff.SOURCES['bootstrap']+'/intervals.csv',handoff.SOURCES['bootstrap']+'/manifest.json'],
        'R11':['git:b35f5cc8f5dd2d07b0a545b6e5a60963afdff6bc:notebooks/03_forecasting_ml/train_ml.ipynb','src/decision/paired_bootstrap.py'],
        'R12':[handoff.SOURCES['bootstrap']+'/intervals.csv',handoff.SOURCES['p4']+'/manifest.json'],
        'R13':['results/tables/decision_matrix.csv'],'R14':['results/tables/decision_matrix.csv','docs/m4_phase4c_compliance.md'],
        'R15':['docs/m4_phase4a_matrix.md','docs/m4_phase4c_compliance.md'],
        'R21':['src/decision/decision_matrix.py','src/decision/decision_matrix_canonical.py'],
        'R22':['src/decision/quantile_models.py','src/decision/quantile_pipeline.py'],
        'R23':['src/decision/cost_simulation.py','src/decision/policy_evaluation.py','src/decision/p4_policy_evaluation.py'],
        'R25':['reports/figures/cost_sensitivity.png','reports/figures/cost_decision_matrix_quantiles.png','reports/figures/cost_decision_matrix_point.png'],
        'R26':['reports/manuscript/04_rq2_decision_value.md'],
        'R27':['results/experiments/m4_handoff_20261109T065456Z_2ef54c17bd6745068a85964fc467f523/tables/T5_break_even_headroom.csv'],
        'R28':[str(Path(audit_path)/'report_integration_outline.md') if audit_path else 'report_integration_outline.md'],
        'R29':['results/predictions/quantiles.csv.gz'], 'R30':[str(REQUIREMENTS)+' §7']}
    for r in rows:
        if r['id'] in exact:r['evidence']='; '.join(exact[r['id']])+' | '+r['evidence']
    return pd.DataFrame(rows)


def inspect_audit(directory):
    directory=Path(directory).resolve()
    if (directory/'FAILED.json').exists():return 'failed'
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text())
        if not {'manifest.json','compliance.csv','COMPLIANCE.md','requirements_extract.md'}.issubset(marker['files']):return 'corrupt'
        for name,digest in marker['files'].items():
            target=(directory/name).resolve()
            if not target.is_relative_to(directory) or handoff.file_hash(target)!=digest:return 'corrupt'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['output_sha256']!={n:h for n,h in marker['files'].items() if n!='manifest.json'}:return 'corrupt'
        return 'complete'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete'


def manuscript(base,rankings,counts,crossings=None):
    method=base.split('## Methods\n',1)[1].split('## Results\n',1)[0]
    indices=rankings.pivot(index='scenario',columns='policy',values='P1_cost_index').reindex(columns=matrix.POLICIES).reset_index()
    result='''# 04 — RQ2 Decision Value (Member 4)

## Methods
'''+method+'''
Shared `primary_quantile_model` vẫn là TODO; actual HGB estimator đã có dated M4
sign-off trước full execution. Đây là documentary protocol gap, không được sửa
hồi tố hoặc đổi estimator sau khi đã xem test results.

## Results
Có 41,076 matched scored test observations/scenario, đầy đủ M0/M3. Direct quantile
raw/corrected metrics, P1 và P4 calibration audit, selected algorithms và per-fold
results nằm trong authoritative bundles. P4 giữ nhãn M4_REPRODUCED_M3; không phải
official M3 deliverable hay lời hứa exact numerical reproduction.

## Newsvendor Cost Comparison
P1-normalized pooled cost indices (ratios of sums):

'''+matrix.markdown(indices)+'''

Lower MAE không nhất thiết tạo lower Newsvendor cost: MAE đánh trọng số đối xứng
cho over/under errors và có optimum ở median, còn asymmetric cost nhắm tau khác
0.5 khi c_under>c_over. Residual adjustment cũng thay order quantity. Point MAE,
quantile pinball loss và simulated policy cost là ba estimands khác nhau. Các
rankings thay đổi theo cost ratio: P4-M0 thấp nhất ở S1, P4-M3 ở S3, P2 ở S5;
đó là descriptive test rankings, không phải deployment selection.

Weather value theo shared config là cost_index(P3)−cost_index(P2), khác với phần
trăm relative cost delta có denominator P2. Break-even chỉ là algebraic index/cost
headroom từ stored results, không phải USD hay estimated operational budget.

## Decision Matrix
Canonical matrix có 154 family-city rows và đủ 15 fields. Ba cost scenarios và
toàn bộ 462 descriptive directions được giữ trong JSON fields; error fields ghi
rõ corrected quantile pinball loss theo từng tau, không chọn một tau hậu nghiệm.
Tất cả recommendation là **Further validation** vì thiếu cell-level paired CI,
operational availability chưa xác minh và test exposure unresolved. Không tự động
map Favorable→Prioritize pilot hoặc Unfavorable→Do not deploy. Recommendation
này là Phase4C research triage, không nhận là rule đã preregistered trước test.

'''+matrix.markdown(counts)+'''

Bảy Salinas cells chỉ có F2-F4, không điền bù F1. M2 weather_signature chỉ là
family-level RQ1 context, không phải city evidence hoặc p-value filter cho matrix.

## Robustness
Raw/corrected quantile diagnostics và raw-clipped economic sensitivity đã được
giữ riêng trong source bundles. Báo raw crossing counts và corrected crossing=0
trong canonical export audit. Existing 95% paired bootstrap chỉ hỗ trợ P2/P3
pooled/fold. M4 dùng non-circular moving blocks L7/5000 draws (L3/14 sensitivity),
khác pinned M3 fixed four weekly blocks/1000 draws. Cùng weekly scale không đồng
nghĩa cùng design; exact shared 'same as M3' compliance là PARTIAL. Không launch
bootstrap mới. Không gán pooled CI cho family-city, P1 indices hoặc P4.

## Limitations
Observed sales chỉ là demand proxy: unobserved stockouts có thể censor demand;
simulated understock không đo actual lost demand. Chi phí dùng ratios giả định,
không có retailer prices/inventory, nên cost indices không phải money, profit
hoặc ROI. Không suy ra causal weather effects hoặc operational deployability.
ERA5 lagged reanalysis có thể chưa available ở forecast origin; masks sales-derived.
F1 test đã exposed trong pilot, không phải untouched confirmation. F2-F4 prior
test exposure chưa được team xác minh. P4 là M4-owned reproduction với input/env
differences; validation reused cho tuning/algorithm selection/calibration và
train-only residuals áp dụng lên fresh-refit predictor khác. Không có independent
calibration/conformal guarantee. Cell/P1/P4 uncertainty và RQ2 multiplicity control
chưa có. Recommendation/timing và exact module-path gaps được ghi trong compliance
matrix thay vì che bằng tên file.

## Conclusions
M4 computational artifacts hoàn tất và canonical outputs đã được schema-validated
mà không thay research results. Strict Roles/G4 readiness vẫn **NOT READY**:
primary-model TODO, exact bootstrap-design/grain coverage, pre-test label timing
và frozen placeholder/API-path gaps cần được resolve/accepted có evidence.
Không retune models, không rerun economic policies hoặc sửa frozen configs để
che các gaps đó. Xem Phase4C compliance matrix và bootstrap gap assessment.
'''
    if crossings is not None:
        result+='\nCrossing correction audit (mỗi row đếm observation triplet, không đếm ba tau riêng):\n\n'+matrix.markdown(crossings)+'\n'
    result+='\nInternal evidence: '+ '; '.join('['+k+'] `'+handoff.SOURCES[name]+'/manifest.json`' for k,name in [('Q','quantiles'),('P1','p1'),('D','p2_p3'),('B','bootstrap'),('R','point_reproduction'),('P4','p4'),('M','matrix')])+'.\n'
    return result


def prepare(requirements=REQUIREMENTS,root=ROOT):
    root=Path(root).resolve();requirements=Path(requirements).resolve()
    if not requirements.is_file():raise FileNotFoundError('Authoritative requirements extract unavailable')
    v=handoff.verify_sources(root);config=matrix.p4.reproduction.read_yaml(root/'configs/cost_scenarios.yaml')
    protocol=matrix.p4.reproduction.read_yaml(root/'configs/research_protocol.yaml')
    original=root/'src/decision/decision_matrix.py';original_hash=handoff.file_hash(original)
    parent=root/'results/experiments';dest=parent/('m4_phase4c_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+dest.name+'.staging-',dir=parent));published={}
    evidence={**v['evidence'],str(requirements):handoff.file_hash(requirements),str(Path(__file__).resolve()):handoff.file_hash(__file__),
              str(root/'docs/m4_phase4c_compliance.md'):handoff.file_hash(root/'docs/m4_phase4c_compliance.md'),str(root/'tests/test_decision.py'):handoff.file_hash(root/'tests/test_decision.py')}
    def csv(path,frame):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
    def text(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f:f.write(value)
    try:
        handoff.export_copy(requirements,stage,'requirements_extract.md')
        qfile=v['dirs']['quantiles']/('m4_quantiles_'+matrix.p4.reproduction.FINAL+'.csv')
        q=pd.read_csv(qfile,parse_dates=['date','forecast_origin'],dtype={'store_nbr':str},float_precision='round_trip')
        expected=pd.read_csv(v['dirs']['point_reproduction']/'expected_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip');expected=expected.loc[expected.split.eq('test')]
        canonical=canonical_quantiles(q,config);validate_quantiles(canonical,expected,config)
        test=q.loc[q.split.eq('test')].reset_index(drop=True)
        if not np.array_equal(canonical.y_pred,test.corrected_prediction) or not np.array_equal(canonical.y_true,test.y_true):raise ValueError('Export changed frozen predictions/targets')
        qpath=stage/'canonical/results/predictions/quantiles.csv.gz';qpath.parent.mkdir(parents=True)
        with qpath.open('xb') as raw:
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:canonical.to_csv(f,index=False)
        loaded=pd.read_csv(qpath,parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip');validate_quantiles(loaded,expected,config)
        pd.testing.assert_frame_equal(loaded,canonical.reset_index(drop=True),check_dtype=False)
        fields=['date','store_nbr','family','city','fold','config','split']
        crossings=q.drop_duplicates(fields).groupby(['split','config']).agg(observations=('raw_crossing','size'),raw_crossed_observations=('raw_crossing','sum'),corrected_crossed_observations=('corrected_crossing','sum')).reset_index()
        csv(stage/'crossing_counts.csv',crossings)
        cm=canonical_matrix(v['cells'],config);validate_matrix(cm,protocol['scope']['deep_study_families'],sorted(v['cells'].city.unique()))
        csv(stage/'canonical/results/tables/decision_matrix.csv',cm)
        cm_reload=pd.read_csv(stage/'canonical/results/tables/decision_matrix.csv',keep_default_na=False)
        validate_matrix(cm_reload,protocol['scope']['deep_study_families'],sorted(v['cells'].city.unique()))
        pd.testing.assert_frame_equal(cm_reload,cm,check_dtype=False)
        comp=pd.read_csv(v['dirs']['p4']/'comparisons.csv',float_precision='round_trip')
        dc=pd.read_csv(v['dirs']['p2_p3']/'comparisons.csv',float_precision='round_trip')
        unit_cols=['scenario','fold','family','city','overstock_units','understock_units']
        pu=pd.read_csv(v['dirs']['p1']/'p1_rows.csv',usecols=unit_cols,float_precision='round_trip')
        p4u=pd.read_csv(v['dirs']['p4']/'p4_rows.csv',usecols=unit_cols+['config'],float_precision='round_trip')
        ci=pd.read_csv(v['dirs']['bootstrap']/'intervals.csv',float_precision='round_trip');matrix.validate_supported_uncertainty(ci)
        costs=canonical_costs(comp,dc,pu,p4u,ci,config);csv(stage/'canonical/results/tables/cost_scenarios.csv',costs)
        costs_reload=pd.read_csv(stage/'canonical/results/tables/cost_scenarios.csv',float_precision='round_trip')
        validate_costs(costs_reload);pd.testing.assert_frame_equal(costs_reload,costs,check_dtype=False)
        figures={'policy_comparison.png':'cost_sensitivity.png','weather_P3_minus_P2.png':'cost_decision_matrix_quantiles.png','weather_P4_M3_minus_M0.png':'cost_decision_matrix_point.png'}
        for source,target in figures.items():handoff.export_copy(v['dirs']['figures']/source,stage,'canonical/reports/figures/'+target)
        previous=root/'results/experiments/m4_handoff_20261109T065456Z_2ef54c17bd6745068a85964fc467f523'
        if handoff.inspect_package(previous)!='complete':raise ValueError('Phase4B report package invalid')
        counts=pd.read_csv(previous/'tables/T4_classification_counts.csv')
        text(stage/'canonical/reports/manuscript/04_rq2_decision_value.md',manuscript((previous/'M4_METHODS_RESULTS.md').read_text(encoding='utf-8'),v['rankings'],counts,crossings))
        handoff.export_copy(previous/'tables/T5_break_even_headroom.csv',stage,'break_even_headroom.csv')
        handoff.export_copy(v['dirs']['bootstrap']/'intervals.csv',stage,'supported_bootstrap_only.csv')
        for file in (stage/'canonical').rglob('*'):
            if file.is_file():
                relative=str(file.relative_to(stage/'canonical')).replace('\\','/')
                published[relative]=publish_exclusive(file,root/relative)
        if handoff.file_hash(original)!=original_hash:raise ValueError('Frozen empty entry point changed')
        compliance=compliance_rows(root,published,dest);csv(stage/'compliance.csv',compliance)
        text(stage/'COMPLIANCE.md','# M4 Phase4C — NOT READY for strict Roles/G4 compliance\n\n'+matrix.markdown(compliance)+'\n')
        text(stage/'bootstrap_gap.md',(root/'docs/m4_phase4c_compliance.md').read_text(encoding='utf-8')+'\nNo new bootstrap was launched. Numeric confidence fields unavailable at cells remain unavailable.\n')
        text(stage/'report_integration_outline.md','# Joint report integration\n\nM1: data/QA/baselines; M2: RQ1 inference; M3: forecasting/ablation; M4: canonical 04 section (Methods, Results, Newsvendor Cost Comparison, Decision Matrix, Robustness, Limitations, Conclusions). Lead coordinates combined report/IEEE/slides. Use frozen results and shared definitions; do not represent M4_REPRODUCED_M3 as official M3. No other member section is edited here.\n')
        for name,digest in evidence.items():
            if handoff.file_hash(name)!=digest:raise ValueError('Source changed during schema export: '+name)
        outputs={str(f.relative_to(stage)).replace('\\','/'):handoff.file_hash(f) for f in stage.rglob('*') if f.is_file()}
        matrix.p4.reproduction._json(stage/'manifest.json',dict(readiness='NOT READY',authority_source=str(requirements),file_sha256=evidence,
            canonical_publications=published,quantile_rows=len(canonical),matrix_rows=len(cm),cost_summary_rows=len(costs),
            recommendation_counts=cm.recommendation.value_counts().to_dict(),primary_quantile_model=protocol['quantile_and_decision']['primary_quantile_model'],
            implementation_model='HistGradientBoostingQuantile',scientific_decisions_unresolved=True,new_bootstrap_launched=False,research_results_changed=False,
            compliance_counts=compliance.status.value_counts().to_dict(),output_sha256=outputs))
        matrix.p4.reproduction._json(stage/'COMPLETE.json',{'files':{**outputs,'manifest.json':handoff.file_hash(stage/'manifest.json')}})
        if inspect_audit(stage)!='complete':raise ValueError('Canonical compliance audit integrity failure')
        if dest.exists():raise FileExistsError('Never overwrite compliance audit')
        stage.rename(dest);return dest
    except Exception as e:
        matrix.p4.reproduction._json(stage/'FAILED.json',{'type':type(e).__name__,'message':str(e)});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--requirements',type=Path,default=REQUIREMENTS);args=parser.parse_args()
    print(prepare(args.requirements))


if __name__=='__main__':main()
