"""Member 4 report/handoff assembly only: no fitting, calibration or policy reruns."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from . import consolidated_matrix as matrix
from . import p4_policy_evaluation as p4
from . import p4_reproduction as points
from . import p1_execution as p1
from . import policy_evaluation as direct
from . import paired_bootstrap as bootstrap
from .quantile_artifacts import inspect_run,_validate_saved_panel
from .quantile_contracts import file_hash,semantic_hash

ROOT=matrix.ROOT
MATRIX='results/experiments/m4_decision_matrix_20261109T062535Z_d3224a2a64ea4331a76bdaeebf706d5e'
FIGURES='results/experiments/m4_matrix_report_figures_20261109T062910Z_43b9c746faf748348d6e1a09b7712d59'
SOURCES=dict(quantiles='results/experiments/m4_quantiles_'+points.FINAL,p1=p4.P1_SOURCE,
             p2_p3=p4.DIRECT_SOURCE,bootstrap=matrix.BOOTSTRAP_SOURCE,point_reproduction=p4.SOURCE,
             p4=matrix.P4_SOURCE,matrix=MATRIX,figures=FIGURES)
CANONICAL={'results/predictions/quantiles.csv':('quantiles','m4_quantiles_'+points.FINAL+'.csv'),
           'results/tables/cost_scenarios.csv':('matrix','policy_summary.csv'),
           'results/tables/decision_matrix.csv':('matrix','matrix.csv')}


def hash_maps(value):
    """Recorded historical evidence, not recomputed live approval directory globs."""
    if isinstance(value,dict):
        for k,v in value.items():
            if k=='file_sha256':yield v
            else:yield from hash_maps(v)
    elif isinstance(value,list):
        for item in value:yield from hash_maps(item)


def recorded_hashes(manifests):
    result={}
    for manifest in manifests:
        for mapping in hash_maps(manifest):
            for name,digest in mapping.items():
                if name in result and result[name]!=digest:raise ValueError('Conflicting historical source hashes: '+name)
                result[name]=digest
    for name,digest in result.items():
        if file_hash(name)!=digest:raise ValueError('Historical provenance mismatch: '+name)
    return result


def reconcile_pooled(cells,rankings,row_costs):
    """Compare stored costs/aggregates, never evaluate a quantity-cost function."""
    records=[]
    if len(rankings)!=15 or set(rankings.scenario)!={'S1','S3','S5'} or any(set(g.policy)!=set(matrix.POLICIES) for _,g in rankings.groupby('scenario')):
        raise ValueError('Require all fifteen scenario-policy summaries')
    for scenario,group in rankings.groupby('scenario'):
        denominator=row_costs[(scenario,'P1')].sum()
        for _,r in group.iterrows():
            values=row_costs[(scenario,r.policy)]
            g=cells.loc[cells.scenario.eq(scenario)]
            matrix_mean=np.average(g['mean_cost_'+r.policy],weights=g.sample_size)
            stored_mean=float(values.mean());index=float(values.sum()/denominator) if denominator else np.nan
            if (len(values)!=r.sample_size or g.sample_size.sum()!=r.sample_size
                or not np.isclose(stored_mean,r.mean_simulated_cost,rtol=1e-12,atol=1e-12)
                or not np.isclose(matrix_mean,r.mean_simulated_cost,rtol=1e-12,atol=1e-12)
                or not np.isclose(index,r.P1_cost_index,rtol=1e-12,atol=1e-12,equal_nan=True)):
                raise ValueError('Pooled cost/index reconciliation failed')
            records.append(dict(scenario=scenario,policy=r.policy,sample_size=len(values),stored_row_mean=stored_mean,
                authoritative_pooled_mean=r.mean_simulated_cost,matrix_weighted_mean=matrix_mean,
                authoritative_index=r.P1_cost_index,reconciled_index=index,status='PASS'))
    return pd.DataFrame(records)


def verify_sources(root=ROOT):
    root=Path(root).resolve();dirs={k:root/v for k,v in SOURCES.items()};manifests={}
    inspectors=dict(p1=p1.inspect_bundle,p2_p3=direct.inspect_evaluation,bootstrap=bootstrap.inspect_bootstrap,
                    point_reproduction=points.inspect_bundle,p4=p4.inspect_bundle,matrix=matrix.inspect_bundle,figures=matrix.inspect_bundle)
    if inspect_run(root,points.FINAL)['status']!='complete':raise ValueError('Quantile bundle incomplete/corrupt')
    for name,folder in dirs.items():
        if name!='quantiles' and inspectors[name](folder)!='complete':raise ValueError('Incomplete/corrupt source: '+name)
        manifests[name]=json.loads((folder/'manifest.json').read_text())
        if 'approval' in manifests[name]:
            a=manifests[name]['approval']
            if a.get('status')!='APPROVED' or a.get('request_sha256')!=semantic_hash(a['request']):raise ValueError('Historical approval digest invalid')
    evidence=recorded_hashes(list(manifests.values()))
    for folder in dirs.values():evidence.update({str(p):file_hash(p) for p in folder.iterdir() if p.is_file()})
    expected=pd.read_csv(dirs['point_reproduction']/'expected_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    point_panel=pd.read_csv(dirs['point_reproduction']/'predictions.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    protocol=points.read_yaml(root/'configs/research_protocol.yaml')
    points.validate_forecasts(point_panel,expected,protocol['rq2']['validation']['test_folds'])
    q=pd.read_csv(dirs['quantiles']/('m4_quantiles_'+points.FINAL+'.csv'),parse_dates=['date','forecast_origin'],dtype={'store_nbr':str},float_precision='round_trip')
    _validate_saved_panel(q,manifests['quantiles'])
    for _,g in q.groupby(['config','tau_key']):points.assert_matched_rows(expected,g)
    test=expected.loc[expected.split.eq('test')];costs={};groups_verified=0
    wanted=set(matrix.p4.MATCH+['y_true','simulated_cost','split','policy','config','scenario','variant'])
    specs=[('p1','p1_rows.csv',['scenario']),('p2_p3','row_costs.csv',['scenario','policy','variant']),('p4','p4_rows.csv',['scenario','config'])]
    for name,file,grouping in specs:
        rows=pd.read_csv(dirs[name]/file,usecols=lambda c:c in wanted,parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
        for key,g in rows.groupby(grouping,sort=True):
            key=key if isinstance(key,tuple) else (key,);fields=dict(zip(grouping,key))
            if 'split' not in g:g=g.assign(split='test')  # frozen direct-cost table is test-only by contract
            if not g.split.eq('test').all():raise ValueError('Non-test economic rows')
            points.assert_matched_rows(test,g);groups_verified+=1
            label='P1' if name=='p1' else ('P4_'+fields['config'] if name=='p4' else fields['policy'])
            if name!='p2_p3' or fields['variant']=='corrected':costs[(fields['scenario'],label)]=g.simulated_cost.to_numpy()
    cells=pd.read_csv(dirs['matrix']/'matrix.csv',keep_default_na=False,na_values=[''],float_precision='round_trip')
    summary=pd.read_csv(dirs['matrix']/'family_city_summary.csv',float_precision='round_trip')
    rankings=pd.read_csv(dirs['matrix']/'policy_summary.csv',float_precision='round_trip')
    families=protocol['scope']['deep_study_families'];cities=sorted(cells.city.unique())
    matrix.validate_matrix(cells,families,cities,['S1','S3','S5'])
    if len(cells)!=462 or len(summary)!=154 or summary.duplicated(['family','city']).any():raise ValueError('Matrix/summary scope failure')
    checks=reconcile_pooled(cells,rankings,costs)
    for _,r in cells.iterrows():
        for policy in matrix.POLICIES:
            expected_index=r['mean_cost_'+policy]/r.mean_cost_P1 if r.mean_cost_P1 else np.nan
            if not np.isclose(expected_index,r['cost_index_'+policy],rtol=1e-12,equal_nan=True):raise ValueError('Cell index mismatch')
    return dict(root=root,dirs=dirs,manifests=manifests,evidence=evidence,cells=cells,summary=summary,rankings=rankings,
                reconciliation=checks,cost_groups_verified=groups_verified,quantile_rows=len(q),point_rows=len(point_panel))


def break_even_table(rankings):
    records=[]
    for scenario,g in rankings.groupby('scenario'):
        indexed=g.set_index('policy')
        for no_weather,weather in [('P2','P3'),('P4_M0','P4_M3')]:
            gap=indexed.loc[no_weather,'mean_simulated_cost']-indexed.loc[weather,'mean_simulated_cost']
            index_gap=indexed.loc[no_weather,'P1_cost_index']-indexed.loc[weather,'P1_cost_index']
            records.append(dict(scenario=scenario,comparison=weather+' minus '+no_weather,
                delta_index_weather_minus_no_weather=-index_gap,raw_cost_headroom=gap,
                nonnegative_conditional_headroom=max(0.,gap),P1_normalized_headroom=index_gap,
                units='relative simulated cost per scored observation; not money',
                interpretation='algebraic headroom only; no operational budget/ROI claim'))
    return pd.DataFrame(records)


def pooled_diagnostics(diag):
    metrics=['pinball_loss','coverage','coverage_gap','strict_less_rate','tie_rate','crossing_rate','negative_rate'];out=[]
    for key,g in diag.groupby(['config','tau_key','variant'],sort=True):
        out.append(dict(zip(['config','tau_key','variant'],key),sample_size=int(g.sample_size.sum()),
                        **{c:float(np.average(g[c],weights=g.sample_size)) for c in metrics}))
    return pd.DataFrame(out)


def csv_report_frame(frame):
    """CSV preserves labels/values, not a pandas column-axis name from crosstab."""
    return frame.copy(deep=True).reset_index(drop=True).rename_axis(None,axis=1)


def export_copy(source,stage,relative):
    stage=Path(stage).resolve();relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe handoff destination')
    target=(stage/relative).resolve()
    if not target.is_relative_to(stage):raise ValueError('Handoff destination outside package')
    target.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    if file_hash(source)!=file_hash(target):raise ValueError('Byte-copy export checksum mismatch')
    return target


def inspect_package(directory):
    directory=Path(directory).resolve()
    if (directory/'FAILED.json').exists():return 'failed'
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text());files=marker['files']
        required={'handoff_manifest.json','README.md','M4_METHODS_RESULTS.md','artifact_map.csv','figure_index.csv','risk_checklist.md','REPRODUCE.md'}
        if not required.issubset(files):return 'corrupt'
        for name,digest in files.items():
            rel=Path(name);target=(directory/rel).resolve()
            if rel.is_absolute() or '..' in rel.parts or not target.is_relative_to(directory) or file_hash(target)!=digest:return 'corrupt'
        manifest=json.loads((directory/'handoff_manifest.json').read_text())
        if manifest['output_sha256']!={n:h for n,h in files.items() if n!='handoff_manifest.json'}:return 'corrupt'
        return 'complete'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete'


def report_text(rankings,diagnostics,classifications):
    indices=rankings.pivot(index='scenario',columns='policy',values='P1_cost_index').reindex(columns=matrix.POLICIES).reset_index()
    return r'''# M4 Methods và Results: Quantile Forecasting, Newsvendor và Decision Matrix

## Methods

Phân tích M4 sử dụng bảy frozen families tại 54 stores/22 cities, với target là
daily observed sales. Shared F1–F4 expanding rolling folds có validation/test
blocks dài 28 ngày; forecast được phát tại cuối t−1 cho ngày t. Observed past
test sales được phép cập nhật lag history; đây không phải 28-day fixed-origin
forecast. Shared masks loại closure/cooldown khỏi scoring; fresh refit loại closure
targets nhưng giữ eligible cooldown targets. Mỗi scenario có 41,076 matched test
observations. Candidate 34-column có SHA-256 e4186aeb3a2da9996dcb9deec164cc70ca11c0a869b932720c9786c64a3ab17d.

Direct Quantile Regression dùng global HistGradientBoostingRegressor(loss='quantile')
tại tau=0.50, 0.75 và 5/6, seed 42. M0 gồm calendar, calendar-aligned sales lags
1/7/14/28, rolling means 7/28 và store/family/city identifiers. M3 chỉ thêm lag1
precipitation và maximum temperature. Numeric median imputation và native
categorical encoding được fit trên eligible train khi selection, và fit mới trên
eligible train+validation khi refit. Không dùng promotions, transactions hoặc
same-day measured weather. Grid giữ max_iter=100, min_samples_leaf=20,
learning_rate=0.05 và max_leaf_nodes ∈ {15,31}; equal search budget, chọn bằng raw
validation pinball loss, tie chọn candidate đầu. Tổng 48 selection fits + 24 fresh
refits. Raw forecasts được giữ; corrected forecasts clip tại zero và sort theo
tau. Sorting không phải calibration và không bảo đảm cải thiện mọi metric. [Q]

Với u=D−Q, pinball loss là rho_tau(u)=max(tau*u,(tau−1)*u). Coverage là tỷ lệ D≤Q;
strict-less D<Q và ties được báo riêng, coverage gap=coverage−tau. P4 clipped point
MAE là metric khác và không được dùng thay quantile pinball loss.

P1 là M1 Seasonal Naive fixed rule (lag7, fallback lag14/rolling7) cộng signed
validation-residual quantile. Residual=y_true−unmodified SeasonalNaive forecast;
pooling fold/model/config/family, linear quantile, n≥30, fallback fold/model/config.
P2/P3 lần lượt dùng corrected direct quantiles M0/M3. [P1, D]

P4-M0/P4-M3 dùng M4_REPRODUCED_M3 point artifacts, không phải official M3 hoặc cam
kết exact numerical reproduction. Adapter pin M3 commit b35f5cc, giữ original
Ridge/ExtraTrees/HGB preprocessing và bảy hyperparameter candidates. Chọn candidate
và algorithm bằng clipped validation MAE, ties theo Ridge→ExtraTrees→HGB; 56
selection fits và tám fresh refits. F1 chọn ExtraTrees, F2–F4 chọn HGB cho cả hai
configs. Train-only validation forecasts được giữ. P4 residual=y_true−clipped
validation forecast; pooling fold/config/family, linear quantile, n≥30, fallback
fold/config. Base là clipped fresh-refit test forecast. P1/P4 đều giữ fractional
Q=max(0,base+residual quantile), không lấy correction từ test residuals. [R, P4]

Simulated Newsvendor cost là C=c_over*max(Q−D,0)+c_under*max(D−Q,0), với D là sales
proxy, ratios 1:1, 1:3, 1:5 và tau=c_under/(c_over+c_under). Cost index được tính
bằng sum(C_policy)/sum(C_P1) trên cùng observations, không trung bình row ratios;
zero denominator trả NA. Đây là relative simulated costs, không actual money.

P2/P3 uncertainty dùng paired non-circular moving blocks trên calendar dates,
giữ toàn bộ store/family/city rows của ngày được chọn và cùng sampled blocks cho
hai policies. Blocks không vượt fold hoặc calendar gap. Có 5,000 resamples, seed
42, percentile intervals, L=7 primary và L=3/14 sensitivity. Design được ghi nhận
analyst-declared, không mô tả là preapproved. Intervals chỉ hỗ trợ pooled/fold
estimands của P3−P2; không có cell/P1/P4 intervals hoặc matrix multiple-testing
correction. [B]

Decision Matrix có 7×22×3=462 rows và summary 154 cells. Mỗi weather comparison
được gọi directionally favorable/unfavorable khi cả bốn fold deltas đều âm/dương;
zeros hoặc opposing signs là mixed; thiếu fold/nonfinite là insufficient. Tổng
hợp chỉ cùng hướng khi cả P3−P2 và P4-M3−P4-M0 đạt cùng classification. Rule được
khai báo Phase 4A, không nhận là preregistered trước test. M2 family-level RQ1
weather signatures chỉ là annotations; không suy ra city-specific causal evidence. [M]

## Results

Pooled simulated cost indices (P1=1; đầy đủ mean costs nằm ở Table 1):

'''+matrix.markdown(indices)+r'''

Descriptive lowest-cost policies lần lượt là P4-M0 (1:1), P4-M3 (1:3) và P2 (1:5).
Không chọn deployment policy từ các rankings này. Fixed source costs/indices và
matrix-weighted aggregates đã được reconciled trên cùng keys/targets.

P3−P2 pooled cost deltas lần lượt là −0.030316%, −0.024620% và +0.139277% so với P2.
Descriptive L7 bootstrap intervals tương ứng [−0.041486,−0.016006]%,
[−0.088921,+0.060195]% và [−0.014263,+0.311632]%. Không chuyển các bounds này xuống
family/city hoặc diễn giải thành formal significance. P4-M3 so với P4-M0 có deltas
+0.073580%, −0.071954%, −0.070703%; uncertainty cho P4 chưa được phân tích. [D,B,P4]

Corrected direct-quantile coverage khoảng 0.511, 0.775 và 0.866 tại ba taus;
coverage gaps dương và corrected crossing rate bằng zero. Pinball/coverage/ties
raw/corrected được tách ở Table 2; không gộp với P4 point MAE.

Classification tổng:

'''+matrix.markdown(classifications)+r'''

BEVERAGES×El Carmen×1:5 directionally favorable, còn BEVERAGES×Babahoyo×1:5
directionally unfavorable theo rule bốn folds/cả hai comparisons. Bảy Salinas
cells mỗi scenario chỉ có F2–F4, vì vậy insufficient; không điền bù F1. Tính pooled
không đồng nghĩa mọi cell đều tốt hơn P1; các exceptions được giữ trong matrix.

## Limitations và phạm vi claims

F1 test metrics đã được xem trong pilots; F1 là exploratory/post-pilot. Prior
exposure của F2–F4 chưa được team xác minh, nên không nhận là untouched
confirmatory sets. ERA5 retrospective lag values chưa chứng minh sẵn có tại
forecast origin; closure/cooldown masks sales-derived và store 52 có limited
history. Observed sales là censored demand proxy, không đo lost demand/stockouts.
Không suy ra actual inventory cost, profit, ROI, causality hoặc operational
deployability. P4 validation được dùng cho cả selection và residual calibration;
calibration không independent, và train-only residuals được áp dụng lên refitted
predictor khác. Không có conformal guarantee. P1/P4/cell-level uncertainty chưa có;
M2 RQ1 FDR không phải multiple-testing correction cho RQ2 matrix.

## Internal evidence references

[Q] Quantile final manifest/metrics/predictions; [P1] SeasonalNaive cost bundle;
[D] P2/P3 cost bundle; [B] P2/P3 bootstrap bundle; [R] M4_REPRODUCED_M3 manifest;
[P4] clipped-residual P4 cost bundle; [M] Decision Matrix and corrected figure
supplement. Exact paths/hashes nằm trong artifact_map.csv và provenance/sources/.
Đây là M4 contribution draft để ghép vào report chung, không phải full paper hay
chứng nhận Roles/Proposal v4.2 compliance khi chưa đọc được hai tài liệu đó.
'''


def build_package(root=ROOT):
    root=Path(root).resolve();v=verify_sources(root);dirs=v['dirs'];files=dict(v['evidence'])
    for path in [Path(__file__).resolve(),root/'docs/m4_phase4b_handoff.md',root/'README.md',root/'configs/research_protocol.yaml',root/'configs/cost_scenarios.yaml']:
        files[str(path.resolve())]=file_hash(path)
    parent=root/'results/experiments';destination=parent/('m4_handoff_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=parent));artifact_map=[]
    def copy(source,relative,deliverable,authority):
        target=export_copy(source,stage,relative)
        artifact_map.append(dict(deliverable=deliverable,authority=authority,authoritative_source=str(Path(source).relative_to(root)),
            source_sha256=file_hash(source),package_file=relative,package_sha256=file_hash(target),status='VERIFIED_BYTE_COPY'))
    def text(relative,value):
        path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f:f.write(value)
    def table(relative,frame):
        frame=csv_report_frame(frame)
        path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
        loaded=pd.read_csv(path,float_precision='round_trip',keep_default_na=False,na_values=[''])
        pd.testing.assert_frame_equal(loaded,frame.reset_index(drop=True),check_dtype=False)
    try:
        for label,folder in dirs.items():
            for name in ['manifest.json','COMPLETE.json']:copy(folder/name,'provenance/sources/'+label+'/'+name,label+' source evidence','completed bundle')
        for relative in ['configs/research_protocol.yaml','configs/cost_scenarios.yaml','docs/m4_final_signoff.approved.json','docs/m4_p1.approved.json',
                         'docs/m4_p4_reproduction.dtypefix.approved.json','docs/m4_p4_calibration.clipped.approved.json']:
            copy(root/relative,'provenance/'+relative,relative,'shared contract or explicit approval evidence')
        copies={'policy_summary.csv':'T1_pooled_policy_costs.csv','weather_summary.csv':'T3_weather_uncertainty.csv',
                'forecast_diagnostics.csv':'quantile_diagnostics_by_cell.csv','point_forecast_diagnostics.csv':'P4_point_MAE_by_cell.csv',
                'supported_bootstrap.csv':'bootstrap_supported_pooled_fold.csv','matrix.csv':'decision_matrix_462.csv',
                'family_city_summary.csv':'family_city_summary_154.csv','family_city_summary.html':'family_city_summary_154.html',
                'fold_consistency.csv':'fold_consistency.csv','matrix_raw_sensitivity.csv':'decision_matrix_raw_sensitivity.csv'}
        for name,dest in copies.items():copy(dirs['matrix']/name,'tables/'+dest,name,'authoritative matrix aggregation')
        diag=pd.read_csv(dirs['matrix']/'forecast_diagnostics.csv',float_precision='round_trip')
        diagnostic=pooled_diagnostics(diag);table('tables/T2_pooled_quantile_diagnostics.csv',diagnostic)
        classification=pd.crosstab(v['cells'].scenario,v['cells'].weather_class).reindex(columns=['directionally_favorable','directionally_unfavorable','mixed_or_inconsistent','insufficient_evidence'],fill_value=0).reset_index()
        table('tables/T4_classification_counts.csv',classification)
        table('tables/T5_break_even_headroom.csv',break_even_table(v['rankings']))
        table('reconciliation_checks.csv',v['reconciliation'])
        comparisons=pd.read_csv(dirs['p4']/'comparisons.csv',float_precision='round_trip');fold_records=[]
        for (scenario,fold),g in comparisons.loc[comparisons.variant.eq('corrected')&comparisons.report_level.eq('fold')].groupby(['scenario','fold']):
            for policy in matrix.POLICIES:
                cfg='M3' if policy=='P4_M3' else 'M0';r=g.loc[g.config.eq(cfg)].iloc[0];column='P4' if policy.startswith('P4_') else policy
                fold_records.append(dict(scenario=scenario,fold=fold,policy=policy,sample_size=int(r.sample_size),
                    mean_simulated_cost=r['mean_cost_'+column],P1_cost_index=r['cost_index_'+column]))
        table('tables/fold_policy_costs.csv',pd.DataFrame(fold_records))
        figure_index=[]
        captions={'policy_comparison':'Pooled descriptive simulated cost indices; no policy confidence intervals or deployment claim.',
                  'weather_P3_minus_P2':'Family-city relative paired P3−P2 cost deltas; no cell CIs; negative means lower simulated cost.',
                  'weather_P4_M3_minus_M0':'Family-city relative P4-M3−P4-M0 cost deltas; M4_REPRODUCED_M3; uncertainty unavailable.'}
        for stem,caption in captions.items():
            for ext in ['png','svg']:
                relative='figures/'+stem+'.'+ext;copy(dirs['figures']/(stem+'.'+ext),relative,stem,'layout-corrected verified figure supplement')
                figure_index.append(dict(file=relative,caption=caption,source=str(dirs['figures'].relative_to(root)),
                                         uncertainty='no family-city CI; supported P2/P3 pooled/fold intervals are a separate table'))
        table('figure_index.csv',pd.DataFrame(figure_index))
        export_info=[]
        for canonical,(label,name) in CANONICAL.items():
            relative='exports/'+canonical;copy(dirs[label]/name,relative,canonical,'candidate canonical export; Roles schema UNVERIFIED')
            source=dirs[label]/name
            export_info.append(dict(proposed_repository_path=canonical,package_file=relative,source_sha256=file_hash(source),
                existing_repository_path=(root/canonical).exists(),schema_status='PENDING_TEAM_CONFIRMATION; no repository canonical file written',
                semantics='full validation/test raw/corrected long panel' if label=='quantiles' else ('15-row pooled policy costs, not cost-assumption inputs' if name=='policy_summary.csv' else '462-row descriptive scenario matrix')))
        p4.reproduction._json(stage/'exports/contract_candidates.json',export_info)
        text('exports/README.md','These are byte-identical candidate exports. Roles/Proposal v4.2 unavailable; canonical schemas not certified. Do not overwrite repository destinations without schema/team agreement. quantiles.csv retains BOTH splits and raw/corrected columns; cost_scenarios.csv is the 15-row pooled policy summary. Source experiment bundles remain authoritative.\n')
        table('artifact_map.csv',pd.DataFrame(artifact_map))
        text('M4_METHODS_RESULTS.md',report_text(v['rankings'],diagnostic,classification))
        text('risk_checklist.md','''# Remaining risks and integration decisions

## Actual M4 implementation readiness
- [x] Quantiles, P1/P2/P3/P4 economic artifacts, P2/P3 bootstrap, 462/154 matrix complete.
- [x] All eight authoritative bundles, stored provenance, matched observations and pooled costs reconciled.
- [x] Report tables, corrected figures, byte-identical candidate exports and manifests available.
- [x] Shared break-even item addressed as algebraic headroom from stored costs, not USD/ROI.

## Verification gaps / team integration (not missing forecasting implementation)
- [ ] Obtain Roles v4.2 and Proposal v4.2; exact rubric, required notebook/report/slides scope and canonical schemas UNVERIFIED.
- [ ] Agree canonical split/column schemas before promoting candidate exports. Root canonical paths remain untouched.
- [ ] Team attest prior F2-F4 test exposure; retain exploratory wording until resolved.
- [ ] Accept/identify M4_REPRODUCED_M3 provenance in joint report; it is not official M3 delivery.
- [ ] Align stale README pipeline status/shared TODO narrative with approved M4 manifests; do not silently edit shared configs.
- [ ] Integrate M4 text/captions into joint manuscript/slides and venue bibliography/AI-disclosure policy.
- [ ] Review analyst-declared bootstrap design status; do not describe it as retrospectively preapproved.

## Research limitations, not repaired by handoff
- [ ] F1 previously exposed; retrospective ERA5 and sales-derived closure masks.
- [ ] Censored demand proxy; no actual costs, profit, ROI, causality or deployment claims.
- [ ] P4 non-independent validation calibration and train/refit predictor mismatch.
- [ ] No cell/P1/P4 uncertainty or RQ2 multiple-testing correction. Separate analysis only if those claims are needed.
- [ ] Salinas lacks scored F1 rows; seven cells/scenario remain insufficient under the declared four-fold rule.
''')
        text('VERIFY_PACKAGE.py', '''"""Standalone checksum verification; no repo/data/packages or writes required."""
from pathlib import Path
from hashlib import sha256
import json
root=Path(__file__).resolve().parent
marker=json.loads((root/'COMPLETE.json').read_text(encoding='utf-8'))
for name,digest in marker['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise SystemExit('Unsafe manifest path')
    h=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=digest:raise SystemExit('Checksum failure: '+name)
print('PACKAGE_CHECKSUMS_PASS',len(marker['files']))
''')
        text('REPRODUCE.md','''# Reproducibility and safe verification

Report package: read M4_METHODS_RESULTS.md, tables/, figures/, artifact_map.csv and figure_index.csv. These report assets are local standalone copies. Full verification of all sources still requires the original repository, .venv and authoritative local bundles/data; raw datasets and every row-level economic file are not embedded.

On any machine with Python 3.9+, run `python -B <package>/VERIFY_PACKAGE.py` to verify the package's own checksums with the standard library only. This does not authenticate unavailable original raw inputs or replace full-source verification.

From repository root, verify ONLY (no training/economic reruns):
```powershell
& '.\\.venv\\Scripts\\python.exe' -B -m src.decision.m4_handoff --verify <this_package_directory>
```
To assemble another uniquely named handoff from immutable sources:
```powershell
& '.\\.venv\\Scripts\\python.exe' -B -m src.decision.m4_handoff --build
```
Do not rerun final_run, point reproduction, calibration, economic evaluation or bootstrap entry points just to read the final report. Model/data/code/package hashes and approvals are in the authoritative manifests; copies are in provenance/. Generated predictions/experiments are Git-ignored. Decide team artifact storage separately; do not commit large outputs by accident.
''')
        text('README.md','''# Member 4 final handoff

Status: **READY_FOR_HANDOFF** for M4 peer review/team report integration.
This is NOT certification of unavailable Roles/Proposal v4.2 compliance, final team submission or deployment readiness.

Start with [M4 Methods/Results](M4_METHODS_RESULTS.md), [artifact map](artifact_map.csv), [figure index](figure_index.csv), [risk checklist](risk_checklist.md) and [reproducibility](REPRODUCE.md).

Primary table: tables/T1_pooled_policy_costs.csv. Matrix: tables/decision_matrix_462.csv; readable summary: tables/family_city_summary_154.html. Use the figures/ copies from the corrected supplement; original matrix P4 heatmap had a label-cropping issue.

exports/ contains schema candidates only; source bundles remain authoritative. No source files, prior bundles, canonical data or shared configs were changed. No model training or economic-policy execution occurs in this handoff assembly.
''')
        for name,digest in files.items():
            if file_hash(name)!=digest:raise ValueError('Source changed during handoff: '+name)
        outputs={str(f.relative_to(stage)).replace('\\','/'):file_hash(f) for f in stage.rglob('*') if f.is_file()}
        p4.reproduction._json(stage/'handoff_manifest.json',dict(status='READY_FOR_HANDOFF',
            readiness_scope='M4 artifacts/report peer review and team integration; not submission/deployment certification',
            roles_v4_2='UNAVAILABLE; compliance unverified',proposal_v4_2='UNAVAILABLE; compliance unverified',
            document_search_evidence='local workspace/docx search, fetched origin/main tree, connected Drive Favorita/STAT3013 searches: no source documents found',
            source_bundles=SOURCES,file_sha256=files,matched_cost_groups_verified=v['cost_groups_verified'],quantile_rows=v['quantile_rows'],
            selected_point_rows=v['point_rows'],matrix_rows=462,summary_cells=154,canonical_candidates=export_info,
            economic_evaluation_rerun=False,forecasting_model_fits=0,team_submission_ready=False,output_sha256=outputs))
        p4.reproduction._json(stage/'COMPLETE.json',{'files':{**outputs,'handoff_manifest.json':file_hash(stage/'handoff_manifest.json')}})
        if inspect_package(stage)!='complete':raise ValueError('Handoff package integrity failure')
        if destination.exists():raise FileExistsError('Never overwrite prior handoff packages')
        stage.rename(destination);return destination
    except Exception as e:
        p4.reproduction._json(stage/'FAILED.json',{'type':type(e).__name__,'message':str(e)});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--build',action='store_true');g.add_argument('--verify',type=Path);args=parser.parse_args()
    if args.verify:
        status=inspect_package(args.verify);print(status)
        if status!='complete':raise SystemExit(1)
        manifest=json.loads((args.verify/'handoff_manifest.json').read_text())
        for name,digest in manifest['file_sha256'].items():
            if file_hash(name)!=digest:raise ValueError('Source evidence changed: '+name)
        print('SOURCE_PROVENANCE_PASS')
    else:print(build_package())


if __name__=='__main__':main()
