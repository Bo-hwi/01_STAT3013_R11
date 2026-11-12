"""Descriptive five-policy Decision Matrix from sealed M4 artifacts; no training."""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from . import p4_policy_evaluation as p4
from . import policy_evaluation as direct
from . import paired_bootstrap as bootstrap
from .quantile_contracts import file_hash,verify_frozen_files,json_snapshot,stable_mean
from .quantile_models import pinball_loss
from .cost_simulation import cost_index,newsvendor_cost

ROOT=p4.ROOT
P4_SOURCE='results/experiments/m4_p4_costs_20261109T060634Z_b0b91a16722f4130bfd89a887a8cc19f'
BOOTSTRAP_SOURCE='results/experiments/m4_bootstrap_p2_p3_20261108T142054Z_cf6d0fc30677'
POLICIES=['P1','P2','P3','P4_M0','P4_M3']
CELL=['family','city','scenario'];FOLDS=['F1','F2','F3','F4']
RULE=dict(version='phase4a_sign_only_v1',required_folds=FOLDS,
          favorable='all four finite fold mean deltas strictly negative',
          unfavorable='all four finite fold mean deltas strictly positive',
          mixed='zeros or opposing signs; or disagreement between weather comparisons',
          insufficient='missing fold or nonfinite delta',consolidated='both comparisons must agree',
          declaration='Phase4A before aggregation; not preregistered before test experiments',
          deployment_claim=False,cell_confidence_intervals=False,multiple_testing_adjustment=False)
QA='retrospective_ERA5; sales_derived_closure_masks; upstream_QA09_wind_excluded; operational_availability_unverified'
LIMITS='F1 exploratory/post-pilot; F2-F4 exposure verification pending; dependent observations; censored sales proxy; P4 nonindependent calibration/refit mismatch; M4_REPRODUCED_M3'


def classify_fold_deltas(rows,column):
    if set(rows.fold)!=set(FOLDS) or rows.fold.duplicated().any() or rows.sample_size.le(0).any():return 'insufficient_evidence'
    values=rows[column].to_numpy(float)
    if not np.isfinite(values).all():return 'insufficient_evidence'
    if (values<0).all():return 'directionally_favorable'
    if (values>0).all():return 'directionally_unfavorable'
    return 'mixed_or_inconsistent'


def combined_direction(a,b):
    if 'insufficient_evidence' in (a,b):return 'insufficient_evidence'
    return a if a==b else 'mixed_or_inconsistent'


def validate_supported_uncertainty(intervals):
    if not set(intervals.scope).issubset(set(FOLDS+['pooled_F1_F4'])):raise ValueError('Unsupported bootstrap scope')
    if any(c in intervals and intervals[c].notna().any() for c in ['family','city']):raise ValueError('No family/city bootstrap intervals exist')
    if not set(intervals.metric).issubset({'mean_cost_delta_P3_minus_P2','relative_cost_delta_pct'}):raise ValueError('Unsupported bootstrap estimand')
    if intervals.duplicated(['scope','scenario','variant','block_length','metric']).any():raise ValueError('Duplicate bootstrap summaries')


def paired_costs(baseline,direct_rows,p4_rows,expected,config,variant='corrected'):
    """Exact one-to-one observation matching, including all-policy missing rows."""
    frames=[]
    for scenario in direct.configured_scenarios(config):
        chosen={'P1':baseline.loc[baseline.scenario.eq(scenario['scenario'])]}
        for label in ['P2','P3']:
            chosen[label]=direct_rows.loc[direct_rows.scenario.eq(scenario['scenario'])&direct_rows.policy.eq(label)&direct_rows.variant.eq(variant)]
        for cfg in ['M0','M3']:chosen['P4_'+cfg]=p4_rows.loc[p4_rows.scenario.eq(scenario['scenario'])&p4_rows.config.eq(cfg)]
        result=None
        for label,rows in chosen.items():
            if rows.empty or not rows.tau_key.eq(scenario['tau_key']).all():raise ValueError('Missing policy/scenario or incorrect tau')
            p4._matched_test(expected,rows)
            required=p4.MATCH+['y_true','quantity','simulated_cost']
            if rows[required].isna().any().any() or rows.duplicated(p4.MATCH).any():raise ValueError('Invalid policy keys')
            values=rows[required].copy()
            costs=newsvendor_cost(values.y_true,values.quantity,scenario['c_over'],scenario['c_under'])
            if not np.array_equal(costs,values.simulated_cost):raise ValueError('Source row costs fail independent recomputation')
            values=values.rename(columns={'y_true':'y_true_'+label,'quantity':'Q_'+label,'simulated_cost':label})
            result=values if result is None else result.merge(values,on=p4.MATCH,how='outer',validate='1:1')
        if result.isna().any().any() or any(not np.array_equal(result.y_true_P1,result['y_true_'+p]) for p in POLICIES):raise ValueError('Policy targets/keys differ')
        result['y_true']=result.y_true_P1;result['scenario']=scenario['scenario'];result['variant']=variant;frames.append(result)
    return pd.concat(frames,ignore_index=True)


def verify_quantile_link(panel,direct_rows):
    for config,policy in [('M0','P2'),('M3','P3')]:
        point=panel.loc[panel.split.eq('test')&panel.config.eq(config)]
        for variant,column in [('corrected','corrected_prediction'),('raw_clipped_sensitivity','raw_prediction')]:
            economic=direct_rows.loc[direct_rows.policy.eq(policy)&direct_rows.variant.eq(variant)]
            pair=point[p4.MATCH+['y_true',column]].merge(economic[p4.MATCH+['y_true','quantity']],on=p4.MATCH,
                how='outer',validate='1:1',suffixes=('_forecast','_economic'),indicator=True)
            if (not pair['_merge'].eq('both').all() or not np.array_equal(pair.y_true_forecast,pair.y_true_economic)
                    or not np.array_equal(np.maximum(pair[column],0),pair.quantity)):
                raise ValueError('Quantile diagnostics and economic forecasts differ')


def verify_bootstrap_points(primary,raw,intervals):
    validate_supported_uncertainty(intervals)
    for _,r in intervals.iterrows():
        wide=primary if r.variant=='corrected' else raw
        g=wide.loc[wide.scenario.eq(r.scenario)]
        if r.scope!='pooled_F1_F4':g=g.loc[g.fold.eq(r.scope)]
        values=_metrics(g)
        point=values['delta_P3_minus_P2'] if r.metric=='mean_cost_delta_P3_minus_P2' else values['relative_delta_pct_P3_minus_P2']
        if len(g)!=r.n_observations or not np.isclose(point,r.point_estimate,rtol=1e-9,atol=1e-9):
            raise ValueError('Bootstrap observation counts/point estimates differ from matched sources')


def attach_annotations(matrix,annotations,families):
    if (annotations.family.isna().any() or annotations.family.duplicated().any() or set(annotations.family)!=set(families)
            or annotations.econometric_signature.isna().any()):raise ValueError('Require unique complete family-level M2 annotation')
    ann=annotations.rename(columns={c:'m2_rq1_'+c for c in annotations if c!='family'})
    result=matrix.merge(ann,on='family',validate='m:1',how='left')
    result['m2_annotation_grain']='family only; repeated context, not city evidence'
    return result


def _metrics(rows):
    record=dict(sample_size=len(rows),n_dates=rows.date.nunique(),n_stores=rows.store_nbr.nunique(),
                zero_rate=float(rows.y_true.eq(0).mean()))
    for policy in POLICIES:
        record['mean_cost_'+policy]=stable_mean(rows[policy].to_numpy())
        record['cost_index_'+policy]=cost_index(rows[policy],rows.P1)
    for tag,left,right in [('P3_minus_P2','P3','P2'),('P4_M3_minus_M0','P4_M3','P4_M0')]:
        record['delta_'+tag]=record['mean_cost_'+left]-record['mean_cost_'+right]
        record['relative_delta_pct_'+tag]=100*(cost_index(rows[left],rows[right])-1)
    return record


def aggregate_matrix(wide,annotations,families,cities,scenarios):
    if wide.duplicated(p4.MATCH+['scenario']).any():raise ValueError('Duplicate consolidated observations')
    if not np.isfinite(wide[POLICIES+['y_true']].to_numpy(float)).all():raise ValueError('Nonfinite consolidated observations')
    if not set(wide.fold).issubset(FOLDS):raise ValueError('Unknown fold')
    folds=[];cells=[]
    for key,g in wide.groupby(CELL+['fold'],observed=True,sort=True):folds.append(dict(zip(CELL+['fold'],key),**_metrics(g)))
    fold_table=pd.DataFrame(folds)
    for key,g in wide.groupby(CELL,observed=True,sort=True):
        group=fold_table
        for name,value in zip(CELL,key):group=group.loc[group[name].eq(value)]
        r=dict(zip(CELL,key),**_metrics(g));r['fold_coverage']=','.join(sorted(g.fold.unique()));r['n_folds']=g.fold.nunique()
        r['contains_store52']=g.store_nbr.astype(str).eq('52').any()
        for tag in ['P3_minus_P2','P4_M3_minus_M0']:
            col='delta_'+tag;r['direction_'+tag]=classify_fold_deltas(group,col)
            r['favorable_folds_'+tag]=int(group[col].lt(0).sum());r['unfavorable_folds_'+tag]=int(group[col].gt(0).sum());r['zero_folds_'+tag]=int(group[col].eq(0).sum())
        r['weather_class']=combined_direction(r['direction_P3_minus_P2'],r['direction_P4_M3_minus_M0']);cells.append(r)
    grid=pd.MultiIndex.from_product([families,cities,scenarios],names=CELL).to_frame(index=False)
    matrix=grid.merge(pd.DataFrame(cells),on=CELL,how='left',validate='1:1')
    absent=matrix.sample_size.isna()
    matrix.loc[absent,['sample_size','n_dates','n_stores','n_folds']]=0
    for col in ['direction_P3_minus_P2','direction_P4_M3_minus_M0','weather_class']:matrix.loc[absent,col]='insufficient_evidence'
    matrix['source_QA_flag']=QA
    matrix.loc[matrix.contains_store52.eq(True),'source_QA_flag']+='; limited_store52_history'
    matrix['methodological_limitations']=LIMITS
    matrix['confidence_interval']='UNAVAILABLE_AT_FAMILY_CITY_GRAIN'
    matrix['uncertainty_flag']='descriptive_only; no_cell_CI_or_significance'
    matrix['P1_comparison_uncertainty']='UNAVAILABLE_NOT_ANALYZED';matrix['P4_uncertainty']='UNAVAILABLE_NOT_ANALYZED'
    matrix['formal_significance_claim']=False;matrix['multiple_testing_adjustment']='none_for_RQ2_matrix'
    matrix['recommendation']='descriptive research comparison only; no deployment claim'
    matrix['interpretation']='forecast metrics, simulated costs and weather deltas are separate evidence domains'
    return attach_annotations(matrix,annotations,families),fold_table


def forecast_diagnostics(panel):
    records=[];test=panel.loc[panel.split.eq('test')]
    for (family,city,config,tau_key),g in test.groupby(['family','city','config','tau_key'],observed=True,sort=True):
        for variant in ['raw','corrected']:
            pred=g[variant+'_prediction'].to_numpy();truth=g.y_true.to_numpy();tau=float(g.tau.iloc[0])
            records.append(dict(family=family,city=city,config=config,tau_key=tau_key,variant=variant,sample_size=len(g),
                pinball_loss=stable_mean(pinball_loss(truth,pred,tau)),coverage=float(np.mean(truth<=pred)),
                strict_less_rate=float(np.mean(truth<pred)),tie_rate=float(np.mean(truth==pred)),coverage_gap=float(np.mean(truth<=pred)-tau),
                crossing_rate=float(g[variant+'_crossing'].mean()),negative_rate=float(np.mean(pred<0))))
    return pd.DataFrame(records)


def enrich_forecasts(matrix,diagnostics,point_panel,config):
    result=matrix.copy();mapping={s['scenario']:s['tau_key'] for s in direct.configured_scenarios(config)}
    result['tau_key']=result.scenario.map(mapping)
    for cfg in ['M0','M3']:
        fields=['pinball_loss','coverage','coverage_gap','strict_less_rate','tie_rate','crossing_rate','negative_rate']
        d=diagnostics.loc[diagnostics.config.eq(cfg)&diagnostics.variant.eq('corrected'),['family','city','tau_key']+fields]
        result=result.merge(d.rename(columns={c:cfg+'_quantile_'+c for c in fields}),on=['family','city','tau_key'],how='left',validate='m:1')
    point=[]
    for (family,city,cfg),g in point_panel.loc[point_panel.split.eq('test')].groupby(['family','city','config'],observed=True):
        point.append(dict(family=family,city=city,config=cfg,point_clipped_MAE=stable_mean(np.abs(g.y_true-g.clipped_prediction)),sample_size=len(g),source_label='M4_REPRODUCED_M3'))
    point=pd.DataFrame(point)
    for cfg in ['M0','M3']:
        d=point.loc[point.config.eq(cfg),['family','city','point_clipped_MAE']].rename(columns={'point_clipped_MAE':'P4_'+cfg+'_point_MAE'})
        result=result.merge(d,on=['family','city'],how='left',validate='m:1')
    result['baseline_error']=result.M0_quantile_pinball_loss;result['weather_error']=result.M3_quantile_pinball_loss
    result['forecast_error_metric']='corrected quantile pinball loss at scenario tau; not P4 point MAE'
    result['delta_error_pct']=100*(result.weather_error/result.baseline_error.where(result.baseline_error.ne(0))-1)
    return result,point


def policy_summary(wide):
    records=[]
    for scenario,g in wide.groupby('scenario'):
        for policy in POLICIES:records.append(dict(scenario=scenario,policy=policy,sample_size=len(g),mean_simulated_cost=stable_mean(g[policy].to_numpy()),P1_cost_index=cost_index(g[policy],g.P1)))
    result=pd.DataFrame(records);result['descriptive_rank']=result.groupby('scenario').mean_simulated_cost.rank(method='min').astype(int)
    result['rank_interpretation']='test descriptive cost ranking; not validated deployment choice'
    return result


def weather_summary(wide,intervals):
    validate_supported_uncertainty(intervals);records=[]
    for scenario,g in wide.groupby('scenario'):
        metrics=_metrics(g)
        for tag in ['P3_minus_P2','P4_M3_minus_M0']:
            record=dict(scenario=scenario,comparison=tag,mean_delta=metrics['delta_'+tag],relative_delta_pct=metrics['relative_delta_pct_'+tag],
                        sample_size=len(g),lower=np.nan,upper=np.nan,uncertainty='unavailable for P4')
            if tag=='P3_minus_P2':
                ci=intervals.loc[intervals.scope.eq('pooled_F1_F4')&intervals.scenario.eq(scenario)&intervals.variant.eq('corrected')&intervals.block_length.eq(7)&intervals.metric.eq('relative_cost_delta_pct')]
                if len(ci)!=1:raise ValueError('Missing supported pooled bootstrap interval')
                if not np.isclose(ci.point_estimate.iloc[0],record['relative_delta_pct'],rtol=1e-10,atol=1e-10):raise ValueError('Bootstrap point estimate differs from matched source costs')
                record.update(lower=ci.lower.iloc[0],upper=ci.upper.iloc[0],uncertainty='descriptive pooled P2/P3 bootstrap; L7; no cell inference')
            records.append(record)
    return pd.DataFrame(records)


def cell_summary(matrix,rankings):
    records=[]
    for (family,city),g in matrix.groupby(['family','city'],sort=True):
        r=dict(family=family,city=city,sample_size=int(g.sample_size.iloc[0]),fold_coverage=g.fold_coverage.iloc[0],
               zero_rate=g.zero_rate.iloc[0],weather_signature=g.m2_rq1_econometric_signature.iloc[0])
        for _,row in g.iterrows():
            r['direction_'+row.scenario]=row.weather_class
            r['P4_M0_index_'+row.scenario]=row['cost_index_P4_M0'];r['P4_M3_index_'+row.scenario]=row['cost_index_P4_M3']
        r['uncertainty']='No cell CI; no deployment claim';records.append(r)
    return pd.DataFrame(records)


def load_sources(root=ROOT):
    root=Path(root);sources=p4.load_sources(root)
    directory=root/P4_SOURCE
    if p4.inspect_bundle(directory)!='complete':raise ValueError('P4 cost bundle incomplete/corrupt')
    manifest=json.loads((directory/'manifest.json').read_text());receipt=manifest['approval']
    if receipt['status']!='APPROVED' or receipt['request_sha256']!=p4.semantic_hash(receipt['request']):raise ValueError('P4 historical approval invalid')
    verify_frozen_files({'file_sha256':receipt['request']['file_sha256'],'input':{}})
    p4_rows=pd.read_csv(directory/'p4_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    quantile,_,hashes=direct.load_verified_source(root,p4.reproduction.FINAL)
    verify_quantile_link(quantile,sources['direct'])
    boot=root/BOOTSTRAP_SOURCE
    if bootstrap.inspect_bootstrap(boot)!='complete':raise ValueError('Bootstrap bundle incomplete/corrupt')
    boot_manifest=json.loads((boot/'manifest.json').read_text())
    verify_frozen_files({'file_sha256':boot_manifest['file_sha256'],'input':{}});verify_frozen_files(boot_manifest['source_forecast_provenance'])
    intervals=pd.read_csv(boot/'intervals.csv',float_precision='round_trip');validate_supported_uncertainty(intervals)
    annotations=pd.read_csv(root/'results/tables/weather_signatures_for_m4.csv',keep_default_na=False)
    families=p4.reproduction.read_yaml(root/'configs/research_protocol.yaml')['scope']['deep_study_families']
    cities=sorted(pd.read_csv(root/'data/raw/favorita/stores.csv').city.unique())
    if len(families)!=7 or len(cities)!=22:raise ValueError('Frozen family/city scope mismatch')
    files={**sources['files'],**hashes,**receipt['request']['file_sha256']}
    for folder in [directory,boot]:files.update({str(f.resolve()):file_hash(f) for f in folder.iterdir() if f.is_file()})
    for f in [Path(__file__).resolve(),root/'docs/m4_phase4a_matrix.md',root/'results/tables/weather_signatures_for_m4.csv',root/'data/raw/favorita/stores.csv']:
        files[str(f.resolve())]=file_hash(f)
    return sources,p4_rows,quantile,intervals,annotations,families,cities,files,boot_manifest


def validate_matrix(matrix,families,cities,scenarios):
    grid=pd.MultiIndex.from_product([families,cities,scenarios],names=CELL)
    if matrix.duplicated(CELL).any() or set(map(tuple,matrix[CELL].to_numpy()))!=set(grid):raise ValueError('Matrix scope/uniqueness failure')
    if not matrix.confidence_interval.eq('UNAVAILABLE_AT_FAMILY_CITY_GRAIN').all() or matrix.formal_significance_claim.any():raise ValueError('Invalid cell uncertainty claim')
    if any(c.endswith(('_ci_lower','_ci_upper')) for c in matrix):raise ValueError('Unsupported cell CI columns')
    if not matrix.recommendation.eq('descriptive research comparison only; no deployment claim').all():raise ValueError('Invalid deployment recommendation')


def inspect_bundle(directory):
    directory=Path(directory)
    if (directory/'FAILED.json').exists():return 'failed'
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text())
        if not marker['files'] or any(Path(n).name!=n or file_hash(directory/n)!=h for n,h in marker['files'].items()):return 'corrupt'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['output_sha256']!={n:h for n,h in marker['files'].items() if n!='manifest.json'}:return 'corrupt'
        return 'complete'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete'


def markdown(frame):
    """Small report table without optional tabulate dependency."""
    if isinstance(frame,pd.Series):frame=frame.rename('count').rename_axis('classification').reset_index()
    def cell(v):
        if pd.isna(v):return 'NA'
        return (f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def figures(stage,matrix,rankings,families,cities,scenarios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for tag in ['P3_minus_P2','P4_M3_minus_M0']:
        column='relative_delta_pct_'+tag;limit=float(matrix[column].abs().max()) or 1.
        fig,axes=plt.subplots(3,1,figsize=(17,12),layout='constrained')
        for scenario,ax in zip(scenarios,axes):
            table=matrix.loc[matrix.scenario.eq(scenario)].pivot(index='family',columns='city',values=column).reindex(index=families,columns=cities)
            im=ax.imshow(table,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_yticks(range(len(families)),families);ax.set_xticks(range(len(cities)),cities,rotation=55,ha='right',fontsize=8)
            ax.set_title(scenario+' | '+tag+' | negative = lower simulated cost')
        fig.colorbar(im,ax=axes,label='Relative paired cost delta (%)')
        fig.suptitle('Descriptive weather cost differences; no family-city confidence intervals')
        for ext in ['png','svg']:fig.savefig(stage/('weather_'+tag+'.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for scenario,ax in zip(scenarios,axes):
        data=rankings.loc[rankings.scenario.eq(scenario)].set_index('policy').reindex(POLICIES)
        ax.bar(POLICIES,data.P1_cost_index,color=['gray','#3b82f6','#14b8a6','#a78bfa','#f59e0b']);ax.axhline(1,color='black',lw=.8)
        ax.set_title(scenario);ax.tick_params(axis='x',rotation=40);ax.set_ylabel('Simulated cost index (P1=1)')
    fig.suptitle('Descriptive pooled policy comparison; no deployment or uncertainty claim')
    for ext in ['png','svg']:fig.savefig(stage/('policy_comparison.'+ext),dpi=180)
    plt.close(fig)


def run_matrix(root=ROOT):
    root=Path(root);sources,rows,quantile,intervals,annotations,families,cities,files,boot_manifest=load_sources(root)
    parent=root/'results/experiments';destination=parent/('m4_decision_matrix_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=parent))
    frozen=dict(classification_rule=RULE,file_sha256=files,source_P4_provenance='M4_REPRODUCED_M3',bootstrap_design=boot_manifest['design'],limitations=LIMITS)
    p4.reproduction._json(stage/'STARTED.json',frozen)
    try:
        expected=sources['expected'].loc[sources['expected'].split.eq('test')]
        wide=paired_costs(sources['baseline'],sources['direct'],rows,expected,sources['config'])
        raw=paired_costs(sources['baseline'],sources['direct'],rows,expected,sources['config'],'raw_clipped_sensitivity')
        verify_bootstrap_points(wide,raw,intervals)
        scenarios=[s['scenario'] for s in direct.configured_scenarios(sources['config'])]
        matrix,folds=aggregate_matrix(wide,annotations,families,cities,scenarios)
        raw_matrix,_=aggregate_matrix(raw,annotations,families,cities,scenarios)
        diagnostics=forecast_diagnostics(quantile)
        matrix,point=enrich_forecasts(matrix,diagnostics,sources['panel'],sources['config'])
        rankings=policy_summary(wide);weather=weather_summary(wide,intervals);summary=cell_summary(matrix,rankings)
        validate_matrix(matrix,families,cities,scenarios)
        if len(matrix)!=462 or len(summary)!=154:raise ValueError('Expected 462 scenario rows / 154 cells')
        tables={'matrix':matrix,'matrix_raw_sensitivity':raw_matrix,'family_city_summary':summary,'fold_consistency':folds,
            'forecast_diagnostics':diagnostics,'point_forecast_diagnostics':point,'policy_summary':rankings,
            'weather_summary':weather,'supported_bootstrap':intervals}
        for name,frame in tables.items():
            with (stage/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
            loaded=pd.read_csv(stage/(name+'.csv'),float_precision='round_trip',keep_default_na=False,na_values=[''])
            pd.testing.assert_frame_equal(loaded,frame.reset_index(drop=True),check_dtype=False)
        with (stage/'family_city_summary.html').open('x',encoding='utf-8') as f:
            f.write('<meta charset="utf-8"><style>body{font:14px sans-serif}table{border-collapse:collapse}th{position:sticky;top:0;background:#e2e8f0}td,th{padding:6px;border:1px solid #ccc}</style><h1>154 descriptive family-city cells</h1><p>No cell confidence intervals or deployment claims. P4 source: M4_REPRODUCED_M3.</p>'+summary.to_html(index=False))
        with (stage/'policy_summary.md').open('x',encoding='utf-8') as f:f.write(markdown(rankings))
        p4.reproduction._json(stage/'source_QA.json',dict(global_flags=QA,limitations=LIMITS,preflight_evidence='verified in frozen upstream manifests',cell_confidence_intervals='unavailable',P4_P1_comparison_uncertainty='unavailable'))
        figures(stage,matrix,rankings,families,cities,scenarios)
        with (stage/'M4_RESULTS.md').open('x',encoding='utf-8') as f:
            f.write('# M4 descriptive results\n\n462 family-city-scenario rows; 154 family-city summaries. All policies evaluated on matched frozen test observations.\n\n'+markdown(rankings)+'\n\nWeather differences (P2/P3 bounds only pooled; P4 bounds unavailable):\n\n'+markdown(weather)+'\n\nConsolidated classifications:\n\n'+markdown(matrix.weather_class.value_counts())+'\n\n'+LIMITS+'\n\nNo actual costs/profit/ROI, causal, significance or deployment claims. M2 annotations are family-level RQ1 context. See supported_bootstrap.csv for supported pooled/fold uncertainty and block-length sensitivity only.\n')
        verify_frozen_files({'file_sha256':files,'input':{}})
        outputs={f.name:file_hash(f) for f in stage.iterdir() if f.is_file()}
        p4.reproduction._json(stage/'manifest.json',dict(frozen,matrix_rows=len(matrix),summary_rows=len(summary),
            classification_counts=matrix.weather_class.value_counts().to_dict(),packages={p:version(p) for p in ['numpy','pandas','matplotlib']},output_sha256=outputs))
        p4.reproduction._json(stage/'COMPLETE.json',{'files':{**outputs,'manifest.json':file_hash(stage/'manifest.json')}})
        if inspect_bundle(stage)!='complete':raise ValueError('Matrix staging integrity failure')
        if destination.exists():raise FileExistsError('Never overwrite previous matrix')
        stage.rename(destination);return destination
    except Exception as e:
        p4.reproduction._json(stage/'FAILED.json',{'type':type(e).__name__,'message':str(e)});raise


def main():
    argparse.ArgumentParser(description=__doc__).parse_args();print(run_matrix())


if __name__=='__main__':main()
