"""Official-test P4 handoff from explicit sealed inputs; never train or evaluate Q."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from . import consolidated_matrix as matrix
from . import decision_matrix_canonical as canonical
from . import fixed_week_bootstrap as fixed
from . import quantile_contracts as contracts
from . import uncertainty_handoff as uncertainty

LABELS={'Pilot','Validate','No-go'}
LABEL_MAP={'Pilot':'Prioritize pilot','Validate':'Further validation','No-go':'Do not deploy'}


def json_text(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)


def verify_archive(directory):
    if directory is None:return {}
    directory=Path(directory).resolve()
    marker=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    for name,digest in marker['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or contracts.file_hash(path)!=digest:raise ValueError('Historical archive corrupt: '+name)
    record=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    return {name:(directory/entry['relative'],entry['sha256']) for name,entry in record['files'].items()}


def verify_bundle(directory,archive=None):
    directory=Path(directory).resolve()
    if not directory.is_dir():raise FileNotFoundError('Missing completed bundle: '+str(directory))
    if (directory/'FAILED.json').exists():raise ValueError('Failed bundle: '+str(directory))
    marker=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    files={}
    for name,digest in marker['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or contracts.file_hash(path)!=digest:raise ValueError('Bundle checksum mismatch: '+str(path))
        files[str(path)]=digest
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    def maps(value):
        if isinstance(value,dict):
            for key,v in value.items():
                if key=='file_sha256':yield v
                else:yield from maps(v)
        elif isinstance(value,list):
            for v in value:yield from maps(v)
    unavailable=[];historical=[]
    for mapping in maps(manifest):
        for name,digest in mapping.items():
            path=Path(name)
            if path.is_file() and contracts.file_hash(path)==digest:continue
            if archive and name in archive and archive[name][1]==digest:
                if contracts.file_hash(archive[name][0])!=digest:raise ValueError('Archived evidence drift')
                historical.append(name);continue
            if not path.is_file():unavailable.append(name);continue
            raise ValueError('Historical evidence differs; provide explicit pre-migration archive: '+name)
    approval=manifest.get('approval')
    if approval:
        request=approval.get('request')
        if approval.get('status')!='APPROVED' or request is None or approval.get('request_sha256')!=contracts.semantic_hash(request):
            raise ValueError('Invalid historical execution receipt')
    return manifest,files,dict(archived_evidence=sorted(set(historical)),unavailable_historical_inputs=sorted(set(unavailable)),
        note='Completed output checksums verified. Missing historical training inputs are not independently revalidated and no training is rerun.')


def load_inputs(p4_bundle,p1_bundle,direct_bundle,quantile_bundle,official_predictions,annotations,historical_archive=None):
    archive=verify_archive(historical_archive);sources={};files={};qa={}
    if historical_archive:
        for name in ['manifest.json','COMPLETE.json']:
            path=(Path(historical_archive)/name).resolve();files[str(path)]=contracts.file_hash(path)
    for name,path in [('p4',p4_bundle),('p1',p1_bundle),('direct',direct_bundle),('quantile',quantile_bundle)]:
        m,f,q=verify_bundle(path,archive);files.update(f);sources[name]=m;qa[name]=q
    if sources['p4'].get('label')!='P4_HYBRID_M4_VALIDATION_OFFICIAL_M3_TEST':raise ValueError('P4 must use official-test hybrid cost bundle')
    request=sources['p4']['approval']['request']
    official_predictions=Path(official_predictions)
    if contracts.file_hash(official_predictions)!=request['specification']['official_sha256']:raise ValueError('Official prediction checksum differs from hybrid execution')
    files[str(official_predictions.resolve())]=contracts.file_hash(official_predictions)
    read=lambda p:pd.read_csv(p,parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    p4=read(Path(p4_bundle)/'p4_rows.csv');p1=read(Path(p1_bundle)/'p1_rows.csv');direct=read(Path(direct_bundle)/'row_costs.csv')
    if not p4.test_source.eq('OFFICIAL_M3_TEST').all():raise ValueError('Mixed P4 test-source labels')
    config=matrix.p4.reproduction.read_yaml(Path(__file__).resolve().parents[2]/'configs/cost_scenarios.yaml')
    expected=p1.loc[p1.scenario.eq('S1'),['date','store_nbr','family','city','fold','y_true']].assign(split='test')
    if expected.duplicated(matrix.p4.reproduction.KEY).any():raise ValueError('Duplicate expected scored keys')
    official=read(official_predictions)
    keys=['date','store_nbr','family','city','fold','config','model']
    for (_,cfg),g in p4.groupby(['scenario','config']):matrix.p4._matched_test(expected,g)
    check=p4.merge(official[keys+['y_true','y_pred']],on=keys,how='left',validate='m:1',suffixes=('','_official'))
    if check.y_pred.isna().any() or not np.array_equal(check.point_prediction,check.y_pred) or not np.array_equal(check.y_true,check.y_true_official):
        raise ValueError('P4 bases/targets do not match official values')
    # This is pairing/aggregation of existing costs, not a call to any cost evaluator.
    frames=[]
    for s in matrix.direct.configured_scenarios(config):
        policies={'P1':p1.loc[p1.scenario.eq(s['scenario'])]}
        for v in fixed.VARIANTS:
            for p in ['P2','P3']:policies[p]=direct.loc[direct.scenario.eq(s['scenario'])&direct.variant.eq(v)&direct.policy.eq(p)]
            for cfg in ['M0','M3']:policies['P4_'+cfg]=p4.loc[p4.scenario.eq(s['scenario'])&p4.config.eq(cfg)]
            result=None
            for name,g in policies.items():
                matrix.p4._matched_test(expected,g)
                if not g.tau_key.eq(s['tau_key']).all():raise ValueError('Scenario/tau mismatch')
                part=g[matrix.p4.MATCH+['y_true','simulated_cost']].rename(columns={'simulated_cost':name})
                if result is None:result=part
                else:
                    result=result.merge(part,on=matrix.p4.MATCH,validate='1:1',how='outer',suffixes=('','_other'))
                    if not np.array_equal(result.y_true,result.y_true_other):raise ValueError('Policy demand mismatch')
                    result=result.drop(columns='y_true_other')
            if result.isna().any().any() or not np.isfinite(result[list(fixed.POLICIES)].to_numpy()).all():raise ValueError('Invalid matched costs')
            frames.append(result.assign(scenario=s['scenario'],variant=v))
    panel=pd.concat(frames,ignore_index=True)
    qfiles=list(Path(quantile_bundle).glob('m4_quantiles_*.csv'))
    if len(qfiles)!=1:raise ValueError('Require one authoritative quantile prediction panel')
    quantiles=pd.read_csv(qfiles[0],parse_dates=['date','forecast_origin'],dtype={'store_nbr':str},float_precision='round_trip')
    contracts.validate_prediction_panel(quantiles)
    ann=pd.read_csv(annotations,keep_default_na=False)
    files[str(Path(annotations).resolve())]=contracts.file_hash(annotations)
    folds=request['selected_mapping']
    fold_specs=[]
    for fold in fixed.SPEC['folds']:
        row=next(r for r in folds if r['fold']==fold)
        fold_specs.append(dict(fold_id=fold,test=[row['test_start'],row['test_end']]))
    support=fixed.calendar_support(panel,fold_specs)
    if len(expected)!=41076 or len(support)!=154 or int(support.eligible.sum())!=140:raise ValueError('Frozen observation/calendar scope drift')
    comparisons=pd.read_csv(Path(p4_bundle)/'comparisons.csv',float_precision='round_trip')
    return dict(panel=panel,p4=p4,p1=p1,direct=direct,expected=expected,quantiles=quantiles,annotations=ann,config=config,comparisons=comparisons,
        folds=fold_specs,support=support,files=files,qa=qa,p4_source=str(Path(p4_bundle).resolve()))


def bootstrap_request(inputs):
    files=dict(inputs['files'])
    root=Path(__file__).resolve().parents[2]
    for name in ['src/decision/final_handoff.py','src/decision/fixed_week_bootstrap.py','src/decision/quantile_contracts.py',
                 'configs/research_protocol.yaml','configs/cost_scenarios.yaml']:
        path=root/name;files[str(path)]=contracts.file_hash(path)
    return dict(status='PROPOSED_NOT_APPROVED',purpose='Fixed-week downstream uncertainty for OFFICIAL-test hybrid P4 costs',
        p4_source=inputs['p4_source'],file_sha256=files,specification=fixed.SPEC,
        scope='six contrasts, 3 scenarios, corrected and direct-only raw sensitivity, pooled/F1-F4/140 complete cells; 14 incomplete cells CI unavailable',
        no_forecasting_fits=True,no_economic_recalculation=True,execution_authorized=False)


def run_bootstrap(inputs,approval,output):
    receipt=json.loads(Path(approval).read_text(encoding='utf-8'));request=receipt.get('request',{})
    current=bootstrap_request(inputs)
    if receipt.get('status')!='APPROVED' or request!=current or receipt.get('request_sha256')!=contracts.semantic_hash(current):
        raise ValueError('Hybrid downstream bootstrap requires explicit exact-request approval')
    for field in ['approved_by','approval_evidence','approval_communicated_date']:
        if not receipt.get(field):raise ValueError('Missing bootstrap approval evidence')
    for path,digest in current['file_sha256'].items():
        if contracts.file_hash(path)!=digest:raise ValueError('Frozen downstream input/code drift')
    result=fixed.bootstrap_supplement(inputs['panel'],inputs['folds'])
    return fixed.save_bundle(result,receipt,current,output,0.)


def recommendation(*,hard_qa_failure=False,complete=True,operational_verified=False,fold_direction='mixed_or_inconsistent',ci_upper=None,ci_lower=None):
    """Conservative Proposal §10 mapping; no tuned thresholds or forced label counts."""
    if hard_qa_failure:return 'No-go'
    if not complete or ci_lower is None or ci_upper is None:return 'Validate'
    if fold_direction=='directionally_unfavorable' and ci_lower>0:return 'No-go'
    if operational_verified and fold_direction=='directionally_favorable' and ci_upper<0:return 'Pilot'
    return 'Validate'


def make_tables(inputs,bootstrap=None):
    primary=inputs['panel'].loc[inputs['panel'].variant.eq('corrected')]
    families=sorted(primary.family.unique());cities=sorted(primary.city.unique())
    cells,folds=matrix.aggregate_matrix(primary,inputs['annotations'],families,cities,['S1','S3','S5'])
    diagnostics=matrix.forecast_diagnostics(inputs['quantiles'])
    points=inputs['p4'][['date','store_nbr','family','city','fold','model','config','y_true','point_prediction']].drop_duplicates().assign(split='test').rename(columns={'point_prediction':'clipped_prediction'})
    cells,_=matrix.enrich_forecasts(cells,diagnostics,points,inputs['config'])
    cells['methodological_limitations']=cells.methodological_limitations.str.replace('M4_REPRODUCED_M3','M4 validation residuals + OFFICIAL M3 rounded test',regex=False)
    base=canonical.canonical_matrix(cells,inputs['config'])
    if bootstrap:
        manifest,_,_=verify_bundle(bootstrap)
        if manifest['approval']['request'].get('p4_source')!=inputs['p4_source']:raise ValueError('P4 bootstrap belongs to a different/historical source')
        ci=pd.read_csv(Path(bootstrap)/'intervals.csv',float_precision='round_trip')
        base=uncertainty.integrate_matrix(base,ci,inputs['support'])
    else:
        # Valid old P3-P2 CI may be reused only for its exact unchanged cost/key estimand.
        base['confidence_interval']=base.apply(lambda r:json_text(dict(status='P4_HYBRID_CI_PENDING_APPROVAL',
            economic_P4_CI=None,forecast_error_CI=None,reason='Historical reproduced-P4 CI cannot label official-P4 costs',
            calendar_complete=bool(inputs['support'].set_index(['family','city']).loc[(r.family,r.city),'eligible']))),axis=1)
    base['recommendation']='Validate'
    base['uncertainty_flag']+='; official_P4_hybrid; no_multiplicity_control'
    base['interpretation']='Validate = Further validation under Proposal §10. Official M3 rounded test base + unchanged M4 train-only residual offsets. Labels are post-test descriptive triage, not preregistered or deployment authorization. Operational weather availability/prior exposure unverified.'
    # All labels are Validate for the CURRENT evidence; other supported labels are tested, not forced into data.
    cost=inputs['comparisons'].copy()
    cost['P4_source']='OFFICIAL_M3_TEST + M4_TRAIN_ONLY_VALIDATION'
    if bootstrap:
        lookup=uncertainty.ci_lookup(ci.fillna({'family':'','city':''}));evidence=[]
        for r in cost.fillna({'fold':'','family':'','city':''}).itertuples():
            scope='pooled_F1_F4' if r.report_level=='pooled' else r.fold if r.report_level=='fold' else 'family_city_pooled' if r.report_level=='family_city' else None
            if scope:
                g=lookup[(scope,r.family if scope=='family_city_pooled' else '',r.city if scope=='family_city_pooled' else '',r.scenario,r.variant)]
                records=[]
                for item in g.itertuples():
                    records.append(dict(metric=item.metric,point_estimate=item.point_estimate,lower=None if pd.isna(item.lower) else item.lower,upper=None if pd.isna(item.upper) else item.upper,
                        confidence_level=.95,valid_replicates=int(item.valid_replicates),status=item.ci_status,sampling='fixed_calendar_week_paired_1000_seed42',denominator=item.denominator))
                evidence.append(json_text(records))
            else:evidence.append(json_text(dict(status='UNAVAILABLE_AT_THIS_GRAIN')))
        cost['fixed_week_uncertainty']=evidence
    else:cost['fixed_week_uncertainty']=json_text(dict(status='P4_HYBRID_CI_PENDING_APPROVAL'))
    if list(base)!=canonical.MCOLS or len(base)!=154 or base.duplicated(['family','city']).any() or not set(base.recommendation).issubset(LABELS):raise ValueError('Final Proposal schema/key/label failure')
    return base,cost,cells,folds


def export(inputs,output_dir,bootstrap=None):
    output=Path(output_dir).resolve()
    if output.exists():raise FileExistsError('Output exists; never overwrite without explicit migration')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.staging-',dir=output.parent))
    try:
        base,cost,cells,folds=make_tables(inputs,bootstrap)
        for name,frame in [('decision_matrix',base),('cost_scenarios',cost),('scenario_cells',cells),('fold_costs',folds)]:
            p=stage/'results/tables'/(name+'.csv');p.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(p,index=False)
        generate_report(stage,base,cost,inputs,bootstrap)
        files={str(p.relative_to(stage)).replace('\\','/'):contracts.file_hash(p) for p in stage.rglob('*') if p.is_file()}
        manifest=dict(P4_source=inputs['p4_source'],file_sha256=inputs['files'],output_sha256=files,
            bootstrap=str(bootstrap) if bootstrap else None,qa=inputs['qa'],model_fits=0,economic_evaluation_reruns=0,
            label_map=LABEL_MAP,readiness='READY_FOR_TEAM_REVIEW' if bootstrap else 'RQ2_POINT_HANDOFF_READY_P4_CI_PENDING',
            created_at_utc=datetime.now(timezone.utc).isoformat())
        fixed.write_json(stage/'manifest.json',manifest)
        fixed.write_json(stage/'COMPLETE.json',dict(files={**files,'manifest.json':contracts.file_hash(stage/'manifest.json')}))
        output.parent.mkdir(parents=True,exist_ok=True);stage.rename(output);return output
    except Exception as error:
        fixed.write_json(stage/'FAILED.json',dict(type=type(error).__name__,message=str(error)));raise


def generate_report(stage,base,cost,inputs,bootstrap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=stage/'reports/figures';figures.mkdir(parents=True)
    primary=cost.loc[cost.variant.eq('corrected')&cost.report_level.eq('pooled')];records=[]
    for scenario,g in primary.groupby('scenario'):
        m0=g.loc[g.config.eq('M0')].iloc[0];m3=g.loc[g.config.eq('M3')].iloc[0]
        records.append(dict(scenario=scenario,P1=m0.cost_index_P1,P2=m0.cost_index_P2,P3=m0.cost_index_P3,P4_M0=m0.cost_index_P4,P4_M3=m3.cost_index_P4))
    table=pd.DataFrame(records).set_index('scenario').reindex(columns=matrix.POLICIES)
    fig,ax=plt.subplots(figsize=(8,4))
    table.plot.bar(ax=ax);ax.set_ylabel('Simulated cost index (P1 = 1)');ax.set_title('Matched official-test hybrid P4; descriptive comparison');ax.tick_params(axis='x',rotation=0);fig.tight_layout();fig.savefig(figures/'cost_scenarios.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3));base.recommendation.value_counts().reindex(['Pilot','Validate','No-go'],fill_value=0).plot.bar(ax=ax)
    ax.set_ylabel('Family–city cells');ax.set_title('Proposal §10 research-triage labels; no deployment claim');ax.tick_params(axis='x',rotation=0);fig.tight_layout();fig.savefig(figures/'cost_decision_matrix.png',dpi=160);plt.close(fig)
    manuscript=stage/'reports/manuscript/04_rq2_decision_value.md';manuscript.parent.mkdir(parents=True)
    text='''# M4 — RQ2 Quantile & Decision Value

## Methods
Direct quantiles use HistGradientBoostingRegressor(loss="quantile"), tau .50/.75/5⁄6,
seed42, M0/M3 and shared F1-F4; two-candidate validation-only raw pinball selection,
train-only preprocessing and fresh train+validation refit. Corrections clip Q>=0
and sort quantiles; raw forecasts remain in Drive bundles. P1=Seasonal Naive with
validation-residual quantile; P2/P3=direct M0/M3 quantiles. P4 uses ACTUAL official
M3 rounded test forecasts, with unchanged M4 train-only clipped-validation residual
quantiles (linear, n>=30, fold/config/family; fallback fold/config). No official raw
forecast is invented. Exact 4-decimal agreement is not full-precision equality.

Simulated C=c_over*(Q-D)+ + c_under*(D-Q)+; ratios 1:1/1:3/1:5. Indices are ratios
of cost sums on identical 41,076 scored test observations, not mean row ratios.
MAE weights over/under symmetrically whereas Newsvendor cost is asymmetric, so
lower point MAE need not yield lower simulated cost. Pinball and point MAE differ.

## Results — Newsvendor Cost Comparison
'''+matrix.markdown(table.reset_index())+'''

Values above and cost_scenarios.png come from the SAME verified stored costs.
Raw-clipped sensitivity changes P2/P3 only and stays separate in cost_scenarios.csv.
All existing fold-level results and 462 scenario cells are provided beside the
154x15 canonical Decision Matrix. M2 weather signatures are family-level context,
not city-specific causal evidence. Pilot/Validate/No-go are short display aliases
for Prioritize pilot/Further validation/Do not deploy. All actual cells remain
Validate because operational availability and exposure are unresolved; do not
force examples of each label or infer deployment from pointwise intervals.

## Robustness and uncertainty
'''+('Approved fixed-week CIs use this hybrid P4 cost source, four fixed weekly blocks/fold, 1,000 paired shared draws, seed42, 95% percentile. Only 140 complete 112-date cells receive their OWN CIs; 14 incomplete Esmeraldas/Salinas cells remain unavailable.' if bootstrap else 'Hybrid P4 CIs are UNAVAILABLE/PENDING downstream approval. Historical reproduced-P4 intervals are not attached to official P4. A concrete frozen downstream request is supplied; do not infer that tiny point differences preserve CIs.')+'''

## Limitations
Only four weekly blocks/fold; no selection/refit/calibration uncertainty or multiple-
testing control. F1 was exposed in pilots; F2-F4 prior exposure unverified. ERA5 is
retrospective with unverified operational availability. Observed sales may be
censored by unobserved stockouts, hence a demand proxy. Costs/indices are not
money, profit, ROI or causal weather value. Calibration is not independent and
train-only residuals transfer to fresh-refit forecasts from a different source.
Original official input hashes/software differ; raw official forecasts unavailable.

## Conclusions
This is a research/report handoff, not operational deployment. Keep source QA and
uncertainty flags, conservative labels, and M4 self-approval versus team sign-off
distinct. Historical bundles remain unchanged. Team must review G4 acceptance.
'''
    manuscript.write_text(text,encoding='utf-8')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['p4-bundle','p1-bundle','direct-bundle','quantile-bundle','official-predictions','annotations']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--historical-archive',type=Path);p.add_argument('--bootstrap-bundle',type=Path)
    p.add_argument('--prepare-bootstrap-request',type=Path)
    p.add_argument('--bootstrap-approval',type=Path);p.add_argument('--bootstrap-output',type=Path)
    p.add_argument('--output-dir',type=Path)
    return p


def main():
    args=parser().parse_args()
    try:
        inputs=load_inputs(args.p4_bundle,args.p1_bundle,args.direct_bundle,args.quantile_bundle,args.official_predictions,args.annotations,args.historical_archive)
        if args.prepare_bootstrap_request:
            request=bootstrap_request(inputs);fixed.write_json(args.prepare_bootstrap_request,request)
            print('PROPOSED_NOT_APPROVED',contracts.semantic_hash(request));return
        if args.bootstrap_approval:
            if args.bootstrap_output is None:raise ValueError('--bootstrap-output required')
            print(run_bootstrap(inputs,args.bootstrap_approval,args.bootstrap_output));return
        if args.output_dir is None:raise ValueError('--output-dir required; never infer a live canonical overwrite')
        print(export(inputs,args.output_dir,args.bootstrap_bundle))
    except (OSError,ValueError,KeyError) as error:
        raise SystemExit('M4 handoff error: '+str(error)) from error


if __name__=='__main__':main()
