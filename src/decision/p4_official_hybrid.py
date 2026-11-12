"""Separate approval gate: frozen M4 validation offsets + official M3 test base."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from . import p4_policy_evaluation as old
from . import fixed_week_bootstrap as fixed

ROOT=old.ROOT
RECON='results/experiments/m4_official_m3_reconciliation_20261111T134400Z_0f08ccb997e341a49d4c749bd9a720c2'
OLD_COST='results/experiments/m4_p4_costs_20261109T060634Z_b0b91a16722f4130bfd89a887a8cc19f'
OFFICIAL='data/processed/point_models.csv.gz'
OFFICIAL_SHA='71e790d360f80a6c3c6d86142391e2ff2947d4d79ec808c2ba3e8095d1623927'
STATUS='PROPOSED_NOT_APPROVED'
SPEC=dict(label='P4_HYBRID_M4_VALIDATION_OFFICIAL_M3_TEST',
    folds=['F1','F2','F3','F4'],configs=['M0','M3'],test_observations_per_config=41076,
    selection='unchanged frozen M4 train-only validation MAE model/parameter mapping; no test selection',
    validation_source=old.SOURCE,test_source=OFFICIAL,official_sha256=OFFICIAL_SHA,
    residual='y_true - M4 clipped train-only validation prediction',pooling=['fold','config','family'],
    quantile_method='linear',minimum_n=30,fallback=['fold','config'],
    offsets='reuse unchanged 192 approved historical validation quantile records; validate against immutable source and group counts',
    base='official y_pred as delivered, clipped and rounded to four decimals; do NOT unround or invent raw predictions',
    Q='max(0, official_y_pred + frozen_validation_residual_quantile)',fractional=True,
    scenarios='exact existing shared YAML S1/S3/S5; ratios of summed P1 costs on matched rows',
    sensitivity='P4 is fixed; corrected/raw-clipped variants change P2/P3 only',
    scope='P4 quantities/costs/indices only; no bootstrap/matrix/canonical migration in this approval',
    limitations=['hybrid validation/test sources and different environments; not independent calibration',
        'M4 train-only residuals applied to official fresh-refit forecast; transfer mismatch possible',
        'original official feature values and raw forecasts unavailable; Drive copy has no independent published output checksum',
        'F1 exposed; F2-F4 exposure unverified; no causal/conformal/operational/money/profit/ROI claim',
        'fresh Parquet mask rebuild unavailable under Windows Application Control; exact frozen scored-key contracts verified',
        'downstream P4 bootstrap/evidence update needs separate versioned execution scope; no silent CI reuse'])


def select_official(official,expected,audit):
    required=old.reproduction.KEY+['config','model','y_true','y_pred']
    x=official.copy()
    if 'split' not in x:x['split']='test'
    if x.empty or set(required)-set(x) or x[required].isna().any().any() or not x.split.eq('test').all():
        raise ValueError('Invalid official test fields/stage')
    if x.duplicated(old.reproduction.KEY+['model','config']).any():raise ValueError('Duplicate official keys')
    if not np.isfinite(x[['y_true','y_pred']]).all().all() or x[['y_true','y_pred']].lt(0).any().any():raise ValueError('Invalid official values')
    if audit.duplicated(['fold','config']).any():raise ValueError('Duplicate selected mapping')
    test=expected.loc[expected.split.eq('test')];parts=[]
    for a in audit.itertuples():
        g=x.loc[x.fold.eq(a.fold)&x.config.eq(a.config)&x.model.eq(a.model)].copy()
        wanted=test.loc[test.fold.eq(a.fold)]
        old.reproduction.assert_matched_rows(wanted,g)
        g=old.reproduction.comparison_copy(g)
        g['selected_parameters']=a.selected_parameters
        parts.append(g)
    result=pd.concat(parts,ignore_index=True)
    for _,g in result.groupby('config'):old.reproduction.assert_matched_rows(test,g)
    return result


def frozen_adjuster(calibration,validation,folds):
    """Restore existing approved residual offsets, never learn from test forecasts."""
    spec=old.residual.ResidualSpec(('family',),'linear',30,((),))
    if not validation.split.eq('validation').all() or not validation.fit_stage.eq('train_only').all():
        raise ValueError('Require train-only validation residual source')
    cal=calibration.copy();keys=['fold','model','config','fallback_level','tau_key']
    if cal[keys+['residual_quantile','n','n_dates','date_min','date_max','quantile_method']].isna().any().any():
        raise ValueError('Null frozen calibration metadata')
    if cal.duplicated(keys+['family']).any() or not cal.quantile_method.eq('linear').all():raise ValueError('Duplicate/wrong quantile calibration')
    if not np.isfinite(cal.residual_quantile).all():raise ValueError('Nonfinite residual quantile')
    levels=set(cal.fallback_level)
    if not levels.issubset({0,1}):raise ValueError('Unapproved fallback level')
    starts={f['fold_id']:pd.Timestamp(f['validation'][0]) for f in folds}
    ends={f['fold_id']:pd.Timestamp(f['validation'][1]) for f in folds}
    offsets={};taus=old.QUANTILE_LEVELS
    tau_keys=[old.residual.canonical_tau_key(t) for t in taus]
    for (fold,model,config,level,family),g in cal.groupby(['fold','model','config','fallback_level','family'],dropna=False,sort=True):
        if set(g.tau_key)!=set(tau_keys) or len(g)!=3:raise ValueError('Incomplete frozen tau triplet')
        v=validation.loc[validation.fold.eq(fold)&validation.model.eq(model)&validation.config.eq(config)]
        if level==0:
            if pd.isna(family):raise ValueError('Missing family for primary pooling')
            v=v.loc[v.family.eq(family)]
        elif pd.notna(family) and family!='':raise ValueError('Fallback must be fold/config global')
        if len(v)<30 or not v.date.between(starts[fold],ends[fold]).all():raise ValueError('Invalid validation size/window')
        for r in g.itertuples():
            if r.n!=len(v) or r.n_dates!=v.date.nunique() or pd.Timestamp(r.date_min)!=v.date.min() or pd.Timestamp(r.date_max)!=v.date.max():
                raise ValueError('Frozen calibration group differs from validation evidence')
        q=g.set_index('tau_key').loc[tau_keys,'residual_quantile'].to_numpy(float)
        if np.any(q[1:]<q[:-1]):raise ValueError('Unordered residual quantiles')
        key=(fold,model,config,int(level))+((family,) if level==0 else ())
        offsets[key]=dict(quantiles=tuple(q),n=len(v),n_dates=v.date.nunique(),date_max=v.date.max())
    return old.residual.ResidualAdjuster(spec,old.json_snapshot(folds),tuple(taus),offsets,cal)


def quantities(selected,validation,calibration,folds,audit):
    adjuster=frozen_adjuster(calibration,validation,folds)
    # Only forecast-origin fields and official base enter predict; never test labels.
    test=selected.drop(columns='y_true').copy()
    test['score_mask']=True;test['forecast_origin']=test.date-pd.Timedelta(days=1);test['horizon_days']=1
    q=adjuster.predict(test)
    q=q.merge(selected[old.residual.POINT_KEY+['y_true']],on=old.residual.POINT_KEY,validate='m:1')
    q=q.merge(audit.drop(columns='model'),on=['fold','config'],validate='m:1')
    q['policy']='P4';q['test_source']='OFFICIAL_M3_TEST';q['validation_source']='M4_REPRODUCED_M3_TRAIN_ONLY'
    q['official_raw_available']=False;q['base_prediction_column']='official_y_pred_rounded4';q['methodology_status']=STATUS
    q['cutoff_evidence']='M4 validation stage verified; official test refit declared by pinned PR11 notebook, checkpoint unavailable'
    return q


def evaluate(sources):
    """Synthetic-testable arithmetic. Real execution must pass run_approved()."""
    q=quantities(sources['selected'],sources['validation'],sources['calibration'],sources['folds'],sources['audit'])
    test=sources['expected'].loc[sources['expected'].split.eq('test')]
    old.validate_fixed_costs(test,sources['baseline'],sources['direct'],sources['config'])
    rows=[];pairs=[];summaries=[]
    for scenario in old.configured_scenarios(sources['config']):
        baseline=sources['baseline'].loc[sources['baseline'].scenario.eq(scenario['scenario']),old.MATCH+['y_true','simulated_cost']].rename(columns={'y_true':'y_true_P1','simulated_cost':'cost_P1'})
        for cfg,g in q.loc[q.tau_key.eq(scenario['tau_key'])].groupby('config'):
            g=g.copy();old._matched_test(test,g)
            g['scenario']=scenario['scenario'];g['c_over']=scenario['c_over'];g['c_under']=scenario['c_under']
            g['simulated_cost']=old.newsvendor_cost(g.y_true,g.quantity,scenario['c_over'],scenario['c_under'])
            if not np.allclose(g.simulated_cost,(scenario['c_over']+scenario['c_under'])*old.pinball_loss(g.y_true,g.quantity,scenario['tau']),rtol=1e-12,atol=0):raise ValueError('Cost-pinball identity')
            g['overstock_units']=np.maximum(g.quantity-g.y_true,0);g['understock_units']=np.maximum(g.y_true-g.quantity,0);g['result_label']=old.LABEL;rows.append(g)
            for variant in fixed.VARIANTS:
                pair=g[old.MATCH+['y_true','simulated_cost']].rename(columns={'simulated_cost':'cost_P4'}).merge(baseline,on=old.MATCH,how='outer',validate='1:1')
                for policy in ['P2','P3']:
                    frozen=sources['direct'].loc[sources['direct'].scenario.eq(scenario['scenario'])&sources['direct'].variant.eq(variant)&sources['direct'].policy.eq(policy),old.MATCH+['y_true','simulated_cost']]
                    pair=pair.merge(frozen.rename(columns={'y_true':'y_true_'+policy,'simulated_cost':'cost_'+policy}),on=old.MATCH,how='outer',validate='1:1')
                if pair.isna().any().any() or any(not np.array_equal(pair.y_true,pair['y_true_'+p]) for p in ['P1','P2','P3']):raise ValueError('Fixed P1/P2/P3 key/target mismatch')
                pair['config']=cfg;pair['variant']=variant;pair['scenario']=scenario['scenario'];pairs.append(pair)
                for level,grouping in old.LEVELS.items():
                    groups=[((),pair)] if not grouping else pair.groupby(grouping,observed=True,sort=True)
                    for key,block in groups:
                        key=key if isinstance(key,tuple) else (key,)
                        record=dict(zip(grouping,key),scenario=scenario['scenario'],config=cfg,variant=variant,report_level=level,sample_size=len(block))
                        for p in ['P1','P2','P3','P4']:
                            record['mean_cost_'+p]=old.stable_mean(block['cost_'+p].to_numpy())
                            record['cost_index_'+p]=old.cost_index(block['cost_'+p],block.cost_P1)
                        record['p1_denominator_zero']=not block.cost_P1.gt(0).any();summaries.append(record)
    return dict(p4_rows=pd.concat(rows,ignore_index=True),matched_costs=pd.concat(pairs,ignore_index=True),comparisons=pd.DataFrame(summaries),calibration=sources['calibration'].copy())


def load_sources(root=ROOT):
    root=Path(root).resolve()
    if fixed.current_branch(root)!='feature/m4-quantile-decision':raise ValueError('Wrong branch')
    official=root/OFFICIAL
    if old.file_hash(official)!=OFFICIAL_SHA:raise ValueError('Official file drift')
    recon=root/RECON;marker=json.loads((recon/'COMPLETE.json').read_text())
    for name,d in marker['files'].items():
        path=(recon/name).resolve()
        if not path.is_relative_to(recon.resolve()) or old.file_hash(path)!=d:raise ValueError('Reconciliation corrupt')
    report=json.loads((recon/'reconciliation.json').read_text())
    if report['official_sha256']!=OFFICIAL_SHA or report['matched_selected_keys']!=82152 or not report['y_true_exact_match']:
        raise ValueError('Reconciliation scope mismatch')
    sources=old.load_sources(root)
    historical=root/OLD_COST
    if old.inspect_bundle(historical)!='complete':raise ValueError('Historical P4 bundle incomplete')
    cal=pd.read_csv(historical/'calibration.csv',parse_dates=['date_min','date_max'],float_precision='round_trip')
    official_frame=pd.read_csv(official,parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    selected=select_official(official_frame,sources['expected'],sources['audit'])
    validation=sources['panel'].loc[sources['panel'].split.eq('validation')].copy()
    # Preparation restores offsets/metadata only; no Q/cost calculation.
    frozen_adjuster(cal,validation,sources['folds'])
    files=dict(sources['files'])
    for folder in [recon,historical]:files.update({str(f):old.file_hash(f) for f in folder.iterdir() if f.is_file()})
    files[str(official)]=OFFICIAL_SHA
    return dict(selected=selected,validation=validation,calibration=cal,**{k:sources[k] for k in ['expected','folds','audit','baseline','direct','config']}),files


def environment():return dict(python=platform.python_version(),packages={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']})


def make_request(root=ROOT):
    root=Path(root).resolve();sources,files=load_sources(root)
    for relative in ['src/decision/p4_official_hybrid.py','tests/test_m4_official_hybrid.py','docs/m4_phase5b_hybrid_plan.md']:
        f=root/relative;files[str(f)]=old.file_hash(f)
    selected=sources['selected'].sort_values(old.reproduction.KEY+['config','model'])
    keys=selected[old.reproduction.KEY+['config','model','y_true','y_pred']].assign(date=lambda x:x.date.dt.strftime('%Y-%m-%d')).to_dict(orient='records')
    return old.json_snapshot(dict(status=STATUS,specification=SPEC,environment=environment(),file_sha256=files,
        official_selected_keys_sha256=old.semantic_hash(keys),selected_rows=len(selected),validation_rows=len(sources['validation']),
        frozen_calibration_rows=len(sources['calibration']),selected_mapping=sources['audit'].to_dict(orient='records'),
        scenarios=old.configured_scenarios(sources['config']),reconciliation=RECON,execution_authorized=False))


def validate_approval(receipt,root=ROOT):
    request=receipt.get('request',{})
    if receipt.get('status')!='APPROVED' or request.get('status')!=STATUS:raise ValueError('Separate hybrid P4 approval PENDING_APPROVAL')
    if receipt.get('request_sha256')!=old.semantic_hash(request) or request.get('specification')!=SPEC:raise ValueError('Approval digest/specification mismatch')
    for field in ['approved_by','approval_evidence','approval_communicated_date']:
        if not isinstance(receipt.get(field),str) or not receipt[field].strip():raise ValueError('Missing explicit approval evidence')
    datetime.strptime(receipt['approval_communicated_date'],'%Y-%m-%d')
    old.verify_frozen_files({'file_sha256':request['file_sha256'],'input':{}})
    if request!=make_request(root):raise ValueError('Frozen request/environment drift')
    return request


def run_approved(approval,root=ROOT):
    receipt=json.loads(Path(approval).read_text(encoding='utf-8'));request=validate_approval(receipt,root)
    sources,_=load_sources(root);parent=Path(root)/'results/experiments'
    dest=parent/('m4_p4_official_hybrid_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+dest.name+'.staging-',dir=parent));start=perf_counter()
    old.reproduction._json(stage/'STARTED.json',dict(approval=receipt))
    try:
        result=evaluate(sources)
        for name,frame in result.items():
            frame['methodology_status']='APPROVED_HYBRID';frame['official_sha256']=OFFICIAL_SHA
            with (stage/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
            dates=[c for c in frame if pd.api.types.is_datetime64_any_dtype(frame[c])]
            loaded=pd.read_csv(stage/(name+'.csv'),parse_dates=dates or None,dtype={'store_nbr':str} if 'store_nbr' in frame else None,float_precision='round_trip')
            pd.testing.assert_frame_equal(loaded,frame.reset_index(drop=True),check_dtype=False)
        old.verify_frozen_files({'file_sha256':request['file_sha256'],'input':{}})
        outputs={f.name:old.file_hash(f) for f in stage.iterdir() if f.is_file()}
        old.reproduction._json(stage/'manifest.json',dict(label=SPEC['label'],approval=receipt,output_sha256=outputs,
            elapsed_seconds=perf_counter()-start,model_fits=0,bootstrap_runs=0,canonical_outputs_changed=False))
        old.reproduction._json(stage/'COMPLETE.json',dict(files={**outputs,'manifest.json':old.file_hash(stage/'manifest.json')}))
        if inspect_bundle(stage)!='complete':raise ValueError('Hybrid output integrity failed')
        if dest.exists():raise FileExistsError('Never overwrite hybrid bundle')
        stage.rename(dest);return dest
    except Exception as e:
        old.reproduction._json(stage/'FAILED.json',dict(type=type(e).__name__,message=str(e)));raise


def inspect_bundle(directory):
    directory=Path(directory).resolve()
    if (directory/'FAILED.json').exists():return 'failed'
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text())
        if set(marker['files'])!={'STARTED.json','manifest.json','p4_rows.csv','matched_costs.csv','comparisons.csv','calibration.csv'}:return 'corrupt'
        if any(not (directory/n).resolve().is_relative_to(directory) or old.file_hash(directory/n)!=d for n,d in marker['files'].items()):return 'corrupt'
        manifest=json.loads((directory/'manifest.json').read_text())
        return 'complete' if manifest['output_sha256']=={n:d for n,d in marker['files'].items() if n!='manifest.json'} else 'corrupt'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete'


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-request',type=Path);group.add_argument('--approval',type=Path);args=parser.parse_args()
    if args.prepare_request:
        if args.prepare_request.exists():raise FileExistsError('Never overwrite request')
        request=make_request();old.reproduction._json(args.prepare_request,request);print('SEMANTIC_SHA256',old.semantic_hash(request))
    else:print(run_approved(args.approval))


if __name__=='__main__':main()
