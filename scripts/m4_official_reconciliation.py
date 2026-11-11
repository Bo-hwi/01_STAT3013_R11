"""Non-destructive official-M3 versus M4 point reconciliation; no fitting/costs."""
import argparse
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from src.decision import p4_reproduction as p

ROOT=p.ROOT
SOURCE='results/experiments/m4_reproduced_m3_20261108T200231Z_d1e59483f28f4c01a76f02f972367216'
FEATURE='data/processed/m4_feature_candidates/candidate-t77rmmvr/model_features.parquet'
OFFICIAL_COLUMNS=['date','store_nbr','family','city','fold','model','config','y_true','y_pred']
KEY=p.KEY+['model','config']


def difference_stats(frame):
    a=frame.official_prediction.to_numpy(float);b=frame.clipped_prediction.to_numpy(float)
    delta=np.abs(a-b)
    return dict(n=len(frame),exact_equal=int(np.sum(a==b)),max_abs_difference=float(delta.max()),
        mean_abs_difference=float(delta.mean()),equal_after_M4_round4=int(np.sum(a==np.round(b,4))),
        abs_le_1e_8=int(np.sum(delta<=1e-8)),abs_le_1e_6=int(np.sum(delta<=1e-6)),
        abs_le_5e_5=int(np.sum(delta<=5e-5)),abs_le_1e_4=int(np.sum(delta<=1e-4)),
        isclose_rtol_1e_10_atol_1e_8=int(np.isclose(a,b,rtol=1e-10,atol=1e-8).sum()),
        M4_raw_negative=int(frame.raw_prediction.lt(0).sum()))


def run(official,root=ROOT):
    root=Path(root).resolve();official=Path(official).resolve()
    if not official.is_file():raise FileNotFoundError('Official point_models.csv.gz missing; STOP')
    if p.subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()!='feature/m4-quantile-decision':
        raise ValueError('Wrong branch')
    source=root/SOURCE
    if p.inspect_bundle(source)!='complete':raise ValueError('M4 reproduction incomplete/corrupt')
    manifest=json.loads((source/'manifest.json').read_text());request=manifest['approval']['request']
    if manifest['approval']['status']!='APPROVED' or manifest['approval']['request_sha256']!=p.semantic_hash(request):
        raise ValueError('Invalid reproduction approval')
    for record in [request,manifest]:p.verify_frozen_files({'file_sha256':record['file_sha256'],'input':{}})
    snapshots={str(official):p.file_hash(official),str(Path(__file__).resolve()):p.file_hash(__file__)}
    snapshots.update({str(f):p.file_hash(f) for f in source.iterdir() if f.is_file()})
    a=pd.read_csv(official,parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    if list(a)!=OFFICIAL_COLUMNS or a.empty or a.isna().any().any():raise ValueError('Official schema/null contract failure')
    if not np.isfinite(a[['y_true','y_pred']]).all().all() or a[['y_true','y_pred']].lt(0).any().any():
        raise ValueError('Nonfinite/negative official target/prediction')
    if a.date.dt.tz is not None or not a.date.eq(a.date.dt.normalize()).all():raise ValueError('Invalid official dates')
    a['split']='test'
    if a.duplicated(KEY).any():raise ValueError('Duplicate official keys')
    if set(a.model)!=set(p.MODELS) or set(a.config)!={'M0','M1','M2','M3','M4'} or set(a.fold)!=set(p.SPEC['folds']):
        raise ValueError('Official model/config/fold scope differs from PR11')
    expected=pd.read_csv(source/'expected_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    forecasts=pd.read_csv(source/'predictions.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    tuning=pd.read_csv(source/'tuning.csv',float_precision='round_trip')
    protocol=p.read_yaml(root/'configs/research_protocol.yaml');ablation=p.read_yaml(root/'configs/ablation_M0_M5.yaml')
    p.validate_forecasts(forecasts,expected,protocol['rq2']['validation']['test_folds'])
    test=expected.loc[expected.split.eq('test')].copy()
    # Normalize identifier copies only; file and model inputs remain unchanged.
    normalized=[];coverage=[]
    for (model,config),g in a.groupby(['model','config'],sort=True):
        g=p.comparison_copy(g);p.assert_matched_rows(test,g);normalized.append(g)
        for fold,part in g.groupby('fold'):
            coverage.append(dict(model=model,config=config,fold=fold,n=len(part),n_dates=part.date.nunique(),n_cities=part.city.nunique(),
                n_families=part.family.nunique(),start=str(part.date.min().date()),end=str(part.date.max().date())))
    a=pd.concat(normalized,ignore_index=True)
    # Independently rebuild score eligibility through existing shared iter_folds.
    cols=['date','store_nbr','family','city','sales','store_active','store_zero_day','score_mask']
    mask_rebuild=dict(status='VERIFIED',method='fresh shared iter_folds over immutable candidate projection')
    try:
        data=pd.read_parquet(root/FEATURE,columns=cols)
    except ImportError as error:
        # Do not bypass Windows Application Control or install another engine.
        # Frozen candidate/code hashes and TWO historical scored-key contracts
        # still permit exact CSV reconciliation, not a claim of a fresh rebuild.
        mask_rebuild=dict(status='UNAVAILABLE',reason=str(error),
            method='Exact comparison against immutable frozen shared-fold key/target contracts; no Parquet recomputation')
        qdir=root/'results/experiments'/('m4_quantiles_'+p.FINAL)
        if p.inspect_run(root,p.FINAL)['status']!='complete':raise ValueError('Frozen quantile contract incomplete')
        q=pd.read_csv(qdir/('m4_quantiles_'+p.FINAL+'.csv'),usecols=p.KEY+['y_true'],parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
        p.assert_matched_rows(test,q.loc[q.split.eq('test')].drop_duplicates())
    else:
        shared=[]
        for fold in p.iter_folds(data,protocol,ablation,protocol['scope']['deep_study_families']):
            part=fold.test.loc[fold.test.score_mask,['date','store_nbr','family','city','sales']].rename(columns={'sales':'y_true'})
            shared.append(part.assign(fold=fold.fold_id,split='test'))
        del data
        p.assert_matched_rows(test,pd.concat(shared,ignore_index=True))
    print('Verified all 15 official panels against frozen scored keys; fresh mask rebuild '+mask_rebuild['status'],flush=True)
    selected=tuning.loc[tuning.selected_candidate&tuning.selected_algorithm].copy()
    if len(selected)!=8 or selected.duplicated(['fold','config']).any():raise ValueError('Invalid frozen selected mapping')
    published=pd.read_csv(StringIO(p.reference_bytes('results/tables/rq2_ml_experiments.csv',root).decode()))
    published_provenance=json.loads(p.reference_bytes('results/tables/rq2_ml_provenance.json',root))
    # Selection comes ONLY from historical train-only validation tuning, never test scores.
    parts=[];selection=[];reference_mae=[]
    for row in selected.itertuples():
        official_rows=a.loc[a.fold.eq(row.fold)&a.config.eq(row.config)&a.model.eq(row.model)].copy()
        m4=forecasts.loc[forecasts.split.eq('test')&forecasts.fold.eq(row.fold)&forecasts.config.eq(row.config)]
        m4=p.comparison_copy(m4)
        pair=official_rows.rename(columns={'y_pred':'official_prediction'}).merge(
            m4[KEY+['y_true','raw_prediction','clipped_prediction']],on=KEY,how='outer',validate='1:1',indicator=True,suffixes=('_official','_M4'))
        if not pair['_merge'].eq('both').all():raise ValueError('Selected official/M4 unmatched keys')
        if not np.array_equal(pair.y_true_official,pair.y_true_M4):raise ValueError('Selected official/M4 demand mismatch')
        pair['abs_prediction_difference']=np.abs(pair.official_prediction-pair.clipped_prediction)
        pair['exact_equal']=pair.official_prediction.eq(pair.clipped_prediction)
        pair['equal_after_M4_round4']=pair.official_prediction.eq(pair.clipped_prediction.round(4))
        parts.append(pair.drop(columns='_merge'))
        reference=published.loc[published.fold.eq(row.fold)&published.ablation.eq(row.config)&published.model.eq(row.model)]
        if len(reference)!=1:raise ValueError('Missing pinned selection reference')
        ref=reference.iloc[0]
        selection.append(dict(fold=row.fold,config=row.config,model=row.model,M4_parameters=row.parameters,
            published_parameters=ref.best_parameters,parameters_equal=json.loads(row.parameters)==json.loads(ref.best_parameters),
            M4_validation_MAE=row.validation_MAE,published_validation_MAE=ref.validation_MAE,
            official_test_MAE=float(np.mean(np.abs(pair.y_true_official-pair.official_prediction))),published_test_MAE=ref.test_MAE))
    for (fold,model,config),g in a.groupby(['fold','model','config']):
        ref=published.loc[published.fold.eq(fold)&published.model.eq(model)&published.ablation.eq(config)].iloc[0]
        mae=float(np.mean(np.abs(g.y_true-g.y_pred)))
        reference_mae.append(dict(fold=fold,model=model,config=config,official_file_MAE=mae,published_unrounded_MAE=ref.test_MAE,
            absolute_summary_difference=abs(mae-ref.test_MAE),within_round4_error_bound=abs(mae-ref.test_MAE)<=5e-5+1e-10))
    pairs=pd.concat(parts,ignore_index=True);statistics=[]
    for level,names in [('overall',[]),('model',['model']),('fold_config',['fold','config','model'])]:
        groups=[((),pairs)] if not names else pairs.groupby(names,sort=True)
        for key,g in groups:
            key=key if isinstance(key,tuple) else (key,)
            statistics.append(dict(zip(names,key),level=level,**difference_stats(g)))
    summary=difference_stats(pairs)
    original_hash=published_provenance['source_sha256']['data/processed/model_features.parquet']
    input_hashes={x:p.file_hash(root/x) for x in [FEATURE,'data/processed/model_features.parquet']}
    notes=dict(official_source='User-provided Drive copy at local path; PR11 export/schema/summary consistency checked',
        independently_published_output_checksum='UNAVAILABLE: PR11 provenance does not hash prediction output',
        raw_official_predictions='UNAVAILABLE: pinned export clips and rounds y_pred to four decimals; raw-to-raw comparison not possible',
        notebook_reference=dict(commit=p.COMMIT,sha256=p.PINS[p.NOTEBOOK],preprocessing_cell=7,export_cell=23),
        feature_input_file_hashes=input_hashes,published_M3_feature_sha256=original_hash,
        authenticated_original_feature_input_available=original_hash in input_hashes.values(),
        feature_value_identity_with_original_M3='NOT_VERIFIABLE unless original input with the published SHA-256 is supplied',
        preprocessing_refit='M4 reuses pinned M3 preprocessing/float32, candidate grids, clipping and fresh eligible train+validation refit; original artifact execution is inferred from provenance, not a model checkpoint',
        official_environment=published_provenance,M4_environment=request['environment'],
        selection='Frozen M4 clipped validation-MAE algorithm mapping; no test-target algorithm/parameter selection',
        independent_current_mask_rebuild=mask_rebuild)
    classification='EXACT_EQUAL' if summary['exact_equal']==summary['n'] else 'ROUND4_EXPORT_EQUIVALENT_NOT_EXACT' if summary['equal_after_M4_round4']==summary['n'] else 'MATERIAL_OR_NON_ROUNDING_DIFFERENCES'
    report=dict(status='RECONCILIATION_COMPLETE',integration='NO_GO_PENDING_REVISED_P4_PLAN_AND_APPROVAL' if classification!='EXACT_EQUAL' else 'GO_FOR_PROVENANCE_ACCEPTANCE_REVIEW',
        classification=classification,official_file=str(official),official_sha256=snapshots[str(official)],official_bytes=official.stat().st_size,
        official_rows=len(a),official_model_config_panels=15,matched_selected_keys=len(pairs),unmatched_official_keys=0,unmatched_M4_keys=0,
        y_true_exact_match=True,frozen_shared_scored_keys_verified=True,fresh_shared_mask_rebuild=mask_rebuild['status'],statistics=summary,provenance=notes,
        historical_P4_reusable_without_recalculation=classification=='EXACT_EQUAL',new_economic_evaluation_performed=False)
    parent=root/'results/experiments';dest=parent/('m4_official_m3_reconciliation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+dest.name+'.staging-',dir=parent))
    try:
        for name,frame in [('prediction_differences',pairs),('difference_statistics',pd.DataFrame(statistics)),('selected_mapping',pd.DataFrame(selection)),
                           ('official_coverage',pd.DataFrame(coverage)),('published_MAE_checks',pd.DataFrame(reference_mae))]:
            with (stage/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
        p._json(stage/'reconciliation.json',report)
        impact='''# Official M3 reconciliation and dependency impact

This is prediction reconciliation only. No model fitting, residual calibration,
order-quantity or simulated cost evaluation has been executed.

Official y_pred is clipped and rounded to four decimals. Raw official forecasts
are unavailable; do not compare clipped official values to raw M4 as raw equality.
Read difference_statistics.csv for exact, rounding and explicit tolerance counts.
Row-level equality is established only where exact_equal=True, never from MAE.

If ANY selected prediction differs, historical P4 test inputs fail exact-source
substitution. A new frozen hybrid P4 plan must retain M4 train-only clipped
validation residuals, fixed selected algorithms/calibration settings, but replace
test base forecasts with official y_pred. Recalculation requires separate approval.
Clipping can make some Q unchanged, but no such outcome is assumed or computed here.

Affected: P4-M0/M3 Q, costs, units, P1-normalized indices, P4 weather-value, fixed-week
contrasts involving P4, Decision Matrix P4 evidence/fold consistency, cost tables,
P4-related figures and manuscript. P1/P2/P3 point costs, direct quantile forecasts,
and P3-P2 bootstrap contrasts can remain authoritative if keys/inputs stay fixed;
carry them into a NEW versioned bundle rather than overwrite historical outputs.
Reevaluate descriptive directions under the same old rule after approved costs;
never retune classification/recommendation rules or claim preregistration.

Even rounding-only differences are not exact equality. Do not claim official-source
economic costs/CI by reusing M4 full-precision costs without an explicit approved
equivalence policy; no equivalence tolerance for economic reuse was preapproved.
Original M3 feature-file SHA differs from locally available candidate/canonical
files. Same named source functions are verified; original feature-value identity
and original raw predictions are unavailable. A Drive file's self-computed hash
identifies the supplied copy, not an independently signed upstream checksum.

Next: team review reconciliation; freeze new hybrid official-test/M4-validation P4
execution request, input hashes, algorithm mapping, rounding and calibration
definitions, matched rows, downstream bootstrap scope and versioned export plan;
obtain explicit execution approval before revised economics. Preserve all old
bundles and canonical exports. No final integration until this condition is met.
'''
        with (stage/'REPORT.md').open('x',encoding='utf-8') as f:
            f.write(impact+'\n## Exact outcome\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
        for name,digest in snapshots.items():
            if p.file_hash(name)!=digest:raise ValueError('Source changed during reconciliation: '+name)
        for name in ['configs/research_protocol.yaml','configs/ablation_M0_M5.yaml',FEATURE,'data/processed/model_features.parquet']:
            snapshots[str(root/name)]=p.file_hash(root/name)
        outputs={f.name:p.file_hash(f) for f in stage.iterdir() if f.is_file()}
        p._json(stage/'manifest.json',dict(file_sha256=snapshots,source_bundle=SOURCE,reference_commit=p.COMMIT,
            reference_sha256=p.PINS,output_sha256=outputs,training_fits=0,economic_evaluations=0,classification=classification))
        p._json(stage/'COMPLETE.json',dict(files={**outputs,'manifest.json':p.file_hash(stage/'manifest.json')}))
        if dest.exists():raise FileExistsError('Never overwrite reconciliation')
        stage.rename(dest)
    except Exception as e:
        p._json(stage/'FAILED.json',dict(type=type(e).__name__,message=str(e)));raise
    print('BUNDLE',dest);print(json.dumps(report,indent=2));print(pd.DataFrame(statistics).to_string(index=False))
    return dest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--official',type=Path,default=ROOT/'data/processed/point_models.csv.gz')
    run(parser.parse_args().official)


if __name__=='__main__':main()
