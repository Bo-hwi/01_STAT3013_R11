"""Approval-gated M4 reproduction of pinned M3 functions, never official M3."""
import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.forecasting.feature_spec import FEATURES, m0_numeric
from src.forecasting.point_models import prepare_features
from src.forecasting.rolling_validation import iter_folds, read_yaml
from .quantile_artifacts import inspect_run
from .quantile_contracts import (file_hash, json_snapshot, semantic_hash, scored_key_digest,
                                 verify_frozen_files, stable_mean)
from .quantile_preflight import production_preflight

ROOT=Path(__file__).resolve().parents[2]
COMMIT='b35f5cc8f5dd2d07b0a545b6e5a60963afdff6bc'
NOTEBOOK='notebooks/03_forecasting_ml/train_ml.ipynb'
PINS={NOTEBOOK:'40564f6805d90470269490574bd2b0262896063887defbe02e9a9d3328aca422',
      'results/tables/rq2_ml_tuning_log.csv':'9fa281c1cc52c964176ab02c310b0453c2c3ba51f80f09a10e9da27fc0a9d40d',
      'results/tables/rq2_ml_experiments.csv':'2ad15fb4f2c97d4ce4924ca2dc518ce4c4881fd7100a22387af0f3425c480199',
      'results/tables/rq2_ml_provenance.json':'f49e06eb40dbe9b9713050c2288bde77dda1f7f6026c066e875f87995f899667'}
MODELS=('Ridge','ExtraTrees','HistGradientBoosting')
CONFIGS=('M0','M3')
KEY=['fold','split','date','store_nbr','family','city']
LABEL='M4_REPRODUCED_M3'
FINAL='final_f1_f4_m0_m3_20261108T123157Z_eb881cc4c56a'
PACKAGES=('numpy','pandas','scipy','scikit-learn','pyarrow','threadpoolctl')
SPEC=dict(label=LABEL,source_commit=COMMIT,folds=['F1','F2','F3','F4'],configs=list(CONFIGS),seed=42,
          comparison_identifier='positive int64 decimal store IDs; lossless comparison copies only; reject noncanonical text',
          selection='minimum clipped validation MAE; first candidate; algorithm order Ridge, ExtraTrees, HistGradientBoosting',
          expected_selection_fits=56,expected_refits=8,full_precision=True,
          differences=['verified M4 candidate, not authenticated original M3 input',
                       'local package versions recorded; no exact numerical reproduction claim',
                       'validation-only algorithm selection added; only winner refitted',
                       'full precision raw and clipped exports; no four-decimal rounding'])


def reference_bytes(path, root=ROOT):
    if path not in PINS:raise ValueError('Unpinned M3 reference path')
    data=subprocess.check_output(['git','show',COMMIT+':'+path],cwd=root)
    if sha256(data).hexdigest()!=PINS[path]:raise ValueError('Pinned M3 source mismatch')
    return data


def reference_functions(root=ROOT):
    """Compile four reviewed functions and the grid, not notebook top-level runs."""
    nb=json.loads(reference_bytes(NOTEBOOK,root))
    functions={'preprocess','estimator','audit_imputer','nonnegative_predict'}
    nodes=[]
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
            elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRIDS' for t in node.targets):nodes.append(node)
    if len(nodes)!=5:raise ValueError('Pinned function extraction differs from reviewed contract')
    namespace={k:globals()[k] for k in ['np','ColumnTransformer','ExtraTreesRegressor','HistGradientBoostingRegressor',
        'SimpleImputer','Ridge','Pipeline','OneHotEncoder','OrdinalEncoder','StandardScaler']}
    namespace.update(SEED=42,CATEGORICAL=list(FEATURES['categorical']))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),NOTEBOOK,'exec'),namespace)
    if tuple(namespace['GRIDS'])!=MODELS or [len(namespace['GRIDS'][m]) for m in MODELS]!=[3,2,2]:
        raise ValueError('Original candidate grid contract changed')
    return namespace


def feature_columns(config, ablation):
    if config not in CONFIGS:raise ValueError('Only M0/M3 are supported')
    weather=ablation['experiments'][config]['weather_features']
    if weather!=([] if config=='M0' else FEATURES['weather'][:2]):raise ValueError('Weather whitelist drift')
    return m0_numeric()+list(weather),list(FEATURES['categorical'])


def select_on_validation(train, validation, numeric, reference):
    """No test frame accepted. Cache candidate validation predictions; no extra fits."""
    features=numeric+FEATURES['categorical'];records=[];winners=[]
    for model in MODELS:
        prep=reference['preprocess'](model,numeric)
        x=prep.fit_transform(train[features]);reference['audit_imputer'](prep,train,numeric)
        xv=prep.transform(validation[features]);best=None
        for i,parameters in enumerate(reference['GRIDS'][model]):
            fitted=reference['estimator'](model,parameters,prep)
            fitted.fit(x,train.sales.to_numpy())
            raw=np.asarray(fitted.predict(xv),dtype=float)
            if raw.shape!=(len(validation),) or not np.isfinite(raw).all():raise ValueError('Invalid validation forecasts')
            clipped=np.maximum(raw,0);mae=stable_mean(np.abs(validation.sales.to_numpy()-clipped))
            record=dict(model=model,candidate=i,parameters=json.dumps(parameters,sort_keys=True),validation_MAE=mae)
            records.append(record)
            if best is None or mae<best['validation_MAE']:
                best=dict(record,raw=raw.copy(),clipped=clipped.copy())
            del fitted
        winners.append(best)
    winner=min(winners,key=lambda w:w['validation_MAE'])  # stable original algorithm order
    for r in records:
        own=next(w for w in winners if w['model']==r['model'])
        r['selected_candidate']=r['candidate']==own['candidate']
        r['selected_algorithm']=r['model']==winner['model']
    return winner,records


def comparison_copy(frame):
    """Only store_nbr changes, on a copy; never coerce through float or mutate inputs."""
    if frame.empty or set(KEY+['y_true'])-set(frame) or frame[KEY+['y_true']].isna().any().any():
        raise ValueError('Missing/null comparison keys or targets')
    def identifier(value):
        if isinstance(value,(bool,np.bool_)):
            raise ValueError('Invalid store_nbr: boolean identifier')
        if isinstance(value,str):
            if not re.fullmatch(r'[1-9][0-9]*',value) or len(value)>19:
                raise ValueError('Invalid store_nbr: require canonical positive decimal integer text')
            number=int(value)
        elif isinstance(value,(int,np.integer)):
            number=int(value)
        elif isinstance(value,(float,np.floating)):
            if not np.isfinite(value) or not value.is_integer() or abs(value)>2**53:
                raise ValueError('Invalid store_nbr: non-representable integer identifier')
            number=int(value)
        else:
            raise ValueError('Invalid store_nbr: unsupported identifier type')
        if not 1<=number<=np.iinfo(np.int64).max:
            raise ValueError('Invalid store_nbr: outside positive int64 range')
        return str(number)
    out=frame.copy(deep=True)
    out['store_nbr']=out.store_nbr.map(identifier).astype(object)
    if out.duplicated(KEY).any():raise ValueError('Duplicate normalized comparison keys')
    return out


def assert_matched_rows(expected, actual):
    expected_copy=comparison_copy(expected);actual_copy=comparison_copy(actual)
    merged=expected_copy.merge(actual_copy,on=KEY,how='outer',validate='1:1',
                               indicator=True,suffixes=('_expected','_actual'))
    if not merged['_merge'].eq('both').all() or not np.array_equal(merged.y_true_expected,merged.y_true_actual):
        raise ValueError('Frozen keys/targets mismatch')


def validate_forecasts(panel, expected, fold_specs):
    required=KEY+['config','model','y_true','raw_prediction','clipped_prediction','fit_stage','selected_parameters']
    if panel.empty or set(required)-set(panel) or panel[required].isna().any().any():raise ValueError('Missing/null prediction fields')
    if panel.duplicated(KEY+['config']).any():raise ValueError('Duplicate prediction keys')
    if set(panel.config)!=set(CONFIGS) or not set(panel.model).issubset(MODELS):raise ValueError('Unexpected model/config')
    if set(panel.split)!={'validation','test'}:raise ValueError('Require validation and test forecasts')
    if not np.isfinite(panel[['raw_prediction','clipped_prediction','y_true']].to_numpy(float)).all():raise ValueError('Nonfinite forecasts/targets')
    if not np.array_equal(panel.clipped_prediction,np.maximum(panel.raw_prediction,0)):raise ValueError('Clipping mismatch')
    if panel.groupby(['fold','config']).model.nunique().ne(1).any() or panel.groupby(['fold','config']).selected_parameters.nunique().ne(1).any():
        raise ValueError('Validation/refit selected model/parameters differ')
    if expected.duplicated(KEY).any() or expected[KEY+['y_true']].isna().any().any():raise ValueError('Invalid expected rows')
    for config in CONFIGS:
        actual=panel.loc[panel.config.eq(config)]
        assert_matched_rows(expected,actual)
    schedule={f['fold_id']:f for f in fold_specs}
    for (fold,split),g in panel.groupby(['fold','split']):
        if fold not in schedule:raise ValueError('Unknown fold')
        start,end=map(pd.Timestamp,schedule[fold][split])
        if not pd.to_datetime(g.date).between(start,end).all():raise ValueError('Prediction date outside shared fold')
        stage='train_only' if split=='validation' else 'fresh_train_validation_refit'
        if not g.fit_stage.eq(stage).all():raise ValueError('Prediction fit-stage mismatch')


def reproduce(data, protocol, ablation, reference=None):
    """Generic core for synthetic tests; real CLI must first pass approval/preflight."""
    reference=reference or reference_functions();protocol=json_snapshot(protocol);ablation=json_snapshot(ablation)
    if protocol['project']['seed']!=42:raise ValueError('Require seed 42')
    families=protocol['scope']['deep_study_families']
    if len(families)!=7 or len(set(families))!=7:raise ValueError('Require frozen seven families')
    data=prepare_features(data.loc[data.family.isin(families)].copy(),ready=True)
    numeric_cast=list(dict.fromkeys(m0_numeric()+FEATURES['weather']))
    numeric_cast=[c for c in numeric_cast if c in data]
    data[numeric_cast]=data[numeric_cast].astype('float32')
    if np.isinf(data[numeric_cast].to_numpy()).any():raise ValueError('Float32 numeric conversion overflow')
    parts=[];tuning=[];expected=[];counts=[];fits=0
    for fold in iter_folds(data,protocol,ablation,families):
        validation=fold.validation.loc[fold.validation.score_mask];test=fold.test.loc[fold.test.score_mask]
        for split,rows in [('validation',validation),('test',test)]:
            e=rows[['date','store_nbr','family','city','sales']].rename(columns={'sales':'y_true'}).copy()
            e['fold'],e['split']=fold.fold_id,split;expected.append(e)
        for config in CONFIGS:
            numeric,categorical=feature_columns(config,ablation)
            with threadpool_limits(limits=2):
                winner,records=select_on_validation(fold.train,validation,numeric,reference);fits+=7
                refit=fold.refit_rows()
                if refit.date.max()>=test.date.min() or refit.store_zero_day.ne(0).any():raise ValueError('Unsafe refit rows')
                prep=reference['preprocess'](winner['model'],numeric)
                xr=prep.fit_transform(refit[numeric+categorical]);reference['audit_imputer'](prep,refit,numeric)
                fitted=reference['estimator'](winner['model'],json.loads(winner['parameters']),prep)
                fitted.fit(xr,refit.sales.to_numpy());fits+=1
                raw=np.asarray(fitted.predict(prep.transform(test[numeric+categorical])),dtype=float)
                if raw.shape!=(len(test),) or not np.isfinite(raw).all():raise ValueError('Invalid test forecasts')
                for split,rows,raw_values in [('validation',validation,winner['raw']),('test',test,raw)]:
                    out=rows[['date','store_nbr','family','city','sales']].rename(columns={'sales':'y_true'}).copy()
                    out['fold'],out['split'],out['config'],out['model']=fold.fold_id,split,config,winner['model']
                    out['raw_prediction'],out['clipped_prediction']=raw_values,np.maximum(raw_values,0)
                    out['selected_parameters']=winner['parameters'];out['result_label']=LABEL
                    out['fit_stage']='train_only' if split=='validation' else 'fresh_train_validation_refit'
                    parts.append(out)
                tuning.extend(dict(r,fold=fold.fold_id,config=config) for r in records)
                counts.append(dict(fold=fold.fold_id,config=config,train=len(fold.train),refit=len(refit),validation=len(validation),test=len(test)))
                del fitted,prep,xr
    panel=pd.concat(parts,ignore_index=True);expected=pd.concat(expected,ignore_index=True)
    validate_forecasts(panel,expected,protocol['rq2']['validation']['test_folds'])
    return panel,pd.DataFrame(tuning),expected,dict(fits=fits,selection_fits=fits//8*7,refits=fits//8,counts=counts)


def published_comparison(tuning, panel, root=ROOT):
    published=pd.read_csv(BytesIO(reference_bytes('results/tables/rq2_ml_tuning_log.csv',root)))
    published=published.rename(columns={'ablation':'config','validation_MAE':'published_validation_MAE','selected':'published_selected'})
    # JSON parameters use canonical serialization for joins, not display whitespace.
    for frame in [published,tuning]:
        frame['parameters']=frame.parameters.map(lambda x:json.dumps(json.loads(x),sort_keys=True))
    result=tuning.merge(published,on=['fold','model','config','parameters'],how='left',validate='1:1',indicator=True)
    if not result['_merge'].eq('both').all():raise ValueError('Candidate differs from pinned M3 grid')
    result['delta_validation_MAE']=result.validation_MAE-result.published_validation_MAE
    published_test=pd.read_csv(BytesIO(reference_bytes('results/tables/rq2_ml_experiments.csv',root))).rename(columns={'ablation':'config'})
    records=[]
    for (fold,config),g in panel.loc[panel.split.eq('test')].groupby(['fold','config']):
        model=g.model.iloc[0];p=published_test.loc[published_test.fold.eq(fold)&published_test.config.eq(config)&published_test.model.eq(model)].iloc[0]
        mae=stable_mean(np.abs(g.y_true-g.clipped_prediction))
        records.append(dict(fold=fold,config=config,model=model,test_MAE=mae,published_test_MAE=p.test_MAE,
                            delta_test_MAE=mae-p.test_MAE,parameters_match=g.selected_parameters.iloc[0]==p.best_parameters))
    return result.drop(columns='_merge'),pd.DataFrame(records)


def approval_request(root=ROOT):
    from scripts.m4_pilot import FEATURE_RELATIVE, verify_candidate_bundle
    root=Path(root).resolve();evidence=verify_candidate_bundle(root/FEATURE_RELATIVE)
    for path in PINS:reference_bytes(path,root)
    files=[root/'configs/research_protocol.yaml',root/'configs/ablation_M0_M5.yaml',root/'scripts/m4_pilot.py',
           root/'docs/m4_phase3e_reproduction.md']
    for directory in ['src/decision','src/forecasting','src/features','src/data']:files+=list((root/directory).glob('*.py'))
    return json_snapshot(dict(status='PENDING_APPROVAL',specification=SPEC,pinned_git_objects=PINS,
                environment=dict(python=platform.python_version(),packages={p:version(p) for p in PACKAGES}),
                file_sha256={**evidence,**{str(p.resolve()):file_hash(p) for p in files}}))


def validate_approval(receipt, root=ROOT):
    request=receipt.get('request',{})
    if receipt.get('status')!='APPROVED' or request.get('status')!='PENDING_APPROVAL':raise ValueError('Reproduction approval required before fitting')
    if receipt.get('request_sha256')!=semantic_hash(request) or request!=approval_request(root):raise ValueError('Approved reproduction bindings drifted')
    for key in ['approved_by','approval_evidence','approval_communicated_date']:
        if not isinstance(receipt.get(key),str) or not receipt[key].strip():raise ValueError('Missing approval evidence')
    datetime.strptime(receipt['approval_communicated_date'],'%Y-%m-%d')
    verify_frozen_files({'file_sha256':request['file_sha256'],'input':{}})


def inspect_bundle(directory):
    directory=Path(directory)
    if (directory/'FAILED.json').exists():return 'failed'
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text())
        expected={'predictions.csv','tuning.csv','expected_rows.csv','published_validation_comparison.csv','published_test_comparison.csv','manifest.json','STARTED.json'}
        if set(marker['files'])!=expected:return 'corrupt'
        if any(file_hash(directory/n)!=h for n,h in marker['files'].items()):return 'corrupt'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['output_sha256']!={n:h for n,h in marker['files'].items() if n!='manifest.json'}:return 'corrupt'
        return 'complete'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete'


def _json(path, record):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False)


def prefit_validation(root=ROOT):
    """Read-only path shared by dry checks and approved runs; never calls fitting."""
    from scripts.m4_pilot import FEATURE_RELATIVE, verify_candidate_bundle
    root=Path(root).resolve()
    candidate_evidence=verify_candidate_bundle(root/FEATURE_RELATIVE)
    for path in PINS:reference_bytes(path,root)
    protocol=read_yaml(root/'configs/research_protocol.yaml');ablation=read_yaml(root/'configs/ablation_M0_M5.yaml')
    if [f['fold_id'] for f in protocol['rq2']['validation']['test_folds']]!=SPEC['folds']:raise ValueError('Require F1-F4')
    data,preflight=production_preflight(root/FEATURE_RELATIVE,protocol,ablation,root)
    if preflight['hard_failures']:raise ValueError('Preflight hard failures')
    if inspect_run(root,FINAL)['status']!='complete':raise ValueError('Frozen quantile reference incomplete')
    reference_dir=root/'results/experiments'/('m4_quantiles_'+FINAL)
    frozen=json.loads((reference_dir/'manifest.json').read_text());verify_frozen_files(frozen['provenance'])
    reference_panel=pd.read_csv(reference_dir/('m4_quantiles_'+FINAL+'.csv'),parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    expected=reference_panel[KEY+['y_true']].drop_duplicates()
    # Validate exact target/key scope BEFORE model fitting, not only at export.
    actual=[]
    for fold in iter_folds(data,protocol,ablation,protocol['scope']['deep_study_families']):
        for split in ['validation','test']:
            rows=getattr(fold,split).loc[lambda x:x.score_mask]
            e=rows[['date','store_nbr','family','city','sales']].rename(columns={'sales':'y_true'}).copy()
            e['fold'],e['split']=fold.fold_id,split;actual.append(e)
    actual=pd.concat(actual,ignore_index=True)
    assert_matched_rows(expected,actual)
    for record in frozen['scored_key_contract']:
        rows=actual.loc[actual.fold.eq(record['fold'])&actual.split.eq(record['split'])]
        if len(rows)!=record['rows'] or scored_key_digest(rows)!=record['sha256']:raise ValueError('Frozen scored-key digest mismatch')
    evidence={**preflight['file_sha256'],**candidate_evidence,
              **{str(p.resolve()):file_hash(p) for p in reference_dir.iterdir() if p.is_file()}}
    verify_frozen_files({'file_sha256':evidence,'input':{}})
    return data,protocol,ablation,expected,preflight,evidence


def run_approved(approval_path,root=ROOT):
    root=Path(root).resolve();approval_path=Path(approval_path).resolve();receipt=json.loads(approval_path.read_text())
    validate_approval(receipt,root);approval_digest=file_hash(approval_path)
    data,protocol,ablation,expected,preflight,evidence=prefit_validation(root)
    evidence[str(approval_path)]=approval_digest
    validate_approval(receipt,root)
    verify_frozen_files({'file_sha256':evidence,'input':{}})
    parent=root/'results/experiments';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/('m4_reproduced_m3_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=parent));started=perf_counter()
    _json(stage/'STARTED.json',dict(approval=receipt,file_sha256=evidence,preflight=preflight,label=LABEL))
    try:
        panel,tuning,_,diagnostics=reproduce(data,protocol,ablation)
        if diagnostics['fits']!=64 or diagnostics['selection_fits']!=56 or diagnostics['refits']!=8:raise ValueError('Unexpected fit budget')
        validate_forecasts(panel,expected,protocol['rq2']['validation']['test_folds'])
        comparison,test_comparison=published_comparison(tuning.copy(),panel,root)
        tables={'predictions.csv':panel,'tuning.csv':tuning,'expected_rows.csv':expected,
                'published_validation_comparison.csv':comparison,'published_test_comparison.csv':test_comparison}
        for name,frame in tables.items():
            with (stage/name).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
            loaded=pd.read_csv(stage/name,parse_dates=['date'] if 'date' in frame else None,
                               dtype={'store_nbr':str} if 'store_nbr' in frame else None,float_precision='round_trip')
            pd.testing.assert_frame_equal(loaded,frame.reset_index(drop=True),check_dtype=False)
        validate_approval(receipt,root);verify_frozen_files({'file_sha256':evidence,'input':{}})
        _json(stage/'manifest.json',dict(label=LABEL,approval=receipt,file_sha256=evidence,preflight=preflight,
            diagnostics=diagnostics,wall_seconds=perf_counter()-started,calibration_status='PENDING_APPROVAL',
            output_sha256={n:file_hash(stage/n) for n in [*tables,'STARTED.json']}))
        _json(stage/'COMPLETE.json',{'files':{n:file_hash(stage/n) for n in [*tables,'manifest.json','STARTED.json']}})
        if inspect_bundle(stage)!='complete':raise ValueError('Output integrity failure')
        if destination.exists():raise FileExistsError('Never overwrite previous runs')
        stage.rename(destination);return destination
    except Exception as exc:
        _json(stage/'FAILED.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-request',type=Path);g.add_argument('--approval',type=Path)
    g.add_argument('--prefit-check',action='store_true');args=parser.parse_args()
    if args.prepare_request:_json(args.prepare_request,approval_request())
    elif args.prefit_check:
        _,_,_,expected,report,_=prefit_validation()
        print(json.dumps(dict(mode='PREFIT_ONLY_NO_FITTING',hard_failures=report['hard_failures'],
            warnings=report['warnings'],unavailable=report['unavailable'],
            scored_rows=expected.groupby(['fold','split']).size().reset_index(name='rows').to_dict(orient='records')),indent=2))
    else:print(run_approved(args.approval))


if __name__=='__main__':main()
