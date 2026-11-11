"""P1 infrastructure only. Residual methodology PENDING_APPROVAL; no real policy CLI."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import inspect
import json

import numpy as np
import pandas as pd

from src.forecasting import baselines as m1
from src.forecasting.point_models import prepare_features
from src.forecasting.rolling_validation import iter_folds, read_yaml
from .cost_simulation import newsvendor_cost
from .policy_evaluation import configured_scenarios
from .quantile_models import _finite_vector, nonnegative_quantity, normalize_tau, canonical_tau_key
from .quantile_contracts import file_hash, json_snapshot, freeze_provenance, verify_frozen_files, scored_key_digest, semantic_hash
from .quantile_artifacts import inspect_run
from .quantile_preflight import production_preflight, ProductionPreflightError


ROOT = Path(__file__).resolve().parents[2]
STATUS = 'PENDING_APPROVAL'
OBS = ['date','store_nbr','family','city','fold']
BASE = ['fold','model','config']
POINT_KEY = OBS + ['model','config']
FINAL_ID = 'final_f1_f4_m0_m3_20261108T123157Z_eb881cc4c56a'


class MethodologyPendingError(ValueError):
    pass


@dataclass(frozen=True)
class ResidualSpec:
    """Every statistical choice explicit; no defaults imply team approval."""
    pooling: tuple
    quantile_method: str
    minimum_samples: int
    fallback: tuple
    methodology_status: str = field(default=STATUS, init=False)

    def __post_init__(self):
        levels=(self.pooling,)+self.fallback if isinstance(self.fallback,tuple) else ()
        if (not levels or any(not isinstance(level,tuple) or any(not isinstance(c,str) for c in level) or len(set(level))!=len(level)
                             or set(level)-{'store_nbr','family','city'} for level in levels)):
            raise ValueError('Pooling/fallback must be explicit tuples of store_nbr/family/city columns')
        if len(set(levels))!=len(levels) or any(not set(b)<set(a) for a,b in zip(levels,levels[1:])):
            raise ValueError('Fallback must be a unique, strictly coarser predeclared hierarchy')
        if self.quantile_method not in {'linear','lower','higher','nearest','midpoint'}:
            raise ValueError('Specify a supported empirical quantile method')
        if isinstance(self.minimum_samples,(bool,np.bool_)) or not isinstance(self.minimum_samples,(int,np.integer)) or self.minimum_samples<1:
            raise ValueError('minimum_samples must be a positive integer')
        object.__setattr__(self,'minimum_samples',int(self.minimum_samples))

    def metadata(self):
        return json_snapshot(asdict(self))


@dataclass(frozen=True)
class P1Inputs:
    validation: pd.DataFrame
    test: pd.DataFrame
    diagnostics: pd.DataFrame
    fold_specs: list
    data_kind: str
    provenance: dict


def _normalise(frame, *, target_required=True, point_required=True):
    required=POINT_KEY+(['y_true'] if target_required else [])+(['y_pred'] if point_required else [])
    if frame.empty or set(required)-set(frame) or frame[required].isna().any().any():
        raise ValueError('Point panel has missing/null required keys or values')
    out=frame.copy();out['date']=pd.to_datetime(out.date,errors='raise')
    if out.date.dt.tz is not None or not out.date.eq(out.date.dt.normalize()).all():
        raise ValueError('Point dates must be timezone-naive calendar days')
    for c in ['store_nbr','family','city','fold','model','config']:out[c]=out[c].astype(str)
    if out.duplicated(POINT_KEY).any():raise ValueError('Duplicate point prediction keys')
    for c in (['y_true'] if target_required else [])+(['y_pred'] if point_required else []):
        out[c]=_finite_vector(out[c].to_numpy(),c)
    if target_required and out.y_true.lt(0).any():raise ValueError('Observed sales must be nonnegative')
    return out


def _alignment(expected, actual, *, check_prediction=False):
    columns=POINT_KEY+['y_true']+(['y_pred'] if check_prediction else [])
    expected=_normalise(expected,point_required=check_prediction)
    actual=_normalise(actual,point_required=check_prediction)
    paired=expected[columns].merge(actual[columns],on=POINT_KEY,how='outer',validate='1:1',
                                   suffixes=('_expected','_actual'),indicator=True)
    if not paired['_merge'].eq('both').all():raise ValueError('Exact scored point keys do not match')
    for c in ['y_true']+(['y_pred'] if check_prediction else []):
        if not np.array_equal(paired[c+'_expected'],paired[c+'_actual']):
            raise ValueError(f'Point {c} disagrees with scored source rows/rule')


def _point_output(rows, fold_id, split):
    forecast,fallbacks=m1.seasonal_naive(rows)
    out=_normalise(m1.to_output(rows,forecast,fold_id,'SeasonalNaive'))
    if out.y_pred.lt(0).any():raise ValueError('Seasonal Naive rule produced a negative prediction')
    out['split'],out['score_mask']=split,True
    out['fit_stage']='fixed_rule_no_fitting_one_day_ahead'
    out['forecast_origin']=out.date-pd.Timedelta(days=1);out['horizon_days']=1
    return out,dict(fold=fold_id,split=split,rows=len(rows),lag7_missing=int(fallbacks),
                    lag14_used=int((rows.sales_lag7.isna()&rows.sales_lag14.notna()).sum()),
                    rolling7_used=int((rows.sales_lag7.isna()&rows.sales_lag14.isna()).sum()))


def build_p1_inputs(features, baseline_predictions, protocol, ablation, *, data_kind='unverified'):
    """Reconstruct validation only; audit and retain existing SeasonalNaive tests."""
    if data_kind not in {'synthetic','real','unverified'}:raise ValueError('Unknown input data_kind')
    if {'excluded_closure','excluded_post_reopening'}&set(features):
        raise ValueError('Use clean M1 feature contract, not legacy scoring-helper columns')
    protocol,ablation=json_snapshot(protocol),json_snapshot(ablation)
    families=protocol['scope']['deep_study_families']
    if set(m1.OUTPUT_COLUMNS)-set(baseline_predictions):raise ValueError('Missing M1 baseline prediction columns')
    baseline=baseline_predictions.loc[baseline_predictions.model.eq('SeasonalNaive')].copy()
    baseline=_normalise(baseline)
    if set(baseline.config)!={'M0'}:raise ValueError('P1 SeasonalNaive must use M0')
    baseline=baseline.loc[baseline.family.isin(families)]
    expected_fold_ids={f['fold_id'] for f in protocol['rq2']['validation']['test_folds']}
    if set(baseline.fold)!=expected_fold_ids:raise ValueError('Baseline fold IDs differ from shared schedule')
    data=prepare_features(features,ready=True)
    validation,test,diagnostics=[],[],[]
    for fold in iter_folds(data,protocol,ablation,families):
        for split in ['validation','test']:
            scored=getattr(fold,split).loc[lambda x:x.score_mask]
            expected,log=_point_output(scored,fold.fold_id,split)
            diagnostics.append(log)
            if split=='validation':validation.append(expected)
            else:
                existing=baseline.loc[baseline.fold.eq(fold.fold_id)]
                _alignment(expected,existing,check_prediction=True)
                keep=existing.copy()
                for c in ['split','score_mask','fit_stage','forecast_origin','horizon_days']:
                    if c=='forecast_origin':keep[c]=keep.date-pd.Timedelta(days=1)
                    else:keep[c]=expected[c].iloc[0]
                test.append(keep)
    return P1Inputs(pd.concat(validation,ignore_index=True),pd.concat(test,ignore_index=True),
                    pd.DataFrame(diagnostics),json_snapshot(protocol['rq2']['validation']['test_folds']),data_kind,
                    {'stage':'point_forecasts_only','baseline_provenance_limitation':
                     'Existing baseline CSV has no upstream manifest/completion marker; exact source keys/truth/rule checked locally'})


def load_real_p1_inputs(root=ROOT):
    """Read-only point-input loader. NO residual quantiles, policy Q or cost evaluation."""
    from scripts.m4_pilot import verify_candidate_bundle, FEATURE_RELATIVE
    root=Path(root).resolve();evidence=verify_candidate_bundle(root/FEATURE_RELATIVE)
    baseline_path=root/'data/processed/baselines.csv';evidence[str(baseline_path)]=file_hash(baseline_path)
    protocol=read_yaml(root/'configs/research_protocol.yaml');ablation=read_yaml(root/'configs/ablation_M0_M5.yaml')
    data,preflight=production_preflight(root/FEATURE_RELATIVE,protocol,ablation,root)
    if preflight['hard_failures']:raise ProductionPreflightError(preflight)
    verify_frozen_files({'file_sha256':evidence,'input':{}})
    baseline=pd.read_csv(baseline_path,parse_dates=['date'],dtype={'store_nbr':str,'family':str,'city':str},float_precision='round_trip')
    inputs=build_p1_inputs(data,baseline,protocol,ablation,data_kind='real')
    if inspect_run(root,FINAL_ID)['status']!='complete':raise ValueError('Reference final quantile bundle is not complete')
    reference_dir=root/'results/experiments'/('m4_quantiles_'+FINAL_ID)
    reference=json.loads((reference_dir/'manifest.json').read_text())
    verify_frozen_files(reference['provenance'])
    for key in reference['scored_key_contract']:
        rows=getattr(inputs,key['split']).loc[lambda x:x.fold.eq(key['fold'])]
        if len(rows)!=key['rows'] or scored_key_digest(rows)!=key['sha256']:
            raise ValueError('P1 point rows differ from frozen M4 scored-key contract')
    provenance=freeze_provenance(root,protocol,ablation,{'mode':'p1_point_inputs_only',**preflight})
    provenance['input']['file_sha256'].update(evidence)
    for path in [Path(__file__).resolve(),root/'docs/m4_phase3d_p1_infrastructure.md',reference_dir/'manifest.json',Path(inspect.getfile(m1)).resolve()]:
        provenance['file_sha256'][str(path)]=file_hash(path)
    verify_frozen_files(provenance)
    return P1Inputs(inputs.validation,inputs.test,inputs.diagnostics,inputs.fold_specs,'real',provenance)


def _window_checked(points, fold_specs, split, *, target_required):
    out=_normalise(points,target_required=target_required)
    required={'split','score_mask','forecast_origin','horizon_days'}
    if required-set(out) or out[list(required)].isna().any().any():raise ValueError('Missing point eligibility/origin metadata')
    if not out.split.eq(split).all():raise ValueError(f'Expected {split} predictions only')
    if not out.score_mask.isin([True,False]).all() or not out.score_mask.astype(bool).all():
        raise ValueError('Residual/policy rows must be scored eligible rows')
    if not pd.to_datetime(out.forecast_origin).eq(out.date-pd.Timedelta(days=1)).all() or not out.horizon_days.eq(1).all():
        raise ValueError('Point forecast origin/horizon mismatch')
    mapping={f['fold_id']:f for f in fold_specs}
    if not mapping or len(mapping)!=len(fold_specs):raise ValueError('Require unique nonempty declared fold specifications')
    if not set(out.fold).issubset(mapping):raise ValueError('Unknown fold in point predictions')
    for fold,g in out.groupby('fold'):
        f=mapping[fold]
        train_end=pd.Timestamp(f['train'][1]);va,ve=map(pd.Timestamp,f['validation']);ta,te=map(pd.Timestamp,f['test'])
        if not train_end<va<=ve<ta<=te:raise ValueError('Invalid declared fold chronology')
        start,end=(va,ve) if split=='validation' else (ta,te)
        if not g.date.between(start,end).all():raise ValueError('Point dates outside their declared fold/split')
    return out


@dataclass(frozen=True)
class ResidualAdjuster:
    spec: ResidualSpec
    fold_specs: list
    taus: tuple
    offsets: dict
    calibration: pd.DataFrame

    def predict(self, test_points):
        """Targets are not required or consulted for Q construction."""
        test=_window_checked(test_points,self.fold_specs,'test',target_required=False)
        output=[];levels=(self.spec.pooling,)+self.spec.fallback
        for row in test.to_dict(orient='records'):
            chosen=None
            for level,columns in enumerate(levels):
                key=(row['fold'],row['model'],row['config'],level)+tuple(row[c] for c in columns)
                if key in self.offsets:chosen=(level,columns,self.offsets[key]);break
            if chosen is None:raise ValueError('No sufficiently populated predeclared calibration group/fallback')
            level,columns,details=chosen
            for tau,offset in zip(self.taus,details['quantiles']):
                with np.errstate(over='ignore',invalid='ignore'):raw=float(row['y_pred']+offset)
                if not np.isfinite(raw):raise ValueError('Adjusted quantity exceeds finite float64 range')
                output.append({**{c:row[c] for c in POINT_KEY},'split':'test','tau':tau,'tau_key':canonical_tau_key(tau),
                    'point_prediction':row['y_pred'],'residual_quantile':offset,'quantity_before_clip':raw,
                    'quantity':float(nonnegative_quantity([raw])[0]),'calibration_samples':details['n'],
                    'calibration_dates':details['n_dates'],'calibration_end':details['date_max'],
                    'pooling_columns':','.join(columns) or 'fold_model_config_global',
                    'fallback_level':level,'used_fallback':level>0,'quantile_method':self.spec.quantile_method,
                    'minimum_samples':self.spec.minimum_samples,'methodology_spec_sha256':semantic_hash(self.spec.metadata()),
                    'methodology_status':STATUS})
        return pd.DataFrame(output)


def fit_residual_adjuster(validation, spec, fold_specs, taus):
    """Synthetic/dev primitive; validation only, never future-fold/test residuals."""
    if not isinstance(spec,ResidualSpec):raise ValueError('Explicit ResidualSpec required')
    levels=(spec.pooling,)+spec.fallback
    taus=tuple(normalize_tau(t) for t in taus)
    if tuple(sorted(set(taus)))!=taus or not taus:raise ValueError('Quantile levels must be unique and increasing')
    data=_window_checked(validation,fold_specs,'validation',target_required=True)
    with np.errstate(over='ignore',invalid='ignore'):residual=data.y_true.to_numpy()-data.y_pred.to_numpy()
    if not np.isfinite(residual).all():raise ValueError('Signed residuals exceed finite float64 range')
    data['residual']=residual
    offsets={};audit=[]
    for level,columns in enumerate(levels):
        names=BASE+list(columns)
        for key,g in data.groupby(names,observed=True,sort=True):
            if len(g)<spec.minimum_samples:continue
            try:
                with np.errstate(over='raise',invalid='raise'):
                    values=np.quantile(g.residual.to_numpy(),taus,method=spec.quantile_method)
            except FloatingPointError as exc:raise ValueError('Residual quantile interpolation overflow') from exc
            if not np.isfinite(values).all():raise ValueError('Residual quantiles must be finite')
            keys=tuple(key[:3])+(level,)+tuple(key[3:])
            details=dict(quantiles=tuple(float(x) for x in values),n=len(g),n_dates=g.date.nunique(),date_max=g.date.max())
            offsets[keys]=details
            for tau,value in zip(taus,values):
                audit.append({**dict(zip(names,key)),'fallback_level':level,'tau_key':canonical_tau_key(tau),
                              'residual_quantile':float(value),'n':len(g),'n_dates':g.date.nunique(),
                              'date_min':g.date.min(),'date_max':g.date.max(),'quantile_method':spec.quantile_method,
                              'methodology_status':STATUS})
    if not offsets:raise ValueError('No calibration groups satisfy minimum_samples')
    return ResidualAdjuster(spec,json_snapshot(fold_specs),taus,offsets,pd.DataFrame(audit))


def evaluate_p1_synthetic(inputs, spec, cost_config):
    """Development-only integration. No approved real-data evaluation path yet."""
    if not isinstance(inputs,P1Inputs) or inputs.data_kind!='synthetic':
        raise MethodologyPendingError('P1 residual methodology PENDING_APPROVAL: real/unverified policy evaluation is disabled')
    return _evaluate_p1(inputs, spec, cost_config)


def _evaluate_p1(inputs, spec, cost_config):
    """Internal arithmetic; real callers must pass the separate approval gate."""
    scenarios=configured_scenarios(cost_config)
    adjuster=fit_residual_adjuster(inputs.validation,spec,inputs.fold_specs,sorted(s['tau'] for s in scenarios))
    points=_window_checked(inputs.test,inputs.fold_specs,'test',target_required=True)
    quantities=adjuster.predict(points.drop(columns='y_true'))
    quantities=quantities.merge(points[POINT_KEY+['y_true']],on=POINT_KEY,validate='m:1')
    output=[]
    for s in scenarios:
        rows=quantities.loc[quantities.tau_key.eq(s['tau_key'])].copy()
        rows['policy'],rows['scenario'],rows['c_over'],rows['c_under']='P1',s['scenario'],s['c_over'],s['c_under']
        rows['simulated_cost']=newsvendor_cost(rows.y_true,rows.quantity,s['c_over'],s['c_under'])
        rows['overstock_units']=np.maximum(rows.quantity-rows.y_true,0)
        rows['understock_units']=np.maximum(rows.y_true-rows.quantity,0)
        rows['result_label']='SYNTHETIC DEVELOPMENT ONLY; PENDING_APPROVAL; no independent/conformal calibration guarantee'
        output.append(rows)
    return pd.concat(output,ignore_index=True),adjuster.calibration
