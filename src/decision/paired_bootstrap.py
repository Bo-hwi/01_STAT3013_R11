"""Descriptive paired calendar moving-block bootstrap for fixed P2/P3 costs."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from . import policy_evaluation as costs
from .quantile_contracts import file_hash, json_snapshot, verify_frozen_files


ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'm4_costs_p2_p3_20261108T134955Z_b112ad3e7866'
OBS = ['date', 'store_nbr', 'family', 'city', 'fold']
KEY = OBS + ['scenario', 'variant', 'tau_key']
DESIGN = dict(seed=42, resamples=5000, primary_block_length=7, block_lengths=[3,7,14],
              confidence_level=.95, interval='percentile_linear_quantile',
              method_status='ANALYST_DECLARED; not previously team-approved',
              sampling='non-circular moving blocks within each contiguous fold segment',
              pooled='stratified by fold/segment; same replicate combines independently sampled folds',
              estimand='row-weighted fixed-policy cost means and ratio of summed paired costs',
              series_pairing='same date draws for P2/P3, all scenarios and both variants',
              zero_denominator='relative delta undefined; finite-replicate intervals report valid counts',
              sign_flip='strict opposite sign to observed point estimate; undefined if observed point is zero/undefined')
LABEL = 'DESCRIPTIVE BOOTSTRAP INTERVALS FOR SIMULATED COSTS; no causal/operational/significance guarantee'


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def date_segments(dates):
    dates = pd.DatetimeIndex(dates)
    if (dates.empty or dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize())
            or not dates.is_unique or not dates.is_monotonic_increasing):
        raise ValueError('Dates must be unique sorted non-null timezone-naive calendar days')
    cuts = np.flatnonzero(np.diff(dates.to_numpy()).astype('timedelta64[D]') != np.timedelta64(1,'D'))+1
    return dates, np.split(np.arange(len(dates)), cuts)


def sample_date_indices(dates, block_length, resamples, rng):
    """Retain segment lengths; NEVER form a sampled block across a gap."""
    block_length = _positive_integer(block_length, 'block_length')
    resamples = _positive_integer(resamples, 'resamples')
    dates, segments = date_segments(dates)
    sampled = []
    for segment in segments:
        n = len(segment)
        if n < block_length:
            raise ValueError('Contiguous segment shorter than declared block length; do not drop days or shrink L silently')
        starts = rng.integers(0, n-block_length+1, size=(resamples, (n+block_length-1)//block_length))
        local = (starts[:,:,None] + np.arange(block_length)).reshape(resamples,-1)[:,:n]
        sampled.append(segment[local])
    return np.concatenate(sampled, axis=1)


def pair_daily_costs(rows, expected_scenarios=None):
    """Lossless day totals for mean/ratio estimands; all observations travel together."""
    required = KEY + ['policy','y_true','simulated_cost']
    if rows.empty or set(required)-set(rows) or rows[required].isna().any().any():
        raise ValueError('Missing/null cost keys or values')
    if set(rows.policy) != {'P2','P3'} or not set(rows.variant).issubset(costs.VARIANTS) or 'corrected' not in set(rows.variant):
        raise ValueError('Require P2/P3 and separate corrected/raw sensitivity variants')
    frame = rows.copy()
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError('Require calendar dates without time zones')
    if frame.duplicated(KEY+['policy']).any():
        raise ValueError('Duplicate policy observation keys')
    for column in ['simulated_cost','y_true']:
        values=frame[column].to_numpy(float)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Costs/demand must be finite nonnegative values')
    if expected_scenarios is not None:
        mapping={s['scenario']:s['tau_key'] for s in expected_scenarios}
        if set(frame.scenario)!=set(mapping) or not frame.scenario.map(mapping).eq(frame.tau_key).all():
            raise ValueError('Cost scenarios/tau keys differ from frozen source config')
    pair=frame.loc[frame.policy.eq('P2'),KEY+['y_true','simulated_cost']].merge(
        frame.loc[frame.policy.eq('P3'),KEY+['y_true','simulated_cost']],on=KEY,how='outer',
        suffixes=('_P2','_P3'),validate='1:1',indicator=True)
    if not pair['_merge'].eq('both').all() or not pair.y_true_P2.equals(pair.y_true_P3):
        raise ValueError('P2/P3 keys or observed demand disagree')
    if pair.groupby(OBS,dropna=False).y_true_P2.nunique().gt(1).any():
        raise ValueError('Demand differs across scenarios/variants')
    reference=None
    for _,g in pair.groupby(['scenario','variant'],sort=True):
        keys=g[OBS].sort_values(OBS).reset_index(drop=True)
        if reference is None:reference=keys
        elif not keys.equals(reference):raise ValueError('Scenarios/variants do not share observation keys')
    daily=pair.groupby(['fold','date','scenario','variant'],sort=True,observed=True).agg(
        cost_P2=('simulated_cost_P2','sum'),cost_P3=('simulated_cost_P3','sum'),n=('simulated_cost_P2','size')).reset_index()
    if not np.isfinite(daily[['cost_P2','cost_P3']].to_numpy()).all():
        raise ValueError('Daily cost sums overflow')
    return daily


def summarize_interval(draws, point, confidence):
    values=np.asarray(draws,dtype=float);finite=values[np.isfinite(values)]
    if not finite.size:
        return dict(point_estimate=point,lower=np.nan,upper=np.nan,includes_zero=None,
                    valid_resamples=0,undefined_resamples=len(values),negative_fraction=np.nan,
                    positive_fraction=np.nan,zero_fraction=np.nan,opposite_point_sign_fraction=np.nan)
    lower,upper=np.quantile(finite,[(1-confidence)/2,1-(1-confidence)/2],method='linear')
    flip=float(np.mean(finite*np.sign(point)<0)) if np.isfinite(point) and point!=0 else np.nan
    return dict(point_estimate=point,lower=float(lower),upper=float(upper),includes_zero=bool(lower<=0<=upper),
                valid_resamples=len(finite),undefined_resamples=len(values)-len(finite),
                negative_fraction=float(np.mean(finite<0)),positive_fraction=float(np.mean(finite>0)),
                zero_fraction=float(np.mean(finite==0)),opposite_point_sign_fraction=flip)


def bootstrap_costs(daily, block_lengths=(3,7,14), resamples=5000, seed=42, confidence=.95):
    resamples=_positive_integer(resamples,'resamples')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    lengths=[_positive_integer(x,'block_length') for x in block_lengths]
    if not lengths or len(lengths)!=len(set(lengths)) or not 0<confidence<1:
        raise ValueError('Require unique block lengths and confidence in (0,1)')
    if daily.empty or daily.duplicated(['fold','date','scenario','variant']).any():
        raise ValueError('Daily panels must be nonempty and unique')
    series=sorted(set(zip(daily.scenario,daily.variant)));folds=sorted(daily.fold.unique())
    arrays={};metadata=[]
    for fold in folds:
        g=daily.loc[daily.fold.eq(fold)];dates=pd.DatetimeIndex(sorted(g.date.unique()))
        dates,segments=date_segments(dates)
        values=[];n=None
        for scenario,variant in series:
            part=g.loc[g.scenario.eq(scenario)&g.variant.eq(variant)].set_index('date').reindex(dates)
            if part[['cost_P2','cost_P3','n']].isna().any().any():raise ValueError('Daily series have missing dates')
            counts=part.n.to_numpy(float)
            if n is None:n=counts
            elif not np.array_equal(n,counts):raise ValueError('Daily series counts disagree')
            values.append(part[['cost_P2','cost_P3']].to_numpy(float))
        stack=np.stack(values,axis=1)
        if not np.isfinite(stack).all() or not np.isfinite(n).all() or (stack<0).any() or (n<=0).any() or not np.equal(n,np.floor(n)).all():
            raise ValueError('Invalid daily costs/counts')
        arrays[fold]=(dates,stack,n)
        metadata.extend(dict(fold=fold,index=i,date=str(d.date()),segment=next(j for j,s in enumerate(segments) if i in s),n_observations=int(n[i])) for i,d in enumerate(dates))
    intervals=[];replicates=[];indices={}
    for length in lengths:
        children=np.random.SeedSequence([int(seed),length]).spawn(len(folds))
        totals={};points={}
        for fold,child in zip(folds,children):
            dates,values,n=arrays[fold]
            idx=sample_date_indices(dates,length,resamples,np.random.default_rng(child))
            indices[f'L{length}_{fold}']=idx.astype(np.int32)
            sums=values[idx].sum(axis=1);counts=n[idx].sum(axis=1)
            if not np.isfinite(sums).all():raise ValueError('Bootstrap sums overflow')
            totals[fold]=(sums,counts);points[fold]=(values.sum(axis=0),n.sum())
        totals['pooled_F1_F4']=(sum(s for s,n in totals.values()),sum(n for s,n in totals.values()))
        points['pooled_F1_F4']=(sum(s for s,n in points.values()),sum(n for s,n in points.values()))
        for scope,(sums,counts) in totals.items():
            point_sums,point_n=points[scope]
            if not np.isfinite(sums).all() or not np.isfinite(point_sums).all() or not np.isfinite(counts).all():
                raise ValueError('Pooled costs/counts exceed finite range')
            for j,(scenario,variant) in enumerate(series):
                a,b=sums[:,j,0],sums[:,j,1]
                delta=(b-a)/counts
                relative=np.full(resamples,np.nan);np.divide(b-a,a,out=relative,where=a>0);relative*=100
                pa,pb=point_sums[j];point_delta=float((pb-pa)/point_n)
                point_relative=float(100*(pb-pa)/pa) if pa>0 else np.nan
                if not np.isfinite(delta).all() or np.isinf(relative).any():raise ValueError('Bootstrap estimates exceed finite range')
                fields=dict(result_label=LABEL,scope=scope,scenario=scenario,variant=variant,
                            block_length=length,resamples=resamples,confidence_level=confidence,
                            n_dates=sum(len(x[0]) for x in arrays.values()) if scope=='pooled_F1_F4' else len(arrays[scope][0]),
                            n_observations=int(point_n))
                for metric,draws,point in [('mean_cost_delta_P3_minus_P2',delta,point_delta),('relative_cost_delta_pct',relative,point_relative)]:
                    intervals.append(dict(fields,metric=metric,**summarize_interval(draws,point,confidence)))
                replicates.append(pd.DataFrame(dict(scope=scope,scenario=scenario,variant=variant,block_length=length,
                    replicate=np.arange(resamples),sampled_observations=counts.astype(int),
                    mean_cost_delta_P3_minus_P2=delta,relative_cost_delta_pct=relative)))
    return pd.DataFrame(intervals),pd.concat(replicates,ignore_index=True),indices,pd.DataFrame(metadata)


def _json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def inspect_bootstrap(directory):
    directory=Path(directory)
    if not directory.exists():return 'absent'
    if (directory/'FAILED.json').exists():return 'failed'
    if not (directory/'COMPLETE.json').exists():return 'incomplete'
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text())
        required={'STARTED.json','daily_costs.csv','intervals.csv','replicates.csv','date_layout.csv','resampling_indices.npz','manifest.json'}
        if set(marker['files'])!=required:return 'corrupt'
        if any(Path(n).name!=n or file_hash(directory/n)!=h for n,h in marker['files'].items()):return 'corrupt'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['output_sha256']!={n:h for n,h in marker['files'].items() if n!='manifest.json'}:return 'corrupt'
    except (OSError,ValueError,KeyError,TypeError):return 'corrupt'
    return 'complete'


def run_bootstrap(root=ROOT):
    root=Path(root).resolve();source=root/'results/experiments'/SOURCE
    if costs.inspect_evaluation(source)!='complete':raise ValueError('Source cost bundle is not complete')
    source_manifest=json.loads((source/'manifest.json').read_text())
    verify_frozen_files({'file_sha256':source_manifest['file_sha256'],'input':{}})
    verify_frozen_files(source_manifest['source_forecast_provenance'])
    hashes={str(p.resolve()):file_hash(p) for p in source.iterdir() if p.is_file()}
    for path in [Path(__file__).resolve(),root/'docs/m4_phase3b_bootstrap.md']:
        hashes[str(path)]=file_hash(path)
    started=perf_counter()
    usecols=KEY+['policy','y_true','simulated_cost']
    rows=pd.read_csv(source/'row_costs.csv',usecols=usecols,parse_dates=['date'],
                     dtype={'store_nbr':str,'family':str,'city':str},float_precision='round_trip')
    if len(rows)!=source_manifest['row_cost_records']:raise ValueError('Source cost rows truncated')
    if set(rows.variant)!=set(costs.VARIANTS):raise ValueError('Source raw/corrected variants incomplete')
    daily=pair_daily_costs(rows,costs.configured_scenarios(source_manifest['scenarios']))
    for _,g in daily.groupby('fold'):date_segments(pd.DatetimeIndex(sorted(g.date.unique())))
    verify_frozen_files({'file_sha256':hashes,'input':{}})
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:12]
    parent=root/'results/experiments';destination=parent/('m4_bootstrap_p2_p3_'+tag)
    if destination.exists():raise FileExistsError('Bootstrap destination already exists; never overwrite')
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=parent))
    frozen=dict(result_label=LABEL,design=json_snapshot(DESIGN),source_cost_bundle=str(source),
                file_sha256=hashes,source_forecast_provenance=source_manifest['source_forecast_provenance'],
                F1_status='exploratory/post-pilot',F2_F4_status='pending prior exposure verification',
                packages={p:version(p) for p in ['numpy','pandas']})
    finalized=False
    try:
        _json(stage/'STARTED.json',frozen)  # Declaration is persisted before any resampling.
        intervals,replicates,indices,layout=bootstrap_costs(daily,DESIGN['block_lengths'],DESIGN['resamples'],DESIGN['seed'],DESIGN['confidence_level'])
        for name,frame in [('daily_costs.csv',daily),('intervals.csv',intervals),('replicates.csv',replicates),('date_layout.csv',layout)]:
            with (stage/name).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
            loaded=pd.read_csv(stage/name,float_precision='round_trip',parse_dates=['date'] if name=='daily_costs.csv' else None)
            pd.testing.assert_frame_equal(loaded,frame,check_dtype=False)
        with (stage/'resampling_indices.npz').open('xb') as f:np.savez_compressed(f,**indices)
        with np.load(stage/'resampling_indices.npz',allow_pickle=False) as archive:
            if set(archive.files)!=set(indices) or any(not np.array_equal(archive[n],idx) for n,idx in indices.items()):
                raise ValueError('Index archive round-trip mismatch')
        verify_frozen_files({'file_sha256':hashes,'input':{}});verify_frozen_files(source_manifest['source_forecast_provenance'])
        output_hashes={p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}
        _json(stage/'manifest.json',dict(frozen,wall_seconds=perf_counter()-started,interval_rows=len(intervals),replicate_rows=len(replicates),
             undefined_relative_replicates=int(intervals.loc[intervals.metric.eq('relative_cost_delta_pct'),'undefined_resamples'].sum()),output_sha256=output_hashes))
        _json(stage/'COMPLETE.json',{'files':{**output_hashes,'manifest.json':file_hash(stage/'manifest.json')}})
        if inspect_bootstrap(stage)!='complete':raise ValueError('Staged bootstrap integrity failure')
        if destination.exists():raise FileExistsError('Bootstrap destination already exists')
        stage.rename(destination)
        finalized=True
        if inspect_bootstrap(destination)!='complete':raise ValueError('Bootstrap bundle integrity failure')
        return destination
    except Exception as exc:
        failure_dir=destination if finalized else stage
        try:_json(failure_dir/'FAILED.json',dict(exception=type(exc).__name__,message=str(exc)))
        except OSError:pass
        raise


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(run_bootstrap())


if __name__=='__main__':main()
