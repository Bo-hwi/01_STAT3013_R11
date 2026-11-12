"""Approval-gated fixed-week uncertainty supplement; consumes stored costs only."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from . import m4_handoff as handoff
from . import p4_reproduction as points
from .quantile_contracts import file_hash, json_snapshot, semantic_hash

ROOT = handoff.ROOT
STATUS = 'PROPOSED_NOT_APPROVED'
POLICIES = ('P1', 'P2', 'P3', 'P4_M0', 'P4_M3')
OBS = ['fold', 'date', 'store_nbr', 'family', 'city']
CONTRASTS = (('P3', 'P2'), ('P4_M3', 'P4_M0'), ('P2', 'P1'),
             ('P3', 'P1'), ('P4_M0', 'P1'), ('P4_M3', 'P1'))
VARIANTS = ('corrected', 'raw_clipped_sensitivity')
SPEC = dict(label='M4_FIXED_WEEK_ROBUSTNESS_COMPLIANCE_SUPPLEMENT', seed=42,
    replicates=1000, confidence=.95, interval='percentile_linear',
    folds=['F1', 'F2', 'F3', 'F4'], scenarios=['S1', 'S3', 'S5'],
    block='floor((date - shared fold test_start).days / 7); four fixed calendar blocks',
    draws='default_rng(42).integers(0,4,size=(1000,4,4)); shared across all comparisons',
    contrasts=[list(p) for p in CONTRASTS], variants=list(VARIANTS),
    raw_sensitivity='ONLY P2/P3 change; P1 and clipped-residual P4 remain fixed',
    pooling='sum sampled costs and counts within each fold, then across fixed folds',
    cell_primary='all 112 scored dates, all four complete folds; 140 eligible cells',
    incomplete_cells='retain point estimates, CI NA; no imputation or partial-week reindexing',
    denominator='P1 resampled jointly; zero denominator -> NA, report valid/undefined counts',
    metrics=['six mean-cost differences', 'six P1-normalized differences',
             'six relative differences versus comparison policy', 'four P1 policy indices'],
    scopes=['pooled_F1_F4', 'F1', 'F2', 'F3', 'F4', 'family_city_pooled'],
    limitations=['only four weekly blocks per fold; no guaranteed nominal coverage',
        'fixed policies; excludes model-selection, calibration and refit uncertainty',
        'F1 exploratory/post-pilot; F2-F4 prior exposure verification pending',
        'no multiple-testing control, causal or operational guarantee',
        'M4_REPRODUCED_M3 is not official M3 or exact numerical reproduction',
        'cell/cost estimands extend M3 pooled MAE design; not preregistered before tests',
        'observed sales is a censored demand proxy; simulated costs are not money/profit/ROI',
        'old moving-block results remain separate; no relabeling or recommendation changes'])


def fold_calendar(folds):
    if [f['fold_id'] for f in folds] != SPEC['folds']:
        raise ValueError('Require ordered shared F1-F4, not a new splitter')
    calendar = {}
    for f in folds:
        start, end = map(pd.Timestamp, f['test'])
        days = pd.date_range(start, end)
        if len(days) != 28 or start != start.normalize() or end != end.normalize():
            raise ValueError('Each shared test fold must contain 28 calendar days')
        calendar[f['fold_id']] = days
    if len(set().union(*(set(v) for v in calendar.values()))) != 112:
        raise ValueError('Shared test folds overlap')
    return calendar


def align_costs(rows, expected, folds):
    """One-to-one pairing, even when a row is absent from every policy."""
    required = OBS + ['scenario', 'variant', 'policy', 'y_true', 'simulated_cost']
    if rows.empty or set(required)-set(rows) or rows[required].isna().any().any():
        raise ValueError('Missing/null stored-cost fields')
    if set(rows.policy) != set(POLICIES) or set(rows.scenario) != set(SPEC['scenarios']):
        raise ValueError('Unexpected policy/scenario scope')
    if set(rows.variant) != set(VARIANTS):
        raise ValueError('Require separate corrected and direct-only raw sensitivity')
    calendar = fold_calendar(folds)
    expected = expected.copy()
    if 'split' not in expected: expected['split'] = 'test'
    if not expected.split.eq('test').all(): raise ValueError('Expected keys must be test-only')
    expected = points.comparison_copy(expected)
    for f, g in expected.groupby('fold'):
        if f not in calendar or not pd.to_datetime(g.date).isin(calendar[f]).all():
            raise ValueError('Expected dates outside shared fold')
        if set(pd.to_datetime(g.date)) != set(calendar[f]):
            raise ValueError('Global fold calendar incomplete')
    parts = []
    for scenario in SPEC['scenarios']:
        for variant in VARIANTS:
            combined = None
            for policy in POLICIES:
                part = rows.loc[rows.scenario.eq(scenario) & rows.variant.eq(variant)
                                & rows.policy.eq(policy), required].copy()
                part['split'] = 'test'
                part = points.comparison_copy(part)
                part['date'] = pd.to_datetime(part.date, errors='raise')
                if part.date.dt.tz is not None or not part.date.eq(part.date.dt.normalize()).all():
                    raise ValueError('Require timezone-naive calendar dates')
                vals = part[['y_true', 'simulated_cost']].to_numpy(float)
                if not np.isfinite(vals).all() or (vals < 0).any():
                    raise ValueError('Stored demand/cost must be finite nonnegative')
                points.assert_matched_rows(expected, part)
                # Exact y_true already matched against frozen expected rows.
                part = part[OBS + ['y_true', 'simulated_cost']].rename(columns={'simulated_cost': policy})
                if combined is None: combined = part
                else: combined = combined.merge(part.drop(columns='y_true'), on=OBS, validate='1:1')
            parts.append(combined.assign(scenario=scenario, variant=variant))
    out = pd.concat(parts, ignore_index=True)
    # Fixed P1/P4 policies must NOT change with the direct-only raw sensitivity.
    a = out.loc[out.variant.eq(VARIANTS[0])].set_index(OBS + ['scenario']).sort_index()
    b = out.loc[out.variant.eq(VARIANTS[1])].set_index(OBS + ['scenario']).sort_index()
    if not a[['P1', 'P4_M0', 'P4_M3']].equals(b[['P1', 'P4_M0', 'P4_M3']]):
        raise ValueError('Sensitivity unexpectedly changes fixed P1/P4 costs')
    return out


def calendar_support(panel, folds):
    calendar = fold_calendar(folds)
    base = panel.loc[panel.scenario.eq('S1') & panel.variant.eq('corrected')]
    records = []
    for (family, city), g in base.groupby(['family', 'city'], sort=True):
        counts = {f: g.loc[g.fold.eq(f), 'date'].nunique() for f in calendar}
        complete = all(set(g.loc[g.fold.eq(f), 'date']) == set(dates) for f, dates in calendar.items())
        records.append(dict(family=family, city=city, sample_size=len(g), n_dates=g.date.nunique(),
                            fold_dates=json.dumps(counts, sort_keys=True), eligible=complete,
                            calendar_flag='COMPLETE_112_DATES' if complete else 'INCOMPLETE_CALENDAR_CI_UNAVAILABLE'))
    return pd.DataFrame(records)


def weekly_totals(panel, folds):
    calendar = fold_calendar(folds)
    x = panel.copy()
    x['block'] = [(d-calendar[f][0]).days//7 for f, d in zip(x.fold, x.date)]
    if not x.block.between(0, 3).all(): raise ValueError('Dates outside fixed blocks')
    grouping = ['scenario', 'variant', 'family', 'city', 'fold', 'block']
    out = x.groupby(grouping, observed=True, sort=True).agg(
        **{p: (p, 'sum') for p in POLICIES}, n=('date', 'size'), n_dates=('date', 'nunique')).reset_index()
    if not np.isfinite(out[list(POLICIES)].to_numpy()).all(): raise ValueError('Weekly totals overflow')
    return out


def fixed_draws():
    """Exact pinned M3 RNG call; never invoked by prepare-request."""
    return np.random.default_rng(42).integers(0, 4, size=(1000, 4, 4))


def ratios(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    out = np.full(a.shape, np.nan)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        np.divide(a, b, out=out, where=b > 0)
    if np.isinf(out).any(): raise ValueError('Ratio outside representable range')
    return out


def statistics(costs, counts):
    costs = np.asarray(costs, float); counts = np.asarray(counts, float)
    if costs.shape[-1] != 5 or costs.shape[:-1] != counts.shape or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Invalid sampled totals')
    if not np.isfinite(counts).all() or (counts < 0).any(): raise ValueError('Invalid sampled counts')
    values = {}
    for weather, base in CONTRASTS:
        i, j = POLICIES.index(weather), POLICIES.index(base)
        difference = costs[..., i] - costs[..., j]
        name = weather + '_minus_' + base
        values[name + ':mean_cost_difference'] = (ratios(difference, counts), 0.)
        values[name + ':P1_normalized_difference'] = (ratios(difference, costs[..., 0]), 0.)
        values[name + ':relative_difference_pct'] = (100*ratios(difference, costs[..., j]), 0.)
    for i, p in enumerate(POLICIES[1:], 1):
        values[p + ':P1_cost_index'] = (ratios(costs[..., i], costs[..., 0]), 1.)
    if any(np.isinf(value).any() for value, _ in values.values()): raise ValueError('Statistic overflow')
    return values


def interval_summary(draws, point, null=0., eligible=True):
    values = np.asarray(draws, float)
    finite = values[np.isfinite(values)] if eligible else np.array([])
    lower, upper = np.quantile(finite, [.025, .975], method='linear') if len(finite) else (np.nan, np.nan)
    direction = np.sign(point-null) if np.isfinite(point) else np.nan
    return dict(point_estimate=float(point), lower=float(lower), upper=float(upper), null_reference=null,
        includes_null=bool(lower <= null <= upper) if len(finite) else None,
        replicates_requested=1000, resamples_attempted=len(values) if eligible else 0,
        valid_replicates=len(finite), undefined_replicates=len(values)-len(finite) if eligible else 0,
        below_null_fraction=float(np.mean(finite < null)) if len(finite) else np.nan,
        above_null_fraction=float(np.mean(finite > null)) if len(finite) else np.nan,
        sign_change_fraction=float(np.mean((finite-null)*direction < 0)) if len(finite) and direction in (-1, 1) else np.nan,
        ci_status='AVAILABLE' if len(finite) else ('UNDEFINED_DENOMINATOR' if eligible else 'INCOMPLETE_CALENDAR_CI_UNAVAILABLE'))


def bootstrap_supplement(panel, folds):
    """Pure synthetic-testable core. Production caller MUST validate approval first."""
    required = OBS + ['scenario', 'variant', 'y_true'] + list(POLICIES)
    if panel.empty or set(required)-set(panel) or panel[required].isna().any().any():
        raise ValueError('Missing/null paired panel fields')
    if panel.duplicated(OBS+['scenario','variant']).any(): raise ValueError('Duplicate paired observations')
    if set(panel.scenario) != set(SPEC['scenarios']) or set(panel.variant) != set(VARIANTS):
        raise ValueError('Unexpected paired scenario/variant scope')
    numbers = panel[list(POLICIES)+['y_true']].to_numpy(float)
    if not np.isfinite(numbers).all() or (numbers < 0).any(): raise ValueError('Invalid paired costs/demand')
    reference = panel.loc[panel.scenario.eq('S1') & panel.variant.eq('corrected'), OBS+['y_true']].sort_values(OBS).reset_index(drop=True)
    for _, g in panel.groupby(['scenario','variant']):
        if not g[OBS+['y_true']].sort_values(OBS).reset_index(drop=True).equals(reference):
            raise ValueError('Paired series keys/targets differ')
    for f, dates in fold_calendar(folds).items():
        if set(reference.loc[reference.fold.eq(f),'date']) != set(dates):
            raise ValueError('Global fold calendar incomplete')
    support = calendar_support(panel, folds)
    weekly = weekly_totals(panel, folds)
    draws = fixed_draws()
    intervals, point_rows = [], []
    scopes = [('pooled_F1_F4', '', '', True)] + [(f, '', '', True) for f in SPEC['folds']]
    scopes += [('family_city_pooled', r.family, r.city, bool(r.eligible)) for r in support.itertuples()]
    for scenario in SPEC['scenarios']:
        for variant in VARIANTS:
            part = weekly.loc[weekly.scenario.eq(scenario) & weekly.variant.eq(variant)]
            for scope, family, city, eligible in scopes:
                g = part if not family else part.loc[part.family.eq(family) & part.city.eq(city)]
                if scope in SPEC['folds']: g = g.loc[g.fold.eq(scope)]
                point_cost = g[list(POLICIES)].sum().to_numpy(float); point_n = int(g.n.sum())
                context = dict(scope=scope, family=family, city=city, scenario=scenario, variant=variant,
                               n_observations=point_n, ci_eligible=eligible)
                point_rows.append(dict(context, **{p+'_total': float(point_cost[i]) for i, p in enumerate(POLICIES)}))
                pc = statistics(point_cost, np.asarray(point_n))
                samples = None
                if eligible:
                    sums = np.zeros((1000, 5)); counts = np.zeros(1000)
                    selected_folds = [scope] if scope in SPEC['folds'] else SPEC['folds']
                    for fold in selected_folds:
                        blocks = g.loc[g.fold.eq(fold)].groupby('block')[list(POLICIES)+['n']].sum().sort_index()
                        if list(blocks.index) != [0, 1, 2, 3]: raise ValueError('Eligible scope lacks four fixed weekly blocks')
                        idx = draws[:, SPEC['folds'].index(fold), :]
                        sums += blocks[list(POLICIES)].to_numpy()[idx].sum(axis=1)
                        counts += blocks.n.to_numpy()[idx].sum(axis=1)
                    samples = statistics(sums, counts)
                for metric, (point, null) in pc.items():
                    vals = samples[metric][0] if samples is not None else np.array([])
                    summary = interval_summary(vals, point, null, eligible)
                    intervals.append(dict(context, metric=metric, confidence_level=.95,
                        denominator=('sampled_observation_count' if metric.endswith(':mean_cost_difference') else
                                     'sampled_comparison_policy_cost' if metric.endswith(':relative_difference_pct') else 'sampled_P1_cost'),
                        zero_denominator_replicates=summary['undefined_replicates'], **summary))
    return dict(intervals=pd.DataFrame(intervals), point_totals=pd.DataFrame(point_rows),
                calendar_support=support, weekly_totals=weekly, draws=draws)


def load_verified_costs(root=ROOT):
    """No calibration, cost evaluation or bootstrap; verify and read historical outputs."""
    root = Path(root).resolve(); verified = handoff.verify_sources(root)
    expected = pd.read_csv(verified['dirs']['point_reproduction']/'expected_rows.csv', parse_dates=['date'], dtype={'store_nbr': str}, float_precision='round_trip')
    expected = expected.loc[expected.split.eq('test')].copy()
    protocol = points.read_yaml(root/'configs/research_protocol.yaml'); folds = protocol['rq2']['validation']['test_folds']
    config = points.read_yaml(root/'configs/cost_scenarios.yaml')
    scenarios = handoff.direct.configured_scenarios(config)
    frames = []
    for name, filename in [('p1','p1_rows.csv'), ('p2_p3','row_costs.csv'), ('p4','p4_rows.csv')]:
        x = pd.read_csv(verified['dirs'][name]/filename, parse_dates=['date'], dtype={'store_nbr': str}, float_precision='round_trip')
        for s in scenarios:
            part = x.loc[x.scenario.eq(s['scenario'])]
            if not part.tau_key.eq(s['tau_key']).all(): raise ValueError('Stored cost tau/scenario mismatch')
            if not part.c_over.eq(s['c_over']).all() or not part.c_under.eq(s['c_under']).all(): raise ValueError('Stored cost parameters differ from shared config')
        if 'split' in x and not x.split.eq('test').all(): raise ValueError('Non-test source costs')
        if name == 'p4': x['policy'] = 'P4_' + x.config
        if name == 'p2_p3': frames.append(x[OBS+['scenario','variant','policy','y_true','simulated_cost']])
        else:
            for variant in VARIANTS: frames.append(x.assign(variant=variant)[OBS+['scenario','variant','policy','y_true','simulated_cost']])
    panel = align_costs(pd.concat(frames, ignore_index=True), expected, folds)
    support = calendar_support(panel, folds)
    if len(expected) != 41076 or len(support) != 154 or int(support.eligible.sum()) != 140:
        raise ValueError('Frozen population/calendar eligibility drift')
    excluded = support.loc[~support.eligible]
    if len(excluded) != 14 or set(excluded.city) != {'Esmeraldas', 'Salinas'}:
        raise ValueError('Unexpected incomplete-calendar cells')
    return panel, folds, support, verified


def environment():
    return dict(python=platform.python_version(), packages={p: version(p) for p in ['numpy','pandas','scipy','scikit-learn']})


def make_request(root=ROOT):
    root = Path(root).resolve()
    branch = current_branch(root)
    if branch != 'feature/m4-quantile-decision': raise ValueError('Require feature/m4-quantile-decision')
    panel, folds, support, verified = load_verified_costs(root)
    # Hash-check the exact pinned bytes, without executing notebook code.
    source = points.reference_bytes(points.NOTEBOOK, root)
    files = dict(verified['evidence'])
    for relative in ['src/decision/fixed_week_bootstrap.py','tests/test_m4_fixed_week_bootstrap.py','docs/m4_phase4e_fixed_week.md',
                     'configs/cost_scenarios.yaml','configs/research_protocol.yaml']:
        f = root/relative; files[str(f)] = file_hash(f)
    return json_snapshot(dict(status=STATUS, branch=branch, specification=SPEC, environment=environment(),
        m3_reference=dict(commit=points.COMMIT, path=points.NOTEBOOK, sha256=points.sha256(source).hexdigest(), cell=17),
        file_sha256=files, folds=folds, test_observations=41076,
        aligned_rows=len(panel), calendar_support=support.to_dict(orient='records'),
        scored_keys_sha256=semantic_hash(panel.loc[panel.scenario.eq('S1') & panel.variant.eq('corrected'), OBS+['y_true']]
            .sort_values(OBS).assign(date=lambda x: x.date.dt.strftime('%Y-%m-%d')).to_dict(orient='records')),
        execution_authorized=False))


def current_branch(root):
    return subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()


def validate_approval(receipt, root=ROOT):
    request = receipt.get('request', {})
    if receipt.get('status') != 'APPROVED' or request.get('status') != STATUS:
        raise ValueError('Separate fixed-week supplement approval required; PENDING_APPROVAL')
    if receipt.get('request_sha256') != semantic_hash(request) or request.get('specification') != SPEC:
        raise ValueError('Approval digest/specification mismatch')
    for k in ['approved_by','approval_evidence','approval_communicated_date']:
        if not isinstance(receipt.get(k), str) or not receipt[k].strip(): raise ValueError('Missing approval evidence: '+k)
    datetime.strptime(receipt['approval_communicated_date'], '%Y-%m-%d')
    if not request.get('file_sha256'): raise ValueError('Empty frozen file evidence')
    for path, digest in request['file_sha256'].items():
        if file_hash(path) != digest: raise ValueError('Frozen file drift: '+path)
    current = make_request(root)
    if semantic_hash(current) != semantic_hash(request): raise ValueError('Current source/environment/request drift')
    return request


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)


def inspect_bundle(directory):
    directory = Path(directory).resolve()
    if (directory/'FAILED.json').exists(): return 'failed'
    try:
        marker = json.loads((directory/'COMPLETE.json').read_text())
        required = {'manifest.json','STARTED.json','intervals.csv','point_totals.csv','calendar_support.csv','weekly_totals.csv','draws.npz'}
        if set(marker['files']) != required: return 'corrupt'
        for name, digest in marker['files'].items():
            path = (directory/name).resolve()
            if not path.is_relative_to(directory) or file_hash(path) != digest: return 'corrupt'
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['output_sha256'] != {k:v for k,v in marker['files'].items() if k != 'manifest.json'}: return 'corrupt'
        return 'complete'
    except (OSError, ValueError, KeyError, TypeError): return 'incomplete'


def save_bundle(result, receipt, request, destination, elapsed):
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError('Never overwrite previous experiments')
    stage = Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-', dir=destination.parent))
    try:
        write_json(stage/'STARTED.json', dict(approval=receipt, request_sha256=semantic_hash(request)))
        for name in ['intervals','point_totals','calendar_support','weekly_totals']:
            with (stage/(name+'.csv')).open('x', encoding='utf-8', newline='') as f: result[name].to_csv(f,index=False)
        with (stage/'draws.npz').open('xb') as f: np.savez_compressed(f, draws=result['draws'])
        with np.load(stage/'draws.npz', allow_pickle=False) as archive:
            if not np.array_equal(archive['draws'], result['draws']): raise ValueError('Draw serialization mismatch')
        reloaded = pd.read_csv(stage/'intervals.csv', float_precision='round_trip', keep_default_na=False)
        expected = result['intervals'].copy()
        # Empty cell labels are valid scope labels; missing CI values are not labels.
        for column in expected:
            if column == 'includes_null':
                reloaded[column] = reloaded[column].map({True:True,False:False,'True':True,'False':False,'':pd.NA}).astype('boolean')
                expected[column] = expected[column].astype('boolean')
            elif pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(expected[column]):
                reloaded[column] = reloaded[column].map(lambda v: np.nan if v == '' else v).astype(float)
        pd.testing.assert_frame_equal(reloaded, expected, check_dtype=False)
        outputs = {p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}
        write_json(stage/'manifest.json', dict(label=SPEC['label'], approval=receipt, specification=SPEC,
            file_sha256=request['file_sha256'], environment=environment(), elapsed_seconds=elapsed,
            previous_moving_block_unchanged=True, output_sha256=outputs))
        write_json(stage/'COMPLETE.json', dict(files={**outputs,'manifest.json':file_hash(stage/'manifest.json')}))
        if inspect_bundle(stage) != 'complete': raise ValueError('Staged supplement incomplete/corrupt')
        if destination.exists(): raise FileExistsError('Destination appeared; never overwrite')
        stage.rename(destination)
        return destination
    except Exception as e:
        write_json(stage/'FAILED.json', dict(type=type(e).__name__, message=str(e)))
        raise


def run_approved(approval, root=ROOT):
    root = Path(root).resolve()
    receipt = json.loads(Path(approval).read_text(encoding='utf-8'))
    request = validate_approval(receipt, root)  # Gate before ANY random draw.
    panel, folds, _, _ = load_verified_costs(root)
    parent = root/'results/experiments'
    destination = parent/('m4_fixed_week_supplement_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    # Persist frozen authorization before resampling. Failure evidence is retained.
    start_record = parent/('.'+destination.name+'.authorized.json')
    write_json(start_record, dict(approval=receipt, request_sha256=semantic_hash(request), status='AUTHORIZED_NOT_COMPLETE'))
    started = perf_counter()
    try:
        result = bootstrap_supplement(panel, folds)
        for path, digest in request['file_sha256'].items():
            if file_hash(path) != digest: raise ValueError('Source drift during bootstrap: '+path)
        return save_bundle(result, receipt, request, destination, perf_counter()-started)
    except Exception as e:
        write_json(parent/('.'+destination.name+'.FAILED.json'), dict(type=type(e).__name__, message=str(e)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-request', type=Path)
    group.add_argument('--approval', type=Path)
    args = parser.parse_args()
    if args.prepare_request:
        if args.prepare_request.exists(): raise FileExistsError('Never overwrite an approval request')
        request = make_request(); write_json(args.prepare_request, request)
        print('PROPOSED_NOT_APPROVED semantic SHA-256:', semantic_hash(request))
    else: print(run_approved(args.approval))


if __name__ == '__main__': main()
