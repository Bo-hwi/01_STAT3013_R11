"""P2/P3 simulated economic evaluation; no training, calibration or P1/P4."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from src.forecasting.rolling_validation import read_yaml
from .cost_simulation import newsvendor_cost, cost_index
from .quantile_models import canonical_tau_key, normalize_tau, nonnegative_quantity, pinball_loss, _real_scalar
from .quantile_contracts import TAU_KEYS, file_hash, json_snapshot, stable_mean, validate_prediction_panel, verify_frozen_files
from .quantile_artifacts import inspect_run, _validate_saved_panel
from .quantile_pipeline import summarize_predictions


ROOT = Path(__file__).resolve().parents[2]
LABEL = 'SIMULATED ECONOMIC OUTCOMES; not actual inventory costs, profit or ROI'
MATCH = ['date', 'store_nbr', 'family', 'city', 'fold', 'tau_key']
VARIANTS = {'corrected': 'corrected_prediction', 'raw_clipped_sensitivity': 'raw_prediction'}
LEVELS = {'pooled': [], 'fold': ['fold'], 'family': ['family'], 'city': ['city'],
          'family_city': ['family', 'city'], 'fold_family': ['fold', 'family'],
          'fold_city': ['fold', 'city'], 'fold_family_city': ['fold', 'family', 'city']}


def configured_scenarios(config):
    """Read costs from YAML; reject duplicate/inconsistent canonical tau mapping."""
    if (config['decision']['demand_proxy'] != 'observed_sales'
            or config['decision']['nonnegative_quantity'] is not True
            or config['cost_function']['units'] != 'relative'):
        raise ValueError('Require the existing observed-sales, nonnegative, relative-cost contract')
    scenarios = []
    for item in config['scenarios']:
        co, cu = (_real_scalar(item[k], k) for k in ['c_over', 'c_under'])
        if co <= 0 or cu <= 0:
            raise ValueError('Scenario costs must be positive finite scalars')
        scale = max(co, cu)
        critical = (cu / scale) / (co / scale + cu / scale)
        tau = normalize_tau(item['tau'])
        if canonical_tau_key(critical) != canonical_tau_key(tau):
            raise ValueError('Scenario tau disagrees with cost critical ratio')
        scenario_id = item['id']
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError('Scenario IDs must be nonempty strings')
        scenarios.append(dict(scenario=scenario_id, c_over=co, c_under=cu,
                              tau=tau, tau_key=canonical_tau_key(tau)))
    if (len(scenarios) != 3 or len({s['scenario'] for s in scenarios}) != 3
            or {s['tau_key'] for s in scenarios} != set(TAU_KEYS)):
        raise ValueError('Require unique scenarios covering exactly the three canonical taus')
    if ([canonical_tau_key(t) for t in config['quantile_targets']['taus']] != list(TAU_KEYS)
            or config['quantile_targets']['feature_sets'] != ['M0', 'M3']):
        raise ValueError('Quantile targets differ from P2/M0 and P3/M3 contract')
    return scenarios


def _sum(values):
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('Simulated aggregate exceeds finite float64 range') from exc
    if not math.isfinite(total):
        raise ValueError('Simulated aggregate must be finite')
    return total


def evaluate_policies(panel, config, evaluation_id, expected_rows=None):
    """Pure test-only paired evaluation; synthetic panels need no production data."""
    scenarios = configured_scenarios(config)
    validate_prediction_panel(panel, expected_rows=expected_rows, expected_configs=['M0', 'M3'])
    if panel.experiment_id.nunique() != 1 or panel.model.nunique() != 1:
        raise ValueError('Evaluate one source experiment/model at a time')
    test = panel.loc[panel.split.eq('test')]
    if test.empty:
        raise ValueError('Economic evaluation requires scored test predictions')
    columns = MATCH + ['tau', 'y_true', 'raw_prediction', 'corrected_prediction']
    paired = test.loc[test.config.eq('M0'), columns].merge(
        test.loc[test.config.eq('M3'), columns], on=MATCH, how='outer',
        suffixes=('_P2', '_P3'), validate='1:1', indicator=True)
    if not paired['_merge'].eq('both').all() or not paired.y_true_P2.equals(paired.y_true_P3):
        raise ValueError('P2/P3 require exact matched keys and y_true')
    output, pairs = [], []
    for scenario in scenarios:
        g = paired.loc[paired.tau_key.eq(scenario['tau_key'])]
        for variant, source_column in VARIANTS.items():
            pair = g[MATCH].copy()
            pair['scenario'], pair['variant'] = scenario['scenario'], variant
            for policy in ['P2', 'P3']:
                demand = g.y_true_P2.to_numpy(float)
                original_q = g[source_column + '_' + policy].to_numpy(float)
                q = nonnegative_quantity(original_q)
                cost = newsvendor_cost(demand, q, scenario['c_over'], scenario['c_under'])
                loss = pinball_loss(demand, q, scenario['tau'])
                # Avoid overflowing c_over+c_under when scaled costs remain representable.
                scale = max(scenario['c_over'], scenario['c_under'])
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    identity = loss * (scenario['c_over']/scale + scenario['c_under']/scale) * scale
                if not np.isfinite(identity).all() or not np.allclose(cost, identity, rtol=1e-12, atol=0):
                    raise ValueError('Per-row cost-pinball identity failed at feasible Q')
                over, under = np.maximum(q-demand, 0), np.maximum(demand-q, 0)
                rows = g[MATCH].copy()
                rows['result_label'], rows['evaluation_id'] = LABEL, evaluation_id
                rows['source_experiment_id'] = panel.experiment_id.iloc[0]
                rows['scenario'], rows['variant'], rows['policy'] = scenario['scenario'], variant, policy
                rows['config'] = 'M0' if policy == 'P2' else 'M3'
                rows['tau'], rows['c_over'], rows['c_under'] = scenario['tau'], scenario['c_over'], scenario['c_under']
                rows['y_true'], rows['source_prediction'], rows['quantity'] = demand, original_q, q
                rows['clipped_at_zero'] = original_q < 0
                rows['simulated_cost'], rows['overstock_units'], rows['understock_units'] = cost, over, under
                rows['pinball_at_feasible_quantity'] = loss
                rows['interpretation'] = np.where(rows.fold.eq('F1'), 'exploratory/post-pilot',
                                                  'confirmatory status pending exposure verification')
                output.append(rows)
                pair['cost_' + policy], pair['over_' + policy], pair['under_' + policy] = cost, over, under
            pairs.append(pair)
    row_costs = pd.concat(output, ignore_index=True)
    paired_costs = pd.concat(pairs, ignore_index=True)
    records = []
    for level, grouping in LEVELS.items():
        names = ['scenario', 'variant'] + grouping
        for key, g in paired_costs.groupby(names, observed=True, sort=True):
            record = dict(zip(names, key))
            mean2, mean3 = stable_mean(g.cost_P2.to_numpy()), stable_mean(g.cost_P3.to_numpy())
            ratio = cost_index(g.cost_P3.to_numpy(), g.cost_P2.to_numpy())
            relative = (ratio-1)*100
            if not np.isnan(relative) and not np.isfinite(relative):
                raise ValueError('Relative cost delta exceeds finite float64 range')
            record.update(result_label=LABEL, evaluation_id=evaluation_id, report_level=level,
                          sample_size=len(g), mean_cost_P2=mean2, mean_cost_P3=mean3,
                          delta_mean_cost_P3_minus_P2=mean3-mean2,
                          relative_delta_pct=relative, cost_ratio_P3_over_P2=ratio,
                          p2_denominator_zero=not g.cost_P2.gt(0).any())
            for policy in ['P2', 'P3']:
                for field in ['cost', 'over', 'under']:
                    values = g[field+'_'+policy].to_numpy()
                    record['total_'+field+'_'+policy] = _sum(values)
                    if field != 'cost':
                        record['mean_'+field+'_'+policy] = stable_mean(values)
            records.append(record)
    return row_costs, pd.DataFrame(records)


def load_verified_source(root, experiment_id):
    """Full-panel, input/code and CSV metric verification before cost arithmetic."""
    root = Path(root)
    if inspect_run(root, experiment_id)['status'] != 'complete':
        raise ValueError('Source quantile bundle is absent, incomplete or corrupt')
    directory = root / 'results/experiments' / ('m4_quantiles_'+experiment_id)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('pilot_only') is not False or manifest['selected_scope'] != {
            'fold_ids': ['F1','F2','F3','F4'], 'configs': ['M0','M3']}:
        raise ValueError('Phase 3A requires the completed full final quantile run')
    verify_frozen_files(manifest['provenance'])
    hashes = {str(p.resolve()): file_hash(p) for p in directory.iterdir() if p.is_file()}
    panel = pd.read_csv(directory/f'm4_quantiles_{experiment_id}.csv', parse_dates=['date','forecast_origin'],
                        dtype={'store_nbr':str,'family':str,'city':str}, float_precision='round_trip')
    _validate_saved_panel(panel, manifest)
    pd.testing.assert_frame_equal(pd.read_csv(directory/'metrics.csv', float_precision='round_trip'),
                                  summarize_predictions(panel), check_dtype=False)
    verify_frozen_files(manifest['provenance'])
    if any(file_hash(Path(p)) != h for p,h in hashes.items()):
        raise ValueError('Source bundle changed during verification')
    return panel, manifest, hashes


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def inspect_evaluation(directory):
    directory = Path(directory)
    if not directory.exists():
        return 'absent'
    if (directory/'FAILED.json').is_file():
        return 'failed'
    if not (directory/'COMPLETE.json').is_file():
        return 'incomplete'
    try:
        marker = json.loads((directory/'COMPLETE.json').read_text())
        if set(marker['files']) != {'row_costs.csv','comparisons.csv','manifest.json','STARTED.json'}:
            return 'corrupt'
        if any(Path(n).name != n or file_hash(directory/n) != h for n,h in marker['files'].items()):
            return 'corrupt'
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['output_sha256'] != {n:h for n,h in marker['files'].items() if n!='manifest.json'}:
            return 'corrupt'
    except (OSError, ValueError, KeyError, TypeError):
        return 'corrupt'
    return 'complete'


def run_evaluation(root, source_id, evaluation_id=None):
    root = Path(root).resolve()
    evaluation_id = evaluation_id or ('p2_p3_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:12])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', evaluation_id):
        raise ValueError('Unsafe evaluation identifier')
    parent = root/'results/experiments'; destination = parent/('m4_costs_'+evaluation_id)
    if destination.exists() or list(parent.glob('.'+destination.name+'.staging-*')):
        raise FileExistsError('Evaluation ID exists or is incomplete; never overwrite')
    started = perf_counter()
    panel, source, hashes = load_verified_source(root, source_id)
    config_path = root/'configs/cost_scenarios.yaml'
    hashes[str(config_path)] = file_hash(config_path)
    hashes[str(Path(__file__).resolve())] = file_hash(Path(__file__).resolve())
    doc = root/'docs/m4_phase3a_costs.md'
    hashes[str(doc)] = file_hash(doc)
    config = json_snapshot(read_yaml(config_path)); configured_scenarios(config)
    # Freeze evaluation design/evidence BEFORE the economic calculations.
    frozen = dict(evaluation_id=evaluation_id, result_label=LABEL, source_experiment_id=source_id,
                  source_manifest_sha256=file_hash(parent/('m4_quantiles_'+source_id)/'manifest.json'),
                  source_forecast_provenance=source['provenance'], file_sha256=hashes,
                  scenarios=config, variants=VARIANTS, report_levels=LEVELS,
                  primary_policies={'P2':'corrected direct quantile M0','P3':'corrected direct quantile M3'},
                  raw_sensitivity='clip raw Q at zero without sorting; preserve source_prediction',
                  unavailable=['P1/P4 evaluation','P1-normalized cost_index','bootstrap CI/significance'],
                  F1_status='exploratory/post-pilot', F2_F4_status='pending exposure verification',
                  metric_definition='Relative delta = 100*(sum(P3)/sum(P2)-1); NaN if P2 denominator is zero')
    verify_frozen_files({'file_sha256': hashes, 'input': {}})
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-', dir=parent))
    try:
        _json(stage/'STARTED.json', frozen)
        rows, comparisons = evaluate_policies(panel, config, evaluation_id)
        for name, frame in [('row_costs.csv', rows), ('comparisons.csv', comparisons)]:
            with (stage/name).open('x', encoding='utf-8', newline='') as stream:
                frame.to_csv(stream, index=False)
            loaded = pd.read_csv(stage/name, float_precision='round_trip', parse_dates=['date'] if name=='row_costs.csv' else None,
                                 dtype={'store_nbr':str,'family':str,'city':str} if name=='row_costs.csv' else None)
            pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
        verify_frozen_files(source['provenance']); verify_frozen_files({'file_sha256': hashes, 'input': {}})
        result = dict(frozen, wall_seconds=perf_counter()-started,
                      row_cost_records=len(rows), comparison_records=len(comparisons),
                      cost_pinball_identity='verified per observation at nonnegative feasible Q; rtol=1e-12',
                      zero_p2_denominator_groups=int(comparisons.p2_denominator_zero.sum()),
                      output_sha256={n:file_hash(stage/n) for n in ['row_costs.csv','comparisons.csv','STARTED.json']})
        _json(stage/'manifest.json', result)
        marker = {n:file_hash(stage/n) for n in ['row_costs.csv','comparisons.csv','manifest.json','STARTED.json']}
        _json(stage/'COMPLETE.json', {'files':marker})
        if inspect_evaluation(stage) != 'complete':
            raise ValueError('Staged economic evaluation failed integrity checks')
        if destination.exists():
            raise FileExistsError('Destination appeared during evaluation')
        stage.rename(destination)
        if inspect_evaluation(destination) != 'complete':
            raise ValueError('Final economic evaluation failed integrity checks')
        return destination
    except Exception as exc:
        failure_dir = destination if destination.exists() else stage
        try:
            _json(failure_dir/'FAILED.json', {'exception':type(exc).__name__, 'message':str(exc)})
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment', required=True)
    args = parser.parse_args(argv)
    print(run_evaluation(ROOT, args.source_experiment))


if __name__ == '__main__':
    main()
