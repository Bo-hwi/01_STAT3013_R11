"""Locked F1/M0 real-data pilot. Execution trains models; --help does not.

Run from the repository root: python -B -m scripts.m4_pilot
PILOT / NOT FINAL RESEARCH RESULTS. Canonical M1 data are never replaced.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

from src.decision import quantile_pipeline as qp
from src.decision.quantile_artifacts import inspect_run
from src.decision.quantile_contracts import file_hash, verify_frozen_files
from src.decision.quantile_preflight import production_preflight, ProductionPreflightError
from src.forecasting.rolling_validation import read_yaml


ROOT = Path(__file__).resolve().parents[1]
FEATURE_RELATIVE = Path('data/processed/m4_feature_candidates/candidate-t77rmmvr/model_features.parquet')
EXPECTED_FEATURE_SHA256 = 'e4186aeb3a2da9996dcb9deec164cc70ca11c0a869b932720c9786c64a3ab17d'
# Independent pilot-only constant: never mutate shared DEFAULT_GRID/configs.
PILOT_PARAMETERS = dict(max_iter=100, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.05)


def verify_candidate_bundle(feature_path):
    """Bind the approved candidate, four evidence files, report and marker."""
    feature_path = Path(feature_path)
    stage = feature_path.parent
    if (stage / 'FAILED.json').exists():
        raise ValueError('Candidate staging bundle contains FAILED.json')
    report_path, marker_path = stage / 'reconciliation_report.json', stage / 'CANDIDATE_COMPLETE.json'
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    if marker.get('report_sha256') != file_hash(report_path) or marker.get('adopted') is not False:
        raise ValueError('Candidate completion marker is inconsistent')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('recommendation') != 'GO_FOR_ADOPTION_REVIEW'
            or report.get('preflight_status') != 'PASS'
            or report.get('original_inputs_unchanged') is not True
            or report.get('code_configs_unchanged') is not True):
        raise ValueError('Candidate reconciliation evidence is not approved for review')
    required = {'model_features.parquet', 'model_features.meta.json',
                'curated_sales_weather.parquet', 'curated_sales_weather.meta.json'}
    outputs = report.get('output_sha256', {})
    if set(outputs) != required:
        raise ValueError('Candidate report must bind exactly the four processed evidence files')
    hashes = {str(p.resolve()): file_hash(p) for p in [report_path, marker_path]}
    for name in required:
        path = stage / name
        digest = file_hash(path)
        if outputs[name] != digest:
            raise ValueError(f'Candidate evidence hash mismatch: {name}')
        hashes[str(path.resolve())] = digest
    if hashes[str(feature_path.resolve())] != EXPECTED_FEATURE_SHA256:
        raise ValueError('Feature digest differs from the verified 34-column candidate')
    return hashes


def run_pilot(experiment_id, root=ROOT):
    """Orchestrate provenance/preflight, then delegate all modelling and saving."""
    root = Path(root)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', experiment_id):
        raise ValueError('Unsafe experiment identifier')
    if inspect_run(root, experiment_id)['status'] != 'absent':
        raise FileExistsError('Experiment exists or is incomplete; use a new ID')
    features = root / FEATURE_RELATIVE
    evidence = verify_candidate_bundle(features)
    protocol = read_yaml(root / 'configs/research_protocol.yaml')
    ablation = read_yaml(root / 'configs/ablation_M0_M5.yaml')
    data, preflight = production_preflight(features, protocol, ablation, root)
    if preflight['hard_failures']:
        raise ProductionPreflightError(preflight)
    # Keep original digests: do not accept changed bytes by recomputing a baseline.
    verify_frozen_files({'file_sha256': {}, 'input': {'file_sha256': evidence}})
    context = {'mode': 'pilot', **preflight,
               'file_sha256': {**preflight['file_sha256'], **evidence},
               'pilot_configuration': {'pilot_only': True, 'label': qp.PILOT_LABEL,
                                       'expected_model_fits': 6, 'parameters': dict(PILOT_PARAMETERS)}}
    result = qp.run_direct_quantiles(data, protocol, ablation, experiment_id,
                                    parameter_grid=[dict(PILOT_PARAMETERS)], input_context=context,
                                    fold_ids=['F1'], configs=['M0'])
    destination = qp.save_run(*result, root=root)
    if inspect_run(root, experiment_id)['status'] != 'complete':
        raise ValueError('Saved pilot bundle failed integrity verification')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', default=None)
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id or (
        'pilot_f1_m0_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:12])
    destination = run_pilot(experiment_id)
    print(f'{qp.PILOT_LABEL}\nSaved: {destination}')


if __name__ == '__main__':
    main()
