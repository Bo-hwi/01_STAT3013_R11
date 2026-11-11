"""M4 final-run proposal: prepare sign-off, or execute ONLY with explicit approval.

No modelling logic lives here. --help and --prepare-signoff never fit models.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from uuid import uuid4

import pandas as pd

from scripts.m4_pilot import FEATURE_RELATIVE, EXPECTED_FEATURE_SHA256, verify_candidate_bundle
from src.decision import quantile_pipeline as qp
from src.decision.quantile_artifacts import inspect_run, _validate_saved_panel
from src.decision.quantile_contracts import (
    file_hash, freeze_provenance, json_snapshot, semantic_hash, verify_frozen_files,
)
from src.decision.quantile_preflight import production_preflight, ProductionPreflightError
from src.forecasting.rolling_validation import read_yaml


ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'feature/m4-quantile-decision'
FOLDS = ['F1', 'F2', 'F3', 'F4']
CONFIGS = ['M0', 'M3']
FAMILIES = ['GROCERY I', 'BEVERAGES', 'CLEANING', 'DAIRY', 'BREAD/BAKERY', 'PERSONAL CARE', 'FROZEN FOODS']
RANGES = [
    ('2016-08-03', '2016-08-04', '2016-08-31', '2016-09-01', '2016-09-28'),
    ('2016-10-17', '2016-10-18', '2016-11-14', '2016-11-15', '2016-12-12'),
    ('2017-01-31', '2017-02-01', '2017-02-28', '2017-03-01', '2017-03-28'),
    ('2017-06-20', '2017-06-21', '2017-07-18', '2017-07-19', '2017-08-15'),
]
GRID = [dict(max_iter=100, max_leaf_nodes=n, min_samples_leaf=20, learning_rate=0.05) for n in (15, 31)]
COUNTS = {
    'F1': dict(train=446544, validation=10199, test=9996, refit=456743),
    'F2': dict(train=473522, validation=9989, test=10108, refit=483644),
    'F3': dict(train=511700, validation=10388, test=10388, refit=522088),
    'F4': dict(train=564074, validation=10584, test=10584, refit=574658),
}
DECISIONS = ('estimator_grid', 'direct_quantile_calibration_refit', 'p1_p4_calibration',
             'paired_block_bootstrap', 'artifact_handoff', 'claims_and_exposure')
DEFER_ALLOWED = {'p1_p4_calibration', 'paired_block_bootstrap'}


def write_json_exclusive(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def current_snapshot(root=ROOT):
    """Read-only binding of the proposed protocol, versions, code and evidence."""
    root = Path(root).resolve()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    if branch != BRANCH:
        raise ValueError(f'Require branch {BRANCH}')
    evidence = verify_candidate_bundle(root / FEATURE_RELATIVE)
    catalog = root / 'docs/source_catalog.csv'
    evidence[str(catalog.resolve())] = file_hash(catalog)
    protocol = read_yaml(root / 'configs/research_protocol.yaml')
    ablation = read_yaml(root / 'configs/ablation_M0_M5.yaml')
    if protocol['project']['seed'] != 42 or list(qp.DEFAULT_GRID) != GRID:
        raise ValueError('Seed or original two-candidate grid changed')
    if [f['fold_id'] for f in protocol['rq2']['validation']['test_folds']] != FOLDS:
        raise ValueError('Require the complete shared F1-F4 schedule')
    for fold, dates in zip(protocol['rq2']['validation']['test_folds'], RANGES):
        if (fold['train'] != ['2013-01-01', dates[0]] or fold['validation'] != list(dates[1:3])
                or fold['test'] != list(dates[3:5])):
            raise ValueError('Shared fold boundaries differ from the sign-off protocol')
    if protocol['scope']['deep_study_families'] != FAMILIES:
        raise ValueError('Frozen family shortlist changed')
    if tuple(qp.normalize_tau(t) for t in protocol['quantile_and_decision']['quantiles']) != (.5, .75, 5/6):
        raise ValueError('Canonical quantile scope changed')
    for config in CONFIGS:
        qp.feature_columns(config, ablation)
    provenance = freeze_provenance(root, protocol, ablation)
    code = dict(provenance['file_sha256'])
    for name in ['scripts/m4_final_run.py', 'docs/m4_final_protocol.md', 'docs/m4_final_signoff.md']:
        path = root / name
        code[str(path.resolve())] = file_hash(path)
    spec = dict(protocol=protocol, ablation=ablation, scope=dict(fold_ids=FOLDS, configs=CONFIGS),
                model=qp.MODEL, quantiles=list(qp.QUANTILE_LEVELS), seed=42, grid=GRID,
                feature_path=str(FEATURE_RELATIVE), feature_sha256=EXPECTED_FEATURE_SHA256,
                selection='raw_scored_validation_pinball_first_candidate_tie_break',
                refit='fresh_eligible_train_plus_validation', calibration='none',
                correction='clip_zero_then_sort_per_row', expected_counts=COUNTS,
                expected_fits=dict(selection=48, refit=24), prediction_rows=493416,
                F1_status='exploratory_previously_inspected',
                F2_F4_confirmatory_status='pending_team_exposure_verification',
                result_label='retrospective weather-aware; F1 exploratory; F2-F4 status per sign-off',
                packages={p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'PyYAML']})
    return json_snapshot(dict(spec=spec, code_config_sha256=code, input_sha256=evidence))


def request_record(snapshot):
    return dict(schema='m4_final_signoff_v1', status='PENDING_TEAM_SIGNOFF', approved=False,
                snapshot=snapshot, snapshot_sha256=semantic_hash(snapshot),
                approved_by='', approved_at_utc='', approval_evidence_reference='',
                f1_exposure_acknowledged=False, f2_f4_exposure_status='pending',
                f2_f4_exposure_evidence='',
                decisions={name: dict(status='pending', resolution='') for name in DECISIONS})


def validate_approval(record, snapshot):
    """Validate attestation and exact bytes; this is not an identity service."""
    if record.get('schema') != 'm4_final_signoff_v1' or record.get('approved') is not True:
        raise ValueError('Explicit team approval is required; pending requests cannot train')
    if record.get('status') != 'APPROVED_FOR_DIRECT_QUANTILE_EXECUTION':
        raise ValueError('Approval status is not executable')
    if (record.get('snapshot') != snapshot or record.get('snapshot_sha256') != semantic_hash(snapshot)):
        raise ValueError('Approved snapshot differs from current code/config/input/protocol/versions')
    for field in ['approved_by', 'approved_at_utc', 'approval_evidence_reference']:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f'Missing approval field: {field}')
    try:
        approved_at = datetime.fromisoformat(record['approved_at_utc'].replace('Z', '+00:00'))
        if approved_at.utcoffset() is None or approved_at.utcoffset().total_seconds() != 0:
            raise ValueError('UTC required')
        if approved_at > datetime.now(timezone.utc):
            raise ValueError('Future approval timestamp')
    except (ValueError, TypeError) as exc:
        raise ValueError('approved_at_utc must be a valid non-future UTC timestamp') from exc
    if record.get('f1_exposure_acknowledged') is not True:
        raise ValueError('F1 prior test exposure must be acknowledged')
    exposure = record.get('f2_f4_exposure_status')
    if exposure not in ['pending', 'exploratory', 'team_verified_uninspected']:
        raise ValueError('Invalid F2-F4 exposure status')
    exposure_evidence = record.get('f2_f4_exposure_evidence')
    if exposure == 'team_verified_uninspected' and (not isinstance(exposure_evidence, str) or not exposure_evidence.strip()):
        raise ValueError('Team-verified exposure status requires evidence')
    decisions = record.get('decisions', {})
    if set(decisions) != set(DECISIONS):
        raise ValueError('Incomplete team decision checklist')
    for name in DECISIONS:
        decision = decisions[name]
        allowed = ['approved', 'deferred_out_of_scope'] if name in DEFER_ALLOWED else ['approved']
        if (not isinstance(decision, dict) or decision.get('status') not in allowed
                or not isinstance(decision.get('resolution'), str) or not decision['resolution'].strip()):
            raise ValueError(f'Unresolved decision: {name}')


def verify_snapshot_files(snapshot):
    verify_frozen_files({'file_sha256': snapshot['code_config_sha256'],
                         'input': {'file_sha256': snapshot['input_sha256']}})


def validate_counts(result):
    predictions, metrics, diagnostics = result
    manifest = diagnostics['manifest']
    if manifest['selected_scope'] != dict(fold_ids=FOLDS, configs=CONFIGS) or manifest['parameter_grid'] != GRID:
        raise ValueError('Final scope/grid mismatch')
    tuning = diagnostics['tuning']
    if (len(tuning) != 48 or tuning['selected'].sum() != 24
            or not tuning.groupby(['fold', 'config', 'tau_key']).size().eq(2).all()
            or not tuning.groupby(['fold', 'config', 'tau_key'])['selected'].sum().eq(1).all()):
        raise ValueError('Final tuning budget mismatch')
    if len(predictions) != 493416 or len(metrics) != 14712:
        raise ValueError('Final prediction/metric counts mismatch')
    split_frame = diagnostics['splits']
    if len(split_frame) != 12 or len(diagnostics['exclusions']) != 8 or split_frame.duplicated(['fold', 'split']).any():
        raise ValueError('Final split/exclusion panel mismatch')
    if set(zip(split_frame.fold, split_frame.split)) != {(f, s) for f in FOLDS for s in ['train', 'validation', 'test']}:
        raise ValueError('Final split scope mismatch')
    splits = split_frame.set_index(['fold', 'split'])
    for fold, expected in COUNTS.items():
        if int(splits.loc[(fold, 'train'), 'rows']) != expected['train']:
            raise ValueError(f'{fold} training count mismatch')
        for split in ['validation', 'test']:
            if int(splits.loc[(fold, split), 'rows']) != (10584 if fold == 'F4' else 10388):
                raise ValueError(f'{fold} split total count mismatch')
            if int(splits.loc[(fold, split), 'scored_rows']) != expected[split]:
                raise ValueError(f'{fold} scored count mismatch')
        actual_refit = sum(int(splits.loc[(fold, split), 'eligible_fit_rows']) for split in ['train', 'validation'])
        if actual_refit != expected['refit']:
            raise ValueError(f'{fold} eligible refit count mismatch')


def execute_approved(approval_path, root=ROOT):
    root = Path(root).resolve()
    approval_path = Path(approval_path).resolve()
    approval_digest = file_hash(approval_path)
    record = json.loads(approval_path.read_text(encoding='utf-8'))
    snapshot = current_snapshot(root)
    validate_approval(record, snapshot)  # No filesystem writes/training before this gate.
    eid = 'final_f1_f4_m0_m3_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:12]
    if inspect_run(root, eid)['status'] != 'absent':
        raise FileExistsError('Final experiment ID exists or is incomplete')
    audit = root / 'results/experiments' / ('m4_final_audit_' + eid)
    audit.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        receipt = audit / 'approval_record.json'
        with approval_path.open('rb') as src, receipt.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if file_hash(receipt) != approval_digest:
            raise ValueError('Approval bytes changed during receipt creation')
        write_json_exclusive(audit / 'execution_start.json', dict(
            experiment_id=eid, started_at_utc=datetime.now(timezone.utc).isoformat(),
            approval_sha256=approval_digest, snapshot_sha256=semantic_hash(snapshot)))
        protocol = snapshot['spec']['protocol']; ablation = snapshot['spec']['ablation']
        verify_snapshot_files(snapshot)
        data, preflight = production_preflight(root / FEATURE_RELATIVE, protocol, ablation, root)
        if preflight['hard_failures']:
            raise ProductionPreflightError(preflight)
        for count in preflight['fold_scored_counts']:
            if count['rows_scored'] != COUNTS[count['fold_id']][count['split']]:
                raise ValueError('Pre-fit scored count mismatch')
        if len(preflight['fold_scored_counts']) != 8 or {(r['fold_id'], r['split']) for r in preflight['fold_scored_counts']} != {
                (fold, split) for fold in FOLDS for split in ['validation', 'test']}:
            raise ValueError('Preflight fold/split set mismatch')
        write_json_exclusive(audit / 'preflight.json', preflight)
        verify_snapshot_files(snapshot)
        evidence = {**preflight['file_sha256'], **snapshot['input_sha256'],
                    **snapshot['code_config_sha256'], str(approval_path): approval_digest,
                    str(receipt): approval_digest}
        context = dict(mode='production', **preflight)
        context['file_sha256'] = evidence
        context['team_signoff'] = record
        context['F1_test_previously_inspected'] = True
        context['F2_F4_exposure_status'] = record['f2_f4_exposure_status']
        print('Approval receipt written; preflight PASS; starting 48 selections + 24 fresh refits.', flush=True)
        result = qp.run_direct_quantiles(data, protocol, ablation, eid, parameter_grid=None,
                                        input_context=context, fold_ids=FOLDS, configs=CONFIGS)
        validate_counts(result)
        directory = qp.save_run(*result, root=root)
        if inspect_run(root, eid)['status'] != 'complete':
            raise ValueError('Finalized bundle integrity failure')
        panel = pd.read_csv(directory / f'm4_quantiles_{eid}.csv', parse_dates=['date', 'forecast_origin'],
                            dtype={'store_nbr': str, 'family': str, 'city': str}, float_precision='round_trip')
        _validate_saved_panel(panel, result[2]['manifest'])
        pd.testing.assert_frame_equal(pd.read_csv(directory / 'metrics.csv', float_precision='round_trip'),
                                      qp.summarize_predictions(panel), check_dtype=False)
        verify_snapshot_files(snapshot)
        verify_frozen_files(result[2]['manifest']['provenance'])
        resources = {'measurement': 'unavailable'}
        try:
            import psutil
            memory = psutil.Process().memory_info()
            resources = dict(os_process_lifetime_peak_working_set_bytes=getattr(memory, 'peak_wset', None))
        except ImportError:
            pass
        write_json_exclusive(audit / 'execution_result.json', dict(status='complete', bundle=str(directory),
            wall_seconds=perf_counter()-started, expected_completed_fits=dict(selection=48, refit=24),
            fit_count_evidence='Completed frozen loops and 48 tuning records; not an independent call counter',
            resources=resources, artifact_sha256={p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}))
        hashes = {p.name: file_hash(p) for p in audit.iterdir() if p.is_file()}
        write_json_exclusive(audit / 'AUDIT_COMPLETE.json', {'files': hashes})
        return directory
    except Exception as exc:
        try:
            write_json_exclusive(audit / 'FAILED.json', dict(exception=type(exc).__name__, message=str(exc),
                                                           elapsed_seconds=perf_counter()-started))
        except OSError:
            pass  # Preserve the original execution failure if the audit disk is unavailable.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-signoff', type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare_signoff is not None:
        if args.approval is not None or args.execute:
            parser.error('Preparation cannot execute training')
        destination = args.prepare_signoff.resolve()
        if destination.parent != (ROOT / 'docs').resolve() or not destination.name.startswith('m4_') or destination.suffix != '.json':
            parser.error('Sign-off requests must be M4 JSON files directly under docs/')
        write_json_exclusive(destination, request_record(current_snapshot()))
        print(f'PENDING TEAM SIGN-OFF; no training: {destination}')
    elif args.execute and args.approval is not None:
        print(execute_approved(args.approval))
    else:
        parser.error('Use --prepare-signoff PATH, or --approval PATH --execute after sign-off')


if __name__ == '__main__':
    main()
