"""Approval/orchestration guards only. NO estimator fitting, including synthetic."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts import m4_final_run as final


@pytest.fixture(autouse=True)
def forbid_estimator_fitting(monkeypatch):
    monkeypatch.setattr(HistGradientBoostingRegressor, 'fit',
                        lambda *a, **k: pytest.fail('Non-training tests must never fit an estimator'))


@pytest.fixture
def snapshot(tmp_path):
    evidence = tmp_path / 'synthetic_evidence.txt'
    evidence.write_text('synthetic bytes, not a dataset', encoding='utf-8')
    return dict(spec=dict(protocol={}, ablation={}), code_config_sha256={},
                input_sha256={str(evidence): final.file_hash(evidence)})


def approved_record(snapshot):
    value = final.request_record(snapshot)
    value.update(approved=True, status='APPROVED_FOR_DIRECT_QUANTILE_EXECUTION',
                 approved_by='SYNTHETIC TEST REVIEWER',
                 approved_at_utc=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat(),
                 approval_evidence_reference='synthetic-test-only', f1_exposure_acknowledged=True)
    for decision in value['decisions'].values():
        decision.update(status='approved', resolution='synthetic test resolution')
    return value


def test_request_is_not_approval_and_cannot_execute(snapshot):
    request = final.request_record(snapshot)
    assert request['approved'] is False and request['f2_f4_exposure_status'] == 'pending'
    with pytest.raises(ValueError, match='Explicit team approval'):
        final.validate_approval(request, snapshot)


@pytest.mark.parametrize('fault', ['status', 'digest', 'snapshot', 'signer', 'reference',
                                  'f1', 'decision', 'missing_decision', 'timestamp', 'future',
                                  'exposure_evidence', 'exposure_status'])
def test_invalid_approval_is_rejected(snapshot, fault):
    record = approved_record(snapshot)
    if fault == 'status': record['status'] = 'PENDING_TEAM_SIGNOFF'
    elif fault == 'digest': record['snapshot_sha256'] = '0'*64
    elif fault == 'snapshot': record['snapshot']['spec']['changed'] = True
    elif fault == 'signer': record['approved_by'] = ''
    elif fault == 'reference': record['approval_evidence_reference'] = ''
    elif fault == 'f1': record['f1_exposure_acknowledged'] = False
    elif fault == 'decision': record['decisions']['estimator_grid']['status'] = 'deferred_out_of_scope'
    elif fault == 'missing_decision': record['decisions'].pop('artifact_handoff')
    elif fault == 'timestamp': record['approved_at_utc'] = '2026-01-01T00:00:00'
    elif fault == 'future': record['approved_at_utc'] = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    elif fault == 'exposure_evidence':
        record['f2_f4_exposure_status'] = 'team_verified_uninspected'
        record['f2_f4_exposure_evidence'] = None
    else: record['f2_f4_exposure_status'] = 'confirmed_by_pilot_scores'
    with pytest.raises(ValueError):
        final.validate_approval(record, snapshot)


def test_explicit_deferrals_limit_scope_and_do_not_assert_confirmation(snapshot):
    record = approved_record(snapshot)
    for name in final.DEFER_ALLOWED:
        record['decisions'][name].update(status='deferred_out_of_scope', resolution='No cost/CI claims in prediction-only run')
    final.validate_approval(record, snapshot)
    assert record['f2_f4_exposure_status'] == 'pending'
    record['f2_f4_exposure_status'] = 'team_verified_uninspected'
    record['f2_f4_exposure_evidence'] = 'synthetic team review reference'
    final.validate_approval(record, snapshot)


def test_stale_code_or_input_snapshot_is_rejected(snapshot):
    record = approved_record(snapshot)
    changed = deepcopy(snapshot)
    changed['code_config_sha256']['new.py'] = 'changed'
    with pytest.raises(ValueError, match='snapshot differs'):
        final.validate_approval(record, changed)


def test_exclusive_request_writer_never_overwrites(tmp_path, snapshot):
    path = tmp_path / 'request.json'
    final.write_json_exclusive(path, final.request_record(snapshot))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        final.write_json_exclusive(path, {})
    assert path.read_bytes() == before


def test_prepare_is_nontraining_and_refuses_unsafe_paths(tmp_path, monkeypatch, snapshot):
    (tmp_path / 'docs').mkdir()
    monkeypatch.setattr(final, 'ROOT', tmp_path)
    monkeypatch.setattr(final, 'current_snapshot', lambda *a: snapshot)
    monkeypatch.setattr(final, 'execute_approved', lambda *a: pytest.fail('Prepare executed training path'))
    path = tmp_path / 'docs/m4_request.json'
    final.main(['--prepare-signoff', str(path)])
    assert json.loads(path.read_text())['approved'] is False
    with pytest.raises(SystemExit):
        final.main(['--prepare-signoff', str(tmp_path/'data/model_features.json')])
    with pytest.raises(SystemExit):
        final.main(['--prepare-signoff', str(path), '--execute'])


def test_pending_approval_stops_before_writes_or_preflight(tmp_path, monkeypatch, snapshot):
    path = tmp_path / 'request.json'
    final.write_json_exclusive(path, final.request_record(snapshot))
    monkeypatch.setattr(final, 'current_snapshot', lambda *a: snapshot)
    monkeypatch.setattr(final, 'production_preflight', lambda *a: pytest.fail('Pending approval reached preflight'))
    with pytest.raises(ValueError, match='Explicit team approval'):
        final.execute_approved(path, tmp_path)
    assert not (tmp_path / 'results').exists()


@pytest.mark.parametrize('failure', ['raised', 'returned'])
def test_approval_receipt_precedes_preflight_and_failure_never_fits(tmp_path, monkeypatch, snapshot, failure):
    path = tmp_path / 'approved.json'
    final.write_json_exclusive(path, approved_record(snapshot))
    monkeypatch.setattr(final, 'current_snapshot', lambda *a: snapshot)

    def fail_preflight(*args):
        receipt, = (tmp_path/'results/experiments').glob('m4_final_audit_*/approval_record.json')
        assert receipt.read_bytes() == path.read_bytes()
        report = {'hard_failures': ['synthetic preflight failure']}
        if failure == 'raised': raise final.ProductionPreflightError(report)
        return None, report

    monkeypatch.setattr(final, 'production_preflight', fail_preflight)
    monkeypatch.setattr(final.qp, 'run_direct_quantiles', lambda *a, **k: pytest.fail('Failed preflight reached training'))
    with pytest.raises(final.ProductionPreflightError): final.execute_approved(path, tmp_path)
    failed, = (tmp_path/'results/experiments').glob('m4_final_audit_*/FAILED.json')
    assert 'synthetic preflight failure' in failed.read_text()
    assert not list((tmp_path/'results/experiments').glob('m4_quantiles_*'))


def test_orchestration_delegates_defaults_and_freezes_approval_before_fit(tmp_path, monkeypatch, snapshot):
    path = tmp_path / 'approved.json'
    final.write_json_exclusive(path, approved_record(snapshot))
    monkeypatch.setattr(final, 'current_snapshot', lambda *a: snapshot)
    counts = [dict(fold_id=f, split=s, rows_scored=final.COUNTS[f][s])
              for f in final.FOLDS for s in ['validation', 'test']]
    monkeypatch.setattr(final, 'production_preflight', lambda *a: (None, dict(hard_failures=[],file_sha256={},fold_scored_counts=counts)))

    class StopBeforeAnyFit(Exception): pass

    def no_actual_training(*args, **kwargs):
        receipt, = (tmp_path/'results/experiments').glob('m4_final_audit_*/approval_record.json')
        assert receipt.read_bytes() == path.read_bytes()
        assert kwargs['parameter_grid'] is None
        assert kwargs['fold_ids'] == final.FOLDS and kwargs['configs'] == final.CONFIGS
        assert args[3].startswith('final_f1_f4_m0_m3_')
        context = kwargs['input_context']
        assert context['F1_test_previously_inspected'] is True
        assert context['F2_F4_exposure_status'] == 'pending'
        assert context['file_sha256'][str(receipt)] == final.file_hash(path)
        raise StopBeforeAnyFit()

    monkeypatch.setattr(final.qp, 'run_direct_quantiles', no_actual_training)
    with pytest.raises(StopBeforeAnyFit): final.execute_approved(path, tmp_path)


def count_fixture():
    tuning = pd.DataFrame([dict(fold=f, config=c, tau_key=t, candidate_id=i, selected=i==0)
                           for f in final.FOLDS for c in final.CONFIGS for t in ['tau_0.50','tau_0.75','tau_5_6'] for i in [0,1]])
    splits = []
    for f, counts in final.COUNTS.items():
        for split in ['train','validation','test']:
            splits.append(dict(fold=f,split=split,rows=counts['train'] if split=='train' else 10584 if f=='F4' else 10388,
                               scored_rows=counts[split],eligible_fit_rows=counts['train'] if split=='train' else counts['refit']-counts['train'] if split=='validation' else counts['test']))
    return (pd.DataFrame(index=range(493416)),pd.DataFrame(index=range(14712)),
            dict(manifest=dict(selected_scope=dict(fold_ids=final.FOLDS,configs=final.CONFIGS),parameter_grid=final.GRID),
                 tuning=tuning,splits=pd.DataFrame(splits),exclusions=pd.DataFrame(index=range(8))))


@pytest.mark.parametrize('fault', ['none','f2_cooldown','selection_budget','prediction_count','scope'])
def test_count_assertions_without_training(fault):
    result = count_fixture()
    if fault == 'f2_cooldown':
        rows=result[2]['splits'];rows.loc[rows.fold.eq('F2') & rows.split.eq('validation'),'eligible_fit_rows']=9989
    elif fault == 'selection_budget': result[2]['tuning']=result[2]['tuning'].iloc[:-1]
    elif fault == 'prediction_count': result=(result[0].iloc[:-1],result[1],result[2])
    elif fault == 'scope': result[2]['manifest']['selected_scope']={'fold_ids':['F1'],'configs':['M0']}
    if fault=='none':final.validate_counts(result)
    else:
        with pytest.raises(ValueError): final.validate_counts(result)


def test_help_cannot_execute(monkeypatch):
    monkeypatch.setattr(final,'execute_approved',lambda *a:pytest.fail('Help triggered execution'))
    with pytest.raises(SystemExit) as result:final.main(['--help'])
    assert result.value.code==0
