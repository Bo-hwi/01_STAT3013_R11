"""Scoped-run and locked entry-point checks; all training inputs are synthetic."""

from copy import deepcopy
import json
import shutil

import pandas as pd
import pytest

from test_m4_quantile_pipeline import contracts, features, small_thread_budget, synthetic_run, GRID
from test_m4_production_readiness import delivery
from test_m4_feature_staging import prepared_delivery
from scripts import m4_pilot as pilot
from src.decision import quantile_pipeline as qp
from src.decision.feature_staging import collect_m1_metadata
from src.decision.quantile_contracts import file_hash, semantic_hash, validate_prediction_panel
from src.decision.quantile_artifacts import inspect_run
from src.decision.quantile_preflight import ProductionPreflightError


@pytest.mark.parametrize('kwargs', [
    {'fold_ids': []}, {'fold_ids': ['F1', 'F1']}, {'fold_ids': ['F9']},
    {'fold_ids': 'F1'}, {'fold_ids': [1]}, {'fold_ids': 1},
    {'configs': []}, {'configs': ['M0', 'M0']}, {'configs': ['M1']},
    {'configs': 'M0'}, {'configs': [None]}, {'configs': 1},
])
def test_invalid_scope_rejected_before_fit(features, contracts, monkeypatch, kwargs):
    monkeypatch.setattr(qp, 'build_quantile_pipeline', lambda *a: pytest.fail('Unexpected fitting'))
    with pytest.raises(ValueError):
        qp.run_direct_quantiles(features, *contracts, 'synthetic_invalid_scope', [GRID[0]], **kwargs)


def test_defaults_preserve_all_shared_folds_and_both_configs(synthetic_run):
    manifest = synthetic_run[2]['manifest']
    assert manifest['selected_scope'] == {'fold_ids': ['F1'], 'configs': ['M0', 'M3']}
    assert manifest['folds'] == manifest['contracts']['protocol']['rq2']['validation']['test_folds']
    assert qp._scope_ids(None, ['F1', 'F2', 'F3', 'F4'], 'fold_ids') == ('F1', 'F2', 'F3', 'F4')
    assert qp._scope_ids(['M3', 'M0'], qp.CONFIGS, 'configs') == qp.CONFIGS
    assert manifest['result_label'] == manifest['contracts']['ablation']['result_label']
    assert not manifest['pilot_only']


def test_f1_m0_exact_six_fits_shared_masks_and_frozen_spec(features, contracts, monkeypatch):
    # Full unmodified shared schedule: scope selection must stop before F2 data are needed.
    protocol = qp.read_yaml(qp.ROOT / 'configs/research_protocol.yaml')
    original = deepcopy(protocol)
    builder, shared, freezer = qp.build_quantile_pipeline, qp.iter_folds, qp.freeze_provenance
    fits, yielded, frozen = [], [], []

    def record_shared(*args):
        for fold in shared(*args):
            yielded.append(fold.fold_id)
            yield fold

    def record_freeze(*args):
        value = freezer(*args)
        frozen.append(value)
        return value

    def record_builder(*args):
        assert frozen, 'Provenance must be frozen before constructing fit models'
        model = builder(*args)
        fit = model.fit

        def record_fit(X, y):
            fits.append((X.copy(), y.copy()))
            return fit(X, y)

        model.fit = record_fit
        return model

    monkeypatch.setattr(qp, 'iter_folds', record_shared)
    monkeypatch.setattr(qp, 'freeze_provenance', record_freeze)
    monkeypatch.setattr(qp, 'build_quantile_pipeline', record_builder)
    before_grid = deepcopy(qp.DEFAULT_GRID)
    predictions, metrics, diagnostics = qp.run_direct_quantiles(
        features, protocol, contracts[1], 'synthetic_scoped_pilot', [pilot.PILOT_PARAMETERS],
        input_context={'mode': 'pilot'}, fold_ids=['F1'], configs=['M0'])
    assert len(fits) == 6
    assert yielded == ['F1', 'F1']
    assert protocol == original and qp.DEFAULT_GRID == before_grid
    fold = next(shared(features, protocol, contracts[1], protocol['scope']['deep_study_families']))
    numeric, categorical = qp.feature_columns('M0', contracts[1])
    for i, (X, y) in enumerate(fits):
        expected = fold.train if i < 3 else fold.refit_rows()
        pd.testing.assert_frame_equal(X, expected[numeric + categorical])
        pd.testing.assert_series_equal(y, expected.sales)
        assert expected.date.max() < fold.test.date.min()
    expected = pd.concat([getattr(fold, split).loc[getattr(fold, split).score_mask, [*qp.KEY, 'city']]
                          .assign(fold='F1', split=split) for split in ['validation', 'test']])
    validate_prediction_panel(predictions, expected, ['M0'])
    assert set(predictions.config) == {'M0'} and set(predictions.fold) == {'F1'}
    assert predictions.corrected_prediction.ge(0).all() and not predictions.corrected_crossing.any()
    assert metrics.sample_size.gt(0).all()
    manifest = diagnostics['manifest']
    digest = manifest.pop('run_spec_sha256')
    assert digest == semantic_hash(manifest)
    assert manifest['selected_scope'] == {'fold_ids': ['F1'], 'configs': ['M0']}
    assert manifest['folds'] == original['rq2']['validation']['test_folds'][:1]
    assert manifest['parameter_grid'] == [pilot.PILOT_PARAMETERS]
    assert manifest['quantiles'] == [0.5, 0.75, 5 / 6]
    assert manifest['result_label'] == qp.PILOT_LABEL and manifest['pilot_only']
    assert any(path.endswith('scripts\\m4_pilot.py') or path.endswith('scripts/m4_pilot.py')
               for path in manifest['provenance']['file_sha256'])


@pytest.fixture
def pilot_bundle(prepared_delivery, monkeypatch):
    root, old_path, frame = prepared_delivery
    path = root / pilot.FEATURE_RELATIVE
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    path.with_suffix('.meta.json').write_text(json.dumps(collect_m1_metadata(frame)), encoding='utf-8')
    for name in ['curated_sales_weather.parquet', 'curated_sales_weather.meta.json']:
        shutil.copyfile(old_path.parent / name, path.parent / name)
    outputs = {p.name: file_hash(p) for p in path.parent.iterdir()}
    report = {'recommendation': 'GO_FOR_ADOPTION_REVIEW', 'preflight_status': 'PASS',
              'original_inputs_unchanged': True, 'code_configs_unchanged': True, 'output_sha256': outputs}
    report_path = path.parent / 'reconciliation_report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    (path.parent / 'CANDIDATE_COMPLETE.json').write_text(
        json.dumps({'report_sha256': file_hash(report_path), 'adopted': False}), encoding='utf-8')
    # Only the test's M4 expected digest changes; no real files or M1 globals are modified.
    monkeypatch.setattr(pilot, 'EXPECTED_FEATURE_SHA256', file_hash(path))
    return root, path


@pytest.mark.parametrize('failure', ['raised', 'returned'])
def test_preflight_failure_never_calls_training(pilot_bundle, monkeypatch, failure):
    root, _ = pilot_bundle

    def failed_preflight(*args):
        report = {'hard_failures': ['synthetic failure']}
        if failure == 'raised':
            raise ProductionPreflightError(report)
        return None, report

    monkeypatch.setattr(pilot, 'production_preflight', failed_preflight)
    monkeypatch.setattr(qp, 'run_direct_quantiles', lambda *a, **k: pytest.fail('Training after failed preflight'))
    with pytest.raises(ProductionPreflightError):
        pilot.run_pilot('synthetic_gate', root)
    assert not (root / 'results').exists()


@pytest.mark.parametrize('fault', ['features', 'metadata', 'marker', 'failed', 'digest'])
def test_provenance_failures_prevent_preflight_and_training(pilot_bundle, monkeypatch, fault):
    root, path = pilot_bundle
    if fault in ['features', 'metadata']:
        target = path if fault == 'features' else path.with_suffix('.meta.json')
        with target.open('ab') as stream:
            stream.write(b'corruption')
    elif fault == 'marker':
        (path.parent / 'CANDIDATE_COMPLETE.json').write_text('{}', encoding='utf-8')
    elif fault == 'failed':
        (path.parent / 'FAILED.json').write_text('{}', encoding='utf-8')
    else:
        monkeypatch.setattr(pilot, 'EXPECTED_FEATURE_SHA256', '0' * 64)
    monkeypatch.setattr(pilot, 'production_preflight', lambda *a: pytest.fail('Invalid bundle reached preflight'))
    with pytest.raises(ValueError):
        pilot.run_pilot('synthetic_invalid_bundle', root)


def test_synthetic_entrypoint_reuses_full_pipeline_and_transaction(pilot_bundle, monkeypatch):
    root, _ = pilot_bundle
    runner = qp.run_direct_quantiles
    calls = []

    def record_runner(*args, **kwargs):
        calls.append(kwargs)
        return runner(*args, **kwargs)

    monkeypatch.setattr(qp, 'run_direct_quantiles', record_runner)
    directory = pilot.run_pilot('synthetic_pilot_entry', root)
    assert inspect_run(root, 'synthetic_pilot_entry')['status'] == 'complete'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['result_label'] == qp.PILOT_LABEL and manifest['pilot_only']
    assert calls[0]['configs'] == ['M0'] and calls[0]['fold_ids'] == ['F1']
    assert calls[0]['parameter_grid'] == [pilot.PILOT_PARAMETERS]
    assert calls[0]['input_context']['pilot_configuration']['expected_model_fits'] == 6
    assert len(pd.read_csv(directory / 'tuning.csv')) == 3
    panel = pd.read_csv(directory / 'm4_quantiles_synthetic_pilot_entry.csv')
    validate_prediction_panel(panel, expected_configs=['M0'])
    assert set(panel.fold) == {'F1'} and len(manifest['scored_key_contract']) == 2
    before = file_hash(directory / 'manifest.json')
    with pytest.raises(FileExistsError):
        pilot.run_pilot('synthetic_pilot_entry', root)
    assert file_hash(directory / 'manifest.json') == before and len(calls) == 1


def test_evidence_changed_during_preflight_prevents_training(pilot_bundle, monkeypatch):
    root, path = pilot_bundle
    preflight = pilot.production_preflight

    def mutate_after_preflight(*args):
        result = preflight(*args)
        with path.with_suffix('.meta.json').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        return result

    monkeypatch.setattr(pilot, 'production_preflight', mutate_after_preflight)
    monkeypatch.setattr(qp, 'run_direct_quantiles', lambda *a, **k: pytest.fail('Changed evidence reached training'))
    with pytest.raises(ValueError, match='Input evidence changed'):
        pilot.run_pilot('synthetic_mutation', root)


def test_help_never_runs_pilot(monkeypatch):
    monkeypatch.setattr(pilot, 'run_pilot', lambda *a: pytest.fail('Help triggered training'))
    with pytest.raises(SystemExit) as result:
        pilot.main(['--help'])
    assert result.value.code == 0
