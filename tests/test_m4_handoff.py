"""Non-training tests for read-only reconciliation, export safety and claim scope."""
import json
import numpy as np
import pandas as pd
import pytest

from src.decision import m4_handoff as h


def summaries():
    cells=[];ranks=[];costs={}
    for scenario in ['S1','S3','S5']:
        row=dict(scenario=scenario,sample_size=2)
        for i,policy in enumerate(h.matrix.POLICIES):
            values=np.array([10.,10.]) if policy=='P1' else np.array([4.+i,6.+i])
            costs[(scenario,policy)]=values
            mean=values.mean();row['mean_cost_'+policy]=mean
            ranks.append(dict(scenario=scenario,policy=policy,sample_size=2,mean_simulated_cost=mean,P1_cost_index=mean/10))
        cells.append(row)
    return pd.DataFrame(cells),pd.DataFrame(ranks),costs


def test_pooled_reconciliation_of_stored_costs_and_indices():
    checks=h.reconcile_pooled(*summaries())
    assert len(checks)==15 and checks.status.eq('PASS').all()


@pytest.mark.parametrize('fault',['matrix','index','count','missing_policy'])
def test_reconciliation_fails_on_source_disagreement(fault):
    cells,ranks,costs=summaries()
    if fault=='matrix':cells.loc[0,'mean_cost_P2']+=1
    elif fault=='index':ranks.loc[0,'P1_cost_index']+=.01
    elif fault=='count':ranks.loc[0,'sample_size']=3
    else:ranks=ranks.iloc[1:]
    with pytest.raises(ValueError):h.reconcile_pooled(cells,ranks,costs)


def test_break_even_is_algebra_not_money_or_policy_redefinition():
    _,ranks,_=summaries();before=ranks.copy(deep=True)
    result=h.break_even_table(ranks)
    assert len(result)==6 and result.nonnegative_conditional_headroom.ge(0).all()
    assert result.units.str.contains('not money').all()
    pd.testing.assert_frame_equal(ranks,before)


def test_byte_export_is_exclusive_and_never_writes_repository_canonical_path(tmp_path):
    source=tmp_path/'authoritative.csv';source.write_bytes(b'a,b\n1,2\n')
    stage=tmp_path/'handoff';stage.mkdir()
    destination=h.export_copy(source,stage,'exports/results/tables/cost_scenarios.csv')
    assert destination.read_bytes()==source.read_bytes()
    assert not (tmp_path/'results/tables/cost_scenarios.csv').exists()
    with pytest.raises(FileExistsError):h.export_copy(source,stage,'exports/results/tables/cost_scenarios.csv')
    with pytest.raises(ValueError):h.export_copy(source,stage,'../../data.csv')
    with pytest.raises(ValueError):h.export_copy(source,stage,tmp_path/'outside.csv')


def test_conflicting_or_changed_provenance_fails(tmp_path):
    source=tmp_path/'source';source.write_text('original');digest=h.file_hash(source)
    assert h.recorded_hashes([{'file_sha256':{str(source):digest}}])=={str(source):digest}
    with pytest.raises(ValueError):h.recorded_hashes([{'file_sha256':{str(source):digest}},{'nested':{'file_sha256':{str(source):'other'}}}])
    source.write_text('changed')
    with pytest.raises(ValueError):h.recorded_hashes([{'file_sha256':{str(source):digest}}])


def test_report_retains_required_limitations_and_metric_distinctions():
    _,rankings,_=summaries()
    classes=pd.DataFrame({'scenario':['S1'],'mixed_or_inconsistent':[154]})
    text=h.report_text(rankings,pd.DataFrame(),classes)
    for phrase in ['F1','F2–F4','ERA5','censored','ROI','causality','independent','M4_REPRODUCED_M3','cell/P1/P4','pinball','point MAE']:
        assert phrase in text
    assert 'không phải official M3' in text


def test_package_checksum_and_path_safety(tmp_path):
    required=['README.md','M4_METHODS_RESULTS.md','artifact_map.csv','figure_index.csv','risk_checklist.md','REPRODUCE.md']
    for name in required:(tmp_path/name).write_text('synthetic')
    outputs={n:h.file_hash(tmp_path/n) for n in required}
    (tmp_path/'handoff_manifest.json').write_text(json.dumps({'output_sha256':outputs}))
    marker={'files':{**outputs,'handoff_manifest.json':h.file_hash(tmp_path/'handoff_manifest.json')}}
    (tmp_path/'COMPLETE.json').write_text(json.dumps(marker))
    assert h.inspect_package(tmp_path)=='complete'
    (tmp_path/'README.md').write_text('tamper')
    assert h.inspect_package(tmp_path)=='corrupt'


def test_classification_crosstab_csv_roundtrip_preserves_values_and_source(tmp_path):
    original=pd.crosstab(pd.Series(['S1','S1'],name='scenario'),pd.Series(['mixed','mixed'],name='weather_class')).reset_index()
    before=original.copy(deep=True);frame=h.csv_report_frame(original)
    assert original.columns.name=='weather_class' and frame.columns.name is None
    path=tmp_path/'classification.csv';frame.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),frame,check_dtype=False)
    pd.testing.assert_frame_equal(original,before)
