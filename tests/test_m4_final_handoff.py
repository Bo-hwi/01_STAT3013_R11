"""Portable tests of the final official-source handoff entry point."""
import json
import sys
from pathlib import Path

import pytest
import pandas as pd

from src.decision import final_handoff as f


@pytest.mark.parametrize('kwargs,expected',[
    ({},'Validate'),
    ({'hard_qa_failure':True},'No-go'),
    ({'complete':False,'ci_lower':-.2,'ci_upper':-.1},'Validate'),
    ({'operational_verified':True,'fold_direction':'directionally_favorable','ci_lower':-.2,'ci_upper':-.1},'Pilot'),
    ({'fold_direction':'directionally_favorable','ci_lower':-.2,'ci_upper':-.1},'Validate'),
    ({'fold_direction':'directionally_unfavorable','ci_lower':.1,'ci_upper':.2},'No-go'),
    ({'fold_direction':'mixed_or_inconsistent','ci_lower':-.2,'ci_upper':.1},'Validate'),
])
def test_labels_follow_evidence_not_forced_counts(kwargs,expected):
    assert f.recommendation(**kwargs)==expected


def test_missing_explicit_inputs_fail_clearly_without_source_overwrite(tmp_path,monkeypatch,capsys):
    source=Path(f.__file__).with_name('decision_matrix.py');before=source.read_bytes()
    args=['decision_matrix']
    for name in ['p4-bundle','p1-bundle','direct-bundle','quantile-bundle','official-predictions','annotations']:
        args+=['--'+name,str(tmp_path/'missing')]
    args+=['--output-dir',str(tmp_path/'output')]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(SystemExit,match='Missing completed bundle'):f.main()
    assert source.read_bytes()==before and not (tmp_path/'output').exists()


def test_existing_export_never_overwritten(tmp_path):
    output=tmp_path/'existing';output.mkdir();(output/'evidence').write_text('keep')
    with pytest.raises(FileExistsError):f.export({},output)
    assert (output/'evidence').read_text()=='keep'


def test_export_writes_only_output_not_entry_source(tmp_path,monkeypatch):
    source=Path(f.__file__).with_name('decision_matrix.py');before=source.read_bytes()
    table=pd.DataFrame({'test':[1]})
    monkeypatch.setattr(f,'make_tables',lambda *a:(table,table,table,table))
    monkeypatch.setattr(f,'generate_report',lambda *a:None)
    inputs={'p4_source':'synthetic-only','files':{},'qa':{}}
    output=f.export(inputs,tmp_path/'new')
    assert (output/'COMPLETE.json').is_file() and source.read_bytes()==before


def test_historical_changed_file_requires_exact_explicit_archive(tmp_path):
    historical=tmp_path/'old_config';historical.write_text('old')
    digest=f.contracts.file_hash(historical)
    saved=tmp_path/'saved';saved.write_text('old')
    bundle=tmp_path/'bundle';bundle.mkdir()
    manifest={'file_sha256':{str(historical):digest}}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    (bundle/'COMPLETE.json').write_text(json.dumps({'files':{'manifest.json':f.contracts.file_hash(bundle/'manifest.json')}}))
    historical.write_text('new authorized migration')
    with pytest.raises(ValueError,match='explicit pre-migration archive'):f.verify_bundle(bundle)
    _,_,qa=f.verify_bundle(bundle,{str(historical):(saved,digest)})
    assert qa['archived_evidence']==[str(historical)]
    saved.write_text('corrupt')
    with pytest.raises(ValueError,match='Archived evidence drift'):f.verify_bundle(bundle,{str(historical):(saved,digest)})


def test_pending_downstream_approval_never_resamples(tmp_path,monkeypatch):
    approval=tmp_path/'pending.json';approval.write_text(json.dumps({'status':'PENDING_APPROVAL'}))
    def forbidden(*a,**k):pytest.fail('Pending approval reached bootstrap')
    monkeypatch.setattr(f.fixed,'bootstrap_supplement',forbidden)
    with pytest.raises(ValueError,match='explicit exact-request approval'):
        f.run_bootstrap({'p4_source':'synthetic','files':{}},approval,tmp_path/'out')


def test_receipt_migration_preserves_original_request_hashes():
    root=Path(f.__file__).resolve().parents[2]
    for path in (root/'docs').glob('m4*.approved.json'):
        record=json.loads(path.read_text(encoding='utf-8'))
        assert record['approved_by']=='M4 (Huy) self-approved'
        if 'request_sha256' in record:
            assert record['request_sha256']==f.contracts.semantic_hash(record['request'])
    protocol=(root/'configs/research_protocol.yaml').read_text(encoding='utf-8')
    assert 'primary_quantile_model: HistGradientBoostingRegressor' in protocol
    assert 'decided 2026-11-08 (team chat), committed later' in protocol
