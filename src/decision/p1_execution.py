"""P1 proposal and explicit approval gate. Never trains forecasting models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from . import p1_residual_policy as p1
from .cost_simulation import cost_index, newsvendor_cost
from .policy_evaluation import MATCH, LEVELS, LABEL, configured_scenarios, inspect_evaluation
from .quantile_contracts import TAU_KEYS, file_hash, semantic_hash, verify_frozen_files, stable_mean, json_snapshot

PROPOSED = 'PROPOSED_NOT_APPROVED'
COST_RELATIVE = 'results/experiments/m4_costs_p2_p3_20261108T134955Z_b112ad3e7866'
PROPOSAL = dict(residual='y_true - y_pred_raw', pooling=['fold','model','config','family'],
                empirical_quantile_method='linear', minimum_samples=30,
                fallback=['fold','model','config'],
                quantity='max(0, test forecast + validation residual quantile)',
                models=['SeasonalNaive'], configs=['M0'], folds=['F1','F2','F3','F4'],
                tau_keys=list(TAU_KEYS),
                limitation='No independent calibration or conformal guarantee; F1 exploratory; F2-F4 exposure pending')


def proposed_spec():
    return p1.ResidualSpec(('family',), 'linear', 30, ((),))


def make_request(root=p1.ROOT):
    """Bind proposed method and existing files, without calculating residuals/costs."""
    from scripts.m4_pilot import FEATURE_RELATIVE
    root=Path(root).resolve()
    paths=[root/'data/processed/baselines.csv',root/'configs/cost_scenarios.yaml',
           root/'configs/research_protocol.yaml',root/'configs/ablation_M0_M5.yaml',
           root/'docs/m4_phase3d_p1_methodology.md']
    paths += list((root/FEATURE_RELATIVE).parent.glob('*'))
    paths += list((root/COST_RELATIVE).glob('*'))
    paths += list((root/'src/decision').glob('*.py'))
    paths += list((root/'src/forecasting').glob('*.py'))
    paths += list((root/'src/features').glob('*.py'))
    paths += list((root/'src/data').glob('*.py'))
    paths += [root/'scripts/m4_pilot.py']
    if not (root/FEATURE_RELATIVE).is_file() or inspect_evaluation(root/COST_RELATIVE)!='complete':
        raise ValueError('Require existing candidate and complete P2/P3 cost bundle')
    record=dict(status=PROPOSED, methodology=json_snapshot(PROPOSAL),
                file_sha256={str(p):file_hash(p) for p in sorted(set(paths)) if p.is_file()})
    return record


def validate_approval(record):
    """Approval is external evidence, never inferred from a proposal or old sign-off."""
    request=record.get('request',{})
    if record.get('status')!='APPROVED' or request.get('status')!=PROPOSED:
        raise p1.MethodologyPendingError('Require APPROVED P1 record; proposal is PROPOSED_NOT_APPROVED')
    if request.get('methodology')!=PROPOSAL or record.get('request_sha256')!=semantic_hash(request):
        raise ValueError('Approval methodology/request digest mismatch')
    for key in ['approved_by','approval_evidence','approval_communicated_date']:
        if not isinstance(record.get(key),str) or not record[key].strip():
            raise ValueError('Missing explicit approval evidence: '+key)
    datetime.strptime(record['approval_communicated_date'],'%Y-%m-%d')
    if not request.get('file_sha256'):raise ValueError('Approval has no frozen file evidence')
    verify_frozen_files({'file_sha256':request['file_sha256'],'input':{}})
    return request


def compare_with_p1(p1_rows, direct_rows, config):
    """Exact test alignment and P1-normalized ratios of sums, not mean row ratios."""
    scenarios=configured_scenarios(config); frames=[]; summaries=[]
    expected_scenarios={s['scenario'] for s in scenarios}
    if set(p1_rows.scenario)!=expected_scenarios or set(direct_rows.scenario)!=expected_scenarios:
        raise ValueError('Require exactly configured scenarios')
    if set(direct_rows.policy)!={'P2','P3'} or set(direct_rows.variant)!={'corrected','raw_clipped_sensitivity'}:
        raise ValueError('Unexpected direct policies/variants')
    if set(p1_rows.model)!={'SeasonalNaive'} or set(p1_rows.config)!={'M0'}:
        raise ValueError('P1 must be SeasonalNaive/M0')
    for s in scenarios:
        base=p1_rows.loc[p1_rows.scenario.eq(s['scenario'])].copy()
        if base.empty or base[MATCH+['y_true','quantity','simulated_cost']].isna().any().any() or base.duplicated(MATCH).any():
            raise ValueError('Invalid/missing/duplicate P1 keys')
        if not base.tau_key.eq(s['tau_key']).all():raise ValueError('P1 scenario/tau mapping mismatch')
        if not base.split.eq('test').all():raise ValueError('P1 economic evaluation is test-only')
        for policy in ['P2','P3']:
            for variant in ['corrected','raw_clipped_sensitivity']:
                direct=direct_rows.loc[direct_rows.scenario.eq(s['scenario']) & direct_rows.policy.eq(policy) & direct_rows.variant.eq(variant)].copy()
                if direct.empty or direct[MATCH+['y_true','quantity','simulated_cost']].isna().any().any() or direct.duplicated(MATCH).any():
                    raise ValueError('Invalid/missing/duplicate direct policy keys')
                pair=base.merge(direct,on=MATCH,how='outer',suffixes=('_P1','_direct'),indicator=True,validate='1:1')
                if not pair['_merge'].eq('both').all() or not np.array_equal(pair.y_true_P1,pair.y_true_direct):
                    raise ValueError('Exact P1/direct keys or y_true differ')
                for suffix in ['P1','direct']:
                    q=pair['quantity_'+suffix].to_numpy(float)
                    if not np.isfinite(q).all() or (q<0).any():raise ValueError('Invalid feasible quantity')
                    actual=newsvendor_cost(pair.y_true_P1,q,s['c_over'],s['c_under'])
                    if not np.array_equal(actual,pair['simulated_cost_'+suffix]):raise ValueError('Stored cost/scenario mismatch')
                pair['policy'],pair['variant'],pair['scenario']=policy,variant,s['scenario']
                frames.append(pair.drop(columns='_merge'))
                for level,grouping in LEVELS.items():
                    groups=[((),pair)] if not grouping else pair.groupby(grouping,observed=True,sort=True)
                    for key,g in groups:
                        keys=key if isinstance(key,tuple) else (key,)
                        denominator=g.simulated_cost_P1.to_numpy(float)
                        summaries.append(dict(zip(grouping,keys),scenario=s['scenario'],policy=policy,variant=variant,
                            report_level=level,sample_size=len(g),mean_cost_P1=stable_mean(denominator),
                            mean_cost_policy=stable_mean(g.simulated_cost_direct.to_numpy(float)),
                            cost_index_P1=cost_index(denominator,denominator),
                            cost_index_policy=cost_index(g.simulated_cost_direct,denominator),
                            p1_denominator_zero=not np.any(denominator>0),result_label=LABEL))
    return pd.concat(frames,ignore_index=True),pd.DataFrame(summaries)


def _json(path, record):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)


def inspect_bundle(directory):
    directory=Path(directory)
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text())
        if set(marker['files'])!={'p1_rows.csv','calibration.csv','paired_rows.csv','cost_indices.csv','manifest.json'}:
            return 'corrupt'
        return 'complete' if all(file_hash(directory/n)==h for n,h in marker['files'].items()) else 'corrupt'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete'


def run_approved(approval_path, root=p1.ROOT):
    root=Path(root).resolve();approval_path=Path(approval_path).resolve()
    approval=json.loads(approval_path.read_text());request=validate_approval(approval)
    # Exact binding prevents a truncated approval hash map bypassing preflight guards.
    if request!=make_request(root):raise ValueError('Current execution bindings differ from approved request')
    approval_digest=file_hash(approval_path)
    inputs=p1.load_real_p1_inputs(root)  # mandatory marker/preflight/exact point & frozen-key checks
    source=root/COST_RELATIVE
    if inspect_evaluation(source)!='complete':raise ValueError('P2/P3 cost source integrity failed')
    manifest=json.loads((source/'manifest.json').read_text())
    verify_frozen_files(manifest['source_forecast_provenance'])
    verify_frozen_files({'file_sha256':manifest['file_sha256'],'input':{}})
    direct=pd.read_csv(source/'row_costs.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    validate_approval(approval)
    if file_hash(approval_path)!=approval_digest:raise ValueError('Approval changed during preflight')
    rows,cal=p1._evaluate_p1(inputs,proposed_spec(),p1.read_yaml(root/'configs/cost_scenarios.yaml'))
    rows['methodology_status']='APPROVED';rows['result_label']=LABEL
    cal['methodology_status']='APPROVED'
    for frame in [rows,cal]:frame['approval_sha256']=approval_digest
    pairs,summary=compare_with_p1(rows,direct,p1.read_yaml(root/'configs/cost_scenarios.yaml'))
    parent=root/'results/experiments';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/('m4_p1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=parent))
    tables={'p1_rows.csv':rows,'calibration.csv':cal,'paired_rows.csv':pairs,'cost_indices.csv':summary}
    for name,table in tables.items():
        with (stage/name).open('x',encoding='utf-8',newline='') as stream:table.to_csv(stream,index=False)
        dates=[c for c in table if pd.api.types.is_datetime64_any_dtype(table[c])]
        strings={c:str for c in ['store_nbr','model','config','family','city'] if c in table}
        loaded=pd.read_csv(stage/name,parse_dates=dates or None,dtype=strings,float_precision='round_trip')
        pd.testing.assert_frame_equal(loaded,table.reset_index(drop=True),check_dtype=False)
    validate_approval(approval);verify_frozen_files(inputs.provenance)
    if inspect_evaluation(source)!='complete' or file_hash(approval_path)!=approval_digest:raise ValueError('Evidence changed during evaluation')
    _json(stage/'manifest.json',dict(approval=approval,approval_sha256=approval_digest,provenance=inputs.provenance,
        result_label=LABEL,F1='exploratory/post-pilot',F2_F4='exposure verification pending',
        calibration_guarantees='none; dependent validation residuals',
        output_sha256={n:file_hash(stage/n) for n in tables}))
    _json(stage/'COMPLETE.json',{'files':{n:file_hash(stage/n) for n in [*tables,'manifest.json']}})
    if inspect_bundle(stage)!='complete':raise ValueError('Staged P1 bundle invalid')
    if destination.exists():raise FileExistsError('Never overwrite prior P1 outputs')
    stage.rename(destination)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-request',type=Path)
    group.add_argument('--approval',type=Path)
    args=parser.parse_args()
    if args.prepare_request:_json(args.prepare_request,make_request())
    else:print(run_approved(args.approval))


if __name__=='__main__':main()
