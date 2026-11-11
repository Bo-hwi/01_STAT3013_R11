"""Clipped-residual P4 proposal and approval-gated evaluation; no model fitting."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from . import p4_reproduction as reproduction
from . import p1_residual_policy as residual
from . import p1_execution as p1_artifacts
from .cost_simulation import newsvendor_cost, cost_index
from .policy_evaluation import configured_scenarios, inspect_evaluation, LEVELS, LABEL, MATCH
from .quantile_contracts import file_hash, semantic_hash, verify_frozen_files, json_snapshot, stable_mean
from .quantile_models import QUANTILE_LEVELS, pinball_loss

ROOT=reproduction.ROOT
SOURCE='results/experiments/m4_reproduced_m3_20261108T200231Z_d1e59483f28f4c01a76f02f972367216'
P1_SOURCE='results/experiments/m4_p1_20261108T173741Z_4d68ed38ce2f4d1299d1e464d47dfa64'
DIRECT_SOURCE=p1_artifacts.COST_RELATIVE
STATUS='PENDING_APPROVAL'
PROPOSAL=dict(policy='P4',version='clipped_validation_residual_v1',
    residual='y_true - clipped_validation_prediction',pooling=['fold','config','family'],
    quantile_method='linear',minimum_samples=30,fallback=['fold','config'],
    base_prediction='clipped fresh_train_validation_refit test prediction',
    quantity='max(0, base_prediction + validation_residual_quantile)',
    algorithm='fixed selected algorithm per fold/config; audit metadata, never reselected',
    sensitivity='P4 uses this single calibration rule; fixed P2/P3 raw-clipped sensitivity remains separate',
    change_from_prior='Replaces raw validation residuals; selected_model removed from declared pooling because unique per fold/config',
    limitations=['validation reused for hyperparameter/algorithm selection and calibration; not independent',
                 'train-only residuals applied to fresh train+validation refit predictor; distribution mismatch possible',
                 'calendar/cross-store dependence; n counts observations, not independent days',
                 'no conformal, causal, operational or actual profit/ROI guarantee',
                 'F1 exploratory/post-pilot; F2-F4 prior test exposure verification pending'])


def selection_audit(panel, folds, tuning):
    mapping={f['fold_id']:f for f in folds};records=[]
    for (fold,config),rows in panel.groupby(['fold','config'],sort=True):
        selected=tuning.loc[tuning.fold.eq(fold)&tuning.config.eq(config)&tuning.selected_candidate&tuning.selected_algorithm]
        if len(selected)!=1 or rows.model.nunique()!=1 or rows.selected_parameters.nunique()!=1:
            raise ValueError('Require one fixed selected algorithm/parameters per fold/config')
        chosen=selected.iloc[0]
        if not rows.model.eq(chosen.model).all() or not rows.selected_parameters.eq(chosen.parameters).all():
            raise ValueError('Selected algorithm/parameters disagree with tuning records')
        f=mapping[fold]
        records.append(dict(fold=fold,config=config,model=chosen.model,selected_parameters=chosen.parameters,
            validation_MAE=float(chosen.validation_MAE),training_cutoff=f['train'][1],refit_cutoff=f['validation'][1],
            validation_start=f['validation'][0],validation_end=f['validation'][1],
            test_start=f['test'][0],test_end=f['test'][1],validation_fit_stage='train_only',test_fit_stage='fresh_train_validation_refit',
            cutoff_evidence='declared shared schedule and verified prediction fit stages'))
    return pd.DataFrame(records)


def _matched_test(expected, actual):
    actual=actual.copy()
    if 'split' not in actual:actual['split']='test'
    if not actual.split.eq('test').all():raise ValueError('Economic observations must be test-only')
    reproduction.assert_matched_rows(expected,actual)


def validate_fixed_costs(expected, baseline, direct, config):
    scenarios=configured_scenarios(config);ids={s['scenario'] for s in scenarios}
    if set(baseline.scenario)!=ids or set(direct.scenario)!=ids:raise ValueError('Fixed scenario set mismatch')
    if set(baseline.policy)!={'P1'} or set(direct.policy)!={'P2','P3'} or set(direct.variant)!={'corrected','raw_clipped_sensitivity'}:
        raise ValueError('Fixed policy/variant set mismatch')
    for frame in [baseline,direct]:
        if not np.isfinite(frame[['y_true','quantity','simulated_cost']].to_numpy(float)).all() or frame[['quantity','simulated_cost']].lt(0).any().any():
            raise ValueError('Invalid fixed cost values')
    for s in scenarios:
        base=baseline.loc[baseline.scenario.eq(s['scenario'])]
        if not base.tau_key.eq(s['tau_key']).all():raise ValueError('Fixed P1 tau mismatch')
        _matched_test(expected,base)
        for policy in ['P2','P3']:
            for variant in ['corrected','raw_clipped_sensitivity']:
                rows=direct.loc[direct.scenario.eq(s['scenario'])&direct.policy.eq(policy)&direct.variant.eq(variant)]
                if not rows.tau_key.eq(s['tau_key']).all():raise ValueError('Fixed direct tau mismatch')
                _matched_test(expected,rows)


def load_sources(root=ROOT):
    """Verify completed historical artifacts without invoking any calibration."""
    root=Path(root).resolve();directory=root/SOURCE
    if reproduction.inspect_bundle(directory)!='complete':raise ValueError('Reproduction bundle incomplete/corrupt')
    manifest=json.loads((directory/'manifest.json').read_text())
    receipt=manifest['approval'];request=receipt['request']
    if (manifest['label']!=reproduction.LABEL or receipt['status']!='APPROVED'
            or receipt['request_sha256']!=semantic_hash(request)
            or request['specification']!=reproduction.SPEC or manifest['diagnostics']['fits']!=64):
        raise ValueError('Invalid historical reproduction scope/approval')
    # Historical file maps remain authoritative: do not recompute a live runner
    # request whose directory glob can include newly added M4 modules.
    verify_frozen_files({'file_sha256':request['file_sha256'],'input':{}})
    verify_frozen_files({'file_sha256':manifest['file_sha256'],'input':{}})
    panel=pd.read_csv(directory/'predictions.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    expected=pd.read_csv(directory/'expected_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    tuning=pd.read_csv(directory/'tuning.csv',float_precision='round_trip')
    protocol=reproduction.read_yaml(root/'configs/research_protocol.yaml');folds=protocol['rq2']['validation']['test_folds']
    reproduction.validate_forecasts(panel,expected,folds)
    audit=selection_audit(panel,folds,tuning)
    if list(sorted(audit.fold.unique()))!=['F1','F2','F3','F4'] or len(audit)!=8:raise ValueError('Require F1-F4/M0/M3')
    config=reproduction.read_yaml(root/'configs/cost_scenarios.yaml');configured_scenarios(config)
    p1=root/P1_SOURCE;direct_dir=root/DIRECT_SOURCE
    if p1_artifacts.inspect_bundle(p1)!='complete' or inspect_evaluation(direct_dir)!='complete':
        raise ValueError('Fixed cost bundle incomplete/corrupt')
    p1_manifest=json.loads((p1/'manifest.json').read_text());direct_manifest=json.loads((direct_dir/'manifest.json').read_text())
    verify_frozen_files(p1_manifest['provenance'])
    verify_frozen_files({'file_sha256':p1_manifest['approval']['request']['file_sha256'],'input':{}})
    verify_frozen_files(direct_manifest['source_forecast_provenance'])
    verify_frozen_files({'file_sha256':direct_manifest['file_sha256'],'input':{}})
    baseline=pd.read_csv(p1/'p1_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    direct=pd.read_csv(direct_dir/'row_costs.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    test=expected.loc[expected.split.eq('test')]
    if len(test)!=41076:raise ValueError('Expected exactly 41,076 frozen test observations')
    validate_fixed_costs(test,baseline,direct,config)
    files={str(p.resolve()):file_hash(p) for directory in [directory,p1,direct_dir] for p in directory.iterdir() if p.is_file()}
    for record in [request,manifest,p1_manifest['approval']['request'],direct_manifest]:files.update(record['file_sha256'])
    files.update(p1_manifest['provenance']['file_sha256']);files.update(p1_manifest['provenance']['input']['file_sha256'])
    return dict(panel=panel,expected=expected,folds=folds,audit=audit,tuning=tuning,baseline=baseline,direct=direct,config=config,files=files)


def make_request(root=ROOT):
    root=Path(root).resolve();sources=load_sources(root)
    files=dict(sources['files'])
    for path in [Path(__file__).resolve(),root/'docs/m4_phase3e3_p4_proposal.md',root/'tests/test_m4_p4_policy.py',
                 root/'configs/cost_scenarios.yaml',root/'configs/research_protocol.yaml']:
        files[str(path.resolve())]=file_hash(path)
    group_sizes=sources['panel'].loc[lambda x:x.split.eq('validation')].groupby(['fold','config','family'],observed=True).size().reset_index(name='n_validation')
    return json_snapshot(dict(status=STATUS,methodology=PROPOSAL,source_bundle=str(root/SOURCE),
        fixed_p1_bundle=str(root/P1_SOURCE),fixed_p2_p3_bundle=str(root/DIRECT_SOURCE),
        test_observations_per_scenario=41076,scenarios=configured_scenarios(sources['config']),
        selected_mapping=sources['audit'].to_dict(orient='records'),validation_group_sizes=group_sizes.to_dict(orient='records'),
        environment=dict(python=platform.python_version(),packages={p:version(p) for p in ['numpy','pandas','scikit-learn']}),
        file_sha256=files))


def validate_approval(receipt,root=ROOT):
    request=receipt.get('request',{})
    if receipt.get('status')!='APPROVED' or request.get('status')!=STATUS:
        raise residual.MethodologyPendingError('Separate clipped-residual P4 approval PENDING_APPROVAL')
    if request.get('methodology')!=PROPOSAL or receipt.get('request_sha256')!=semantic_hash(request):raise ValueError('P4 method/request digest mismatch')
    for field in ['approved_by','approval_evidence','approval_communicated_date']:
        if not isinstance(receipt.get(field),str) or not receipt[field].strip():raise ValueError('Missing P4 approval evidence')
    datetime.strptime(receipt['approval_communicated_date'],'%Y-%m-%d')
    verify_frozen_files({'file_sha256':request['file_sha256'],'input':{}})
    if request!=make_request(root):raise ValueError('P4 frozen request drift')


def _calibrate(panel, expected, folds, audit):
    """Pure synthetic-test arithmetic; official callers must use run_approved()."""
    reproduction.validate_forecasts(panel,expected,folds)
    observed=panel[['fold','config','model','selected_parameters']].drop_duplicates()
    check=observed.merge(audit[['fold','config','model','selected_parameters']],on=['fold','config'],
                         how='outer',validate='1:1',suffixes=('_observed','_audit'),indicator=True)
    if (not check['_merge'].eq('both').all() or not check.model_observed.eq(check.model_audit).all()
            or not check.selected_parameters_observed.eq(check.selected_parameters_audit).all()):
        raise ValueError('Selected model/parameters audit mapping mismatch')
    frames={}
    for split in ['validation','test']:
        frame=panel.loc[panel.split.eq(split)].copy()
        frame['y_pred']=frame.clipped_prediction  # deliberately clipped for BOTH splits
        frame['score_mask']=True;frame['forecast_origin']=frame.date-pd.Timedelta(days=1);frame['horizon_days']=1
        frames[split]=frame
    # model is an invariant within fold/config. Keeping it as a guard in the
    # shared residual API produces exactly the declared fold/config pooling.
    adjuster=residual.fit_residual_adjuster(frames['validation'],residual.ResidualSpec(('family',),'linear',30,((),)),folds,QUANTILE_LEVELS)
    q=adjuster.predict(frames['test'].drop(columns='y_true'))
    q=q.merge(frames['test'][residual.POINT_KEY+['y_true']],on=residual.POINT_KEY,validate='m:1')
    metadata=audit.drop(columns=['model'])
    q=q.merge(metadata,on=['fold','config'],validate='m:1')
    cal=adjuster.calibration.merge(metadata,on=['fold','config'],validate='m:1')
    q['policy']='P4';q['methodology_status']=STATUS;q['base_prediction_column']='clipped_prediction'
    return q,cal


def _evaluate(panel, expected, folds, audit, baseline, direct, config):
    test=expected.loc[expected.split.eq('test')];validate_fixed_costs(test,baseline,direct,config)
    q,cal=_calibrate(panel,expected,folds,audit);rows=[];pairs=[];records=[]
    for s in configured_scenarios(config):
        base=baseline.loc[baseline.scenario.eq(s['scenario']),MATCH+['y_true','simulated_cost']].rename(columns={'y_true':'y_true_P1','simulated_cost':'cost_P1'})
        for cfg in reproduction.CONFIGS:
            g=q.loc[q.config.eq(cfg)&q.tau_key.eq(s['tau_key'])].copy()
            _matched_test(test,g)
            g['scenario']=s['scenario'];g['c_over'],g['c_under']=s['c_over'],s['c_under']
            g['simulated_cost']=newsvendor_cost(g.y_true,g.quantity,s['c_over'],s['c_under'])
            loss=pinball_loss(g.y_true,g.quantity,s['tau'])
            if not np.allclose(g.simulated_cost,(s['c_over']+s['c_under'])*loss,rtol=1e-12,atol=0):raise ValueError('Cost-pinball identity failed')
            g['overstock_units']=np.maximum(g.quantity-g.y_true,0);g['understock_units']=np.maximum(g.y_true-g.quantity,0)
            g['result_label']=LABEL;rows.append(g)
            for variant in ['corrected','raw_clipped_sensitivity']:
                pair=g[MATCH+['y_true','simulated_cost']].rename(columns={'simulated_cost':'cost_P4'}).merge(base,on=MATCH,validate='1:1',how='outer')
                for policy in ['P2','P3']:
                    fixed=direct.loc[direct.scenario.eq(s['scenario'])&direct.policy.eq(policy)&direct.variant.eq(variant),MATCH+['y_true','simulated_cost']]
                    pair=pair.merge(fixed.rename(columns={'y_true':'y_true_'+policy,'simulated_cost':'cost_'+policy}),on=MATCH,how='outer',validate='1:1')
                if pair.isna().any().any() or any(not np.array_equal(pair.y_true,pair['y_true_'+p]) for p in ['P1','P2','P3']):raise ValueError('Matched policy targets/keys mismatch')
                pair['config'],pair['variant'],pair['scenario']=cfg,variant,s['scenario'];pairs.append(pair)
                for level,grouping in LEVELS.items():
                    groups=[((),pair)] if not grouping else pair.groupby(grouping,observed=True,sort=True)
                    for key,block in groups:
                        key=key if isinstance(key,tuple) else (key,)
                        r=dict(zip(grouping,key),config=cfg,scenario=s['scenario'],variant=variant,report_level=level,sample_size=len(block),result_label=LABEL)
                        for policy in ['P1','P2','P3','P4']:
                            r['mean_cost_'+policy]=stable_mean(block['cost_'+policy].to_numpy())
                            r['cost_index_'+policy]=cost_index(block['cost_'+policy],block.cost_P1)
                        r['p1_denominator_zero']=not block.cost_P1.gt(0).any();records.append(r)
    return dict(p4_rows=pd.concat(rows,ignore_index=True),calibration=cal,matched_costs=pd.concat(pairs,ignore_index=True),comparisons=pd.DataFrame(records))


OUTPUTS={'p4_rows.csv','calibration.csv','matched_costs.csv','comparisons.csv','STARTED.json','manifest.json'}


def inspect_bundle(directory):
    directory=Path(directory)
    if (directory/'FAILED.json').exists():return 'failed'
    try:
        marker=json.loads((directory/'COMPLETE.json').read_text())
        if set(marker['files'])!=OUTPUTS or any(file_hash(directory/n)!=h for n,h in marker['files'].items()):return 'corrupt'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['output_sha256']!={n:h for n,h in marker['files'].items() if n!='manifest.json'}:return 'corrupt'
        return 'complete'
    except (OSError,ValueError,KeyError,TypeError):return 'incomplete'


def run_approved(approval_path,root=ROOT):
    root=Path(root).resolve();approval_path=Path(approval_path).resolve();receipt=json.loads(approval_path.read_text())
    validate_approval(receipt,root);digest=file_hash(approval_path);sources=load_sources(root)
    parent=root/'results/experiments';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/('m4_p4_costs_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=parent))
    reproduction._json(stage/'STARTED.json',dict(approval=receipt,approval_sha256=digest,result_label=LABEL))
    try:
        result=_evaluate(sources['panel'],sources['expected'],sources['folds'],sources['audit'],sources['baseline'],sources['direct'],sources['config'])
        for name,frame in result.items():
            frame['methodology_status']='APPROVED';frame['approval_sha256']=digest
            if name=='p4_rows':frame['interpretation']=np.where(frame.fold.eq('F1'),'exploratory/post-pilot','exposure verification pending')
            with (stage/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
            dates=[c for c in frame if pd.api.types.is_datetime64_any_dtype(frame[c])]
            loaded=pd.read_csv(stage/(name+'.csv'),parse_dates=dates or None,dtype={'store_nbr':str} if 'store_nbr' in frame else None,float_precision='round_trip')
            pd.testing.assert_frame_equal(loaded,frame.reset_index(drop=True),check_dtype=False)
        validate_approval(receipt,root)
        if file_hash(approval_path)!=digest:raise ValueError('Approval changed during evaluation')
        reproduction._json(stage/'manifest.json',dict(approval=receipt,result_label=LABEL,calibration_guarantees='none',
            output_sha256={n:file_hash(stage/n) for n in OUTPUTS-{'manifest.json'}}))
        reproduction._json(stage/'COMPLETE.json',{'files':{n:file_hash(stage/n) for n in OUTPUTS}})
        if inspect_bundle(stage)!='complete':raise ValueError('Staged P4 integrity failed')
        if destination.exists():raise FileExistsError('Never overwrite prior runs')
        stage.rename(destination);return destination
    except Exception as e:
        reproduction._json(stage/'FAILED.json',{'type':type(e).__name__,'message':str(e)});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-request',type=Path);group.add_argument('--approval',type=Path);args=parser.parse_args()
    if args.prepare_request:reproduction._json(args.prepare_request,make_request())
    else:print(run_approved(args.approval))


if __name__=='__main__':main()
