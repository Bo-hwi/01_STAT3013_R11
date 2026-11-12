"""Separate P4 residual proposal. No implied P4 approval from P1 or reproduction."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import p4_reproduction as reproduction
from . import p1_residual_policy as residual
from .quantile_contracts import file_hash, semantic_hash, verify_frozen_files, json_snapshot
from .quantile_models import QUANTILE_LEVELS

STATUS='PENDING_APPROVAL'
PROPOSAL=dict(policy='P4',residual='y_true - raw_validation_prediction',
              pooling=['fold','selected_model','config','family'],quantile_method='linear',minimum_samples=30,
              fallback=['fold','selected_model','config'],
              quantity='max(0, clipped_refitted_test_prediction + validation_residual_quantile)',
              limitations=['validation reused for tuning/algorithm selection and calibration',
                           'train-only residuals applied to a different train+validation refitted predictor',
                           'no independent calibration/conformal guarantee'])


def calibration_request(directory):
    directory=Path(directory).resolve()
    if reproduction.inspect_bundle(directory)!='complete':raise ValueError('Complete reproduction bundle required')
    return json_snapshot(dict(status=STATUS,methodology=PROPOSAL,source_bundle=str(directory),
        file_sha256={**{str(p):file_hash(p) for p in directory.iterdir() if p.is_file()},
            str(Path(__file__).resolve()):file_hash(__file__),
            str(Path(residual.__file__).resolve()):file_hash(residual.__file__),
            str(Path(reproduction.__file__).resolve()):file_hash(reproduction.__file__)}))


def validate_approval(receipt):
    request=receipt.get('request',{})
    if receipt.get('status')!='APPROVED' or request.get('status')!=STATUS:
        raise residual.MethodologyPendingError('Separate P4 calibration approval PENDING_APPROVAL')
    if request.get('methodology')!=PROPOSAL or receipt.get('request_sha256')!=semantic_hash(request):raise ValueError('P4 method/digest mismatch')
    if request!=calibration_request(request['source_bundle']):raise ValueError('P4 source/code changed')
    for key in ['approved_by','approval_evidence','approval_communicated_date']:
        if not isinstance(receipt.get(key),str) or not receipt[key].strip():raise ValueError('Missing P4 approval evidence')
    datetime.strptime(receipt['approval_communicated_date'],'%Y-%m-%d')
    verify_frozen_files({'file_sha256':request['file_sha256'],'input':{}})
    return request


def _calibrate(panel, fold_specs):
    """Internal arithmetic for synthetic tests; no official execution authorization."""
    frames={}
    for split in ['validation','test']:
        p=panel.loc[panel.split.eq(split)].copy()
        p['y_pred']=p['raw_prediction' if split=='validation' else 'clipped_prediction']
        p['score_mask']=True;p['forecast_origin']=pd.to_datetime(p.date)-pd.Timedelta(days=1);p['horizon_days']=1
        frames[split]=p
    adjuster=residual.fit_residual_adjuster(frames['validation'],residual.ResidualSpec(('family',),'linear',30,((),)),fold_specs,QUANTILE_LEVELS)
    quantities=adjuster.predict(frames['test'].drop(columns='y_true'))
    quantities['policy']='P4';quantities['methodology_status']=STATUS
    return quantities,adjuster.calibration


def apply_approved(approval_path):
    """Return approved quantities/audit in memory; no economic output or model fitting."""
    receipt=json.loads(Path(approval_path).read_text());request=validate_approval(receipt)
    directory=Path(request['source_bundle']);manifest=json.loads((directory/'manifest.json').read_text())
    verify_frozen_files({'file_sha256':manifest['file_sha256'],'input':{}})
    verify_frozen_files({'file_sha256':manifest['approval']['request']['file_sha256'],'input':{}})
    panel=pd.read_csv(directory/'predictions.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    expected=pd.read_csv(directory/'expected_rows.csv',parse_dates=['date'],dtype={'store_nbr':str},float_precision='round_trip')
    protocol_path=next(p for p in manifest['approval']['request']['file_sha256'] if Path(p).name=='research_protocol.yaml')
    folds=reproduction.read_yaml(protocol_path)['rq2']['validation']['test_folds']
    reproduction.validate_forecasts(panel,expected,folds)
    quantities,audit=_calibrate(panel,folds)
    validate_approval(receipt)
    for frame in [quantities,audit]:
        frame['methodology_status']='APPROVED';frame['approval_sha256']=file_hash(approval_path)
    return quantities,audit
