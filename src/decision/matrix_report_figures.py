"""Layout-corrected standalone report figures; never changes a sealed matrix bundle."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import consolidated_matrix as matrix


def heatmap(data,tag,families,cities,scenarios):
    column='relative_delta_pct_'+tag;limit=float(data[column].abs().max()) or 1.
    fig,axes=plt.subplots(3,1,figsize=(18,14))
    fig.subplots_adjust(left=.17,right=.88,top=.93,bottom=.10,hspace=.70)
    for scenario,ax in zip(scenarios,axes):
        table=data.loc[data.scenario.eq(scenario)].pivot(index='family',columns='city',values=column).reindex(index=families,columns=cities)
        im=ax.imshow(table,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_yticks(range(len(families)),families,fontsize=10)
        ax.set_xticks(range(len(cities)),cities,rotation=55,ha='right',fontsize=9)
        ax.set_title(scenario+' | '+tag.replace('_',' ')+' | negative = lower simulated cost',fontsize=12)
    cax=fig.add_axes([.91,.12,.015,.78]);fig.colorbar(im,cax=cax,label='Relative paired cost delta (%)')
    fig.suptitle('Descriptive weather cost differences; no family-city confidence intervals',y=.98,fontsize=14)
    return fig


def run_figures(source):
    source=Path(source).resolve()
    if matrix.inspect_bundle(source)!='complete':raise ValueError('Complete source matrix bundle required')
    manifest=json.loads((source/'manifest.json').read_text());matrix.verify_frozen_files({'file_sha256':manifest['file_sha256'],'input':{}})
    data=pd.read_csv(source/'matrix.csv',keep_default_na=False,na_values=[''],float_precision='round_trip')
    ranks=pd.read_csv(source/'policy_summary.csv',float_precision='round_trip')
    families=matrix.p4.reproduction.read_yaml(matrix.ROOT/'configs/research_protocol.yaml')['scope']['deep_study_families']
    cities=sorted(data.city.unique());scenarios=['S1','S3','S5']
    matrix.validate_matrix(data,families,cities,scenarios)
    hashes={str(p):matrix.file_hash(p) for p in source.iterdir() if p.is_file()}
    hashes[str(Path(__file__).resolve())]=matrix.file_hash(__file__)
    destination=source.parent/('m4_matrix_report_figures_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=source.parent))
    try:
        for tag in ['P3_minus_P2','P4_M3_minus_M0']:
            fig=heatmap(data,tag,families,cities,scenarios)
            for ext in ['png','svg']:fig.savefig(stage/('weather_'+tag+'.'+ext),dpi=180,bbox_inches='tight',pad_inches=.2)
            plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(16,5));fig.subplots_adjust(left=.06,right=.99,bottom=.23,top=.85,wspace=.3)
        for scenario,ax in zip(scenarios,axes):
            g=ranks.loc[ranks.scenario.eq(scenario)].set_index('policy').reindex(matrix.POLICIES)
            ax.bar(matrix.POLICIES,g.P1_cost_index,color=['gray','#3b82f6','#14b8a6','#a78bfa','#f59e0b']);ax.axhline(1,color='black',lw=.8)
            ax.set_title(scenario);ax.tick_params(axis='x',rotation=40);ax.set_ylabel('Simulated cost index (P1=1)')
        fig.suptitle('Descriptive pooled policy comparison; no deployment or uncertainty claim',y=.97)
        for ext in ['png','svg']:fig.savefig(stage/('policy_comparison.'+ext),dpi=180,bbox_inches='tight',pad_inches=.2)
        plt.close(fig)
        matrix.verify_frozen_files({'file_sha256':hashes,'input':{}})
        with (stage/'README.md').open('x',encoding='utf-8') as f:
            f.write('# Report-ready figure supplement\n\nSource: '+str(source)+'\n\nLayout-only replacement for source figures: visual QA found clipped labels on the original P4 heatmap. Tables, values, classification rules and prior bundle remain unchanged. Use these PNG/SVG figures in the report. No new analysis, bootstrap or model fits.\n')
        outputs={f.name:matrix.file_hash(f) for f in stage.iterdir() if f.is_file()}
        matrix.p4.reproduction._json(stage/'manifest.json',dict(source_matrix_bundle=str(source),file_sha256=hashes,
            purpose='layout-only report figure supplement; no numerical changes',output_sha256=outputs))
        matrix.p4.reproduction._json(stage/'COMPLETE.json',{'files':{**outputs,'manifest.json':matrix.file_hash(stage/'manifest.json')}})
        if matrix.inspect_bundle(stage)!='complete':raise ValueError('Figure supplement integrity failure')
        if destination.exists():raise FileExistsError('Never overwrite report supplements')
        stage.rename(destination);return destination
    except Exception as e:
        matrix.p4.reproduction._json(stage/'FAILED.json',{'type':type(e).__name__,'message':str(e)});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);args=parser.parse_args()
    print(run_figures(args.source))


if __name__=='__main__':main()
