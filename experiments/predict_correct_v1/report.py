#!/usr/bin/env python3
"""Current learning curves and final WER/compute summaries, without retraining."""
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    cfg=json.loads((HERE/'config.json').read_text());out=ROOT/cfg['output'];histories={}
    for variant in cfg['variants']:
        p=out/variant/'log.jsonl'
        if not p.exists():continue
        values=[]
        for line in p.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row.get('event')=='eval':values.append(row)
        if values:histories[variant]=values
    artifacts=HERE/'artifacts';artifacts.mkdir(exist_ok=True)
    if histories:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for variant,values in histories.items():
            x=[r['epoch'] for r in values]
            axes[0].plot(x,[r['selection_wer']['wer_percent'] for r in values],'.-',label=variant)
            axes[1].plot(x,[r['validation']['mae'] for r in values],'.-',label=variant)
            axes[2].plot(x,[r['validation']['kl'] for r in values],'.-',label=variant)
        for axis,title,ylabel in zip(axes,['Free-running checkpoint selection','Held-out random-window representation','Held-out teacher distribution'],['Selection WER (%)','Normalized MAE','Distillation KL']):
            axis.set(title=title,xlabel='Epoch',ylabel=ylabel);axis.grid(alpha=.2);axis.legend()
        fig.suptitle('A: audio only; B: audio + frozen FGN; C: audio + previous corrected state',fontsize=10)
        fig.tight_layout();fig.savefig(artifacts/'learning_curves.png',dpi=160,bbox_inches='tight');plt.close(fig)
    value=dict(status='training in progress',completed_epochs={k:v[-1]['epoch'] for k,v in histories.items()},
        epoch_history=histories,test_used=False)
    if set(histories)==set(cfg['variants']) and all(v[-1]['epoch']==cfg['epochs'] for v in histories.values()):
        value['status']='training complete; evaluation pending or running'
    if (out/'results.json').exists():
        result=json.loads((out/'results.json').read_text());value['status']='evaluation complete';value['primary']=result['primary'];value['paired_comparisons']=result['paired_comparisons'];value['decisions']=result['decisions']
        runner_path=out/'runner.jsonl'
        if runner_path.exists():
            runner=[json.loads(line) for line in runner_path.read_text().splitlines()]
            if runner[-1]['event']=='done':value['status']='completed'
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        colors={'full_encoder':'black','A':'tab:blue','B':'tab:orange','C':'tab:green'}
        for axis,split in zip(axes,['dev-clean','dev-other']):
            for variant,color in colors.items():
                score=result['primary'][f'{variant}/{split}'];axis.scatter(score['total_ms_per_chunk'],score['wer_percent'],color=color,s=60)
                offset={'A':(8,12),'B':(8,4),'C':(8,-18),'full_encoder':(-8,9)}[variant]
                axis.annotate(variant,(score['total_ms_per_chunk'],score['wer_percent']),xytext=offset,textcoords='offset points',
                    ha='right' if variant=='full_encoder' else 'left',arrowprops=dict(arrowstyle='-',lw=.6,color=color))
            axis.set(title=split,xlabel='Total inference ms/chunk (lower is better)',ylabel='WER % (lower is better)');axis.grid(alpha=.2);axis.margins(x=.1,y=.12)
        fig.suptitle('Full fbank + frontend + feedback + free-running decoder cost; CPU batch 1',fontsize=10)
        fig.tight_layout();fig.savefig(artifacts/'wer_compute.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (artifacts/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],completed_epochs=value['completed_epochs'])))


if __name__=='__main__':main()
