#!/usr/bin/env python3
"""Render current fitting/generalization curves; never read the test set."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'runs/fgn_capacity_v1'


def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    artifacts = HERE / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    histories = {}
    summaries = []
    for layers in cfg['generator_layers']:
        path = OUT / f'layers_{layers}/log.jsonl'
        if not path.exists():
            continue
        records = []
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        values = [row for row in records if row.get('event') == 'eval']
        starts = [row for row in records if row.get('event') == 'start']
        if not values:
            continue
        histories[layers] = values
        best = min(values, key=lambda row: row['dev']['all']['nll'])
        last = values[-1]
        detected = OUT / f'layers_{layers}/overfit.json'
        summaries.append(dict(layers=layers, parameters=starts[-1]['generator_parameters'],
                              completed_epochs=last['epoch'], best_epoch=best['epoch'],
                              best_dev_nll=best['dev']['all']['nll'],
                              last_train_nll=last['train_probe']['all']['nll'], last_dev_nll=last['dev']['all']['nll'],
                              last_train_objective=last['train_probe']['all']['objective'],
                              last_dev_objective=last['dev']['all']['objective'],
                              last_online_objective=last.get('online_training_objective'),
                              overfitting=json.loads(detected.read_text()) if detected.exists() else None))
    matched = None
    if len(histories) == len(cfg['generator_layers']):
        epoch = min(values[-1]['epoch'] for values in histories.values())
        matched = dict(epoch=epoch, models={})
        for layers, values in histories.items():
            row = next(item for item in values if item['epoch'] == epoch)
            matched['models'][str(layers)] = dict(train_nll=row['train_probe']['all']['nll'],
                                                 dev_nll=row['dev']['all']['nll'],
                                                 train_objective=row['train_probe']['all']['objective'],
                                                 dev_objective=row['dev']['all']['objective'],
                                                 online_objective=row.get('online_training_objective'))
    (artifacts / 'summary.json').write_text(json.dumps(dict(scope=cfg['scope'], models=summaries,
                                                          matched_epoch_comparison=matched, test_used=False), indent=2)+'\n')
    if not histories:
        print('No completed evaluation points yet.')
        return
    with (artifacts / 'learning_curves.csv').open('w', newline='') as stream:
        fields=['layers','epoch','train_nll','dev_nll','dev_clean_nll','dev_other_nll','train_objective','dev_objective','online_objective','train_ce','dev_ce','train_crps','dev_crps']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for layers, values in histories.items():
            for row in values:
                train=row['train_probe']['all'];dev=row['dev']
                writer.writerow(dict(layers=layers,epoch=row['epoch'],train_nll=train['nll'],dev_nll=dev['all']['nll'],
                                     dev_clean_nll=dev['dev-clean']['nll'],dev_other_nll=dev['dev-other']['nll'],
                                     train_objective=train['objective'],dev_objective=dev['all']['objective'],
                                     online_objective=row.get('online_training_objective'),
                                     train_ce=train['ce'],dev_ce=dev['all']['ce'],
                                     train_crps=train['crps'],dev_crps=dev['all']['crps']))
    fig,axes=plt.subplots(3,len(histories),figsize=(max(8,5*len(histories)),10),squeeze=False,sharex=True,sharey='row')
    for col,(layers,values) in enumerate(histories.items()):
        epochs=[row['epoch'] for row in values]
        for group,key,label in [('train_probe','all','Fixed train probe'),('dev','dev-clean','dev-clean'),('dev','dev-other','dev-other')]:
            axes[0,col].plot(epochs,[row[group][key]['nll'] for row in values],'.-',label=label)
        parameters = next(value['parameters'] for value in summaries if value['layers'] == layers)
        axes[0,col].set(title=f'FGN {layers} blocks\n{parameters/1e6:.2f}M generator parameters',ylabel='Unsmoothed token NLL')
        for group,label in [('train_probe','Fixed train probe'),('dev','Mixed dev')]:
            axes[1,col].plot(epochs,[row[group]['all']['objective'] for row in values],'.-',label=label)
        online = [row for row in values if 'online_training_objective' in row]
        axes[1,col].plot([row['epoch'] for row in online],
                         [row['online_training_objective'] for row in online],
                         '--',label='Online batch-mean training loss')
        epsilon=cfg['label_smoothing'];correct=1-epsilon+epsilon/500;other=epsilon/500
        floor=-correct*math.log(correct)-499*other*math.log(other)
        axes[1,col].axhline(floor,color='gray',linestyle=':',label='Smoothed-CE lower bound')
        axes[1,col].set(xlabel='Continuation epoch',ylabel='CE + 0.1 CRPS')
        for group,label in [('train_probe','Fixed train probe'),('dev','Mixed dev')]:
            axes[2,col].plot(epochs,[row[group]['all']['crps'] for row in values],'.-',label=label)
        axes[2,col].set(xlabel='Continuation epoch',ylabel='Acoustic-state fair CRPS')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('FGN generator capacity: fixed cases, H2, identical initial function\nTeacher-forced train/dev probes; no test evaluation',fontsize=12)
    fig.tight_layout();fig.savefig(artifacts/'learning_curves.png',dpi=160,bbox_inches='tight');plt.close(fig)
    lines=['# FGN capacity study — current status','','Fixed-case fitting diagnostic; no test use. Residual correction/encoder skipping is not implemented.','',
           '| FGN blocks | Generator parameters | Epochs | Best epoch | Best dev NLL | Last train-probe NLL | Last dev NLL | Overfit confirmed |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for value in summaries:
        lines.append(f"| {value['layers']} | {value['parameters']:,} | {value['completed_epochs']} | {value['best_epoch']} | {value['best_dev_nll']:.4f} | {value['last_train_nll']:.4f} | {value['last_dev_nll']:.4f} | {'Yes' if value['overfitting'] else 'No'} |")
    if matched:
        lines += ['', f"Matched-epoch comparison at epoch {matched['epoch']}:", '',
                  '| FGN blocks | Train-probe NLL | Dev NLL | Train-probe objective | Dev objective |',
                  '|---|---:|---:|---:|---:|']
        for layers,value in matched['models'].items():
            lines.append(f"| {layers} | {value['train_nll']:.4f} | {value['dev_nll']:.4f} | {value['train_objective']:.4f} | {value['dev_objective']:.4f} |")
    lines+=['','A completed stage without the predeclared pattern is not evidence of overfitting. Online loss is the unweighted mean of training-batch objectives during each epoch; fixed-probe objectives use evaluation mode after that epoch. They differ in both examples and model state. Neither is unsmoothed token NLL.','',
            '![Current learning curves](learning_curves.png)']
    (artifacts/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'models_reported':len(summaries),'completed_epochs':{v['layers']:v['completed_epochs'] for v in summaries}}),flush=True)


if __name__=='__main__':main()
