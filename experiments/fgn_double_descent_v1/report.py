#!/usr/bin/env python3
"""Render observed long-horizon curves and full-training interpolation diagnostics."""
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def records(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def main():
    cfg = json.loads((HERE/'config.json').read_text())
    out = ROOT/cfg['output']
    spec = importlib.util.spec_from_file_location('dd_curve_renderer',ROOT/cfg['base_experiment']/'report.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.HERE, renderer.OUT = HERE, out
    renderer.main()
    summary = json.loads((HERE/'artifacts/summary.json').read_text())
    runner = records(out/'runner.jsonl')
    full = {str(layers): sorted([json.loads(path.read_text()) for path in (out/f'layers_{layers}').glob('full_train_epoch_*.json')],key=lambda row:row['epoch']) for layers in cfg['generator_layers']}
    windows = {}
    for layers in cfg['generator_layers']:
        history = [row for row in records(out/f'layers_{layers}/log.jsonl') if row.get('event')=='eval']
        values = []
        for lo,hi in cfg['late_curve_windows']:
            selected = [row for row in history if lo<=row['epoch']<=hi]
            values.append(dict(epoch_window=[lo,hi],observed_points=len(selected),complete=len(selected)==hi-lo+1,
                dev_nll_mean=sum(row['dev']['all']['nll'] for row in selected)/len(selected) if selected else None,
                dev_crps_mean=sum(row['dev']['all']['crps'] for row in selected)/len(selected) if selected else None))
        windows[str(layers)] = values
    summary.update(status='completed' if runner[-1]['event']=='done' else 'in progress',
        inherited_epoch_window=[0,cfg['resume_epoch']],new_epoch_window=[cfg['resume_epoch']+1,cfg['target_epoch']],
        full_training_measurements=full,predeclared_window_means=windows,
        scope=cfg['scope'],test_used=False)
    (HERE/'artifacts/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    heading = ('# Epoch-wise double-descent investigation\n\n'+
               f"Status: {summary['status']}. Epochs 0–27 are inherited; only epochs 28–96 are newly trained here. "
               'No overfit-based stopping. Three warm-started depths and one seed do not establish a model-wise double-descent curve.\n\n')
    path = HERE/'artifacts/summary.md'
    path.write_text(heading+path.read_text())
    if any(full.values()):
        fig,axes = plt.subplots(3,3,figsize=(14,10),sharex=True,sharey='row')
        for column,layers in enumerate(cfg['generator_layers']):
            measurements = full[str(layers)]
            epochs = [row['epoch'] for row in measurements]
            axes[0,column].plot(epochs,[row['nll'] for row in measurements],'.-')
            axes[0,column].set(title=f'FGN {layers} blocks',ylabel='Full fixed-train token NLL')
            axes[1,column].plot(epochs,[100*row['errors']/row['examples'] for row in measurements],'.-',label='All fixed cases')
            axes[1,column].plot(epochs,[100*row['subsets']['arrived_audio']['errors']/row['subsets']['arrived_audio']['examples'] for row in measurements],'.-',label='Arrived-audio cases')
            axes[1,column].legend(fontsize=8)
            axes[1,column].set(ylabel='Hard next-token error (%)')
            for key,label in [('minimum','Minimum'),('p05','5th percentile'),('median','Median')]:
                axes[2,column].plot(epochs,[row['target_vs_best_other_logp_margin'][key] for row in measurements],'.-',label=label)
            axes[2,column].axhline(0,color='gray',linestyle=':')
            axes[2,column].set(xlabel='Continuation epoch',ylabel='Target minus best-other log probability')
            axes[2,column].legend(fontsize=8)
            for axis in axes[:,column]:axis.grid(alpha=.2)
        fig.suptitle('Whole fixed-training-set diagnostics: 28,539 cases\nOne fixed-noise two-branch pass per milestone; no test evaluation')
        fig.tight_layout()
        fig.savefig(HERE/'artifacts/full_training_curves.png',dpi=160,bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(dict(status=summary['status'],completed_epochs={row['layers']:row['completed_epochs'] for row in summary['models']},full_training_milestones={key:[row['epoch'] for row in values] for key,values in full.items()})))


if __name__=='__main__':
    main()
