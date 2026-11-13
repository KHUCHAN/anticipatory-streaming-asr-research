#!/usr/bin/env python3
"""Reuse the unchanged aggregate renderer and append extension-specific evidence."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    spec = importlib.util.spec_from_file_location('capacity_aggregate_renderer', ROOT / cfg['base_experiment'] / 'report.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.HERE = HERE
    renderer.OUT = ROOT / cfg['output']
    renderer.main()
    artifacts = HERE / 'artifacts'
    summary = json.loads((artifacts / 'summary.json').read_text())
    runner = [json.loads(line) for line in (ROOT / cfg['output'] / 'runner.jsonl').read_text().splitlines()]
    summary.update(status='completed' if runner[-1]['event'] == 'done' else 'in progress',
                   inherited_epochs=[0, cfg['resume_epoch']],
                   requested_additional_epochs=cfg['additional_epochs'], target_epoch=cfg['target_epoch'],
                   early_stopping_disabled=True)
    comparisons = []
    for layers in cfg['generator_layers']:
        records = [json.loads(line) for line in (ROOT / cfg['output'] / f'layers_{layers}/log.jsonl').read_text().splitlines()]
        history = [row for row in records if row.get('event') == 'eval']
        initial = history[0]
        first = next(row for row in history if row['epoch'] == cfg['resume_epoch'])
        last = history[-1]
        comparisons.append(dict(layers=layers, start_epoch=first['epoch'], end_epoch=last['epoch'],
                                initial_train_nll=initial['train_probe']['all']['nll'], initial_dev_nll=initial['dev']['all']['nll'],
                                initial_train_crps=initial['train_probe']['all']['crps'], initial_dev_crps=initial['dev']['all']['crps'],
                                start_train_nll=first['train_probe']['all']['nll'], last_train_nll=last['train_probe']['all']['nll'],
                                start_dev_nll=first['dev']['all']['nll'], last_dev_nll=last['dev']['all']['nll'],
                                start_train_hit1=first['train_probe']['all']['hit1'], last_train_hit1=last['train_probe']['all']['hit1'],
                                start_dev_hit1=first['dev']['all']['hit1'], last_dev_hit1=last['dev']['all']['hit1'],
                                start_train_crps=first['train_probe']['all']['crps'], last_train_crps=last['train_probe']['all']['crps'],
                                start_dev_crps=first['dev']['all']['crps'], last_dev_crps=last['dev']['all']['crps']))
    summary['extension_comparison'] = comparisons
    (artifacts / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    heading = (f"# Longer FGN training — {summary['status']}\n\n"
               f"Epochs 0–{cfg['resume_epoch']} are copied read-only evidence from the completed capacity study. "
               f"Only epochs {cfg['resume_epoch']+1}–{cfg['target_epoch']} are the requested additional training. "
               "The original best/last checkpoints remain preserved. Overfit observations do not stop this extension. "
               "The prior confirmation marker means the rule was met before extension; it is not a new stopping event.\n\n")
    body = (artifacts / 'summary.md').read_text()
    (artifacts / 'summary.md').write_text(heading+body)
    print(json.dumps(dict(status=summary['status'],epochs={value['layers']:value['end_epoch'] for value in comparisons})))


if __name__ == '__main__':
    main()
