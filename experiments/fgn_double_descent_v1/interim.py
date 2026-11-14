#!/usr/bin/env python3
"""Snapshot running progress and separate token-risk recovery from state loss."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    raw = path.read_bytes()
    rows = []
    for line in raw.decode().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows, hashlib.sha256(raw).hexdigest()


def main():
    cfg = json.loads((HERE/'config.json').read_text())
    out = ROOT/cfg['output']
    runner,runner_sha = read(out/'runner.jsonl')
    models = []
    added = 0
    for layers in cfg['generator_layers']:
        folder = out/f'layers_{layers}'
        rows,log_sha = read(folder/'log.jsonl')
        history = [row for row in rows if row.get('event')=='eval']
        latest = history[-1]
        best = min(history,key=lambda row:row['dev']['all']['nll'])
        peak = max(history,key=lambda row:row['dev']['all']['crps'])
        start = next(row for row in history if row['epoch']==cfg['resume_epoch'])
        windows = []
        for lo,hi in cfg['late_curve_windows']:
            selected = [row for row in history if lo<=row['epoch']<=hi]
            windows.append(dict(epoch_window=[lo,hi],observed_points=len(selected),complete=len(selected)==hi-lo+1,
                dev_nll_mean=sum(row['dev']['all']['nll'] for row in selected)/len(selected) if selected else None,
                dev_crps_mean=sum(row['dev']['all']['crps'] for row in selected)/len(selected) if selected else None))
        full = sorted([json.loads(path.read_text()) for path in folder.glob('full_train_epoch_*.json')],key=lambda row:row['epoch'])
        added += max(0,latest['epoch']-cfg['resume_epoch'])
        models.append(dict(layers=layers,completed_epoch=latest['epoch'],current_epoch=rows[-1].get('epoch'),
            last_event=rows[-1]['event'],source_log_snapshot_sha256=log_sha,
            start_dev_nll=start['dev']['all']['nll'],last_dev_nll=latest['dev']['all']['nll'],
            last_train_probe_nll=latest['train_probe']['all']['nll'],last_dev_hit1=latest['dev']['all']['hit1'],
            best_dev_nll=best['dev']['all']['nll'],best_dev_epoch=best['epoch'],
            state_crps_peak_epoch=peak['epoch'],state_crps_peak=peak['dev']['all']['crps'],
            last_dev_crps=latest['dev']['all']['crps'],predeclared_window_means=windows,
            full_training_milestones=full))
    result = dict(status='interim running-study snapshot',captured_utc=datetime.now(timezone.utc).isoformat(),
        runner_last_event=runner[-1]['event'],runner_snapshot_sha256=runner_sha,
        additional_epochs_completed=added,additional_epochs_requested=cfg['additional_epochs']*3,
        models=models,test_used=False,
        interpretation='Token NLL and state CRPS are distinct objectives. Window means are descriptive, not significance tests. Component recovery does not identify an interpolation-driven double-descent mechanism. Other models remain incomplete; no universal presence or absence conclusion.')
    (HERE/'artifacts/interim.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Epoch-wise double-descent interim assessment — 2026-11-13','',
        f"Snapshot captured at {result['captured_utc']}. This is an incomplete-study report; the training runner remains active. {added} of {result['additional_epochs_requested']} requested new epochs have completed across the three models. It does not stop or alter training.",'',
        '| Generator blocks | Completed epoch / target | Train-probe NLL | Dev NLL | Best dev NLL (epoch) |',
        '|---|---:|---:|---:|---:|']
    for row in models:
        lines.append(f"| {row['layers']} | {row['completed_epoch']} / {cfg['target_epoch']} | {row['last_train_probe_nll']:.5f} | {row['last_dev_nll']:.5f} | {row['best_dev_nll']:.5f} ({row['best_dev_epoch']}) |")
    lines += ['','## Declared late windows','',
        '| Blocks | Window | Observed points / planned | Mean dev token NLL | Mean dev state CRPS |',
        '|---|---|---:|---:|---:|']
    for row in models:
        for window in row['predeclared_window_means']:
            if not window['observed_points']:
                continue
            lo,hi = window['epoch_window']
            lines.append(f"| {row['layers']} | {lo}–{hi} | {window['observed_points']} / {hi-lo+1} | {window['dev_nll_mean']:.5f} | {window['dev_crps_mean']:.5f} |")
    lines += ['','The completed 16-block run has increasing mean token NLL across the three declared windows, with no sustained second descent visible in that comparison. Its state CRPS recovers after an earlier peak. The other models are still incomplete; this is a metric-specific finding within the observed horizon, not proof of a general double-descent mechanism or its absence.','',
        '## State component and full-training fitting','']
    for row in models:
        lines.append(f"- {row['layers']} blocks: dev CRPS peak {row['state_crps_peak']:.6f} at epoch {row['state_crps_peak_epoch']}; latest {row['last_dev_crps']:.6f}.")
        if row['full_training_milestones']:
            first,last = row['full_training_milestones'][0],row['full_training_milestones'][-1]
            lines.append(f"  Full fixed-training arrived-audio errors: {first['subsets']['arrived_audio']['errors']} at epoch {first['epoch']} → {last['subsets']['arrived_audio']['errors']} at epoch {last['epoch']}, out of {last['subsets']['arrived_audio']['examples']} cases. Zero-audio errors remain separately reported in the numeric snapshot.")
    lines += ['','The fixed training inputs contain contradictory no-audio targets; a conservative expected contribution of 193 hard errors is unavoidable under independent fresh noise. A single fixed-noise pass need not obey that expectation exactly. Full-training fitting and improving state loss do not establish better held-out token prediction. The study uses one seed and has no new final-ASR WER measurement.','',
        '[Protocol](../experiments/fgn_double_descent_v1/README.md) · [Numeric interim snapshot](../experiments/fgn_double_descent_v1/artifacts/interim.json) · [Current complete/incomplete curves](../experiments/fgn_double_descent_v1/artifacts/learning_curves.png)']
    (ROOT/'docs/fgn_double_descent_interim_2026-11-13.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(additional_epochs_completed=added,additional_epochs_requested=result['additional_epochs_requested'],epochs={row['layers']:row['completed_epoch'] for row in models})))


if __name__=='__main__':
    main()
