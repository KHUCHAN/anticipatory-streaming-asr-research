#!/usr/bin/env python3
"""Write completed finite-window findings without declaring universal absence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    cfg = json.loads((HERE/'config.json').read_text())
    summary = json.loads((HERE/'artifacts/summary.json').read_text())
    verified = json.loads((HERE/'artifacts/verification.json').read_text())
    runner = [json.loads(line) for line in (ROOT/cfg['output']/'runner.jsonl').read_text().splitlines()]
    assert runner[-1]['event']=='done' and summary['status']=='completed'
    assert len(verified['models'])==3 and all(row['status']=='verified' and row['epoch']==cfg['target_epoch'] for row in verified['models'])
    histories = {}
    trajectories = []
    for layers in cfg['generator_layers']:
        rows = [json.loads(line) for line in (ROOT/cfg['output']/f'layers_{layers}/log.jsonl').read_text().splitlines()]
        history = [row for row in rows if row.get('event')=='eval']
        histories[layers] = history
        assert [row['epoch'] for row in history] == list(range(cfg['target_epoch']+1))
        initial,at27,last = history[0],history[cfg['resume_epoch']],history[-1]
        peak = max(history,key=lambda row:row['dev']['all']['crps'])
        minimum = min(history,key=lambda row:row['dev']['all']['crps'])
        selected = next(row for row in summary['models'] if row['layers']==layers)
        assert selected['completed_epochs'] == last['epoch']
        assert selected['last_dev_nll'] == last['dev']['all']['nll']
        windows = summary['predeclared_window_means'][str(layers)]
        assert all(row['complete'] for row in windows)
        trajectories.append(dict(layers=layers,parameters=selected['parameters'],
            dev_nll_at27=at27['dev']['all']['nll'],dev_nll_final=last['dev']['all']['nll'],
            dev_hit1_at27=at27['dev']['all']['hit1'],dev_hit1_final=last['dev']['all']['hit1'],
            state_crps_initial=initial['dev']['all']['crps'],state_crps_peak=peak['dev']['all']['crps'],
            state_crps_peak_epoch=peak['epoch'],state_crps_final=last['dev']['all']['crps'],
            state_crps_minimum=minimum['dev']['all']['crps'],state_crps_minimum_epoch=minimum['epoch'],
            dev_nll_window_means_increase=all(a['dev_nll_mean']<b['dev_nll_mean'] for a,b in zip(windows,windows[1:])),
            dev_crps_window_means_decrease=all(a['dev_crps_mean']>b['dev_crps_mean'] for a,b in zip(windows,windows[1:]))))
    assert all(row['dev_nll_window_means_increase'] for row in trajectories)
    assert all(row['dev_crps_window_means_decrease'] for row in trajectories)
    conclusion = ('All three generators completed epoch 96. Token validation NLL rises across every predeclared late window for each model; no sustained second token-risk descent is observed in this horizon. '
                  'State CRPS shows late recovery in every model, and the 6.32M final score is slightly below its shared trained epoch-0 score. '
                  'That component-specific shape does not establish an interpolation-driven mechanism or improved held-out token prediction.')
    result = dict(status='completed',scope=cfg['scope'],models=summary['models'],
        predeclared_window_means=summary['predeclared_window_means'],
        full_training_measurements=summary['full_training_measurements'],
        added_epochs=cfg['additional_epochs']*3,elapsed_seconds=runner[-1]['elapsed_seconds'],
        verification_passed=True,test_used=False,
        reviewed_date='2026-11-14',trajectory_comparison=trajectories,conclusion=conclusion,
        reporting_source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['report.py','verify.py','analyze.py']})
    result['input_conflicts'] = json.loads((HERE/'artifacts/input_conflicts.json').read_text())
    (HERE/'artifacts/completion.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# FGN epoch-wise double-descent investigation — 2026-11-13','',
        'Final results reviewed on 2026-11-14, America/Los_Angeles. The filename retains the experiment launch date.','',
        conclusion,'',
        f"Completed {cfg['additional_epochs']} additional epochs for each generator, from {cfg['resume_epoch']} through {cfg['target_epoch']}, without early stopping. Sequential runner time: {result['elapsed_seconds']/3600:.2f} hours. Original model/AdamW states were restored; the prior experiment and best checkpoints remain unchanged.",'',
        '## Token risk and checkpoint selection','',
        '| FGN blocks | Generator parameters | Last train-probe NLL | Last dev NLL | Selected best dev NLL | Best epoch |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['models']:
        lines.append(f"| {row['layers']} | {row['parameters']:,} | {row['last_train_nll']:.5f} | {row['last_dev_nll']:.5f} | {row['best_dev_nll']:.5f} | {row['best_epoch']} |")
    lines += ['','## Changes from epoch 27 to 96','',
        '| Generator parameters | Dev token NLL: 27 → 96 | Dev next-token Hit@1: 27 → 96 | Dev state CRPS: initial → peak → final |',
        '|---|---:|---:|---:|']
    for row in trajectories:
        lines.append(f"| {row['parameters']:,} | {row['dev_nll_at27']:.5f} → {row['dev_nll_final']:.5f} | {100*row['dev_hit1_at27']:.2f}% → {100*row['dev_hit1_final']:.2f}% | {row['state_crps_initial']:.5f} → {row['state_crps_peak']:.5f} → {row['state_crps_final']:.5f} |")
    lines += ['','These are reference-prefix next-token probe metrics, not ASR WER. The probes contain 512 fixed cases. Epoch 0 is an already-trained shared function, not random initialization. The small model reaches its minimum dev state CRPS of 0.49808 at epoch 90; no statistical significance is established for that small difference from the initial function. A second descent does not need to beat the first minimum.','',
        '## Declared epoch-window means','',
        'Window means are descriptive, single-seed summaries. They do not supply a significance test or identify a double-descent mechanism. A second decline need not outperform the first minimum.','',
        '| Blocks | Window | Mean dev NLL | Mean dev CRPS |','|---|---|---:|---:|']
    for layers,values in summary['predeclared_window_means'].items():
        for row in values:
            assert row['complete']
            lines.append(f"| {layers} | {row['epoch_window'][0]}–{row['epoch_window'][1]} | {row['dev_nll_mean']:.5f} | {row['dev_crps_mean']:.5f} |")
    lines += ['','## Full fixed-training-set diagnostics','',
        '| Blocks | Epoch | Cases | Hard errors | Token NLL | State CRPS | Minimum log-probability margin |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for layers,values in summary['full_training_measurements'].items():
        for row in values:
            lines.append(f"| {layers} | {row['epoch']} | {row['examples']} | {row['errors']} | {row['nll']:.5f} | {row['crps']:.5f} | {row['target_vs_best_other_logp_margin']['minimum']:.5f} |")
    lines += ['','Final arrived-audio training errors out of 28,318 cases are 23 / 19 / 24 for 4 / 8 / 16 blocks, versus 70 / 69 / 68 at epoch 27. Zero-audio errors are 193 in each final fixed-noise pass. Better training fit does not translate into better validation token risk.','']
    lines += ['','All 28,539 fixed training cases are evaluated at milestones. Each measurement uses one fixed-noise two-branch pass; zero errors in one pass would not establish correctness for every noise draw. The input audit finds 221 identical complete no-audio prefixes with conflicting targets, contributing at least 193 expected hard errors under fresh independent functional noise. This expectation is not a bound on every finite pass. Arrived-audio subset errors are retained separately in the numeric evidence. A raw parameter/example ratio, a smoothed-loss floor or a 512-case probe cannot locate the interpolation threshold.','',
        '## Limits and provenance','',
        '- Epochs 0–27 are inherited evidence. Only 28–96 are new training in this namespace.',
        '- This investigates epoch-wise behavior for three existing warm-started depths, with one seed. It is not a broad model-size/width/memory sweep or an independent replication.',
        '- Architecture, functional noise, CE(smoothing 0.1) + 0.1 state CRPS, AdamW learning rate 3e-5, weight decay 0.01 and global epoch seeds remain fixed.',
        '- Token NLL, hard error, state CRPS and joint objective must be interpreted separately. Recovery in one component need not improve another.',
        '- No test evaluation, new AR-only/no-noise control, final ASR WER, raw-audio correction or compute-acceleration claim.',
        '- Verification checks saved model/optimizer/snapshot epochs, source hashes, original-run preservation and full-training milestone counts. Finite-window findings do not prove eventual presence or absence of double descent.','',
        '![Compact metric comparison](../experiments/fgn_double_descent_v1/artifacts/double_descent_summary.png)','',
        '![Train/dev curves](../experiments/fgn_double_descent_v1/artifacts/learning_curves.png)','',
        '![Full-training diagnostics](../experiments/fgn_double_descent_v1/artifacts/full_training_curves.png)','',
        '[Protocol](../experiments/fgn_double_descent_v1/README.md) · [Earlier review](double_descent_review_2026-11-13.md) · [Deep Double Descent](https://arxiv.org/abs/1912.02292)']
    (ROOT/'docs/fgn_double_descent_2026-11-13.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed',added_epochs=result['added_epochs'],verification_passed=True)))


if __name__=='__main__':
    main()
