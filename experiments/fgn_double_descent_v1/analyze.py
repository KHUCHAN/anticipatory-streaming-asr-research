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
    result = dict(status='completed',scope=cfg['scope'],models=summary['models'],
        predeclared_window_means=summary['predeclared_window_means'],
        full_training_measurements=summary['full_training_measurements'],
        added_epochs=cfg['additional_epochs']*3,elapsed_seconds=runner[-1]['elapsed_seconds'],
        verification_passed=True,test_used=False,
        reporting_source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['report.py','verify.py','analyze.py']})
    result['input_conflicts'] = json.loads((HERE/'artifacts/input_conflicts.json').read_text())
    (HERE/'artifacts/completion.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# FGN epoch-wise double-descent investigation — 2026-11-13','',
        f"Completed {cfg['additional_epochs']} additional epochs for each generator, from {cfg['resume_epoch']} through {cfg['target_epoch']}, without early stopping. Sequential runner time: {result['elapsed_seconds']/3600:.2f} hours. Original model/AdamW states were restored; the prior experiment and best checkpoints remain unchanged.",'',
        '## Token risk and checkpoint selection','',
        '| FGN blocks | Generator parameters | Last train-probe NLL | Last dev NLL | Selected best dev NLL | Best epoch |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['models']:
        lines.append(f"| {row['layers']} | {row['parameters']:,} | {row['last_train_nll']:.5f} | {row['last_dev_nll']:.5f} | {row['best_dev_nll']:.5f} | {row['best_epoch']} |")
    lines += ['','## Declared epoch-window means','',
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
    lines += ['','All 28,539 fixed training cases are evaluated at milestones. Each measurement uses one fixed-noise two-branch pass; zero errors in one pass would not establish correctness for every noise draw. The input audit finds 221 identical complete no-audio prefixes with conflicting targets, contributing at least 193 expected hard errors under fresh independent functional noise. This expectation is not a bound on every finite pass. Arrived-audio subset errors are retained separately in the numeric evidence. A raw parameter/example ratio, a smoothed-loss floor or a 512-case probe cannot locate the interpolation threshold.','',
        '## Limits and provenance','',
        '- Epochs 0–27 are inherited evidence. Only 28–96 are new training in this namespace.',
        '- This investigates epoch-wise behavior for three existing warm-started depths, with one seed. It is not a broad model-size/width/memory sweep or an independent replication.',
        '- Architecture, functional noise, CE(smoothing 0.1) + 0.1 state CRPS, AdamW learning rate 3e-5, weight decay 0.01 and global epoch seeds remain fixed.',
        '- Token NLL, hard error, state CRPS and joint objective must be interpreted separately. Recovery in one component need not improve another.',
        '- No test evaluation, new AR-only/no-noise control, final ASR WER, raw-audio correction or compute-acceleration claim.',
        '- Verification checks saved model/optimizer/snapshot epochs, source hashes, original-run preservation and full-training milestone counts. Finite-window findings do not prove eventual presence or absence of double descent.','',
        '![Train/dev curves](../experiments/fgn_double_descent_v1/artifacts/learning_curves.png)','',
        '![Full-training diagnostics](../experiments/fgn_double_descent_v1/artifacts/full_training_curves.png)','',
        '[Protocol](../experiments/fgn_double_descent_v1/README.md) · [Earlier review](double_descent_review_2026-11-13.md) · [Deep Double Descent](https://arxiv.org/abs/1912.02292)']
    (ROOT/'docs/fgn_double_descent_2026-11-13.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed',added_epochs=result['added_epochs'],verification_passed=True)))


if __name__=='__main__':
    main()
