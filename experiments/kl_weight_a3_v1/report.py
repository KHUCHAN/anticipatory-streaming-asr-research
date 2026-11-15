#!/usr/bin/env python3
"""Write matched-loss evidence and learning curves without changing training."""
import datetime
import json
import os
from zoneinfo import ZoneInfo

from common import ROOT, HERE, sha, write

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = ROOT / 'runs/kl_weight_a3_v1'


def main():
    study = json.loads((HERE / 'config.json').read_text())
    histories = {}
    for name in ['control', *study['training_arms']]:
        p = ROOT / ('runs/predict_correct_a2_v1/kl/log.jsonl' if name == 'control' else f'runs/kl_weight_a3_v1/{name}/kl/log.jsonl')
        if p.exists():
            histories[name] = [r for r in [json.loads(s) for s in p.read_text().splitlines()] if r['event'] == 'eval']
    histories = {k: v for k, v in histories.items() if v}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {'control': 'black', 'kl4': 'tab:blue', 'kl16': 'tab:orange'}
    for name, rows in histories.items():
        x = [r['epoch'] for r in rows]
        for key, axis in [('mae', axes[0, 0]), ('kl', axes[0, 1])]:
            axis.plot(x, [r['train_' + key] for r in rows], '--', color=colors[name], label=name + ' train')
            axis.plot(x, [r['validation'][key] for r in rows], '-', color=colors[name], label=name + ' selection')
        for split, style in [('dev-clean', '-'), ('dev-other', '--')]:
            axes[1, 0].plot(x, [r['selection'][split]['wer_percent'] for r in rows], style, color=colors[name], label=name + ' ' + split)
        axes[1, 1].plot(x, [r['gradient']['weighted_kl_to_mae_ratio'] for r in rows], color=colors[name], label=name)
    for axis, title, label in zip(axes.flat,
            ['Representation MAE', 'Teacher-distribution KL', 'Free-running selection WER', 'Fixed training-probe gradient balance'],
            ['Normalized MAE', 'KL', 'WER %', 'Weighted KL / MAE gradient norm']):
        axis.set(title=title, xlabel='Continuation epoch', ylabel=label)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[1, 1].set_yscale('log')
    fig.tight_layout()
    (HERE / 'artifacts').mkdir(exist_ok=True)
    fig.savefig(HERE / 'artifacts/learning_curves.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    result_path = RUN / 'results.json'
    summary = dict(status='training or evaluation in progress', completed_epochs={k: len(v) for k, v in histories.items()}, epoch_history=histories, test_used=False)
    if not result_path.exists():
        write(HERE / 'artifacts/summary.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'epoch_history'}))
        return
    result = json.loads(result_path.read_text())
    verified = json.loads((HERE / 'artifacts/verification.json').read_text())
    assert verified['status'] == 'verified'
    runner = [json.loads(s) for s in (RUN / 'runner.jsonl').read_text().splitlines()]
    assert runner[-1]['event'] == 'done'
    elapsed = runner[-1]['elapsed_seconds']
    chosen = result['selected_candidate']
    summary.update(status='completed', results=result, verification=verified)
    write(HERE / 'artifacts/summary.json', summary)
    completion = dict(status='completed', completed_local=datetime.datetime.fromtimestamp(runner[0]['started_unix'] + elapsed, ZoneInfo('America/Los_Angeles')).isoformat(),
        elapsed_seconds=elapsed, selected_candidate=chosen, verification=verified,
        report_source_sha256=sha(HERE / 'report.py'), test_used=False)
    write(HERE / 'artifacts/completion.json', completion)
    selected = result['primary'][f'{chosen}/dev-clean']
    q = result['qualification'][chosen]
    lines = ['# A3 matched KL-weight comparison — 2026-11-15', '',
        f"Selected arm: **{chosen}**, chosen on the original model-selection role before final decoding. Final dev-clean WER **{selected['wer_percent']:.3f}%**; complete cost **{100*q['same_run_cost_fraction']:.2f}%** of the same-run teacher. The joint <10% WER / <=50% cost gate {'passes' if q['passes_both'] else 'fails'}.", '',
        f"Two new 24-epoch continuations plus a disposable 100-update control replay, evaluation and verification completed in {elapsed/3600:.2f} runner hours. The original A2 24-epoch arm is reused as control. One seed; 400 dev-clean + 400 dev-other final-role utterances; no test split. Completion: {completion['completed_local']}.", '',
        '## Final WER and complete CPU cost', '',
        '| Path | Clean WER % | Other WER % | Clean ms/chunk | Other ms/chunk |', '|---|---:|---:|---:|---:|']
    for name in ['teacher', 'small', 'control', 'kl4', 'kl16']:
        a, b = [result['primary'][f'{name}/{s}'] for s in ['dev-clean', 'dev-other']]
        lines.append(f"| {name} | {a['wer_percent']:.3f} | {b['wer_percent']:.3f} | {a['total_ms_per_chunk']:.3f} | {b['total_ms_per_chunk']:.3f} |")
    lines += ['', 'Small uses the repaired, model-specific blank-filled initial context. Full teacher and A2/A3 retain their original decoder convention. Original A2 control token sequences reproduce on all 800 utterances. Small input availability is 60ms shorter (39 versus 45 feature frames), so this is not an exactly matched-lookahead architecture comparison.', '',
        '## Fixed training conditions and treatment', '',
        'All arms start from the same A2 epoch-12 MAE checkpoint. Architecture, seed 237, sampled windows/order, 24 epochs, fresh AdamW, 3e-4 schedule, clipping, temperature 2, blank weight 0.1, two-epoch KL ramp and checkpoint rule are unchanged. Only the KL coefficient in `MAE + lambda * KL` changes. Native teacher contexts supervise the loss; inference is free-running and sees no teacher encoder state or reference transcript.', '',
        '| Arm | KL multiplier | Final coefficient | Selected epoch | Selection clean WER % |', '|---|---:|---:|---:|---:|']
    base_weight = json.loads((ROOT / study['base_calibration']).read_text())['weight']
    for name, m in study['arms'].items():
        s = result['model_selection'][name]
        lines.append(f"| {name} | {m:g} | {base_weight*m:.8f} | {s['epoch']} | {s['dev_clean_wer']:.3f} |")
    lines += ['', 'The first 100 control updates were replayed with disposable weights and verified against the original logged MAE/KL/coefficient values before training. Six implementation files are byte-identical to A2. Original source/input hashes, both full epoch histories and selected checkpoints pass verification. The training-only gradient probe is diagnostic, not a complete reconstruction of every update.', '',
        '## Paired WER differences', '', '| Split | Comparison | Difference pp | Paired 95% CI |', '|---|---|---:|---|']
    for key, value in result['paired_comparisons'].items():
        split, comparison = key.split('/')
        lo, hi = value['ci95_pp']
        lines.append(f"| {split} | {comparison} | {value['wer_difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', 'Intervals are paired utterance bootstrap intervals conditional on the selected single-seed checkpoints; they do not include seed variability or correct for an exhaustive hyperparameter search. These development roles have been used previously in this project and are not untouched external tests.', '',
        '## Runtime scope and limits', '', result['timing_scope'], '',
        'All neural paths use FP32 ONNX on Apple M5 Pro 24GiB. A2/A3 exports change neither weights nor precision and pass random-stream numerical checks plus eight real selection-utterance token checks per arm. The teacher encoder is absent from student inference. These are processing costs, not microphone/display latency, mobile power or adaptive-fallback measurements.', '',
        'This isolates two declared coefficient changes within the tested recipe. It does not establish a general limit on 100h distillation, automatically justify 960h scaling, or evaluate FGN/cascade acceleration. No reference-supervised transducer objective or frame-weight treatment was added.', '',
        '![Matched learning curves](../experiments/kl_weight_a3_v1/artifacts/learning_curves.png)', '',
        '[Protocol](../experiments/kl_weight_a3_v1/README.md) · [Aggregate results](../experiments/kl_weight_a3_v1/artifacts/results.json) · [20M repair](small_zipformer_reproduction_2026-11-15.md) · [A2](predict_correct_a2_2026-11-15.md)']
    (ROOT / 'docs/kl_weight_a3_2026-11-15.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(completion))


if __name__ == '__main__':
    main()
