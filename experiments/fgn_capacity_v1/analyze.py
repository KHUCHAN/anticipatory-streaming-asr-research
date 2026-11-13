#!/usr/bin/env python3
"""Write the completed capacity diagnosis without training or using test data."""
import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'runs/fgn_capacity_v1'


def main():
    runner = [json.loads(line) for line in (OUT / 'runner.jsonl').read_text().splitlines()]
    assert runner[-1]['event'] == 'done', 'Study is still running or failed; no completion report.'
    cfg = json.loads((HERE / 'config.json').read_text())
    summary = json.loads((HERE / 'artifacts/summary.json').read_text())
    verification = json.loads((HERE / 'artifacts/verification.json').read_text())
    assert len(verification['models']) == len(cfg['generator_layers'])
    assert all(value['status'] == 'verified through saved checkpoint' for value in verification['models'])
    assert len(summary['models']) == len(cfg['generator_layers'])
    for value in summary['models']:
        verified = next(item for item in verification['models'] if item['layers'] == value['layers'])
        assert verified['completed_epochs'] == value['completed_epochs']
        assert verified['best_epoch'] == value['best_epoch']
    matched = summary['matched_epoch_comparison']
    assert matched is not None
    histories = {}
    for layers in cfg['generator_layers']:
        rows = [json.loads(line) for line in (OUT / f'layers_{layers}/log.jsonl').read_text().splitlines()]
        histories[layers] = [row for row in rows if row.get('event') == 'eval']
    initial = histories[cfg['generator_layers'][0]][0]
    initial_nll = initial['dev']['all']['nll']
    assert all(abs(values[0]['dev']['all']['nll'] - initial_nll) < 1e-5 for values in histories.values())
    elapsed = runner[-1]['elapsed_seconds']
    winner = min(summary['models'], key=lambda value: value['best_dev_nll'])
    base_layers, largest_layers = cfg['generator_layers'][0], cfg['generator_layers'][-1]
    base = matched['models'][str(base_layers)]
    largest = matched['models'][str(largest_layers)]
    base_summary = next(value for value in summary['models'] if value['layers'] == base_layers)
    overfit_count = sum(bool(value['overfitting']) for value in summary['models'])
    conclusion = (f"{overfit_count} of {len(summary['models'])} capacities met the declared overfit rule. "
                  f"At the common epoch {matched['epoch']}, moving from {base_layers} to {largest_layers} generator blocks "
                  f"changes train-probe NLL from {base['train_nll']:.5f} to {largest['train_nll']:.5f} "
                  f"and dev NLL from {base['dev_nll']:.5f} to {largest['dev_nll']:.5f}.")
    if base_summary['overfitting']:
        conclusion += ' The original 6.32M generator can already overfit this fixed-case task; increasing capacity is not a prerequisite for observing that pattern.'
    result = dict(status='completed', reason=runner[-1]['reason'], elapsed_seconds=elapsed,
                  completed_epochs=sum(value['completed_epochs'] for value in summary['models']),
                  models=summary['models'], matched_epoch_comparison=matched,
                  lowest_observed_dev_nll_generator_layers=winner['layers'],
                  initial_dev_nll=initial_nll, verification_passed=True, test_used=False,
                  conclusion=conclusion,
                  final_reporting_source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                                 for name in ['report.py', 'verify.py', 'analyze.py']},
                  interpretation='Fixed-case single-seed depth/capacity diagnosis. No final-WER, encoder-acceleration or residual-corrector result.')
    (HERE / 'artifacts/completion.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# FGN generator capacity and overfitting — 2026-11-12', '',
             conclusion, '',
             'Completed the user-requested capacity diagnosis on the existing AR+FGN hybrid. This enlarges the generator itself, not AR depth. The raw-audio corrector and encoder-skipping system remain unimplemented.', '',
             f"Training ran for {elapsed/3600:.2f} hours from runner start through its completed stage, excluding prior cached-feature preparation and benchmarks. Total completed continuation epochs: {result['completed_epochs']}. No test evaluation or test-driven checkpoint selection.", '',
             '## Best and final checkpoints', '',
             '| FGN blocks | Generator parameters | Epochs completed | Best dev epoch | Best dev NLL | Last train-probe NLL | Last dev NLL | Overfit rule met |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for value in summary['models']:
        lines.append(f"| {value['layers']} | {value['parameters']:,} | {value['completed_epochs']} | {value['best_epoch']} | {value['best_dev_nll']:.4f} | {value['last_train_nll']:.4f} | {value['last_dev_nll']:.4f} | {'Yes' if value['overfitting'] else 'No'} |")
    if len({value['completed_epochs'] for value in summary['models']}) == 1:
        last_budget_note = f"All capacities completed {matched['epoch']} continuation epochs. Their last-checkpoint columns therefore share an epoch budget, although parameter count and wall-clock/compute cost differ."
    else:
        last_budget_note = 'Model-specific last epochs differ because of early stopping. Use the common epoch below for a matched-epoch comparison.'
    lines += ['', last_budget_note, '',
              f"## Matched comparison at epoch {matched['epoch']}", '',
              '| FGN blocks | Train-probe NLL | Dev NLL | Train-probe objective | Dev objective |',
              '|---|---:|---:|---:|---:|']
    for layers, value in matched['models'].items():
        lines.append(f"| {layers} | {value['train_nll']:.4f} | {value['dev_nll']:.4f} | {value['train_objective']:.4f} | {value['dev_objective']:.4f} |")
    lines += ['', 'Observed capacity differences at this common budget:', '']
    for layers in cfg['generator_layers'][1:]:
        value = matched['models'][str(layers)]
        lines.append(f"- {layers} versus 4 generator blocks: train-probe NLL {value['train_nll']-base['train_nll']:+.5f}; dev NLL {value['dev_nll']-base['dev_nll']:+.5f}; evaluation-mode train-probe objective {value['train_objective']-base['train_objective']:+.5f}.")
    lines += ['', f"The lowest observed selected dev NLL is {winner['best_dev_nll']:.5f} with {winner['layers']} generator blocks. This is a descriptive single-seed minimum on model-selection data, not a statistically established architecture winner.", '',
              '## Acoustic-state loss', '',
              '| FGN blocks | Initial train-probe CRPS | Last train-probe CRPS | Initial dev CRPS | Last dev CRPS |',
              '|---|---:|---:|---:|---:|']
    for layers, values in histories.items():
        first, last = values[0], values[-1]
        lines.append(f"| {layers} | {first['train_probe']['all']['crps']:.5f} | {last['train_probe']['all']['crps']:.5f} | {first['dev']['all']['crps']:.5f} | {last['dev']['all']['crps']:.5f} |")
    lines += ['', 'Token likelihood and acoustic-state forecast loss must be assessed separately. The optimized weighted sum can improve while the state component worsens. A lower token NLL does not establish a more useful predicted acoustic representation for a future corrector.', '',
              '## What the curves establish', '',
              '- Larger nested function classes can attain an equal or lower global empirical minimum, but the finite optimizer need not realize that advantage.',
              '- The predeclared overfit rule requires five successive held-out NLL regressions of at least 0.02 and training-probe improvements of at least 0.01 relative to the dev-optimal checkpoint, after at least eight epochs. A limit or a plateau does not count.',
              '- The optimized objective is smoothed token CE + 0.1 state CRPS. The figures separately show unsmoothed NLL, that objective and state CRPS. Improvement in one component does not establish improvement in another.',
              '- Online training loss is an unweighted average of training-batch objectives while weights change during the epoch. Fixed-probe objectives are measured in evaluation mode after the epoch. Their numerical difference is not a generalization gap on identical inputs.', '',
              '## Task and limitations', '',
              '- All 28,539 train-clean-100 utterances contribute one fixed token/cutoff case. This deliberately tests fitting a fixed problem, not every possible prefix in 100 hours. Do not directly compare absolute values with the earlier random-cutoff/curriculum curves.',
              '- Train and model-selection dev probes each have 512 cases. Reference prefixes are teacher-forced; these NLLs are not final ASR WER or recognized-prefix acceptance.',
              '- AR remains four layers and is jointly trainable. Only generator depth changes: hidden size, memory size, noise dimension, data and optimization settings are held fixed. This is not a study of every possible width/capacity scaling strategy.',
              '- Every model starts from the same trained function through identity initialization of added residual blocks. Best and last checkpoints are preserved separately.',
              '- Label smoothing 0.1 over 500 classes gives a CE lower bound of approximately 0.94466 nats. Zero optimized loss is not an appropriate target. Separately, identical no-audio prefixes have conflicting labels; see the input-ambiguity diagnostic.',
              '- One seed, development-based stopping and a fixed-case task limit generalization claims. No independent test or final-WER claim is made.',
              '- These results do not demonstrate efficient raw-audio correction, valid cache recovery or lower inference cost. Those remain the next architecture question.', '',
              '## Verification and files', '',
              '- Training sources and recorded input hashes match the pre-result capture.',
              '- Checkpoint epochs, best-dev selection, full epoch step counts and the declared overfit events match the logs.',
              '- Added blocks have nonzero trained output weights; initial checks establish identical logits and nonzero gradients.',
              '- Weights and logs remain local in `runs/fgn_capacity_v1/`. Public aggregates are in `experiments/fgn_capacity_v1/artifacts/`.', '',
              '![Learning curves](../experiments/fgn_capacity_v1/artifacts/learning_curves.png)', '',
              '[Protocol and commands](../experiments/fgn_capacity_v1/README.md) · [Current research direction](predict_correct_acceleration_plan_2026-11-12.md) · [Experiment index](../experiments/README.md)']
    (ROOT / 'docs/fgn_capacity_2026-11-12.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
