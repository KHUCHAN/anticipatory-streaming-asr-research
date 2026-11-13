#!/usr/bin/env python3
"""Write the final longer-training report after all requested epochs finish."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    out = ROOT / cfg['output']
    runner = [json.loads(line) for line in (out / 'runner.jsonl').read_text().splitlines()]
    summary = json.loads((HERE / 'artifacts/summary.json').read_text())
    verification = json.loads((HERE / 'artifacts/verification.json').read_text())
    assert runner[-1]['event'] == 'done' and summary['status'] == 'completed'
    assert len(verification['models']) == len(cfg['generator_layers'])
    assert all(value['status'] == 'verified' and value['epoch'] == cfg['target_epoch'] for value in verification['models'])
    comparison = summary['extension_comparison']
    assert all(value['end_epoch'] == cfg['target_epoch'] for value in comparison)
    counts = dict(train_nll_improved=sum(value['last_train_nll'] < value['start_train_nll'] for value in comparison),
                  dev_nll_improved=sum(value['last_dev_nll'] < value['start_dev_nll'] for value in comparison),
                  dev_crps_improved=sum(value['last_dev_crps'] < value['start_dev_crps'] for value in comparison),
                  dev_crps_better_than_epoch0=sum(value['last_dev_crps'] < value['initial_dev_crps'] for value in comparison),
                  new_best_dev_checkpoints=sum(value['best_epoch'] > cfg['resume_epoch'] for value in summary['models']))
    conclusion = (f"From epoch {cfg['resume_epoch']} to {cfg['target_epoch']}, train-probe NLL improved in {counts['train_nll_improved']}/3 models, "
                  f"dev NLL improved in {counts['dev_nll_improved']}/3, and dev acoustic-state CRPS improved in {counts['dev_crps_improved']}/3. "
                  f"{counts['new_best_dev_checkpoints']}/3 models selected a new best-dev-NLL checkpoint during the extension.")
    final_train = [value['last_train_nll'] for value in comparison]
    interpretation = (f"Observed final train-probe NLLs span {min(final_train):.4f}–{max(final_train):.4f} across a threefold generator-size range. "
                      'Additional epochs improve fitting, while validation token likelihood and accuracy worsen in these runs. '
                      'The small generator already reaches similar fitting values; this is not statistical equivalence or a test of wider generators, larger memory or the full distribution of temporal cutoffs.')
    result = dict(status='completed', target_epoch=cfg['target_epoch'],
                  additional_epochs_per_model=cfg['additional_epochs'], additional_epochs_total=cfg['additional_epochs']*len(comparison),
                  runner_elapsed_seconds=runner[-1]['elapsed_seconds'],
                  conclusion=conclusion, improvement_counts=counts,
                  interpretation=interpretation,
                  comparison=comparison, models=summary['models'], verification_passed=True, test_used=False,
                  reporting_source_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                           for name in ['report.py', 'verify.py', 'analyze.py']})
    (HERE / 'artifacts/completion.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Longer FGN training — 2026-11-13', '',
             conclusion, '',
             interpretation, '',
             f"Completed {cfg['additional_epochs']} additional epochs per model, extending the prior capacity experiment from epoch {cfg['resume_epoch']} to {cfg['target_epoch']}. All three models restored their exact saved model and AdamW states. Overfit observations were recorded without stopping. Original experiments and best checkpoints remain unchanged.", '',
             f"Sequential extension runner time: {runner[-1]['elapsed_seconds']/3600:.2f} hours. This excludes the earlier 33 training epochs and prior feature/checkpoint preparation. No test evaluation or final-WER claim.", '',
             '## Before and after the extra epochs', '',
             '| Generator parameters | Train-probe NLL: epoch 11 → 27 | Dev NLL: epoch 11 → 27 | Best dev NLL | Best epoch |',
             '|---|---:|---:|---:|---:|']
    for value in comparison:
        selected = next(item for item in summary['models'] if item['layers'] == value['layers'])
        lines.append(f"| {selected['parameters']:,} | {value['start_train_nll']:.4f} → {value['last_train_nll']:.4f} | {value['start_dev_nll']:.4f} → {value['last_dev_nll']:.4f} | {selected['best_dev_nll']:.4f} | {selected['best_epoch']} |")
    lines += ['', 'All final comparisons share the same epoch budget. Larger models use more parameters and compute; this is not a fixed-compute scaling comparison.', '',
              '## Auxiliary probe accuracy', '',
              '| FGN blocks | Train-probe Hit@1: epoch 11 → 27 | Dev Hit@1: epoch 11 → 27 |',
              '|---|---:|---:|']
    for value in comparison:
        lines.append(f"| {value['layers']} | {100*value['start_train_hit1']:.2f}% → {100*value['last_train_hit1']:.2f}% | {100*value['start_dev_hit1']:.2f}% → {100*value['last_dev_hit1']:.2f}% |")
    lines += ['', 'These are teacher-forced next-token probe accuracies, not final ASR WER. NLL also depends on probability assigned to the target, so a change in NLL is not the same quantity as a change in token accuracy. Checkpoint selection remains fixed to dev NLL.', '',
              '## Acoustic-state prediction', '',
              '| FGN blocks | Train-probe CRPS: epoch 0 → 11 → 27 | Dev CRPS: epoch 0 → 11 → 27 |',
              '|---|---:|---:|']
    for value in comparison:
        lines.append(f"| {value['layers']} | {value['initial_train_crps']:.5f} → {value['start_train_crps']:.5f} → {value['last_train_crps']:.5f} | {value['initial_dev_crps']:.5f} → {value['start_dev_crps']:.5f} → {value['last_dev_crps']:.5f} |")
    lines += ['', f"Epoch 0 is the shared, already-trained function before the capacity study, not random initialization. {counts['dev_crps_better_than_epoch0']}/3 models have lower final dev CRPS than that function. Recovery relative to epoch 11 does not necessarily establish an improvement relative to epoch 0.", '',
              'Token likelihood, state CRPS and optimized loss must be read separately. The objective remains branch-mean token CE with label smoothing 0.1 + 0.1 state CRPS. A falling token NLL alone does not demonstrate better acoustic representations.', '',
              '## Settings and scope', '',
              '- FGN uses SiLU, with hidden dimension 256, FFN dimension 1,024, four heads and 256-dimensional GRU memory. Its output is a linear residual with no terminal ReLU.',
              '- The generator has 4 / 8 / 16 blocks; AR remains four layers. Width and memory are fixed, so these runs do not assess width or memory scaling. Two acoustic forecast steps and two noise branches; new 32-dimensional noise at each step.',
              '- AdamW states continue from epoch 11, learning rate 3e-5, weight decay 0.01, gradient clip 5. No activation, objective, sampling or learning-rate change accompanies the extra epochs.',
              '- One fixed token/cutoff per each of 28,539 train-clean-100 utterances; fixed train/dev probes of 512 each. Teacher-forced prefixes and one seed limit the conclusion to this fitting diagnostic.',
              '- These data do not establish a causal effect of functional noise, final recognition quality, useful residual correction or lower inference cost.',
              '- No AR-only or noise-free control receives the extra epochs here. This is an epoch-extension diagnosis of the same stochastic hybrids, not a new matched-budget FGN-versus-AR comparison.', '',
              '## Verification and provenance', '',
              '- Restored-model train/dev NLL agrees with each saved source checkpoint within 1e-5 before updates; optimizer step counters continue from 4,521.',
              '- All requested epochs, best-dev selection, final checkpoint metrics, global step counts and original input/source/checkpoint hashes passed verification.',
              '- Epochs 0–11 in the figure are inherited evidence. Only epochs 12–27 are new training.',
              '- Public aggregates and settings are in `experiments/fgn_capacity_long_v1/artifacts/`; raw weights and logs remain local-only in `runs/fgn_capacity_long_v1/`.', '',
              '![Extended learning curves](../experiments/fgn_capacity_long_v1/artifacts/learning_curves.png)', '',
              '[Protocol](../experiments/fgn_capacity_long_v1/README.md) · [Original capacity results](fgn_capacity_2026-11-12.md)']
    (ROOT / 'docs/fgn_capacity_long_2026-11-13.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],additional_epochs=result['additional_epochs_total'],verification_passed=True)))


if __name__ == '__main__':
    main()
