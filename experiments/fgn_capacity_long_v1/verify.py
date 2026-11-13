#!/usr/bin/env python3
"""Audit additional epochs, exact resume metadata and unchanged original runs."""
from collections import Counter
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from train import base, configuration


def main():
    extension, cfg = configuration()
    settings = json.loads((HERE / 'artifacts/settings.json').read_text())
    assert settings['optimizer']['learning_rate'] == cfg['learning_rate']
    assert settings['optimizer']['weight_decay'] == cfg['weight_decay']
    assert settings['loss']['label_smoothing'] == cfg['label_smoothing']
    assert settings['loss']['state_weight'] == cfg['crps_weight']
    provenance = json.loads((HERE / 'artifacts/provenance.json').read_text())
    for name, expected in provenance['sha256'].items():
        assert base.sha(ROOT / name) == expected, name
    train, dev, probe = base.get_data(cfg)
    coverage = [index for group in train.batches(cfg) for index in group]
    assert sorted(coverage) == list(range(len(train.rows)))
    assert len(train.rows) == 28539 and len(dev.rows) == len(probe.rows) == 512
    assert all(row['split'] == 'train-clean-100' for row in train.rows)
    assert all(row['split'] in {'dev-clean', 'dev-other'} for row in dev.rows)
    assert not {row['utterance_id'] for row in train.rows} & {row['utterance_id'] for row in dev.rows}
    data = dict(train_utterances=len(train.rows), each_utterance_once_per_epoch=True,
                fixed_cases_per_utterance=1, train_probe_utterances=len(probe.rows),
                dev_split_counts=dict(Counter(row['split'] for row in dev.rows)), test_used=False)
    del train, dev, probe
    models = []
    for layers in extension['generator_layers']:
        out = ROOT / extension['output'] / f'layers_{layers}'
        original = ROOT / extension['base_runs'] / f'layers_{layers}'
        original_text = (original / 'log.jsonl').read_text()
        text = (out / 'log.jsonl').read_text()
        assert text.startswith(original_text), 'Original history changed during import'
        rows = [json.loads(line) for line in text.splitlines()]
        if rows[-1]['event'] != 'continuation_complete':
            models.append(dict(layers=layers, status='pending or running; verification deferred'))
            continue
        history = [row for row in rows if row.get('event') == 'eval']
        assert [row['epoch'] for row in history] == list(range(extension['target_epoch'] + 1))
        resumes = [row for row in rows[len(original_text.splitlines()):] if row['event'] == 'start']
        assert resumes and resumes[0]['resumed_from_epoch'] == extension['resume_epoch']
        expected_source = str((original / 'last.pt').relative_to(ROOT))
        assert resumes[0]['source_checkpoint'] == expected_source
        assert resumes[0]['source_checkpoint_sha256'] == provenance['sha256'][expected_source]
        assert all(row['resumed_nll_max_difference'] < 1e-5 for row in resumes)
        assert all(row['optimizer_step_range'] == [row['resumed_from_epoch'] * 411] * 2 for row in resumes)
        assert all(not row['stop_on_overfit'] for row in resumes)
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        assert last['cfg'] == best['cfg'] == cfg
        assert last['continuation_cfg'] == extension
        assert last['epoch'] == extension['target_epoch']
        assert last['step'] == extension['target_epoch'] * 411
        optimizer_steps = [int(value['step']) for value in last['optimizer']['state'].values() if 'step' in value]
        assert optimizer_steps and min(optimizer_steps) == max(optimizer_steps) == last['step']
        for group in last['optimizer']['param_groups']:
            assert group['lr'] == settings['optimizer']['learning_rate']
            assert group['weight_decay'] == settings['optimizer']['weight_decay']
            assert list(group['betas']) == settings['optimizer']['betas']
        assert last['evaluation_record'] == history[-1]
        assert last['dev'] == history[-1]['dev'] and last['train_probe'] == history[-1]['train_probe']
        optimum = min(history, key=lambda row: row['dev']['all']['nll'])
        assert best['epoch'] == optimum['epoch']
        assert best['dev'] == optimum['dev']
        for name, expected in last['source_sha256'].items():
            assert base.sha(ROOT / name) == expected, name
        best_original_preserved = base.sha(original / 'best.pt') == provenance['sha256'][str((original / 'best.pt').relative_to(ROOT))]
        assert best_original_preserved
        if best['epoch'] <= extension['resume_epoch']:
            assert base.sha(out / 'best.pt') == base.sha(original / 'best.pt')
        model = base.make_model(layers, cfg)
        model.load_state_dict(last['model'])
        assert len(model.state.blocks) == layers
        assert all(type(block.ff[1]).__name__ == 'SiLU' for block in model.state.blocks)
        assert not any(isinstance(module, torch.nn.ReLU) for module in model.state.modules())
        for key in ['hidden_dim', 'ff_dim', 'heads', 'memory_dim', 'noise_dim', 'state_frames', 'state_features']:
            assert model.state.cfg[key] == settings['generator'][key], key
        assert len(model.ar.decoder.layers) == settings['ar']['layers']
        assert type(model.ar.decoder.layers[0].ff[1]).__name__ == settings['ar']['ff_activation']
        parameters = sum(value.numel() for value in model.state.parameters())
        assert parameters == resumes[0]['generator_parameters']
        status = json.loads((out / 'overfit_status.json').read_text())
        assert status['epoch'] == extension['target_epoch']
        assert status['result'] == base.overfit_status(history, cfg)
        models.append(dict(layers=layers, status='verified', epoch=last['epoch'],
                           added_epochs=last['epoch']-extension['resume_epoch'], generator_parameters=parameters,
                           best_epoch=best['epoch'], best_dev_nll=best['dev']['all']['nll'],
                           original_best_preserved=best_original_preserved,
                           settings_match_checkpoint_architecture=True, generator_relu_modules=0,
                           resumed_nll_max_difference=resumes[0]['resumed_nll_max_difference'],
                           final_overfit_rule_met=status['criterion_met'],
                           last_sha256=base.sha(out / 'last.pt'), best_sha256=base.sha(out / 'best.pt')))
        del model, last, best
    result = dict(captured_sources_and_inputs_unchanged=True, original_runs_unchanged=True,
                  data_audit=data, models=models, test_used=False)
    base.write(HERE / 'artifacts/verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
