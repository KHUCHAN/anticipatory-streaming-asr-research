#!/usr/bin/env python3
"""Verify completed capacity runs against frozen inputs, logs and checkpoints."""
import json
from collections import Counter
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from study import OUT, get_data, make_model, overfit_status, sha, write


def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    provenance = json.loads((HERE / 'artifacts/provenance_initial.json').read_text())
    # Reporting/runner versions are recorded separately; the mathematical
    # training implementation and all recorded inputs must remain unchanged.
    training_sources = [name for name in provenance['source_sha256'] if not name.endswith(('/report.py', '/run.py'))]
    for name in training_sources:
        assert sha(ROOT / name) == provenance['source_sha256'][name], name
    for name, expected in provenance['input_sha256'].items():
        assert sha(ROOT / name) == expected, name
    train, dev, probe = get_data(cfg)
    train_ids = {row['utterance_id'] for row in train.rows}
    dev_ids = {row['utterance_id'] for row in dev.rows}
    probe_ids = {row['utterance_id'] for row in probe.rows}
    assert len(train_ids) == len(train.rows) == 28539
    assert len(dev_ids) == len(dev.rows) == cfg['dev_selection_utterances']
    assert len(probe_ids) == len(probe.rows) == cfg['train_probe_utterances']
    assert probe_ids <= train_ids and not train_ids & dev_ids
    assert all(row['split'] == cfg['train_split'] for row in train.rows)
    assert all(row['split'] in {'dev-clean', 'dev-other'} for row in dev.rows)
    coverage = [index for group in train.batches(cfg) for index in group]
    assert sorted(coverage) == list(range(len(train.rows)))
    roles = {}
    for role in ['model_selection', 'gate_fit', 'gate_calibration']:
        rows = [json.loads(line) for line in (ROOT / cfg['gate_protocol'] / f'{role}.jsonl').read_text().splitlines()]
        roles[role] = {row['utterance_id'].split('-')[0] for row in rows}
        if role == 'model_selection':
            assert dev_ids <= {row['utterance_id'] for row in rows}
    assert not roles['model_selection'] & (roles['gate_fit'] | roles['gate_calibration'])
    data_audit = dict(train_utterances=len(train.rows), cases_per_utterance=1,
                      train_batches_per_epoch=len(train.batches(cfg)),
                      each_training_utterance_once_per_epoch=True,
                      train_probe_utterances=len(probe.rows),
                      dev_split_counts=dict(Counter(row['split'] for row in dev.rows)),
                      selected_dev_role='model_selection', gate_speaker_overlap=False,
                      train_dev_overlap=False, no_test_rows_in_training_or_probes=True)
    del train, dev, probe
    results = []
    for layers in cfg['generator_layers']:
        out = OUT / f'layers_{layers}'
        if not (out / 'last.pt').exists():
            results.append(dict(layers=layers, status='not completed'))
            continue
        records = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
        if records[-1].get('event') not in {'stage_complete', 'overfit_confirmed', 'already_overfit'}:
            results.append(dict(layers=layers, status='running; deferred to avoid checkpoint/log race'))
            continue
        history = [row for row in records if row.get('event') == 'eval']
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        assert last['cfg'] == best['cfg'] == cfg
        assert [row['epoch'] for row in history] == list(range(last['epoch'] + 1))
        assert best['epoch'] == min(history, key=lambda row: row['dev']['all']['nll'])['epoch']
        assert abs(best['dev']['all']['nll'] - min(row['dev']['all']['nll'] for row in history)) < 1e-12
        assert last['dev'] == history[-1]['dev']
        assert last['train_probe'] == history[-1]['train_probe']
        assert last['step'] == last['epoch'] * next(row for row in records if row['event'] == 'start')['steps_per_epoch']
        assert last['source_sha256']['study.py'] == sha(HERE / 'study.py')
        assert last['source_sha256']['config.json'] == sha(HERE / 'config.json')
        model = make_model(layers, cfg)
        model.load_state_dict(last['model'])
        count = sum(value.numel() for value in model.state.parameters())
        assert count == next(row for row in records if row['event'] == 'start')['generator_parameters']
        declared = json.loads((out / 'overfit.json').read_text()) if (out / 'overfit.json').exists() else None
        assert declared == overfit_status(history, cfg)
        added_norms = [float(block.ff[-1].weight.detach().norm()) for block in model.state.blocks[4:]]
        assert all(value > 0 for value in added_norms), 'Added blocks remained inactive'
        results.append(dict(layers=layers, completed_epochs=last['epoch'], best_epoch=best['epoch'],
                            generator_parameters=count, best_sha256=sha(out / 'best.pt'),
                            last_sha256=sha(out / 'last.pt'), overfitting=declared,
                            added_block_output_norms=added_norms,
                            status='verified through saved checkpoint'))
        del model, last, best
    value = dict(training_sources_unchanged=True, recorded_inputs_unchanged=True,
                 data_audit=data_audit, models=results, test_used=False,
                 scope='Checks saved checkpoints, not any currently running unsaved partial epoch.')
    write(HERE / 'artifacts/verification.json', value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
