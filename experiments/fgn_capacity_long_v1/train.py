#!/usr/bin/env python3
"""Continue the frozen capacity protocol without stopping on overfit events."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/fgn_capacity_v1'))
import study as base


def configuration():
    extension = json.loads((HERE / 'config.json').read_text())
    cfg = json.loads((ROOT / extension['base_experiment'] / 'config.json').read_text())
    assert extension['target_epoch'] == extension['resume_epoch'] + extension['additional_epochs']
    assert not extension['stop_on_overfit'] and not extension['test_used']
    assert all(extension[key] == cfg[key] for key in ['generator_layers', 'label_smoothing', 'crps_weight'])
    return extension, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, required=True)
    args = parser.parse_args()
    extension, cfg = configuration()
    assert args.layers in extension['generator_layers']
    torch.set_num_threads(1)
    original = ROOT / extension['base_runs'] / f'layers_{args.layers}'
    out = ROOT / extension['output'] / f'layers_{args.layers}'
    out.mkdir(parents=True, exist_ok=True)
    lease = (out / 'training.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'continuation_config.json').exists():
        assert json.loads((out / 'continuation_config.json').read_text()) == extension
    else:
        base.write(out / 'continuation_config.json', extension)
        base.write(out / 'config.json', cfg)
    if not (out / 'log.jsonl').exists():
        shutil.copy2(original / 'log.jsonl', out / 'log.jsonl')
    if not (out / 'best.pt').exists():
        shutil.copy2(original / 'best.pt', out / 'best.pt')
    records = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
    history = [row for row in records if row.get('event') == 'eval']
    resumed = (out / 'last.pt').exists()
    source = out / 'last.pt' if resumed else original / 'last.pt'
    checkpoint = torch.load(source, map_location='mps', weights_only=False)
    assert checkpoint['cfg'] == cfg
    assert checkpoint['epoch'] >= extension['resume_epoch']
    if not resumed:
        assert checkpoint['epoch'] == extension['resume_epoch']
    # A checkpoint is atomically written before its log line. Recover that
    # completed evaluation after an interruption in the small intervening gap.
    if checkpoint['epoch'] == history[-1]['epoch'] + 1 and 'evaluation_record' in checkpoint:
        with (out / 'log.jsonl').open('a') as stream:
            stream.write(json.dumps(checkpoint['evaluation_record']) + '\n')
        history.append(checkpoint['evaluation_record'])
    assert checkpoint['epoch'] == history[-1]['epoch']
    log = (out / 'log.jsonl').open('a', buffering=1)
    def emit(value):
        print(json.dumps(value), flush=True)
        log.write(json.dumps(value) + '\n')
    if checkpoint['epoch'] >= extension['target_epoch']:
        emit(dict(event='continuation_complete', completed_epoch=checkpoint['epoch'], already_completed=True))
        return
    model = base.make_model(args.layers, cfg).to('mps')
    model.load_state_dict(checkpoint['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], betas=(.9, .98), weight_decay=cfg['weight_decay'])
    opt.load_state_dict(checkpoint['optimizer'])
    step, first = checkpoint['step'], checkpoint['epoch']
    assert step == first * 411
    optimizer_steps = [int(value['step']) for value in opt.state.values() if 'step' in value]
    assert optimizer_steps and min(optimizer_steps) == max(optimizer_steps) == step
    train, dev, probe = base.get_data(cfg)
    resumed_dev = base.evaluate(model, dev, cfg, 'mps')
    resumed_probe = base.evaluate(model, probe, cfg, 'mps')
    delta = max(abs(resumed_dev['all']['nll'] - checkpoint['dev']['all']['nll']),
                abs(resumed_probe['all']['nll'] - checkpoint['train_probe']['all']['nll']))
    assert delta < 1e-5, f'Resumed predictions changed before updates: {delta}'
    emit(dict(event='start', layers=args.layers, resumed_from_epoch=first,
              target_epoch=extension['target_epoch'], stop_on_overfit=False,
              generator_parameters=sum(p.numel() for p in model.state.parameters()),
              total_parameters=sum(p.numel() for p in model.parameters()),
              source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=base.sha(source),
              optimizer_step_range=[min(optimizer_steps), max(optimizer_steps)],
              resumed_nll_max_difference=delta, train_utterances=len(train.rows),
              steps_per_epoch=len(train.batches(cfg))))
    del checkpoint
    best = min(row['dev']['all']['nll'] for row in history)
    source_hashes = {str(p.relative_to(ROOT)): base.sha(p) for p in [HERE / 'train.py', HERE / 'config.json',
                     ROOT / extension['base_experiment'] / 'study.py', ROOT / extension['base_experiment'] / 'config.json']}
    for epoch in range(first + 1, extension['target_epoch'] + 1):
        started = time.time()
        torch.manual_seed(cfg['seed'] * 1000 + epoch)
        model.train()
        running = []
        for indices in train.batches(cfg, 8100 + epoch):
            batch = train.collate(indices, None, 'mps')
            factor = min(1., (step + 1) / cfg['warmup_steps'])
            for group in opt.param_groups:
                group['lr'] = cfg['learning_rate'] * factor
            opt.zero_grad(set_to_none=True)
            loss, parts = base.objective(model, batch, cfg, cfg['forecast_horizon'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite gradient')
            opt.step()
            step += 1
            running.append([float(loss.detach()), float(parts['ce']), float(parts['crps'])])
            if step % 100 == 0:
                memory = torch.mps.driver_allocated_memory() / 2**30
                if memory > cfg['max_driver_memory_gib']:
                    raise RuntimeError('Predeclared MPS memory limit exceeded')
                emit(dict(event='train', epoch=epoch, step=step,
                          objective=sum(value[0] for value in running[-100:]) / len(running[-100:]),
                          ce=sum(value[1] for value in running[-100:]) / len(running[-100:]),
                          crps=sum(value[2] for value in running[-100:]) / len(running[-100:]),
                          seconds=time.time()-started, memory_gib=memory))
        record = dict(event='eval', epoch=epoch, step=step,
                      dev=base.evaluate(model, dev, cfg, 'mps'),
                      train_probe=base.evaluate(model, probe, cfg, 'mps'),
                      online_training_objective=sum(value[0] for value in running)/len(running),
                      seconds=time.time()-started)
        history.append(record)
        saved = dict(model=model.state_dict(), optimizer=opt.state_dict(), cfg=cfg,
                     continuation_cfg=extension, epoch=epoch, step=step,
                     dev=record['dev'], train_probe=record['train_probe'],
                     evaluation_record=record, source_sha256=source_hashes)
        torch.save(saved, out / 'last.tmp.pt')
        (out / 'last.tmp.pt').replace(out / 'last.pt')
        if record['dev']['all']['nll'] < best:
            best = record['dev']['all']['nll']
            torch.save(saved, out / 'best.tmp.pt')
            (out / 'best.tmp.pt').replace(out / 'best.pt')
        emit(record)
        detected = base.overfit_status(history, cfg)
        base.write(out / 'overfit_status.json', dict(epoch=epoch, criterion_met=bool(detected), result=detected))
        emit(dict(event='overfit_observed', epoch=epoch, criterion_met=bool(detected), stopped=False))
        del saved
    emit(dict(event='continuation_complete', completed_epoch=extension['target_epoch'],
              additional_epochs_completed=extension['target_epoch']-extension['resume_epoch'],
              best_dev_nll=best, stopped_on_overfit=False))


if __name__ == '__main__':
    main()
