#!/usr/bin/env python3
"""Controlled FGN generator growth, fixed-case fitting and generalization curves."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'runs/fgn_capacity_v1'
sys.path.insert(0, str(ROOT / 'experiments/band_fgn_v1'))
from common import Dataset, datasets, oldstate, sha
from model import Hybrid, objective


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class FixedCases(Dataset):
    """Fix each token/cutoff independently of epoch and batch composition."""
    def __init__(self, source, cfg):
        super().__init__(source.rows, source.meta)
        self.seed = cfg['fixed_case_seed']

    def collate(self, indices, rng, device):
        # The original collator consumes randrange, then randint, for each
        # utterance. Use independent utterance seeds with the same sampling law.
        class Cases:
            def __init__(self, rows, seed):
                self.rows, self.seed, self.index = rows, seed, 0
            def randrange(self, n):
                uid = self.rows[self.index]['utterance_id']
                seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{uid}'.encode()).digest()[:8], 'big')
                self.rng = random.Random(seed)
                return self.rng.randrange(n)
            def randint(self, lo, hi):
                result = self.rng.randint(lo, hi)
                self.index += 1
                return result
        cases = Cases([self.rows[i] for i in indices], self.seed)
        return super().collate(indices, cases, device)


def make_model(layers, cfg):
    torch.manual_seed(cfg['seed'])
    model = Hybrid('small_fgn')
    checkpoint = torch.load(ROOT / cfg['initial_checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.state.cfg = copy.deepcopy(model.state.cfg)
    model.state.cfg['layers'] = layers
    while len(model.state.blocks) < layers:
        block = oldstate.StateBlock(model.state.cfg)
        # Exact identity residual branch at initialization.
        torch.nn.init.zeros_(block.attention.out_proj.weight)
        torch.nn.init.zeros_(block.attention.out_proj.bias)
        torch.nn.init.zeros_(block.ff[-1].weight)
        torch.nn.init.zeros_(block.ff[-1].bias)
        model.state.blocks.append(block)
    assert len(model.state.blocks) == layers
    return model


def get_data(cfg):
    train, dev, probe = datasets(cfg)
    return tuple(FixedCases(ds, cfg) for ds in [train, dev, probe])


@torch.inference_mode()
def evaluate(model, ds, cfg, device):
    model.eval()
    cpu_rng = torch.get_rng_state()
    mps_rng = torch.mps.get_rng_state() if device == 'mps' else None
    torch.manual_seed(502)
    sums = {}
    total = 0
    for indices in ds.batches(cfg):
        batch = ds.collate(indices, None, device)
        logits, crps, _ = model(batch, cfg['forecast_horizon'])
        y = batch[-1]
        logp = torch.logsumexp(logits.log_softmax(-1), 0) - math.log(len(logits))
        nll = F.nll_loss(logp, y, reduction='none')
        ce = torch.stack([F.cross_entropy(branch, y, label_smoothing=cfg['label_smoothing'], reduction='none') for branch in logits]).mean(0)
        hit = logp.argmax(-1) == y
        n = len(indices)
        values = sums.setdefault('all', dict(nll=0., ce=0., crps=0., hit1=0., count=0))
        values['nll'] += float(nll.sum())
        values['ce'] += float(ce.sum())
        values['crps'] += n * float(crps)
        values['hit1'] += int(hit.sum())
        values['count'] += n
        for j, index in enumerate(indices):
            split = ds.rows[index]['split']
            values = sums.setdefault(split, dict(nll=0., ce=0., crps=0., hit1=0., count=0))
            values['nll'] += float(nll[j])
            values['ce'] += float(ce[j])
            values['hit1'] += int(hit[j])
            values['count'] += 1
        total += n
    torch.set_rng_state(cpu_rng)
    if mps_rng is not None:
        torch.mps.set_rng_state(mps_rng)
    result = {}
    for key, values in sums.items():
        result[key] = {k: v / values['count'] for k, v in values.items() if k != 'count'}
        result[key]['examples'] = values['count']
        if key == 'all':
            result[key]['objective'] = result[key]['ce'] + cfg['crps_weight'] * result[key]['crps']
        else:
            result[key].pop('crps')
    model.train()
    return result


def overfit_status(history, cfg):
    if history[-1]['epoch'] < cfg['overfit_min_epochs']:
        return None
    best = min(history, key=lambda row: row['dev']['all']['nll'])
    recent = history[-cfg['overfit_patience']:]
    if len(recent) < cfg['overfit_patience']:
        return None
    # A rising validation curve alone is insufficient: the same fixed training
    # probe must improve relative to the validation-optimal checkpoint.
    valid = all(row['epoch'] > best['epoch'] and
                row['dev']['all']['nll'] >= best['dev']['all']['nll'] + cfg['overfit_dev_nll_delta'] and
                row['train_probe']['all']['nll'] <= best['train_probe']['all']['nll'] - cfg['overfit_train_nll_improvement']
                for row in recent)
    if not valid:
        return None
    return dict(detected=True, best_epoch=best['epoch'], confirmed_epoch=history[-1]['epoch'],
                criterion='five successive evaluation points: mixed-dev NLL worse by >=0.02 and fixed-train-probe NLL better by >=0.01 relative to dev-optimal epoch',
                best_dev_nll=best['dev']['all']['nll'], last_dev_nll=history[-1]['dev']['all']['nll'],
                train_at_best=best['train_probe']['all']['nll'], last_train_nll=history[-1]['train_probe']['all']['nll'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--bench', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    cfg = json.loads((HERE / 'config.json').read_text())
    assert args.layers in cfg['generator_layers'] and 1 <= args.epochs <= cfg['max_epochs_per_model']
    torch.set_num_threads(1)
    model = make_model(args.layers, cfg).to(args.device)
    train, dev, probe = get_data(cfg)
    out = OUT / f'layers_{args.layers}' / ('bench_' + sha(HERE / 'config.json')[:8] if args.bench else '')
    out.mkdir(parents=True, exist_ok=True)
    lease = (out / 'training.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg_path = out / 'config.json'
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == cfg, 'Study config changed; use a new namespace.'
    else:
        write(cfg_path, cfg)
    log = (out / 'log.jsonl').open('a', buffering=1)
    def emit(value):
        print(json.dumps(value), flush=True)
        log.write(json.dumps(value) + '\n')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], betas=(.9, .98), weight_decay=cfg['weight_decay'])
    step = first = 0
    history = []
    if not args.bench and (out / 'last.pt').exists():
        checkpoint = torch.load(out / 'last.pt', map_location=args.device, weights_only=False)
        assert checkpoint['cfg'] == cfg
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        first = checkpoint['epoch']
        history = [row for row in map(json.loads, (out / 'log.jsonl').read_text().splitlines()) if row.get('event') == 'eval']
        if overfit_status(history, cfg):
            emit(dict(event='already_overfit', result=overfit_status(history, cfg)))
            return
    emit(dict(event='start', layers=args.layers, total_parameters=sum(p.numel() for p in model.parameters()),
              generator_parameters=sum(p.numel() for p in model.state.parameters()),
              train_utterances=len(train.rows), train_cases=len(train.rows), steps_per_epoch=len(train.batches(cfg)),
              target_epochs=args.epochs, resumed_from_epoch=first,
              config_sha256=sha(HERE / 'config.json'), initial_checkpoint_sha256=sha(ROOT / cfg['initial_checkpoint'])))
    if not history and not args.bench:
        record = dict(event='eval', epoch=0, step=0,
                      dev=evaluate(model, dev, cfg, args.device),
                      train_probe=evaluate(model, probe, cfg, args.device), seconds=0.)
        history.append(record)
        emit(record)
        torch.save(dict(model=model.state_dict(), cfg=cfg, epoch=0, step=0, dev=record['dev'], train_probe=record['train_probe']), out / 'best.pt')
    best = min([row['dev']['all']['nll'] for row in history], default=float('inf'))
    for epoch in range(first + 1, args.epochs + 1):
        started = time.time()
        torch.manual_seed(cfg['seed'] * 1000 + epoch)
        model.train()
        running = []
        for indices in train.batches(cfg, 8100 + epoch):
            batch = train.collate(indices, None, args.device)
            factor = min(1., (step + 1) / cfg['warmup_steps'])
            for group in opt.param_groups:
                group['lr'] = cfg['learning_rate'] * factor
            opt.zero_grad(set_to_none=True)
            loss, parts = objective(model, batch, cfg, cfg['forecast_horizon'])
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
                memory = torch.mps.driver_allocated_memory() / 2**30 if args.device == 'mps' else None
                if memory is not None and memory > cfg['max_driver_memory_gib']:
                    raise RuntimeError('Predeclared MPS memory limit exceeded')
                emit(dict(event='train', epoch=epoch, step=step,
                          objective=sum(v[0] for v in running[-100:]) / len(running[-100:]),
                          ce=sum(v[1] for v in running[-100:]) / len(running[-100:]),
                          crps=sum(v[2] for v in running[-100:]) / len(running[-100:]),
                          seconds=time.time() - started, memory_gib=memory))
            if args.bench and len(running) >= args.bench:
                if args.device == 'mps':
                    torch.mps.synchronize()
                emit(dict(event='bench_done', steps=len(running), seconds=time.time() - started,
                          last_objective=running[-1][0], memory_gib=torch.mps.driver_allocated_memory()/2**30 if args.device == 'mps' else None))
                return
        record = dict(event='eval', epoch=epoch, step=step,
                      dev=evaluate(model, dev, cfg, args.device), train_probe=evaluate(model, probe, cfg, args.device),
                      online_training_objective=sum(v[0] for v in running) / len(running), seconds=time.time() - started)
        history.append(record)
        checkpoint = dict(model=model.state_dict(), optimizer=opt.state_dict(), cfg=cfg, epoch=epoch,
                          step=step, dev=record['dev'], train_probe=record['train_probe'],
                          source_sha256={'study.py': sha(HERE / 'study.py'), 'config.json': sha(HERE / 'config.json')})
        torch.save(checkpoint, out / 'last.tmp.pt')
        (out / 'last.tmp.pt').replace(out / 'last.pt')
        if record['dev']['all']['nll'] < best:
            best = record['dev']['all']['nll']
            torch.save(checkpoint, out / 'best.pt')
        emit(record)
        overfit = overfit_status(history, cfg)
        if overfit:
            write(out / 'overfit.json', overfit)
            emit(dict(event='overfit_confirmed', result=overfit))
            return
    emit(dict(event='stage_complete', completed_epochs=args.epochs, overfitting_detected=False, best_dev_nll=best))


if __name__ == '__main__':
    main()
