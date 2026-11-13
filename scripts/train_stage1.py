#!/usr/bin/env python3
"""Stage 1: chunk-causal encoder with joint CTC + attention-decoder training (B0 base).

Dynamic chunk training: each batch runs in offline mode (full context) or in a
streaming mode with a random chunk size and a bounded left context.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, spec_augment, FEATS  # noqa: E402
from avs.model import AVS  # noqa: E402
from avs.text import Tokenizer, wer  # noqa: E402

CHUNKS = [4, 8, 16, 32]
LEFT = 64  # encoder frames (2.56 s) of left context per layer in streaming modes


def sync(dev):
    if dev == 'mps':
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, sets, tok, dev, max_frames):
    model.eval()
    out = {}
    for name, (ds, subset) in sets.items():
        for mode, (chunk, left) in {'full': (0, -1), 'c16': (16, LEFT), 'c4': (4, LEFT)}.items():
            refs, hyps = [], []
            for idxs in ds.batches(max_frames, shuffle=False, subset=subset):
                mel, lens, _, _ = ds.collate(idxs, dev)
                hyps += [tok.decode(h) for h in model.ctc_greedy(mel, lens, chunk, left)]
                refs += [ds.texts[i] for i in idxs]
            out[f'{name}/{mode}'] = round(100 * wer(refs, hyps), 2)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/stage1')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--max-frames', type=int, default=40000, help='padded mel frames per batch')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=4000)
    ap.add_argument('--train', default='train-clean-100')
    ap.add_argument('--dev-utts', type=int, default=300)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--bench', type=int, default=0, help='run N steps, report throughput, exit')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(FEATS / 'spm.model')
    train = FeatSet(args.train, tok)
    sets = {}
    for name in ('dev-clean', 'dev-other'):
        ds = FeatSet(name, tok)
        sets[name] = (ds, random.Random(0).sample(range(len(ds)), min(args.dev_utts, len(ds))))
    model = AVS(tok.vocab).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    steps_per_epoch = len(train.batches(args.max_frames, shuffle=False))
    total = steps_per_epoch * args.epochs
    sched = lambda s: (s + 1) / args.warmup if s < args.warmup else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, total - args.warmup)))
    step, start_epoch = 0, 0
    last = out / 'last.pt'
    if last.exists() and not args.bench:
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        step, start_epoch = ck['step'], ck['epoch'] + 1
    log = (out / 'log.jsonl').open('a')

    def emit(rec):
        line = json.dumps(rec)
        print(line, flush=True)
        if not args.bench:
            log.write(line + '\n'); log.flush()

    emit({'event': 'start', 'params': n_params, 'steps_per_epoch': steps_per_epoch, 'total_steps': total, 'device': dev,
          'resume_epoch': start_epoch, 'args': vars(args), 'torch': torch.__version__})
    model.train()
    bad = 0
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        agg, n_agg, frames, t0 = {}, 0, 0, time.time()
        for bi, idxs in enumerate(train.batches(args.max_frames, seed=args.seed * 1000 + epoch)):
            mel, lens, ys, ylens = train.collate(idxs, dev)
            mel = spec_augment(mel, lens)
            chunk, left = (0, -1) if random.random() < 0.4 else (random.choice(CHUNKS), LEFT)
            for g in opt.param_groups:
                g['lr'] = args.lr * sched(step)
            with torch.autocast(dev, dtype=torch.float16, enabled=args.amp):
                loss, parts = model(mel, lens, ys, ylens, chunk, left)
            if not torch.isfinite(loss):
                bad += 1
                emit({'event': 'nonfinite', 'step': step, 'consecutive': bad})
                if bad >= 20:
                    emit({'event': 'abort', 'reason': '20 consecutive non-finite losses'}); sys.exit(2)
                opt.zero_grad(set_to_none=True); step += 1
                continue
            bad = 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            ntok = int(ylens.sum())
            for k, v in {'loss': loss.detach() * len(idxs) / ntok, 'ctc': parts['ctc'] * len(idxs) / ntok,
                         'att': parts['att'] * len(idxs) / ntok, 'acc': parts['acc'], 'gnorm': gn}.items():
                agg[k] = agg.get(k, 0.0) + v
            n_agg += 1; frames += int(lens.sum())
            if step % 100 == 0 or (args.bench and step == args.bench):
                sync(dev)
                dt = time.time() - t0
                emit({'event': 'train', 'epoch': epoch, 'step': step, 'lr': round(args.lr * sched(step), 6),
                      **{k: round(float(v) / n_agg, 4) for k, v in agg.items()},
                      'x_realtime': round(frames / 100 / dt, 1), 'sec_per_step': round(dt / n_agg, 3),
                      'mem_gb': round(torch.mps.driver_allocated_memory() / 2 ** 30, 2) if dev == 'mps' else None})
                agg, n_agg, frames, t0 = {}, 0, 0, time.time()
            if args.bench and step >= args.bench:
                return
        ev = evaluate(model, sets, tok, dev, args.max_frames)
        emit({'event': 'eval', 'epoch': epoch, 'step': step, 'wer': ev, 'epoch_min': round((time.time() - t_epoch) / 60, 1)})
        torch.save({'model': model.state_dict(), 'cfg': model.cfg, 'epoch': epoch, 'step': step}, out / f'ep{epoch:02d}.pt')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'cfg': model.cfg, 'epoch': epoch, 'step': step}, last)
    emit({'event': 'done', 'epochs': args.epochs, 'step': step})


if __name__ == '__main__':
    main()
