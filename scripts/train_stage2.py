#!/usr/bin/env python3
"""Stage 2: anticipatory predictors on top of the frozen stage-1 encoder.

--kind text  : B2, decoder without cross-attention (transcripts only).
--kind audio : B3, decoder whose cross-attention is masked to audio that had
               arrived before a cutoff placed ahead of the tokens it predicts.

A token sequence is cut into segments of 1..4 tokens. All tokens of a segment
see only encoder frames before one chunk boundary that precedes the segment's
first token, so the h-th token of a segment is predicted h tokens ahead of audio.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS, Decoder, sos_eos  # noqa: E402
from avs.text import Tokenizer  # noqa: E402

CHUNKS, LEFT, HMAX = [4, 8, 16, 32], 64, 4


WINDOW, WPOS = 0, 16


def cutoffs(spikes, enc_len, chunk, rng, seg=None):
    """-> (cut, hor) lists of length U+1 (targets incl. eos)."""
    sp = list(spikes) + [enc_len]
    cut, hor, u = [], [], 0
    while u < len(sp):
        hi = (sp[u] // chunk) * chunk
        lo = min(hi, (sp[u - 1] // chunk + 1) * chunk) if u else 0
        a = lo + chunk * rng.randint(0, (hi - lo) // chunk)
        m = seg or rng.randint(1, HMAX)
        for h in range(m):
            if u >= len(sp):
                break
            cut.append(a); hor.append(h); u += 1
    return cut, hor


def batch_masks(idxs, align, enc_lens, t_enc, chunk, rng, seg, dev):
    u_max = max(len(align[i]) for i in idxs) + 1
    cut = torch.zeros(len(idxs), u_max, dtype=torch.long)
    hor = torch.full((len(idxs), u_max), -1, dtype=torch.long)
    for b, i in enumerate(idxs):
        c, h = cutoffs(align[i].tolist(), int(enc_lens[b]), chunk, rng, seg)
        cut[b, :len(c)] = torch.tensor(c); hor[b, :len(h)] = torch.tensor(h)
    mask = torch.arange(t_enc)[None, None, :] < cut[:, :, None]
    return mask.to(dev), hor.to(dev)


def forward(kind, base, pred, ds, idxs, align_modes, chunk, rng, seg, dev, amp):
    mel, lens, ys, ylens = ds.collate(idxs, dev)
    ys_in, tgt = sos_eos(ys, ylens)
    with torch.autocast(dev, dtype=torch.float16, enabled=amp):
        with torch.no_grad():
            enc, el = base.encoder(mel, lens, chunk, LEFT)
        mask, hor = batch_masks(idxs, align_modes[f'c{chunk}'], el.cpu(), enc.shape[1], chunk, rng, seg, dev)
        if WINDOW and kind == 'audio':
            # strict bounded context: every position is predicted from its own last WINDOW input tokens only.
            # Training scores WPOS random positions per utterance (memory); evaluation scores all positions in slices.
            b, u = ys_in.shape
            if pred.training:
                pos = torch.stack([torch.randperm(u, device='cpu')[:min(WPOS, u)] for _ in range(b)]).to(dev)
                tgt, hor = tgt.gather(1, pos), hor.gather(1, pos)
            else:
                pos = torch.arange(u, device=dev)[None].expand(b, -1)
            outs = []
            for j in range(0, pos.shape[1], WPOS):
                pj = pos[:, j:j + WPOS]
                k = pj.shape[1]
                idx = (pj[..., None] - (WINDOW - 1) + torch.arange(WINDOW, device=dev)).clamp(min=0)      # [B,k,W], left pad = SOS
                win = ys_in.gather(1, idx.flatten(1)).view(b * k, WINDOW)
                mem = enc[:, None].expand(-1, k, -1, -1).reshape(b * k, enc.shape[1], enc.shape[2])
                m = mask.gather(1, pj[..., None].expand(-1, -1, mask.shape[2])).reshape(b * k, 1, -1)
                outs.append(pred(win, mem, m)[:, -1].view(b, k, -1))
            logits = torch.cat(outs, 1)
        else:
            logits = pred(ys_in) if kind == 'text' else pred(ys_in, enc, mask)
    return logits.float(), tgt, hor


@torch.no_grad()
def evaluate(kind, base, pred, sets, dev, amp):
    pred.eval()
    out = {}
    for name, (ds, subset, align) in sets.items():
        for chunk in (8, 16):
            rng = random.Random(1234)
            stat = torch.zeros(HMAX, 5, dtype=torch.float64)  # n, nll, hit1, hit5, path1
            for idxs in ds.batches(40000, shuffle=False, subset=subset):
                logits, tgt, hor = forward(kind, base, pred, ds, idxs, align, chunk, rng, HMAX, dev, amp)
                logp = logits.log_softmax(-1).cpu(); tgt = tgt.cpu(); hor = hor.cpu()
                ok = tgt != -100
                nll = -logp.gather(2, tgt.clamp(min=0)[..., None])[..., 0]
                top5 = logp.topk(5, -1).indices
                hit1 = top5[..., 0] == tgt
                hit5 = (top5 == tgt[..., None]).any(-1)
                path = hit1.clone()
                for h in range(1, HMAX):  # path correct through h: all of 0..h correct within the segment
                    sel = hor == h
                    prev = torch.roll(path, 1, 1)
                    path = torch.where(sel, path & prev, path)
                for h in range(HMAX):
                    sel = (hor == h) & ok
                    stat[h] += torch.tensor([sel.sum(), nll[sel].sum(), hit1[sel].sum(), hit5[sel].sum(), path[sel].sum()], dtype=torch.float64)
            for h in range(HMAX):
                n = stat[h, 0].item()
                out[f'{name}/c{chunk}/h{h}'] = {'nll': round(stat[h, 1].item() / n, 4), 'hit1': round(100 * stat[h, 2].item() / n, 2),
                                               'hit5': round(100 * stat[h, 3].item() / n, 2), 'path1': round(100 * stat[h, 4].item() / n, 2)}
    pred.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['text', 'audio'], required=True)
    ap.add_argument('--base', default='runs/stage1/avg.pt')
    ap.add_argument('--out', default=None)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--train', default='train-clean-100')
    ap.add_argument('--dev-utts', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: use only the first N train utterances')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--band', type=int, default=0, help='audio predictor whose every layer attends to at most this many earlier tokens (bounded cached state)')
    ap.add_argument('--no-init', action='store_true', help='do not start from the stage-1 decoder')
    ap.add_argument('--window', type=int, default=0, help='audio predictor with a strict context of this many input tokens (0 = full history)')
    ap.add_argument('--seed', type=int, default=2, help='Training RNG seed; default reproduces the original seed-2 recipe')
    a = ap.parse_args()
    dev, amp = a.device, not a.no_amp and a.device == 'mps'
    global WINDOW
    WINDOW = a.window
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = ROOT / (a.out or f'runs/stage2_{a.kind}')
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / a.base, map_location='cpu')
    base = AVS(**ck['cfg']); base.load_state_dict(ck['model']); base.to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    cfg = ck['cfg']
    pred = Decoder(cfg['vocab'], cfg['d'], cfg['h'], cfg['ff'], cfg['dec_layers'], cfg['drop'], cross=a.kind == 'audio')
    if a.kind == 'audio' and not a.no_init:
        pred.load_state_dict(base.decoder.state_dict())
    pred.band = a.band
    pred.to(dev).train()
    align_dir = (ROOT / a.base).parent / 'align'
    train = FeatSet(a.train, tok)
    train_align = torch.load(align_dir / f'{a.train}.pt')
    subset = range(a.limit) if a.limit else None
    sets = {}
    for name in ('dev-clean', 'dev-other'):
        ds = FeatSet(name, tok)
        n = min(a.dev_utts, a.limit or len(ds), len(ds))
        sets[name] = (ds, range(n) if a.limit else random.Random(0).sample(range(len(ds)), n), torch.load(align_dir / f'{name}.pt'))
    opt = torch.optim.AdamW(pred.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    spe = len(train.batches(40000, shuffle=False, subset=subset))
    total = spe * a.epochs
    sched = lambda s: (s + 1) / a.warmup if s < a.warmup else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, total - a.warmup)))
    log = (out / 'log.jsonl').open('a')

    def emit(rec):
        line = json.dumps(rec); print(line, flush=True); log.write(line + '\n'); log.flush()

    emit({'event': 'start', 'kind': a.kind, 'params': sum(p.numel() for p in pred.parameters()), 'steps_per_epoch': spe,
          'total_steps': total, 'base': a.base, 'args': vars(a)})
    rng = random.Random(a.seed + 1)
    step, best = 0, float('inf')
    for epoch in range(a.epochs):
        t0, agg, n_agg, t_ep = time.time(), 0.0, 0, time.time()
        for idxs in train.batches(40000, subset=subset, seed=7000 + epoch + 100000 * (a.seed - 2)):
            chunk = rng.choice(CHUNKS)
            for g in opt.param_groups:
                g['lr'] = a.lr * sched(step)
            logits, tgt, _ = forward(a.kind, base, pred, train, idxs, train_align, chunk, rng, None, dev, amp)
            loss = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, label_smoothing=0.1)
            if not torch.isfinite(loss):
                emit({'event': 'nonfinite', 'step': step}); opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 5.0)
            opt.step(); step += 1
            agg += loss.detach(); n_agg += 1
            if step % 100 == 0:
                emit({'event': 'train', 'epoch': epoch, 'step': step, 'lr': round(a.lr * sched(step), 6),
                      'loss': round(float(agg) / n_agg, 4), 'sec_per_step': round((time.time() - t0) / n_agg, 3)})
                t0, agg, n_agg = time.time(), 0.0, 0
        ev = evaluate(a.kind, base, pred, sets, dev, amp)
        score = sum(v['nll'] for v in ev.values()) / len(ev)
        brief = {k: ev[k] for k in ('dev-clean/c16/h0', 'dev-clean/c16/h1', 'dev-clean/c16/h3')}
        emit({'event': 'eval', 'epoch': epoch, 'step': step, 'mean_nll': round(score, 4), 'metrics': brief, 'full': ev,
              'epoch_min': round((time.time() - t_ep) / 60, 1)})
        state = {'model': pred.state_dict(), 'cfg': cfg, 'kind': a.kind, 'epoch': epoch, 'mean_nll': score, 'window': a.window, 'band': a.band, 'seed': a.seed}
        torch.save(state, out / 'last.pt')
        if score < best:
            best = score; torch.save(state, out / 'best.pt')
    emit({'event': 'done', 'best_mean_nll': round(best, 4)})


if __name__ == '__main__':
    main()
