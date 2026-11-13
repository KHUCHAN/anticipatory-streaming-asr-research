#!/usr/bin/env python3
"""FGN-style predictor on top of the frozen stage-1 encoder.

--noise shared : FGN (global 32-d noise -> conditional layer-norms, CRPS on marginals)
--noise slot   : ablation, independent noise per future position (CRPS)
--noise none   : ablation, deterministic non-autoregressive predictor (cross-entropy)
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS  # noqa: E402
from avs.fgn import FGN, crps_loss, topk_product, build_examples  # noqa: E402
from avs.text import Tokenizer  # noqa: E402

CHUNKS, LEFT = [4, 8, 16, 32], 64


def contexts(base, model, ds, idxs, align_modes, chunk, rng, m, dev, amp):
    mel, lens, _, _ = ds.collate(idxs, dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=amp):
        with torch.no_grad():
            enc, el = base.encoder(mel, lens, chunk, LEFT)
    bidx, cut, prefix, pmask, aidx, tgt = build_examples(idxs, ds.tokens, align_modes[f'c{chunk}'], el.cpu(), chunk, rng, m, model)
    audio = enc.float()[bidx.to(dev)[:, None], aidx.clamp(min=0).to(dev)]
    ctx, pad = model.context(prefix.to(dev), pmask.to(dev), audio, (aidx >= 0).to(dev))  # float32: float16 diverged at step ~6800
    return ctx, pad, tgt.to(dev)


class Shape:
    """Stand-in for a model when building examples with a different horizon."""
    def __init__(self, model, H):
        self.P, self.A, self.H = model.P, model.A, H


def pieces(base, model, ds, idxs, align_modes, chunk, rng, m, dev, amp, H):
    mel, lens, _, _ = ds.collate(idxs, dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=amp):
        with torch.no_grad():
            enc, el = base.encoder(mel, lens, chunk, LEFT)
    bidx, cut, prefix, pmask, aidx, tgt = build_examples(idxs, ds.tokens, align_modes[f'c{chunk}'], el.cpu(), chunk, rng, m, Shape(model, H))
    audio = enc.float()[bidx.to(dev)[:, None], aidx.clamp(min=0).to(dev)]
    return prefix.to(dev), pmask.to(dev), audio, (aidx >= 0).to(dev), tgt.to(dev)


def rollout_loss(model, prefix, pmask, audio, amask, tgt, noise, aux, dev):
    """FGN-style autoregressive rollout: every member feeds its own prediction back as the newest context token.

    The loss is averaged over the rollout steps and always scored against the true token of that step.
    Tokens are discrete, so no gradient flows through the fed-back prediction (unlike the continuous weather state).
    """
    n, R = prefix.shape[0], tgt.shape[1]
    trajs = 1 if noise == 'none' else 2
    pre, pm = [prefix.clone() for _ in range(trajs)], [pmask.clone() for _ in range(trajs)]
    one = torch.ones(n, 1, dtype=torch.bool, device=dev)
    loss, steps = 0.0, 0
    for r in range(R):
        outs = []
        for j in range(trajs):
            ctx, pad = model.context(pre[j], pm[j], audio, amask)
            l = model.sample(ctx, pad, model.draw(n, dev), torch.full((n,), r, device=dev) if model.lead else None)
            outs.append(l)
            nxt = l[:, 0].float().argmax(-1).detach()
            pre[j] = torch.cat([pre[j][:, 1:], nxt[:, None]], 1); pm[j] = torch.cat([pm[j][:, 1:], one], 1)
        t = tgt[:, r:r + 1]
        if not (t != -100).any():
            continue
        if noise == 'none':
            loss = loss + F.cross_entropy(outs[0].float().transpose(1, 2), t, ignore_index=-100, label_smoothing=0.1)
        else:
            loss = loss + crps_loss(outs[0], outs[1], t, aux)
        steps += 1
    return loss / max(1, steps)


def path_stats(paths, tgt, ks=(1, 4, 16), ns=(1, 2, 3, 4)):
    """paths [N,K,H], tgt [N,H] -> {(k,n): (hits, count)}: some of the first k paths equals the first n targets."""
    out = {}
    for n in ns:
        valid = (tgt[:, :n] != -100).all(1)
        eq = (paths[:, :, :n] == tgt[:, None, :n]).all(-1)
        for k in ks:
            out[(k, n)] = (int((eq[:, :k].any(1) & valid).sum()), int(valid.sum()))
    return out


@torch.no_grad()
def evaluate(base, model, sets, dev, amp, k=16, m=8):
    model.eval()
    out, score = {}, []
    gen = torch.Generator().manual_seed(99)
    for name, (ds, subset, align) in sets.items():
        for chunk in (8, 16):
            rng = random.Random(1234)
            acc, loss_s, loss_n, hit, sharp, distinct, n_ex = {}, 0.0, 0, torch.zeros(model.H, 2), 0.0, 0.0, 0
            for idxs in ds.batches(40000, shuffle=False, subset=subset):
                ctx, pad, tgt = contexts(base, model, ds, idxs, align, chunk, rng, m, dev, amp)
                with torch.autocast(dev, dtype=torch.float16, enabled=False):
                    if model.noise == 'none':
                        logits = model.sample(ctx, pad, None).float()
                        ok = tgt != -100
                        loss_s += float(F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction='sum')); loss_n += int(ok.sum())
                        mean = logits.softmax(-1)
                        paths = topk_product(logits.log_softmax(-1), k)
                        sharp += float(mean.max(-1).values.mean()) * len(tgt)
                    else:
                        probs = model.members(ctx, pad, k, gen)
                        ok = tgt != -100
                        p1, p2 = probs[:, 0], probs[:, 1]
                        y = tgt.clamp(min=0)[..., None]
                        c = (1 - p1.gather(2, y)[..., 0]) + (1 - p2.gather(2, y)[..., 0]) - 0.5 * (p1 - p2).abs().sum(-1)
                        loss_s += float((c * ok).sum()); loss_n += int(ok.sum())
                        mean = probs.mean(1)
                        paths = probs.argmax(-1)
                        sharp += float(probs.max(-1).values.mean()) * len(tgt)
                paths, tgt_c, ok_c = paths.cpu(), tgt.cpu(), ok.cpu()
                top = mean.argmax(-1).cpu()
                for h in range(model.H):
                    hit[h] += torch.tensor([float(((top[:, h] == tgt_c[:, h]) & ok_c[:, h]).sum()), float(ok_c[:, h].sum())])
                for key, (a, b) in path_stats(paths, tgt_c).items():
                    x = acc.get(key, (0, 0)); acc[key] = (x[0] + a, x[1] + b)
                distinct += sum(len({tuple(p) for p in ex.tolist()}) for ex in paths); n_ex += len(tgt)
            rec = {'loss': round(loss_s / loss_n, 4), 'sharp': round(sharp / n_ex, 3), 'distinct': round(distinct / n_ex, 2),
                   'hit1': [round(100 * float(hit[h, 0] / hit[h, 1]), 1) for h in range(model.H)]}
            for (kk, n), (a, b) in acc.items():
                rec[f'path{n}@{kk}'] = round(100 * a / b, 2)
            out[f'{name}/c{chunk}'] = rec
            score.append(rec['loss'])
    model.train()
    return out, sum(score) / len(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', choices=['shared', 'slot', 'none'], default='shared')
    ap.add_argument('--base', default='runs/stage1/avg.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--cuts', type=int, default=8, help='cutoffs sampled per utterance')
    ap.add_argument('--horizon', type=int, default=6)
    ap.add_argument('--prefix', type=int, default=8, help='context length in tokens')
    ap.add_argument('--rollout', type=int, default=1, help='>1: train on autoregressive rollouts of this many steps (needs --horizon 1)')
    ap.add_argument('--lead', type=int, default=0, help='>0: teacher-forced training with 0..lead-1 true tokens ahead of the audio in the context (needs --horizon 1)')
    ap.add_argument('--split-ctx', action='store_true', help='context encoder over audio only (one pass per boundary)')
    ap.add_argument('--init', default=None, help='checkpoint to start from (rollout fine-tuning)')
    ap.add_argument('--aux', type=float, default=0.5, help='weight of the ensemble-mean log score added to CRPS')
    ap.add_argument('--dev-utts', type=int, default=300)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--resume', action='store_true', help='continue from <out>/last.pt if it exists')
    a = ap.parse_args()
    dev, amp = a.device, not a.no_amp and a.device == 'mps'
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / a.base, map_location='cpu')
    base = AVS(**ck['cfg']); base.load_state_dict(ck['model']); base.to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    c = ck['cfg']
    model = FGN(c['vocab'], c['d'], c['h'], c['ff'], H=a.horizon, P=a.prefix, noise=a.noise, drop=c['drop'], split=a.split_ctx, lead=a.lead).to(dev).train()
    with torch.no_grad():  # token embedding and output layer start from the stage-1 decoder
        model.emb.weight.copy_(base.decoder.emb.weight)
        model.out.load_state_dict(base.decoder.out.state_dict())
    if a.init:
        model.load_state_dict(torch.load(ROOT / a.init, map_location='cpu')['model'])
    align_dir = (ROOT / a.base).parent / 'align'
    train = FeatSet('train-clean-100', tok)
    train_align = torch.load(align_dir / 'train-clean-100.pt')
    subset = range(a.limit) if a.limit else None
    sets = {}
    for name in ('dev-clean', 'dev-other'):
        ds = FeatSet(name, tok)
        n = min(a.dev_utts, a.limit or len(ds), len(ds))
        sets[name] = (ds, range(n) if a.limit else random.Random(0).sample(range(len(ds)), n), torch.load(align_dir / f'{name}.pt'))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    spe = len(train.batches(40000, shuffle=False, subset=subset))
    total = spe * a.epochs
    sched = lambda s: (s + 1) / a.warmup if s < a.warmup else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, total - a.warmup)))
    log = (out / 'log.jsonl').open('a')

    def emit(rec):
        line = json.dumps(rec); print(line, flush=True); log.write(line + '\n'); log.flush()

    emit({'event': 'start', 'noise': a.noise, 'params': sum(p.numel() for p in model.parameters()), 'steps_per_epoch': spe,
          'total_steps': total, 'args': vars(a)})
    rng = random.Random(a.seed + 100)
    tgen = torch.Generator().manual_seed(a.seed + 200)
    step, best, start = 0, float('inf'), 0
    if a.resume and (out / 'last.pt').exists():
        st = torch.load(out / 'last.pt', map_location='cpu')
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        step, best, start = st['step'], st['best'], st['epoch'] + 1
        rng.setstate(st['rng']); torch.set_rng_state(st['torch_rng'])
        emit({'event': 'resume', 'epoch': start, 'step': step})
    mem = lambda: round(torch.mps.driver_allocated_memory() / 2 ** 30, 2) if dev == 'mps' else 0.0
    for epoch in range(start, a.epochs):
        t0, agg, n_agg, t_ep = time.time(), 0.0, 0, time.time()
        for idxs in train.batches(40000, subset=subset, seed=7000 + epoch):
            chunk = rng.choice(CHUNKS)
            for g in opt.param_groups:
                g['lr'] = a.lr * sched(step)
            kk = None
            if a.rollout > 1:
                pz = pieces(base, model, train, idxs, train_align, chunk, rng, a.cuts, dev, amp, a.rollout)
            elif a.lead:
                prefix, pmask, audio, amask, t4 = pieces(base, model, train, idxs, train_align, chunk, rng, a.cuts, dev, amp, a.lead)
                n = prefix.shape[0]
                kk = torch.randint(0, a.lead, (n,), generator=tgen).to(dev)
                idx = kk[:, None] + torch.arange(model.P, device=dev)[None]
                prefix = torch.cat([prefix, t4.clamp(min=0)], 1).gather(1, idx)       # the k newest context tokens are true future tokens
                pmask = torch.cat([pmask, t4 != -100], 1).gather(1, idx)
                tgt = t4.gather(1, kk[:, None])
                ctx, pad = model.context(prefix, pmask, audio, amask)
            else:
                ctx, pad, tgt = contexts(base, model, train, idxs, train_align, chunk, rng, a.cuts, dev, amp)
            with torch.autocast(dev, dtype=torch.float16, enabled=False):
                if a.rollout > 1:
                    loss = rollout_loss(model, *pz, a.noise, a.aux, dev)
                elif a.noise == 'none':
                    loss = F.cross_entropy(model.sample(ctx, pad, None, kk).float().transpose(1, 2), tgt, ignore_index=-100, label_smoothing=0.1)
                else:
                    n = ctx.shape[0]
                    loss = crps_loss(model.sample(ctx, pad, model.draw(n, dev), kk), model.sample(ctx, pad, model.draw(n, dev), kk), tgt, a.aux)
            if not torch.isfinite(loss):
                emit({'event': 'nonfinite', 'step': step}); opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                emit({'event': 'nonfinite', 'step': step, 'where': 'grad'}); opt.zero_grad(set_to_none=True); step += 1; continue
            opt.step(); step += 1
            if dev == 'mps' and step % (5 if a.rollout > 1 else 20) == 0:   # variable batch shapes make the MPS buffer cache grow without bound (19 GB by step 6000)
                torch.mps.empty_cache()
            agg += loss.detach(); n_agg += 1
            if step % 100 == 0:
                emit({'event': 'train', 'epoch': epoch, 'step': step, 'lr': round(a.lr * sched(step), 6),
                      'loss': round(float(agg) / n_agg, 4), 'sec_per_step': round((time.time() - t0) / n_agg, 3), 'mps_gb': mem()})
                t0, agg, n_agg = time.time(), 0.0, 0
        ev, score = evaluate(base, model, sets, dev, amp)
        emit({'event': 'eval', 'epoch': epoch, 'step': step, 'score': round(score, 4), 'metrics': {'dev-clean/c16': ev['dev-clean/c16']},
              'full': ev, 'epoch_min': round((time.time() - t_ep) / 60, 1)})
        state = {'model': model.state_dict(), 'cfg': model.cfg, 'epoch': epoch, 'score': score, 'seed': a.seed}
        if score < best:
            best = score; torch.save(state, out / 'best.pt')
        torch.save({**state, 'opt': opt.state_dict(), 'step': step, 'best': best, 'rng': rng.getstate(), 'torch_rng': torch.get_rng_state()}, out / 'last.pt')
        if dev == 'mps':
            torch.mps.empty_cache()
    emit({'event': 'done', 'best_score': round(best, 4)})


if __name__ == '__main__':
    main()
