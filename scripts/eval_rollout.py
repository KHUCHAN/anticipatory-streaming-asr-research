#!/usr/bin/env python3
"""Multi-step accuracy of bounded-context one-step predictors that feed their own predictions back.

Same protocol as scripts/eval_fgn.py (same utterances, cutoffs and path n@K metric), so rows are comparable:
  <name>_beam   deterministic one-step model, beam 16 over its own predictions
  <name>_roll   noise model, 16 independent rollouts (fresh noise every step, as in FGN)
  ar_audio      full-history autoregressive predictor (B3), beam 16
Also reports CPU time per predictor step and the per-hypothesis state size.
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS  # noqa: E402
from avs.fgn import build_examples, ShortPrior  # noqa: E402
from avs.stream import Node, Predictor  # noqa: E402
from avs.text import Tokenizer, SOS  # noqa: E402
from eval_fgn import ar_beam, load_ar, load_fgn, tally, LEFT, KS, NS  # noqa: E402


class Shape:
    def __init__(self, m, H):
        self.P, self.A, self.H = m.P, m.A, H


def shift(prefix, pmask, tok):
    one = torch.ones(tok.shape[0], 1, dtype=torch.bool, device=tok.device)
    return torch.cat([prefix[:, 1:], tok[:, None]], 1), torch.cat([pmask[:, 1:], one], 1)


def roll_beam(m, prefix, pmask, audio, amask, k, depth):
    n, dev = prefix.shape[0], prefix.device
    score = torch.full((n, k), -1e30, device=dev); score[:, 0] = 0
    pre, pm = prefix[:, None].repeat(1, k, 1), pmask[:, None].repeat(1, k, 1)
    ext = torch.zeros(n, k, 0, dtype=torch.long, device=dev)
    au, am = audio.repeat_interleave(k, 0), amask.repeat_interleave(k, 0)
    for s in range(depth):
        out = []
        for i in range(0, n * k, 1024):
            ctx, pad = m.context(pre.flatten(0, 1)[i:i + 1024], pm.flatten(0, 1)[i:i + 1024], au[i:i + 1024], am[i:i + 1024])
            out.append(m.sample(ctx, pad, None, torch.full((ctx.shape[0],), s, device=dev) if m.lead else None)[:, 0].float().log_softmax(-1))
        logp = torch.cat(out).view(n, k, -1)
        logp[..., 0] = -1e30
        score, idx = (score[..., None] + logp).flatten(1).topk(k, -1)
        src, tk = idx // logp.shape[-1], idx % logp.shape[-1]
        ext = torch.cat([ext.gather(1, src[..., None].expand(-1, -1, s)), tk[..., None]], 2)
        pre, pm = pre.gather(1, src[..., None].expand(-1, -1, pre.shape[2])), pm.gather(1, src[..., None].expand(-1, -1, pm.shape[2]))
        p2, m2 = shift(pre.flatten(0, 1), pm.flatten(0, 1), tk.flatten())
        pre, pm = p2.view(n, k, -1), m2.view(n, k, -1)
    return ext


def roll_sample(m, prefix, pmask, audio, amask, k, depth, gen):
    n, dev = prefix.shape[0], prefix.device
    pre, pm = prefix.repeat_interleave(k, 0), pmask.repeat_interleave(k, 0)
    au, am = audio.repeat_interleave(k, 0), amask.repeat_interleave(k, 0)
    ext = []
    for s in range(depth):
        out = []
        for i in range(0, n * k, 1024):
            ctx, pad = m.context(pre[i:i + 1024], pm[i:i + 1024], au[i:i + 1024], am[i:i + 1024])
            lg = m.sample(ctx, pad, m.draw(ctx.shape[0], dev, gen), torch.full((ctx.shape[0],), s, device=dev) if m.lead else None)[:, 0].float()
            lg[:, 0] = -1e30
            out.append(lg.argmax(-1))
        tk = torch.cat(out)
        ext.append(tk)
        pre, pm = shift(pre, pm, tk)
    return torch.stack(ext, 1).view(n, k, depth)


def bench(models, ar, reps=40):
    """CPU, 1 thread: one predictor step for a 4-hypothesis beam after a 30-token prefix, 64 audio frames."""
    torch.set_num_threads(1)
    enc = torch.randn(64, 256)
    out = {}

    def run(p):
        p.set_memory(enc)
        node = Node(SOS, None); p.step([node])
        for _ in range(30):
            node = Node(5, node); p.step([node])
        kids = [Node(7 + i, node) for i in range(4)]
        ts = []
        for _ in range(reps):
            for kd in kids:
                kd.kv = None
            t = time.perf_counter(); p.step(kids); ts.append(time.perf_counter() - t)
        return round(1000 * statistics.median(ts[5:]), 3), p.bytes_per_node
    for name, m in models.items():
        ms, by = run(ShortPrior(m.cpu(), 1))
        out[name] = {'step4_ms': ms, 'state_bytes_per_hyp': by, 'grows_with_history': False}
    ms, by = run(Predictor(ar.cpu()))
    out['ar_audio'] = {'step4_ms': ms, 'state_bytes_per_hyp_per_token': by, 'grows_with_history': True}
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='runs/stage1/avg.pt')
    ap.add_argument('--models', nargs='+', required=True, help='name=checkpoint ...')
    ap.add_argument('--audio', default='runs/stage2_audio/best.pt')
    ap.add_argument('--splits', nargs='+', default=['dev-clean', 'test-clean', 'dev-other'])
    ap.add_argument('--chunks', type=int, nargs='+', default=[8, 16])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--cuts', type=int, default=8)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default='runs/rollout_eval.json')
    a = ap.parse_args()
    dev = a.device
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / a.base, map_location='cpu')
    base = AVS(**ck['cfg']); base.load_state_dict(ck['model']); base.to(dev).eval()
    models = {kv.split('=')[0]: load_fgn(kv.split('=')[1], dev) for kv in a.models}
    ar = load_ar(a.audio, dev)
    gen = torch.Generator().manual_seed(5)
    results = {}
    for split in a.splits:
        ds = FeatSet(split, tok)
        align = torch.load((ROOT / a.base).parent / 'align' / f'{split}.pt')
        subset = random.Random(21).sample(range(len(ds)), min(a.n, len(ds)))
        for chunk in a.chunks:
            acc = {}
            rngs = {name: random.Random(77) for name in list(models) + ['ar']}
            for idxs in ds.batches(40000, shuffle=False, subset=subset):
                mel, lens, _, _ = ds.collate(idxs, dev)
                enc, el = base.encoder(mel, lens, chunk, LEFT)
                for name, m in models.items():   # same cutoffs for every model (same rng seed), own context length
                    bidx, cut, prefix, pmask, aidx, tgt = build_examples(idxs, ds.tokens, align[f'c{chunk}'], el.cpu(), chunk, rngs[name], a.cuts, Shape(m, 4))
                    audio = enc[bidx.to(dev)[:, None], aidx.clamp(min=0).to(dev)].float()
                    args = (prefix.to(dev), pmask.to(dev), audio, (aidx >= 0).to(dev))
                    if m.noise == 'none':
                        tally(acc, name + '_beam', roll_beam(m, *args, 16, 4).cpu(), tgt)
                    else:
                        tally(acc, name + '_roll', roll_sample(m, *args, 16, 4, gen).cpu(), tgt)
                bidx, cut, prefix, pmask, aidx, tgt = build_examples(idxs, ds.tokens, align[f'c{chunk}'], el.cpu(), chunk, rngs['ar'], a.cuts, Shape(next(iter(models.values())), 4))
                out = torch.zeros(len(tgt), 16, 4, dtype=torch.long)
                for b, i in enumerate(idxs):
                    sel = (bidx == b).nonzero()[:, 0]
                    if not len(sel):
                        continue
                    sp = align[f'c{chunk}'][i].tolist()
                    cuts = cut[sel].tolist()
                    counts = [sum(s < c for s in sp) for c in cuts]
                    out[sel] = ar_beam(ar, ds.tokens[i] + [SOS], counts, cuts, enc[b, :int(el[b])].float(), 16, 4, dev).cpu()
                tally(acc, 'ar_audio', out, tgt)
            rec = {}
            for key, (x, y) in acc.items():
                if key[1] == 'distinct':
                    rec.setdefault(key[0], {})['distinct_of_16'] = round(x / y, 2)
                else:
                    rec.setdefault(key[0], {})[f'path{key[2]}@{key[1]}'] = round(100 * x / y, 2)
            results[f'{split}/c{chunk}'] = rec
            print(f'\n## {split} chunk {chunk * 40} ms  (path n@K, %)')
            cols = [f'path{n}@{k}' for n in NS for k in KS]
            print('| method | ' + ' | '.join(cols) + ' | distinct/16 |')
            print('|' + '---|' * (len(cols) + 2))
            for name, r in rec.items():
                print(f'| {name} | ' + ' | '.join(str(r[c]) for c in cols) + f' | {r["distinct_of_16"]} |', flush=True)
    results['bench_cpu_1thread'] = bench(models, ar)
    print('\nbench', json.dumps(results['bench_cpu_1thread']))
    json.dump(results, (ROOT / a.out).open('w'), indent=1)


if __name__ == '__main__':
    main()
