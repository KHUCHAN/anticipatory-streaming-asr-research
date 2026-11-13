#!/usr/bin/env python3
"""F1/F2: do FGN members form coherent multi-token paths, and what does each FGN ingredient buy?

Every method proposes K paths for the next tokens at the same cutoffs. path n@K = some of the K paths
equals the true next n tokens.

  fgn        K members of one FGN (shared 32-d noise)
  fgn_shuf   the same members with each position permuted across members (same marginals, joint destroyed)
  fgn_x2     K/2 members from each of two independently trained FGNs (model ensemble)
  slot       members with independent noise per position
  nar        deterministic marginals, exact K best product paths
  ar_audio   autoregressive audio-aware predictor (B3), beam K
  ar_text    autoregressive text-only predictor (B2), beam K
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS, Decoder  # noqa: E402
from avs.fgn import FGN, topk_product, build_examples  # noqa: E402
from avs.stream import Node, Predictor  # noqa: E402
from avs.text import Tokenizer, SOS  # noqa: E402

LEFT, KS, NS = 64, (1, 4, 16), (1, 2, 3, 4)


def load_fgn(path, dev):
    ck = torch.load(ROOT / path, map_location='cpu')
    m = FGN(**ck['cfg']); m.load_state_dict(ck['model'])
    return m.to(dev).eval()


def load_ar(path, dev):
    ck = torch.load(ROOT / path, map_location='cpu')
    c = ck['cfg']
    d = Decoder(c['vocab'], c['d'], c['h'], c['ff'], c['dec_layers'], 0.0, cross=ck['kind'] == 'audio')
    d.load_state_dict(ck['model'])
    return d.to(dev).eval()


def ar_beam(dec, toks, counts, cuts, mem, k, depth, dev):
    """Beam search from each cutoff of one utterance. -> paths [N,k,depth]."""
    n, t = len(cuts), mem.shape[0]
    score = torch.full((n, k), -1e30, device=dev); score[:, 0] = 0
    ext = torch.zeros(n, k, 0, dtype=torch.long, device=dev)
    cmax = max(counts)
    pre = torch.zeros(n, cmax + 1, dtype=torch.long, device=dev)
    for j, c in enumerate(counts):
        pre[j, 0] = SOS
        pre[j, 1:c + 1] = torch.tensor(toks[:c], device=dev)
    cnt = torch.tensor(counts, device=dev)
    mask = (torch.arange(t, device=dev)[None] < torch.tensor(cuts, device=dev)[:, None])[:, None, :]   # [N,1,T]
    for s in range(depth):
        ys = torch.zeros(n, k, cmax + 1 + s, dtype=torch.long, device=dev)
        ys[:, :, :cmax + 1] = pre[:, None]
        for q in range(s):   # extension tokens go right after each row's own prefix
            ys.scatter_(2, (cnt[:, None, None] + 1 + q).expand(n, k, 1), ext[:, :, q:q + 1])
        flat = ys.view(n * k, -1)
        if dec.cross:
            logits = dec(flat, mem[None].expand(n * k, -1, -1), mask.repeat_interleave(k, 0))
        else:
            logits = dec(flat)
        at = (cnt[:, None] + s).expand(n, k).reshape(-1)
        logp = logits[torch.arange(n * k, device=dev), at].float().log_softmax(-1).view(n, k, -1)
        logp[..., 0] = -1e30
        cand = (score[..., None] + logp).flatten(1)
        score, idx = cand.topk(k, -1)
        src, tok = idx // logp.shape[-1], idx % logp.shape[-1]
        ext = torch.cat([ext.gather(1, src[..., None].expand(-1, -1, s)), tok[..., None]], 2)
    return ext


def tally(acc, name, paths, tgt):
    for n in NS:
        valid = (tgt[:, :n] != -100).all(1)
        eq = (paths[:, :, :n] == tgt[:, None, :n]).all(-1)
        for k in KS:
            a = acc.setdefault((name, k, n), [0, 0])
            a[0] += int((eq[:, :k].any(1) & valid).sum()); a[1] += int(valid.sum())
    d = acc.setdefault((name, 'distinct'), [0, 0])
    d[0] += sum(len({tuple(p[:4]) for p in ex.tolist()}) for ex in paths[:, :16]); d[1] += len(paths)


def bench(fgn, ar, reps=30):
    """CPU, 1 thread: wall time to propose 16 paths of 4 tokens after a 12-token prefix with 64 audio frames."""
    torch.set_num_threads(1)
    fgn, d = fgn.cpu(), fgn.d
    enc = torch.randn(64, d)
    prefix = torch.randint(2, 500, (1, fgn.P)); pm = torch.ones(1, fgn.P, dtype=torch.bool)
    am = torch.ones(1, fgn.A, dtype=torch.bool)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        ctx, pad = fgn.context(prefix, pm, enc[None], am)
        fgn.members(ctx, pad, 16).argmax(-1)
        ts.append(time.perf_counter() - t)
    out = {'fgn_16x%d_ms' % fgn.H: round(1000 * statistics.median(ts[5:]), 2)}
    p = Predictor(ar.cpu())
    ts = []
    for _ in range(reps):
        p.set_memory(enc)
        node = Node(SOS, None); p.step([node])
        for _ in range(12):
            node = Node(5, node); p.step([node])
        t = time.perf_counter()
        p.set_memory(enc)
        front = [node]
        for _ in range(4):
            cand = []
            for nd in front:
                for tk in nd.logp[2:].topk(16).indices.tolist()[:16 // len(front) or 1]:
                    cand.append(Node(tk + 2, nd))
            front = cand[:16]
            p.step(front)
        ts.append(time.perf_counter() - t)
    out['ar_beam16x4_ms'] = round(1000 * statistics.median(ts[5:]), 2)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='runs/stage1/avg.pt')
    ap.add_argument('--fgn', default='runs/fgn_s1/best.pt')
    ap.add_argument('--fgn2', default='runs/fgn_s2/best.pt')
    ap.add_argument('--slot', default='runs/fgn_slot/best.pt')
    ap.add_argument('--nar', default='runs/fgn_none/best.pt')
    ap.add_argument('--audio', default='runs/stage2_audio/best.pt')
    ap.add_argument('--text', default='runs/stage2_text/best.pt')
    ap.add_argument('--splits', nargs='+', default=['dev-clean', 'dev-other', 'test-clean'])
    ap.add_argument('--chunks', type=int, nargs='+', default=[8, 16])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--cuts', type=int, default=8)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default='runs/fgn_eval.json')
    a = ap.parse_args()
    dev = a.device
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / a.base, map_location='cpu')
    base = AVS(**ck['cfg']); base.load_state_dict(ck['model']); base.to(dev).eval()
    models = {}
    for name, path in (('fgn', a.fgn), ('fgn2', a.fgn2), ('slot', a.slot), ('nar', a.nar)):
        if (ROOT / path).exists():
            models[name] = load_fgn(path, dev)
    ars = {name: load_ar(path, dev) for name, path in (('ar_audio', a.audio), ('ar_text', a.text)) if (ROOT / path).exists()}
    ref = models['fgn']
    gen = torch.Generator().manual_seed(5)
    results = {}
    for split in a.splits:
        ds = FeatSet(split, tok)
        align = torch.load((ROOT / a.base).parent / 'align' / f'{split}.pt')
        subset = random.Random(21).sample(range(len(ds)), min(a.n, len(ds)))
        for chunk in a.chunks:
            acc, rng = {}, random.Random(77)
            for idxs in ds.batches(40000, shuffle=False, subset=subset):
                mel, lens, _, _ = ds.collate(idxs, dev)
                enc, el = base.encoder(mel, lens, chunk, LEFT)
                bidx, cut, prefix, pmask, aidx, tgt = build_examples(idxs, ds.tokens, align[f'c{chunk}'], el.cpu(), chunk, rng, a.cuts, ref)
                audio = enc[bidx.to(dev)[:, None], aidx.clamp(min=0).to(dev)]
                args = (prefix.to(dev), pmask.to(dev), audio, (aidx >= 0).to(dev))
                probs = {}
                for name, m in models.items():
                    ctx, pad = m.context(*args)
                    if m.noise == 'none':
                        tally(acc, name, topk_product(m.sample(ctx, pad, None).float().log_softmax(-1), 16).cpu(), tgt)
                    else:
                        probs[name] = m.members(ctx, pad, 16, gen).argmax(-1).cpu()   # [N,16,H]
                        tally(acc, name, probs[name], tgt)
                p = probs['fgn']
                shuf = torch.stack([p[:, torch.randperm(16, generator=gen), h] for h in range(p.shape[2])], 2)
                tally(acc, 'fgn_shuf', shuf, tgt)
                if 'fgn2' in probs:   # interleave so the first K members are split evenly over the two models
                    tally(acc, 'fgn_x2', torch.stack([p[:, :8], probs['fgn2'][:, :8]], 2).flatten(1, 2), tgt)
                for name, dec in ars.items():
                    out = torch.zeros(len(tgt), 16, 4, dtype=torch.long)
                    for b, i in enumerate(idxs):
                        sel = (bidx == b).nonzero()[:, 0]
                        if not len(sel):
                            continue
                        sp = align[f'c{chunk}'][i].tolist()
                        cuts = cut[sel].tolist()
                        counts = [sum(s < c for s in sp) for c in cuts]
                        toks = ds.tokens[i] + [SOS]   # eos shares the sos id
                        out[sel] = ar_beam(dec, toks, counts, cuts, enc[b, :int(el[b])].float(), 16, 4, dev).cpu()
                    # beams are sorted by score, so the first K rows of a beam-16 search approximate beam K from above
                    tally(acc, name, out, tgt)
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
    if 'ar_audio' in ars:
        results['bench_cpu_1thread'] = bench(ref, ars['ar_audio'])
        print('\nbench', json.dumps(results['bench_cpu_1thread']))
    json.dump(results, (ROOT / a.out).open('w'), indent=1)


if __name__ == '__main__':
    main()
