#!/usr/bin/env python3
"""G0: how many not-yet-heard tokens could be committed early, and at what precision?

At every chunk boundary a (frames), the true prefix = tokens whose CTC spike is before a. Each predictor
names the next token and a confidence, seeing audio only up to a. A token is committed early at the first
boundary where the confidence passes a threshold. Without early commit the token is committed at the end
of the chunk that contains its spike; lead = that time minus a.

The threshold is chosen on the first split (precision >= target) and then applied unchanged to the others.
Teacher-forced prefix: an upper bound on what a real decoder could do.
"""
import argparse, json, math, random, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS, Decoder  # noqa: E402
from avs.fgn import FGN  # noqa: E402
from avs.text import Tokenizer, SOS  # noqa: E402

LEFT, FRAME_MS = 64, 40


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


def ar_next(dec, toks, counts, cuts, mem, dev):
    """Next-token distribution after the true prefix at each cutoff of one utterance -> [N,V]."""
    n, t, cmax = len(cuts), mem.shape[0], max(counts)
    ys = torch.zeros(n, cmax + 1, dtype=torch.long, device=dev)
    ys[:, 0] = SOS
    for j, c in enumerate(counts):
        ys[j, 1:c + 1] = torch.tensor(toks[:c], device=dev)
    if dec.cross:
        mask = (torch.arange(t, device=dev)[None] < torch.tensor(cuts, device=dev)[:, None])[:, None, :]
        logits = dec(ys, mem[None].expand(n, -1, -1), mask)
    else:
        logits = dec(ys)
    lp = logits[torch.arange(n, device=dev), torch.tensor(counts, device=dev)].float()
    lp[:, 0] = -1e30
    return lp.softmax(-1).cpu()


def curve(recs, ntok, th):
    """recs: list of (utt, tokidx, conf, correct, lead) in boundary order. First boundary passing th commits."""
    seen, ok, bad, lead = set(), 0, 0, 0.0
    for u, c, conf, good, ld in recs:
        if conf < th or (u, c) in seen:
            continue
        seen.add((u, c))
        if good:
            ok += 1; lead += ld
        else:
            bad += 1
    n = ok + bad
    return {'precision': round(100 * ok / n, 2) if n else None, 'coverage': round(100 * ok / ntok, 2),
            'wrong_per_100tok': round(100 * bad / ntok, 2), 'lead_ms': round(lead / ok, 1) if ok else None,
            'mean_saving_ms_per_tok': round(lead / ntok, 1)}


def pick(recs, ntok, target):
    """Lowest threshold whose precision is >= target (max coverage)."""
    best = None
    for th in sorted({round(r[2], 3) for r in recs}):
        c = curve(recs, ntok, th)
        if c['precision'] is not None and c['precision'] >= target:
            return th, c
    return best, None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='runs/stage1/avg.pt')
    ap.add_argument('--splits', nargs='+', default=['dev-clean', 'test-clean', 'dev-other'])
    ap.add_argument('--chunks', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default='runs/g0_early_commit.json')
    a = ap.parse_args()
    dev = a.device
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / a.base, map_location='cpu')
    base = AVS(**ck['cfg']); base.load_state_dict(ck['model']); base.to(dev).eval()
    f1, f2, nar = (load_fgn(f'runs/fgn_{x}/best.pt', dev) for x in ('s1', 's2', 'none'))
    ars = {'ar_audio': load_ar('runs/stage2_audio/best.pt', dev), 'ar_text': load_ar('runs/stage2_text/best.pt', dev)}
    P, A, d = f1.P, f1.A, f1.d
    gen = torch.Generator().manual_seed(3)
    results, thresholds = {}, {}
    for si, split in enumerate(a.splits):
        ds = FeatSet(split, tok)
        align = torch.load((ROOT / a.base).parent / 'align' / f'{split}.pt')
        subset = random.Random(21).sample(range(len(ds)), min(a.n, len(ds)))
        for chunk in a.chunks:
            recs, ntok = {}, 0
            for idxs in ds.batches(40000, shuffle=False, subset=subset):
                mel, lens, _, _ = ds.collate(idxs, dev)
                enc, el = base.encoder(mel, lens, chunk, LEFT)
                for b, i in enumerate(idxs):
                    al = align[f'c{chunk}'][i]
                    if al is None:
                        continue
                    sp, toks, t = al.tolist(), ds.tokens[i], int(el[b])
                    ntok += len(toks)
                    cuts = [chunk * j for j in range(1, t // chunk + 1)]
                    counts = [sum(s < c for s in sp) for c in cuts]
                    keep = [(c, k) for c, k in zip(cuts, counts) if k < len(toks)]   # a next token exists
                    if not keep:
                        continue
                    cuts, counts = [c for c, _ in keep], [k for _, k in keep]
                    tgt = torch.tensor([toks[k] for k in counts])
                    lead = [FRAME_MS * (math.ceil((sp[k] + 1) / chunk) * chunk - c) for c, k in keep]
                    e = enc[b, :t].float()
                    prefix = torch.tensor([[0] * (P - len(toks[max(0, k - P):k])) + toks[max(0, k - P):k] for k in counts], device=dev)
                    pmask = torch.tensor([[False] * (P - min(P, k)) + [True] * min(P, k) for k in counts], device=dev)
                    aidx = torch.tensor(cuts)[:, None] - A + torch.arange(A)[None]
                    audio = e[aidx.clamp(min=0).to(dev)]
                    amask = (aidx >= 0).to(dev)
                    out = {}
                    votes = []
                    for m in (f1, f2):
                        ctx, pad = m.context(prefix, pmask, audio, amask)
                        votes.append(m.members(ctx, pad, a.k, gen)[:, :, 0].cpu())      # [N,k,V]
                    for name, pr in (('fgn', votes[0]), ('fgn_x2', torch.cat([votes[0][:, :a.k // 2], votes[1][:, :a.k // 2]], 1))):
                        hard = torch.zeros(pr.shape[0], pr.shape[2]).scatter_add_(1, pr.argmax(-1), torch.ones(pr.shape[:2])) / pr.shape[1]
                        out[name + '_vote'] = hard.max(-1)          # share of members naming the modal token
                        out[name + '_mean'] = pr.mean(1).max(-1)    # ensemble-mean probability
                    ctx, pad = nar.context(prefix, pmask, audio, amask)
                    out['nar'] = nar.sample(ctx, pad, None)[:, 0].float().softmax(-1).cpu().max(-1)
                    for name, dec in ars.items():
                        out[name] = ar_next(dec, toks, counts, cuts, e, dev).max(-1)
                    # agreement gate: AR names the token, FGN vote share must back the same token
                    agree = (out['ar_audio'][1] == out['fgn_x2_vote'][1]).float()
                    out['ar_audio*fgn'] = (torch.minimum(out['ar_audio'][0], out['fgn_x2_vote'][0]) * agree, out['ar_audio'][1])
                    for name, mx in out.items():
                        good = (mx[1] == tgt).tolist()
                        conf = mx[0].tolist()
                        recs.setdefault(name, []).extend((i, k, conf[j], good[j], lead[j]) for j, k in enumerate(counts))
            key = f'{split}/c{chunk}'
            results[key] = {'tokens': ntok}
            print(f'\n## {split} chunk {chunk * FRAME_MS} ms  ({ntok} tokens)')
            print('| method | top-1 at boundaries | target | threshold | precision | coverage of all tokens | wrong /100 tok | lead ms | mean saving ms/tok |')
            print('|---|---|---|---|---|---|---|---|---|')
            for name, r in recs.items():
                top1 = round(100 * sum(x[3] for x in r) / len(r), 2)
                results[key][name] = {'top1': top1}
                for target in (95, 90):
                    if si == 0:
                        th, _ = pick(r, ntok, target)
                        thresholds[(name, chunk, target)] = th
                    th = thresholds[(name, chunk, target)]
                    c = curve(r, ntok, th) if th is not None else None
                    results[key][name][f'p{target}'] = {'threshold': th, **(c or {})}
                    if c:
                        print(f'| {name} | {top1} | {target} | {th} | {c["precision"]} | {c["coverage"]} | {c["wrong_per_100tok"]} | {c["lead_ms"]} | {c["mean_saving_ms_per_tok"]} |', flush=True)
                    else:
                        print(f'| {name} | {top1} | {target} | - | not reachable | 0 | 0 | - | 0 |', flush=True)
    json.dump(results, (ROOT / a.out).open('w'), indent=1)


if __name__ == '__main__':
    main()
