#!/usr/bin/env python3
"""Corrections to the H1/H3 analysis.

A. Gate comparison redone: (1) the FGN forward time is added to the time a forecast appears, (2) gates are
   compared at the SAME retraction rate (threshold sweep on the evaluation set, interpolated), and
   (3) precision is also reported against the reference transcript, not only against the decoder.
B. The two-pass display is replayed as an actual screen over time: the slow pass owns the text up to the audio
   it has decoded, the fast pass (plus forecast) supplies the tail. Reported: when each finally-correct word
   becomes stable on screen, when it first appears, and how many on-screen words are rewritten.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_final import Gate, SETS, labelled, match_words, pct, tok, words_with_time, boot_mean  # noqa: E402

USES_FGN = {'raw': False, 'lr_all': True, 'lr_no_fgn': False}
load = lambda name: torch.load(ROOT / f'runs/final/{name}.pt')['chunks']


def token_correct(ref, hyp):
    """Set of hyp token positions that an edit-distance alignment matches to the reference."""
    return {j for _, j in match_words(ref, hyp)[1]}


def simulate(u, score, th, extra_fgn):
    """-> display time per committed token, positions shown early, retracted count."""
    pend, etimes, early, rev, pos = None, [], [], 0, 0
    for b in u['bounds']:
        new = b['new']
        tms = [b['t']] * len(new)
        if pend and new:
            if new[0] == pend[0]:
                tms[0] = min(pend[1], b['t']); early.append(pos)
            else:
                rev += 1
            pend = None
        etimes += tms; pos += len(new)
        if b['f'] is not None and score(b['f']) >= th:
            g = b['g'][0][0]
            if pend is None:
                pend = (g, b['t'] + b['tg'] + (b['tn'] if extra_fgn else 0.0))
            elif pend[0] != g:
                rev += 1; pend = (g, b['t'] + b['tg'] + (b['tn'] if extra_fgn else 0.0))
        elif pend:
            rev += 1; pend = None
    rev += pend is not None
    return etimes, early, rev


def evaluate(utts, score, th, extra_fgn):
    lat, ok, okref, rev, ntok, per = [], 0, 0, 0, 0, []
    for u in utts:
        et, early, r = simulate(u, score, th, extra_fgn)
        good = u.setdefault('_good', token_correct(u['ref_ids'], u['ids']))
        hyp, ref = words_with_time(u['ids'], et), words_with_time(u['ref_ids'], u['ref_t'])
        pairs = u.setdefault('_pairs', match_words([w for w, _ in ref], [w for w, _ in hyp])[1])
        ul = [hyp[j][1] - ref[r_][1] for r_, j in pairs]
        lat += ul; per.append(ul); ok += len(early); okref += sum(p in good for p in early); rev += r; ntok += len(u['ids'])
    return {'mean': 1000 * sum(lat) / len(lat), 'p50': 1000 * pct(lat, 0.5), 'p95': 1000 * pct(lat, 0.95), 'early': 100 * ok / ntok,
            'retracted': 100 * rev / ntok, 'prec_dec': 100 * ok / max(1, ok + rev), 'prec_ref': 100 * okref / max(1, ok + rev), '_per': per}


def at_retraction(curve, target):
    """Interpolate a threshold sweep (sorted by retraction) at a retraction rate."""
    curve = sorted(curve, key=lambda r: r['retracted'])
    for a, b in zip(curve, curve[1:]):
        if a['retracted'] <= target <= b['retracted']:
            w = (target - a['retracted']) / max(1e-9, b['retracted'] - a['retracted'])
            return {k: a[k] + w * (b[k] - a[k]) for k in ('mean', 'p50', 'p95', 'early', 'prec_dec', 'prec_ref')}
    return None


# ---------------------------------------------------------------- B: screen replay

def to_words(ids):
    out, cur = [], ''
    for i in ids:
        piece = tok.piece(i)
        if piece.startswith('▁') and cur:
            out.append(cur); cur = ''
        cur += piece.replace('▁', '')
    if cur:
        out.append(cur)
    return out


def match_prefix(ref, hyp):
    """Matched ref positions when hyp is a partial transcript: align hyp to the best-fitting PREFIX of ref.

    Aligning a partial screen against the whole reference lets a word match a later identical word, which makes
    words look unstable. Anchoring at the cheapest reference prefix (shortest on ties) removes that artefact.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i = min(range(n + 1), key=lambda k: (d[k][m], k))
    j, out = m, set()
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            out.add(i - 1); i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def replay(uf, us, cf, cs, score, th, extra_fgn):
    """Screen over time. us is None for single pass. -> (events [(t, words)], final words)."""
    ev = []
    fb, n = [], 0
    for k, b in enumerate(uf['bounds']):
        n += len(b['new'])
        last = k == len(uf['bounds']) - 1
        fc = [b['g'][0][0]] if (b['f'] is not None and score is not None and score(b['f']) >= th) else []
        fb.append(((k + 1) * cf if not last else 10 ** 9, n))
        ev.append((b['t'], 0, 'f', k, None))
        if fc:
            ev.append((b['t'] + b['tg'] + (b['tn'] if extra_fgn else 0.0), 1, 'c', k, fc))
    if us is not None:
        sb, n = [], 0
        for k, b in enumerate(us['bounds']):
            n += len(b['new'])
            last = k == len(us['bounds']) - 1
            sb.append(((k + 1) * cs if not last else 10 ** 9, n))
            ev.append((b['t'], 2, 's', k, None))
    ev.sort(key=lambda e: (e[0], e[1]))
    fk, sk, fc, out, prev = -1, -1, [], [], None
    for t, _, kind, k, payload in ev:
        if kind == 'f':
            fk, fc = k, []
        elif kind == 'c':
            fc = payload if k == fk else fc
        else:
            sk = k
        if us is None:
            ids = uf['ids'][:fb[fk][1]] + fc if fk >= 0 else []
        else:
            cover, ns = (sb[sk] if sk >= 0 else (0, 0))
            ids = us['ids'][:ns]
            if fk >= 0 and fb[fk][0] > cover:                       # fast pass is ahead of the slow pass
                j = next((i for i, (e, _) in enumerate(fb) if e > cover), None)
                start = fb[j - 1][1] if j else 0                    # fast tokens committed at boundaries beyond the covered audio
                ids = ids + uf['ids'][start:fb[fk][1]] + fc
        if ids != prev:
            out.append((t, to_words(ids))); prev = ids
    return out, to_words((us or uf)['ids'])


def screen_metrics(args):
    name, fast, slow, cf, cs, gate, th, extra = args
    score = gate
    stable, first, rewrites, nfinal, errs, nref = [], [], 0, 0, 0, 0
    for i, uf in enumerate(fast):
        us = slow[i] if slow is not None else None
        events, final = replay(uf, us, cf, cs, score, th, extra)
        ref = words_with_time((us or uf)['ref_ids'], (us or uf)['ref_t'])
        rw = [w for w, _ in ref]
        e, pairs = match_words(rw, final)
        keep = {r for r, _ in pairs}
        errs += e; nref += len(rw); nfinal += len(final)
        seen_first, last_bad = {}, {r: -1 for r in keep}
        prev = []
        for k, (t, words) in enumerate(events):
            m = match_prefix(rw, words) if k < len(events) - 1 else keep
            for r in keep:
                if r in m:
                    seen_first.setdefault(r, t)
                else:
                    last_bad[r] = k
            lcp = 0
            while lcp < len(prev) and lcp < len(words) and prev[lcp] == words[lcp]:
                lcp += 1
            growing = lcp == len(prev) - 1 and len(words) >= len(prev) and words[lcp].startswith(prev[lcp])
            if not growing:
                rewrites += len(prev) - lcp
            prev = words
        for r in keep:
            k = last_bad[r] + 1
            if k < len(events):
                stable.append(events[k][0] - ref[r][1]); first.append(seen_first[r] - ref[r][1])
    f = lambda xs, q: round(1000 * pct(xs, q), 1)
    return name, {'wer': round(100 * errs / nref, 2), 'stable_p50': f(stable, 0.5), 'stable_mean': round(1000 * sum(stable) / len(stable), 1),
                  'stable_p95': f(stable, 0.95), 'first_p50': f(first, 0.5), 'first_mean': round(1000 * sum(first) / len(first), 1),
                  'first_p95': f(first, 0.95), 'rewritten_words_per_100': round(100 * rewrites / nfinal, 2)}


def main():
    gate_sets = [load('gate_dev-clean'), load('gate_dev-other')]
    chunks = sorted(gate_sets[0])
    X, y = [], []
    for c in chunks:
        a, b = labelled([u for g in gate_sets for u in g[c]['utts']]); X += a; y += b
    scores = {'raw': lambda f: f[0], 'lr_all': Gate(SETS['lr_all']).fit(X, y), 'lr_no_fgn': Gate(SETS['lr_no_fgn']).fit(X, y)}
    out = {}
    splits = ('dev-clean', 'test-clean', 'dev-other', 'test-other')
    evals = {s: load(f'eval_{s}') for s in splits}

    print('# A  gates at the same retraction rate (FGN forward time included in the display time)')
    ths = [x / 100 for x in range(30, 100, 3)]
    for split in ([] if '--only-b' in sys.argv else splits):
        print(f'\n## {split}')
        print('| chunk | retraction /100 tok | gate | shown early % | mean ms | P50 | P95 | precision vs decoder | precision vs reference |')
        print('|---|---|---|---|---|---|---|---|---|')
        for c in chunks:
            utts = evals[split][c]['utts']
            curves = {n: [evaluate(utts, s, th, USES_FGN[n]) for th in ths] for n, s in scores.items()}
            for target in (0.5, 1.0):
                for n in scores:
                    r = at_retraction(curves[n], target)
                    if r is None:
                        print(f'| {c * 40} ms | {target} | {n} | out of sweep range | | | | | |'); continue
                    out[f'A/{split}/c{c}/ret{target}/{n}'] = {k: round(v, 2) for k, v in r.items()}
                    print(f'| {c * 40} ms | {target} | {n} | {r["early"]:.2f} | {r["mean"]:.1f} | {r["p50"]:.1f} | {r["p95"]:.1f} | {r["prec_dec"]:.1f} | {r["prec_ref"]:.1f} |', flush=True)
            # cost of the FGN forward on the display time: same threshold with and without it
            a_, b_ = evaluate(utts, scores['lr_all'], 0.87, False), evaluate(utts, scores['lr_all'], 0.87, True)
            out[f'A/{split}/c{c}/fgn_time_cost_ms'] = round(b_['mean'] - a_['mean'], 2)
            print(f'| {c * 40} ms | - | lr_all th 0.87: FGN time adds {b_["mean"] - a_["mean"]:.2f} ms to the mean, CI {boot_mean(a_["_per"], b_["_per"])} | | | | | | |')

    print('\n# B  screen replay (300 utterances per split, gate lr_all at th 0.87)')
    jobs = []
    for split in ('test-clean', 'test-other', 'dev-clean'):
        ev = evals[split]
        sub = {c: ev[c]['utts'][:300] for c in chunks}
        for c in chunks:
            jobs.append((f'{split}|single {c * 40} ms', sub[c], None, c, None, None, 0, False))
            jobs.append((f'{split}|single {c * 40} ms + forecast', sub[c], None, c, None, scores['lr_all'], 0.87, True))
        for cf, cs in ((4, 16), (8, 16), (4, 8)):
            jobs.append((f'{split}|{cf * 40} ms -> {cs * 40} ms', sub[cf], sub[cs], cf, cs, None, 0, False))
            jobs.append((f'{split}|{cf * 40} ms + forecast -> {cs * 40} ms', sub[cf], sub[cs], cf, cs, scores['lr_all'], 0.87, True))
    with Pool(8) as pool:
        res = pool.map(screen_metrics, jobs)
    cur = None
    for name, r in res:
        split, cfg = name.split('|')
        if split != cur:
            cur = split
            print(f'\n## {split}')
            print('| configuration | final WER | stable P50 | stable mean | stable P95 | first-seen P50 | first-seen mean | first-seen P95 | rewritten words /100 |')
            print('|---|---|---|---|---|---|---|---|---|')
        out[f'B/{split}/{cfg}'] = r
        print(f'| {cfg} | {r["wer"]} | {r["stable_p50"]} | {r["stable_mean"]} | {r["stable_p95"]} | {r["first_p50"]} | {r["first_mean"]} | {r["first_p95"]} | {r["rewritten_words_per_100"]} |')
    json.dump(out, (ROOT / ('runs/final/summary2b.json' if '--only-b' in sys.argv else 'runs/final/summary2.json')).open('w'), indent=1)


if __name__ == '__main__':
    main()
