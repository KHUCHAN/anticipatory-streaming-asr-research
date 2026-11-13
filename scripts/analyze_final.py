#!/usr/bin/env python3
"""Final experiments on the forecast-verify front end, evaluated offline from scripts/collect_events.py logs.

H1 learned confidence gate  : logistic regression on boundary features vs the raw predictor probability.
H2 forecast chain           : show up to 3 forecast tokens instead of 1.
H3 two-pass display         : small-chunk pass (with forecasts) for display, large-chunk pass for the final text.

Gates and thresholds are fitted on held-out dev utterances (runs/final/gate_*.pt) and applied unchanged to
the evaluation sets. The final transcript is never changed by a forecast, so WER is that of the decoder.
"""
import json, random, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FEATS  # noqa: E402
from avs.text import Tokenizer  # noqa: E402

tok = Tokenizer(FEATS / 'spm.model')
FEATURES = ['p1', 'p2', 'entropy', 'ctc_blank_last', 'ctc_tok_last', 'ctc_tok_max2', 'nar_p', 'nar_agree', 'continuation', 'n_new', 'chunk']
SETS = {'raw': None, 'lr_all': list(range(11)), 'lr_no_fgn': [0, 1, 2, 3, 4, 5, 8, 9, 10], 'lr_no_ctc': [0, 1, 2, 6, 7, 8, 9, 10]}


def words_with_time(ids, times):
    out, cur, t = [], '', 0.0
    for i, tm in zip(ids, times):
        piece = tok.piece(i)
        if piece.startswith('▁') and cur:
            out.append((cur, t)); cur = ''
        cur += piece.replace('▁', ''); t = tm
    if cur:
        out.append((cur, t))
    return out


def match_words(ref, hyp):
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return d[n][m], pairs


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float('nan')


def xform(f):
    """Feature vector -> model input (probabilities to logits, counts kept)."""
    lg = lambda p: torch.logit(torch.tensor(p).clamp(1e-4, 1 - 1e-4)).item()
    return [lg(f[0]), lg(f[1]), f[2], lg(f[3]), lg(f[4]), lg(f[5]), lg(f[6]), f[7], f[8], min(f[9], 4.0), f[10] / 8.0]


class Gate:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        X = torch.tensor(X)[:, self.cols]
        self.mu, self.sd = X.mean(0), X.std(0).clamp_min(1e-6)
        X = (X - self.mu) / self.sd
        y = torch.tensor(y, dtype=torch.float32)
        w = torch.zeros(X.shape[1] + 1, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w[:-1] + w[-1], y) + 1e-3 * (w[:-1] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.w = w.detach()
        return self

    def __call__(self, f):
        x = (torch.tensor(xform(f))[self.cols] - self.mu) / self.sd
        return float(torch.sigmoid(x @ self.w[:-1] + self.w[-1]))


def labelled(utts):
    """(features, confirmed) for every boundary with a forecast that the next commit could verify."""
    X, y = [], []
    for u in utts:
        bs = u['bounds']
        for k, b in enumerate(bs):
            if b['f'] is None:
                continue
            nxt = next((c['new'] for c in bs[k + 1:] if c['new']), None)
            if nxt is None:
                X.append(xform(b['f'])); y.append(0.0)
            else:
                X.append(xform(b['f'])); y.append(float(nxt[0] == b['g'][0][0]))
    return X, y


def simulate(u, score, th, depth=1, th_chain=None):
    """Display timeline of one utterance. Returns per-token display times, confirmed and retracted counts."""
    th_chain = th if th_chain is None else th_chain
    pend, etimes, ok, rev = [], [], 0, 0
    for b in u['bounds']:
        new = b['new']
        tms = [b['t']] * len(new)
        j = 0
        while j < len(pend) and j < len(new) and pend[j][0] == new[j]:
            tms[j] = pend[j][1]; ok += 1; j += 1
        if j < len(pend) and j < len(new):   # mismatch with the verified audio
            rev += len(pend) - j; pend = []
        else:
            pend = pend[j:]
        etimes += tms
        want = []
        if b['f'] is not None and score(b['f']) >= th:
            want = [b['g'][0][0]]
            for g, p in b['g'][1:depth]:
                if p < th_chain:
                    break
                want.append(g)
        k = 0
        while k < len(pend) and k < len(want) and pend[k][0] == want[k]:
            k += 1
        rev += len(pend) - k
        pend = pend[:k] + [(g, b['t'] + b['tg']) for g in want[k:]]
    rev += len(pend)
    return etimes, ok, rev


def evaluate(utts, score, th, depth=1, th_chain=None, base=False):
    lat, ok, rev, ntok, errs, nref, per = [], 0, 0, 0, 0, 0, []
    for u in utts:
        if base:
            et, o, r = u['times'], 0, 0
        else:
            et, o, r = simulate(u, score, th, depth, th_chain)
        hyp = words_with_time(u['ids'], et)
        ref = words_with_time(u['ref_ids'], u['ref_t'])
        e, pairs = match_words([w for w, _ in ref], [w for w, _ in hyp])
        ul = [hyp[j][1] - ref[r_][1] for r_, j in pairs]
        lat += ul; ok += o; rev += r; ntok += len(u['ids']); errs += e; nref += len(ref); per.append(ul)
    return {'wer': round(100 * errs / nref, 2), 'p50': round(1000 * pct(lat, 0.5), 1), 'mean': round(1000 * sum(lat) / len(lat), 1),
            'p95': round(1000 * pct(lat, 0.95), 1), 'early': round(100 * ok / ntok, 2), 'retracted': round(100 * rev / ntok, 2),
            'precision': round(100 * ok / max(1, ok + rev), 1), '_per': per}


def boot_mean(a, b, reps=1000, seed=0):
    """95% CI of the difference in mean latency (b - a), ms, resampling utterances."""
    rng, n, out = random.Random(seed), len(a), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        xa = [x for i in idx for x in a[i]]; xb = [x for i in idx for x in b[i]]
        out.append(1000 * (sum(xb) / len(xb) - sum(xa) / len(xa)))
    out.sort()
    return round(out[int(0.025 * reps)], 1), round(out[int(0.975 * reps) - 1], 1)


def pick_threshold(utts, score, target, depth=1):
    """Lowest threshold (max coverage) whose precision on the fitting set is >= target."""
    for th in [x / 100 for x in range(30, 100)]:
        r = evaluate(utts, score, th, depth)
        if r['precision'] >= target and r['early'] > 0:
            return th
    return 0.99


def two_pass(fast, slow, score, th, depth):
    """Fast pass shows words (with forecasts); slow pass owns the final text."""
    lat, changed, nslow, errs, nref = [], 0, 0, 0, 0
    for uf, us in zip(fast, slow):
        assert uf['utt'] == us['utt']
        et, _, _ = simulate(uf, score, th, depth)
        fw, sw = words_with_time(uf['ids'], et), words_with_time(us['ids'], us['times'])
        ref = words_with_time(us['ref_ids'], us['ref_t'])
        rw = [w for w, _ in ref]
        e, ps = match_words(rw, [w for w, _ in sw])
        _, pf = match_words(rw, [w for w, _ in fw])
        ft = {r: fw[j][1] for r, j in pf}
        lat += [min(sw[j][1], ft.get(r, 1e9)) - ref[r][1] for r, j in ps]
        d, _ = match_words([w for w, _ in sw], [w for w, _ in fw])
        changed += d; nslow += len(sw); errs += e; nref += len(ref)
    return {'wer': round(100 * errs / nref, 2), 'p50': round(1000 * pct(lat, 0.5), 1), 'mean': round(1000 * sum(lat) / len(lat), 1),
            'p95': round(1000 * pct(lat, 0.95), 1), 'changed_words_per_100': round(100 * changed / nslow, 2)}


def main():
    load = lambda name: torch.load(ROOT / f'runs/final/{name}.pt')['chunks']
    gate_sets = [load('gate_dev-clean'), load('gate_dev-other')]
    chunks = sorted(gate_sets[0])
    fit_utts = {c: [u for g in gate_sets for u in g[c]['utts']] for c in chunks}
    X, y = [], []
    for c in chunks:
        a, b = labelled(fit_utts[c]); X += a; y += b
    print(f'gate fitting set: {len(y)} boundaries, {100 * sum(y) / len(y):.1f}% confirmed')
    scores = {'raw': lambda f: f[0]}
    for name, cols in SETS.items():
        if cols is not None:
            scores[name] = Gate(cols).fit(X, y)
            print(name, 'weights', {FEATURES[c]: round(float(w), 2) for c, w in zip(cols, scores[name].w[:-1])})
    out = {}
    evals = {s: load(f'eval_{s}') for s in ('dev-clean', 'test-clean', 'dev-other', 'test-other')}
    strip = lambda r: {k: v for k, v in r.items() if k != '_per'}

    print('\n# H1  confidence gate (1 forecast token). Threshold fitted for 95 % precision on held-out dev, per chunk size.')
    ths = {(n, c): pick_threshold(fit_utts[c], s, 95.0) for n, s in scores.items() for c in chunks}
    for split, ev in evals.items():
        print(f'\n## {split}')
        print('| chunk | gate | th | WER | P50 | mean | P95 | shown early % | retracted /100 | precision | mean vs raw (95% CI) |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for c in chunks:
            utts = ev[c]['utts']
            base = evaluate(utts, None, 0, base=True)
            out[f'H1/{split}/c{c}/none'] = strip(base)
            print(f'| {c * 40} ms | none | - | {base["wer"]} | {base["p50"]} | {base["mean"]} | {base["p95"]} | 0 | 0 | - | - |')
            raw = None
            for n, s in scores.items():
                r = evaluate(utts, s, ths[(n, c)])
                raw = r if n == 'raw' else raw
                ci = '-' if n == 'raw' else '%s … %s' % boot_mean(raw['_per'], r['_per'])
                out[f'H1/{split}/c{c}/{n}'] = {**strip(r), 'th': ths[(n, c)], 'ci_mean_vs_raw': ci}
                print(f'| {c * 40} ms | {n} | {ths[(n, c)]} | {r["wer"]} | {r["p50"]} | {r["mean"]} | {r["p95"]} | {r["early"]} | {r["retracted"]} | {r["precision"]} | {ci} |')

    best = 'lr_all'
    print(f'\n# H2  forecast chain with gate {best}. First token as in H1; later tokens need predictor probability >= th_chain.')
    for split, ev in evals.items():
        print(f'\n## {split}')
        print('| chunk | depth | th_chain | P50 | mean | P95 | shown early % | retracted /100 | precision |')
        print('|---|---|---|---|---|---|---|---|---|')
        for c in chunks:
            utts = ev[c]['utts']
            for depth, tc in ((1, None), (2, 0.9), (2, 0.7), (3, 0.9), (3, 0.7)):
                r = evaluate(utts, scores[best], ths[(best, c)], depth, tc)
                out[f'H2/{split}/c{c}/d{depth}_{tc}'] = strip(r)
                print(f'| {c * 40} ms | {depth} | {tc or "-"} | {r["p50"]} | {r["mean"]} | {r["p95"]} | {r["early"]} | {r["retracted"]} | {r["precision"]} |')

    print(f'\n# H3  two-pass display: fast pass (gate {best}, depth 1) shows words, slow pass owns the final text.')
    for split, ev in evals.items():
        print(f'\n## {split}')
        print('| display pass | final pass | final WER | display P50 | mean | P95 | words changed by final /100 | encoder cost ms per s of audio |')
        print('|---|---|---|---|---|---|---|---|')
        cost = lambda c: 1000 * ev[c]['enc_s'] / (c * 0.04)
        for c in chunks:
            r = evaluate(ev[c]['utts'], scores[best], ths[(best, c)])
            out[f'H3/{split}/single_c{c}'] = {**strip(r), 'enc_ms_per_s': round(cost(c), 1)}
            print(f'| {c * 40} ms + forecast | same | {r["wer"]} | {r["p50"]} | {r["mean"]} | {r["p95"]} | {r["retracted"]} (tokens) | {cost(c):.1f} |')
        for cf, cs in ((4, 8), (4, 16), (8, 16)):
            for fc in (False, True):
                r = two_pass(ev[cf]['utts'], ev[cs]['utts'], scores[best], ths[(best, cf)] if fc else 2.0, 1)
                out[f'H3/{split}/fast{cf}_slow{cs}_{"fc" if fc else "plain"}'] = {**r, 'enc_ms_per_s': round(cost(cf) + cost(cs), 1)}
                print(f'| {cf * 40} ms{" + forecast" if fc else ""} | {cs * 40} ms | {r["wer"]} | {r["p50"]} | {r["mean"]} | {r["p95"]} | {r["changed_words_per_100"]} | {cost(cf) + cost(cs):.1f} |')
    json.dump(out, (ROOT / 'runs/final/summary.json').open('w'), indent=1)


if __name__ == '__main__':
    main()
