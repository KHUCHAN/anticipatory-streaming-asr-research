#!/usr/bin/env python3
"""Checks to run before scaling up, from the boundary logs of scripts/collect_events.py (no training).

P1  Where does the latency gain come from? For every forecast that the next chunk confirmed, compare the
    reference time of that token (full-context alignment) with the moment the forecast was shown.
    "already spoken" = the token's reference frame lies inside audio that had arrived; the forecast then
    recovers the emission delay of the streaming CTC. "not yet spoken" = genuine anticipation.
    A transducer with FastEmit emits earlier, so only the second share is safe to expect at scale.
P2  Gate label: decoder agreement (used so far) vs reference correctness (planned for the scale-up).
P3  Threshold transfer: a threshold fitted on one dev half for a target retraction rate, applied to the other
    dev half and to test; pooled vs per-domain (clean/other) calibration.
P4  Does the forecast work better where recognition is better? Coverage and precision by utterance WER.
"""
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_final import Gate, SETS, xform, match_words, words_with_time  # noqa: E402

load = lambda name: torch.load(ROOT / f'runs/final/{name}.pt')['chunks']
arrive_time = lambda e: (160 * (4 * (e - 1) + 6) + 400) / 16000
FRAME = 0.04


def tok_align(u):
    """hyp token position -> ref token position, for tokens an edit-distance alignment matches."""
    if '_ta' not in u:
        u['_ta'] = {j: r for r, j in match_words(u['ref_ids'], u['ids'])[1]}
    return u['_ta']


def boundary_rows(utts, chunk):
    """One row per boundary that carries a forecast."""
    rows = []
    for ui, u in enumerate(utts):
        ta, pos, bs = tok_align(u), 0, u['bounds']
        ref_words = None
        for k, b in enumerate(bs):
            pos += len(b['new'])
            if b['f'] is None:
                continue
            nxt = next((c['new'] for c in bs[k + 1:] if c['new']), None)
            g = b['g'][0][0]
            confirmed = nxt is not None and nxt[0] == g
            r = ta.get(pos) if confirmed else None            # ref index of the confirmed token (None = decoder kept a wrong token)
            correct = confirmed and r is not None and u['ref_ids'][r] == g
            audio_end = arrive_time((k + 1) * chunk)          # audio that had arrived when the forecast was made
            rows.append({'u': ui, 'k': k, 'pos': pos, 'f': b['f'], 'confirmed': confirmed, 'correct': correct,
                         'ref_t': u['ref_t'][r] if correct else None, 'audio_end': audio_end, 't': b['t']})
    return rows


def simulate_rate(utts, score, th):
    """Retractions and confirmed/correct early tokens per 100 tokens for a gate threshold (first-token forecasts)."""
    ok = okref = rev = ntok = 0
    for u in utts:
        ta, pend, pos = tok_align(u), None, 0
        for b in u['bounds']:
            new = b['new']
            if pend is not None and new:
                if new[0] == pend:
                    ok += 1; okref += pos in ta
                else:
                    rev += 1
                pend = None
            pos += len(new)
            if b['f'] is not None and score(b['f']) >= th:
                g = b['g'][0][0]
                if pend is not None and pend != g:
                    rev += 1
                pend = g
            elif pend is not None:
                rev += 1; pend = None
        rev += pend is not None
        ntok += len(u['ids'])
    return {'early': 100 * ok / ntok, 'retracted': 100 * rev / ntok, 'wrong_shown': 100 * (rev + ok - okref) / ntok,
            'prec_dec': 100 * ok / max(1, ok + rev), 'prec_ref': 100 * okref / max(1, ok + rev)}


def fit_threshold(utts, score, key, target):
    """Lowest threshold (max coverage) with rate[key] <= target on these utterances."""
    best = 0.99
    for th in [x / 100 for x in range(99, 29, -1)]:
        if simulate_rate(utts, score, th)[key] <= target:
            best = th
        else:
            break
    return best


def at_rate(utts, score, key, target):
    """Coverage at a given error rate, interpolated on a threshold sweep of these utterances."""
    curve = sorted((simulate_rate(utts, score, th / 100) for th in range(30, 100, 3)), key=lambda r: r[key])
    for a, b in zip(curve, curve[1:]):
        if a[key] <= target <= b[key]:
            w = (target - a[key]) / max(1e-9, b[key] - a[key])
            return {k: a[k] + w * (b[k] - a[k]) for k in a}
    return None


def main():
    chunks = [4, 8, 16]
    gate = {s: load(f'gate_{s}') for s in ('dev-clean', 'dev-other')}
    evals = {s: load(f'eval_{s}') for s in ('dev-clean', 'test-clean', 'dev-other', 'test-other')}
    out = {}

    # gates: label = decoder agreement vs reference correctness, same features (lr_all)
    X, yd, yc = [], [], []
    for s in gate:
        for c in chunks:
            for r in boundary_rows(gate[s][c]['utts'], c):
                X.append(xform(r['f'])); yd.append(float(r['confirmed'])); yc.append(float(r['correct']))
    g_dec, g_ref = Gate(SETS['lr_all']).fit(X, yd), Gate(SETS['lr_all']).fit(X, yc)
    raw = lambda f: f[0]
    print(f'gate fit: {len(yd)} boundaries, confirmed {100 * sum(yd) / len(yd):.1f} %, correct {100 * sum(yc) / len(yc):.1f} %')

    print('\n# P1  source of the gain: confirmed-and-correct forecasts shown by gate lr_all (th 0.87)')
    print('| split | chunk | n | already spoken when shown | not yet spoken | median lead of ref time over audio end (ms) | share with ref time > 80 ms after audio end |')
    print('|---|---|---|---|---|---|---|')
    for s, ev in evals.items():
        for c in chunks:
            rows = [r for r in boundary_rows(ev[c]['utts'], c) if r['correct'] and g_dec(r['f']) >= 0.87]
            d = sorted(1000 * (r['ref_t'] - r['audio_end']) for r in rows)
            spoken = sum(x <= 0 for x in d)
            rec = {'n': len(d), 'already_spoken': round(100 * spoken / len(d), 1), 'not_yet': round(100 - 100 * spoken / len(d), 1),
                   'median_ms': round(d[len(d) // 2], 1), 'beyond_80ms': round(100 * sum(x > 80 for x in d) / len(d), 1)}
            out[f'P1/{s}/c{c}'] = rec
            print(f'| {s} | {c * 40} ms | {rec["n"]} | {rec["already_spoken"]} % | {rec["not_yet"]} % | {rec["median_ms"]} | {rec["beyond_80ms"]} % |', flush=True)

    print('\n# P2  gate label. Coverage at the same rate of wrong tokens shown (retracted + confirmed-but-wrong) = 1.0 per 100 tokens')
    print('| split | chunk | gate | shown early % | retracted /100 | wrong shown /100 | precision vs reference |')
    print('|---|---|---|---|---|---|---|')
    for s in ('test-clean', 'test-other'):
        for c in chunks:
            for name, sc in (('raw', raw), ('label=decoder', g_dec), ('label=reference', g_ref)):
                r = at_rate(evals[s][c]['utts'], sc, 'wrong_shown', 1.0)
                if r is None:
                    print(f'| {s} | {c * 40} ms | {name} | out of range | | | |'); continue
                out[f'P2/{s}/c{c}/{name}'] = {k: round(v, 2) for k, v in r.items()}
                print(f'| {s} | {c * 40} ms | {name} | {r["early"]:.2f} | {r["retracted"]:.2f} | {r["wrong_shown"]:.2f} | {r["prec_ref"]:.1f} |', flush=True)

    print('\n# P3  threshold transfer for a target of 0.5 retractions per 100 tokens (gate label=decoder), chunk 320 ms')
    print('| calibrated on | applied to | threshold | achieved retraction /100 | shown early % |')
    print('|---|---|---|---|---|')
    c = 8
    halves = {s: (gate[s][c]['utts'][:400], gate[s][c]['utts'][400:]) for s in gate}
    cal = {'dev-clean (half A)': halves['dev-clean'][0], 'dev-other (half A)': halves['dev-other'][0],
           'pooled dev (half A)': halves['dev-clean'][0] + halves['dev-other'][0]}
    targets = {'dev-clean (half B)': halves['dev-clean'][1], 'dev-other (half B)': halves['dev-other'][1],
               'test-clean': evals['test-clean'][c]['utts'], 'test-other': evals['test-other'][c]['utts']}
    for cn, cu in cal.items():
        th = fit_threshold(cu, g_dec, 'retracted', 0.5)
        for tn, tu in targets.items():
            r = simulate_rate(tu, g_dec, th)
            out[f'P3/{cn}->{tn}'] = {'th': th, 'retracted': round(r['retracted'], 2), 'early': round(r['early'], 2)}
            print(f'| {cn} | {tn} | {th} | {r["retracted"]:.2f} | {r["early"]:.2f} |', flush=True)

    print('\n# P4  forecast quality by utterance WER (gate label=decoder, th 0.87, chunk 320 ms, test-clean + test-other)')
    print('| utterance WER | utterances | shown early % | retracted /100 | precision vs reference |')
    print('|---|---|---|---|---|')
    utts = evals['test-clean'][8]['utts'] + evals['test-other'][8]['utts']
    for u in utts:
        ref, hyp = [w for w, _ in words_with_time(u['ref_ids'], u['ref_t'])], [w for w, _ in words_with_time(u['ids'], u['times'])]
        u['_wer'] = 100 * match_words(ref, hyp)[0] / max(1, len(ref))
    for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 60), (60, 1e9)):
        sub = [u for u in utts if lo <= u['_wer'] < hi]
        if not sub:
            continue
        r = simulate_rate(sub, g_dec, 0.87)
        out[f'P4/{lo}-{hi}'] = {'n': len(sub), **{k: round(v, 2) for k, v in r.items()}}
        print(f'| {lo}–{hi if hi < 1e9 else "∞"} % | {len(sub)} | {r["early"]:.2f} | {r["retracted"]:.2f} | {r["prec_ref"]:.1f} |', flush=True)
    json.dump(out, (ROOT / 'runs/final/prescale.json').open('w'), indent=1)


if __name__ == '__main__':
    main()
