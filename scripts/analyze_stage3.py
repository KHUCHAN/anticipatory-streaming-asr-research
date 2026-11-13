"""Summarise Stage-3 result files: per-method table and paired bootstrap CIs over utterances.

    .venv/bin/python scripts/analyze_stage3.py runs/stage3/main/dev-clean.json [...]
"""
import json
import random
import sys

PAIRS = [('B0', 'B1'), ('B0', 'B1a'), ('B0', 'B3'), ('B1', 'B1a'), ('B1', 'B2'), ('B1a', 'B3'), ('B1a', 'B1f')]
COLS = ['wer', 'lat_p50_ms', 'lat_p95_ms', 'onpath_steps_per_tok', 'spec_steps_per_tok', 'refresh_steps_per_tok',
        'reuse_rate', 'spec_waste', 'onpath_pred_ms_per_chunk', 'spec_ms_per_chunk', 'peak_nodes', 'peak_pred_state_kb', 'rtf', 'late_chunk_frac']


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float('nan')


def boot(a, b, reps=2000, seed=0):
    """a, b: per-utterance records of two methods on the same utterances. Returns CIs of (b - a)."""
    rng = random.Random(seed)
    n = len(a)
    dw, dl = [], []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        nref = sum(a[i]['nref'] for i in idx)
        dw.append(100 * (sum(b[i]['err'] for i in idx) - sum(a[i]['err'] for i in idx)) / nref)
        dl.append(1000 * (med([x for i in idx for x in b[i]['lat']]) - med([x for i in idx for x in a[i]['lat']])))
    dw.sort(); dl.sort()
    lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
    return (dw[lo], dw[hi]), (dl[lo], dl[hi])


def main():
    out = {}
    for path in sys.argv[1:]:
        d = json.load(open(path))
        split = d['args']['split']
        print(f'\n## {split}  (n={d["results"][0]["utts"]}, w_text={d["args"]["w_text"]}, w_audio={d["args"]["w_audio"]}, hold={d["args"]["hold"]})')
        print('| chunk | method | ' + ' | '.join(COLS) + ' |')
        print('|' + '---|' * (len(COLS) + 2))
        for r in d['results']:
            print(f'| {r["chunk_ms"]} ms | {r["method"]} | ' + ' | '.join(str(r[c]) for c in COLS) + ' |')
        chunks = sorted({r['chunk'] for r in d['results']})
        print('\npaired bootstrap 95% CI of (second - first):')
        for c in chunks:
            for x, y in PAIRS:
                kx, ky = f'{x}/c{c}', f'{y}/c{c}'
                if kx not in d['per_utt'] or ky not in d['per_utt']:
                    continue
                a, b = d['per_utt'][kx], d['per_utt'][ky]
                assert [u['utt'] for u in a] == [u['utt'] for u in b]
                (wl, wh), (ll, lh) = boot(a, b)
                print(f'  c{c:<2} {x:>3} -> {y:<3}  dWER [{wl:+.2f}, {wh:+.2f}] pt   dP50 latency [{ll:+.1f}, {lh:+.1f}] ms')
                out[f'{split}/c{c}/{x}->{y}'] = {'dwer': [round(wl, 3), round(wh, 3)], 'dlat_p50_ms': [round(ll, 2), round(lh, 2)]}
    json.dump(out, open('runs/stage3/bootstrap.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
