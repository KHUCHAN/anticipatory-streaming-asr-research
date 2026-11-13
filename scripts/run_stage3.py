#!/usr/bin/env python3
"""Stage 3: predict-verify streaming decoding, B0-B3, with a paced-replay clock.

Audio is released at real-time pace. A chunk can be processed only after its last
mel frame has arrived and after the system has finished earlier work (speculation
included). Compute times are measured on this machine and added to the clock.
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS, Decoder  # noqa: E402
from avs.fgn import FGN, FGNPrior, ShortPrior  # noqa: E402
from avs.stream import Decoder3, Predictor, WindowPredictor  # noqa: E402
from avs.text import Tokenizer, edit_distance  # noqa: E402

LEFT = 64
frame_time = lambda f: (160 * (4 * f + 3) + 200) / 16000   # centre of encoder frame f
arrive_time = lambda e: (160 * (4 * (e - 1) + 6) + 400) / 16000  # audio needed for frames < e


def words_with_time(tok, ids, times):
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
    """Edit-distance alignment -> (errors, [(ref_idx, hyp_idx)] of exact matches)."""
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


def bench_encoder(model, chunk, reps=60):
    """Median wall time of one true incremental encoder step (CPU)."""
    enc = model.encoder
    cache = enc.stream_init()
    mel = torch.randn(1, 4 * (chunk - 1) + 7, 80)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); enc.stream_chunk(mel, cache, LEFT); ts.append(time.perf_counter() - t)
    return statistics.median(ts[10:]), enc.stream_state_bytes(cache)


METHODS = {
    'B0': dict(pred=None, beam=1),
    'B0b': dict(pred=None),
    'B1': dict(pred='text'),
    'B2': dict(pred='text', speculate=True),
    'B1a': dict(pred='audio'),
    'B3': dict(pred='audio', speculate=True),
    'B1f': dict(pred='audio', fresh=True),
    'H1a': dict(pred='band'),    # AR predictor trained with banded self-attention; cached state of the last `band` tokens only
    'W1a': dict(pred='window'),  # full-history AR architecture restricted to a strict token window
    'S1a': dict(pred='short'),  # bounded-context one-step predictor (optionally rollout-trained) as the prior
    'B4': dict(pred='fgn'),   # FGN ensemble prior: one forward per boundary, lookups on the critical path
}


def load_pred(path, hist, release_cache=False):
    ck = torch.load(path, map_location='cpu')
    c = ck['cfg']
    dec = Decoder(c['vocab'], c['d'], c['h'], c['ff'], c['dec_layers'], 0.0, cross=ck['kind'] == 'audio')
    dec.load_state_dict(ck['model'])
    return Predictor(dec, hist, release_cache=release_cache)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='runs/stage1/avg.pt')
    ap.add_argument('--text', default='runs/stage2_text/best.pt')
    ap.add_argument('--audio', default='runs/stage2_audio/best.pt')
    ap.add_argument('--split', default='dev-clean')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--chunks', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--methods', nargs='+', default=['B0', 'B0b', 'B1', 'B2', 'B1a', 'B3'])
    ap.add_argument('--beam', type=int, default=4)
    ap.add_argument('--w-text', type=float, default=0.3)
    ap.add_argument('--w-audio', type=float, default=0.5)
    ap.add_argument('--bonus', type=float, default=0.0)
    ap.add_argument('--fgn', nargs='+', default=['runs/fgn_s1/best.pt'], help='one or more FGN checkpoints (model ensemble)')
    ap.add_argument('--w-fgn', type=float, default=0.5)
    ap.add_argument('--band-ckpt', default='runs/stage2_audio_b2/best.pt', help='checkpoint for method H1a')
    ap.add_argument('--window-ckpt', default='runs/stage2_audio_w3/best.pt', help='checkpoint for method W1a')
    ap.add_argument('--short', default=None, help='checkpoint for method S1a')
    ap.add_argument('--short-k', type=int, default=1, help='members averaged per step for a noise model')
    ap.add_argument('--w-short', type=float, default=0.5)
    ap.add_argument('--fgn-k', type=int, default=32, help='ensemble members per boundary')
    ap.add_argument('--spec-k', type=int, default=2)
    ap.add_argument('--spec-h', type=int, default=2)
    ap.add_argument('--budget', type=int, default=32)
    ap.add_argument('--hold', type=int, default=1, help='chunks before a best-path token is force-committed; -1 = beam agreement only')
    ap.add_argument('--margin', type=float, default=10.0)
    ap.add_argument('--hist', type=int, default=0, help='predictor token-history window (0 = unbounded)')
    ap.add_argument('--release-cache', action='store_true', help='H1a only: release text KV/logp outside the live bounded-history frontier')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--early-fgn', nargs='*', default=[], help='G2 for methods without a predictor: FGN checkpoint(s) that name the next token (one forward per boundary)')
    ap.add_argument('--early-k', type=int, default=1, help='members for --early-fgn')
    ap.add_argument('--early', type=float, nargs='*', default=[], help='G2: show the predicted next token before its audio arrives when its probability >= threshold; verified by the next chunk')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if a.release_cache and any(METHODS[m]['pred'] != 'band' for m in a.methods):
        ap.error('--release-cache requires only the band-limited H1a method')
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / a.base, map_location='cpu')
    model = AVS(**ck['cfg']); model.load_state_dict(ck['model']); model.eval()
    preds = {}
    if any(METHODS[m]['pred'] == 'text' for m in a.methods):
        preds['text'] = load_pred(ROOT / a.text, a.hist)
    if any(METHODS[m]['pred'] == 'audio' for m in a.methods):
        preds['audio'] = load_pred(ROOT / a.audio, a.hist)
    if any(METHODS[m]['pred'] == 'fgn' for m in a.methods):
        ms = []
        for path in a.fgn:
            fk = torch.load(ROOT / path, map_location='cpu')
            fm = FGN(**fk['cfg']); fm.load_state_dict(fk['model']); ms.append(fm)
        preds['fgn'] = FGNPrior(ms, a.fgn_k)
    if any(METHODS[m]['pred'] == 'short' for m in a.methods):
        fk = torch.load(ROOT / a.short, map_location='cpu')
        fm = FGN(**fk['cfg']); fm.load_state_dict(fk['model'])
        preds['short'] = ShortPrior(fm, a.short_k)
    if any(METHODS[m]['pred'] == 'band' for m in a.methods):
        preds['band'] = load_pred(ROOT / a.band_ckpt, torch.load(ROOT / a.band_ckpt, map_location='cpu')['band'], a.release_cache)
    if any(METHODS[m]['pred'] == 'window' for m in a.methods):
        wk = torch.load(ROOT / a.window_ckpt, map_location='cpu'); wc = wk['cfg']
        wd = Decoder(wc['vocab'], wc['d'], wc['h'], wc['ff'], wc['dec_layers'], 0.0, cross=True); wd.load_state_dict(wk['model'])
        preds['window'] = WindowPredictor(wd, wk['window'])
    gate = None
    if a.early_fgn:
        ms = []
        for path in a.early_fgn:
            fk = torch.load(ROOT / path, map_location='cpu')
            fm = FGN(**fk['cfg']); fm.load_state_dict(fk['model']); ms.append(fm)
        gate = FGNPrior(ms, a.early_k)
    ds = FeatSet(a.split, tok)
    ref_align = torch.load((ROOT / a.base).parent / 'align' / f'{a.split}.pt')['full']
    pool = [i for i in range(len(ds)) if ref_align[i] is not None]
    idxs = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    results, per_utt = [], {}
    for chunk in a.chunks:
        enc_s, enc_bytes = bench_encoder(model, chunk)
        feats = {}
        for batch in ds.batches(40000, shuffle=False, subset=idxs):
            mel, lens, _, _ = ds.collate(batch, 'cpu')
            enc, el = model.encoder(mel, lens, chunk, LEFT)
            logp = model.ctc(enc).log_softmax(-1)
            for b, i in enumerate(batch):
                feats[i] = (enc[b, :el[b]].clone(), logp[b, :el[b]].clone())
        for name in a.methods:
            cfg = dict(METHODS[name])
            kind = cfg.pop('pred')
            w = {'text': a.w_text, 'audio': a.w_audio, 'fgn': a.w_fgn, 'short': a.w_short, 'window': a.w_audio, 'band': a.w_audio, None: 0.0}[kind]
            if kind == 'fgn':
                preds['fgn'].lookups = preds['fgn'].hits = preds['fgn'].net_calls = 0
            dec = Decoder3(preds.get(kind), beam=cfg.pop('beam', a.beam), weight=w, bonus=a.bonus if kind else 0.0,
                           spec_k=a.spec_k, spec_h=a.spec_h, node_budget=a.budget,
                           hold=None if a.hold < 0 else a.hold, margin=a.margin, **cfg)
            errs = nref = 0
            lat, tot = [], dict(audio=0.0, enc=0.0, chunks=0, late=0, late_s=0.0, toks=0)
            agg = {}
            hyps = {}
            early = {th: dict(ok=0, rev=0, lat=[]) for th in a.early} if kind in ('audio', 'text', 'short', 'window', 'band') or (kind is None and gate) else {}
            for i in idxs:
                enc, logp = feats[i]
                T = enc.shape[0]
                dec.reset()
                free, ids, times = 0.0, [], []
                pend, etimes = {th: None for th in early}, {th: [] for th in early}
                for s in range(0, T, chunk):
                    e = min(s + chunk, T)
                    t_arr = arrive_time(e)
                    start = max(t_arr, free)
                    if free > t_arr:
                        tot['late'] += 1; tot['late_s'] += free - t_arr
                    t0 = time.perf_counter()
                    new = dec.advance(logp[s:e], enc[:e])
                    dt = time.perf_counter() - t0
                    t_commit = start + enc_s + dt
                    if e == T:
                        new = new + dec.finish()
                        free = t_commit
                    else:
                        free = t_commit + dec.speculate_now(enc[:e])
                    ids += new; times += [t_commit] * len(new)
                    if early:   # the committed node's next-token scores were refreshed with the audio up to e inside advance()
                        guess, node, pr, tg = None, dec.committed, None, 0.0
                        if e < T and kind is None:
                            tg = time.perf_counter()
                            gate.condition(node.prefix()); gate.set_memory(enc[:e])
                            pr = gate.probs[:, 0].mean(0).clone(); pr[0] = 0
                            tg = time.perf_counter() - tg
                            free += tg; dec.stats['spec_s'] = dec.stats.get('spec_s', 0) + tg   # the guess is ready tg after the commit
                        elif e < T and getattr(node, 'logp', None) is not None and node.cut == dec.cut:
                            pr = node.logp.exp(); pr[0] = 0
                        if pr is not None:
                            g = int(pr.argmax())
                            if g >= 2:
                                guess = (g, float(pr[g]))
                        for th, st in early.items():
                            tms = [t_commit] * len(new)
                            if pend[th] and new:
                                if new[0] == pend[th][0]:
                                    tms[0] = pend[th][1]; st['ok'] += 1
                                else:
                                    st['rev'] += 1
                                pend[th] = None
                            etimes[th] += tms
                            if guess and guess[1] >= th:
                                if pend[th] is None:
                                    pend[th] = (guess[0], t_commit + tg)
                                elif pend[th][0] != guess[0]:
                                    st['rev'] += 1; pend[th] = (guess[0], t_commit + tg)
                            elif pend[th]:
                                st['rev'] += 1; pend[th] = None
                    tot['chunks'] += 1
                tot['audio'] += arrive_time(T); tot['enc'] += enc_s * ((T + chunk - 1) // chunk); tot['toks'] += len(ids)
                for k, v in dec.stats.items():
                    agg[k] = max(agg.get(k, 0), v) if k == 'peak_nodes' else agg.get(k, 0) + v
                hyp = words_with_time(tok, ids, times)
                ref = words_with_time(tok, ds.tokens[i], [frame_time(int(f)) for f in ref_align[i]])
                e_, pairs = match_words([w_ for w_, _ in ref], [w_ for w_, _ in hyp])
                errs += e_; nref += len(ref)
                ul = [hyp[j][1] - ref[r][1] for r, j in pairs]
                for th, st in early.items():
                    st['rev'] += pend[th] is not None
                    hy = words_with_time(tok, ids, etimes[th])
                    st['lat'] += [hy[j][1] - ref[r][1] for r, j in pairs]
                lat += ul
                hyps[i] = ids
                per_utt.setdefault(f'{name}/c{chunk}', []).append({'utt': ds.ids[i], 'err': e_, 'nref': len(ref), 'lat': [round(x, 4) for x in ul]})
            p = preds.get(kind)
            spec_steps, on = agg.get('spec_steps', 0), agg.get('onpath_steps', 0)
            rec = {'method': name, 'chunk': chunk, 'chunk_ms': chunk * 40, 'wer': round(100 * errs / nref, 2),
                   'lat_p50_ms': round(1000 * pct(lat, 0.5), 1), 'lat_p95_ms': round(1000 * pct(lat, 0.95), 1),
                   'lat_mean_ms': round(1000 * sum(lat) / max(1, len(lat)), 1),
                   'enc_ms_per_chunk': round(1000 * enc_s, 2),
                   'onpath_pred_ms_per_chunk': round(1000 * (agg.get('onpath_s', 0) + agg.get('mem_s', 0)) / tot['chunks'], 3),
                   'spec_ms_per_chunk': round(1000 * agg.get('spec_s', 0) / tot['chunks'], 3),
                   'onpath_steps_per_tok': round(on / max(1, tot['toks']), 3),
                   'spec_steps_per_tok': round(spec_steps / max(1, tot['toks']), 3),
                   'refresh_steps_per_tok': round(agg.get('refresh_steps', 0) / max(1, tot['toks']), 3),
                   'reuse_rate': round(agg.get('spec_used', 0) / max(1, agg.get('spec_used', 0) + on), 4) if kind else None,
                   'spec_waste': round(1 - agg.get('spec_used', 0) / spec_steps, 4) if spec_steps else None,
                   'peak_nodes': agg.get('peak_nodes'),
                   'peak_pred_state_kb': round(agg.get('peak_nodes', 0) * p.bytes_per_node / 1024, 1) if p else 0,
                   'enc_state_kb': round(enc_bytes / 1024, 1),
                   'revised_per_100tok': round(100 * agg.get('revised', 0) / max(1, tot['toks']), 2),
                   'rtf': round((tot['enc'] + agg.get('onpath_s', 0) + agg.get('mem_s', 0) + agg.get('spec_s', 0)) / tot['audio'], 4),
                   'late_chunk_frac': round(tot['late'] / tot['chunks'], 4), 'matched_words': len(lat), 'utts': len(idxs)}
            if kind == 'fgn':
                rec['fgn_hit_rate'] = round(p.hits / max(1, p.lookups), 4)
                rec['fgn_forwards_per_chunk'] = round(p.net_calls / tot['chunks'], 3)
            if early:
                rec['early'] = [{'th': th, 'lat_p50_ms': round(1000 * pct(st['lat'], 0.5), 1), 'lat_mean_ms': round(1000 * sum(st['lat']) / max(1, len(st['lat'])), 1),
                                 'lat_p95_ms': round(1000 * pct(st['lat'], 0.95), 1), 'early_tok_per_100': round(100 * st['ok'] / max(1, tot['toks']), 2),
                                 'retracted_per_100': round(100 * st['rev'] / max(1, tot['toks']), 2),
                                 'precision': round(100 * st['ok'] / max(1, st['ok'] + st['rev']), 1)} for th, st in early.items()]
            results.append(rec)
            print(json.dumps(rec), flush=True)
            per_utt[f'{name}/c{chunk}/hyps'] = hyps
        for x, y in (('B1', 'B2'), ('B1a', 'B3')):
            if x in a.methods and y in a.methods:
                hx, hy = per_utt[f'{x}/c{chunk}/hyps'], per_utt[f'{y}/c{chunk}/hyps']
                print(json.dumps({'check': f'{x}=={y}', 'chunk': chunk, 'different_utts': sum(hx[i] != hy[i] for i in idxs)}), flush=True)
    if a.out:
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({'args': vars(a), 'results': results, 'per_utt': {k: v for k, v in per_utt.items() if not k.endswith('/hyps')}}, out.open('w'))


if __name__ == '__main__':
    main()
