#!/usr/bin/env python3
"""Run the B1a streaming decoder and log, at every chunk boundary, what a forecast-verify front end could show.

Per boundary: commit time, newly committed tokens, a greedy 3-token forecast chain from the committed prefix
(audio up to the boundary only) and features of the first forecast for a learned confidence gate.
All decisions (thresholds, gates, two-pass display) are evaluated offline by scripts/analyze_final.py.
"""
import argparse, random, statistics, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS, Decoder  # noqa: E402
from avs.fgn import FGN, FGNPrior  # noqa: E402
from avs.stream import Decoder3, Node, Predictor  # noqa: E402
from avs.text import Tokenizer  # noqa: E402

LEFT, DEPTH = 64, 3
frame_time = lambda f: (160 * (4 * f + 3) + 200) / 16000
arrive_time = lambda e: (160 * (4 * (e - 1) + 6) + 400) / 16000


def bench_encoder(model, chunk, reps=60):
    enc = model.encoder
    cache = enc.stream_init()
    mel = torch.randn(1, 4 * (chunk - 1) + 7, 80)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); enc.stream_chunk(mel, cache, LEFT); ts.append(time.perf_counter() - t)
    return statistics.median(ts[10:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='runs/stage1/avg.pt')
    ap.add_argument('--audio', default='runs/stage2_audio/best.pt')
    ap.add_argument('--nar', default='runs/fgn_none/best.pt')
    ap.add_argument('--split', default='dev-clean')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--exclude-seed', type=int, default=None, help='sample from the utterances NOT in the --n sample of this seed')
    ap.add_argument('--exclude-n', type=int, default=500)
    ap.add_argument('--chunks', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / a.base, map_location='cpu')
    model = AVS(**ck['cfg']); model.load_state_dict(ck['model']); model.eval()
    pk = torch.load(ROOT / a.audio, map_location='cpu'); c = pk['cfg']
    d = Decoder(c['vocab'], c['d'], c['h'], c['ff'], c['dec_layers'], 0.0, cross=True); d.load_state_dict(pk['model'])
    pred = Predictor(d, 0)
    fk = torch.load(ROOT / a.nar, map_location='cpu')
    fm = FGN(**fk['cfg']); fm.load_state_dict(fk['model'])
    nar = FGNPrior([fm], 1)
    ds = FeatSet(a.split, tok)
    ref_align = torch.load((ROOT / a.base).parent / 'align' / f'{a.split}.pt')['full']
    pool = [i for i in range(len(ds)) if ref_align[i] is not None]
    if a.exclude_seed is not None:
        held = set(random.Random(a.exclude_seed).sample(pool, min(a.exclude_n, len(pool))))
        pool = [i for i in pool if i not in held]
    idxs = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    cont = [not tok.piece(i).startswith('▁') for i in range(c['vocab'])]
    out = {'args': vars(a), 'chunks': {}}
    for chunk in a.chunks:
        enc_s = bench_encoder(model, chunk)
        feats = {}
        for batch in ds.batches(40000, shuffle=False, subset=idxs):
            mel, lens, _, _ = ds.collate(batch, 'cpu')
            enc, el = model.encoder(mel, lens, chunk, LEFT)
            logp = model.ctc(enc).log_softmax(-1)
            for b, i in enumerate(batch):
                feats[i] = (enc[b, :el[b]].clone(), logp[b, :el[b]].clone())
        dec = Decoder3(pred, beam=4, weight=0.5, bonus=0.0, hold=0, margin=10.0)
        utts, t_all = [], time.time()
        for i in idxs:
            enc, logp = feats[i]
            T = enc.shape[0]
            dec.reset()
            free, ids, times, bounds = 0.0, [], [], []
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                start = max(arrive_time(e), free)
                t0 = time.perf_counter()
                new = dec.advance(logp[s:e], enc[:e])
                t_commit = start + enc_s + (time.perf_counter() - t0)
                free = t_commit
                if e == T:
                    new = new + dec.finish()
                ids += new; times += [t_commit] * len(new)
                b = {'t': t_commit, 'new': new, 'g': [], 'f': None, 'tg': 0.0, 'tn': 0.0}
                node = dec.committed
                if e < T and node.logp is not None and node.cut == dec.cut:
                    t1 = time.perf_counter()
                    chain, cur, first = [], node, None
                    for k in range(DEPTH):
                        pr = cur.logp.exp(); pr[0] = 0
                        if k == 0:
                            first = pr
                        g = int(pr.argmax())
                        if g < 2:
                            break
                        chain.append((g, float(pr[g])))
                        if k + 1 < DEPTH:
                            nxt = Node(g, cur); pred.step([nxt], cut=dec.cut, spec=True); cur = nxt
                    b['tg'] = time.perf_counter() - t1
                    if chain:
                        g = chain[0][0]
                        t1 = time.perf_counter()
                        nar.condition(node.prefix()); nar.set_memory(enc[:e])
                        npr = nar.probs[0, 0]
                        b['tn'] = time.perf_counter() - t1
                        top2 = first.topk(2).values
                        ent = float(-(first.clamp_min(1e-9) * first.clamp_min(1e-9).log()).sum())
                        ctc = logp[max(0, e - 2):e].exp()
                        b['f'] = [float(top2[0]), float(top2[1]), ent, float(ctc[-1, 0]), float(ctc[-1, g]), float(ctc[:, g].max()),
                                  float(npr[g]), float(int(npr[2:].argmax()) + 2 == g), float(cont[g]), float(len(new)), float(chunk)]
                    b['g'] = chain
                bounds.append(b)
            ref_t = [frame_time(int(f)) for f in ref_align[i]]
            utts.append({'utt': ds.ids[i], 'ref_ids': ds.tokens[i], 'ref_t': ref_t, 'ids': ids, 'times': times, 'bounds': bounds,
                         'audio_s': arrive_time(T), 'chunks': len(bounds)})
        out['chunks'][chunk] = {'enc_s': enc_s, 'utts': utts}
        print(f'{a.split} chunk {chunk}: {len(utts)} utts in {time.time() - t_all:.0f}s', flush=True)
    path = ROOT / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)


if __name__ == '__main__':
    main()
