#!/usr/bin/env python3
"""True incremental streaming: the encoder is advanced chunk by chunk with its caches, on a simulated real-time clock.

Checks that (1) the transcripts equal those of the masked full-utterance forward used by every other experiment,
and (2) the measured time of one whole step (encoder + CTC + beam search with the AR prior + forecast read-out
+ FGN second opinion + gate) stays far below the chunk duration. All times are single-thread CPU wall time.
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS, Decoder  # noqa: E402
from avs.fgn import FGN, FGNPrior  # noqa: E402
from avs.stream import Decoder3, Predictor  # noqa: E402
from avs.text import Tokenizer  # noqa: E402

LEFT = 64
arrive_time = lambda e: (160 * (4 * (e - 1) + 6) + 400) / 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test-clean')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--chunks', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--out', default='runs/final/paced.json')
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    tok = Tokenizer(FEATS / 'spm.model')
    ck = torch.load(ROOT / 'runs/stage1/avg.pt', map_location='cpu')
    model = AVS(**ck['cfg']); model.load_state_dict(ck['model']); model.eval()
    pk = torch.load(ROOT / 'runs/stage2_audio/best.pt', map_location='cpu'); c = pk['cfg']
    d = Decoder(c['vocab'], c['d'], c['h'], c['ff'], c['dec_layers'], 0.0, cross=True); d.load_state_dict(pk['model'])
    pred = Predictor(d, 0)
    fk = torch.load(ROOT / 'runs/fgn_none/best.pt', map_location='cpu')
    fm = FGN(**fk['cfg']); fm.load_state_dict(fk['model'])
    nar = FGNPrior([fm], 1)
    ds = FeatSet(a.split, tok)
    idxs = random.Random(11).sample(range(len(ds)), a.n)
    out = {}
    for chunk in a.chunks:
        dec = Decoder3(pred, beam=4, weight=0.5, bonus=0.0, hold=0, margin=10.0)
        dur = chunk * 0.04
        t_enc, t_dec, t_fc, t_all, late, same, maxdiff, audio = [], [], [], [], 0, 0, 0.0, 0.0
        for i in idxs:
            mel, lens, _, _ = ds.collate([i], 'cpu')
            full, el = model.encoder(mel, lens, chunk, LEFT)
            T = int(el[0])
            ref_logp = model.ctc(full[0, :T]).log_softmax(-1)
            dec.reset(); ref_ids = []
            for s in range(0, T, chunk):                      # reference: masked full-utterance forward
                e = min(s + chunk, T)
                ref_ids += dec.advance(ref_logp[s:e], full[0, :e])
            ref_ids += dec.finish()
            dec.reset()
            cache, encs, ids, free = model.encoder.stream_init(), [], [], 0.0
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                t0 = time.perf_counter()
                y = model.encoder.stream_chunk(mel[:, 4 * s:4 * (e - 1) + 7], cache, LEFT)[0]
                logp = model.ctc(y).log_softmax(-1)
                t1 = time.perf_counter()
                encs.append(y); enc = torch.cat(encs, 0)
                new = dec.advance(logp, enc)
                t2 = time.perf_counter()
                if e < T and dec.committed.logp is not None:   # forecast read-out + second opinion + gate features
                    pr = dec.committed.logp.exp(); g = int(pr[2:].argmax()) + 2
                    nar.condition(dec.committed.prefix()); nar.set_memory(enc)
                    _ = float(pr[g]), float(nar.probs[0, 0][g])
                t3 = time.perf_counter()
                if e == T:
                    new = new + dec.finish()
                ids += new
                start = max(arrive_time(e), free)
                late += free > arrive_time(e)
                free = start + (t3 - t0)
                t_enc.append(t1 - t0); t_dec.append(t2 - t1); t_fc.append(t3 - t2); t_all.append(t3 - t0)
            maxdiff = max(maxdiff, float((torch.cat(encs, 0) - full[0, :T]).abs().max()))
            same += ids == ref_ids
            audio += arrive_time(T)
        ms = lambda xs, q: round(1000 * sorted(xs)[min(len(xs) - 1, int(q * len(xs)))], 2)
        rec = {'chunk_ms': chunk * 40, 'utts': a.n, 'identical_transcripts': same, 'max_encoder_diff': round(maxdiff, 6),
               'step_ms_p50': ms(t_all, 0.5), 'step_ms_p95': ms(t_all, 0.95), 'step_ms_max': ms(t_all, 1.0),
               'encoder_ms_p50': ms(t_enc, 0.5), 'search_ms_p50': ms(t_dec, 0.5), 'forecast_gate_ms_p50': ms(t_fc, 0.5),
               'chunks_started_late': late, 'chunks': len(t_all), 'rtf': round(sum(t_all) / audio, 4),
               'step_over_chunk_p95': round(ms(t_all, 0.95) / (1000 * dur), 4)}
        out[f'c{chunk}'] = rec
        print(json.dumps(rec), flush=True)
    json.dump(out, (ROOT / a.out).open('w'), indent=1)


if __name__ == '__main__':
    main()
