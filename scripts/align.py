#!/usr/bin/env python3
"""Forced-align every split with the frozen stage-1 model, in each encoder mode.

Output: <run>/align/<split>.pt = {mode: [LongTensor(first frame per token) per utterance]}
Offline mode ('full') is the timing reference; streaming modes define what the
streaming recogniser itself would have emitted before a chunk boundary.
"""
import argparse, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.align import ctc_align  # noqa: E402
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS  # noqa: E402
from avs.text import Tokenizer  # noqa: E402

MODES = {'full': (0, -1), 'c4': (4, 64), 'c8': (8, 64), 'c16': (16, 64), 'c32': (32, 64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/stage1/avg.pt')
    ap.add_argument('--splits', nargs='+', default=['train-clean-100', 'dev-clean', 'dev-other', 'test-clean', 'test-other'])
    ap.add_argument('--modes', nargs='+', default=list(MODES))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    ck = torch.load(ROOT / a.ckpt, map_location='cpu')
    model = AVS(**ck['cfg']); model.load_state_dict(ck['model']); model.to(a.device).eval()
    tok = Tokenizer(FEATS / 'spm.model')
    out_dir = (ROOT / a.ckpt).parent / 'align'
    out_dir.mkdir(exist_ok=True)
    for split in a.splits:
        ds = FeatSet(split, tok)
        subset = range(a.limit) if a.limit else None
        result = {}
        for mode in a.modes:
            chunk, left = MODES[mode]
            spikes = [None] * len(ds)
            for idxs in ds.batches(40000, shuffle=False, subset=subset):
                mel, lens, ys, ylens = ds.collate(idxs, a.device)
                with torch.no_grad():
                    enc, el = model.encoder(mel, lens, chunk, left)
                    logp = model.ctc(enc).float().log_softmax(-1)
                for i, sp in zip(idxs, ctc_align(logp, el, ys, ylens)):
                    spikes[i] = sp.to(torch.int32)
            result[mode] = spikes
        torch.save(result, out_dir / f'{split}.pt')
        done = [s for s in result[a.modes[0]] if s is not None]
        print(f'ALIGN {split}: {len(done)} utts, modes={a.modes}', flush=True)


if __name__ == '__main__':
    main()
