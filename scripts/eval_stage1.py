#!/usr/bin/env python3
"""Stage-1 report: CTC-greedy WER of a checkpoint on full splits, per encoder mode."""
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.data import FeatSet, FEATS  # noqa: E402
from avs.model import AVS  # noqa: E402
from avs.text import Tokenizer, wer  # noqa: E402

MODES = {'full': (0, -1), 'c32': (32, 64), 'c16': (16, 64), 'c8': (8, 64), 'c4': (4, 64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/stage1/avg.pt')
    ap.add_argument('--splits', nargs='+', default=['dev-clean', 'dev-other', 'test-clean', 'test-other'])
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    ck = torch.load(ROOT / a.ckpt, map_location='cpu')
    model = AVS(**ck['cfg']); model.load_state_dict(ck['model']); model.to(a.device).eval()
    tok = Tokenizer(FEATS / 'spm.model')
    table = {}
    for split in a.splits:
        ds = FeatSet(split, tok)
        for mode, (chunk, left) in MODES.items():
            refs, hyps = [], []
            for idxs in ds.batches(40000, shuffle=False):
                mel, lens, _, _ = ds.collate(idxs, a.device)
                with torch.autocast(a.device, dtype=torch.float16, enabled=a.device == 'mps'):
                    out = model.ctc_greedy(mel, lens, chunk, left)
                hyps += [tok.decode(h) for h in out]; refs += [ds.texts[i] for i in idxs]
            table.setdefault(split, {})[mode] = round(100 * wer(refs, hyps), 2)
        print(split, table[split], flush=True)
    out = (ROOT / a.ckpt).parent / 'wer_ctc_greedy.json'
    json.dump({'ckpt': a.ckpt, 'averaged': ck.get('averaged'), 'decoding': 'CTC greedy, no LM', 'wer': table}, out.open('w'), indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
