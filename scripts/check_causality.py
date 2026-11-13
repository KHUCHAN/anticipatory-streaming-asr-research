#!/usr/bin/env python3
"""Causality test: future mel frames must not change already-emitted encoder frames."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.model import AVS, mel_needed, sub_len  # noqa: E402

torch.manual_seed(0)
ckpt = sys.argv[1] if len(sys.argv) > 1 else None
if ckpt:
    ck = torch.load(ckpt, map_location='cpu'); model = AVS(**ck['cfg']); model.load_state_dict(ck['model'])
else:
    model = AVS(500)
model.eval()
mel = torch.randn(2, 1000, 80)
lens = torch.tensor([1000, 700])
ok = True
for chunk, left in [(4, 64), (8, 64), (16, 64), (32, 64), (16, -1)]:
    base, el = model.encoder(mel, lens, chunk, left)
    for J in [chunk * 2, chunk * 5, (sub_len(700) // chunk) * chunk]:
        m = mel_needed(J - 1)
        pert = mel.clone(); pert[:, m:] += torch.randn_like(pert[:, m:])
        out, _ = model.encoder(pert, lens, chunk, left)
        past = (out[:, :J] - base[:, :J]).abs().max().item()
        future = (out[0, J:] - base[0, J:]).abs().max().item()
        good = past < 1e-5 and future > 1e-3
        ok &= good
        print(f'chunk={chunk:2d} left={left:3d} boundary={J:3d} mel_cut={m:4d} past_diff={past:.2e} future_diff={future:.2e} {"OK" if good else "LEAK"}')
base, _ = model.encoder(mel, lens, 0, -1)
pert = mel.clone(); pert[:, 500:] += 1
print('offline mode (expected non-causal) past_diff=%.2e' % (model.encoder(pert, lens, 0, -1)[0][:, :100] - base[:, :100]).abs().max().item())
print("CAUSALITY", "PASS" if ok else "FAIL")
if not ok: sys.exit(1)

# Incremental streaming must reproduce the masked full-utterance forward.
with torch.no_grad():
    for chunk in (4, 16):
        full, el = model.encoder(mel[:1], lens[:1], chunk, 64)
        cache, outs, s, T = model.encoder.stream_init(), [], 0, int(el[0])
        while s < T:
            e = min(s + chunk, T)
            outs.append(model.encoder.stream_chunk(mel[:1, 4 * s:4 * (e - 1) + 7], cache, 64)); s = e
        diff = (torch.cat(outs, 1) - full[:, :T]).abs().max().item()
        print(f'incremental vs masked forward chunk={chunk}: max diff {diff:.2e}, state {model.encoder.stream_state_bytes(cache) / 1e6:.2f} MB', 'OK' if diff < 1e-3 else 'MISMATCH')
        if diff >= 1e-3:
            sys.exit(1)
