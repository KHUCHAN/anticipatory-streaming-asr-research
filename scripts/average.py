#!/usr/bin/env python3
"""Average the last N epoch checkpoints of a run -> avg.pt."""
import sys
from pathlib import Path
import torch

run, n = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(run.glob('ep[0-9][0-9].pt'))[-n:]
avg, cfg = None, None
for f in files:
    ck = torch.load(f, map_location='cpu')
    cfg = ck['cfg']
    if avg is None:
        avg = {k: v.clone().double() if v.is_floating_point() else v for k, v in ck['model'].items()}
    else:
        for k, v in ck['model'].items():
            if v.is_floating_point():
                avg[k] += v.double()
avg = {k: (v / len(files)).float() if v.is_floating_point() else v for k, v in avg.items()}
torch.save({'model': avg, 'cfg': cfg, 'averaged': [f.name for f in files]}, run / 'avg.pt')
print('averaged', [f.name for f in files], '->', run / 'avg.pt')
