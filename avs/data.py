"""Feature-cache dataset, length-bucketed batching, SpecAugment."""
import json, random
from pathlib import Path
import numpy as np
import torch

FEATS = Path(__file__).resolve().parents[1] / 'data/processed/feats'


class FeatSet:
    def __init__(self, split, tok, root=FEATS):
        meta = json.load((root / f'{split}.json').open())
        self.ids, self.texts, self.offsets, self.lens = meta['ids'], meta['texts'], meta['offsets'], meta['lens']
        self.mm = np.memmap(root / f'{split}.f16', dtype=np.float16, mode='r').reshape(-1, 80)
        st = json.load((root / 'stats.json').open())
        self.mean, self.std = np.asarray(st['mean'], np.float32), np.asarray(st['std'], np.float32)
        self.tokens = [tok.encode(t) for t in self.texts]

    def __len__(self):
        return len(self.ids)

    def feat(self, i):
        x = np.asarray(self.mm[self.offsets[i]:self.offsets[i] + self.lens[i]], dtype=np.float32)
        return (x - self.mean) / self.std

    def batches(self, max_frames, max_utts=128, shuffle=True, subset=None, seed=0):
        idx = sorted(subset if subset is not None else range(len(self)), key=lambda i: self.lens[i])
        out, cur = [], []
        for i in idx:
            if cur and ((len(cur) + 1) * self.lens[i] > max_frames or len(cur) >= max_utts):
                out.append(cur); cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        if shuffle:
            random.Random(seed).shuffle(out)
        return out

    def collate(self, idxs, device):
        feats = [self.feat(i) for i in idxs]
        lens = torch.tensor([f.shape[0] for f in feats])
        mel = torch.zeros(len(idxs), int(lens.max()), 80)
        for b, f in enumerate(feats):
            mel[b, :f.shape[0]] = torch.from_numpy(f)
        ylens = torch.tensor([len(self.tokens[i]) for i in idxs])
        ys = torch.zeros(len(idxs), int(ylens.max()), dtype=torch.long)
        for b, i in enumerate(idxs):
            ys[b, :ylens[b]] = torch.tensor(self.tokens[i])
        return mel.to(device), lens.to(device), ys.to(device), ylens.to(device)


def spec_augment(mel, lens, n_freq=2, f_max=27, n_time=3, t_max=40, t_ratio=0.2):
    """In-place masking with 0 (= train mean after normalisation). Training only."""
    for b in range(mel.shape[0]):
        t = int(lens[b])
        for _ in range(n_freq):
            w = random.randint(0, f_max); f0 = random.randint(0, 80 - w)
            mel[b, :, f0:f0 + w] = 0
        for _ in range(n_time):
            w = min(random.randint(0, t_max), int(t * t_ratio)); t0 = random.randint(0, max(0, t - w))
            mel[b, t0:t0 + w] = 0
    return mel
