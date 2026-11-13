"""Batched CTC Viterbi forced alignment (CPU). Returns the first frame of every token."""
import torch

NEG = -1e30


def ctc_align(logp, lens, ys, ylens):
    """logp [B,T,V] log-probs, lens [B], ys [B,U] (pad anything), ylens [B] -> list of LongTensor[U_b]."""
    logp, lens, ys, ylens = logp.float().cpu(), lens.cpu(), ys.cpu(), ylens.cpu()
    b, t, _ = logp.shape
    u = ys.shape[1]
    s = 2 * u + 1
    ext = torch.zeros(b, s, dtype=torch.long)
    ext[:, 1::2] = ys
    em = logp.gather(2, ext[:, None, :].expand(b, t, s))
    skip = torch.zeros(b, s, dtype=torch.bool)
    skip[:, 2:] = (ext[:, 2:] != 0) & (ext[:, 2:] != ext[:, :-2])
    alpha = torch.full((b, s), NEG)
    alpha[:, 0] = em[:, 0, 0]
    alpha[:, 1] = em[:, 0, 1]
    bp = torch.zeros(t, b, s, dtype=torch.int8)
    for i in range(1, t):
        p1 = torch.cat([torch.full((b, 1), NEG), alpha[:, :-1]], 1)
        p2 = torch.cat([torch.full((b, 2), NEG), alpha[:, :-2]], 1).masked_fill(~skip, NEG)
        best, arg = torch.stack([alpha, p1, p2], 0).max(0)
        live = (i < lens)[:, None]
        alpha = torch.where(live, best + em[:, i], alpha)
        bp[i] = torch.where(live, arg, torch.zeros_like(arg)).to(torch.int8)
    out = []
    for k in range(b):
        n, tk = int(ylens[k]), int(lens[k])
        if n == 0:
            out.append(torch.zeros(0, dtype=torch.long)); continue
        last = 2 * n
        state = last if alpha[k, last] >= alpha[k, last - 1] else last - 1
        first = torch.full((n,), tk - 1, dtype=torch.long)
        for i in range(tk - 1, -1, -1):
            if state % 2 == 1:
                first[state // 2] = i
            state -= int(bp[i, k, state])
        out.append(first)
    return out
