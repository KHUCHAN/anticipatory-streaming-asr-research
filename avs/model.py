"""AVS architecture.

Encoder : valid-conv x4 subsampling -> chunk-causal Conformer blocks
          (RoPE self-attention under a chunk mask + strictly causal depthwise conv).
          Encoder frame j depends on mel frames <= 4*j+6 of its own chunk end, so the
          masked full-utterance forward equals true chunk-by-chunk streaming.
Heads   : CTC (acoustic verifier / alignment) and a Transformer decoder.
Decoder : also used as the stage-2 predictor. Its cross-attention takes a per-token
          memory mask, so each token can be restricted to audio that had arrived
          before a cutoff. A learned always-visible slot `mem0` stands for "no audio".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import SOS


def sub_len(t):
    """Encoder frames produced from t mel frames (two valid k3/s2 convolutions)."""
    return ((t - 3) // 2 + 1 - 3) // 2 + 1


def mel_needed(j):
    """Mel frames that must have arrived before encoder frame j exists (count)."""
    return 4 * j + 7


def chunk_mask(t, chunk, left, device):
    """[t,t] bool, True = may attend. chunk<=0 -> full context (offline mode)."""
    if chunk <= 0:
        return torch.ones(t, t, dtype=torch.bool, device=device)
    i = torch.arange(t, device=device)
    start = (i // chunk) * chunk
    end = start + chunk
    j = i[None, :]
    lo = (start - left)[:, None] if left >= 0 else torch.zeros_like(start)[:, None]
    return (j < end[:, None]) & (j >= lo)


def rope(x, pos):
    """x [B,H,T,D], pos [T] float -> rotary-encoded x."""
    half = x.shape[-1] // 2
    freqs = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float()[:, None] * freqs
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class MHA(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.drop = h, drop
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, mem, mask, q_pos=None, k_pos=None):
        b, tq, d = x.shape
        split = lambda y: y.view(b, -1, self.h, d // self.h).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(mem)), split(self.v(mem))
        if q_pos is not None:
            q, k = rope(q, q_pos), rope(k, k_pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(b, tq, d))


class EncLayer(nn.Module):
    def __init__(self, d, h, ff, kernel, drop):
        super().__init__()
        self.kernel = kernel
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))
        self.att = MHA(d, h, drop)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.pw2 = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def conv(self, x, hist=None):
        """Strictly causal depthwise conv. hist: [B,kernel-1,d] left context (streaming)."""
        y = F.glu(self.pw1(x), -1)
        new_hist = None
        if hist is None:
            z = F.pad(y.transpose(1, 2), (self.kernel - 1, 0))
        else:
            cat = torch.cat([hist, y], 1)
            new_hist = cat[:, -(self.kernel - 1):]
            z = cat.transpose(1, 2)
        if z.device.type == 'cpu':  # grouped Conv1d is slow on CPU; same arithmetic via unfold
            z = (z.unfold(2, self.kernel, 1) * self.dw.weight[:, 0][None, :, None, :]).sum(-1) + self.dw.bias[None, :, None]
        else:
            z = self.dw(z)
        z = z.transpose(1, 2)
        return self.pw2(F.silu(self.n3(z))), new_hist

    def forward(self, x, mask, pos):
        y = self.n1(x)
        x = x + self.drop(self.att(y, y, mask, pos, pos))
        x = x + self.drop(self.conv(self.n2(x))[0])
        return x + self.drop(self.ff(self.n4(x)))

    def stream(self, x, c, pos, left):
        """True incremental step for one chunk x [1,n,d]; c holds k/v of past frames and conv history."""
        b, n, d = x.shape
        att, h = self.att, self.att.h
        y = self.n1(x)
        split = lambda z: z.view(b, n, h, d // h).transpose(1, 2)
        q, k, v = rope(split(att.q(y)), pos), rope(split(att.k(y)), pos), split(att.v(y))
        K = k if c['k'] is None else torch.cat([c['k'], k], 2)
        V = v if c['v'] is None else torch.cat([c['v'], v], 2)
        o = F.scaled_dot_product_attention(q, K, V)
        c['k'], c['v'] = (K, V) if left < 0 else (K[:, :, -left:], V[:, :, -left:]) if left else (None, None)
        x = x + att.o(o.transpose(1, 2).reshape(b, n, d))
        hist = c['hist'] if c['hist'] is not None else x.new_zeros(b, self.kernel - 1, d)
        z, c['hist'] = self.conv(self.n2(x), hist)
        x = x + z
        return x + self.ff(self.n4(x))


class Encoder(nn.Module):
    def __init__(self, d=256, h=4, ff=1024, layers=12, kernel=15, drop=0.1, n_mel=80, c=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, c, 3, 2), nn.ReLU(), nn.Conv2d(c, c, 3, 2), nn.ReLU())
        self.proj = nn.Linear(c * sub_len(n_mel), d)
        self.drop = nn.Dropout(drop)
        self.layers = nn.ModuleList(EncLayer(d, h, ff, kernel, drop) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def subsample(self, mel):
        x = self.conv(mel.unsqueeze(1))
        b, c, t, f = x.shape
        return self.proj(x.transpose(1, 2).reshape(b, t, c * f))

    def forward(self, mel, mel_lens, chunk=0, left=-1):
        x = self.drop(self.subsample(mel))
        lens = sub_len(mel_lens)
        t = x.shape[1]
        valid = torch.arange(t, device=x.device)[None, :] < lens[:, None]
        mask = chunk_mask(t, chunk, left, x.device)[None] & valid[:, None, :]
        mask = (mask | torch.eye(t, dtype=torch.bool, device=x.device)[None]).unsqueeze(1)
        pos = torch.arange(t, device=x.device)
        for layer in self.layers:
            x = layer(x, mask, pos)
        return self.norm(x), lens

    def stream_init(self):
        return {'pos': 0, 'layers': [{'k': None, 'v': None, 'hist': None} for _ in self.layers]}

    @torch.no_grad()
    def stream_chunk(self, mel, cache, left=64):
        """mel [1, 4n+3, 80]: the mel frames 4s .. 4(s+n-1)+6 for encoder frames s..s+n-1."""
        x = self.subsample(mel)
        pos = cache['pos'] + torch.arange(x.shape[1], device=x.device)
        for layer, c in zip(self.layers, cache['layers']):
            x = layer.stream(x, c, pos, left)
        cache['pos'] += x.shape[1]
        return self.norm(x)

    @staticmethod
    def stream_state_bytes(cache):
        n = 0
        for c in cache['layers']:
            n += sum(t.numel() * t.element_size() for t in (c['k'], c['v'], c['hist']) if t is not None)
        return n


class DecLayer(nn.Module):
    def __init__(self, d, h, ff, drop, cross=True):
        super().__init__()
        self.n1, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_att = MHA(d, h, drop)
        self.cross = cross
        if cross:
            self.n2 = nn.LayerNorm(d)
            self.cross_att = MHA(d, h, drop)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, self_mask, pos, mem, mem_mask):
        y = self.n1(x)
        x = x + self.drop(self.self_att(y, y, self_mask, pos, pos))
        if self.cross:
            x = x + self.drop(self.cross_att(self.n2(x), mem, mem_mask))
        return x + self.drop(self.ff(self.n3(x)))


class Decoder(nn.Module):
    """Causal token decoder. cross=False gives a text-only predictor (B2)."""

    def __init__(self, vocab, d=256, h=4, ff=1024, layers=4, drop=0.1, cross=True):
        super().__init__()
        self.cross = cross
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(DecLayer(d, h, ff, drop, cross) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)
        self.drop = nn.Dropout(drop)
        if cross:
            self.mem0 = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, ys_in, mem=None, mem_mask=None):
        """ys_in [B,U]; mem [B,T,d]; mem_mask bool [B,1|U,T] (True = visible)."""
        b, u = ys_in.shape
        x = self.drop(self.emb(ys_in))
        pos = torch.arange(u, device=x.device)
        self_mask = torch.ones(u, u, dtype=torch.bool, device=x.device).tril()
        if getattr(self, 'band', 0):   # every layer attends to the current token and at most `band` earlier ones
            self_mask = self_mask & ~torch.ones(u, u, dtype=torch.bool, device=x.device).tril(-self.band - 1)
        self_mask = self_mask[None, None]
        if self.cross:
            mem = torch.cat([self.mem0.expand(b, 1, -1).to(mem.dtype), mem], 1)
            always = torch.ones(b, mem_mask.shape[1], 1, dtype=torch.bool, device=x.device)
            mem_mask = torch.cat([always, mem_mask], 2).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, self_mask, pos, mem, mem_mask)
        return self.out(self.norm(x))


def sos_eos(ys, ylens):
    """-> decoder input (sos + y) and target (y + eos, pad=-100)."""
    b, u = ys.shape
    pad = torch.arange(u + 1, device=ys.device)[None, :] > ylens[:, None]
    ys_in = torch.cat([torch.full((b, 1), SOS, device=ys.device), ys], 1)
    tgt = torch.cat([ys, torch.zeros(b, 1, dtype=torch.long, device=ys.device)], 1)
    tgt.scatter_(1, ylens[:, None], SOS)
    return ys_in, tgt.masked_fill(pad, -100)


class AVS(nn.Module):
    def __init__(self, vocab, d=256, h=4, ff=1024, enc_layers=12, dec_layers=4, drop=0.1):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, h=h, ff=ff, enc_layers=enc_layers, dec_layers=dec_layers, drop=drop)
        self.encoder = Encoder(d, h, ff, enc_layers, drop=drop)
        self.ctc = nn.Linear(d, vocab)
        self.decoder = Decoder(vocab, d, h, ff, dec_layers, drop)

    def forward(self, mel, mel_lens, ys, ylens, chunk=0, left=-1, ctc_weight=0.3, smoothing=0.1):
        enc, lens = self.encoder(mel, mel_lens, chunk, left)
        b = mel.shape[0]
        logp = self.ctc(enc).float().log_softmax(-1)
        ctc = F.ctc_loss(logp.transpose(0, 1), ys, lens, ylens, blank=0, reduction='sum', zero_infinity=True) / b
        ys_in, tgt = sos_eos(ys, ylens)
        valid = (torch.arange(enc.shape[1], device=enc.device)[None, :] < lens[:, None])[:, None, :]
        logits = self.decoder(ys_in, enc, valid).float()
        att = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction='sum', label_smoothing=smoothing) / b
        acc = ((logits.argmax(-1) == tgt).sum() / (tgt != -100).sum()).detach()
        return ctc_weight * ctc + (1 - ctc_weight) * att, {'ctc': ctc.detach(), 'att': att.detach(), 'acc': acc}

    @torch.no_grad()
    def ctc_greedy(self, mel, mel_lens, chunk=0, left=-1):
        enc, lens = self.encoder(mel, mel_lens, chunk, left)
        best = self.ctc(enc).argmax(-1)
        hyps = []
        for b in range(best.shape[0]):
            seq = torch.unique_consecutive(best[b, :lens[b]])
            hyps.append([int(i) for i in seq if i != 0])
        return hyps
