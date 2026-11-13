"""FGN-style anticipatory predictor (after Alet et al. 2025, "Skillful joint probabilistic
weather forecasting from marginals").

What is carried over from FGN:
  * one low-dimensional global noise vector z ~ N(0, I_32) is the only source of randomness;
    it enters every conditional layer-norm through one shared noise encoder;
  * one forward pass produces one whole future (here: the next H tokens), members in parallel;
  * training scores per-position marginals only (CRPS, two members per example);
    joint structure across positions is never supervised;
  * the model conditions on a bounded recent state (last P tokens, last A encoder frames),
    the counterpart of FGN's second-order Markov conditioning;
  * epistemic uncertainty = several independently trained models sharing the member budget.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EOS = 1


class CondLN(nn.Module):
    """LayerNorm whose gain and bias are functions of the noise embedding."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.gb = nn.Linear(d, 2 * d)

    def forward(self, x, e):
        g, b = self.gb(e).chunk(2, -1)
        return F.layer_norm(x, (self.d,)) * (1 + g) + b


class FGNLayer(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.n1, self.n2, self.n3 = CondLN(d), CondLN(d), CondLN(d)
        self.sa = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ca = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, e, ctx, pad):
        y = self.n1(x, e)
        x = x + self.drop(self.sa(y, y, y, need_weights=False)[0])
        x = x + self.drop(self.ca(self.n2(x, e), ctx, ctx, key_padding_mask=pad, need_weights=False)[0])
        return x + self.drop(self.ff(self.n3(x, e)))


class FGN(nn.Module):
    """noise: 'shared' (FGN), 'slot' (independent noise per future position), 'none' (deterministic)."""

    def __init__(self, vocab, d=256, h=4, ff=1024, layers=4, ctx_layers=2, H=6, P=8, A=64, zdim=32, noise='shared', drop=0.1, split=False, lead=0):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, h=h, ff=ff, layers=layers, ctx_layers=ctx_layers, H=H, P=P, A=A, zdim=zdim, noise=noise, drop=drop, split=split, lead=lead)
        self.lead = lead     # >0: the slot is told how many context tokens lie beyond the audio cutoff (0 .. lead-1)
        if lead:
            self.kemb = nn.Embedding(lead, d)
            nn.init.zeros_(self.kemb.weight)
        self.split = split   # True: the context encoder sees audio only, so it runs once per boundary, not once per hypothesis
        self.H, self.P, self.A, self.zdim, self.noise, self.d = H, P, A, zdim, noise, d
        self.emb = nn.Embedding(vocab, d)
        self.ppos = nn.Parameter(torch.randn(P, d) * 0.02)   # index P-1 = most recent token
        self.apos = nn.Parameter(torch.randn(A, d) * 0.02)   # index A-1 = frame just before the cutoff
        self.aproj = nn.Linear(d, d)
        self.ctx0 = nn.Parameter(torch.zeros(1, 1, d))       # always visible
        self.ctx = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, ff, drop, activation='gelu', batch_first=True, norm_first=True),
                                         ctx_layers, enable_nested_tensor=False)
        self.slots = nn.Parameter(torch.randn(H, d) * 0.02)
        self.znet = nn.Sequential(nn.Linear(zdim, d), nn.SiLU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(FGNLayer(d, h, ff, drop) for _ in range(layers))
        self.norm = CondLN(d)
        self.out = nn.Linear(d, vocab)

    def context(self, prefix, pmask, audio, amask):
        """prefix [B,P], audio [B,A,d], both right-aligned; masks True = present. Noise-free."""
        if self.split:
            return self.join(*self.encode_audio(audio, amask), prefix, pmask)
        b = prefix.shape[0]
        x = torch.cat([self.ctx0.expand(b, 1, -1), self.emb(prefix) + self.ppos, self.aproj(audio).float() + self.apos], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~pmask, ~amask], 1)
        return self.ctx(x, src_key_padding_mask=pad), pad

    def encode_audio(self, audio, amask):
        """split mode: audio-only memory [B,1+A,d], shared by every hypothesis at a boundary."""
        b = audio.shape[0]
        x = torch.cat([self.ctx0.expand(b, 1, -1), self.aproj(audio).float() + self.apos], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~amask], 1)
        return self.ctx(x, src_key_padding_mask=pad), pad

    def join(self, mem, mpad, prefix, pmask):
        """split mode: append the (un-contextualised) recent tokens to the audio memory."""
        return torch.cat([mem, self.emb(prefix) + self.ppos], 1), torch.cat([mpad, ~pmask], 1)

    def draw(self, n, device, gen=None):
        if self.noise == 'none':
            return None
        shape = (n, self.zdim) if self.noise == 'shared' else (n, self.H, self.zdim)
        return torch.randn(shape, generator=gen).to(device)

    def sample(self, ctx, pad, z, k=None):
        """One member per row: logits [B,H,V]. k [B] = number of context tokens that are ahead of the audio."""
        b = ctx.shape[0]
        if z is None:
            e = torch.zeros(b, 1, self.d, device=ctx.device)
        else:
            e = self.znet(z)
            e = e[:, None] if e.dim() == 2 else e
        x = self.slots.expand(b, -1, -1)
        if self.lead and k is not None:
            x = x + self.kemb(k.clamp(max=self.lead - 1))[:, None]
        for layer in self.layers:
            x = layer(x, e, ctx, pad)
        return self.out(self.norm(x, e))

    def members(self, ctx, pad, k, gen=None):
        """k members per context: probabilities [B,k,H,V]."""
        out = []
        for i in range(0, ctx.shape[0], 48):   # bounded memory: 48 contexts x k members at a time
            c, p = ctx[i:i + 48].repeat_interleave(k, 0), pad[i:i + 48].repeat_interleave(k, 0)
            out.append(self.sample(c, p, self.draw(c.shape[0], ctx.device, gen)).float().softmax(-1).view(-1, k, self.H, self.cfg['vocab']))
        out = torch.cat(out)
        if ctx.device.type == 'mps':
            torch.mps.empty_cache()
        return out


def crps_loss(l1, l2, tgt, aux=0.0):
    """Fair two-member CRPS summed over the vocabulary indicators of each position (marginals only).

    Per indicator v: 0.5 (|p1_v - y_v| + |p2_v - y_v|) - 0.5 |p1_v - p2_v|, with y one-hot.
    """
    p1, p2 = l1.float().softmax(-1), l2.float().softmax(-1)
    ok = tgt != -100
    y = tgt.clamp(min=0)[..., None]
    skill = (1 - p1.gather(2, y)[..., 0]) + (1 - p2.gather(2, y)[..., 0])
    spread = 0.5 * (p1 - p2).abs().sum(-1)
    crps = ((skill - spread) * ok).sum() / ok.sum()
    if not aux:
        return crps
    # log score of the two-member marginal predictive distribution (still marginals only). Pure CRPS on
    # softmax outputs collapses: members turn into sharp random one-hots and the skill gradient vanishes.
    mix = -(0.5 * (p1.gather(2, y)[..., 0] + p2.gather(2, y)[..., 0]) + 1e-8).log()
    return crps + aux * (mix * ok).sum() / ok.sum()


def topk_product(logp, k):
    """Exact k best paths under independent per-position marginals. logp [B,H,V] -> [B,k,H]."""
    b, h, v = logp.shape
    score = torch.zeros(b, 1, device=logp.device)
    paths = torch.zeros(b, 1, 0, dtype=torch.long, device=logp.device)
    for j in range(h):
        top, idx = logp[:, j].topk(min(k, v), -1)                      # [B,k]
        cand = (score[:, :, None] + top[:, None, :]).flatten(1)        # [B,n*k]
        score, flat = cand.topk(min(k, cand.shape[1]), -1)
        src, tok = flat // top.shape[1], idx.gather(1, flat % top.shape[1])
        paths = torch.cat([paths.gather(1, src[..., None].expand(-1, -1, j)), tok[..., None]], 2)
    return paths


def build_examples(idxs, tokens, align, enc_lens, chunk, rng, m, model, cuts=None):
    """m random chunk-boundary cutoffs per utterance.

    -> bidx [N], cut [N], prefix [N,P], pmask [N,P], aidx [N,A] (encoder frame index, <0 = absent), tgt [N,H]
    """
    P, A, H = model.P, model.A, model.H
    bidx, cut, prefix, pmask, tgt = [], [], [], [], []
    for b, i in enumerate(idxs):
        if align[i] is None:
            continue
        sp, toks, t = align[i].tolist(), tokens[i], int(enc_lens[b])
        for _ in range(m):
            a = chunk * rng.randint(0, t // chunk)
            c = sum(s < a for s in sp)
            pre = toks[max(0, c - P):c]
            nxt = (toks[c:] + [EOS])[:H]
            bidx.append(b); cut.append(a)
            prefix.append([0] * (P - len(pre)) + pre); pmask.append([False] * (P - len(pre)) + [True] * len(pre))
            tgt.append(nxt + [-100] * (H - len(nxt)))
    cut_t = torch.tensor(cut)
    aidx = cut_t[:, None] - A + torch.arange(A)[None]
    return torch.tensor(bidx), cut_t, torch.tensor(prefix), torch.tensor(pmask), aidx, torch.tensor(tgt)


class FGNPrior:
    """Drop-in prior for the stage-3 decoder: one ensemble forward per chunk boundary, then table lookups.

    At a boundary the ensemble (k members, split over the given models) proposes futures of H tokens
    conditioned on the committed prefix and the audio so far. The score of a token after a hypothesis is
    the mean member probability at that position among members whose path agrees with the hypothesis
    (all members if none agrees). No network call happens on the critical path.
    """
    cross = True

    def __init__(self, models, k=32, eps=0.05, seed=0):
        self.models = [m.eval() for m in models]
        self.k, self.eps = k, eps
        self.gen = torch.Generator().manual_seed(seed)
        m = self.models[0]
        self.d, self.H, self.P, self.A, self.V = m.d, m.H, m.P, m.A, m.cfg['vocab']
        self.prefix, self.base = [], 0
        self.bytes_per_node = self.V * 4
        self.calls = self.steps = self.lookups = self.hits = self.net_calls = 0

    def condition(self, prefix):
        self.prefix = list(prefix)

    @torch.no_grad()
    def set_memory(self, enc):
        P, A = self.P, self.A
        pre = self.prefix[-P:]
        prefix = torch.tensor([[0] * (P - len(pre)) + pre])
        pmask = torch.tensor([[False] * (P - len(pre)) + [True] * len(pre)])
        t = min(A, enc.shape[0])
        audio = torch.zeros(1, A, self.d)
        amask = torch.zeros(1, A, dtype=torch.bool)
        if t:
            audio[0, A - t:] = enc[-t:]; amask[0, A - t:] = True
        probs = []
        for m in self.models:
            ctx, pad = m.context(prefix, pmask, audio, amask)
            probs.append(m.members(ctx, pad, self.k // len(self.models), self.gen)[0])
        self.probs = torch.cat(probs, 0)                 # [k,H,V]
        self.paths = self.probs.argmax(-1).tolist()
        self.base = len(self.prefix)
        self.net_calls += 1

    def step(self, nodes, cut=-1, spec=False):
        self.calls += 1; self.steps += len(nodes)
        for n in nodes:
            path = n.prefix()
            s = path[self.base:]
            j = len(s)
            self.lookups += 1
            if j >= self.H or path[:self.base] != self.prefix:
                p = torch.full((self.V,), 1.0 / self.V)
            else:
                idx = [i for i, pp in enumerate(self.paths) if pp[:j] == s]
                if idx:
                    self.hits += 1
                    p = self.probs[idx, j].mean(0)
                else:
                    p = self.probs[:, j].mean(0)
                p = (1 - self.eps) * p + self.eps / self.V
            n.logp, n.kv, n.spec, n.cut = p.log(), (), spec, cut


class ShortPrior:
    """Bounded-context one-step predictor as a drop-in for the stage-3 Predictor.

    The only state of a hypothesis is its last P tokens, so nothing is cached per node and memory does not grow
    with the utterance. Each step is one forward over (P tokens + A audio frames).
    """
    cross = True

    def __init__(self, model, k=1, seed=0):
        self.m = model.eval()
        self.k = k if model.noise != 'none' else 1
        self.gen = torch.Generator().manual_seed(seed)
        self.d, self.P, self.A, self.V = model.d, model.P, model.A, model.cfg['vocab']
        self.bytes_per_node = self.P * 8
        self.calls = self.steps = 0
        self.base = 0

    def condition(self, prefix):
        self.base = len(prefix)   # tokens verified by the audio that set_memory() will receive

    @torch.no_grad()
    def set_memory(self, enc):
        t = min(self.A, enc.shape[0])
        self.audio = torch.zeros(1, self.A, self.d)
        self.amask = torch.zeros(1, self.A, dtype=torch.bool)
        if t:
            self.audio[0, self.A - t:] = enc[-t:]; self.amask[0, self.A - t:] = True
        self.mem = self.m.encode_audio(self.audio, self.amask) if self.m.split else None

    @torch.no_grad()
    def step(self, nodes, cut=-1, spec=False):
        if not nodes:
            return
        self.calls += 1; self.steps += len(nodes)
        P, b = self.P, len(nodes)
        full = [n.prefix() for n in nodes]
        pre = [f[-P:] for f in full]
        k = torch.tensor([max(0, len(f) - self.base) for f in full]) if self.m.lead else None
        prefix = torch.tensor([[0] * (P - len(p)) + p for p in pre])
        pmask = torch.tensor([[False] * (P - len(p)) + [True] * len(p) for p in pre])
        if self.m.split:
            ctx, pad = self.m.join(self.mem[0].expand(b, -1, -1), self.mem[1].expand(b, -1), prefix, pmask)
        else:
            ctx, pad = self.m.context(prefix, pmask, self.audio.expand(b, -1, -1), self.amask.expand(b, -1))
        if self.m.noise == 'none':
            logp = self.m.sample(ctx, pad, None, k)[:, 0].log_softmax(-1)
        else:
            c, q = ctx.repeat_interleave(self.k, 0), pad.repeat_interleave(self.k, 0)
            kk = k.repeat_interleave(self.k, 0) if k is not None else None
            pr = self.m.sample(c, q, self.m.draw(c.shape[0], ctx.device, self.gen), kk)[:, 0].softmax(-1)
            logp = pr.view(b, self.k, -1).mean(1).clamp_min(1e-8).log()
        for i, n in enumerate(nodes):
            n.logp, n.kv, n.spec, n.cut = logp[i], (), spec, cut
