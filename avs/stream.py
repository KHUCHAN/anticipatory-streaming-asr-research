"""Stage 3: predict-verify streaming decoding over a prefix trie.

Acoustic evidence is the CTC posterior of each arrived chunk. Hypotheses are
trie nodes; a node owns the predictor state of its own position only (per-layer
key/value), so common prefixes are stored once. Predictor states can be computed
ahead of time (while waiting for the next chunk) and are reused on arrival.
"""
import math, time
import torch
import torch.nn.functional as F

from .text import SOS

NEG = -1e30


def rope_b(x, pos):
    """x [B,H,1,D], pos [B] -> rotary with one position per batch item."""
    half = x.shape[-1] // 2
    freqs = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
    ang = pos[:, None] * freqs
    cos, sin = ang.cos()[:, None, None, :], ang.sin()[:, None, None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


def logadd(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG / 2 else a + math.log1p(math.exp(b - a))


class Node:
    __slots__ = ('tok', 'parent', 'depth', 'children', 'kv', 'logp', 'spec', 'used', 'cut', 'gen', 'born')

    def __init__(self, tok, parent):
        self.tok, self.parent = tok, parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.children = {}
        self.kv = None      # per-layer (k, v) of this position, each [H, D]
        self.logp = None    # log P(next token | prefix ending here), [V]
        self.spec = False   # state was computed speculatively (off the critical path)
        self.used = False   # state was consumed by the search
        self.cut = -1       # audio cutoff (encoder frames) the state was computed with
        self.gen = 0        # last round in which the node was created or visited
        self.born = -1      # round in which the node first entered the beam

    def prefix(self):
        out, n = [], self
        while n.parent is not None:
            out.append(n.tok); n = n.parent
        return out[::-1]


class Predictor:
    """Incremental, batched single-step interface to a (text-only or audio-aware) Decoder."""

    def __init__(self, dec, hist=0, release_cache=False):
        self.dec, self.cross, self.hist = dec.eval(), dec.cross, hist
        if release_cache and hist <= 0:
            raise ValueError('Cache release requires a positive per-layer history limit')
        self.release_cache = release_cache
        self.d = dec.emb.embedding_dim
        self.h = dec.layers[0].self_att.h
        self.mem_kv = None
        self.calls = self.steps = 0
        self.bytes_per_node = 0

    def set_memory(self, enc):
        """enc [T,d] = all encoder frames that have arrived. Recomputed per chunk."""
        if not self.cross:
            return
        mem = torch.cat([self.dec.mem0[0], enc], 0)[None]
        d = mem.shape[-1]
        split = lambda y: y.view(1, -1, self.h, d // self.h).transpose(1, 2)
        self.mem_kv = [(split(l.cross_att.k(mem)), split(l.cross_att.v(mem))) for l in self.dec.layers]

    @torch.no_grad()
    def step(self, nodes, cut=-1, spec=False):
        """Fill kv/logp for nodes whose parents already have state (or are the root)."""
        if not nodes:
            return
        self.calls += 1; self.steps += len(nodes)
        b, dec, h = len(nodes), self.dec, self.h
        paths = []
        for n in nodes:
            p, m = [], n.parent
            while m is not None and (not self.hist or len(p) < self.hist):
                p.append(m); m = m.parent
            paths.append(p[::-1])
        tmax = max(len(p) for p in paths) + 1
        x = dec.emb(torch.tensor([n.tok for n in nodes]))[:, None]
        d = x.shape[-1]; dh = d // h
        pos = torch.tensor([float(n.depth) for n in nodes])
        allow = torch.zeros(b, 1, 1, tmax, dtype=torch.bool)
        for i, p in enumerate(paths):
            allow[i, 0, 0, :len(p)] = True
            allow[i, 0, 0, tmax - 1] = True
        new_kv = [[] for _ in nodes]
        for li, layer in enumerate(dec.layers):
            y = layer.n1(x)
            att = layer.self_att
            q = att.q(y).view(b, 1, h, dh).transpose(1, 2)
            k = att.k(y).view(b, 1, h, dh).transpose(1, 2)
            v = att.v(y).view(b, 1, h, dh).transpose(1, 2)
            q, k = rope_b(q, pos), rope_b(k, pos)
            K = torch.zeros(b, h, tmax, dh); V = torch.zeros(b, h, tmax, dh)
            for i, p in enumerate(paths):
                if p:
                    K[i, :, :len(p)] = torch.stack([m.kv[li][0] for m in p], 1)
                    V[i, :, :len(p)] = torch.stack([m.kv[li][1] for m in p], 1)
            K[:, :, tmax - 1] = k[:, :, 0]; V[:, :, tmax - 1] = v[:, :, 0]
            for i in range(b):
                new_kv[i].append((k[i, :, 0], v[i, :, 0]))
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=allow)
            x = x + att.o(o.transpose(1, 2).reshape(b, 1, d))
            if self.cross:
                ca = layer.cross_att
                mk, mv = self.mem_kv[li]
                q2 = ca.q(layer.n2(x)).view(b, 1, h, dh).transpose(1, 2)
                o2 = F.scaled_dot_product_attention(q2, mk.expand(b, -1, -1, -1), mv.expand(b, -1, -1, -1))
                x = x + ca.o(o2.transpose(1, 2).reshape(b, 1, d))
            x = x + layer.ff(layer.n3(x))
        logp = dec.out(dec.norm(x))[:, 0].log_softmax(-1)
        for i, n in enumerate(nodes):
            n.kv, n.logp, n.spec, n.cut = new_kv[i], logp[i], spec, cut
        self.bytes_per_node = sum(k.numel() + v.numel() for k, v in new_kv[0]) * 4 + logp.shape[1] * 4


class WindowPredictor:
    """Same decoder as Predictor, but every step sees only the last `window` input tokens (strict bounded context).

    Nothing is cached per hypothesis: the state of a hypothesis is its last `window` tokens.
    """

    def __init__(self, dec, window):
        self.dec, self.cross, self.window = dec.eval(), True, window
        self.d = dec.emb.embedding_dim
        self.bytes_per_node = window * 8
        self.calls = self.steps = 0

    def set_memory(self, enc):
        self.mem = enc[None]

    @torch.no_grad()
    def step(self, nodes, cut=-1, spec=False):
        if not nodes:
            return
        self.calls += 1; self.steps += len(nodes)
        w, b = self.window, len(nodes)
        wins = []
        for n in nodes:
            seq = ([SOS] + n.prefix())[-w:]
            wins.append([SOS] * (w - len(seq)) + seq)
        mask = torch.ones(b, 1, self.mem.shape[1], dtype=torch.bool)
        logp = self.dec(torch.tensor(wins), self.mem.expand(b, -1, -1), mask)[:, -1].log_softmax(-1)
        for i, n in enumerate(nodes):
            n.logp, n.kv, n.spec, n.cut = logp[i], (), spec, cut


class Decoder3:
    """CTC prefix beam search + predictor fusion, commit by beam agreement, optional speculation.

    fresh=False: predictor scores used while chunk n+1 is decoded were computed with audio
                 up to boundary n (anticipatory scores; the only mode that allows reuse).
    fresh=True : scores are recomputed with the audio of the current chunk (control).
    """

    def __init__(self, predictor=None, beam=4, weight=0.3, bonus=0.0, top_ctc=6, ctc_prune=-5.0,
                 speculate=False, spec_k=2, spec_h=2, node_budget=32, fresh=False, hold=1, margin=10.0):
        self.p, self.beam, self.w, self.bonus = predictor, beam, weight, bonus
        self.top_ctc, self.ctc_prune = top_ctc, ctc_prune
        self.speculate, self.spec_k, self.spec_h, self.budget, self.fresh = speculate, spec_k, spec_h, node_budget, fresh
        self.hold, self.margin = hold, margin  # hold: chunks a token of the best path waits before forced commit (None = agreement only)

    def reset(self):
        self.root = Node(SOS, None)
        self.hyps = {self.root: (0.0, NEG, 0.0)}  # node -> (log p_blank, log p_nonblank, fusion score)
        self.committed = self.root
        self.round = 0
        self.cut = 0
        self.best_prev = []
        self.stats = dict(onpath_steps=0, onpath_calls=0, onpath_s=0.0, spec_steps=0, spec_calls=0, spec_s=0.0,
                          reused=0, spec_used=0, peak_nodes=1, revised=0, mem_s=0.0)
        if self.p:
            if hasattr(self.p, 'condition'):
                self.p.condition([])
            if self.p.cross:
                self.p.set_memory(torch.zeros(0, self.p.d))
            self.p.step([self.root], cut=0, spec=True)  # computed before any audio arrives

    @staticmethod
    def score(v):
        return logadd(v[0], v[1]) + v[2]

    def _stale(self, n):
        """Audio-aware states are valid only for the audio cutoff they were computed with."""
        return n.kv is None or (self.p.cross and n.cut != self.cut)

    def _refresh(self):
        todo = [n for n in self.hyps if self._stale(n)]
        for depth in sorted({n.depth for n in todo}):
            self.p.step([n for n in todo if n.depth == depth], cut=self.cut, spec=True)
        for n in todo:
            n.used = False
        if todo:
            self.stats['spec_calls'] += 1
        self.stats['refresh_steps'] = self.stats.get('refresh_steps', 0) + len(todo)
        return len(todo)

    def _need(self, nodes):
        """Predictor state for nodes: reuse if present, else compute on the critical path."""
        missing = []
        for n in nodes:
            if self._stale(n):
                missing.append(n)
            elif not n.used:
                n.used = True
                self.stats['reused'] += 1
                self.stats['spec_used'] += n.spec
        if missing:
            t = time.perf_counter()
            for depth in sorted({n.depth for n in missing}):
                self.p.step([n for n in missing if n.depth == depth], cut=self.cut, spec=False)
            self.stats['onpath_s'] += time.perf_counter() - t
            self.stats['onpath_steps'] += len(missing); self.stats['onpath_calls'] += 1
            for n in missing:
                n.used = True

    def child(self, node, tok):
        c = node.children.get(tok)
        if c is None:
            c = node.children[tok] = Node(tok, node)
        c.gen = self.round
        return c

    def advance(self, logp_chunk, enc_so_far=None):
        """logp_chunk [n,V]: CTC log-probs of newly arrived frames. Returns newly committed tokens."""
        self.round += 1
        if self.p and self.p.cross and self.fresh:
            t = time.perf_counter()
            self.cut = enc_so_far.shape[0]
            self.p.set_memory(enc_so_far)
            self.stats['mem_s'] += time.perf_counter() - t
        rows = logp_chunk.tolist()
        for row in rows:
            blank = row[0]
            cand = sorted(((lp, t) for t, lp in enumerate(row) if t >= 2 and lp > self.ctc_prune), reverse=True)[:self.top_ctc]
            if self.p and cand:
                self._need(list(self.hyps))
            nxt = {}
            for node, (pb, pnb, ps) in self.hyps.items():
                tot = logadd(pb, pnb)
                e = nxt.get(node, (NEG, NEG, ps))
                nb_same = pnb + row[node.tok] if node.parent is not None else NEG
                nxt[node] = (logadd(e[0], tot + blank), logadd(e[1], nb_same), ps)
                if not cand:
                    continue
                plog = node.logp.tolist() if self.p else None
                for lp, tok in cand:
                    base = pb if (node.parent is not None and tok == node.tok) else tot
                    if base <= NEG / 2:
                        continue
                    c = self.child(node, tok)
                    if c.born < 0:
                        c.born = self.round
                    ps_c = ps + self.bonus + (self.w * plog[tok] if self.p else 0.0)
                    e = nxt.get(c, (NEG, NEG, ps_c))
                    nxt[c] = (e[0], logadd(e[1], base + lp), ps_c)
            ranked = sorted(nxt.items(), key=lambda kv: -self.score(kv[1]))[:self.beam]
            floor = self.score(ranked[0][1]) - self.margin
            self.hyps = {n: v for n, v in ranked if self.score(v) >= floor}
        best = max(self.hyps.items(), key=lambda kv: self.score(kv[1]))[0].prefix()
        keep = 0
        while keep < min(len(best), len(self.best_prev)) and best[keep] == self.best_prev[keep]:
            keep += 1
        self.stats['revised'] += len(self.best_prev) - keep
        self.best_prev = best
        out = self._commit()
        if not self.speculate:
            if self.p and self.p.cross and not self.fresh:  # no rollout: only refresh live hypotheses for the new cutoff
                t = time.perf_counter()
                self.cut = enc_so_far.shape[0]
                if hasattr(self.p, 'condition'):
                    self.p.condition(self.committed.prefix())
                self.p.set_memory(enc_so_far)
                self._refresh()
                self.stats['spec_s'] += time.perf_counter() - t
            self._prune()
        return out

    def _commit(self, final=False):
        if final:
            target = max(self.hyps.items(), key=lambda kv: self.score(kv[1]))[0]
        else:
            paths = []
            for n in self.hyps:
                p, m = [], n
                while m is not None:
                    p.append(m); m = m.parent
                paths.append(p[::-1])
            target, i = self.root, 0
            while all(len(p) > i for p in paths) and all(p[i] is paths[0][i] for p in paths):
                target = paths[0][i]; i += 1
            if self.hold is not None:
                m = max(self.hyps.items(), key=lambda kv: self.score(kv[1]))[0]
                while m.parent is not None and m.born > self.round - self.hold:
                    m = m.parent
                if m.depth > target.depth:
                    target = m
                    alive = {}
                    for n, v in self.hyps.items():
                        k = n
                        while k is not None and k.depth > target.depth:
                            k = k.parent
                        if k is target:
                            alive[n] = v
                    self.hyps = alive
        new, m = [], target
        while m is not None and m.depth > self.committed.depth:
            new.append(m.tok); m = m.parent
        if target.depth > self.committed.depth:
            self.committed = target
        return new[::-1]

    def finish(self):
        return self._commit(final=True)

    def speculate_now(self, enc_so_far=None):
        """While waiting for the next chunk: roll out likely continuations of each live hypothesis."""
        if not (self.p and self.speculate):
            return 0.0
        t = time.perf_counter()
        if self.p.cross:
            self.cut = enc_so_far.shape[0]
            self.p.set_memory(enc_so_far)
        made = 0

        def compute(nodes):
            nonlocal made
            todo = [n for n in nodes if self._stale(n)]
            if todo:
                for depth in sorted({n.depth for n in todo}):
                    self.p.step([n for n in todo if n.depth == depth], cut=self.cut, spec=True)
                for n in todo:
                    n.used = False
                made += len(todo); self.stats['spec_calls'] += 1

        frontier = [(0.0, n) for n in self.hyps]
        self._refresh()
        for _ in range(self.spec_h):
            cand = []
            for sc, n in frontier:
                vals, toks = n.logp[2:].topk(self.spec_k)
                cand += [(sc + float(v), self.child(n, int(tk) + 2)) for v, tk in zip(vals, toks)]
            cand.sort(key=lambda x: -x[0])
            frontier = cand[:max(0, self.budget - made)]
            if not frontier:
                break
            compute([c for _, c in frontier])
        self.stats['spec_steps'] += made
        self._prune()
        dt = time.perf_counter() - t
        self.stats['spec_s'] += dt
        return dt

    def _prune(self):
        """Keep ancestors of live hypotheses and nodes touched in this round; drop the rest."""
        keep = set()
        for n in self.hyps:
            m = n
            while m is not None and m not in keep:
                keep.add(m); m = m.parent
        tensor_keep = None
        if self.p and getattr(self.p, 'release_cache', False):
            tensor_keep = set()
            frontier = list(self.hyps) + [self.committed]
            if self.speculate:
                # Speculative candidates can be consumed in the next chunk.
                frontier += [n for n in keep if n.gen == self.round]
            for n in frontier:
                for _ in range(self.p.hist + 1):
                    if n is None:
                        break
                    tensor_keep.add(n)
                    n = n.parent
        count, cached, stack = 0, 0, [self.root]
        while stack:
            n = stack.pop()
            if tensor_keep is not None and n not in tensor_keep:
                n.kv = None
                n.logp = None
            cached += n.kv is not None
            count += n.kv is not None and n.depth > self.committed.depth
            for tok in list(n.children):
                c = n.children[tok]
                if c in keep or c.gen == self.round:
                    stack.append(c)
                else:
                    del n.children[tok]
        self.stats['peak_nodes'] = max(self.stats['peak_nodes'], count)
        self.stats['cached_nodes'] = cached
        self.stats['peak_cached_nodes'] = max(self.stats.get('peak_cached_nodes', 0), cached)

    def state_usage(self):
        """Resident predictor tensor storage, separate from audio and token metadata.

        Count backing storages once: per-node tensors can be views into a batched
        projection. Model weights, temporary workspaces and process RSS are not
        part of these cache sizes. Parent/child token objects remain for output.
        """
        text_storages, audio_storages = {}, {}
        logical, nodes, cached, stack = 0, 0, 0, [self.root]

        def add(tensor, storages):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()

        while stack:
            n = stack.pop()
            nodes += 1
            if n.kv is not None:
                cached += 1
                for k, v in n.kv:
                    logical += (k.numel() + v.numel()) * k.element_size()
                    add(k, text_storages); add(v, text_storages)
            if n.logp is not None:
                logical += n.logp.numel() * n.logp.element_size()
                add(n.logp, text_storages)
            stack.extend(n.children.values())
        if self.p and getattr(self.p, 'mem_kv', None) is not None:
            for k, v in self.p.mem_kv:
                add(k, audio_storages); add(v, audio_storages)
        return {'token_nodes':nodes, 'cached_nodes':cached,
                'text_logical_bytes':logical, 'text_storage_bytes':sum(text_storages.values()),
                'audio_storage_bytes':sum(audio_storages.values())}
