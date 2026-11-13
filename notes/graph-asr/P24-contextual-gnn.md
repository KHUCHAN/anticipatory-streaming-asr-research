# P24. Graph Neural Networks for Contextual ASR with the Tree-Constrained Pointer Generator

| Field | Value |
|---|---|
| Authors | Sun, Guangzhi; Zhang, Chao; Woodland, Phil |
| Source date | 2023-05-30 |
| Publication | Reviewed source: arXiv preprint, 2023 |
| Category | Graph-based speech recognition |
| Reading priority | 2 — see candidate graph |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 10 pages; `papers/pdf/P24_contextual-gnn.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2305.18824) · [PDF](https://arxiv.org/pdf/2305.18824v1)

## Main idea and method

Contextual biasing The prefix tree of words is encoded with GNN to improve recognition of rare and unregistered words.

By applying node encoding such as tree-RNN, GCN, and GraphSAGE to TCPGen, lower wordpiece information is used for the lookahead.

## Source-reported evidence

We evaluate contextual ASR in LibriSpeech and AMI. Distinguish between rare-word WER and full WER.

Evidence location: GNN prefix-tree encoding and contextual ASR results.

## Connection to this project — interpretation

Relevant to representing shared prefixes and branches of multiple candidates as a graph.

## Limitations when applying the result

The graph mainly represents biasing vocabulary, not acoustic frames. It does not verify words before their audio arrives.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `f55d77c0478930645875d6bf0d4ef7848e8d0d8dd4ad5387a3f8782bfa5d70c7`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
