# P15. Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition

| Field | Value |
|---|---|
| Authors | Shi, Yangyang; Wang, Yongqiang; Wu, Chunyang; Yeh, Ching-Feng; Chan, Julian; Zhang, Frank; Le, Duc; Seltzer, Mike |
| Source date | 2020-10-21 |
| Publication | ICASSP 2021; arXiv first released 2020 |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 2 — Limited context comparison |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 5 pages; `papers/pdf/P15_emformer.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2010.10759) · [PDF](https://arxiv.org/pdf/2010.10759v4)

## Main idea and method

A low-latency chunked Transformer acoustic model that summarizes historical context in a memory bank.

Combines left-context KV caches, summarized memory and parallel block training.

## Source-reported evidence

Compare WER and RTF at different latency settings in LibriSpeech.

Evidence location: Evaluation by emformer memory structure and latency.

## Connection to this project — interpretation

This shows that processing history through retained state rather than storing everything can be studied without removing all attention.

## Limitations when applying the result

It is not the same structure as a recurrent state like Mamba. You need to compare the lookahead, chunk, and memory bank budgets.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `ec1141086fdd61862a9baf64e6128bd4d62d3f9f50351d23fe739b92a94dc369`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
