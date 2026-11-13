# P04. ParaASR: Multi-Token Prediction for Fast and Long-Context LLM-Based Speech Recognition

| Field | Value |
|---|---|
| Authors | Lin, Qingjian; Li, Yuxin; Zhang, Haoyang; Chen, Jun; Huang, Yechang; Tian, Feng; Li, Xie; Zhang, Xiangyu Tony; Liu, Daijiao; Zhang, Yuxin; Gong, Jinglan; Zhao, Bo; Tian, Fei; Yang, Xuerui; Yu, Gang; Zhang, Xiangyu; Jiang, Daxin |
| Source date | 2026-07-31 |
| Publication | arXiv preprint, 2026-07-31 |
| Category | Speculative decoding and path reuse |
| Reading priority | 1 — read first |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 14 pages; `papers/pdf/P04_paraasr.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2607.29279) · [PDF](https://arxiv.org/pdf/2607.29279v1)

## Main idea and method

It is a Multi-Token Prediction system that proposes multiple future tokens simultaneously in LLM-based ASR and adopts the confirmed prefix.

We propose 6 tokens at once as five future token branches of the 4B decoder. Match basic ASR and MTP through step-by-step learning.

## Source-reported evidence

MTP-5 in WenetSpeech meeting evaluation adopts the average 5.0/6 token. RTF 0.0053 is a single H800, single concurrency, 30 second voice 100 evaluation conditions.

Evidence location: PDF Page 8–11, Tables 1–4, Section 4.

## Connection to this project — interpretation

Helps balance accepted length, candidate width and rollback cost, distinguishing acoustic verification from linguistic plausibility.

## Limitations when applying the result

Multi-token decoding with received audio differs from prediction before future audio arrives. Some dataset numbers differ between Tables 1 and 4; record table and conditions with each accuracy comparison.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `b6b1148d08ea3cc48d28d64d7c469646ca71bfc1f18fa129a27998004481decc`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
