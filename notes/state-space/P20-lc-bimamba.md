# P20. Attention-Free Dual-Mode ASR with Latency-Controlled Selective State Spaces

| Field | Value |
|---|---|
| Authors | Moriya, Takafumi; Mimura, Masato; Matsui, Kiyoaki; Sato, Hiroshi; Matsuura, Kohei |
| Source date | 2025 |
| Publication | Interspeech 2025 |
| Category | State-space models and ConMamba |
| Reading priority | 1 — streaming state space basis |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 5 pages; `papers/pdf/P20_lc-bimamba.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://www.isca-archive.org/interspeech_2025/moriya25_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2025/moriya25_interspeech.pdf)

## Main idea and method

This is an attention-free dual-mode ASR study using a Mamba-based encoder for streaming and offline.

We use limited future context within chunks with LC-BiMamba and improve transducer search with AdaALSD.

## Source-reported evidence

Compare Conformer and WER·RTF in TLv2 and LibriSpeech. The LibriSpeech table shows the results of specific settings: RTF 0.58 vs. 0.50/0.51.

Evidence location: PDF 1–4 page, Table 5 and latency settings.

## Connection to this project — interpretation

Direct support for controlling chunks and future context when comparing state-space models in actual streaming conditions.

## Limitations when applying the result

latency-controlled does not mean that the future context is 0. The expression constant-time should not be read to mean that the entire voice is processed in O(1) time.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `de00b9bd79b0da7b7fb421e167df1deb6398628ae5fc2e7bf2f91b47cb3cf155`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
