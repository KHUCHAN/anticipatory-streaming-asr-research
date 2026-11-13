# P01. Speculative Speech Recognition by Audio-Prefixed Low-Rank Adaptation of Language Models

| Field | Value |
|---|---|
| Authors | Yusuf, Bolaji; Baskar, Murali Karthick; Rosenberg, Andrew; Ramabhadran, Bhuvana |
| Source date | 2024-07-05 |
| Publication | Interspeech 2024 |
| Category | Prediction before audio arrival |
| Reading priority | 1 — read first |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 5 pages; `papers/pdf/P01_ssr.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2407.04641) · [PDF](https://arxiv.org/pdf/2407.04641v1) · [Additional primary source 1](https://www.isca-archive.org/interspeech_2024/yusuf24_interspeech.html)

## Main idea and method

Speculative Speech Recognition predicts an unheard continuation from observed audio and transcript.

Connect a language model applying voice conditional soft prefix and LoRA to the Conformer-Transducer output. We also propose a learning label alignment that takes ASR errors into account.

## Source-reported evidence

Evaluating speech conditional prediction in LibriSpeech and multi-domain experiments. The default SOWER setting is 1 seconds in the future, 8 candidates, and oracle evaluation to select the best candidate a posteriori.

Evidence location: PDF Page 1–4, Figure 1, Section 2.2, Eq. (8), Tables 1–3.

## Connection to this project — interpretation

The closest prior work to preparing the next words and verifying them with new audio; it motivates a text-only versus audio-aware forecasting comparison.

## Limitations when applying the result

The SOWER improvement cannot be translated into a reduction in word commit latency in the actual system. This paper distinguishes the problem from general speculative decoding, which reduces the computation of a large model into a small model.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `8c556bc0943cc66a658cf9f2048743d5f548714cc683e5c20dc3a83fa9cca7ec`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
