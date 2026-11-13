# P05. Fast Inference from Transformers via Speculative Decoding

| Field | Value |
|---|---|
| Authors | Leviathan, Yaniv; Kalman, Matan; Matias, Yossi |
| Source date | 2022-11-30 |
| Publication | ICML 2023; arXiv first released 2022 |
| Category | Speculative decoding and path reuse |
| Reading priority | 1 — read first |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 13 pages; `papers/pdf/P05_speculative-decoding.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2211.17192) · [PDF](https://arxiv.org/pdf/2211.17192v2) · [Additional primary source 1](https://proceedings.mlr.press/v202/leviathan23a.html)

## Main idea and method

Foundational speculative decoding: a small draft model proposes candidates for parallel verification by a larger target.

The target distribution is maintained through adoption and correction sampling using the draft distribution and target distribution.

## Source-reported evidence

Targeting T5-XXL, we report 2–3 times acceleration compared to the standard implementation. What is preserved is the generation distribution of the target model.

Evidence location: Algorithm 1, correctness analysis, experiments.

## Connection to this project — interpretation

A starting point for probabilistic acceptance rules in an accept-if-correct, otherwise-correct procedure.

## Limitations when applying the result

ASR correct answers or audio content that has not yet been heard are not guaranteed. If the input changes, it must be verified again with the target distribution under that input.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `8967e2daab74178b8a73a6f0b0780c5cbb40f9f6dcbd5c456b24a89c128c52ed`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
