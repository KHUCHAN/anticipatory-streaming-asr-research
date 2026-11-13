# P08. Blockwise Parallel Decoding for Deep Autoregressive Models

| Field | Value |
|---|---|
| Authors | Stern, Mitchell; Shazeer, Noam; Uszkoreit, Jakob |
| Source date | 2018-11-07 |
| Publication | arXiv preprint, 2018 |
| Category | Speculative decoding and path reuse |
| Reading priority | 2 — multi-token base |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 10 pages; `papers/pdf/P08_blockwise-parallel.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/1811.03115) · [PDF](https://arxiv.org/pdf/1811.03115v1)

## Main idea and method

Proposes several future outputs in parallel, then advances along the longest prefix accepted by a scoring model.

An auxiliary model predicts several subsequent positions and uses verification/back-off to reduce sequential decoding iterations.

## Source-reported evidence

Evaluates iteration counts and measured runtime improvements in machine translation and image super-resolution.

Evidence location: blockwise decoding algorithm and experiments.

## Connection to this project — interpretation

Earlier work on continuous candidate proposals and verified-prefix acceptance, relevant to bounding novelty claims.

## Limitations when applying the result

Does not address acoustic arrival or word alignment. Distinguish quality-preserving settings from quality/speed trade-offs.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `e34c3c9ddd1066b7dc2c11406580a586a6d483a23e6362e1414f760b655c52e1`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
