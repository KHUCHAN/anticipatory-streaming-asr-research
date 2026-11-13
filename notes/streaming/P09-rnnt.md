# P09. Sequence Transduction with Recurrent Neural Networks

| Field | Value |
|---|---|
| Authors | Graves, Alex |
| Source date | 2012-11-14 |
| Publication | arXiv preprint, 2012 |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 1 — Sort theory |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 9 pages; `papers/pdf/P09_rnnt.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/1211.3711) · [PDF](https://arxiv.org/pdf/1211.3711v1)

## Main idea and method

Original RNN Transducer paper probabilistically handles unequal audio and label sequence lengths.

Combines the input encoder and output-history prediction network in a joint distribution and sums over possible alignments.

## Source-reported evidence

Sequence transduction is evaluated using TIMIT's phoneme recognition.

Evidence location: transducer formulation and TIMIT experiments.

## Connection to this project — interpretation

Provides acoustic-alignment foundations for candidate verification when speech frames and words are not one-to-one.

## Limitations when applying the result

The original setup does not automatically specify a modern causal streaming implementation. Define encoder future context and decoding policy separately.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `d28fddcfc95a646ad60d3007a7f5fbd0fbba35eec418c018a595da81089f06cf`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
