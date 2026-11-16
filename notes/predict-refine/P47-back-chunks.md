# P47. Learning from Back Chunks: Acquiring More Future Knowledge for Streaming ASR Models via Self Distillation

| Field | Value |
|---|---|
| Authors | Yang, Yuting; Ma, Guodong; Li, Yuke; Du, Binbin; Zhu, Haoqi; Ruan, Liang |
| Source date | 2024-09-01 |
| Publication | Interspeech 2024 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P47_back-chunks.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2024/yang24m_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2024/yang24m_interspeech.pdf)

## Main idea and method

Future-aware Transformer self-distillation transfers information from later chunks into look-ahead representations of earlier chunks.

Match corresponding look-ahead/next-chunk features using MSE, or prediction distributions using KL, alongside the hybrid CTC/attention ASR objective. Dynamic chunking and a real look-ahead window are retained.

## Source-reported evidence

AISHELL-1, chunk 12 plus look-ahead 4 (640ms window), no LM: Table 2 feature-distilled test CER 6.8% first pass and 5.9% rescored, versus same-look-ahead baseline 7.1/6.1%. AISHELL-2 is also evaluated.

Evidence location: PDF p. 3, Eqs. (2)-(5) and Table 1; p. 4, Table 2.

## Connection to this project — interpretation

A prior for jointly learning representations from future-aware supervision. It argues for task-aware objectives and a matched real-context baseline before interpreting representation MAE alone.

## Limitations when applying the result

Distilling future knowledge during training does not supply actual future audio at inference or demonstrate encoder omission. Report the look-ahead window and both decoding passes with any CER claim.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review; this is not evidence that they do not exist.

## Source version

PDF SHA-256: `f99bfd7603cefb397edd811389f0d79e2204bb900fa4365392dd818bf26d6bcd`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
