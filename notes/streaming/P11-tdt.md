# P11. Efficient Sequence Transduction by Jointly Predicting Tokens and Durations

| Field | Value |
|---|---|
| Authors | Hainan Xu; Fei Jia; Somshubra Majumdar; He Huang; Shinji Watanabe; Boris Ginsburg |
| Source date | 2023-07-03 |
| Publication | ICML 2023, PMLR 202 |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 2 — Skip time base |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 23 pages; `papers/pdf/P11_tdt.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://proceedings.mlr.press/v202/xu23g.html) · [PDF](https://proceedings.mlr.press/v202/xu23g/xu23g.pdf)

## Main idea and method

Token-and-Duration Transducer predicts input-frame advancement together with tokens.

Independently normalized token and duration outputs let transducer decoding advance by multiple frames.

## Source-reported evidence

Reports ASR inference speedup up to 2.82 times over a conventional transducer.

Evidence location: TDT formulation and ASR experiments.

## Connection to this project — interpretation

A comparison that reduces joiner calls during verification/output, independently of speculative candidate generation.

## Limitations when applying the result

This does not skip all acoustic encoding. Native streaming support of TDT checkpoints must be checked per model configuration.

## Code and models

- [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

Original NVIDIA/NeMo links now redirect to NVIDIA-NeMo/Speech. Per-model recipes, weights and Mac operation support were not execution-tested.

## Source version

PDF SHA-256: `b637cea35e648e9f3469dac7f76d67b09308947b53d1ada81f5991100fd45519`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
