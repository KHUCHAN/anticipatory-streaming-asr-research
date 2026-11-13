# P16. Mamba: Linear-Time Sequence Modeling with Selective State Spaces

| Field | Value |
|---|---|
| Authors | Gu, Albert; Dao, Tri |
| Source date | 2023-12-01 |
| Publication | Mamba original paper, arXiv 2023 |
| Category | State-space models and ConMamba |
| Reading priority | 2 — State compression candidate |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 36 pages; `papers/pdf/P16_mamba.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2312.00752) · [PDF](https://arxiv.org/pdf/2312.00752v2)

## Main idea and method

An input-selective state-space model for efficient long-sequence processing.

Uses input-dependent SSM parameters and hardware-aware scan, updating recurrent state at inference.

## Source-reported evidence

Evaluates quality and throughput in sequence modeling including language and audio.

Evidence location: selective SSM, hardware-aware scan and evaluation.

## Connection to this project — interpretation

Core background for retaining bounded state instead of a complete historical KV cache.

## Limitations when applying the result

Mamba does not automatically resolve ASR alignment or future word validation. The CUDA optimization results of the original implementation do not generalize to Apple MPS performance.

## Code and models

- [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `adf70ed1803c85b1899dec3e21f3af0b124411439e8654b840ea65f7b9f52b2e`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
