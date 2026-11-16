# P40. CUSIDE-array: A Streaming Multi-Channel End-to-End Speech Recognition System with Realistic Evaluations

| Field | Value |
|---|---|
| Authors | Kong, Xiangzhu; Ning, Tianqi; Huang, Hao; Ou, Zhijian |
| Source date | 2024-07-13 |
| Publication | ISCSLP 2024 (acceptance stated on arXiv) |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P40_cuside-array.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2407.09807) · [PDF](https://arxiv.org/pdf/2407.09807v1)

## Main idea and method

CUSIDE is combined with a streaming neural beamformer for multi-channel ASR and distribution-shift evaluation.

Streaming neural MVDR feeds the acoustic backend. Future-feature simulation is applied to that backend; streaming, non-streaming and frame-simulation losses are jointly optimized.

## Source-reported evidence

Collected v1, AISHELL-4 Table 1: jointly trained multi-channel baseline CER 36.68% at 400ms; real right context 32.51% at 800ms; simulated future context 35.96% at 402ms.

Evidence location: Collected v1 PDF pp. 2-4, system diagram and Table 1.

## Connection to this project — interpretation

A useful robustness extension of the learned-future family. It motivates matched noise and channel controls, rather than assuming clean-speech gains survive a new input distribution.

## Limitations when applying the result

The multi-channel/MVDR setting is not our single-channel benchmark. Simulation does not universally recover real-future accuracy. A later v2 exists; these collected results remain explicitly tied to v1.

## Code and models

- [https://github.com/thu-spmi/CAT/blob/master/docs/cuside-array.md](https://github.com/thu-spmi/CAT/blob/master/docs/cuside-array.md)

Official CAT recipe documentation was located and its public page checked. No model was run.

## Source version

PDF SHA-256: `39b10ce64a60ab68b884ce6fdcd0653ece0f9846aa62e4941e018662909c6116`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
