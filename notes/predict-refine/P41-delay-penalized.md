# P41. Delay-penalized transducer for low-latency streaming ASR

| Field | Value |
|---|---|
| Authors | Kang, Wei; Yao, Zengwei; Kuang, Fangjun; Guo, Liyong; Yang, Xiaoyu; Lin, Long; Żelasko, Piotr; Povey, Daniel |
| Source date | 2022-10-31 |
| Publication | ICASSP 2023 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P41_delay-penalized.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2211.00490) · [PDF](https://arxiv.org/pdf/2211.00490v1) · [Additional primary source 1](https://ieeexplore.ieee.org/document/10096378/) · [Additional primary source 2](https://www.danielpovey.com/files/2023_icassp_delay_penalized_transducer.pdf)

## Main idea and method

A time-dependent training penalty encourages nonblank transducer emissions earlier in the alignment lattice.

Add lambda*((T-1)/2-t) to normalized nonblank log scores used by transducer recursion. The approximation penalizes expected delay without external alignments.

## Source-reported evidence

LibriSpeech Conformer, Table 1: lambda 0 to 0.003 changes clean WER 3.40 to 3.70% and mean alignment delay 373 to 102ms; other WER 8.66 to 9.35%, delay 374 to 137ms.

Evidence location: PDF Eq. (19), Table 1; Figure 3 total-latency discussion.

## Connection to this project — interpretation

An especially relevant k2/icefall-family control for reducing intrinsic emission delay before adding our gate or forecast.

## Limitations when applying the result

The table alignment delay and a complete streaming delay are different: the later total-latency figure adds half a chunk. Earlier emission trades some accuracy; no unconditional native-transcript preservation is claimed.

## Code and models

- [https://github.com/k2-fsa/k2](https://github.com/k2-fsa/k2)

The paper identifies k2 implementation support. Exact recipe and checkpoint behavior were not reproduced.

## Source version

PDF SHA-256: `96048cf708dbcf6d367f7ae96ca87c482556267271caa2d377380c968c00f4ae`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
