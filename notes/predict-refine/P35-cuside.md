# P35. CUSIDE: Chunking, Simulating Future Context and Decoding for Streaming ASR

| Field | Value |
|---|---|
| Authors | An, Keyu; Zheng, Huahuan; Ou, Zhijian; Xiang, Hongyu; Ding, Ke; Wan, Guanglu |
| Source date | 2022-03-31 |
| Publication | Interspeech 2022 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P35_cuside.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2022/an22_interspeech.html) · [PDF](https://arxiv.org/pdf/2203.16758v2) · [Additional primary source 1](https://arxiv.org/abs/2203.16758)

## Main idea and method

Learned future acoustic-feature simulation replaces the wait for real right context in streaming ASR.

A GRU and feed-forward simulator predict future input features. Joint streaming ASR, non-streaming ASR and L1 simulation objectives use stochastic real, absent and simulated future context. The current chunk still traverses the acoustic encoder.

## Source-reported evidence

AISHELL-1, Table 3: CTC-CRF/WFST system at 400ms chunk plus approximately 2ms simulation reports 5.47% CER, or 4.79% with NNLM rescoring. Simulation timing uses a GTX 1080. Tables 1-2 test stochastic-context and chunk-jitter training.

Evidence location: PDF pp. 2-4, Figure 1, training objective, Tables 1-3 (p. 4).

## Connection to this project — interpretation

The closest required comparison for learned acoustic padding. Joint ASR adaptation is a material difference from our frozen-teacher state predictors and A2/A3 student replacement. A fair learned-fill study should separate joint training from a frozen plug-in.

## Limitations when applying the result

Mandarin CER and LM-assisted results are not English WER. Chunk plus simulation time is not our reference-word stable latency. Avoiding future-context waiting does not demonstrate omission of current-chunk encoder computation.

## Code and models

- [https://github.com/thu-spmi/CAT](https://github.com/thu-spmi/CAT)
- [https://github.com/thu-spmi/CAT/blob/master/docs/whatsnew.md](https://github.com/thu-spmi/CAT/blob/master/docs/whatsnew.md)

The official CAT changelog identifies a CUSIDE implementation. Exact recipe/checkpoint reproducibility was not exercised.

## Source version

PDF SHA-256: `1f79320f37a0e614d2f9b3932fd7846c4c9525ce42610ac842f327851e6463cd`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
