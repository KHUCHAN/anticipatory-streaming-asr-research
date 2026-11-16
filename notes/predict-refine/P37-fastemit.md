# P37. FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization

| Field | Value |
|---|---|
| Authors | Yu, Jiahui; Chiu, Chung-Cheng; Li, Bo; Chang, Shuo-yiin; Sainath, Tara N.; He, Yanzhang; Narayanan, Arun; Han, Wei; Gulati, Anmol; Wu, Yonghui; Pang, Ruoming |
| Source date | 2020-10-21 |
| Publication | ICASSP 2021 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P37_fastemit.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2010.11148) · [PDF](https://arxiv.org/pdf/2010.11148v2)

## Main idea and method

Alignment-free transducer training encourages earlier nonblank emissions.

Sequence-level regularization scales label-transition gradients by 1+lambda while leaving blank-transition gradients unchanged. No future-feature generation or inference-time forecast module is required.

## Source-reported evidence

Table 1, LibriSpeech ContextNet-L: test clean/other WER 4.4/8.9 to 3.1/7.5%, PR90 210 to 30ms. Table 2 shows that stronger regularization can eventually worsen WER.

Evidence location: PDF p. 3, gradient equations and Section 3.2 latency definition; p. 4, Tables 1-3.

## Connection to this project — interpretation

A necessary emission-timing control before attributing gains to speculative prediction. Recalibrate our gate and boundary-gain analysis on an early-emission backbone. Frame-weighted KL in A3 is not FastEmit.

## Limitations when applying the result

PR latency anchors the finalized last token to the utterance speech end; it is not average word latency or first-display time. Results depend on architecture, objective weight and data; exact WER preservation is not guaranteed.

## Code and models

- [https://github.com/tensorflow/lingvo](https://github.com/tensorflow/lingvo)

Lingvo is the implementation toolkit named in the paper. An exact runnable FastEmit recipe/checkpoint was not confirmed here.

## Source version

PDF SHA-256: `f74ae11b2459e6c019dbf143b743a351296fc704f6d2704121f05b58ad406db3`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
