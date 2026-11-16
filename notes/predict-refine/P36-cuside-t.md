# P36. CUSIDE-T: Chunking, Simulating Future and Decoding for Transducer based Streaming ASR

| Field | Value |
|---|---|
| Authors | Zhao, Wenbo; Li, Ziwei; Yu, Chuan; Ou, Zhijian |
| Source date | 2024-07-14 |
| Publication | ISCSLP 2024 (acceptance stated on arXiv) |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P36_cuside-t.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2407.10255) · [PDF](https://arxiv.org/pdf/2407.10255v2)

## Main idea and method

CUSIDE future-feature simulation is extended to transducer ASR.

A Conformer RNN-T and GRU simulator are jointly trained using streaming and non-streaming transducer losses plus L1 simulation. The simulator adds about 5% parameters; it supplies right context rather than replacing the current encoder.

## Source-reported evidence

AISHELL-1, 400ms chunk: adapted U2++ versus CUSIDE-T beam CER is 6.11 versus 6.02%; with LM rescoring, 5.74 versus 5.51%. CUSIDE-T labels its latency 400+2ms, with another roughly 50-100ms for rescoring. WenetSpeech 10k-hour tests are also reported.

Evidence location: PDF pp. 2-4, Figure 1, joint objective, Tables 1-2 (p. 4).

## Connection to this project — interpretation

More directly relevant than CTC-only future simulation to our Zipformer transducer. Include it before claiming that learned future acoustic context in RNN-T is new.

## Limitations when applying the result

The paper adapts the comparator to the same RNN-T structure. Chinese CER, beam search, LM rescoring and simulation hardware must stay attached to the numbers. A 2ms simulator is not a measured Mac end-to-end speedup.

## Code and models

- [https://github.com/thu-spmi/CAT](https://github.com/thu-spmi/CAT)

The paper points to CAT for code, scripts and checkpoints. Their specific release was not downloaded or reproduced in this review.

## Source version

PDF SHA-256: `8c31661acf56626b71c04f8ab4ec34f732b981c8cd1ac1673f0c9cee39c2268c`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
