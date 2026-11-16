# P42. Minimum latency training of sequence transducers for streaming end-to-end speech recognition

| Field | Value |
|---|---|
| Authors | Shinohara, Yusuke; Watanabe, Shinji |
| Source date | 2022-09-18 |
| Publication | Interspeech 2022 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P42_minimum-latency.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2022/shinohara22_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2022/shinohara22_interspeech.pdf) · [Additional primary source 1](https://arxiv.org/abs/2211.02333)

## Main idea and method

Optimize expected alignment latency jointly with transducer recognition loss.

Dynamic programming over the transducer lattice estimates an expected delay objective using reference alignments. This differs from FastEmit, which does not need such external alignment.

## Source-reported evidence

WSJ eval92, causal Conformer, no LM, means of three independent seeds: baseline WER 12.1% and PR90 220ms; minimum-latency training 12.8% and 27ms; FastEmit comparator 12.7% and 67ms.

Evidence location: PDF Sections 2-3, Tables 1-2 and latency definition.

## Connection to this project — interpretation

Demonstrates a task-aware latency objective rather than a feature reconstruction objective. It provides a control for whether forecast gains survive direct training for earlier emission.

## Limitations when applying the result

WSJ is not LibriSpeech. The WER change is 0.7 percentage points, not 0.7% relative. Alignment availability and utterance-tail timing prevent direct ranking against our word display metric.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review; this is not evidence that they do not exist.

## Source version

PDF SHA-256: `fcebca0196b3be5539a3b8a52dd950375484cb9affbc90b3fc315fa86ffdb4bc`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
