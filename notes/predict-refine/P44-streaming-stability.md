# P44. Analyzing the Quality and Stability of a Streaming End-to-End On-Device Speech Recognizer

| Field | Value |
|---|---|
| Authors | Shangguan, Yuan; Knister, Kate; He, Yanzhang; McGraw, Ian; Beaufays, Françoise |
| Source date | 2020-06-02 |
| Publication | Interspeech 2020 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P44_streaming-stability.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2020/shangguan20_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2020/shangguan20_interspeech.pdf)

## Main idea and method

Streaming transcript quality includes stability of intermediate results, beyond final WER.

A prefix is stable only if all subsequent hypotheses keep it. UPWR counts unstable word occurrences per final-hypothesis word; UPSR counts revised segments per utterance. Both can exceed one.

## Source-reported evidence

Table 1 illustrates repeated corrections: 11 unstable-word occurrences over 9 final words gives UPWR 1.22; five revised segments in one utterance gives UPSR 5. The study separates normalization and streaming instability.

Evidence location: PDF p. 2, Section 2 and Table 1; Sections 3-4.

## Connection to this project — interpretation

Use event-level display replay to catch repeat rewrites hidden by final edit distance. Report reference correctness separately from agreement with the recognizer final output.

## Limitations when applying the result

UPWR is not bounded probability and is not interchangeable with a forecast withdrawal fraction. Stability can hold for a consistently wrong transcript, so it does not replace reference-aligned accuracy.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review; this is not evidence that they do not exist.

## Source version

PDF SHA-256: `650597e8dd40cc1a43ad0622bf69347fef6ae4c572eaf133e04d1ffa4321f36d`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
