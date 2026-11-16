# P48. Delayed-KD: Delayed Knowledge Distillation based CTC for Low-Latency Streaming ASR

| Field | Value |
|---|---|
| Authors | Li, Longhao; Li, Yangze; Xue, Hongfei; Liu, Jie; Fang, Shuai; Wang, Kai; Xie, Lei |
| Source date | 2025-05-28 |
| Publication | Interspeech 2025 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P48_delayed-kd.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2025/li25u_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2025/li25u_interspeech.pdf) · [Additional primary source 1](https://arxiv.org/abs/2505.22069)

## Main idea and method

Align a streaming CTC student with a non-streaming teacher over an allowable temporal-delay range.

A Temporal Alignment Buffer selects the delay with minimum KL divergence, then combines distillation with CTC/attention ASR losses. Inference does not need the offline teacher.

## Source-reported evidence

AISHELL-1, 40ms inference chunks, Table 1: TAB 0 gives streaming/rescored CER 7.02/5.77%; TAB 80ms gives 6.11/5.42%, FTD50 140ms and LTD50 90ms. Table 2 shows degradation when distillation weight exceeds the selected optimum.

Evidence location: PDF p. 2, Eqs. (1)-(4); p. 3, Section 3.3; p. 4, Tables 1-3.

## Connection to this project — interpretation

A useful diagnostic reference for A2/A3: raw same-index KL can ignore emission misalignment. This motivates a separate alignment audit, not a retrospective claim that misalignment caused our A3 failure.

## Limitations when applying the result

The headline 5.42% is utterance-level rescoring CER, not first-pass streaming CER. A 40ms chunk is not total measured delay. CTC temporal buffering cannot be assumed to transfer unchanged to a frozen RNN-T decoder.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review; this is not evidence that they do not exist.

## Source version

PDF SHA-256: `0b27d8a7376ddd9d0a3040dd5fb707e694879bb1d9f69b831dd75b5c2b1658ce`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
