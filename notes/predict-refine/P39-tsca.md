# P39. Improving Streaming Speech Recognition With Time-Shifted Contextual Attention And Dynamic Right Context Masking

| Field | Value |
|---|---|
| Authors | Le, Khanh; Chau, Duc |
| Source date | 2024-09-01 |
| Publication | Interspeech 2024 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P39_tsca.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2024/le24_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2024/le24_interspeech.pdf) · [Additional primary source 1](https://arxiv.org/abs/2502.15158)

## Main idea and method

Time-Shifted Contextual Attention revisits trailing frames when the next real chunk arrives; dynamic right-context masking supports its training.

Move the contextual window so previous trailing frames can use already received current audio. The inference window spans c+r frames with stride c; this is overlapping refinement, not synthesizing unheard audio.

## Source-reported evidence

LibriSpeech 960h, CTC/attention system: Section 4.2 reports average test clean/other WER 4.74/12.23 to 4.44/11.62% for one paired group of configurations. The RTF denominator is c, not c+r.

Evidence location: PDF pp. 2-4, TSCA/DRC methods, Table 2 and Section 4.2.

## Connection to this project — interpretation

A control for the possibility that apparent future-context gains come from revising earlier representations with newly available evidence. Account for overlap recomputation in total inference cost.

## Limitations when applying the result

This uses real arrived audio and can revise prior results. Averages across configurations are not one deployable operating point. Its RTF, decoding and delay definitions differ from our Zipformer measurements.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review; this is not evidence that they do not exist.

## Source version

PDF SHA-256: `0025d24656ffee61af3cce09e51eda41fa924b504fe47d9366544b41e70dfda1`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
