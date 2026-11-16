# P45. Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding

| Field | Value |
|---|---|
| Authors | Zhou, Haoran; Song, Xingchen; Fahy, Brendan; Song, Qiaochu; Zhang, Binbin; Peng, Zhendong; Wadhawan, Anshul; Jiang, Denglin; Verma, Apurv; Ramesh, Vinay; Prasad, Srivas; Franceschini, Michele M. |
| Source date | 2025-06-13 |
| Publication | Interspeech 2025 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P45_whisper-two-pass.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2025/zhou25_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2025/zhou25_interspeech.pdf) · [Additional primary source 1](https://arxiv.org/abs/2506.12154)

## Main idea and method

Adapt pretrained Whisper to causal CTC partial decoding and attention-decoder rescoring using a U2 structure.

Fine-tune a causal encoder and additional CTC head while preserving the attention decoder for final rescoring. A hybrid tokenizer assigns a smaller CTC vocabulary. This is two-pass ASR, not prediction of unheard acoustic frames.

## Source-reported evidence

Earnings-call evaluation, 5800h adaptation: Table 2 gives 17.60/17.30% WER without/with rescoring at 1000ms chunks. Table 3, 12s maximum delay: RTF 0.30 and mean finalize-operation latency 1126ms on four Xeon virtual cores.

Evidence location: PDF Sections 2-4, Tables 1-3 (p. 3).

## Connection to this project — interpretation

Shows that two-pass adaptation of a strong pretrained recognizer and vocabulary choice are existing alternatives. Compare our display method against native streaming and final-rescoring behavior explicitly.

## Limitations when applying the result

Whisper pretraining, earnings-call adaptation and 1s chunking differ from our 960h Zipformer reference. Finalize-operation latency is rescoring compute, not per-word stable display delay.

## Code and models

- [https://github.com/wenet-e2e/wenet](https://github.com/wenet-e2e/wenet)

WeNet is the paper toolkit. A complete exact recipe/checkpoint reproduction was not performed.

## Source version

PDF SHA-256: `fa37a188f26bdd6447807e079492683592534a9dd3cae996e1a866f3061043c0`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
