# P50. Selective Lookahead for Attention-Based Streaming ASR

| Field | Value |
|---|---|
| Authors | Jia, Yichen; Tamm, Bastiaan; Van hamme, Hugo |
| Source date | 2026-09-29 |
| Publication | IEEE SLT 2026 (acceptance stated on arXiv) |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 7 pages; `papers/pdf/P50_selective-lookahead.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2609.37611) · [PDF](https://arxiv.org/pdf/2609.37611v1)

## Main idea and method

A bounded-lookahead encoder and token-level wait policy spend future-audio delay selectively.

Age-selected encoder states bound future context across layers. A learned wait trigger or calibrated confidence trigger defers uncertain tokens; stable-prefix commitment and a wait budget control rollback and tail delay.

## Source-reported evidence

LibriSpeech train-clean-360, full test-clean, Table V: static R=1 gives WER 6.7%, median/p90 860/1298ms; dynamic learned trigger plus stable prefix gives 6.5%, 306/2800ms. Budget B=1 gives 6.8%, 328/964ms. Table III subset RTF rises 0.40 to 0.60.

Evidence location: Collected 7-page v1, Sections III-V, Table III (timing subset), Table V (full test-clean), Table VI.

## Connection to this project — interpretation

Direct recent prior for confidence-driven adaptive waiting. This constrains novelty of an adaptive gate or chunk policy even without synthesizing future audio.

## Limitations when applying the result

Per-word delay excludes computation and incorrectly recognized words. The unbudgeted median gain worsens p90; RTF uses a 95-utterance GPU subset. Training uses 360h, so this is not a matched 960h Zipformer ranking.

## Code and models

- [https://github.com/windskylionheart1023/selective-lookahead-asr](https://github.com/windskylionheart1023/selective-lookahead-asr)

Official repository linked by the arXiv authors confirmed. Code was not executed; the collected PDF has 7 pages although the arXiv comment says 5.

## Source version

PDF SHA-256: `de3daf8e656a9626c301e0837f238eeab6a56a2249b5d73cdf5607397dde8179`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
