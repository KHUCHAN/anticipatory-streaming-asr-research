# P49. Fast-U2++: Fast and Accurate End-to-End Speech Recognition in Joint CTC/Attention Frames

| Field | Value |
|---|---|
| Authors | Liang, Chengdong; Zhang, Xiao-Lei; Zhang, Binbin; Wu, Di; Li, Shengqiang; Song, Xingchen; Peng, Zhendong; Pan, Fuping |
| Source date | 2022-11-02 |
| Publication | arXiv preprint; collected v1 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P49_fast-u2pp.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2211.00941) · [PDF](https://arxiv.org/pdf/2211.00941v1)

## Main idea and method

Use lower encoder layers with small chunks for partial output and upper layers with larger chunks for higher-quality final recognition.

Joint streaming/non-streaming ASR training and a second distillation stage train the lower path. The upper path and attention rescoring remain in the final pipeline; this is not a future-frame generator.

## Source-reported evidence

AISHELL-1 Table 1, lower/top chunks 4/24 and D2 beta 0.01: lower greedy CER 7.34%, final rescoring 5.06%, FTD50 170ms and LTD50 70ms. The headline 80ms is half the 160ms lower chunk, not final end-to-end latency.

Evidence location: PDF pp. 2-3, Figure 1, Eqs. (1)-(6); p. 4, Tables 1-3.

## Connection to this project — interpretation

A direct architecture control for a cheap provisional path followed by strong refinement. Compare against sharing lower-layer work before paying for two independent encoders.

## Limitations when applying the result

The good final CER cannot be assigned to the cheap lower path. Cost must include upper-layer processing and rescoring. The source quotes an additional rescoring delay; headline model latency and word-aligned emission latency differ.

## Code and models

- [https://github.com/wenet-e2e/wenet](https://github.com/wenet-e2e/wenet)

WeNet is the toolkit linked in the paper; the exact Fast-U2++ checkpoint/recipe was not separately confirmed or run.

## Source version

PDF SHA-256: `64d9c7d17cd09481cd42605f22db77d9a4d97f18b1b9357991c73003c2c154e3`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
