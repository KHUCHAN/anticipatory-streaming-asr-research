# P34. ZeroPrompt: Streaming Acoustic Encoders are Zero-Shot Masked LMs

| Field | Value |
|---|---|
| Authors | Song, Xingchen; Wu, Di; Zhang, Binbin; Peng, Zhendong; Dang, Bo; Pan, Fuping; Wu, Zhiyong |
| Source date | 2023-05-18 |
| Publication | Interspeech 2023 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 5 pages; `papers/pdf/P34_zeroprompt.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2023/song23c_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2023/song23c_interspeech.pdf) · [Additional primary source 1](https://arxiv.org/abs/2305.10649)

## Main idea and method

A training-free zero-feature prompt lets a chunked acoustic encoder produce provisional text before real future audio arrives.

Append zero feature frames; use chunk-causal masking and prompt-and-refine to protect the real-audio path. Intermediate ZeroPrompt starts the prompt at a hidden encoder layer to trade preview quality against cost.

## Source-reported evidence

Table 1, LibriSpeech test-clean, 640ms chunk and 0 to 640ms prompt: TDT-F 1136 to 662ms, TDT-L 7328 to 7098ms; greedy/rescored WER stays 4.41/3.80%. Greedy RTF increases 0.04826 to 0.07006. PER-A is 10.4%.

Evidence location: PDF pp. 2-4, Figures 1-3, Section 3.1, Table 1 (p. 3), Table 2 (p. 4).

## Connection to this project — interpretation

A required baseline for speculative display. Simply filling unavailable frames or moving preview into hidden layers is already covered in prior work. Porting the mechanism to Zipformer/RNN-T still needs cache isolation and native-output equivalence checks.

## Limitations when applying the result

TDT is utterance-relative first/last display time and excludes compute time. PER counts characters, not word revisions. Final-output preservation depends on the mask and refinement path; it is not automatic for arbitrary padding. Lower display latency does not imply less inference work.

## Code and models

- [https://github.com/wenet-e2e/wenet](https://github.com/wenet-e2e/wenet)

WeNet baseline toolkit/checkpoints are identified in the paper. A separate exact ZeroPrompt implementation was not confirmed; no code or checkpoint was run here.

## Source version

PDF SHA-256: `7b544f5446295a0e7fce70228aaf7988e8ec5144031b32f7b3d9e0bab0c0b8e9`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
