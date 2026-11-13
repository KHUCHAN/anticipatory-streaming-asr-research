# P02. Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models

| Field | Value |
|---|---|
| Authors | Bian, Yutong; Zhang, Yanglin; Wang, Yuxiang; Zhang, Luoyuan; He, Rongshen; Wang, Zihao; Qiu, Junwen; Li, Jiaqi; Wu, Zhizheng |
| Source date | 2026-11-04 |
| Publication | arXiv preprint, 2026-11-04 |
| Category | Prediction before audio arrival |
| Reading priority | 1 — read first |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 13 pages; `papers/pdf/P02_speculative-listening.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2611.06982) · [PDF](https://arxiv.org/pdf/2611.06982v1)

## Main idea and method

Full-duplex speech-language-model research drafts during idle time before the next audio chunk and verifies after arrival.

It uses silence hypothesis, next-order branch of the first token, and hypothesis pool between chunks. Only the prefix confirmed by the target model under the new voice is adopted.

## Source-reported evidence

Four models of TickTock-Bench report acceleration of speak chunks by 2.6–2.8 times and reduction of decoding delay before response starts. This text uses 1 second chunks and a low-concurrency environment.

Evidence location: PDF 3–7 Page, Sections 3–5, Tables 1–2.

## Connection to this project — interpretation

Directly relevant to scheduling precomputation during audio waiting and reusing/correcting wrong branches.

## Limitations when applying the result

Not an experiment in predicting the next words of pure ASR. High output agreement under a silence hypothesis is not future-word accuracy. Lossless refers to preserving target output.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `31dbfc0ef28ccd92add078b8e04e230660e6a5ddb4330b3ef2b64a3c1d767f5c`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
