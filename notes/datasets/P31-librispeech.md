# P31. LibriSpeech: An ASR Corpus Based on Public Domain Audio Books

| Field | Value |
|---|---|
| Authors | Vassil Panayotov; Guoguo Chen; Daniel Povey; Sanjeev Khudanpur |
| Source date | 2015 |
| Publication | ICASSP 2015 |
| Category | Original dataset papers |
| Reading priority | 1 — Understand basic data |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 5 pages; `papers/pdf/P31_librispeech.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://www.openslr.org/resources/12/about.html) · [PDF](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf) · [Additional primary source 1](https://www.openslr.org/12)

## Main idea and method

This is the original paper of the 16 kHz English reading ASR corpus collected from LibriVox public audiobooks.

Provides cleaned transcripts, segmented utterances, train/dev/test and separately distributed language-model resources.

## Source-reported evidence

Presents an approximately 1,000-hour corpus and baseline ASR. Initial preparation in this project covers train-clean-100 and four dev/test splits.

Evidence location: corpus construction and official OpenSLR split list.

## Connection to this project — interpretation

Provides a reproducible small-scale comparison and multiple temporal prefixes from identical utterances.

## Limitations when applying the result

Read, cleaned and segmented speech differs from conversation or continuous microphones. Word-end timing needs separate alignment.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `0e6ee6ace1a55631b54e8671c25203418f8af00dcec3273ca06b1945b7a8a953`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
