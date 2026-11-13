# P12. Conformer: Convolution-augmented Transformer for Speech Recognition

| Field | Value |
|---|---|
| Authors | Gulati, Anmol; Qin, James; Chiu, Chung-Cheng; Parmar, Niki; Zhang, Yu; Yu, Jiahui; Han, Wei; Wang, Shibo; Zhang, Zhengdong; Wu, Yonghui; Pang, Ruoming |
| Source date | 2020-05-16 |
| Publication | Conformer original paper, arXiv 2020 |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 2 — Encoder comparison |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 5 pages; `papers/pdf/P12_conformer.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2005.08100) · [PDF](https://arxiv.org/pdf/2005.08100v1)

## Main idea and method

By combining convolution and self-attention, local and global patterns of speech are modeled.

Combine feed-forward, attention, and convolution in the conformer block.

## Source-reported evidence

LibriSpeech reports high recognition performance compared to existing encoders.

Evidence location: Conformer block and LibriSpeech results.

## Connection to this project — interpretation

Provides acoustic-encoder controls for isolating candidate-generation effects.

## Limitations when applying the result

The original architecture includes full-context settings. Adding an attention causal mask does not automatically make convolution/subsampling causal.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `4d53a2d87f7b68b3b5a86d8e4301c5092ae61e260f9c76dab48af34faac9a81f`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
