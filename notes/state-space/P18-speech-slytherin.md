# P18. Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation, Recognition, and Synthesis

| Field | Value |
|---|---|
| Authors | Jiang, Xilin; Li, Yinghao Aaron; Florea, Adrian Nicolas; Han, Cong; Mesgarani, Nima |
| Source date | 2024-07-13 |
| Publication | arXiv preprint, 2024 |
| Category | State-space models and ConMamba |
| Reading priority | 1 — Grounds for applying state space |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 9 pages; `papers/pdf/P18_speech-slytherin.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2407.09732) · [PDF](https://arxiv.org/pdf/2407.09732v1)

## Main idea and method

This study proposes ConMamba by applying Mamba to sound source separation, voice recognition, and synthesis.

For ASR, evaluates replacing Conformer attention with Mamba-family modules.

## Source-reported evidence

They report that Mamba's efficiency advantage depends on input length and operation, and is not always advantageous for short speech.

Evidence location: ConMamba and speed/memory evaluation.

## Connection to this project — interpretation

Useful for assessing state reduction and the risk that gains disappear on short real-time chunks.

## Limitations when applying the result

Bidirectional configurations and full speech experiments are not treated as strict streaming results. The state processing of chunk boundaries and the learning budget must be verified separately.

## Code and models

- [https://github.com/xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `5cc3c3aeebac17170e5880843ca760aad202424aa4a232101086bb1add8eaa76`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
