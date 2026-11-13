# P25. Tree-constrained Pointer Generator with Graph Neural Network Encodings for Contextual Speech Recognition

| Field | Value |
|---|---|
| Authors | Sun, Guangzhi; Zhang, Chao; Woodland, Phil |
| Source date | 2022 |
| Publication | Interspeech 2022 |
| Category | Graph-based speech recognition |
| Reading priority | 2 — contextual graph prototype |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 5 pages; `papers/pdf/P25_tcpgen.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://www.isca-archive.org/interspeech_2022/sun22_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2022/sun22_interspeech.pdf)

## Main idea and method

This is a contextual ASR study that combines GNN encoding with a tree-constrained pointer generator.

The branch information of the biasing word prefix tree is reflected in the node representation.

## Source-reported evidence

We evaluate LibriSpeech's biasing task and context word recognition in AMI.

Evidence location: method and contextual biasing experiments.

## Connection to this project — interpretation

Prior work on structural representation of candidate trees; read alongside its successor P24.

## Limitations when applying the result

Knowing a contextual word list is not equivalent to knowing the correct future utterance.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `0aa25d8719f22aeb91c726dbf9bc5d8dafd0feec69dcd5b46ad114cbb68215ec`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
