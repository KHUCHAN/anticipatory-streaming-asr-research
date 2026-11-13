# P13. Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition

| Field | Value |
|---|---|
| Authors | Rekesh, Dima; Koluguri, Nithin Rao; Kriman, Samuel; Majumdar, Somshubra; Noroozi, Vahid; Huang, He; Hrinchuk, Oleksii; Puvvada, Krishna; Kumar, Ankur; Balam, Jagadeesh; Ginsburg, Boris |
| Source date | 2023-05-08 |
| Publication | ASRU 2023 accepted; Check original text arXiv |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 2 — Efficiency baseline candidate |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 8 pages; `papers/pdf/P13_fastconformer.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2305.05084) · [PDF](https://arxiv.org/pdf/2305.05084v6)

## Main idea and method

Increase learning and inference efficiency by changing the subsampling and attention configuration of the conformer.

Uses aggressive temporal downsampling and limited-context attention.

## Source-reported evidence

Reports 2.8 times the speed of baseline Conformer and configurations for long audio.

Evidence location: architecture redesign and efficiency experiments.

## Connection to this project — interpretation

Use this as a reference when comparing whether the gains from speculation are better than simple encoder optimization.

## Limitations when applying the result

Equivalent acceleration is not guaranteed on different hardware or streaming chunks. Processing an 11-hour recording does not imply native streaming support of that duration.

## Code and models

- [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

Original NVIDIA/NeMo links now redirect to NVIDIA-NeMo/Speech. Per-model recipes, weights and Mac operation support were not execution-tested.

## Source version

PDF SHA-256: `eb8c7706a567ec5795333139f8cd41ff9c127ba99b6dc85d378eb3fa3213bf8d`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
