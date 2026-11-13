# P19. Assessing the Performance and Efficiency of Mamba ASR in Low-Resource Scenarios

| Field | Value |
|---|---|
| Authors | Zevallos, Rodolfo; Cortada Garcia, Martí; Solito, Sarah; Mena, Carlos; Peiró-Lilja, Alex; Hernando, Javier |
| Source date | 2025 |
| Publication | Interspeech 2025 |
| Category | State-space models and ConMamba |
| Reading priority | 2 — Source of reported figures |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 5 pages; `papers/pdf/P19_mamba-low-resource.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://www.isca-archive.org/interspeech_2025/zevallos25_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2025/zevallos25_interspeech.pdf)

## Main idea and method

We compare the performance and efficiency of Conformer and ConMamba under low-resource conditions in 9 languages.

Compares both architectures across training-data sizes and input-context lengths.

## Source-reported evidence

Reports 40–45% less training time, 50% less memory and 63–70% inference acceleration under its stated conditions.

Evidence location: PDF 1 page abstract, main text experiment settings and results.

## Connection to this project — interpretation

Identifies the correct paper/conditions behind the supplied ConMamba 63–70% claim.

## Limitations when applying the result

This is not a universal ConMamba speed. Remeasure for different languages, training budgets, utterance lengths and hardware.

## Code and models

- [https://github.com/xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)

ConMamba implementation referenced by the paper; this does not establish that its authors released newly trained weights.

## Source version

PDF SHA-256: `7e075d3fb67015a3cdc6c803275194591f2356fa80d276aa5b7f902f348f2b3a`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
