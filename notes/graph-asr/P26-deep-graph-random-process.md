# P26. Deep Graph Random Process for Relational-Thinking-Based Speech Recognition

| Field | Value |
|---|---|
| Authors | Hengguan Huang; Fuzhao Xue; Hao Wang; Ye Wang |
| Source date | 2020-11-21 |
| Publication | ICML 2020, PMLR 119 |
| Category | Graph-based speech recognition |
| Reading priority | 3 — relational ASR background |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 11 pages; `papers/pdf/P26_deep-graph-random-process.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://proceedings.mlr.press/v119/huang20k.html) · [PDF](https://proceedings.mlr.press/v119/huang20k/huang20k.pdf)

## Main idea and method

Deep Graph Random Process models inter-utterance relationships as a latent probabilistic graph.

Several percept graphs are inferred using the Bayesian nonparametric method and reflected in acoustic modeling.

## Source-reported evidence

CHiME-2 and CHiME-5 evaluate the effectiveness of relationship modeling in ASR.

Evidence location: DGP formulation and CHiME experiments.

## Connection to this project — interpretation

This is the background for understanding the scope of graph-based voice recognition. Avoid confusing the candidate transcription tree with the utterance relationship graph.

## Limitations when applying the result

This is not ASR using Google's FGN or Factor Graph NN. Preliminary calculation of low-delay candidates is not the core purpose.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

https://github.com/GlenHGHUANG/Deep_graph_random_process linked by the original PMLR text was 2026-11-10 HTTP 404. We have not confirmed any official implementation currently available.

## Source version

PDF SHA-256: `15111002f2b1dc815af017f564e830b9fa62d1bd652c48471e11677289154828`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
