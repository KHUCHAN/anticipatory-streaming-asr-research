# P43. Partial Rewriting for Multi-Stage ASR

| Field | Value |
|---|---|
| Authors | Bruguier, Antoine; Qiu, David; He, Yanzhang |
| Source date | 2023-12-08 |
| Publication | arXiv preprint; venue not independently confirmed in this review |
| Category | Future context, provisional output and emission latency |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 15 pages; `papers/pdf/P43_partial-rewriting.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2312.09463) · [PDF](https://arxiv.org/pdf/2312.09463v1)

## Main idea and method

Merge fast causal partial transcripts with slower, higher-quality partials while keeping the final recognizer output.

Levenshtein alignment, trimming and hysteresis combine the two streams. Text merging itself is lightweight and needs no acoustic retraining; both acoustic paths still have a cost.

## Source-reported evidence

The paper evaluates partial word error, emission latency and flicker across multiple test sets. Table 1 shows that quality and revision behavior must be assessed together; Sections 2.2-2.3 define the metrics and merge rules.

Evidence location: Collected 15-page v1, Sections 2-4, Algorithm 1 and Table 1.

## Connection to this project — interpretation

A direct prior for our fast-display/slow-final two-pass path and an alternative to simplistic screen merging. A final WER unchanged by construction is not a complete display-quality result.

## Limitations when applying the result

Merger overhead is not total two-model inference overhead. Private production data and architectures limit direct numeric comparison. Exact-match final output does not ensure reference-correct intermediate words or a low rewrite count.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review; this is not evidence that they do not exist.

## Source version

PDF SHA-256: `342da32f240cbf43b3524d1086126047fe292c11db62c343da640acc3a82415f`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
