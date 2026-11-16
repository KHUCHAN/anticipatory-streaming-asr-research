# P55. ProGRes: Prompted Generative Rescoring on ASR n-Best

| Field | Value |
|---|---|
| Authors | Tur, Ada Defne; Moumen, Adel; Ravanelli, Mirco |
| Source date | 2024-08-30 |
| Publication | IEEE SLT 2024 |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 8 pages; `papers/pdf/P55_progres.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2409.00217) · [PDF](https://arxiv.org/pdf/2409.00217v2) · [Additional primary source 1](https://ieeexplore.ieee.org/document/10832194/)

## Main idea and method

Generate an additional candidate, then rescore the extended list with interpolated ASR and LLM scores.

Prompt an instruction model, append its transcription, and use Llama-3 sequence scores plus an ASR score. The reported implementation gives the newly generated candidate the best original ASR candidate score.

## Source-reported evidence

Table 1, GPT-4 expansion: CommonVoice ASR1/ASR2 WER 40.84/9.32%; Section 5.1 baselines are 42.94/12.38%. The prose reports 9.82 for ASR2, conflicting with the table. Original-list ProGRes gives 42.84/11.42% without generated additions.

Evidence location: Collected v2 PDF pp. 2-3 methods; p. 4 Table 1 and Sections 5.1-5.2.

## Connection to this project — interpretation

The closest control if our method expands candidates and reselects them using linguistic scores. Compare original-list reranking separately from generation plus reranking.

## Limitations when applying the result

ASR1 and ASR2 use different CommonVoice versions and recognition systems. Borrowing the best ASR score is not an independently computed acoustic likelihood for a generated string. No measured streaming stable-delay or total-RTF result was found; table/prose discrepancy remains unresolved.

## Code and models

- [https://github.com/AdaDTur/ProGRes](https://github.com/AdaDTur/ProGRes)

Official code/data repository and IEEE publication metadata confirmed. Code, API inference and author metrics were not reproduced.

## Source version

PDF SHA-256: `06aedec89d099f523299e9022e429b1897ac476f09465b3ca097819edc4aea2f`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
