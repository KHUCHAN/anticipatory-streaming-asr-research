# P06. Accelerating Large Language Model Decoding with Speculative Sampling

| Field | Value |
|---|---|
| Authors | Chen, Charlie; Borgeaud, Sebastian; Irving, Geoffrey; Lespiau, Jean-Baptiste; Sifre, Laurent; Jumper, John |
| Source date | 2023-02-02 |
| Publication | arXiv technical report, 2023 |
| Category | Speculative decoding and path reuse |
| Reading priority | 2 — Check the theory |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 11 pages; `papers/pdf/P06_speculative-sampling.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2302.01318) · [PDF](https://arxiv.org/pdf/2302.01318v1)

## Main idea and method

Scores a short draft in one pass and uses modified rejection sampling to preserve the target language-model distribution.

The target evaluates each draft location in parallel and corrects the rejection location with the residual distribution.

## Source-reported evidence

70B Reports 2–2.5 times decoding acceleration in Chinchilla's distributed environment.

Evidence location: sampling algorithm and Chinchilla experiments.

## Connection to this project — interpretation

High branch probability alone does not justify commitment; define a separate verification rule.

## Limitations when applying the result

Results concern a language model with fixed input. Streaming acoustic alignment and newly arriving input are separate issues.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `ffa03c6ae46f3122570bacd7da358cae8659b6421162bbc25088622fd4889c37`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
