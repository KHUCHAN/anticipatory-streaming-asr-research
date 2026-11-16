# P57. Delayed Fusion: Integrating Large Language Models into First-Pass Decoding in End-to-end Speech Recognition

| Field | Value |
|---|---|
| Authors | Hori, Takaaki; Kocour, Martin; Haider, Adnan; McDermott, Erik; Zhuang, Xiaodan |
| Source date | 2025-01-16 |
| Publication | ICASSP 2025 (acceptance stated on arXiv) |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 5 pages; `papers/pdf/P57_delayed-fusion.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2501.09258) · [PDF](https://arxiv.org/pdf/2501.09258v1)

## Main idea and method

Reduce language-model calls by applying LLM scores after pruning and only at selected decoding steps.

Retokenize stable word-ended prefixes to handle ASR/LLM vocabulary mismatch and cache previous LM scores. Compare shortest-hypothesis and fixed-interval scheduling with shallow fusion and second-pass rescoring.

## Source-reported evidence

Table I, LibriHeavy 50kh, CTC-prefix mode: no LM clean/other WER 3.47/6.41%, RTF 0.018; OpenLLaMA-3B delayed fusion 3.05/5.68%, RTF 0.115. RTF uses the first 200 lh-other utterances on Xeon+V100. Table II exposes an accuracy-cost frontier.

Evidence location: PDF pp. 2-3 Algorithm 1 and retokenization; p. 3 Section IV-A/Table I; p. 4 Tables II-III.

## Connection to this project — interpretation

Direct computation-scheduling prior for efficient online candidate scoring. Fewer LLM calls can help relative to other LLM integrations without making the system faster than plain ASR.

## Limitations when applying the result

The algorithm is streamable, but experiments explicitly use full-utterance encoder context. They do not establish live causal word-delay improvement. LibriHeavy is not LibriSpeech 960h; the 2.2x comparison also changes the ASR tokenizer/model, so do not import it as a matched-backbone speedup.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review. This does not establish their absence.

## Source version

PDF SHA-256: `6b30ee77895f3de66ea42fe710e31c1611f2d29aa4b25b883909bd85b42af2a3`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
