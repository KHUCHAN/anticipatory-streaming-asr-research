# P52. Can Generative Large Language Models Perform ASR Error Correction?

| Field | Value |
|---|---|
| Authors | Ma, Rao; Qian, Mengjie; Manakul, Potsawee; Gales, Mark; Knill, Kate |
| Source date | 2023-07-09 |
| Publication | arXiv preprint; collected v2 |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 8 pages; `papers/pdf/P52_chatgpt-asr-correction.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2307.04172) · [PDF](https://arxiv.org/pdf/2307.04172v2) · [Additional primary source 1](https://ieeexplore.ieee.org/document/10930744/)

## Main idea and method

Prompt ChatGPT for ASR correction, then compare unrestricted output against constrained candidate selection.

Use zero/one-shot prompts on ASR N-best lists; either request a candidate directly or map a generated result to the closest original hypothesis by edit distance. The experimental endpoint is normalized final WER.

## Source-reported evidence

Table 2, LibriSpeech test-other Conformer-Transducer: WER 6.90 to 6.24% for one-shot closest mapping; Whisper small.en 7.37 to 7.03%. On TED, unrestricted zero-shot correction of Whisper worsens 3.89 to 5.84%. The historical model is gpt-3.5-turbo-0613.

Evidence location: PDF Section 3 methods; Section 4.1 setup; Table 2 and error-analysis Tables 3-6.

## Connection to this project — interpretation

Direct prior for our reported LLM candidate-selection direction. Provides both a constrained control and evidence that plausible language correction can damage a strong recognizer.

## Limitations when applying the result

The reported results are utterance-level and use an old proprietary API snapshot; no measured causal display-delay or full-pipeline RTF result was found. No local dev-other improvement or significance claim is verified by this paper.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review. This does not establish their absence.

## Source version

PDF SHA-256: `ebdc75dda7f1b1bcd36e7ad1e7071c74fcc0b8dfd292d677dfaa363723f7e280`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
