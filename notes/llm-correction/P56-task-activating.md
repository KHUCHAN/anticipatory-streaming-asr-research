# P56. Generative Speech Recognition Error Correction with Large Language Models and Task-Activating Prompting

| Field | Value |
|---|---|
| Authors | Yang, Chao-Han Huck; Gu, Yile; Liu, Yi-Chieh; Ghosh, Shalini; Bulyko, Ivan; Stolcke, Andreas |
| Source date | 2023-09-27 |
| Publication | IEEE ASRU 2023 |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 2 - generation or fusion control |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 8 pages; `papers/pdf/P56_task-activating.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2309.15649) · [PDF](https://arxiv.org/pdf/2309.15649v2) · [Additional primary source 1](https://ieeexplore.ieee.org/document/10389673/)

## Main idea and method

Task-activating prompts and demonstrations adapt an LLM to ASR rescoring/error correction.

Compare prompted correction followed by a trained rescorer against direct LLM-based second-pass output; also study LoRA/residual adapters and multi-turn task setup. The first-pass Conformer-transducer uses substantial multilingual/public speech pretraining.

## Source-reported evidence

Table 3, frozen InstructGPT with TAP: WSJ WER 11.87 to 8.72%, ATIS 8.82 to 6.39%. Tables 2 and 4 separate prompt variants and fine-tuning; those are distinct conditions.

Evidence location: PDF Sections 3-4; p. 6 Tables 2-4 and Figure 5 discussion.

## Connection to this project — interpretation

Prior for prompt-based candidate reasoning, in-context examples and generative correction. Prompt sophistication alone is not a new ASR contribution.

## Limitations when applying the result

This is second-pass utterance evaluation with historical proprietary models, not streaming display measurement. Generation/correction can leave the original list, so beating its oracle does not show a perfect constrained selector. No measured streaming word-delay/RTF result was found.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review. This does not establish their absence.

## Source version

PDF SHA-256: `19f01fa8b4b3f4da445e1f90ba278ea36d99d9fd2c30fa05ccff2226f4e7e4d5`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
