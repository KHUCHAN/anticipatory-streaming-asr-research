# P14. Robust Speech Recognition via Large-Scale Weak Supervision

| Field | Value |
|---|---|
| Authors | Radford, Alec; Kim, Jong Wook; Xu, Tao; Brockman, Greg; McLeavey, Christine; Sutskever, Ilya |
| Source date | 2022-12-06 |
| Publication | Whisper original paper, arXiv 2022 |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 2 — External accuracy reference |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 28 pages; `papers/pdf/P14_whisper.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2212.04356) · [PDF](https://arxiv.org/pdf/2212.04356v1)

## Main idea and method

Whisper uses large-scale weakly supervised audio/transcript training across languages and domains.

It learns with an audio encoder and an autoregressive text decoder and uses multilingual and multi-task supervision.

## Source-reported evidence

We report 680,000time-scale learning and various zero-shot recognition evaluations.

Evidence location: training data and evaluation sections.

## Connection to this project — interpretation

Defines the boundary between training all models from scratch and comparing externally pretrained large models.

## Limitations when applying the result

The base architecture is not exclusively real-time streaming. External pretrained Whisper and a locally trained 100-hour model are not matched-training WER comparisons.

## Code and models

- [https://github.com/openai/whisper](https://github.com/openai/whisper)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `6337bde031b2f237547a977b022f831169a7e05b4d9047f29501166d83594566`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
