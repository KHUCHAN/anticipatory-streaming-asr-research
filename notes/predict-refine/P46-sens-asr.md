# P46. SENS-ASR: Semantic Embedding injection in Neural-transducer for Streaming Automatic Speech Recognition

| Field | Value |
|---|---|
| Authors | Dkhissi, Youness; Vielzeuf, Valentin; Allesiardo, Elys; Larcher, Anthony |
| Source date | 2026-02-17 |
| Publication | LREC 2026 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 9 pages; `papers/pdf/P46_sens-asr.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://aclanthology.org/2026.lrec-1.803/) · [PDF](https://arxiv.org/pdf/2603.10005v2) · [Additional primary source 1](https://arxiv.org/abs/2603.10005)

## Main idea and method

Distill semantic sentence information into a streaming transducer context module using arrived acoustic representations.

Attention pooling produces a context vector concatenated with encoder frames before the joiner. Train with RNN-T plus MSE to a fine-tuned sentence-embedding teacher; the recipe also uses FastEmit.

## Source-reported evidence

LibriSpeech 960h, greedy/no external LM, Table 1: at 160ms chunks clean/other WER 7.55/18.34 to 7.21/17.89%; at 1280ms other worsens 8.39 to 8.55%. TEDLIUM-2 is also evaluated.

Evidence location: Collected arXiv v2 pp. 2-5, Eq. (3), Sections 3.1-3.3 and Table 1; LREC publisher metadata separately checked.

## Connection to this project — interpretation

Relevant to our proposed language-informed prior, including on a FastEmit backbone. It is semantic augmentation, not an encoder-skipping method or a next-token preview generator.

## Limitations when applying the result

The small-chunk gains do not establish lower total runtime or irreversible commit latency. The frozen sentence teacher, text paraphrase data and new context module must be included in training/resource descriptions.

## Code and models

- [https://github.com/Orange-OpenSource/sens-asr](https://github.com/Orange-OpenSource/sens-asr)

Official public training/evaluation repository confirmed. No weights were downloaded and no author result was run.

## Source version

PDF SHA-256: `8893da89deee97cbd8d96682df4d724030af972bf8ebf159a7c89c0af8917165`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
