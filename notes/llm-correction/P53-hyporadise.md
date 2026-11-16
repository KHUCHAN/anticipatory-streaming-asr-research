# P53. HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models

| Field | Value |
|---|---|
| Authors | Chen, Chen; Hu, Yuchen; Yang, Chao-Han Huck; Siniscalchi, Sabato Marco; Chen, Pin-Yu; Chng, Eng Siong |
| Source date | 2023-09-27 |
| Publication | NeurIPS 2023 Datasets and Benchmarks |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 2 - generation or fusion control |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 24 pages; `papers/pdf/P53_hyporadise.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html) · [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/6492267465a7ac507be1f9fd1174e78d-Paper-Datasets_and_Benchmarks.pdf) · [Additional primary source 1](https://arxiv.org/abs/2309.15701)

## Main idea and method

An open N-best-to-reference benchmark for generative ASR correction across speech domains.

Compare fine-tuned text models, LoRA adaptation and prompted in-context correction. Separate the best whole-hypothesis oracle from a compositional oracle; generation can leave the original candidate set.

## Source-reported evidence

Table 2 reports cross-domain WER benefits. Appendix Table 6, LibriSpeech: baseline clean/other 1.8/3.7%; H2T-LoRA 1.7/3.8%, H2T-ft 1.8/3.9%. Larger language models do not yield a universal gain on this strong baseline.

Evidence location: Publisher PDF pp. 4-8, benchmark/methods and Tables 1-3; p. 20 Appendix Table 6 and failure-case discussion.

## Connection to this project — interpretation

Useful public correction data and controls. Direct evidence against assuming that a successful correction recipe improves every strong ASR model or split.

## Limitations when applying the result

Offline hypothesis correction does not demonstrate streaming latency improvement. The publisher landing abstract says more than 316k pairs while the collected PDF says more than 334k; do not merge those counts without version checking. Oracles are not deployable selectors.

## Code and models

- [https://github.com/Hypotheses-Paradise/Hypo2Trans](https://github.com/Hypotheses-Paradise/Hypo2Trans)

Official benchmark/code repository confirmed; its data, models and evaluation were not downloaded or run. The collected publisher PDF is the evidence version.

## Source version

PDF SHA-256: `5259464e7e887f35a44dcdde1d773c536f9146f8fc26ebbb0aefdc630d2c4ab4`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
