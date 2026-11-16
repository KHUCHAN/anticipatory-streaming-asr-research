# P59. StreamCorrect: Self-Supervised Error Correction for Real-Time ASR with Offline Recognizers

| Field | Value |
|---|---|
| Authors | Vu, Tung D.; Xie, Zeying; Zhao, Weiwei; Wei, Victor Junqiu; Jiang, Di; Fung, Yi R. |
| Source date | 2026-10-24 |
| Publication | EMNLP 2026 Industry Track |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 13 pages; `papers/pdf/P59_streamcorrect.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://aclanthology.org/2026.emnlp-industry.142/) · [PDF](https://aclanthology.org/2026.emnlp-industry.142.pdf)

## Main idea and method

A speech-language corrector refines chunk-level hypotheses and feeds the corrected committed prefix back into streaming recognition.

Fine-tune Ultravox-v0.5-Llama-3.2-1B with audio, top-k hypotheses and confirmed prefix. Prefix-to-Reference Alignment derives chunk targets from paired speech/transcripts without extra timestamp annotation. The underlying recognizer remains frozen.

## Source-reported evidence

Table 2, AISHELL-1 100ms chunks, SimulStreaming to StreamCorrect: CER 27.3 to 11.7%; first-token latency 158 to 207ms, last-token latency 200 to 284ms on NVIDIA H20. At 500ms CER 16.3 to 9.8%, FTL 158 to 219ms, LTL 213 to 295ms.

Evidence location: PDF pp. 3-5 method/setup; p. 6 Table 2; p. 7 deployment discussion and p. 8 limitations.

## Connection to this project — interpretation

A direct 2026 streaming correction prior, rather than an offline reranking result. It conditions on actual arrived audio and evolving prefix; relevant to preserving acoustic evidence when language priors might overcorrect.

## Limitations when applying the result

Mandarin/Cantonese CER is not English WER. Latencies are first/last utterance endpoints, not per-word p50/p90. Correction improves accuracy but adds latency. The named 1B corrector has about 1.92B total parameters per the deployment section. The abstract relative-gain headline does not summarize every Table 2 row.

## Code and models

- [https://github.com/vu-duy-tung/StreamCorrect](https://github.com/vu-duy-tung/StreamCorrect)

Official repository confirmed; its README uses a different subtitle from the published paper. Code and metrics were not reproduced.

## Source version

PDF SHA-256: `db5f0153ec0718a424a51d17c4a7c8c75c6e0ba33bb383a78f5f58c9008a7cae`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
