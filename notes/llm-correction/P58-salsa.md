# P58. SALSA: Speedy ASR-LLM Synchronous Aggregation

| Field | Value |
|---|---|
| Authors | Mittal, Ashish; Prabhu, Darshan; Sarawagi, Sunita; Jyothi, Preethi |
| Source date | 2024-09-01 |
| Publication | Interspeech 2024 |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 2 - generation or fusion control |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 5 pages; `papers/pdf/P58_salsa.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2024/mittal24_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2024/mittal24_interspeech.pdf)

## Main idea and method

Couple ASR decoder states into an LLM while advancing both decoders synchronously.

Train small state-projection adapters and use cascading tokenization to handle vocabulary mismatch. Whisper encodes the utterance; LLaMA generates the final text without a separately generated N-best correction pass.

## Source-reported evidence

Section 5 runtime: A100-80GB RTF Whisper 0.42, SALSA 0.64, Whispering-LLaMA 1.30. Table 1, eight selected FLEURS languages: LoRA Whisper average WER 39.4%, SALSA-7B on the fine-tuned recognizer 33.0%.

Evidence location: PDF pp. 2-3 coupling/decoding; p. 4 Tables 1-3 and Runtime Complexity.

## Connection to this project — interpretation

A speed-oriented ASR/LLM fusion control. It can be faster than a different LLM correction pipeline while remaining slower than standalone Whisper.

## Limitations when applying the result

Synchronous token generation is not causal audio streaming: the paper evaluates isolated utterances and does not report streaming word-delay distributions. The 17M adapter size excludes the 1.55B Whisper and 7B LLM weights.

## Code and models

- [https://github.com/csalt-research/salsa](https://github.com/csalt-research/salsa)

Official training/inference repository confirmed. No dependency installation, model download or execution was performed.

## Source version

PDF SHA-256: `0f48126dafac57e8877d77c4ae139b079f43db14e38e0dadd1d372f750c10c40`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
