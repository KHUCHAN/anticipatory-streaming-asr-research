# P38. TrimTail: Low-Latency Streaming ASR with Simple but Effective Spectrogram-Level Length Penalty

| Field | Value |
|---|---|
| Authors | Song, Xingchen; Wu, Di; Wu, Zhiyong; Zhang, Binbin; Zhang, Yuekai; Peng, Zhendong; Li, Wenpeng; Pan, Fuping; Zhu, Changbao |
| Source date | 2022-11-01 |
| Publication | ICASSP 2023 |
| Category | Future context, provisional output and emission latency |
| Reading priority | 2 - related control or measurement |
| Review depth | Full PDF collected; core method, experimental conditions and selected result tables read. Visual checks are listed in the review manifest; author experiments not reproduced. |
| PDF | 6 pages; `papers/pdf/P38_trimtail.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2211.00522) · [PDF](https://arxiv.org/pdf/2211.00522v2)

## Main idea and method

Training-time spectrogram tail trimming encourages emission before the original end of the input.

Remove randomly sampled trailing feature frames while retaining target labels; combine with the existing ASR loss. The technique adds no separate inference network and differs from padding unavailable audio during inference.

## Source-reported evidence

AISHELL-1 Conformer CTC, 640ms chunk, maximum trim 50: LTD50 210 to 60ms; greedy/rescored error 5.81/5.05 to 5.83/5.09%. The paper also evaluates LibriSpeech and transducers.

Evidence location: PDF Section 2, Algorithm 1; Tables 1-3 and Section 3 latency discussion.

## Connection to this project — interpretation

A low-overhead timing baseline alongside FastEmit. Helps separate a model that emits late because of its training objective from a model helped specifically by a learned future prior.

## Limitations when applying the result

AISHELL error is CER despite generic table labeling. The LibriSpeech timing experiment uses baseline-derived pseudo-alignment; do not present its relative delays as independently aligned absolute word delays.

## Code and models

- [https://github.com/wenet-e2e/wenet/pull/1487](https://github.com/wenet-e2e/wenet/pull/1487)

The paper links the WeNet spectrim pull request; its public page was checked. No training or author result was reproduced.

## Source version

PDF SHA-256: `385851e7e9d269579672776ef6d8d4c1e6e11526e583ceb371faa2ca949a73e9`. Collected for the 2026-11-15 to 2026-11-16 related-work extension. PDF retrieval and source inspection only; no author training, inference or reported speedup reproduced.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
