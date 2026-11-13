# P30. Zipformer: A faster and better encoder for automatic speech recognition

| Field | Value |
|---|---|
| Authors | Yao, Zengwei; Guo, Liyong; Yang, Xiaoyu; Kang, Wei; Kuang, Fangjun; Yang, Yifan; Jin, Zengrui; Lin, Long; Povey, Daniel |
| Source date | 2023-10-17 |
| Publication | Zipformer original paper, arXiv 2023 |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 1 — External streaming baseline |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 16 pages; `papers/pdf/P30_zipformer.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2310.11230) · [PDF](https://arxiv.org/pdf/2310.11230v4) · [Additional primary source 1](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)

## Main idea and method

Zipformer combines encoder time resolutions to improve efficiency and recognition.

We propose a U-Net-like multi-frame rate structure, attention weight reuse, BiasNorm, Swoosh activation, and ScaledAdam.

## Source-reported evidence

LibriSpeech, AISHELL-1, WenetSpeech compares and discloses icefall implementation.

Evidence location: encoder design and ASR experiments.

## Connection to this project — interpretation

Provides the architecture/training context for an efficient streaming encoder and short prediction-network context. The locally evaluated backbone is the 66.11M LibriSpeech 960h model described below.

## Limitations when applying the result

Public 960h weights are not matched to this project's 100h acoustic training. ONNX CPU versus PyTorch MPS runtime differences do not isolate architecture effects.

## Code and models

- [https://github.com/k2-fsa/icefall](https://github.com/k2-fsa/icefall)

Initially, links and source connections were checked only. On 2026-11-11, official 66.11M Streaming Zipformer FP32 ONNX weights were evaluated with streaming greedy decoding on all 5,559 test utterances. WER was 3.058/7.820%, near official 3.06/7.79%. This reproduces checkpoint evaluation, not all ICLR 2024 tables/training or our forecaster. [Baseline report](../../docs/baseline_zipformer_2026-11-11.md).

## Source version

PDF SHA-256: `94deedaae51eb7ad2016845c84c8e27c070cc0c0538162851ca41e18778b269b`. The PDF version/hash identifies the source. Original acoustic training and the paper's complete experiments were not reproduced; subsequent public-checkpoint evaluation is documented above.

## Project follow-up

On 2026-11-12, 4.60M full/band2 forecasters were trained with matched initialization for 100h × 15 epochs on the frozen 66.11M backbone and evaluated on the full test. Band2+gate mean display gain was 89.58/70.28ms over eager native output and 30.32/16.54ms over native probability+gate. Actual test risks differ; this does not establish matched-risk Pareto improvement, faster commitment or reproduction of the original paper's acoustic training. [Conditions and diagnostics](../../docs/zipformer_ours_2026-11-12.md).

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
