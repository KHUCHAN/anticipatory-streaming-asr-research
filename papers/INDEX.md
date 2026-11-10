# 논문 전체 목록

검증 기준일: 2026-11-10. 총 33편. 번호는 연도순이 아니라 주제별 식별자다.

논문별 문서에서 **논문이 보고한 내용**과 **이 연구 아이디어에 대한 해석**을 구분한다. 논문 PDF는 로컬에 보관하며 아래 원문 링크에서 다시 받을 수 있다.

## 음성 도착 전 예측

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P01 | [Speculative Speech Recognition by Audio-Prefixed Low-Rank Adaptation of Language Models](../notes/anticipatory/P01-ssr.md) | 2024 / Interspeech 2024 | 1 — 먼저 읽기 |
| P02 | [Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models](../notes/anticipatory/P02-speculative-listening.md) | 2026 / arXiv preprint, 2026-11-04 | 1 — 먼저 읽기 |

## 추측 디코딩과 경로 재사용

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P03 | [SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding](../notes/decoding/P03-specasr.md) | 2025 / arXiv 2025; 원문에 DAC 2025 accepted 기재 | 1 — 먼저 읽기 |
| P04 | [ParaASR: Multi-Token Prediction for Fast and Long-Context LLM-Based Speech Recognition](../notes/decoding/P04-paraasr.md) | 2026 / arXiv preprint, 2026-07-31 | 1 — 먼저 읽기 |
| P05 | [Fast Inference from Transformers via Speculative Decoding](../notes/decoding/P05-speculative-decoding.md) | 2022 / ICML 2023; arXiv 최초 공개 2022 | 1 — 먼저 읽기 |
| P06 | [Accelerating Large Language Model Decoding with Speculative Sampling](../notes/decoding/P06-speculative-sampling.md) | 2023 / arXiv technical report, 2023 | 2 — 이론 확인 |
| P07 | [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](../notes/decoding/P07-specinfer.md) | 2023 / arXiv 2023; SpecInfer 원문 버전 | 2 — 분기 구현 참고 |
| P08 | [Blockwise Parallel Decoding for Deep Autoregressive Models](../notes/decoding/P08-blockwise-parallel.md) | 2018 / arXiv preprint, 2018 | 2 — 다중 토큰 기반 |

## ASR 정렬·인코더·스트리밍 기준선

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P09 | [Sequence Transduction with Recurrent Neural Networks](../notes/streaming/P09-rnnt.md) | 2012 / arXiv preprint, 2012 | 1 — 정렬 이론 |
| P10 | [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](../notes/streaming/P10-ctc.md) | 2006 / ICML 2006 | 1 — 정렬 이론 |
| P11 | [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](../notes/streaming/P11-tdt.md) | 2023 / ICML 2023, PMLR 202 | 2 — 시간축 건너뛰기 |
| P12 | [Conformer: Convolution-augmented Transformer for Speech Recognition](../notes/streaming/P12-conformer.md) | 2020 / Conformer 원 논문, arXiv 2020 | 2 — 인코더 비교 |
| P13 | [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](../notes/streaming/P13-fastconformer.md) | 2023 / ASRU 2023 accepted; 확인 원문 arXiv | 2 — 효율 기준선 후보 |
| P14 | [Robust Speech Recognition via Large-Scale Weak Supervision](../notes/streaming/P14-whisper.md) | 2022 / Whisper 원 논문, arXiv 2022 | 2 — 외부 정확도 참고 |
| P15 | [Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition](../notes/streaming/P15-emformer.md) | 2020 / ICASSP 2021; arXiv 최초 공개 2020 | 2 — 제한 문맥 비교 |
| P30 | [Zipformer: A faster and better encoder for automatic speech recognition](../notes/streaming/P30-zipformer.md) | 2023 / Zipformer 원 논문, arXiv 2023 | 1 — 외부 streaming 기준선 |

## 상태공간 모델과 ConMamba

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P16 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../notes/state-space/P16-mamba.md) | 2023 / Mamba 원 논문, arXiv 2023 | 2 — 상태 압축 후보 |
| P17 | [Efficiently Modeling Long Sequences with Structured State Spaces](../notes/state-space/P17-s4.md) | 2021 / ICLR 2022; arXiv 최초 공개 2021 | 3 — 상태공간 이론 |
| P18 | [Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation, Recognition, and Synthesis](../notes/state-space/P18-speech-slytherin.md) | 2024 / arXiv preprint, 2024 | 1 — 상태공간 적용 근거 |
| P19 | [Assessing the Performance and Efficiency of Mamba ASR in Low-Resource Scenarios](../notes/state-space/P19-mamba-low-resource.md) | 2025 / Interspeech 2025 | 2 — 보고 수치의 출처 |
| P20 | [Attention-Free Dual-Mode ASR with Latency-Controlled Selective State Spaces](../notes/state-space/P20-lc-bimamba.md) | 2025 / Interspeech 2025 | 1 — streaming 상태공간 근거 |

## FGN·GraphCast·WeatherNext

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P21 | [Skillful joint probabilistic weather forecasting from marginals](../notes/weather/P21-fgn.md) | 2025 / arXiv technical report, 2025 | 1 — FGN 원리 확인 |
| P22 | [GraphCast: Learning skillful medium-range global weather forecasting](../notes/weather/P22-graphcast.md) | 2022 / Science 2023; 확인 PDF는 arXiv 2022 최초 공개본의 개정 버전 | 3 — 그래프 기상 배경 |
| P23 | [GenCast: Diffusion-based ensemble forecasting for medium-range weather](../notes/weather/P23-gencast.md) | 2023 / arXiv 2023; 관련 정식 논문 Nature, 2024 online | 3 — FGN 비교 대상 |
| P29 | [Operational tropical cyclone forecasting with AI](../notes/weather/P29-weathernext-cyclones.md) | 2026 / Nature 657, 2026; online 2026-08-06 | 3 — FGN 후속 응용 |

## 그래프 기반 음성인식

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P24 | [Graph Neural Networks for Contextual ASR with the Tree-Constrained Pointer Generator](../notes/graph-asr/P24-contextual-gnn.md) | 2023 / 확인 원문: arXiv preprint, 2023 | 2 — 후보 graph 참고 |
| P25 | [Tree-constrained Pointer Generator with Graph Neural Network Encodings for Contextual Speech Recognition](../notes/graph-asr/P25-tcpgen.md) | 2022 / Interspeech 2022 | 2 — contextual graph 원형 |
| P26 | [Deep Graph Random Process for Relational-Thinking-Based Speech Recognition](../notes/graph-asr/P26-deep-graph-random-process.md) | 2020 / ICML 2020, PMLR 119 | 3 — relational ASR 배경 |

## FGN/FGNN 동명 아키텍처 구분

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P27 | [Factor Graph Neural Networks](../notes/disambiguation/P27-factor-graph.md) | 2023 / JMLR 24(181):1–54, 2023 | 3 — 약어 혼동 방지 |
| P28 | [Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"](../notes/disambiguation/P28-finite-gaussian.md) | 2023 / CUNY PhD dissertation / arXiv, 2023 | 3 — 약어 혼동 방지 |

## 학습 데이터셋 원 논문

| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |
|---|---|---|---|
| P31 | [LibriSpeech: An ASR Corpus Based on Public Domain Audio Books](../notes/datasets/P31-librispeech.md) | 2015 / ICASSP 2015 | 1 — 기본 데이터 이해 |
| P32 | [GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio](../notes/datasets/P32-gigaspeech.md) | 2021 / GigaSpeech 원 논문, arXiv 2021 | 2 — 확장 데이터 이해 |
| P33 | [Common Voice: A Massively-Multilingual Speech Corpus](../notes/datasets/P33-common-voice.md) | 2019 / Common Voice 원 논문, arXiv 2019 | 2 — 다국어 데이터 이해 |
