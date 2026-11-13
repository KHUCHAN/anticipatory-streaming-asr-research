# Paper index

Source verification date: 2026-11-10. Initial catalog: 33 papers. IDs group topics rather than publication years.

Notes separate source-reported evidence from project interpretation. Original PDFs are local-only and can be obtained from the linked sources. Later related-work links appear in the research documents.

## Prediction before audio arrival

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P01 | [Speculative Speech Recognition by Audio-Prefixed Low-Rank Adaptation of Language Models](../notes/anticipatory/P01-ssr.md) | 2024 / Interspeech 2024 | 1 — read first |
| P02 | [Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models](../notes/anticipatory/P02-speculative-listening.md) | 2026 / arXiv preprint, 2026-11-04 | 1 — read first |

## Speculative decoding and path reuse

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P03 | [SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding](../notes/decoding/P03-specasr.md) | 2025 / arXiv 2025; DAC 2025 accepted stated in the original text | 1 — read first |
| P04 | [ParaASR: Multi-Token Prediction for Fast and Long-Context LLM-Based Speech Recognition](../notes/decoding/P04-paraasr.md) | 2026 / arXiv preprint, 2026-07-31 | 1 — read first |
| P05 | [Fast Inference from Transformers via Speculative Decoding](../notes/decoding/P05-speculative-decoding.md) | 2022 / ICML 2023; arXiv first released 2022 | 1 — read first |
| P06 | [Accelerating Large Language Model Decoding with Speculative Sampling](../notes/decoding/P06-speculative-sampling.md) | 2023 / arXiv technical report, 2023 | 2 — Check the theory |
| P07 | [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](../notes/decoding/P07-specinfer.md) | 2023 / arXiv 2023; SpecInfer original version | 2 — Branch implementation notes |
| P08 | [Blockwise Parallel Decoding for Deep Autoregressive Models](../notes/decoding/P08-blockwise-parallel.md) | 2018 / arXiv preprint, 2018 | 2 — multi-token base |

## ASR alignment, encoders and streaming baselines

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P09 | [Sequence Transduction with Recurrent Neural Networks](../notes/streaming/P09-rnnt.md) | 2012 / arXiv preprint, 2012 | 1 — Sort theory |
| P10 | [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](../notes/streaming/P10-ctc.md) | 2006 / ICML 2006 | 1 — Sort theory |
| P11 | [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](../notes/streaming/P11-tdt.md) | 2023 / ICML 2023, PMLR 202 | 2 — Skip time base |
| P12 | [Conformer: Convolution-augmented Transformer for Speech Recognition](../notes/streaming/P12-conformer.md) | 2020 / Conformer original paper, arXiv 2020 | 2 — Encoder comparison |
| P13 | [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](../notes/streaming/P13-fastconformer.md) | 2023 / ASRU 2023 accepted; Check original text arXiv | 2 — Efficiency baseline candidate |
| P14 | [Robust Speech Recognition via Large-Scale Weak Supervision](../notes/streaming/P14-whisper.md) | 2022 / Whisper original paper, arXiv 2022 | 2 — External accuracy reference |
| P15 | [Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition](../notes/streaming/P15-emformer.md) | 2020 / ICASSP 2021; arXiv first released 2020 | 2 — Limited context comparison |
| P30 | [Zipformer: A faster and better encoder for automatic speech recognition](../notes/streaming/P30-zipformer.md) | 2023 / Zipformer original paper, arXiv 2023 | 1 — External streaming baseline |

## State-space models and ConMamba

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P16 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../notes/state-space/P16-mamba.md) | 2023 / Mamba original paper, arXiv 2023 | 2 — State compression candidate |
| P17 | [Efficiently Modeling Long Sequences with Structured State Spaces](../notes/state-space/P17-s4.md) | 2021 / ICLR 2022; arXiv first released 2021 | 3 — State space theory |
| P18 | [Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation, Recognition, and Synthesis](../notes/state-space/P18-speech-slytherin.md) | 2024 / arXiv preprint, 2024 | 1 — Grounds for applying state space |
| P19 | [Assessing the Performance and Efficiency of Mamba ASR in Low-Resource Scenarios](../notes/state-space/P19-mamba-low-resource.md) | 2025 / Interspeech 2025 | 2 — Source of reported figures |
| P20 | [Attention-Free Dual-Mode ASR with Latency-Controlled Selective State Spaces](../notes/state-space/P20-lc-bimamba.md) | 2025 / Interspeech 2025 | 1 — streaming state space basis |

## FGN, GraphCast and WeatherNext

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P21 | [Skillful joint probabilistic weather forecasting from marginals](../notes/weather/P21-fgn.md) | 2025 / arXiv technical report, 2025 | 1 — Confirmation of FGN principle |
| P22 | [GraphCast: Learning skillful medium-range global weather forecasting](../notes/weather/P22-graphcast.md) | 2022 / Science 2023; Confirmation PDF is arXiv 2022 revised version of original release | 3 — graph weather background |
| P23 | [GenCast: Diffusion-based ensemble forecasting for medium-range weather](../notes/weather/P23-gencast.md) | 2023 / arXiv 2023; Related official paper Nature, 2024 online | 3 — FGN comparison target |
| P29 | [Operational tropical cyclone forecasting with AI](../notes/weather/P29-weathernext-cyclones.md) | 2026 / Nature 657, 2026; online 2026-08-06 | 3 — FGN follow-up application |

## Graph-based speech recognition

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P24 | [Graph Neural Networks for Contextual ASR with the Tree-Constrained Pointer Generator](../notes/graph-asr/P24-contextual-gnn.md) | 2023 / Reviewed source: arXiv preprint, 2023 | 2 — see candidate graph |
| P25 | [Tree-constrained Pointer Generator with Graph Neural Network Encodings for Contextual Speech Recognition](../notes/graph-asr/P25-tcpgen.md) | 2022 / Interspeech 2022 | 2 — contextual graph prototype |
| P26 | [Deep Graph Random Process for Relational-Thinking-Based Speech Recognition](../notes/graph-asr/P26-deep-graph-random-process.md) | 2020 / ICML 2020, PMLR 119 | 3 — relational ASR background |

## Disambiguating FGN/FGNN architectures

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P27 | [Factor Graph Neural Networks](../notes/disambiguation/P27-factor-graph.md) | 2023 / JMLR 24(181):1–54, 2023 | 3 — Avoid abbreviation confusion |
| P28 | [Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"](../notes/disambiguation/P28-finite-gaussian.md) | 2023 / CUNY PhD dissertation / arXiv, 2023 | 3 — Avoid abbreviation confusion |

## Original dataset papers

| ID | Paper | Source year / publication | Priority |
|---|---|---|---|
| P31 | [LibriSpeech: An ASR Corpus Based on Public Domain Audio Books](../notes/datasets/P31-librispeech.md) | 2015 / ICASSP 2015 | 1 — Understand basic data |
| P32 | [GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio](../notes/datasets/P32-gigaspeech.md) | 2021 / GigaSpeech original paper, arXiv 2021 | 2 — Understanding extended data |
| P33 | [Common Voice: A Massively-Multilingual Speech Corpus](../notes/datasets/P33-common-voice.md) | 2019 / Common Voice original paper, arXiv 2019 | 2 — Understanding multilingual data |
