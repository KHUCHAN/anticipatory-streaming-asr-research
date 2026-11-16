# P60. It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition

| Field | Value |
|---|---|
| Authors | Chen, Chen; Li, Ruizhe; Hu, Yuchen; Siniscalchi, Sabato Marco; Chen, Pin-Yu; Chng, Eng Siong; Yang, Chao-Han Huck |
| Source date | 2024-02-08 |
| Publication | ICLR 2024 |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 2 - generation or fusion control |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 17 pages; `papers/pdf/P60_late-acoustic-fusion.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://arxiv.org/abs/2402.05457) · [PDF](https://arxiv.org/pdf/2402.05457v1) · [Additional primary source 1](https://openreview.net/forum?id=QqjFHyQwtF)

## Main idea and method

Uncertainty-Aware Dynamic Fusion combines a generative corrector with acoustic token distributions.

Calibrate LLM uncertainty and dynamically mix its token logits with a separately adapted acoustic decoder. Late fusion requires a common output vocabulary; the acoustic decoder is fine-tuned to the LLaMA tokenizer.

## Source-reported evidence

Table 1: GER baseline ATIS/WSJ WER 1.61/2.83%; static late fusion 1.36/2.55%, UADF 1.24/2.47%. Early/mid fusion comparisons and uncertainty ablations are also reported.

Evidence location: Collected arXiv v1 pp. 4-6 methods/setup; p. 7 Table 1; p. 8 Table 2.

## Connection to this project — interpretation

Direct prior for an uncertainty-aware language-plus-acoustic correction rule. A calibrated uncertainty gate or a lightweight acoustic decoder is not by itself a novel combination.

## Limitations when applying the result

The ASR decoder is adapted for vocabulary compatibility, so this is not a zero-training plug-in for arbitrary frozen models. Full-utterance correction results do not show streaming word-delay or net inference speed improvement. An exact causal adaptation would require separate evidence.

## Code and models

- [https://github.com/Hypotheses-Paradise/UADF](https://github.com/Hypotheses-Paradise/UADF)
- [https://huggingface.co/PeacefulData/UADFusionGER](https://huggingface.co/PeacefulData/UADFusionGER)

Official repository and model-card pointers confirmed. OpenReview PDF download returned HTTP 403, so the collected evidence version is the author arXiv v1 with an ICLR-2024 publication header.

## Source version

PDF SHA-256: `de2453e0ccb0b9cd924614724215e0fc8c09bddfe95fa1ac527b7ad9ce4987ea`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
