# P54. Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models

| Field | Value |
|---|---|
| Authors | Hu, Yuchen; Chen, Chen; Qin, Chengwei; Zhu, Qiushi; Chng, Eng Siong; Li, Ruizhe |
| Source date | 2024-05-16 |
| Publication | Findings of ACL 2024 |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 2 - generation or fusion control |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 14 pages; `papers/pdf/P54_clozeger.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://aclanthology.org/2024.findings-acl.37/) · [PDF](https://aclanthology.org/2024.findings-acl.37.pdf) · [Additional primary source 1](https://arxiv.org/abs/2405.10025)

## Main idea and method

ClozeGER reformats candidate disagreements as multiple-choice gaps, optionally using source speech in a multimodal LLM.

Align hypotheses, choose tokens for blanks, calibrate option logits and optionally apply a second generative post-processing stage. SpeechGPT adds acoustic information; text-only LLaMA-2 is a separate ablation.

## Source-reported evidence

Table 1 SpeechGPT branch: CommonVoice WER 14.4% baseline, 13.7% raw ClozeGER, 12.4% calibrated and 8.5% after post-processing. Table 2 text-only branch reaches 8.6%. Final gains cannot be assigned to selection alone or acoustic input alone.

Evidence location: PDF Sections 3-4, Figure 2; p. 6 Tables 1-2; pp. 7-8 calibration and post-processing ablations.

## Connection to this project — interpretation

Direct prior for compactly presenting candidate differences to an LLM and for acoustic verification of linguistic correction. Option-order bias is relevant even when asking only for a candidate ID.

## Limitations when applying the result

Word choices can compose a new string, and the final post-processor can rewrite common context; this is not strict whole-list reranking. Full-utterance SpeechGPT experiments do not establish online causality, word-level stability or cheap execution.

## Code and models

- [https://github.com/Hypotheses-Paradise/Hypo2Trans](https://github.com/Hypotheses-Paradise/Hypo2Trans)

The paper points to the HyPoradise repository, which is accessible. A separate complete ClozeGER implementation/checkpoint was not confirmed in that review.

## Source version

PDF SHA-256: `da2a92c8cf458eeafe137707c3325e7e068e2d7f8e1e558a3f7f36697a1f77ae`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
