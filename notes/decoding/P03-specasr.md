# P03. SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding

| Field | Value |
|---|---|
| Authors | Wei, Linye; Zhong, Shuzhang; Xu, Songqiang; Wang, Runsheng; Huang, Ru; Li, Meng |
| Source date | 2025-07-24 |
| Publication | arXiv 2025; DAC 2025 accepted stated in the original text |
| Category | Speculative decoding and path reuse |
| Reading priority | 1 — read first |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 7 pages; `papers/pdf/P03_specasr.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2507.18181) · [PDF](https://arxiv.org/pdf/2507.18181v2)

## Main idea and method

Accelerates speculative decoding using similarity between small and large audio-conditioned ASR paths.

It combines adaptive draft length, draft recycling, and two-pass sparse token tree and verifies multiple candidate paths in parallel.

## Source-reported evidence

3.04–3.79 times is the result of using Vicuna-13B as a simulated target. The experiment records the Whisper tiny.en/medium.en path on RTX A6000 and then simulates the delay with TinyLlama/Llama/Vicuna.

Evidence location: PDF Page 5 Section V-A/V-B and Figure 11; 6 side ablation.

## Connection to this project — interpretation

Draft recycling and branch-budget control are relevant to this project's candidate-reuse policy.

## Limitations when applying the result

Does not establish forecasting before actual audio arrival. Do not cite its speedups as a deployed Llama-ASR or real-time Mac result.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `5bede7b1c67a5c37559cd584c92c0d25ce188c597985d0352caad3021bafd5ea`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
