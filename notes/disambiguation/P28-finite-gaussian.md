# P28. Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"

| Field | Value |
|---|---|
| Authors | Grezes, Felix |
| Source date | 2023-06-13 |
| Publication | CUNY PhD dissertation / arXiv, 2023 |
| Category | Disambiguating FGN/FGNN architectures |
| Reading priority | 3 — Avoid abbreviation confusion |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 90 pages; `papers/pdf/P28_finite-gaussian.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2306.07796) · [PDF](https://arxiv.org/pdf/2306.07796v1)

## Main idea and method

This is a doctoral thesis that reduces excessive confidence in areas far from the input using Finite Gaussian Neuron.

Converts and retrains existing neural networks as FGNs and evaluates adversarial robustness.

## Source-reported evidence

We experiment on MNIST and SpeechCommands, and also report limitations on more complex speech commands and strong attacks.

Evidence location: Thesis abstract and SpeechCommands experiments.

## Connection to this project — interpretation

Preserve the original FGNN reference while separating it from this project's main research direction.

## Limitations when applying the result

SpeechCommands classification is different from ASR, which transcribes continuous speech into sentences. It is not the same model as Functional Generative Network.

## Code and models

- [https://github.com/grezesf/FGN---Research](https://github.com/grezesf/FGN---Research)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `0249184611054039ba37e6a5129175db141f75fb607a7bd04484f914aa25dd62`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
