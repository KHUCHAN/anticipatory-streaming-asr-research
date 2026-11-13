# P21. Skillful joint probabilistic weather forecasting from marginals

| Field | Value |
|---|---|
| Authors | Alet, Ferran; Price, Ilan; El-Kadi, Andrew; Masters, Dominic; Markou, Stratis; Andersson, Tom R.; Stott, Jacklynn; Lam, Remi; Willson, Matthew; Sanchez-Gonzalez, Alvaro; Battaglia, Peter |
| Source date | 2025-06-12 |
| Publication | arXiv technical report, 2025 |
| Category | FGN, GraphCast and WeatherNext |
| Reading priority | 1 — Confirmation of FGN principle |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 53 pages; `papers/pdf/P21_fgn.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2506.10772) · [PDF](https://arxiv.org/pdf/2506.10772v1)

## Main idea and method

This is a probabilistic weather prediction paper explaining the Functional Generative Network of WeatherNext 2.

Conditions on two recent atmospheric states; shared 32-dimensional noise enters conditional normalization to generate coherent ensemble samples.

## Source-reported evidence

The 2-order Markov model is calculated using Eq. (1), the noise path is specified in the main text and appendix. 8 times acceleration compared to GenCast is the result of weather prediction compared to repeated diffusion sampling.

Evidence location: PDF 3 page Eq. (1), 7 page Section 3.3, 13 page and 22–23 page.

## Connection to this project — interpretation

An optional inspiration for coherent sampling of multiple futures; first isolate the core predict–verify effect.

## Limitations when applying the result

The 32 dimension is not the overall memory state size, but a noise dimension that controls stochastic fluctuations. Weather results do not demonstrate ASR acceleration or next word accuracy.

## Code and models

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `2141e6fe9b314143a2fe3655f6d41e74e804c573de4987e7e94b16faf313e16e`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
