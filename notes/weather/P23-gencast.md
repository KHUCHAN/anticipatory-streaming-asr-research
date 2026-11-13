# P23. GenCast: Diffusion-based ensemble forecasting for medium-range weather

| Field | Value |
|---|---|
| Authors | Price, Ilan; Sanchez-Gonzalez, Alvaro; Alet, Ferran; Andersson, Tom R.; El-Kadi, Andrew; Masters, Dominic; Ewalds, Timo; Stott, Jacklynn; Mohamed, Shakir; Battaglia, Peter; Lam, Remi; Willson, Matthew |
| Source date | 2023-12-25 |
| Publication | arXiv 2023; Related official paper Nature, 2024 online |
| Category | FGN, GraphCast and WeatherNext |
| Reading priority | 3 — FGN comparison target |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 96 pages; `papers/pdf/P23_gencast.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2312.15796) · [PDF](https://arxiv.org/pdf/2312.15796v2) · [Additional primary source 1](https://doi.org/10.1038/s41586-024-08252-9)

## Main idea and method

GenCast is a diffusion-based ensemble forecaster generating multiple plausible future weather trajectories.

Model the conditional distribution of the next state using a graph-based network and iterative denoising sampling.

## Source-reported evidence

Evaluates weather-ensemble skill, extremes, tropical cyclones and wind power. It is a major compute comparison in the FGN paper.

Evidence location: probabilistic forecasting method and sampling procedure.

## Connection to this project — interpretation

Helps distinguish single-step generation from repeated sampling costs and identify the appropriate ASR comparison.

## Limitations when applying the result

The catalog PDF uses the arXiv GenCast title; the Nature publication is titled “Probabilistic weather forecasting with machine learning.”

## Code and models

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `4beaff56fa98f836a77808f9979397e59ad0dcc5f31fec169b77d05e7a1e67dd`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
