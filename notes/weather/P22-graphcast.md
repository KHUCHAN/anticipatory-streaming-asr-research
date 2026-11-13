# P22. GraphCast: Learning skillful medium-range global weather forecasting

| Field | Value |
|---|---|
| Authors | Lam, Remi; Sanchez-Gonzalez, Alvaro; Willson, Matthew; Wirnsberger, Peter; Fortunato, Meire; Alet, Ferran; Ravuri, Suman; Ewalds, Timo; Eaton-Rosen, Zach; Hu, Weihua; Merose, Alexander; Hoyer, Stephan; Holland, George; Vinyals, Oriol; Stott, Jacklynn; Pritzel, Alexander; Mohamed, Shakir; Battaglia, Peter |
| Source date | 2022-12-24 |
| Publication | Science 2023; Confirmation PDF is arXiv 2022 revised version of original release |
| Category | FGN, GraphCast and WeatherNext |
| Reading priority | 3 — graph weather background |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 102 pages; `papers/pdf/P22_graphcast.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2212.12794) · [PDF](https://arxiv.org/pdf/2212.12794v2) · [Additional primary source 1](https://doi.org/10.1126/science.adi2336)

## Main idea and method

GraphCast predicts global medium-range weather using a GNN on a spherical mesh.

Calculate the next weather state autoregressively using a grid-mesh encoder, mesh message passing, and mesh-grid decoder.

## Source-reported evidence

Evaluates accuracy and efficiency on weather reanalysis and operational forecasts.

Evidence location: graph encoder–processor–decoder architecture.

## Connection to this project — interpretation

We summarize that the graph structure and the functional probability modeling of FGN are different components in the Google Weather model.

## Limitations when applying the result

A weather spherical mesh does not automatically define a suitable speech graph. Speech nodes, edges and temporal causality require a separate design.

## Code and models

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `71ce5f63fb0b67955d38df35b766d9cd4ef27b4cd6188c7cbe4f9307b0e758c3`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
