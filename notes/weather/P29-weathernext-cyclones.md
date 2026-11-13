# P29. Operational tropical cyclone forecasting with AI

| Field | Value |
|---|---|
| Authors | Alet, Ferran; Andersson, Tom R.; Price, Ilan; Markou, Stratis; El-Kadi, Andrew; Masters, Dominic; Li, Amy; Merchant, Samier; Williams, Natalie; Thornton, Gregory; MacKay, Ken; Graham, Olivia; Uddin, Akib; Gaiarin, Ben; Shah, Devaja; Kruse, Elinor; Hogsett, Wallace; Zelinsky, David; Cangialosi, John; Martinez, Jonathan; Franklin, James; DeMaria, Mark; Musgrave, Kate; Bain, Caroline L.; Titley, Helen; Stott, Jacklynn; Lam, Remi; Bell, Aaron; Komarek, Paul; Willson, Matthew; Sanchez-Gonzalez, Alvaro; Battaglia, Peter |
| Source date | 2026-08-06 |
| Publication | Nature 657, 2026; online 2026-08-06 |
| Category | FGN, GraphCast and WeatherNext |
| Reading priority | 3 — FGN follow-up application |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 26 pages; `papers/pdf/P29_weathernext-cyclones.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://www.nature.com/articles/s41586-026-10953-2) · [PDF](https://repository.library.noaa.gov/view/noaa/76404/noaa_76404_DS1.pdf) · [Additional primary source 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC13581599/)

## Main idea and method

Operational WeatherNext Cyclones research jointly trains on weather states and cyclone data.

Combines large ensembles and cyclone-specific objectives to forecast track, intensity and size.

## Source-reported evidence

2023–2025 reports the typhoon evaluation and comparison of existing operation models. The official WeatherNext repository links this paper to the weight guide.

Evidence location: Nature text, Figure 1, Methods; NOAA Archive PDF.

## Connection to this project — interpretation

This is supplementary material that confirms the source of the application context and public implementation in which FGN generates actual multiple future paths.

## Limitations when applying the result

Success in weather does not establish an ASR benefit. This is also distinct from the WeatherNext 3 architecture paper.

## Code and models

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `419b33da61eeee115d139518b25df34552d9768331cf9e0922c3f8d0af4abcbf`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
