# P27. Factor Graph Neural Networks

| Field | Value |
|---|---|
| Authors | Zhen Zhang; Mohammed Haroon Dupty; Fan Wu; Javen Qinfeng Shi; Wee Sun Lee |
| Source date | 2023 |
| Publication | JMLR 24(181):1–54, 2023 |
| Category | Disambiguating FGN/FGNN architectures |
| Reading priority | 3 — Avoid abbreviation confusion |
| Review depth | Bibliography, abstract, core sections and experimental setup checked; no exhaustive close reading or reproduction. |
| PDF | 54 pages; `papers/pdf/P27_factor-graph.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://jmlr.org/papers/v24/21-0434.html) · [PDF](https://jmlr.org/papers/volume24/21-0434/21-0434.pdf)

## Main idea and method

FGNN, which stands for Factor Graph Neural Network, is a high-order relational learning study.

Message passing between factors and variables is neurally networked to express sum-product and max-product inference.

## Source-reported evidence

Evaluates synthetic graphical models, LDPC decoding and graph matching.

Evidence location: JMLR abstract and Section 5 experiments.

## Connection to this project — interpretation

Records that this similarly named architecture differs from the user's Google FGN reference.

## Limitations when applying the result

An ASR-trained model was not confirmed in this paper. It alone does not establish that a public FGNN-ASR checkpoint exists.

## Code and models

- [https://github.com/zzhang1987/Factor-Graph-Neural-Network](https://github.com/zzhang1987/Factor-Graph-Neural-Network)

Implementation links and their source connection were checked; installation, execution, weight download and performance reproduction were not performed in that review.

## Source version

PDF SHA-256: `6b198dfceb67a11bf2b807231c67f8c34f8f8f98a5bb5585d671dd077d330cae`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
