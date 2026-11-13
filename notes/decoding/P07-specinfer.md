# P07. SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification

| Field | Value |
|---|---|
| Authors | Miao, Xupeng; Oliaro, Gabriele; Zhang, Zhihao; Cheng, Xinhao; Wang, Zeyu; Zhang, Zhengxin; Wong, Rae Ying Yee; Zhu, Alan; Yang, Lijie; Shi, Xiaoxiang; Shi, Chunan; Chen, Zhuoming; Arfeen, Daiyaan; Abhyankar, Reyna; Jia, Zhihao |
| Source date | 2023-05-16 |
| Publication | arXiv 2023; SpecInfer original version |
| Category | Speculative decoding and path reuse |
| Reading priority | 2 — Branch implementation notes |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 18 pages; `papers/pdf/P07_specinfer.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://arxiv.org/abs/2305.09781) · [PDF](https://arxiv.org/pdf/2305.09781v4)

## Main idea and method

Organizes candidate token paths as a tree for parallel target-LLM verification.

Verification of different prefixes is parallelized using token tree and tree attention mask.

## Source-reported evidence

Reports lower latency than prior serving systems in distributed inference/offloading settings.

Evidence location: token-tree verification design and serving evaluation.

## Connection to this project — interpretation

Relevant to formalizing likely next-word paths through trees, branches and shared-prefix reuse.

## Limitations when applying the result

This is an LLM serving result and does not directly prove the efficiency of acoustic verification or streaming ASR. As the number of branches increases, memory and verification costs also increase.

## Code and models

- [https://github.com/flexflow/FlexFlow](https://github.com/flexflow/FlexFlow)

The source link now redirects to flexflow/flexflow-train. Preservation/executability of the paper's serving artifact there was not confirmed.

## Source version

PDF SHA-256: `38764bfe741d39c9a309d1b715e8dc2cea6ea6c965337752b3562b1785afd9f1`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
