# P51. N-best T5: Robust ASR Error Correction using Multiple Input Hypotheses and Constrained Decoding Space

| Field | Value |
|---|---|
| Authors | Ma, Rao; Gales, Mark J. F.; Knill, Kate M.; Qian, Mengjie |
| Source date | 2023-03-01 |
| Publication | Interspeech 2023 |
| Category | LLM hypothesis selection, correction and fusion |
| Reading priority | 1 - direct comparison |
| Review depth | Full PDF collected; core method and selected experiments/limitations inspected. Visual checks are enumerated in the review manifest. No author model or benchmark was run. |
| PDF | 5 pages; `papers/pdf/P51_nbest-t5.pdf` |
| Source verification date | 2026-11-16 |

[Source](https://www.isca-archive.org/interspeech_2023/ma23e_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2023/ma23e_interspeech.pdf) · [Additional primary source 1](https://arxiv.org/abs/2303.00456) · [Additional primary source 2](https://ieeexplore.ieee.org/document/10930744/)

## Main idea and method

Fine-tune a text-to-text model on multiple ASR hypotheses and constrain its decoding to acoustically supported alternatives.

Concatenate an N-best list for T5-base. Compare free generation, selection within the original list and lattice-constrained decoding; interpolate correction and ASR scores. Training pairs come from perturbed training speech.

## Source-reported evidence

LibriSpeech, Table 4: baseline clean/other WER 2.88/7.06%; 10-best T5 unrestricted 2.67/6.56%, N-best constrained 2.54/6.31%, lattice constrained 2.53/6.27%. No measured streaming word-delay/RTF result was found in the inspected paper.

Evidence location: PDF Sections 2-4, constrained-decoding methods; p. 4 Tables 1-4 and Figure 5.

## Connection to this project — interpretation

Direct prior for choosing among ASR candidates while retaining acoustic evidence. Distinguish selecting a complete hypothesis from composing a new lattice path or generating unrestricted text.

## Limitations when applying the result

A full-utterance correction study, not a causal chunk-level display evaluation. Constraining output does not guarantee that the selected string is correct or improves every utterance. Include candidate extraction and correction costs.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

An exact author implementation and checkpoint were not confirmed in this bounded review. This does not establish their absence.

## Source version

PDF SHA-256: `687e5ceb9c3e67ae05c3ce614e66b424c1c818985fe07371bd5d14ad8401a9a9`. Research-only collection on 2026-11-16. No model, training, benchmark, API inference, checkpoint conversion or experiment migration was executed.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
