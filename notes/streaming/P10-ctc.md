# P10. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks

| Field | Value |
|---|---|
| Authors | Alex Graves; Santiago Fernández; Faustino Gomez; Jürgen Schmidhuber |
| Source date | 2006 |
| Publication | ICML 2006 |
| Category | ASR alignment, encoders and streaming baselines |
| Reading priority | 1 — Sort theory |
| Review depth | Bibliography/abstract checked; full PDF collected and parsed; no exhaustive close reading or reproduction. |
| PDF | 8 pages; `papers/pdf/P10_ctc.pdf` |
| Source verification date | 2026-11-10 |

[Source](https://www.cs.toronto.edu/~graves/icml_2006.pdf) · [PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

## Main idea and method

Original Connectionist Temporal Classification paper learns audio/label sequences without frame-level target alignments.

Combine the probabilities of all frame paths corresponding to one transcription through blank, repeat label collapse, and forward-backward.

## Source-reported evidence

Evaluates CTC on phoneme recognition without supplied label alignment.

Evidence location: Sections 2–4, CTC objective and forward-backward.

## Connection to this project — interpretation

Foundational material for connecting acoustic evidence and text prefixes before architecture selection.

## Limitations when applying the result

CTC itself does not guarantee a streaming encoder. Greedy decoders are already very cheap, so the computational savings in candidate predictions may be small.

## Code and models

An official implementation link was not confirmed in this review. This does not establish that no implementation exists.

Do not assume a public checkpoint exists.

## Source version

PDF SHA-256: `ed7955f5d4ca3b068ef18855632e14195d63fd4ca7ea77d2c6ba02e94346d157`. The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
