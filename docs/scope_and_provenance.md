# Management of collection scope and sources

Initial verification date: 2026-11-10. The 33 records cover papers explicitly mentioned in two supplied research conversations and directly connected foundational work. This is not an exhaustive worldwide systematic review.

Status update: 2026-11-11. The 33 papers are the initial catalog. Later sources on bounded-context prediction, confidence, Stateful Conformer and multi-stage display are linked separately in the [literature map](literature_map.md), [review](results_review_2026-11-11.md) and [960h plan](librispeech960_benchmark_plan.md), not added to that count. Start with [current conclusions](conclusion_and_next_steps.md).

The initial source collection has 2 anticipatory papers, 6 speculative-decoding papers, 8 alignment/encoder/streaming papers, 5 state-space papers, 4 weather-FGN papers, 3 graph-ASR papers, 2 acronym-disambiguation papers and 3 dataset papers.

## Targeted extension, 2026-11-16

The future-context extension brought the catalog to **50 papers**: the initial 33 plus [17 additions on future context, prompt-and-refine, emission latency, rewriting, distillation and adaptive waiting](prior_work_predict_refine_2026-11-16.md). P43/P44 promote two previously supplementary links to collected catalog entries. Other supplementary links remain outside the count. Old note review dates are preserved; new notes carry their own verification date.

The [extension manifest](../references/reviews/predict_refine_2026-11-16.json) records the bounded search, source versions, selected numerical cells and visual checks. All 17 PDFs were collected and their core methods/selected experiments inspected. Author experiments were not reproduced. Source PDFs can be proceedings versions or pinned arXiv versions, as recorded per entry. Later arXiv versions and acceptance metadata do not silently replace the collected evidence. This extension does not certify exhaustive coverage or novelty.

## LLM correction extension, 2026-11-16

The current catalog has **60 papers** after [ten LLM candidate-selection, correction and fusion additions](prior_work_llm_correction_2026-11-16.md), P51-P60. Their full PDFs, original English notes and [review manifest](../references/reviews/llm_correction_2026-11-16.json) distinguish fixed-list reranking, unrestricted generation, acoustic fusion and actual streaming measurements. Supplementary metadata-only sources are not counted as full-text-reviewed entries.

The user limited this task to research. Only source retrieval/inspection and documentation checks were performed; no training, benchmark, model inference, experiment migration or architecture change was executed. The teacher check uses publicly distributed text sources and preserves the uncertainty caused by a dirty upstream training revision. Public notes omit internal paths and network identifiers appearing in the upstream raw log.

arXiv, ISCA, PMLR, JMLR, author pages, Nature/NOAA original text, and official project repositories were used. The search was conducted focusing on SSR, speculative listening/ASR, token tree verification, ConMamba/LC-BiMamba, FGN/WeatherNext, contextual graph ASR, Zipformer, and the attached dataset name.

## record configuration

Each paper records authors, source date, publication, category, priority, method, evidence, interpretation, limitations, source locations and implementation-review status. Downloaded PDFs were parsed and their page counts/SHA-256 saved. Core performance claims were checked against experimental conditions and source tables/sections. This is not reproduction or exhaustive close reading of every paper.

The arXiv PDF was fixed as the reviewed version. Since the title, year, and conditions of the official publication and preprint may be different, they are distinguished in the catalog. BibTeX was created based on verified original texts or verified academic bibliographies. The arXiv DOI is not evidence of completion of peer review.

## Contents of public repositories

Public material includes authored research notes, bibliographic metadata, source links, aggregate data-verification reports and reproduction scripts. Original conversations, private absolute paths, account information, raw audio, utterance-level transcript manifests, third-party PDFs and model weights are Git-excluded.

Third-party originals retain their source copyright/use terms. Public links do not imply identical redistribution rights. Store PDFs locally and publish direct links/checksums.

## Edit/recreate

```bash
python3 scripts/build_index.py
python3 scripts/validate_catalog.py
python3 scripts/validate_catalog.py --pdfs
```

`references/catalog.json` is the standard file for organizing this thesis. Individual notes, CSV, BibTeX, and the entire list are recreated from this file. Goals/methodology and data preparation reports are managed as separate written documents and execution results.
