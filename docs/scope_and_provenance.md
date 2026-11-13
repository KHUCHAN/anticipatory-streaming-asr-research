# Management of collection scope and sources

Initial verification date: 2026-11-10. The 33 records cover papers explicitly mentioned in two supplied research conversations and directly connected foundational work. This is not an exhaustive worldwide systematic review.

Status update: 2026-11-11. The 33 papers are the initial catalog. Later sources on bounded-context prediction, confidence, Stateful Conformer and multi-stage display are linked separately in the [literature map](literature_map.md), [review](results_review_2026-11-11.md) and [960h plan](librispeech960_benchmark_plan.md), not added to that count. Start with [current conclusions](conclusion_and_next_steps.md).

The source collection has 2 anticipatory papers, 6 speculative-decoding papers, 8 alignment/encoder/streaming papers, 5 state-space papers, 4 weather-FGN papers, 3 graph-ASR papers, 2 acronym-disambiguation papers and 3 dataset papers.

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
