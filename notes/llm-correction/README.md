# LLM candidate selection, correction and fusion

Added 2026-11-16: ten English paper notes, P51-P60. Read the [comparison review](../../docs/prior_work_llm_correction_2026-11-16.md) first. This collection fills the LLM gap in the earlier future-context review; no author models or project experiments were run.

| Read | Paper | Focus |
|---|---|---|
| 1 | [P51 N-best T5](P51-nbest-t5.md), [P52 ChatGPT correction](P52-chatgpt-asr-correction.md) | Fixed-list selection, constrained decoding and free-generation controls |
| 2 | [P53 HyPoradise](P53-hyporadise.md) | Open correction benchmark and strong-baseline failure cases |
| 3 | [P55 ProGRes](P55-progres.md), [P56 Task-Activating Prompting](P56-task-activating.md) | Generate/expand/reselect and prompted correction |
| 4 | [P54 ClozeGER](P54-clozeger.md), [P60 UADF](P60-late-acoustic-fusion.md) | Candidate disagreements, option calibration and acoustic evidence |
| 5 | [P57 Delayed Fusion](P57-delayed-fusion.md), [P58 SALSA](P58-salsa.md) | LLM integration cost, timing schedules and vocabulary mismatch |
| 6 | [P59 StreamCorrect](P59-streamcorrect.md) | Direct recent streaming correction with measured latency overhead |

Do not equate offline WER improvement, faster LLM scoring, causal audio streaming and earlier stable display. Runtime definitions and missing timing evidence are recorded per paper. Existing [P01 SSR](../anticipatory/P01-ssr.md) concerns unheard-text prediction; [P34-P50](../predict-refine/README.md) cover synthetic future context, early emission and provisional display.

Original PDFs are local-only under `papers/pdf/`. The [catalog](../../references/catalog.json), [BibTeX](../../references/references.bib), and [evidence manifest](../../references/reviews/llm_correction_2026-11-16.json) preserve versions and source locations. The catalog now has 60 entries.

```bash
python3 scripts/download_papers.py --category llm-correction
python3 scripts/build_index.py
python3 scripts/validate_catalog.py --pdfs
```

Edit catalog metadata before regenerating individual notes. Retrieval and successful validation do not constitute reproduction of author results.
