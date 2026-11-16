# Future context, provisional display and emission latency

Added 2026-11-16: 17 English paper notes, P34-P50. Their source PDFs are available locally in `papers/pdf/`; exact filenames, source links, versions and SHA-256 values appear in each note and the catalog. Original PDFs are excluded from the public repository.

Start with the [comparison and novelty review](../../docs/prior_work_predict_refine_2026-11-16.md). It distinguishes provisional display, final emission, future-context waiting and actual encoder computation. These papers were inspected, not reproduced.

## Reading order

| Order | Papers | Purpose |
|---|---|---|
| 1 | [P34 ZeroPrompt](P34-zeroprompt.md) | Training-free preview, protected final path, hidden-layer prompts, extra runtime |
| 2 | [P35 CUSIDE](P35-cuside.md), [P36 CUSIDE-T](P36-cuside-t.md) | Learned future features jointly trained with ASR |
| 3 | [P39 TSCA/DRC](P39-tsca.md), [P40 CUSIDE-array](P40-cuside-array.md) | Refinement with arrived audio and multi-channel robustness |
| 4 | [P37 FastEmit](P37-fastemit.md), [P38 TrimTail](P38-trimtail.md), [P41 Delay penalty](P41-delay-penalized.md), [P42 Minimum latency](P42-minimum-latency.md) | Train for earlier token emission before adding a forecast |
| 5 | [P49 Fast-U2++](P49-fast-u2pp.md), [P43 Partial Rewriting](P43-partial-rewriting.md), [P44 Stability](P44-streaming-stability.md), [P45 U2 Whisper](P45-whisper-two-pass.md) | Fast/slow paths, display merging and repeated-revision metrics |
| 6 | [P47 Back Chunks](P47-back-chunks.md), [P48 Delayed-KD](P48-delayed-kd.md), [P46 SENS-ASR](P46-sens-asr.md) | Future-aware, alignment-aware and semantic distillation |
| 7 | [P50 Selective Lookahead](P50-selective-lookahead.md) | Recent adaptive waiting policy and median/tail/runtime trade-offs |

For predicting unheard text, also read existing [P01 SSR](../anticipatory/P01-ssr.md) and [P02 Speculative Listening](../anticipatory/P02-speculative-listening.md). Our existing [paper index](../../papers/INDEX.md) now has 50 entries, retaining the original 33 IDs and review dates.

## Reuse

```bash
python3 scripts/download_papers.py --category predict-refine
python3 scripts/build_index.py
python3 scripts/validate_catalog.py --pdfs
```

Individual notes are generated from [catalog.json](../../references/catalog.json). Edit that source when correcting a note. Keep source-reported results separate from project interpretation; exact author checkpoints and local reproductions have different status.
