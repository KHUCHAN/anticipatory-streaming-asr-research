# Local original PDF

Store originals by ID, for example `P01_ssr.pdf`, in this folder. `.gitignore` excludes PDFs; the public repository retains original links and a script to obtain identical versions.

```bash
python3 scripts/download_papers.py
python3 scripts/download_papers.py --ids P01 P02 P03 P04 P21
python3 scripts/download_papers.py --category predict-refine
python3 scripts/download_papers.py --category llm-correction
python3 scripts/validate_catalog.py --pdfs
```

Source links, versions and SHA-256 hashes are in the [catalog](../../references/catalog.json). Accessible scholarly originals were collected under their authors' and publishers' terms. Model weights and speech datasets are outside this PDF collection.

The 2026-11-16 future-context extension adds P34-P50 (17 PDFs), covering ZeroPrompt, CUSIDE/CUSIDE-T, FastEmit and related work. The separate LLM correction extension adds P51-P60 (ten PDFs). All 60 catalog originals are identified separately from their public notes. See the [future-context review](../../docs/prior_work_predict_refine_2026-11-16.md), [LLM review](../../docs/prior_work_llm_correction_2026-11-16.md), and their linked folder guides.
