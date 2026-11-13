# Local original PDF

Store originals by ID, for example `P01_ssr.pdf`, in this folder. `.gitignore` excludes PDFs; the public repository retains original links and a script to obtain identical versions.

```bash
python3 scripts/download_papers.py
python3 scripts/download_papers.py --ids P01 P02 P03 P04 P21
python3 scripts/validate_catalog.py --pdfs
```

Source links, versions and SHA-256 hashes are in the [catalog](../../references/catalog.json). Accessible scholarly originals were collected under their authors' and publishers' terms. Model weights and speech datasets are outside this PDF collection.
