# 로컬 원문 PDF

원문은 `P01_ssr.pdf`처럼 ID별로 이 폴더에 보관한다. PDF 파일은 `.gitignore`에 의해 Git에서 제외된다. 공개 저장소에서는 원문 링크와 다음 스크립트로 동일한 PDF를 받을 수 있다.

```bash
python3 scripts/download_papers.py
python3 scripts/download_papers.py --ids P01 P02 P03 P04 P21
python3 scripts/validate_catalog.py --pdfs
```

각 문서의 원문 링크, 버전, SHA-256은 [카탈로그](../../references/catalog.json)에 기록되어 있다. 접근 가능한 학술 원문을 수집했으며, PDF의 이용 조건은 각 저자·출판사의 조건을 따른다. 모델 가중치와 음성 데이터셋은 수집 대상에 포함하지 않는다.
