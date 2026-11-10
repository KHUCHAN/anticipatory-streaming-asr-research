#!/usr/bin/env python3
"""Regenerate Markdown notes, CSV and BibTeX from the curated JSON catalog."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    'anticipatory': '음성 도착 전 예측',
    'decoding': '추측 디코딩과 경로 재사용',
    'streaming': 'ASR 정렬·인코더·스트리밍 기준선',
    'state-space': '상태공간 모델과 ConMamba',
    'weather': 'FGN·GraphCast·WeatherNext',
    'graph-asr': '그래프 기반 음성인식',
    'disambiguation': 'FGN/FGNN 동명 아키텍처 구분',
    'datasets': '학습 데이터셋 원 논문',
}


def bib_escape(text):
    return text.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('"', "'")


def main():
    data = json.loads((ROOT / 'references/catalog.json').read_text())
    papers = data['papers']
    index = ['# 논문 전체 목록', '', f"검증 기준일: {data['verified_at']}. 총 {len(papers)}편. 번호는 연도순이 아니라 주제별 식별자다.", '',
             '논문별 문서에서 **논문이 보고한 내용**과 **이 연구 아이디어에 대한 해석**을 구분한다. 논문 PDF는 로컬에 보관하며 아래 원문 링크에서 다시 받을 수 있다.', '']
    bibs = []
    for category, label in CATEGORIES.items():
        index += [f'## {label}', '', '| ID | 논문 | 원문 연도 / 발표 정보 | 우선순위 |', '|---|---|---|---|']
        for p in papers:
            if p['category'] != category:
                continue
            note = ROOT / p['note_path']
            note.parent.mkdir(parents=True, exist_ok=True)
            links = [f"[원문]({p['paper_url']})", f"[PDF]({p['pdf_url']})"]
            links += [f"[추가 1차 출처 {i+1}]({u})" for i, u in enumerate(p['additional_sources'])]
            code = '\n'.join(f'- [{u}]({u})' for u in p['code_urls']) if p['code_urls'] else '공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.'
            content = f"""# {p['id']}. {p['title']}

| 항목 | 내용 |
|---|---|
| 저자 | {'; '.join(p['authors'])} |
| 확인한 원문 날짜 | {p['source_date']} |
| 발표 정보 | {p['publication']} |
| 분류 | {label} |
| 읽기 우선순위 | {p['priority']} |
| 원문 확인 범위 | {p['reading_depth']} |
| PDF | {p['pdf_pages']}쪽; `{p['local_pdf_path']}` |
| 검증일 | {data['verified_at']} |

{' · '.join(links)}

## 핵심과 방법

{p['summary_ko']}

{p['method_ko']}

## 논문이 보고한 근거

{p['evidence_ko']}

확인 위치: {p['evidence_location']}.

## 연구 아이디어와의 연결 — 정리자의 해석

{p['relevance_ko']}

## 적용할 때의 제한

{p['limitation_ko']}

## 코드와 모델

{code}

{p['code_note']}

## 원문 버전

PDF SHA-256: `{p['pdf_sha256']}`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
"""
            note.write_text(content)
            index.append(f"| {p['id']} | [{p['title']}](../{p['note_path']}) | {p['source_date'][:4]} / {p['publication']} | {p['priority']} |")
            fields = {'title': bib_escape(p['title']), 'author': ' and '.join(p['authors']), 'year': p['source_date'][:4], 'url': p['paper_url']}
            fields.update(p['bib_fields'])
            if p['arxiv_id']:
                fields.update(eprint=p['arxiv_id'], archivePrefix='arXiv')
            body = ',\n'.join(f'  {k} = {{{v}}}' for k, v in fields.items())
            bibs.append(f"@{p['bib_type']}{{{p['id']}_{p['slug'].replace('-', '_')},\n{body}\n}}")
        index.append('')
    (ROOT / 'papers/INDEX.md').write_text('\n'.join(index))
    (ROOT / 'references/references.bib').write_text('% Verified source versions; see catalog.json for publication-status notes.\n\n' + '\n\n'.join(bibs) + '\n')
    fields = ['id', 'category', 'title', 'source_date', 'publication', 'priority', 'paper_url', 'pdf_url', 'note_path', 'local_pdf_path', 'pdf_sha256']
    with (ROOT / 'references/catalog.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(papers)
    print(f'Generated {len(papers)} notes, index, CSV and BibTeX.')


if __name__ == '__main__':
    main()
