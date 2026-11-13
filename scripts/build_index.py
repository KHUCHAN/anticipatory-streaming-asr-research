#!/usr/bin/env python3
"""Regenerate English paper notes, CSV and BibTeX from the curated catalog."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    'anticipatory': 'Prediction before audio arrival',
    'decoding': 'Speculative decoding and path reuse',
    'streaming': 'ASR alignment, encoders and streaming baselines',
    'state-space': 'State-space models and ConMamba',
    'weather': 'FGN, GraphCast and WeatherNext',
    'graph-asr': 'Graph-based speech recognition',
    'disambiguation': 'Disambiguating FGN/FGNN architectures',
    'datasets': 'Original dataset papers',
}


def bib_escape(text):
    return text.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('"', "'")


def main():
    data = json.loads((ROOT / 'references/catalog.json').read_text())
    papers = data['papers']
    index = ['# Paper index', '',
             f"Source verification date: {data['verified_at']}. Initial catalog: {len(papers)} papers. IDs group topics rather than publication years.", '',
             'Notes separate source-reported evidence from project interpretation. Original PDFs are local-only and can be obtained from the linked sources. Later related-work links appear in the research documents.', '']
    bibs = []
    for category, label in CATEGORIES.items():
        index += [f'## {label}', '', '| ID | Paper | Source year / publication | Priority |', '|---|---|---|---|']
        for p in papers:
            if p['category'] != category:
                continue
            note = ROOT / p['note_path']
            note.parent.mkdir(parents=True, exist_ok=True)
            links = [f"[Source]({p['paper_url']})", f"[PDF]({p['pdf_url']})"]
            links += [f"[Additional primary source {i+1}]({url})" for i, url in enumerate(p['additional_sources'])]
            code = '\n'.join(f'- [{url}]({url})' for url in p['code_urls']) if p['code_urls'] else 'An official implementation link was not confirmed in this review. This does not establish that no implementation exists.'
            version_note = p.get('reproduction_note', 'The source version and this hash identify the collected PDF. Training and author experiments were not reproduced in the initial literature review.')
            project_update = '\n\n## Project follow-up\n\n' + p['project_update_en'] if p.get('project_update_en') else ''
            content = f"""# {p['id']}. {p['title']}

| Field | Value |
|---|---|
| Authors | {'; '.join(p['authors'])} |
| Source date | {p['source_date']} |
| Publication | {p['publication']} |
| Category | {label} |
| Reading priority | {p['priority']} |
| Review depth | {p['reading_depth']} |
| PDF | {p['pdf_pages']} pages; `{p['local_pdf_path']}` |
| Source verification date | {data['verified_at']} |

{' · '.join(links)}

## Main idea and method

{p['summary_en']}

{p['method_en']}

## Source-reported evidence

{p['evidence_en']}

Evidence location: {p['evidence_location']}.

## Connection to this project — interpretation

{p['relevance_en']}

## Limitations when applying the result

{p['limitation_en']}

## Code and models

{code}

{p['code_note']}

## Source version

PDF SHA-256: `{p['pdf_sha256']}`. {version_note}{project_update}

[Paper index](../../papers/INDEX.md) · [Literature map](../../docs/literature_map.md)
"""
            note.write_text(content)
            index.append(f"| {p['id']} | [{p['title']}](../{p['note_path']}) | {p['source_date'][:4]} / {p['publication']} | {p['priority']} |")
            fields = {'title': bib_escape(p['title']), 'author': ' and '.join(p['authors']), 'year': p['source_date'][:4], 'url': p['paper_url']}
            fields.update(p['bib_fields'])
            if p['arxiv_id']:
                fields.update(eprint=p['arxiv_id'], archivePrefix='arXiv')
            body = ',\n'.join(f'  {key} = {{{value}}}' for key, value in fields.items())
            bibs.append(f"@{p['bib_type']}{{{p['id']}_{p['slug'].replace('-', '_')},\n{body}\n}}")
        index.append('')
    (ROOT / 'papers/INDEX.md').write_text('\n'.join(index))
    (ROOT / 'references/references.bib').write_text('% Verified source versions; see catalog.json for publication-status notes.\n\n' + '\n\n'.join(bibs) + '\n')
    fields = ['id', 'category', 'title', 'source_date', 'publication', 'priority', 'paper_url', 'pdf_url', 'note_path', 'local_pdf_path', 'pdf_sha256']
    with (ROOT / 'references/catalog.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(papers)
    print(f'Generated {len(papers)} English notes, index, CSV and BibTeX.')


if __name__ == '__main__':
    main()
