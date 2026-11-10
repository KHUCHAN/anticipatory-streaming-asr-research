#!/usr/bin/env python3
"""Check catalog integrity, generated notes, local links and optional PDF hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdfs', action='store_true', help='Require all original PDFs and matching hashes')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'references/catalog.json').read_text())
    papers = catalog['papers']
    errors = []
    ids = [p['id'] for p in papers]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate IDs')
    for p in papers:
        for key in ['title','authors','source_date','publication','paper_url','pdf_url','summary_ko','method_ko','evidence_ko','evidence_location','relevance_ko','limitation_ko','note_path','pdf_sha256']:
            if not p.get(key):
                errors.append(f"{p['id']}: missing {key}")
        if not re.fullmatch(r'P\d{2}', p['id']):
            errors.append(f"Invalid ID {p['id']}")
        if not re.fullmatch(r'[a-f0-9]{64}', p['pdf_sha256']):
            errors.append(f"{p['id']}: invalid PDF hash")
        if not all(p[k].startswith('https://') for k in ['paper_url','pdf_url']):
            errors.append(f"{p['id']}: source is not HTTPS")
        if not (ROOT / p['note_path']).is_file():
            errors.append(f"{p['id']}: missing note")
        if args.pdfs:
            pdf = ROOT / p['local_pdf_path']
            if not pdf.is_file() or hashlib.sha256(pdf.read_bytes()).hexdigest() != p['pdf_sha256']:
                errors.append(f"{p['id']}: missing or mismatched PDF")
    for path in ROOT.rglob('*.md'):
        if any(x in path.relative_to(ROOT).parts for x in ['.git','.local','tmp','.venv']):
            continue
        body = path.read_text()
        if '/Users/' in body or 'gho_' in body or 'sk-proj-' in body:
            errors.append(f'{path.relative_to(ROOT)}: unexpected private path or credential marker')
        for link in re.findall(r'\]\(([^)]+)\)', body):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',link) or link.startswith('#'):
                continue
            target = unquote(link.split('#',1)[0].strip('<>'))
            if target and not (path.parent / target).exists():
                errors.append(f'{path.relative_to(ROOT)}: broken link {link}')
    bib = (ROOT / 'references/references.bib').read_text()
    if len(re.findall(r'^@',bib,re.M)) != len(papers):
        errors.append('BibTeX/catalog count mismatch')
    for error in errors:
        print('ERROR:',error)
    print(f'{len(papers)} papers; {len(errors)} errors; PDF verification={args.pdfs}.')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
