#!/usr/bin/env python3
"""Check public text, English prose, local Markdown links and excluded assets."""
import ast
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.md', '.py', '.json', '.csv', '.bib', '.txt', '.svg'}
HANGUL = re.compile('[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]')


def main():
    names = set(subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT).decode().split('\0')) - {''}
    errors = []
    python_files = markdown_files = 0
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():
            continue
        if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.onnx', '.npz', '.flac', '.wav', '.pdf'}:
            errors.append(f'{name}: model/data/PDF asset is not a public source file')
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        body = path.read_text(encoding='utf-8-sig')
        if path.suffix == '.json':
            language_text = json.dumps(json.loads(body), ensure_ascii=False)
        else:
            language_text = body
        if HANGUL.search(language_text):
            errors.append(f'{name}: non-English Korean prose or path remains')
        if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', body):
            errors.append(f'{name}: private-key marker')
        if re.search(r'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|sk-proj-[A-Za-z0-9_-]{30,})', body):
            errors.append(f'{name}: credential marker')
        if path.suffix == '.py':
            ast.parse(body, filename=name)
            python_files += 1
        if path.suffix.lower() != '.md':
            continue
        markdown_files += 1
        if '/Users/' in body:
            errors.append(f'{name}: private absolute path in public Markdown')
        for link in re.findall(r'\]\(([^)]+)\)', body):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', link) or link.startswith('#'):
                continue
            target = unquote(link.split('#', 1)[0].strip('<>'))
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            try:
                public_name = str(resolved.relative_to(ROOT))
            except ValueError:
                errors.append(f'{name}: link leaves repository: {link}')
                continue
            if resolved.is_dir():
                if not any(p.startswith(public_name + '/') for p in names):
                    errors.append(f'{name}: directory is local-only: {link}')
            elif public_name not in names:
                errors.append(f'{name}: target is missing from public files: {link}')
    for error in errors:
        print('ERROR:', error)
    print(json.dumps({'public_files': len(names), 'python_files_parsed': python_files,
                      'markdown_files_checked': markdown_files, 'errors': len(errors)}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
