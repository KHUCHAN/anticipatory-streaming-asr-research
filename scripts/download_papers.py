#!/usr/bin/env python3
"""Download original PDFs locally and verify the catalog's SHA-256 hashes.

Requires Python 3.9+ and curl. No datasets, model weights or account data are fetched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', nargs='+', help='Paper IDs, e.g. P01 P03')
    parser.add_argument('--category', help='One category from catalog.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'papers/pdf')
    parser.add_argument('--list', action='store_true', help='List the selection without downloading')
    args = parser.parse_args()
    papers = json.loads((ROOT / 'references/catalog.json').read_text())['papers']
    if args.ids and set(args.ids) - {p['id'] for p in papers}:
        parser.error('Unknown paper ID')
    if args.category and args.category not in {p['category'] for p in papers}:
        parser.error('Unknown category')
    papers = [p for p in papers if (not args.ids or p['id'] in args.ids) and (not args.category or p['category'] == args.category)]
    if args.list:
        for p in papers:
            print(f"{p['id']} {p['title']}\n  {p['pdf_url']}")
        return 0
    if not shutil.which('curl'):
        parser.error('curl is required; install it before downloading')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for p in papers:
        destination = args.output_dir / Path(p['local_pdf_path']).name
        if destination.is_file() and digest(destination) == p['pdf_sha256']:
            print(f"OK   {p['id']} (already downloaded)")
            continue
        fd, name = tempfile.mkstemp(prefix=p['id']+'-', suffix='.part', dir=args.output_dir)
        import os
        os.close(fd)
        temporary = Path(name)
        try:
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--proto', '=https', '--proto-redir', '=https', '--retry', '2', '--max-time', '90', '--output', str(temporary), '--', p['pdf_url']], check=True)
            with temporary.open('rb') as f:
                if not f.read(1024).lstrip().startswith(b'%PDF-'):
                    raise ValueError('response is not a PDF')
            if digest(temporary) != p['pdf_sha256']:
                raise ValueError('SHA-256 mismatch; review the upstream version before changing the catalog')
            temporary.replace(destination)
            print(f"OK   {p['id']} {destination.name}")
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            failed.append(p['id'])
            print(f"FAIL {p['id']}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)
    print(f'{len(papers)-len(failed)}/{len(papers)} PDFs verified.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
