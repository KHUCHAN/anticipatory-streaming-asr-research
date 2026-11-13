#!/usr/bin/env python3
"""Download the pinned official ONNX archive without changing the AVS env."""
import hashlib
import datetime
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = '639e25b578e9e997131402199419c13a941f8e4e198e2da1ce57dbf5cf401282'
REVISIONS = {
    'icefall_commit': '179b5eef823e8c2c38a7972f4762e22c5dd23973',
    'sherpa_onnx_commit': 'bec3acea719faafa6c86c7022d98f19663b82a76',
    'hf_model': '37cb5606808f3d5e55a3fc73554bdf757d82465a',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    config = json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    directory = ROOT/'checkpoints/baselines'
    directory.mkdir(parents=True,exist_ok=True)
    archive = directory/'zipformer-en-2023-06-26.tar.bz2'
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix+'.part')
        subprocess.run(['curl','-fL','--retry','3',config['model_archive'],'-o',str(temporary)],check=True)
        assert digest(temporary) == ARCHIVE_SHA256, 'Downloaded archive differs from the recorded version'
        temporary.replace(archive)
    assert digest(archive) == ARCHIVE_SHA256, 'Archive differs from the recorded version'
    model = ROOT/config['model_dir']
    expected = [model/(part+'-'+config['model_suffix']) for part in ['encoder','decoder','joiner']]
    expected += [model/'tokens.txt',model/'bpe.model']
    # Extract only when needed; repeated runs do not overwrite local model files.
    if not all(path.exists() for path in expected):
        with tarfile.open(archive) as source:
            source.extractall(directory,filter='data')
    assert all(path.exists() for path in expected)
    output = ROOT/'runs/baseline_zipformer'
    output.mkdir(parents=True,exist_ok=True)
    preparation = dict(model_dir=str(model.relative_to(ROOT)),archive_sha256=ARCHIVE_SHA256,
                       files={str(p.relative_to(ROOT)):digest(p) for p in expected})
    (output/'model_preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    snapshots = ROOT/'.local/baseline_sources'
    snapshots.mkdir(parents=True,exist_ok=True)
    urls = {
        'icefall_RESULTS.md':f"https://raw.githubusercontent.com/k2-fsa/icefall/{REVISIONS['icefall_commit']}/egs/librispeech/ASR/RESULTS.md",
        'online-decode-files.py':f"https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/{REVISIONS['sherpa_onnx_commit']}/python-api-examples/online-decode-files.py",
        'zipformer_models.rst':'https://raw.githubusercontent.com/k2-fsa/sherpa/master/docs/source/onnx/pretrained_models/online-transducer/zipformer-transducer-models.rst',
    }
    for name,url in urls.items():
        path = snapshots/name
        if not path.exists():
            subprocess.run(['curl','-fsSL','--retry','3',url,'-o',str(path)],check=True)
    sources = dict(
        retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        upstream_revisions=REVISIONS, source_urls=urls,
        downloaded_source_sha256={name:digest(snapshots/name) for name in urls},
        limitations=[
            'ONNX conversion is linked to the original checkpoint by upstream documentation; direct numerical comparison with PyTorch has not been run.',
            'Export epoch-99/avg-1 is an alias; original results report epoch-30/avg-8. This is not local epoch-99 training.',
            'The sherpa documentation snapshot uses its rolling master URL; its retrieved contents are identified by SHA-256, not a commit pin.',
            'This streaming reference does not reproduce every ICLR 2024 paper table or latest ASR SOTA.',
        ])
    (output/'upstream_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(preparation,indent=2))


if __name__ == '__main__':
    main()
