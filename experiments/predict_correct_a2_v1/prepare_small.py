#!/usr/bin/env python3
"""Fetch the official English small streaming recognizer into ignored storage."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
NAME='sherpa-onnx-streaming-zipformer-en-20M-2023-02-17'
URL=f'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{NAME}.tar.bz2'


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    parent=ROOT/'checkpoints/baselines';parent.mkdir(parents=True,exist_ok=True)
    archive=parent/(NAME+'.tar.bz2');model=parent/NAME
    if not archive.exists():
        part=archive.with_suffix('.part')
        subprocess.run(['curl','-fL','--retry','3',URL,'-o',str(part)],check=True)
        part.replace(archive)
    expected=[model/f'{name}-epoch-99-avg-1.onnx' for name in ['encoder','decoder','joiner']]+[model/'tokens.txt']
    if not all(path.exists() for path in expected):
        with tarfile.open(archive) as source:source.extractall(parent,filter='data')
    assert all(path.exists() for path in expected)
    target=Path(__file__).parent/'artifacts/small_model_provenance.json';target.parent.mkdir(exist_ok=True)
    record=dict(url=URL,archive_sha256=sha(archive),model_directory=str(model.relative_to(ROOT)),
        files={str(path.relative_to(ROOT)):sha(path) for path in expected},
        source_model='https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small',
        model_card='https://huggingface.co/csukuangfj/'+NAME,test_used=False)
    if target.exists():assert json.loads(target.read_text())==record
    else:target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
