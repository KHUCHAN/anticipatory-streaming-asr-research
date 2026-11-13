#!/usr/bin/env python3
"""Profile the existing batch-1 CPU adapter without changing its inference path."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

from zipformer_features import FrozenZipformer, ROOT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class TimedSession:
    def __init__(self, session):
        self.session = session
        self.seconds = 0.0
        self.calls = 0

    def run(self, *args, **kwargs):
        begin = time.perf_counter()
        result = self.session.run(*args, **kwargs)
        self.seconds += time.perf_counter() - begin
        self.calls += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--utterances-per-split', type=int, default=12)
    parser.add_argument('--splits', nargs='+', default=['dev-clean', 'dev-other'])
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.utterances_per_split < 1 or args.warmup < 0:
        parser.error('Require a positive sample size and nonnegative warmup.')
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve the old measurement.')
    model = FrozenZipformer()
    rows = []
    manifests = {}
    for split in args.splits:
        path = ROOT / f'data/processed/manifests/{split}.jsonl'
        pool = [json.loads(line) for line in path.open()]
        if args.utterances_per_split > len(pool):
            parser.error(f'Sample size exceeds {split} size.')
        rows.extend(random.Random(args.seed).sample(pool, args.utterances_per_split))
        manifests[str(path.relative_to(ROOT))] = sha(path)
    for i in range(args.warmup):
        model.batch([rows[i % len(rows)]])
    for name in ['encoder', 'decoder', 'joiner']:
        setattr(model, name, TimedSession(getattr(model, name)))
    chunks = tokens = 0
    begin = time.perf_counter()
    for row in rows:
        value = model.batch([row])[0]
        chunks += len(value['cuts'])
        tokens += len(value['ids'])
    wall = time.perf_counter() - begin
    components = {name: {'seconds': getattr(model, name).seconds,
                         'calls': getattr(model, name).calls}
                  for name in ['encoder', 'decoder', 'joiner']}
    neural = sum(value['seconds'] for value in components.values())
    for value in components.values():
        value.update(percent_neural=100 * value['seconds'] / neural,
                     ms_per_chunk=1000 * value['seconds'] / chunks)
    config = model.config
    model_dir = ROOT / config['model_dir']
    result = {
        'scope': 'Existing FrozenZipformer, batch 1, CPUExecutionProvider, one intra/inter-op thread; no model or decoder changes. Not a live microphone/UI benchmark.',
        'timing_denominator': 'Neural component shares exclude file loading, fbank, bookkeeping and session initialization. Wall time includes per-utterance loading, fbank and bookkeeping; excludes session initialization and warmup.',
        'seed': args.seed, 'warmup_utterances': args.warmup,
        'utterances': len(rows), 'chunks': chunks, 'native_tokens': tokens,
        'audio_seconds': sum(row['duration_seconds'] for row in rows),
        'wall_seconds': wall, 'neural_seconds': neural, 'components': components,
        'native_neural_speedup_ceiling_if_decoder_and_joiner_free': neural / components['encoder']['seconds'],
        'utterance_ids': [row['utterance_id'] for row in rows],
        'manifest_sha256': manifests,
        'source_sha256': {name: sha(ROOT / name) for name in [
            'scripts/profile_zipformer_compute.py', 'scripts/zipformer_features.py',
            'configs/zipformer_baseline.json']},
        'model_sha256': {name: sha(model_dir / f'{name}-{config["model_suffix"]}')
                         for name in ['encoder', 'decoder', 'joiner']},
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'machine': platform.machine(), 'dependencies': {
                            name: importlib.metadata.version(name) for name in [
                                'onnxruntime', 'kaldi-native-fbank', 'sentencepiece', 'soundfile']}},
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'utterances': len(rows), 'chunks': chunks,
                      'components': components, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
