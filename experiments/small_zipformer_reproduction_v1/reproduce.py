#!/usr/bin/env python3
"""Repair the legacy 20M decoder padding contract without changing its weights.

The source icefall ONNX example starts with [blank, blank]. The current
sherpa-onnx native decoder starts with [-1, blank]. This legacy decoder export
does not implement the newer negative-padding convention. Preserve the A2
sources and original graph; make the contract explicit in a new study.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'runs/small_zipformer_reproduction_v1'
sys.path.insert(0, str(ROOT / 'experiments/predict_correct_a2_v1'))
from common import config, sha, write
from inference import Decoder, Inference


class LegacyDecoder(Decoder):
    def __init__(self, graph, convention='blank'):
        self.graph = graph
        self.ids = []
        self.context = np.zeros((1, graph.context), np.int64)
        if convention == 'negative':
            self.context[:, :-1] = -1
        self.pn = graph.decoder.run(None, {'y': self.context})[0]


def make_corrected_inference(name):
    # The original Inference implementation resolves Decoder from its module.
    # Override it only in this separate process/study, never in the A2 source.
    import inference
    inference.Decoder = LegacyDecoder if name == 'small' else Decoder
    return Inference(config(), small=name == 'small')


def aggregate(records):
    result = {}
    for split in ['dev-clean', 'dev-other']:
        own = [r for r in records if r['split'] == split]
        words = sum(r['reference_words'] for r in own)
        errors = sum(r['errors'] for r in own)
        seconds = sum(r['total_seconds'] for r in own)
        audio = sum(r['audio_seconds'] for r in own)
        chunks = sum(r['chunks'] for r in own)
        result[split] = dict(utterances=len(own), errors=errors,
            reference_words=words, wer_percent=100 * errors / words,
            total_seconds=seconds, audio_seconds=audio, chunks=chunks,
            rtf=seconds / audio, total_ms_per_chunk=1000 * seconds / chunks,
            fbank_ms_per_chunk=1000 * sum(r['feature_seconds'] for r in own) / chunks,
            encoder_ms_per_chunk=1000 * sum(r['components']['audio'] for r in own) / chunks,
            decoder_ms_per_chunk=1000 * sum(r['components']['decoder'] for r in own) / chunks)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['selection', 'evaluation'], required=True)
    parser.add_argument('--models', nargs='+', choices=['small', 'teacher'], default=['small'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config()
    roles = json.loads((ROOT / cfg['protocol']).read_text())
    ids = set(roles[args.role + '_ids'])
    rows = sorted([r for r in json.loads((ROOT / cfg['input_cache'] / 'index.json').read_text())
        if r['utterance_id'] in ids], key=lambda r: (r['split'], r['utterance_id']))
    repeats = {name: [] for name in args.models}
    for repeat in range(2):
        for name in args.models if repeat == 0 else args.models[::-1]:
            infer = make_corrected_inference(name)
            infer.utterance(rows[0])  # excluded warmup
            records = []
            for i, row in enumerate(rows, 1):
                records.append(infer.utterance(row))
                if i % 100 == 0:
                    print(json.dumps(dict(model=name, role=args.role, repeat=repeat, done=i, total=len(rows))), flush=True)
            write(OUT / f'{name}_{args.role}_repeat{repeat}.json', dict(records=records))
            repeats[name].append(records)
    signature = lambda records: [(r['utterance_id'], r['token_ids'], r['hypothesis'], r['errors']) for r in records]
    result = {}
    for name, pair in repeats.items():
        assert signature(pair[0]) == signature(pair[1])
        scores = [aggregate(r) for r in pair]
        result[name] = json.loads(json.dumps(scores[0]))
        for split in result[name]:
            for key in ['rtf', 'total_seconds', 'total_ms_per_chunk', 'fbank_ms_per_chunk', 'encoder_ms_per_chunk', 'decoder_ms_per_chunk']:
                result[name][split][key] = sum(s[split][key] for s in scores) / len(scores)
            result[name][split]['timing_repetitions'] = [s[split]['total_ms_per_chunk'] for s in scores]
    record = dict(status='completed', role=args.role, scores=result,
        change='Only legacy decoder initial context [-1,0] -> [0,0]; original ONNX weights and encoder states unchanged',
        source_sha256=sha(Path(__file__)), all_repeated_transcripts_match=True,
        frozen_A2_sources_changed=False, test_used=False)
    write(HERE / 'artifacts' / f'{args.role}.json', record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
