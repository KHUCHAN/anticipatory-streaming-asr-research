#!/usr/bin/env python3
"""One isolated CPU replay; all model starts/export checks are outside timers."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from common import ROOT, HERE, OUT, config, sha, write
from data import datasets
import inference
from inference import Inference, Decoder
from model import A2
from onnx_backend import export
from optimized_inference import OptimizedInference


class Legacy20MDecoder(Decoder):
    def __init__(self, graph):
        self.graph = graph
        self.ids = []
        self.context = np.zeros((1, graph.context), np.int64)
        self.pn = graph.decoder.run(None, {'y': self.context})[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=['teacher', 'small', 'control', 'kl4', 'kl16'], required=True)
    parser.add_argument('--repeat', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = config()
    _, dev, evaluation = datasets(cfg)
    rows = sorted(evaluation, key=lambda r: (r['split'], r['utterance_id']))
    root = ROOT / 'runs/kl_weight_a3_v1'
    path = root / 'evaluation' / f'{args.name}_repeat{args.repeat}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved['cfg'] == cfg and len(saved['records']) == 800
        print(json.dumps(dict(event='evaluation_already_saved', name=args.name, repeat=args.repeat)))
        return
    if args.name in ['teacher', 'small']:
        inference.Decoder = Legacy20MDecoder if args.name == 'small' else Decoder
        infer = Inference(cfg, small=args.name == 'small')
        digest = 'original public weights; explicit legacy padding only for 20M'
    else:
        checkpoint = ROOT / ('runs/predict_correct_a2_v1/kl/best.pt' if args.name == 'control'
            else f'runs/kl_weight_a3_v1/{args.name}/kl/best.pt')
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = A2(cfg).eval()
        model.load_state_dict(ck['model'])
        digest = sha(checkpoint)
        directory, metadata = export(model, cfg, 'kl', digest)
        write(HERE / 'artifacts' / f'onnx_{args.name}.json', dict(directory=str(directory.relative_to(ROOT)), **metadata))
        infer = OptimizedInference(cfg, directory)
        check_path = HERE / 'artifacts' / f'backend_{args.name}.json'
        if not check_path.exists():
            reference = Inference(cfg, model)
            checks = []
            for split in ['dev-clean', 'dev-other']:
                for row in sorted([r for r in dev.rows if r['split'] == split], key=lambda r: r['utterance_id'])[:4]:
                    a = reference.utterance(row)
                    b = infer.utterance(row)
                    checks.append(dict(utterance_id=row['utterance_id'], tokens_match=a['token_ids'] == b['token_ids']))
            assert all(r['tokens_match'] for r in checks)
            write(check_path, dict(checks=checks, all_match=True, model_sha256=digest))
        del model, ck
    infer.utterance(dev.rows[0])
    infer.utterance(dev.rows[1])
    records = []
    for i, row in enumerate(rows, 1):
        records.append(infer.utterance(row))
        if i % 100 == 0:
            print(json.dumps(dict(event='evaluation_progress', name=args.name, repeat=args.repeat, done=i, total=len(rows))), flush=True)
    write(path, dict(cfg=cfg, name=args.name, model_sha256=digest, repeat=args.repeat, records=records))
    print(json.dumps(dict(event='evaluation_saved', name=args.name, repeat=args.repeat)), flush=True)


if __name__ == '__main__':
    main()
