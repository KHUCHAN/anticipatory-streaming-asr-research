#!/usr/bin/env python3
"""Create an explicit padding-compatibility wrapper for the legacy decoder."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.local/a2-export-deps'))
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT / 'experiments/predict_correct_a2_v1/config.json').read_text())
    original = ROOT / cfg['small_model'] / 'decoder-epoch-99-avg-1.onnx'
    destination = ROOT / 'runs/small_zipformer_reproduction_v1/decoder-blank-padding.onnx'
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(original)
    assert len(model.graph.input) == 1 and model.graph.input[0].name == 'y'
    initializers = [(x.name, x.SerializeToString()) for x in model.graph.initializer]
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == 'y':
                node.input[i] = 'legacy_blank_padded_y'
    zero = helper.make_tensor('legacy_blank_id', TensorProto.INT64, [], [0])
    model.graph.initializer.append(zero)
    prefix = [
        helper.make_node('Less', ['y', 'legacy_blank_id'], ['legacy_negative_padding'], name='LegacyPaddingIsNegative'),
        helper.make_node('Where', ['legacy_negative_padding', 'legacy_blank_id', 'y'], ['legacy_blank_padded_y'], name='LegacyPaddingToBlank'),
    ]
    existing = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(prefix + existing)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    assert initializers == [(x.name, x.SerializeToString()) for x in model.graph.initializer[:-1]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    old = ort.InferenceSession(str(original), options, providers=['CPUExecutionProvider'])
    new = ort.InferenceSession(str(destination), options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(78237)
    maximum = 0.
    for context in [np.array([[-1, 0]], np.int64), np.array([[-1, 123]], np.int64), np.zeros((1, 2), np.int64), rng.integers(0, 500, size=(128, 2), dtype=np.int64)]:
        expected = old.run(None, {'y': np.maximum(context, 0)})[0]
        actual = new.run(None, {'y': context})[0]
        maximum = max(maximum, float(np.abs(expected - actual).max()))
    negative = old.run(None, {'y': np.array([[-1, 0]], np.int64)})[0]
    last_id = old.run(None, {'y': np.array([[499, 0]], np.int64)})[0]
    blank = old.run(None, {'y': np.array([[0, 0]], np.int64)})[0]
    record = dict(original_sha256=sha(original), wrapped_sha256=sha(destination),
        wrapper=str(destination.relative_to(ROOT)), original_weights_preserved=True,
        wrapper_vs_explicit_blank_max_difference=maximum,
        original_negative_vs_last_vocab_id_max_difference=float(np.abs(negative - last_id).max()),
        original_negative_vs_blank_max_difference=float(np.abs(negative - blank).max()),
        scope='Legacy 2023 English 20M decoder only; explicit negative padding -> blank input contract; no learned weights changed',
        source_sha256=sha(Path(__file__)))
    assert maximum < 1e-6
    output = HERE / 'artifacts/decoder_wrapper.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
