"""Separate run namespace; immutable A2 inputs and exact implementation reused."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARM = os.environ.get('A3_ARM', 'control')
OUT = ROOT / 'runs/kl_weight_a3_v1' / ARM
sys.path.insert(0, str(ROOT))


def config():
    study = json.loads((HERE / 'config.json').read_text())
    assert ARM in study['arms']
    cfg = json.loads((ROOT / study['base_config']).read_text())
    cfg.update(a3_arm=ARM, a3_multiplier=study['arms'][ARM],
        initial_checkpoint=study['initial_checkpoint'], base_calibration=study['base_calibration'])
    assert cfg['kl_epochs'] == study['epochs_per_arm']
    return cfg


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=scalar) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


state_module = module('a3_original_bridge', 'experiments/fgn_state_v1/model.py')
features = module('a3_original_features', 'experiments/predict_correct_v1/prepare.py').features


def word_error(reference, hypothesis):
    a = reference.split()
    b = hypothesis.split()
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1], len(a)
