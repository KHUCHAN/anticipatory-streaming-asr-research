#!/usr/bin/env python3
"""Run the additional fixed-budget epochs sequentially and preserve provenance."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    lease = (out / 'runner.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logpath = out / 'runner.jsonl'
    if logpath.exists():
        existing = [json.loads(line) for line in logpath.read_text().splitlines()]
        if existing[-1]['event'] == 'done':
            print('Continuation already completed; preserved the original completion log.')
            return
    provenance_path = HERE / 'artifacts/provenance.json'
    if not provenance_path.exists():
        prior = json.loads((ROOT / cfg['base_experiment'] / 'artifacts/provenance_initial.json').read_text())
        names = set(prior['source_sha256']) | set(prior['input_sha256'])
        names |= {str((HERE / name).relative_to(ROOT)) for name in ['train.py', 'config.json']}
        for layers in cfg['generator_layers']:
            names |= {str((ROOT / cfg['base_runs'] / f'layers_{layers}' / name).relative_to(ROOT))
                      for name in ['best.pt', 'last.pt', 'log.jsonl', 'overfit.json']}
        provenance = dict(captured_unix=time.time(), sha256={name: sha(ROOT/name) for name in sorted(names)},
                          scope='Before extended updates. Original experiment inputs and checkpoints are read-only; feature-cache content is not fully rehashed.',
                          test_used=False)
        provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    else:
        provenance = json.loads(provenance_path.read_text())
        for name, expected in provenance['sha256'].items():
            assert sha(ROOT/name) == expected, name
    # The extended figure retains epochs 0--11 for every capacity even while
    # the sequential runner is still training its first model.
    for layers in cfg['generator_layers']:
        model_out = out / f'layers_{layers}'
        model_out.mkdir(exist_ok=True)
        for name in ['log.jsonl', 'overfit.json']:
            if not (model_out / name).exists():
                shutil.copy2(ROOT / cfg['base_runs'] / f'layers_{layers}' / name, model_out / name)
    log = logpath.open('a', buffering=1)
    started = time.time()
    def emit(value):
        print(json.dumps(value), flush=True)
        log.write(json.dumps(value)+'\n')
    emit(dict(event='start', pid=os.getpid(), started_unix=started, config=cfg))
    guard = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        for layers in cfg['generator_layers']:
            emit(dict(event='training', layers=layers, target_epoch=cfg['target_epoch'], elapsed_seconds=time.time()-started))
            subprocess.run([sys.executable, str(HERE / 'train.py'), '--layers', str(layers)], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(HERE / 'report.py')], cwd=ROOT, check=True)
        emit(dict(event='done', target_epoch=cfg['target_epoch'],
                  reason='Completed requested additional epochs; overfit observations did not stop training.',
                  elapsed_seconds=time.time()-started, test_used=False))
    except BaseException as exc:
        emit(dict(event='interrupted_or_failed', error=repr(exc), elapsed_seconds=time.time()-started))
        raise
    finally:
        guard.terminate()
        guard.wait()


if __name__ == '__main__':
    main()
