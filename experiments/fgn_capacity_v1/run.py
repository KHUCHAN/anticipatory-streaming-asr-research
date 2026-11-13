#!/usr/bin/env python3
"""Sequential capacity stages; never interpret a stage limit as overfitting."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'runs/fgn_capacity_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'study.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((HERE / 'config.json').read_text())
    log = (OUT / 'runner.jsonl').open('a', buffering=1)
    def emit(value):
        print(json.dumps(value), flush=True)
        log.write(json.dumps(value) + '\n')
    started = time.time()
    emit(dict(event='start', pid=os.getpid(), started_unix=started, config=cfg))
    guard = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        for epochs in [16, 32, 64, cfg['max_epochs_per_model']]:
            if epochs > cfg['max_epochs_per_model']:
                continue
            for layers in cfg['generator_layers']:
                model_out = OUT / f'layers_{layers}'
                if (model_out / 'overfit.json').exists():
                    continue
                emit(dict(event='training', layers=layers, target_epochs=epochs, elapsed_seconds=time.time()-started))
                subprocess.run([sys.executable, str(HERE / 'study.py'), '--layers', str(layers), '--epochs', str(epochs)], cwd=ROOT, check=True)
                subprocess.run([sys.executable, str(HERE / 'report.py')], cwd=ROOT, check=True)
            found = [layers for layers in cfg['generator_layers'] if (OUT / f'layers_{layers}/overfit.json').exists()]
            emit(dict(event='stage_complete', target_epochs=epochs, overfit_models=found, elapsed_seconds=time.time()-started))
            if found:
                emit(dict(event='done', reason='overfitting observed in capacity comparison', overfit_models=found,
                          elapsed_seconds=time.time()-started, test_used=False))
                return
        emit(dict(event='done', reason='maximum training stage reached without confirmed overfitting',
                  overfit_models=[], elapsed_seconds=time.time()-started, test_used=False))
    except BaseException as exc:
        emit(dict(event='interrupted_or_failed', error=repr(exc), elapsed_seconds=time.time()-started))
        raise
    finally:
        guard.terminate()
        guard.wait()


if __name__ == '__main__':
    main()
