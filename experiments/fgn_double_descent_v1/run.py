#!/usr/bin/env python3
"""Run a finite epoch-wise double-descent experiment, preserving prior evidence."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from train import base, configuration


def main():
    extension, cfg = configuration()
    assert sorted(extension['execution_order']) == sorted(extension['generator_layers'])
    out = ROOT / extension['output']
    out.mkdir(parents=True, exist_ok=True)
    lease = (out / 'runner.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logpath = out / 'runner.jsonl'
    if logpath.exists():
        rows = [json.loads(line) for line in logpath.read_text().splitlines()]
        if rows[-1]['event'] == 'done':
            print('Experiment already completed; retained its completion record.')
            return
    provenance_path = HERE / 'artifacts/provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        for name, expected in provenance['sha256'].items():
            assert base.sha(ROOT/name) == expected, name
    else:
        prior = json.loads((ROOT / extension['base_experiment'] / 'artifacts/provenance_initial.json').read_text())
        names = set(prior['source_sha256']) | set(prior['input_sha256'])
        names |= {str((HERE/name).relative_to(ROOT)) for name in ['config.json','train.py','full_train.py']}
        for layers in extension['generator_layers']:
            names |= {str((ROOT/extension['base_runs']/f'layers_{layers}'/name).relative_to(ROOT))
                      for name in ['best.pt','last.pt','log.jsonl']}
        provenance = dict(captured_unix=time.time(), sha256={name:base.sha(ROOT/name) for name in sorted(names)},
                          training_scope=extension['scope'], feature_cache_fully_rehashed=False, test_used=False)
        base.write(provenance_path, provenance)
    base.write(HERE / 'artifacts/settings.json', json.loads((ROOT/'experiments/fgn_capacity_long_v1/artifacts/settings.json').read_text()))
    for layers in extension['generator_layers']:
        dest = out/f'layers_{layers}'
        dest.mkdir(exist_ok=True)
        for name in ['log.jsonl','overfit.json']:
            if not (dest/name).exists():
                source = ROOT/extension['base_runs']/f'layers_{layers}'/name
                shutil.copy2(source,dest/name)
    log = logpath.open('a',buffering=1)
    started = time.time()
    def emit(value):
        print(json.dumps(value),flush=True)
        log.write(json.dumps(value)+'\n')
    emit(dict(event='start',pid=os.getpid(),started_unix=started,config=extension))
    guard = subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    try:
        for layers in extension['execution_order']:
            emit(dict(event='training',layers=layers,target_epoch=extension['target_epoch'],elapsed_seconds=time.time()-started))
            subprocess.run([sys.executable,str(HERE/'train.py'),'--layers',str(layers)],cwd=ROOT,check=True)
            subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
        emit(dict(event='training_complete',elapsed_seconds=time.time()-started))
        subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(HERE/'verify.py')],cwd=ROOT,check=True)
        emit(dict(event='done',target_epoch=extension['target_epoch'],elapsed_seconds=time.time()-started,
                  reason='All declared additional epochs and checkpoint verification completed; no early stopping.',test_used=False))
        subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(HERE/'analyze.py')],cwd=ROOT,check=True)
    except BaseException as exc:
        emit(dict(event='interrupted_or_failed',error=repr(exc),elapsed_seconds=time.time()-started))
        raise
    finally:
        guard.terminate()
        guard.wait()


if __name__ == '__main__':
    main()
