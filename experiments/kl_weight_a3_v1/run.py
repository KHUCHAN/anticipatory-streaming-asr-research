#!/usr/bin/env python3
"""Verify the repaired baseline, replay control, train two fixed KL arms."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, HERE, sha, write

RUN = ROOT / 'runs/kl_weight_a3_v1'


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lease = (RUN / 'runner.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    study = json.loads((HERE / 'config.json').read_text())
    native = json.loads((ROOT / 'experiments/small_zipformer_reproduction_v1/artifacts/native_audit.json').read_text())
    assert native['all_tokens_match'] and native['all_chunks_match'] and native['utterances'] == 800
    repaired = json.loads((ROOT / 'experiments/small_zipformer_reproduction_v1/artifacts/verification.json').read_text())
    assert repaired['status'] == 'verified' and repaired['teacher_tokens_unchanged']
    base = json.loads((ROOT / study['base_config']).read_text())
    original = json.loads((ROOT / 'experiments/predict_correct_a2_v1/artifacts/provenance.json').read_text())
    for name, digest in original['sha256'].items():
        assert sha(ROOT / name) == digest, name
    identical = ['model.py', 'data.py', 'objective.py', 'inference.py', 'onnx_backend.py', 'optimized_inference.py']
    for name in identical:
        assert sha(HERE / name) == sha(ROOT / 'experiments/predict_correct_a2_v1' / name), name
    inputs = list(HERE.glob('*.py')) + [HERE / 'config.json', ROOT / study['base_config'], ROOT / study['initial_checkpoint'], ROOT / study['base_calibration']]
    inputs += [ROOT / 'experiments/predict_correct_a2_v1/evaluate.py', ROOT / 'experiments/predict_correct_a2_v1/artifacts/results.json']
    inputs += [ROOT / 'experiments/small_zipformer_reproduction_v1/artifacts' / name
        for name in ['evaluation.json', 'decoder_wrapper.json', 'native_audit.json', 'verification.json']]
    hashes = dict(original['sha256'])
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
    path = HERE / 'artifacts/provenance.json'
    if path.exists():
        saved = json.loads(path.read_text())
        for name, digest in saved['sha256'].items():
            assert sha(ROOT / name) == digest, name
    else:
        write(path, dict(captured_unix=time.time(), sha256=hashes, exact_A2_implementation_copies=identical,
            base_config=base, study=study, test_used=False, full_binary_caches_rehashed=False))
    log = (RUN / 'runner.jsonl').open('a', buffering=1)
    started = time.time()
    def emit(event, **kw):
        value = dict(event=event, elapsed_seconds=time.time() - started, **kw)
        print(json.dumps(value), flush=True)
        log.write(json.dumps(value) + '\n')
    emit('start', pid=os.getpid(), started_unix=started)
    guard = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        replay = HERE / 'artifacts/control_replay.json'
        if not replay.exists():
            emit('control_replay_start')
            subprocess.run([sys.executable, str(HERE / 'train.py'), '--phase', 'kl', '--bench', '100'], cwd=ROOT, env=dict(os.environ, A3_ARM='control'), check=True)
            logs = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
            old = next(r for r in logs(ROOT / 'runs/predict_correct_a2_v1/kl/log.jsonl') if r['event'] == 'train' and r['step'] == 100)
            new = [r for r in logs(RUN / 'control/kl/benchmark/log.jsonl') if r['event'] == 'train' and r['step'] == 100][-1]
            for key in ['mae', 'kl', 'weight']:
                assert math.isclose(old[key], new[key], rel_tol=2e-4, abs_tol=2e-5), (key, old[key], new[key])
            write(replay, dict(status='verified', updates=100, old=old, replay=new, disposable_weights=True, tolerance=dict(relative=2e-4, absolute=2e-5)))
        emit('control_replay_verified')
        for arm in study['training_arms']:
            emit('training_start', arm=arm, multiplier=study['arms'][arm])
            subprocess.run([sys.executable, str(HERE / 'train.py'), '--phase', 'kl'], cwd=ROOT, env=dict(os.environ, A3_ARM=arm), check=True)
            emit('training_complete', arm=arm)
        emit('evaluation_start')
        subprocess.run([sys.executable, str(HERE / 'evaluate.py')], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(HERE / 'verify.py')], cwd=ROOT, check=True)
        emit('done', test_used=False)
        subprocess.run([sys.executable, str(HERE / 'report.py')], cwd=ROOT, check=True)
    except BaseException as error:
        emit('failed_or_interrupted', reason=repr(error))
        raise
    finally:
        guard.terminate()
        guard.wait()


if __name__ == '__main__':
    main()
