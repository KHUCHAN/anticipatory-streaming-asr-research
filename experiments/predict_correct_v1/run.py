#!/usr/bin/env python3
"""Train the three ablations sequentially, then evaluate and report all costs."""
import fcntl
import json
import os
import subprocess
import sys
import time

from common import ROOT,HERE,OUT,config,sha,write


def main():
    cfg=config();OUT.mkdir(parents=True,exist_ok=True)
    lease=(OUT/'runner.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    log=(OUT/'runner.jsonl').open('a',buffering=1);started=time.time()
    def emit(value):print(json.dumps(value),flush=True);log.write(json.dumps(value)+'\n')
    provenance_path=HERE/'artifacts/provenance.json'
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text())
        for name,h in provenance['sha256'].items():assert sha(ROOT/name)==h,name
    else:
        paths=[HERE/n for n in ['config.json','common.py','model.py','data.py','objective.py','train.py','inference.py','prepare.py']]
        paths += [ROOT/cfg['predictor_checkpoint'],ROOT/cfg['bridge'],ROOT/cfg['encoder_cache']/'index.json',ROOT/cfg['encoder_cache']/'stats.json',ROOT/'experiments/fgn_state_v1/model.py',OUT/'cache/index.json',OUT/'cache/stats.json',OUT/'protocol.json']
        write(provenance_path,dict(captured_unix=started,sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
            cached_feature_bytes=(OUT/'cache/fbank.f16').stat().st_size,teacher_cache_bytes=(ROOT/cfg['encoder_cache']/'encoder.f16').stat().st_size,
            full_binary_caches_rehashed=False,test_used=False))
    emit(dict(event='start',pid=os.getpid(),started_unix=started,config=cfg))
    guard=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    try:
        for variant in cfg['variants']:
            emit(dict(event='training',variant=variant,elapsed_seconds=time.time()-started))
            subprocess.run([sys.executable,str(HERE/'train.py'),'--variant',variant],cwd=ROOT,check=True)
            subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
        emit(dict(event='evaluating',elapsed_seconds=time.time()-started))
        subprocess.run([sys.executable,str(HERE/'finalize.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(HERE/'verify.py')],cwd=ROOT,check=True)
        emit(dict(event='done',elapsed_seconds=time.time()-started,test_used=False))
        subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(HERE/'analyze.py')],cwd=ROOT,check=True)
    except BaseException as exc:
        emit(dict(event='interrupted_or_failed',error=repr(exc),elapsed_seconds=time.time()-started));raise
    finally:guard.terminate();guard.wait()


if __name__=='__main__':main()
