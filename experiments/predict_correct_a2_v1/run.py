#!/usr/bin/env python3
"""Run the bounded MAE/KL experiment, native controls and final reporting."""
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
    native=json.loads((HERE/'artifacts/native_checks.json').read_text());assert native['all_match']
    preflight=json.loads((HERE/'artifacts/preflight.json').read_text());assert preflight['tiny_fit']['final_mae']<.8*preflight['tiny_fit']['initial_mae']
    inputs=[HERE/n for n in ['common.py','config.json','model.py','data.py','objective.py','inference.py','train.py']]
    inputs+=[ROOT/'experiments/fgn_state_v1/model.py',ROOT/'experiments/predict_correct_v1/prepare.py',ROOT/cfg['protocol'],ROOT/cfg['bridge'],ROOT/cfg['teacher_config']]
    inputs+=[ROOT/cfg['input_cache']/n for n in ['index.json','stats.json','complete.json']]
    inputs+=[ROOT/cfg['encoder_cache']/n for n in ['stats.json','complete.json']]
    teacher=json.loads((ROOT/cfg['teacher_config']).read_text())
    for folder,suffix in [(ROOT/teacher['model_dir'],teacher['model_suffix']),(ROOT/cfg['small_model'],'epoch-99-avg-1.onnx')]:
        inputs+=[folder/f'{n}-{suffix}' for n in ['encoder','decoder','joiner']]+[folder/'tokens.txt']
    path=HERE/'artifacts/provenance.json'
    if path.exists():
        value=json.loads(path.read_text())
        for name,h in value['sha256'].items():assert sha(ROOT/name)==h,name
    else:write(path,dict(captured_unix=started,sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},test_used=False,full_binary_caches_rehashed=False))
    emit(dict(event='start',pid=os.getpid(),started_unix=started,config=cfg))
    guard=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    try:
        emit(dict(event='training',phase='mae',elapsed_seconds=time.time()-started))
        subprocess.run([sys.executable,str(HERE/'train.py'),'--phase','mae'],cwd=ROOT,check=True)
        import torch
        checkpoint=torch.load(OUT/'mae/best.pt',map_location='cpu',weights_only=False)
        representation_ok=checkpoint['record']['validation']['mae']<checkpoint['record']['validation']['constant_mae']
        emit(dict(event='representation_gate',passed=representation_ok,validation=checkpoint['record']['validation'],elapsed_seconds=time.time()-started))
        del checkpoint
        if representation_ok:
            emit(dict(event='training',phase='kl',elapsed_seconds=time.time()-started))
            subprocess.run([sys.executable,str(HERE/'train.py'),'--phase','kl'],cwd=ROOT,check=True)
        emit(dict(event='evaluating',elapsed_seconds=time.time()-started))
        subprocess.run([sys.executable,str(HERE/'evaluate.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(HERE/'verify.py')],cwd=ROOT,check=True)
        emit(dict(event='done',elapsed_seconds=time.time()-started,test_used=False))
        subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
    except BaseException as error:
        emit(dict(event='failed_or_interrupted',reason=repr(error),elapsed_seconds=time.time()-started));raise
    finally:guard.terminate();guard.wait()


if __name__=='__main__':main()
