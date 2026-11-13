#!/usr/bin/env python3
"""Continue OUR experiment after the already running training/cache jobs.

This is a current-task runner, not a scheduled automation. Child failures
stop the pipeline. Logs and completion markers remain local.
"""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/zipformer_ours'


def wait_for(paths,label):
    last=0
    while not all(p.exists() for p in paths):
        if time.time()-last>300:
            print(json.dumps(dict(waiting=label,utc_unix=time.time())),flush=True);last=time.time()
        time.sleep(15)


def run(name,command):
    marker=OUT/(name+'_complete.json')
    if marker.exists():return
    started=time.time();print('START',name,flush=True)
    with (OUT/(name+'.log')).open('a',buffering=1) as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed: inspect {OUT/(name+".log")}')
    marker.write_text(json.dumps(dict(stage=name,seconds=time.time()-started,finished_unix=time.time()),indent=2)+'\n')
    print('DONE',name,round(time.time()-started,1),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    wait_for([OUT/'training_complete.json',OUT/'prepared_dev-clean.json',OUT/'prepared_dev-other.json'],'two trained forecasters and dev cache')
    run('collect_dev',['.venv/bin/python','scripts/collect_zipformer_forecasts.py','--part','dev'])
    run('calibrate',['.venv/bin/python','scripts/analyze_zipformer_forecasts.py','calibrate'])
    wait_for([OUT/'prepared_test-clean.json',OUT/'prepared_test-other.json'],'full test cache')
    run('collect_test',['.venv/bin/python','scripts/collect_zipformer_forecasts.py','--part','test'])
    run('evaluate',['.venv/bin/python','scripts/analyze_zipformer_forecasts.py','evaluate'])
    run('report',['.venv/bin/python','scripts/report_zipformer_ours.py'])
    (OUT/'complete.json').write_text(json.dumps(dict(status='our method trained and evaluated',finished_unix=time.time()),indent=2)+'\n')
    print('OUR METHOD EXPERIMENT COMPLETE',flush=True)


if __name__=='__main__':main()
