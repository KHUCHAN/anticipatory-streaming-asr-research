"""Wait for our training queue, then run dev calibration before test evaluation."""
import json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/fgn_state_v1'
budget=json.loads((OUT/'budget.json').read_text());cfg=json.loads((ROOT/'experiments/fgn_state_v1/config.json').read_text())
while True:
    q=OUT/'training_queue.json'
    if q.exists():
        try:records=json.loads(q.read_text())
        except json.JSONDecodeError:records=[]
        if any(r['returncode'] for r in records):raise RuntimeError('Training queue failed')
        extra=OUT/'crps_only_s37/complete.json'
        if len(records)==len(cfg['variants']) and extra.exists():
            assert json.loads(extra.read_text())['returncode']==0,'CRPS-only training failed'
            break
    if time.time()>budget['deadline_unix']:raise RuntimeError('Six-hour wall budget expired while waiting')
    time.sleep(15)

stages=[('runtime-check',['check_runtime.py']),('marginals-only-check',['check_crps_only.py']),
        ('collection-smoke',['collect.py','--part','dev','--limit','2','--workers','1','--smoke']),
        ('dev-collection',['collect.py','--part','dev','--workers','6']),
        ('calibration',['analyze.py','calibrate']),
        ('test-collection',['collect.py','--part','test','--workers','6']),
        ('paired-evaluation',['analyze.py','evaluate']),
        ('trajectory-diagnostics',['diagnose.py'])]
status=[]
for name,args in stages:
    remaining=budget['deadline_unix']-time.time()
    if remaining<30:raise RuntimeError('Six-hour wall budget exhausted before '+name)
    started=time.time();print(json.dumps(dict(event='evaluation_start',stage=name)),flush=True)
    with (OUT/('pipeline-'+name+'.log')).open('a',buffering=1) as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiments/fgn_state_v1'/args[0]),*args[1:]],
                                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);process.wait();raise RuntimeError('Six-hour wall budget reached during '+name)
    row=dict(stage=name,returncode=code,seconds=time.time()-started,ended_unix=time.time());status.append(row)
    (OUT/'evaluation_queue.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(row),flush=True)
    if code:
        print((OUT/('pipeline-'+name+'.log')).read_text()[-4000:],flush=True);raise SystemExit(code)
print('ALL EVALUATION STAGES COMPLETE',flush=True)
