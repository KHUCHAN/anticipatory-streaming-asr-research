"""Train the pre-test marginals-only control after the original MPS queue."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/fgn_state_v1'
budget=json.loads((OUT/'budget.json').read_text())
while True:
    path=OUT/'training_queue.json'
    if path.exists():
        try:q=json.loads(path.read_text())
        except json.JSONDecodeError:q=[]
        if any(r['returncode'] for r in q):raise RuntimeError('Primary training queue failed')
        if len(q)==5:break
    if time.time()>budget['training_deadline_unix']:raise RuntimeError('No training budget left')
    time.sleep(15)
out=OUT/'crps_only_s37';out.mkdir(exist_ok=True);started=time.time()
print('CRPS-only control started',flush=True)
with (out/'console.log').open('a',buffering=1) as log:
    code=subprocess.run([sys.executable,'-u',str(ROOT/'experiments/fgn_state_v1/train_crps_only.py'),
                         '--variant','crps_only_s37','--deadline',str(budget['training_deadline_unix'])],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
record=dict(returncode=code,started_unix=started,ended_unix=time.time())
tmp=out/'complete.tmp.json';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out/'complete.json')
print(json.dumps(record),flush=True)
raise SystemExit(code)
