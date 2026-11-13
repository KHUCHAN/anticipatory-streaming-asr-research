"""Run the five isolated variants sequentially within the recorded budget."""
import json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/fgn_state_v1'
cfg=json.loads((ROOT/'experiments/fgn_state_v1/config.json').read_text())
budget=json.loads((OUT/'budget.json').read_text())
results=[]
for i,variant in enumerate(cfg['variants']):
    remaining=len(cfg['variants'])-i
    allocation=max(60.,(budget['training_deadline_unix']-time.time())/remaining)
    deadline=min(time.time()+allocation,budget['training_deadline_unix'])
    destination=OUT/variant;destination.mkdir(exist_ok=True)
    record=dict(variant=variant,started_unix=time.time(),deadline_unix=deadline)
    print(json.dumps(dict(event='launch',**record)),flush=True)
    with (destination/'console.log').open('a',buffering=1) as log:
        p=subprocess.run([sys.executable,'-u',str(ROOT/'experiments/fgn_state_v1/train.py'),
                          '--variant',variant,'--deadline',str(deadline)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=p.returncode,ended_unix=time.time())
    results.append(record);(OUT/'training_queue.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(dict(event='finished',**record)),flush=True)
    if p.returncode:raise SystemExit(p.returncode)
print(json.dumps(dict(event='queue_done',seconds=sum(r['ended_unix']-r['started_unix'] for r in results))),flush=True)
