#!/usr/bin/env python3
"""Two extra paired seeds; one MPS training job and at most two CPU decoders."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/followups/seeds'


def run(name, args):
    log = OUT/(name+'.console.log')
    print(json.dumps({'event':'start','job':name,'utc':datetime.now(timezone.utc).isoformat()}, ensure_ascii=False), flush=True)
    start = time.monotonic()
    with log.open('w') as stream:
        subprocess.run([sys.executable, *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'event':'done','job':name,'seconds':round(time.monotonic()-start,2)}, ensure_ascii=False), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'additional_seeds':[3,4], 'existing_seed':2, 'epochs':15,
            'init':'same stage1 decoder for both models', 'decoding_splits':['dev-clean','test-clean','test-other'],
            'utterances_per_split':500, 'chunks_ms':[160,320,640], 'evaluation_sample_seed':11,
            'scope':'predictor training seed only; acoustic checkpoint fixed','hold':0,
            'concurrency':{'mps_training':1,'cpu_decoding':2},
            'timing_scope':'accuracy experiment; decoding times may overlap predictor training'}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[]
        for seed in plan['additional_seeds']:
            for band, tag in ((0,'full'), (2,'band2')):
                name = f'{tag}_s{seed}'
                folder = OUT/name
                processes=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
                running=[int(line.strip().split(None,1)[0]) for line in processes
                         if 'scripts/train_stage2.py --kind audio --out '+str(folder.relative_to(ROOT))+' ' in line]
                if running:
                    print(f'WAIT existing training {name}: {running}',flush=True)
                    while any(_alive(pid) for pid in running):
                        time.sleep(15)
                if (folder/'log.jsonl').exists() and any(json.loads(line).get('event')=='done' for line in (folder/'log.jsonl').read_text().splitlines()):
                    print(f'SKIP completed training {name}', flush=True)
                else:
                    run(name, ['scripts/train_stage2.py','--kind','audio','--out',str(folder.relative_to(ROOT)),
                               '--band',str(band),'--seed',str(seed)])
                checkpoint=str((folder/'best.pt').relative_to(ROOT))
                for split in plan['decoding_splits']:
                    target=OUT/f'{tag}_s{seed}_{split}.json'
                    if target.exists():
                        parsed=json.loads(target.read_text())
                        if parsed['args']['hold']==0 and len(parsed.get('results',[]))==3 and all(row.get('utts')==500 for row in parsed['results']):
                            continue
                    args=['scripts/run_stage3.py','--split',split,'--n','500','--seed','11','--threads','1',
                          '--methods','H1a' if tag=='band2' else 'B1a','--hold','0','--out',str(target.relative_to(ROOT))]
                    args += ['--band-ckpt' if tag=='band2' else '--audio',checkpoint]
                    jobs.append(pool.submit(run,f'eval_{tag}_s{seed}_{split}',args))
        for job in as_completed(jobs):
            job.result()
    (OUT/'completed.json').write_text(json.dumps({'status':'all four trainings and twelve decoding jobs finished',
                                                 'utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    print('ALL SEED RUNS COMPLETE',flush=True)


def _alive(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


if __name__=='__main__':
    main()
