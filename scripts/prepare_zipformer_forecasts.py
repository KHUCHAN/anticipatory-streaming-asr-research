#!/usr/bin/env python3
"""Cache frozen streaming encoder outputs and training labels, resumably."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

from zipformer_features import ROOT,FrozenZipformer,align_gold

MODEL=None


def init():
    global MODEL
    MODEL=FrozenZipformer()


def process(rows):
    begin=time.perf_counter(); values=MODEL.batch(rows)
    result=[]
    for row,value in zip(rows,values):
        timing,quality=align_gold(value['gold'],value['ids'],value['frames'],len(value['enc']))
        value['gold_frames']=timing;value['alignment_match_fraction']=np.array(quality)
        p=ROOT/'data/processed/zipformer_forecasts'/row['split']/(row['utterance_id']+'.npz')
        p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_suffix('.tmp.npz')
        np.savez(temporary,**value);temporary.replace(p)
        result.append(dict(id=row['utterance_id'],split=row['split'],audio_seconds=float(value['duration']),
                           encoder_frames=len(value['enc']),gold_tokens=len(value['gold']),alignment_match_fraction=quality))
    return result,time.perf_counter()-begin


def main():
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',default=['train-clean-100','dev-clean','dev-other','test-clean','test-other'])
    p.add_argument('--workers',type=int,default=4);p.add_argument('--batch',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();out=ROOT/'runs/zipformer_ours';out.mkdir(parents=True,exist_ok=True)
    for split in a.splits:
        rows=[json.loads(l) for l in (ROOT/f'data/processed/manifests/{split}.jsonl').open()]
        if a.limit:rows=rows[:a.limit]
        allrows=rows
        rows=[r for r in rows if not (ROOT/f"data/processed/zipformer_forecasts/{split}/{r['utterance_id']}.npz").exists()]
        # Similar durations reduce inactive padded encoder work in a batch.
        rows.sort(key=lambda r:r['duration_seconds'])
        groups=[rows[i:i+a.batch] for i in range(0,len(rows),a.batch)]
        started=time.time();done=len(allrows)-len(rows)
        log=(out/f'prepare_{split}.jsonl').open('a',buffering=1)
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=init) as ex:
            futures=[ex.submit(process,g) for g in groups]
            for f in as_completed(futures):
                rec,seconds=f.result();done+=len(rec)
                log.write(json.dumps(dict(records=rec,worker_seconds=seconds))+'\n')
                if done%200<a.batch:
                    print(json.dumps(dict(split=split,done=done,total=len(allrows),seconds=round(time.time()-started,1))),flush=True)
        log.close()
        (out/f'prepared_{split}.json').write_text(json.dumps(dict(split=split,count=len(allrows),seconds=time.time()-started,batch=a.batch,workers=a.workers,limited=bool(a.limit)),indent=2)+'\n')
        print('PREPARED',split,len(allrows),flush=True)


if __name__=='__main__':main()
