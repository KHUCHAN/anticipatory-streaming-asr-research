#!/usr/bin/env python3
"""Audit the actual deterministic random-window supervision schedule."""
import json
import random

from common import HERE,OUT,config,write


def main():
    cfg=config();rows=[r for r in json.loads((OUT/'cache/index.json').read_text()) if r['split']==cfg['train_split']]
    window=cfg['burn_chunks']+cfg['loss_chunks']
    available=unique=visits=first_two_available=first_two_seen=0
    for index,row in enumerate(rows):
        count=row['nchunks'];available+=count;seen=set()
        for epoch in range(1,cfg['epochs']+1):
            rng=random.Random((epoch+1)*1000003+index*9176+cfg['seed'])
            start=rng.randint(0,max(0,count-window));n=min(window,count-start)
            loss_start=min(cfg['burn_chunks'],max(0,n-1))
            supervised=range(start+loss_start,start+n)
            visits+=len(supervised);seen.update(supervised)
        unique+=len(seen)
        first_two_available+=min(2,count);first_two_seen+=len(seen&set(range(min(2,count))))
    value=dict(train_utterances=len(rows),epochs=cfg['epochs'],available_chunks=available,
        supervised_chunk_visits=visits,unique_supervised_chunks=unique,
        unique_coverage_fraction=unique/available,first_two_available_chunks=first_two_available,
        first_two_unique_supervised_chunks=first_two_seen,
        interpretation='Coverage of direct losses, not input audio exposure. Burn-in outputs influence later inference feedback but are detached from later chunk gradients. Long-utterance first two chunks are burn-in only under this fixed recipe; this is a pilot training limitation.',
        changes_training=False,test_used=False)
    write(HERE/'artifacts/training_coverage.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
