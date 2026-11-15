#!/usr/bin/env python3
"""Training-only sampled median and direct-supervision coverage diagnostics."""
import json

import numpy as np

from common import ROOT,HERE,OUT,config,write
from data import datasets


def main():
    cfg=config();train,dev,evaluation=datasets(cfg);stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text())
    train_frames=sum(r['nframes'] for r in train.rows)
    assert train.rows[0]['offset']==0 and train.rows[-1]['offset']+train.rows[-1]['nframes']==train_frames
    rng=np.random.default_rng(8237);indices=np.sort(rng.choice(train_frames,size=65536,replace=False))
    mean=np.array(stats['mean'],np.float32);std=np.array(stats['std'],np.float32)
    sample=(np.array(train.states[indices],np.float32)-mean)/std;median=np.median(sample,axis=0)
    epochs=[]
    for phase in ['mae','kl']:
        path=OUT/phase/'log.jsonl'
        if path.exists():
            records=[json.loads(line) for line in path.read_text().splitlines()]
            epochs += [r['epoch']+(cfg['mae_epochs'] if phase=='kl' else 0) for r in records if r['event']=='eval']
    counts=dict(available=0,visits=0,unique=0,first_chunks_seen=0,completed_global_epochs=epochs)
    for index,row in enumerate(train.rows):
        seen=set();counts['available']+=row['nchunks']
        for epoch in epochs:
            start=train.start(index,epoch);end=min(row['nchunks'],start+cfg['chunks_per_window']);seen.update(range(start,end));counts['visits']+=end-start
        counts['unique']+=len(seen);counts['first_chunks_seen']+=int(0 in seen)
    scores={}
    for split in ['dev-clean','dev-other']:
        sums=dict(mean=0.,sampled_median=0.);count=0
        for row in evaluation:
            if row['split']!=split:continue
            values=(np.array(train.states[row['offset']:row['offset']+row['nframes']],np.float32)-mean)/std
            sums['mean']+=float(np.abs(values).sum());sums['sampled_median']+=float(np.abs(values-median).sum());count+=values.size
        scores[split]={key:value/count for key,value in sums.items()}
    write(HERE/'artifacts/state_baselines.json',dict(scores=scores,median_estimation=dict(training_frames_sampled=65536,seed=8237,exact_population_median=False),
        direct_supervision_coverage=counts,scope='Constant estimates from training data only; scores are diagnostic reference metrics, never inference inputs or policy tuning',test_used=False))
    print(json.dumps(dict(scores=scores,coverage=counts),indent=2))


if __name__=='__main__':main()
