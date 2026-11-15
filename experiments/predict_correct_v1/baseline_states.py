#!/usr/bin/env python3
"""Matched-set representation baselines; neither is fed into A/B/C."""
import json
import numpy as np

from common import ROOT,HERE,OUT,config,write
from data import datasets


def main():
    cfg=config();_,_,rows=datasets(cfg)
    n=json.loads((ROOT/cfg['encoder_cache']/'complete.json').read_text())['frames']
    states=np.memmap(ROOT/cfg['encoder_cache']/'encoder.f16',dtype=np.float16,mode='r',shape=(n,512))
    stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text())
    mean=np.array(stats['mean'],np.float32);std=np.array(stats['std'],np.float32)
    scores={}
    for split in ['dev-clean','dev-other']:
        absolute=persistence=count=0
        for row in rows:
            if row['split']!=split:continue
            x=(np.array(states[row['offset']:row['offset']+row['nframes']],np.float32)-mean)/std
            x=x.reshape(-1,8,512)
            previous=np.concatenate([np.zeros_like(x[:1]),x[:-1]],0)
            absolute+=float(np.abs(x).sum(dtype=np.float64));persistence+=float(np.abs(x-previous).sum(dtype=np.float64));count+=x.size
        scores[split]=dict(mean_prediction_mae=absolute/count,oracle_previous_true_chunk_mae=persistence/count,elements=count)
    result=dict(scores=scores,scope='Same final evaluation utterances, frames and train-only normalization as A/B/C. Mean baseline is constant normalized zero. Persistence uses previous TRUE teacher states as an offline oracle diagnostic; it is not the deployed C condition and is never a model input. Neither MAE reduction is a percentage of acoustic information.',test_used=False)
    write(HERE/'artifacts/state_baselines.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
