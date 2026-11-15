#!/usr/bin/env python3
"""Extract baseline-matched fbank once; do not recompute or expose teacher states."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import time

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf

# Preparation can run in the baseline environment without importing torch.
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'runs/predict_correct_v1'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)


def features(wave):
    o=knf.FbankOptions();o.frame_opts.dither=0;o.frame_opts.snip_edges=False
    o.mel_opts.num_bins=80;o.mel_opts.low_freq=20;o.mel_opts.high_freq=-400
    fb=knf.OnlineFbank(o);fb.accept_waveform(16000,wave);fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])


def extract(row):
    wave,sr=sf.read(ROOT/row['audio'],dtype='float32')
    assert sr==16000 and len(wave)==row['num_samples']
    values=features(np.concatenate([wave,np.zeros(10560,np.float32)]))
    return row,values.astype(np.float16)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'cache/complete.json').exists():
        print('Fbank cache already complete.');return
    teacher=json.loads((ROOT/cfg['encoder_cache']/'index.json').read_text())
    train=[r for r in teacher if r['split']==cfg['train_split']]
    role=ROOT/cfg['gate_protocol']
    selection=[json.loads(s) for s in (role/'model_selection.jsonl').read_text().splitlines()]
    evaluation=[json.loads(s) for s in (role/'gate_calibration.jsonl').read_text().splitlines()]
    selected=[];evaluated=[]
    for split in ['dev-clean','dev-other']:
        selected+=random.Random(cfg['selection_seed']).sample([r for r in selection if r['split']==split],cfg['selection_per_split'])
        evaluated+=random.Random(cfg['evaluation_seed']).sample([r for r in evaluation if r['split']==split],cfg['evaluation_per_split'])
    assert not {r['speaker_id'] for r in selected}&{r['speaker_id'] for r in evaluated}
    selection_ids={r['utterance_id'] for r in selected};evaluation_ids={r['utterance_id'] for r in evaluated}
    rows=train+[r for r in teacher if r['utterance_id'] in selection_ids|evaluation_ids]
    protocol=dict(selection_ids=sorted(selection_ids),evaluation_ids=sorted(evaluation_ids),
                  selection_role='model_selection',evaluation_role='gate_calibration',speaker_disjoint=True,
                  selection_seed=cfg['selection_seed'],evaluation_seed=cfg['evaluation_seed'],test_used=False)
    write(OUT/'protocol.json',protocol)
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    path=cache/'fbank.f16';index=[];offset=0;sums=np.zeros(80,np.float64);squares=np.zeros(80,np.float64);count=0
    started=time.time()
    with path.open('wb') as stream,ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        for i,(row,values) in enumerate(pool.map(extract,rows,chunksize=8),1):
            assert len(values)>=32*(row['nchunks']-1)+46
            stream.write(values.tobytes());record=dict(row,feature_offset=offset,feature_frames=len(values));index.append(record)
            offset+=len(values)
            if row['split']==cfg['train_split']:
                x=values.astype(np.float64);sums+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
            if i%1000==0:print(json.dumps(dict(event='features',utterances=i,total=len(rows),seconds=time.time()-started)),flush=True)
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,.01))
    write(cache/'index.json',index);write(cache/'stats.json',dict(mean=mean.tolist(),std=std.tolist(),train_frames=count,scope='train-only global fbank normalization, including matched tail padding'))
    write(cache/'complete.json',dict(utterances=len(rows),frames=offset,bytes=path.stat().st_size,seconds=time.time()-started))
    print('FBANK COMPLETE',len(rows),'utterances',round(time.time()-started,1),'seconds',flush=True)


if __name__=='__main__':main()
