"""Reuse immutable caches with direct supervision of beginning/middle/end."""
import json
import random

import numpy as np
import torch

from common import ROOT


class Data:
    def __init__(self,rows,cfg):
        self.rows=rows;self.cfg=cfg
        cache=ROOT/cfg['input_cache'];teacher=ROOT/cfg['encoder_cache']
        count=json.loads((cache/'complete.json').read_text())['frames']
        self.fbank=np.memmap(cache/'fbank.f16',mode='r',dtype=np.float16,shape=(count,80))
        count=json.loads((teacher/'complete.json').read_text())['frames']
        self.states=np.memmap(teacher/'encoder.f16',mode='r',dtype=np.float16,shape=(count,512))
        self.contexts=np.memmap(teacher/'contexts.i16',mode='r',dtype=np.int16,shape=(count,2))
        self.tokens=np.memmap(teacher/'tokens.i16',mode='r',dtype=np.int16,shape=(count,))
        stats=json.loads((cache/'stats.json').read_text());self.mean=np.array(stats['mean'],np.float32);self.std=np.array(stats['std'],np.float32)
        self.history=(cfg['kernel']-1)+4*sum((cfg['kernel']-1)*d for d in cfg['dilations'])

    def batches(self,epoch=0):
        order=list(range(len(self.rows)));random.Random(self.cfg['seed']*1000+epoch).shuffle(order)
        n=self.cfg['batch_utterances'];return [order[i:i+n] for i in range(0,len(order),n)]

    def start(self,index,epoch):
        n=self.rows[index]['nchunks'];maximum=max(0,n-self.cfg['chunks_per_window'])
        rng=random.Random((epoch+1)*1000003+index*9176+self.cfg['seed']);kind=rng.randrange(8)
        if kind==0:return 0
        if kind==1:return maximum
        return rng.randint(0,maximum)

    def collate(self,indices,epoch,device):
        chunks=self.cfg['chunks_per_window'];length=self.history+32*(chunks-1)+45;b=len(indices)
        features=np.zeros((b,length,80),np.float32);fvalid=np.zeros((b,length),bool)
        states=np.zeros((b,chunks,8,512),np.float32);contexts=np.zeros((b,chunks,8,2),np.int64)
        tokens=np.zeros((b,chunks,8),np.int64);valid=np.zeros((b,chunks,8),bool);starts=[]
        for j,index in enumerate(indices):
            row=self.rows[index];start=self.start(index,epoch);starts.append(start)
            positions=32*start-self.history+np.arange(length);mask=(positions>=0)&(positions<row['feature_frames'])
            features[j,mask]=(np.array(self.fbank[row['feature_offset']+positions[mask]],np.float32)-self.mean)/self.std;fvalid[j]=mask
            n=min(chunks,row['nchunks']-start);lo=row['offset']+start*8;hi=lo+n*8
            states[j,:n]=np.array(self.states[lo:hi],np.float32).reshape(n,8,512)
            contexts[j,:n]=np.array(self.contexts[lo:hi],np.int64).reshape(n,8,2)
            tokens[j,:n]=np.array(self.tokens[lo:hi],np.int64).reshape(n,8);valid[j,:n]=True
        return tuple(torch.from_numpy(x).to(device) for x in [features,fvalid,states,contexts,tokens,valid]),starts


def datasets(cfg):
    rows=json.loads((ROOT/cfg['input_cache']/'index.json').read_text());roles=json.loads((ROOT/cfg['protocol']).read_text())
    selection=set(roles['selection_ids']);evaluation=set(roles['evaluation_ids'])
    return Data([r for r in rows if r['split']==cfg['train_split']],cfg),Data([r for r in rows if r['utterance_id'] in selection],cfg),[r for r in rows if r['utterance_id'] in evaluation]
