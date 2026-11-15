"""Random training windows with audio-only burn-in and cached teacher labels."""
import json
import random

import numpy as np
import torch

from common import ROOT,OUT


class Data:
    def __init__(self,rows,cfg):
        self.rows=rows;self.cfg=cfg
        total=json.loads((OUT/'cache/complete.json').read_text())['frames']
        self.fbank=np.memmap(OUT/'cache/fbank.f16',dtype=np.float16,mode='r',shape=(total,80))
        total=json.loads((ROOT/cfg['encoder_cache']/'complete.json').read_text())['frames']
        self.encoder=np.memmap(ROOT/cfg['encoder_cache']/'encoder.f16',dtype=np.float16,mode='r',shape=(total,512))
        self.contexts=np.memmap(ROOT/cfg['encoder_cache']/'contexts.i16',dtype=np.int16,mode='r',shape=(total,2))
        self.tokens=np.memmap(ROOT/cfg['encoder_cache']/'tokens.i16',dtype=np.int16,mode='r',shape=(total,))
        self.stats=json.loads((OUT/'cache/stats.json').read_text())
        self.mean=np.array(self.stats['mean'],np.float32);self.std=np.array(self.stats['std'],np.float32)
        self.history=sum((cfg['audio_kernel']-1)*d for d in cfg['audio_dilations'])

    def batches(self,epoch=0):
        order=list(range(len(self.rows)));random.Random(8000+epoch).shuffle(order)
        n=self.cfg['batch_utterances']
        return [order[i:i+n] for i in range(0,len(order),n)]

    def collate(self,indices,epoch,device):
        cfg=self.cfg;t=cfg['burn_chunks']+cfg['loss_chunks'];length=self.history+32*(t-1)+45
        batch=len(indices);features=np.zeros((batch,length,80),np.float32)
        fvalid=np.zeros((batch,length),bool);states=np.zeros((batch,t,8,512),np.float32)
        contexts=np.zeros((batch,t,8,2),np.int64);tokens=np.zeros((batch,t,8),np.int64)
        valid=np.zeros((batch,t,8),bool);starts=[]
        for j,index in enumerate(indices):
            row=self.rows[index]
            # Evaluation uses epoch zero; training changes the window every epoch.
            rng=random.Random((epoch+1)*1000003+index*9176+cfg['seed'])
            start=rng.randint(0,max(0,row['nchunks']-t));starts.append(start)
            positions=32*start-self.history+np.arange(length)
            mask=(positions>=0)&(positions<row['feature_frames'])
            features[j,mask]=(np.array(self.fbank[row['feature_offset']+positions[mask]],np.float32)-self.mean)/self.std
            fvalid[j]=mask
            n=min(t,row['nchunks']-start);a=row['offset']+start*8;z=a+n*8
            states[j,:n]=np.array(self.encoder[a:z],np.float32).reshape(n,8,512)
            contexts[j,:n]=np.array(self.contexts[a:z],np.int64).reshape(n,8,2)
            tokens[j,:n]=np.array(self.tokens[a:z],np.int64).reshape(n,8)
            valid[j,min(cfg['burn_chunks'],max(0,n-1)):n]=True
        result=[torch.from_numpy(x).to(device) for x in [features,fvalid,states,contexts,tokens,valid]]
        return (*result,starts)


def datasets(cfg):
    rows=json.loads((OUT/'cache/index.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    selection=set(protocol['selection_ids']);evaluation=set(protocol['evaluation_ids'])
    return (Data([r for r in rows if r['split']==cfg['train_split']],cfg),
            Data([r for r in rows if r['utterance_id'] in selection],cfg),
            [r for r in rows if r['utterance_id'] in evaluation])
