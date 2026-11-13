"""Our audio-aware AR forecaster on frozen Zipformer features, BPE ids unchanged."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import Decoder

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/processed/zipformer_forecasts'


class ZipformerForecaster(nn.Module):
    def __init__(self,cfg,band=0):
        super().__init__();self.cfg=cfg;self.band=band
        d=cfg['decoder']
        self.adapter=nn.Sequential(nn.Linear(cfg['input_encoder_dim'],d['d']),nn.LayerNorm(d['d']))
        self.decoder=Decoder(**d,cross=True);self.decoder.band=band

    def forward(self,tokens,encoder,mask):
        return self.decoder(tokens,self.adapter(encoder),mask)


class CachedSet:
    def __init__(self,rows):
        self.rows=rows
        self.lengths=[int(r['duration_seconds']/.04)+24 for r in rows]

    def __len__(self):return len(self.rows)

    def load(self,index):
        r=self.rows[index]
        with np.load(CACHE/r['split']/(r['utterance_id']+'.npz')) as v:
            return {k:v[k] for k in v.files}

    def batches(self,max_frames=8000,max_utts=24,seed=None):
        order=sorted(range(len(self)),key=lambda i:self.lengths[i]);groups=[];group=[]
        for i in order:
            if group and ((len(group)+1)*self.lengths[i]>max_frames or len(group)>=max_utts):
                groups.append(group);group=[]
            group.append(i)
        if group:groups.append(group)
        if seed is not None:random.Random(seed).shuffle(groups)
        return groups

    def collate(self,indices,rng,device):
        # Same segment/cutoff construction as the successful AVS AR experiments.
        values=[self.load(i) for i in indices];b=len(values)
        t=max(len(v['enc']) for v in values);u=max(len(v['gold']) for v in values)+1
        enc=torch.zeros(b,t,512);ys=torch.full((b,u),1,dtype=torch.long)
        target=torch.full((b,u),-100,dtype=torch.long);cuts=torch.zeros(b,u,dtype=torch.long)
        horizon=torch.full((b,u),-1,dtype=torch.long)
        for i,v in enumerate(values):
            length=len(v['gold']);enc[i,:len(v['enc'])]=torch.from_numpy(v['enc'].astype(np.float32))
            ys[i,1:length+1]=torch.from_numpy(v['gold'].astype(np.int64))
            target[i,:length]=torch.from_numpy(v['gold'].astype(np.int64));target[i,length]=1
            spikes=v['gold_frames'].tolist()+[len(v['enc'])];j=0
            while j<len(spikes):
                hi=(spikes[j]//8)*8
                lo=min(hi,(spikes[j-1]//8+1)*8) if j else 0
                cutoff=lo+8*rng.randint(0,(hi-lo)//8);segment=rng.randint(1,4)
                for h in range(segment):
                    if j>=len(spikes):break
                    cuts[i,j]=cutoff;horizon[i,j]=h;j+=1
        mask=torch.arange(t)[None,None,:]<cuts[:,:,None]
        return ys.to(device),enc.to(device),mask.to(device),target.to(device),horizon.to(device)


def read_rows(path):return [json.loads(line) for line in Path(path).open()]
