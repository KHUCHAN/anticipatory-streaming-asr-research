import json,random
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/fgn_state_v1'


class PackedSet:
    def __init__(self,rows):
        self.rows=rows
        total=json.loads((OUT/'cache/complete.json').read_text())['frames']
        self.enc=np.memmap(OUT/'cache/encoder.f16',dtype=np.float16,mode='r',shape=(total,512))
        self.ctx=np.memmap(OUT/'cache/contexts.i16',dtype=np.int16,mode='r',shape=(total,2))
        self.tgt=np.memmap(OUT/'cache/tokens.i16',dtype=np.int16,mode='r',shape=(total,))

    def batches(self,cfg,seed=None):
        order=sorted(range(len(self.rows)),key=lambda i:self.rows[i]['nchunks']);groups=[];group=[]
        for i in order:
            if group and ((len(group)+1)*self.rows[i]['nchunks']>cfg['max_padded_chunks'] or len(group)>=cfg['batch_utterances']):
                groups.append(group);group=[]
            group.append(i)
        if group:groups.append(group)
        if seed is not None:random.Random(seed).shuffle(groups)
        return groups

    def collate(self,indices,cfg,rng,device):
        rows=[self.rows[i] for i in indices];b=len(rows);t=max(r['nchunks'] for r in rows)
        x=torch.zeros(b,t,8,512);c=torch.zeros(b,t,8,2,dtype=torch.long);y=torch.zeros(b,t,8,dtype=torch.long)
        cuts=[];lengths=[]
        for i,row in enumerate(rows):
            a=row['offset'];z=a+row['nframes'];n=row['nchunks'];lengths.append(n)
            x[i,:n]=torch.from_numpy(np.array(self.enc[a:z],dtype=np.float32).reshape(n,8,512))
            c[i,:n]=torch.from_numpy(np.array(self.ctx[a:z],dtype=np.int64).reshape(n,8,2))
            y[i,:n]=torch.from_numpy(np.array(self.tgt[a:z],dtype=np.int64).reshape(n,8))
            last=min(n-2,max(0,int((row['duration_seconds']-.48)/.32)))
            for _ in range(cfg['cutoffs_per_utterance']):cuts.append((i,rng.randint(0,last)))
        return x.to(device),c.to(device),y.to(device),torch.tensor(cuts,device=device),torch.tensor(lengths,device=device)


def selections(cfg):
    rows=json.loads((OUT/'cache/index.json').read_text())
    train=[r for r in rows if r['split']==cfg['train_split']]
    protocol=ROOT/cfg['gate_protocol']
    dev=[json.loads(l) for l in (protocol/'model_selection.jsonl').open()]
    chosen=random.Random(cfg['selection_seed']).sample(dev,cfg['dev_selection_utterances'])
    ids={r['utterance_id'] for r in chosen}
    selection=[r for r in rows if r['utterance_id'] in ids]
    return train,selection
