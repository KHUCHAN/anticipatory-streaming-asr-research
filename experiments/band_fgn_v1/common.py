import hashlib, importlib.util, json, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/band_fgn_v1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

oldstate=module('band_hybrid_original_state',ROOT/'experiments/fgn_state_v1/model.py')
StateFGN=oldstate.StateFGN
fair_crps_two=oldstate.fair_crps_two

def config():return json.loads((HERE/'config.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n')

class Dataset:
    def __init__(self,rows,meta):
        self.rows=rows;self.meta=meta
        n=json.loads((ROOT/'runs/fgn_state_v1/cache/complete.json').read_text())['frames']
        self.enc=np.memmap(ROOT/'runs/fgn_state_v1/cache/encoder.f16',dtype=np.float16,mode='r',shape=(n,512))
    def batches(self,cfg,seed=None):
        order=sorted(range(len(self.rows)),key=lambda i:self.rows[i]['nchunks']);groups=[];group=[]
        for i in order:
            if group and (len(group)>=cfg['batch_utterances'] or (len(group)+1)*self.rows[i]['nchunks']>cfg['max_padded_chunks']):
                groups.append(group);group=[]
            group.append(i)
        if group:groups.append(group)
        if seed is not None:random.Random(seed).shuffle(groups)
        return groups
    def collate(self,idx,rng,device):
        rows=[self.rows[i] for i in idx];b=len(rows);t=max(r['nchunks'] for r in rows)
        states=torch.zeros(b,t,8,512);lengths=[];cuts=[];prefixes=[];targets=[]
        for i,r in enumerate(rows):
            n=r['nchunks'];a=r['offset'];states[i,:n]=torch.from_numpy(np.array(self.enc[a:a+n*8],dtype=np.float32).reshape(n,8,512));lengths.append(n)
            m=self.meta[r['utterance_id']];gold=m['gold'];frames=m['gold_frames']
            # Targets/timing are offline supervision. The model receives only
            # states strictly before this cutoff and the preceding text prefix.
            j=rng.randrange(len(gold));hi=min(n-1,max(0,frames[j]//8))
            lo=min(hi,max(0,frames[j-1]//8+1)) if j else 0
            cut=rng.randint(lo,hi)
            cuts.append(cut);prefixes.append([1]+gold[:j]);targets.append(gold[j])
        u=max(map(len,prefixes));tokens=torch.ones(b,u,dtype=torch.long)
        for i,p in enumerate(prefixes):tokens[i,:len(p)]=torch.tensor(p)
        vals=(states,torch.tensor(lengths),torch.tensor(cuts),tokens,torch.tensor([len(p) for p in prefixes]),torch.tensor(targets))
        return tuple(v.to(device) for v in vals)

def datasets(cfg):
    index=json.loads((ROOT/'runs/fgn_state_v1/cache/index.json').read_text())
    meta=json.loads((OUT/'gold_metadata.json').read_text())
    train=[r for r in index if r['split']==cfg['train_split']]
    protocol=ROOT/cfg['gate_protocol']
    dev=[json.loads(l) for l in (protocol/'model_selection.jsonl').open()]
    dev=random.Random(cfg['selection_seed']).sample(dev,cfg['dev_selection_utterances'])
    devid={r['utterance_id'] for r in dev}
    selected=[r for r in index if r['utterance_id'] in devid]
    probe=random.Random(812).sample(train,cfg['train_probe_utterances'])
    return Dataset(train,meta),Dataset(selected,meta),Dataset(probe,meta)

def rows_for(part,cfg):
    paths=[ROOT/cfg['gate_protocol']/f'{p}.jsonl' for p in ['gate_fit','gate_calibration']] if part=='dev' else [ROOT/f'data/processed/manifests/{s}.jsonl' for s in cfg['evaluation_splits']]
    return [json.loads(l) for p in paths for l in p.open()]
