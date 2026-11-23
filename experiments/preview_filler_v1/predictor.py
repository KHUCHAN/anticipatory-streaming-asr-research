"""Future-feature predictors used as preview fillers: deterministic (L1) and FGN-style stochastic (fair CRPS)."""
import json
import sys
import time

import numpy as np
import torch
from torch import nn

from shared import OUT,ROOT,config,write


class CondNorm(nn.Module):
    def __init__(self,d,noise):
        super().__init__();self.norm=nn.LayerNorm(d);self.mod=nn.Linear(d,2*d) if noise else None
        if noise:nn.init.normal_(self.mod.weight,std=.01);nn.init.zeros_(self.mod.bias)

    def forward(self,x,z):
        y=self.norm(x)
        if self.mod is None:return y
        scale,shift=self.mod(z).chunk(2,-1);return y*(1+scale)+shift


class Block(nn.Module):
    def __init__(self,d,heads,ff,noise):
        super().__init__();self.n1=CondNorm(d,noise);self.n2=CondNorm(d,noise)
        self.att=nn.MultiheadAttention(d,heads,batch_first=True);self.ff=nn.Sequential(nn.Linear(d,ff),nn.SiLU(),nn.Linear(ff,d))

    def forward(self,x,z):
        y=self.n1(x,z);x=x+self.att(y,y,y,need_weights=False)[0];return x+self.ff(self.n2(x,z))


class Predictor(nn.Module):
    """Past `context` normalized fbank frames -> next `future` frames, as a residual on the last observed frame."""
    def __init__(self,p,context,future,stochastic):
        super().__init__();d=p['width'];self.context=context;self.future=future;self.stochastic=stochastic;self.noise_dim=p['noise_dim']
        self.inp=nn.Linear(80,d);self.pos=nn.Parameter(torch.randn(1,context+future,d)*.02);self.query=nn.Parameter(torch.zeros(1,future,d))
        self.noise=nn.Linear(p['noise_dim'],d) if stochastic else None
        self.blocks=nn.ModuleList([Block(d,p['heads'],p['ff'],stochastic) for _ in range(p['layers'])]);self.out=nn.Linear(d,80)

    def forward(self,x,z=None):
        h=torch.cat([self.inp(x),self.query.expand(len(x),-1,-1)],1)+self.pos
        e=self.noise(z)[:,None] if self.stochastic else None
        for b in self.blocks:h=b(h,e)
        return x[:,-1:]+self.out(h[:,self.context:])


class Windows:
    def __init__(self,cfg,rows,stats):
        a2=json.loads((ROOT/cfg['a2_config']).read_text());cache=ROOT/a2['input_cache'];n=json.loads((cache/'complete.json').read_text())['frames']
        self.f=np.memmap(cache/'fbank.f16',dtype=np.float16,mode='r',shape=(n,80));self.rows=[r for r in rows if r['feature_frames']>40]
        self.mean=np.array(stats['mean'],np.float32);self.std=np.array(stats['std'],np.float32);s=cfg['stage2'];self.C=s['context_frames'];self.F=s['future_frames']

    def sample(self,rng,batch):
        X=np.empty((batch,self.C,80),np.float32);Y=np.empty((batch,self.F,80),np.float32)
        for b in range(batch):
            r=self.rows[rng.integers(len(self.rows))];t=int(rng.integers(1,r['feature_frames']-self.F+1));lo=r['feature_offset']
            past=np.asarray(self.f[lo+max(0,t-self.C):lo+t],np.float32)
            if len(past)<self.C:past=np.concatenate([np.repeat(past[:1],self.C-len(past),0),past])
            X[b]=past;Y[b]=self.f[lo+t:lo+t+self.F]
        return torch.from_numpy((X-self.mean)/self.std),torch.from_numpy((Y-self.mean)/self.std)


def load(kind,device='cpu'):
    cfg=config();s=cfg['stage2'];m=Predictor(s['predictor'],s['context_frames'],s['future_frames'],kind=='fgn')
    m.load_state_dict(torch.load(OUT/f'predictor_{kind}.pt',map_location='cpu'));return m.to(device).eval()


def main():
    kind=sys.argv[1];minutes=float(sys.argv[2]) if len(sys.argv)>2 else None;cfg=config();s=cfg['stage2'];p=s['predictor'];minutes=minutes or p['minutes']
    a2=json.loads((ROOT/cfg['a2_config']).read_text());cache=ROOT/a2['input_cache'];index=json.loads((cache/'index.json').read_text());stats=json.loads((cache/'stats.json').read_text())
    sel=set(json.loads((ROOT/cfg['protocol']).read_text())['selection_ids'])
    train=Windows(cfg,[r for r in index if r['split']==s['train_split']],stats);val=Windows(cfg,[r for r in index if r['utterance_id'] in sel],stats)
    dev=torch.device('mps');torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    model=Predictor(p,s['context_frames'],s['future_frames'],kind=='fgn').to(dev);opt=torch.optim.AdamW(model.parameters(),p['learning_rate'],weight_decay=.01)
    vx,vy=val.sample(np.random.default_rng(1),2048);vx=vx.to(dev);vy=vy.to(dev);t0=time.time();step=0;log=[];run=0.0

    def crps(samples,y):
        k=len(samples);a=sum((x-y).abs() for x in samples)/k
        b=sum((samples[i]-samples[j]).abs() for i in range(k) for j in range(i+1,k))/(k*(k-1)) if k>1 else 0
        return a-b  # fair CRPS: second term already equals 0.5 * mean over ordered pairs

    @torch.no_grad()
    def validate():
        model.eval();out={}
        if kind=='fgn':
            S=[torch.cat([model(vx[i:i+512],torch.randn(len(vx[i:i+512]),p['noise_dim'],device=dev)) for i in range(0,len(vx),512)]) for _ in range(8)]
            mean=sum(S)/8;out['crps']=float(crps(S,vy).mean());out['spread_skill']=float((torch.stack(S).var(0,unbiased=True).mean().sqrt()*np.sqrt(9/8))/((mean-vy)**2).mean().sqrt())
        else:mean=torch.cat([model(vx[i:i+512]) for i in range(0,len(vx),512)])
        out['mae']=float((mean-vy).abs().mean());out['mae_by_frame']=[round(float(v),3) for v in (mean-vy).abs().mean((0,2))[[0,7,15,31]]]
        out['mae_repeat']=float((vx[:,-1:]-vy).abs().mean());out['mae_context_mean']=float((vx.mean(1,keepdim=True)-vy).abs().mean());model.train();return out

    while time.time()-t0<minutes*60:
        x,y=train.sample(rng,p['batch']);x=x.to(dev);y=y.to(dev);frac=(time.time()-t0)/(minutes*60)
        for g in opt.param_groups:g['lr']=p['learning_rate']*min(1,step/500)*max(.05,1-frac)
        if kind=='fgn':loss=crps([model(x,torch.randn(len(x),p['noise_dim'],device=dev)) for _ in range(p['crps_samples'])],y).mean()
        else:loss=(model(x)-y).abs().mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();step+=1;run=loss.item() if step==1 else .99*run+.01*loss.item()
        if step%2000==0:
            v=validate();log.append(dict(step=step,minutes=round((time.time()-t0)/60,1),train=round(run,4),**v));print(json.dumps(log[-1]),flush=True)
    final=validate();torch.save(model.state_dict(),OUT/f'predictor_{kind}.pt')
    write(OUT/f'predictor_{kind}.json',dict(kind=kind,steps=step,minutes=minutes,parameters=sum(q.numel() for q in model.parameters()),final=final,log=log));print('done',json.dumps(final))


if __name__=='__main__':main()
