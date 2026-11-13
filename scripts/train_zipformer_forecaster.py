#!/usr/bin/env python3
"""Actually train our full/band-2 audio forecasters on the strong frozen encoder."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from avs.zipformer_forecaster import ZipformerForecaster,CachedSet,read_rows


@torch.no_grad()
def evaluate(model,ds,cfg,device):
    model.eval();total=0;loss=0.;hits=0;hstats=[[0,0] for _ in range(4)]
    rng=random.Random(20261111)
    for idx in ds.batches(cfg['max_batch_padded_encoder_frames'],cfg['max_batch_utterances']):
        ys,enc,mask,tgt,h=ds.collate(idx,rng,device)
        logits=model(ys,enc,mask);n=int((tgt!=-100).sum())
        loss+=float(F.cross_entropy(logits.transpose(1,2),tgt,ignore_index=-100,reduction='sum'))
        correct=logits.argmax(-1)==tgt;hits+=int(correct.sum());total+=n
        for k in range(4):
            valid=(h==k)&(tgt!=-100);hstats[k][0]+=int(valid.sum());hstats[k][1]+=int((correct&valid).sum())
    model.train()
    return dict(nll=loss/total,hit1=100*hits/total,tokens=total,horizon_hit1=[100*v/max(1,n) for n,v in hstats])


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['full','band2'],required=True)
    p.add_argument('--config',default='configs/zipformer_ours.json');p.add_argument('--device',default='mps')
    p.add_argument('--bench',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--wait-cache',action='store_true',help='Wait for the concurrently prepared feature cache')
    a=p.parse_args();cfg=json.loads((ROOT/a.config).read_text());seed=cfg['seed']
    random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    out=ROOT/'runs/zipformer_ours'/a.kind
    if a.bench:out=out/'bench'
    out.mkdir(parents=True,exist_ok=True)
    model=ZipformerForecaster(cfg,cfg['variants'][a.kind]).to(a.device)
    initial_hash=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.state_dict().values())).hexdigest()
    train_rows=read_rows(ROOT/'data/processed/manifests'/f"{cfg['forecaster_train']}.jsonl")
    if a.limit:train_rows=train_rows[:a.limit]
    train=CachedSet(train_rows)
    dev_rows=read_rows(ROOT/cfg['dev_protocol']/'model_selection.jsonl')
    dev_rows=random.Random(410).sample(dev_rows,min(cfg['model_selection_max_utterances'],len(dev_rows)))
    dev=CachedSet(dev_rows)
    def await_cache(rows):
        waited=time.monotonic();last_message=-60
        for row in rows:
            path=ROOT/cfg['features']/row['split']/(row['utterance_id']+'.npz')
            while not path.exists():
                if not a.wait_cache:raise FileNotFoundError(path)
                elapsed=time.monotonic()-waited
                if elapsed>1800:raise TimeoutError(f'Cache not ready after 30 minutes: {path}')
                if elapsed-last_message>=60:
                    print(json.dumps(dict(event='waiting_cache',utterance=row['utterance_id'],seconds=round(elapsed))),flush=True)
                    last_message=elapsed
                time.sleep(1)
    log=(out/'log.jsonl').open('a',buffering=1)
    def emit(v):log.write(json.dumps(v)+'\n');print(json.dumps(v),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],betas=tuple(cfg['optimizer']['betas']),weight_decay=cfg['optimizer']['weight_decay'])
    batches=train.batches(cfg['max_batch_padded_encoder_frames'],cfg['max_batch_utterances'])
    total_steps=len(batches)*cfg['train_epochs'];warm=cfg['optimizer']['warmup'];step=0;best=float('inf');start_epoch=0
    last=out/'last.pt'
    if last.exists() and not a.bench:
        state=torch.load(last,map_location=a.device,weights_only=False)
        assert state['cfg']==cfg and state['kind']==a.kind
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        step=state['step'];start_epoch=state['epoch']+1;best=state['best_nll']
    emit(dict(event='start',kind=a.kind,params=sum(x.numel() for x in model.parameters()),initial_state_sha256=initial_hash,
              train_utterances=len(train),dev_utterances=len(dev),steps_per_epoch=len(batches),epochs=cfg['train_epochs'],device=a.device,cfg=cfg))
    for epoch in range(start_epoch,cfg['train_epochs']):
        rng=random.Random(seed+epoch);started=time.time();losses=[]
        for idx in train.batches(cfg['max_batch_padded_encoder_frames'],cfg['max_batch_utterances'],None if epoch==0 else 7000+epoch):
            await_cache([train_rows[i] for i in idx])
            ys,enc,mask,tgt,_=train.collate(idx,rng,a.device)
            factor=min(1.,(step+1)/warm) if step<warm else .05+.95*.5*(1+math.cos(math.pi*(step-warm)/max(1,total_steps-warm)))
            for group in optimizer.param_groups:group['lr']=cfg['optimizer']['lr']*factor
            optimizer.zero_grad(set_to_none=True);logits=model(ys,enc,mask)
            loss=F.cross_entropy(logits.transpose(1,2),tgt,ignore_index=-100,label_smoothing=cfg['label_smoothing'])
            if not torch.isfinite(loss):
                emit(dict(event='abort',reason='nonfinite loss',epoch=epoch,step=step));raise RuntimeError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
            optimizer.step();step+=1;losses.append(float(loss.detach()))
            if step%100==0:
                torch.mps.synchronize() if a.device=='mps' else None
                emit(dict(event='train',epoch=epoch,step=step,loss=sum(losses[-100:])/min(100,len(losses)),epoch_seconds=time.time()-started,
                          memory_gb=torch.mps.driver_allocated_memory()/2**30 if a.device=='mps' else None))
            if a.bench and step>=a.bench:
                emit(dict(event='bench_done',steps=step,seconds=time.time()-started));return
        await_cache(dev_rows)
        ev=evaluate(model,dev,cfg,a.device);best=min(best,ev['nll'])
        state=dict(model=model.state_dict(),cfg=cfg,kind=a.kind,band=cfg['variants'][a.kind],epoch=epoch,step=step,optimizer=optimizer.state_dict(),best_nll=best,dev=ev,initial_state_sha256=initial_hash)
        torch.save(state,out/'last.tmp.pt');(out/'last.tmp.pt').replace(last)
        if ev['nll']<=best:torch.save(state,out/'best.pt')
        emit(dict(event='eval',epoch=epoch,step=step,**ev,seconds=time.time()-started))
    emit(dict(event='done',epochs=cfg['train_epochs'],best_nll=best))


if __name__=='__main__':main()
