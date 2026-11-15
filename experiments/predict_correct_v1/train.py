#!/usr/bin/env python3
"""Train audio correction with random cutoffs and corrected-state recurrence."""
import argparse
import fcntl
import json
import math
import random
import time

import torch

from common import ROOT,HERE,OUT,config,write,sha,state_module
from model import Corrector
from data import datasets
from objective import objective
from inference import Inference,corpus


@torch.inference_mode()
def evaluate_loss(model,bridge,data,cfg,mean,std,device):
    model.eval();sums={key:0. for key in ['loss','mae','kl']};count=0
    for group in data.batches():
        batch=data.collate(group,0,device);loss,parts=objective(model,bridge,batch,cfg,mean,std)
        n=len(group);count+=n;sums['loss']+=n*float(loss)
        for key in parts:sums[key]+=n*float(parts[key])
    model.train();return {key:value/count for key,value in sums.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['A','B','C'],required=True)
    parser.add_argument('--bench',type=int,default=0);args=parser.parse_args();cfg=config()
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    assert torch.backends.mps.is_available(),'Mac GPU is required; run with approved GPU access.'
    model=Corrector(cfg,args.variant).to('mps');bridge=state_module.FrozenRNNT(ROOT/cfg['bridge']).to('mps').eval()
    train,dev,_=datasets(cfg);stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text())
    mean=torch.tensor(stats['mean'],device='mps');std=torch.tensor(stats['std'],device='mps')
    out=OUT/args.variant/('benchmark' if args.bench else '');out.mkdir(parents=True,exist_ok=True)
    lease=(out/'training.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
    else:write(out/'config.json',cfg)
    log=(out/'log.jsonl').open('a',buffering=1)
    def emit(value):
        print(json.dumps(value),flush=True);log.write(json.dumps(value)+'\n')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],betas=(.9,.98),weight_decay=cfg['weight_decay'])
    step=first=0;best=float('inf');spe=len(train.batches());total_steps=spe*cfg['epochs']
    if not args.bench and (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='mps',weights_only=False);assert ck['cfg']==cfg
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        step,first,best=ck['step'],ck['epoch'],ck['best_wer'];del ck
    emit(dict(event='start',variant=args.variant,train_utterances=len(train.rows),epochs=cfg['epochs'],
        steps_per_epoch=spe,resume_epoch=first,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),cfg=cfg))
    source_hashes={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in ['train.py','model.py','data.py','objective.py','config.json']}
    for epoch in range(first+1,cfg['epochs']+1):
        model.train();torch.manual_seed(cfg['seed']*1000+epoch);started=time.time();running=[]
        for group in train.batches(epoch):
            batch=train.collate(group,epoch,'mps')
            if step<cfg['warmup_steps']:factor=(step+1)/cfg['warmup_steps']
            else:factor=.1+.9*.5*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/max(1,total_steps-cfg['warmup_steps'])))
            for g in opt.param_groups:g['lr']=cfg['learning_rate']*factor
            opt.zero_grad(set_to_none=True);loss,parts=objective(model,bridge,batch,cfg,mean,std)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'])
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            opt.step();step+=1;running.append([float(loss.detach()),float(parts['mae']),float(parts['kl'])])
            if step%100==0:
                memory=torch.mps.driver_allocated_memory()/2**30
                if memory>cfg['max_driver_memory_gib']:raise RuntimeError('Declared GPU memory guard exceeded')
                emit(dict(event='train',epoch=epoch,step=step,loss=sum(v[0] for v in running[-100:])/len(running[-100:]),
                    mae=sum(v[1] for v in running[-100:])/len(running[-100:]),kl=sum(v[2] for v in running[-100:])/len(running[-100:]),seconds=time.time()-started,memory_gib=memory))
            if args.bench and len(running)>=args.bench:
                torch.mps.synchronize();emit(dict(event='benchmark_done',steps=len(running),seconds=time.time()-started,
                    memory_gib=torch.mps.driver_allocated_memory()/2**30,last_loss=running[-1][0]));return
        validation=evaluate_loss(model,bridge,dev,cfg,mean,std,'mps')
        # Select on actual free-running complete-utterance WER, not state loss.
        torch.manual_seed(cfg['seed']);cpu=Corrector(cfg,args.variant).eval()
        cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()})
        score,_=corpus(Inference(cfg,cpu),dev.rows)
        improved=score['wer_percent']<best;best=min(best,score['wer_percent'])
        record=dict(event='eval',epoch=epoch,step=step,train_loss=sum(v[0] for v in running)/len(running),
            validation=validation,selection_wer=score,seconds=time.time()-started)
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),cfg=cfg,variant=args.variant,
            epoch=epoch,step=step,best_wer=best,record=record,source_sha256=source_hashes)
        torch.save(ck,out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
        if improved:torch.save(ck,out/'best.tmp.pt');(out/'best.tmp.pt').replace(out/'best.pt')
        emit(record);del cpu,ck
    emit(dict(event='done',epochs=cfg['epochs'],best_selection_wer=best))


if __name__=='__main__':main()
