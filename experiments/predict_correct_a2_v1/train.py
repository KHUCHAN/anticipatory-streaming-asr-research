#!/usr/bin/env python3
"""Two declared phases; MAE checkpoint is preserved before calibrated KL."""
import argparse
import fcntl
import json
import math
import random
import statistics
import time

import torch

from common import ROOT,HERE,OUT,config,sha,state_module,write
from data import datasets
from inference import Inference,corpus
from model import A2
from objective import losses


def gradient_stats(model,bridge,batch,cfg,mean,std):
    mae,kl=losses(model,bridge,batch,cfg,mean,std,True)
    parameters=tuple(model.audio.parameters())
    a=torch.autograd.grad(mae,parameters,retain_graph=True);b=torch.autograd.grad(kl,parameters)
    aa=float(torch.stack([x.square().sum() for x in a]).sum())
    bb=float(torch.stack([x.square().sum() for x in b]).sum())
    ab=float(torch.stack([(x*y).sum() for x,y in zip(a,b)]).sum())
    return dict(mae=float(mae.detach()),kl=float(kl.detach()),mae_gradient_norm=aa**.5,kl_gradient_norm=bb**.5,
        raw_kl_to_mae_ratio=(bb/max(aa,1e-30))**.5,cosine=ab/max((aa*bb)**.5,1e-30))


@torch.inference_mode()
def validate(model,bridge,data,cfg,mean,std,with_kl,device):
    sums=dict(mae=0.,kl=0.,constant_mae=0.);frames=0
    model.eval()
    for indices in data.batches():
        batch,_=data.collate(indices,0,device);mae,kl=losses(model,bridge,batch,cfg,mean,std,with_kl)
        valid=batch[-1];n=int(valid.sum());frames+=n;sums['mae']+=float(mae)*n
        if kl is not None:sums['kl']+=float(kl)*n
        constant=((((batch[2]-mean)/std).abs().mean(-1))*valid).sum()/valid.sum()
        sums['constant_mae']+=float(constant)*n
    model.train();return {k:v/frames for k,v in sums.items()}


def selection(model,cfg,dev):
    cpu=A2(cfg).eval();cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()})
    infer=Inference(cfg,cpu);result={}
    for split in ['dev-clean','dev-other']:result[split]=corpus(infer,[r for r in dev.rows if r['split']==split])[0]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['mae','kl'],required=True);parser.add_argument('--bench',type=int,default=0)
    args=parser.parse_args();cfg=config();torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    assert torch.backends.mps.is_available(),'Approved Mac GPU access is required.'
    out=OUT/args.phase/('benchmark' if args.bench else '');out.mkdir(parents=True,exist_ok=True)
    lease=(out/'training.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=A2(cfg).to('mps');bridge=state_module.FrozenRNNT(ROOT/cfg['bridge']).to('mps').eval()
    train,dev,_=datasets(cfg);stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text())
    mean=torch.tensor(stats['mean'],device='mps');std=torch.tensor(stats['std'],device='mps')
    log=(out/'log.jsonl').open('a',buffering=1)
    def emit(value):print(json.dumps(value),flush=True);log.write(json.dumps(value)+'\n')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
    else:write(out/'config.json',cfg)
    first=step=0;best=float('inf');weight=0.;epochs=cfg[args.phase+'_epochs']
    if args.phase=='kl':
        ck=torch.load(OUT/'mae/best.pt',map_location='mps',weights_only=False);model.load_state_dict(ck['model'])
        assert ck['record']['validation']['mae']<ck['record']['validation']['constant_mae'],'Representation gate not passed'
        path=OUT/'kl_calibration.json'
        if path.exists():calibration=json.loads(path.read_text())
        else:
            indices=random.Random(cfg['seed']+224).sample(range(len(train.rows)),cfg['kl_calibration_batches']*cfg['kl_calibration_batch_size'])
            values=[]
            for lo in range(0,len(indices),cfg['kl_calibration_batch_size']):
                batch,_=train.collate(indices[lo:lo+cfg['kl_calibration_batch_size']],cfg['mae_epochs']+1,'mps')
                values.append(gradient_stats(model,bridge,batch,cfg,mean,std))
            ratio=statistics.median(v['mae_gradient_norm']/max(v['kl_gradient_norm'],1e-12) for v in values)
            chosen=max(cfg['kl_weight_min'],min(cfg['kl_weight_max'],cfg['kl_trunk_gradient_target_ratio']*ratio))
            calibration=dict(weight=chosen,batches=values,source_mae_sha256=sha(OUT/'mae/best.pt'),scope='One-time shared-audio-trunk gradient calibration on training data only')
            write(path,calibration);write(HERE/'artifacts/kl_calibration.json',calibration)
        weight=calibration['weight'];emit(dict(event='kl_calibration',**calibration))
    lr=cfg[args.phase+'_learning_rate'];opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=cfg['weight_decay'])
    if (out/'last.pt').exists() and not args.bench:
        ck=torch.load(out/'last.pt',map_location='mps',weights_only=False);assert ck['cfg']==cfg
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['epoch'];step=ck['step'];best=ck['best_metric']
    source={str(p.relative_to(ROOT)):sha(p) for p in [HERE/name for name in ['common.py','config.json','model.py','data.py','objective.py','inference.py','train.py']]}
    spe=len(train.batches());total=spe*epochs
    emit(dict(event='start',phase=args.phase,epochs=epochs,resume_epoch=first,steps_per_epoch=spe,parameters=sum(p.numel() for p in model.parameters())))
    for epoch in range(first+1,epochs+1):
        torch.manual_seed(cfg['seed']*1000+epoch+(0 if args.phase=='mae' else cfg['mae_epochs']));started=time.time();recent=[]
        sample_epoch=epoch+(0 if args.phase=='mae' else cfg['mae_epochs']);current_weight=weight*min(1.,epoch/cfg['kl_weight_warmup_epochs'])
        for indices in train.batches(sample_epoch):
            batch,_=train.collate(indices,sample_epoch,'mps')
            if step<cfg['warmup_steps']:factor=(step+1)/cfg['warmup_steps']
            else:factor=.1+.9*.5*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/max(1,total-cfg['warmup_steps'])))
            for group in opt.param_groups:group['lr']=lr*factor
            opt.zero_grad(set_to_none=True);mae,kl=losses(model,bridge,batch,cfg,mean,std,args.phase=='kl')
            loss=mae if kl is None else mae+current_weight*kl
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            opt.step();step+=1;recent.append((float(mae.detach()),float(kl.detach()) if kl is not None else 0.))
            if step%100==0:
                memory=torch.mps.driver_allocated_memory()/2**30
                if memory>cfg['max_driver_memory_gib']:raise RuntimeError('GPU allocation guard exceeded')
                emit(dict(event='train',phase=args.phase,epoch=epoch,step=step,mae=sum(v[0] for v in recent[-100:])/len(recent[-100:]),kl=sum(v[1] for v in recent[-100:])/len(recent[-100:]),weight=current_weight,seconds=time.time()-started,memory_gib=memory))
            if args.bench and len(recent)>=args.bench:
                torch.mps.synchronize();emit(dict(event='benchmark_done',steps=len(recent),seconds=time.time()-started,memory_gib=torch.mps.driver_allocated_memory()/2**30));return
        val=validate(model,bridge,dev,cfg,mean,std,args.phase=='kl','mps');scores=selection(model,cfg,dev)
        diagnostic=None
        if args.phase=='kl':
            batch,_=train.collate(list(range(4)),cfg['mae_epochs']+1,'mps');diagnostic=gradient_stats(model,bridge,batch,cfg,mean,std)
            diagnostic['weighted_kl_to_mae_ratio']=current_weight*diagnostic['raw_kl_to_mae_ratio']
        metric=val['mae'] if args.phase=='mae' else scores['dev-clean']['wer_percent'];improved=metric<best;best=min(best,metric)
        record=dict(event='eval',phase=args.phase,epoch=epoch,step=step,validation=val,selection=scores,gradient=diagnostic,weight=current_weight,
            train_mae=sum(v[0] for v in recent)/len(recent),train_kl=sum(v[1] for v in recent)/len(recent),seconds=time.time()-started)
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),cfg=cfg,phase=args.phase,epoch=epoch,step=step,best_metric=best,record=record,source_sha256=source)
        torch.save(checkpoint,out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
        if improved:torch.save(checkpoint,out/'best.tmp.pt');(out/'best.tmp.pt').replace(out/'best.pt')
        emit(record)
    emit(dict(event='done',phase=args.phase,epochs=epochs,best_metric=best))


if __name__=='__main__':main()
