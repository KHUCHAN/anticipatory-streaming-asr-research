import argparse,fcntl,hashlib,json,math,random,time
import torch
from torch.nn import functional as F
from common import ROOT,OUT,HERE,config,datasets,sha,write
from model import Hybrid,objective

@torch.no_grad()
def evaluate(model,ds,cfg,device):
    model.eval();rng=random.Random(501)
    cpu_rng=torch.get_rng_state();mps_rng=torch.mps.get_rng_state();torch.manual_seed(502)
    sums=dict(nll=0.,hit1=0.,crps=0.);count=0;split_sums={}
    for ids in ds.batches(cfg):
        batch=ds.collate(ids,rng,device);logits,crps,_=model(batch,2)
        lp=torch.logsumexp(logits.log_softmax(-1),0)-math.log(len(logits));y=batch[-1];b=len(y)
        sums['nll']+=float(F.nll_loss(lp,y,reduction='sum'));sums['hit1']+=int((lp.argmax(-1)==y).sum());sums['crps']+=b*float(crps);count+=b
        nll=F.nll_loss(lp,y,reduction='none').tolist();hit=(lp.argmax(-1)==y).tolist()
        for j,ix in enumerate(ids):
            split=ds.rows[ix]['split'];s=split_sums.setdefault(split,dict(nll=0.,hit1=0.,examples=0))
            s['nll']+=nll[j];s['hit1']+=hit[j];s['examples']+=1
    torch.set_rng_state(cpu_rng);torch.mps.set_rng_state(mps_rng)
    model.train();return {k:v/count for k,v in sums.items()}|dict(examples=count,by_split={k:dict(nll=v['nll']/v['examples'],hit1=v['hit1']/v['examples'],examples=v['examples']) for k,v in split_sums.items()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--bench',type=int,default=0);a=p.parse_args()
    cfg=config();assert a.variant in cfg['variants'];device='mps';torch.set_num_threads(1);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    out=OUT/a.variant/('bench' if a.bench else '');out.mkdir(parents=True,exist_ok=True)
    lease=(out/'training.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX)
    model=Hybrid(a.variant).to(device);train,dev,probe=datasets(cfg)
    log=(out/'log.jsonl').open('a',buffering=1)
    def emit(x):print(json.dumps(x),flush=True);log.write(json.dumps(x)+'\n')
    initial=hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.98),weight_decay=cfg['weight_decay'])
    steps=len(train.batches(cfg));step=0;first=0;best=float('inf');total=steps*cfg['epochs']
    if (out/'last.pt').exists() and not a.bench:
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False);assert ck['cfg']==cfg
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];first=ck['epoch']+1;best=ck['best']
    emit(dict(event='start',variant=a.variant,parameters=sum(t.numel() for t in model.parameters()),ar_parameters=sum(t.numel() for t in model.ar.parameters()),train_utterances=len(train.rows),steps_per_epoch=steps,initial_sha256=initial,cfg=cfg))
    if first==0 and not a.bench:
        devstats=evaluate(model,dev,cfg,device);trainstats=evaluate(model,probe,cfg,device);best=devstats['nll']
        torch.save(dict(model=model.state_dict(),variant=a.variant,cfg=cfg,epoch=-1,step=0,best=best,dev=devstats,train_probe=trainstats,initial_sha256=initial),out/'best.pt')
        emit(dict(event='eval',epoch=0,step=0,dev=devstats,train_probe=trainstats,generalization_gap_nll=devstats['by_split']['dev-clean']['nll']-trainstats['nll'],seconds=0.))
    for epoch in range(first,cfg['epochs']):
        started=time.time();rng=random.Random(7000+epoch);torch.manual_seed(cfg['seed']*1000+epoch);losses=[]
        horizon=next(h for lo,hi,h in cfg['rollout_schedule'] if lo<=epoch+1<=hi)
        for ids in train.batches(cfg,8100+epoch):
            batch=train.collate(ids,rng,device);warm=cfg['warmup_steps']
            factor=min(1.,(step+1)/warm) if step<warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))
            for g in opt.param_groups:g['lr']=cfg['learning_rate']*factor
            opt.zero_grad(set_to_none=True);loss,parts=objective(model,batch,cfg,horizon)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(gn):raise RuntimeError('nonfinite gradient')
            opt.step();step+=1;losses.append(float(loss.detach()))
            if step%100==0:emit(dict(event='train',epoch=epoch+1,step=step,horizon=horizon,loss=sum(losses[-100:])/len(losses[-100:]),seconds=time.time()-started,memory_gib=torch.mps.driver_allocated_memory()/2**30))
            if a.bench and step>=a.bench:emit(dict(event='bench_done',steps=step,seconds=time.time()-started));return
        devstats=evaluate(model,dev,cfg,device);trainstats=evaluate(model,probe,cfg,device);best=min(best,devstats['nll'])
        ck=dict(model=model.state_dict(),variant=a.variant,cfg=cfg,epoch=epoch,step=step,best=best,optimizer=opt.state_dict(),dev=devstats,train_probe=trainstats,initial_sha256=initial,source_sha256={n:sha(HERE/n) for n in ['model.py','train.py','common.py','config.json']})
        torch.save(ck,out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
        if devstats['nll']<=best:torch.save(ck,out/'best.pt')
        emit(dict(event='eval',epoch=epoch+1,step=step,dev=devstats,train_probe=trainstats,generalization_gap_nll=devstats['by_split']['dev-clean']['nll']-trainstats['nll'],seconds=time.time()-started))
    emit(dict(event='done',epochs=cfg['epochs'],best=best))
if __name__=='__main__':main()
