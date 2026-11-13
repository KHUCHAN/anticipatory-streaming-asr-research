#!/usr/bin/env python3
"""Train the CRPS-only control; no token-loss gradients and CRPS-only selection."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import torch
from model import StateFGN,FrozenRNNT,rollout_loss
from data import ROOT,OUT,PackedSet,selections


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def horizon_for(epoch,cfg,variant):
    if variant.startswith('single_step'):return 1
    for first,last,h in cfg['rollout_schedule']:
        if first<=epoch+1<=last:return h
    return cfg['rollout_schedule'][-1][-1]


@torch.no_grad()
def evaluate(model,bridge,ds,cfg,mean,std,device):
    cpu_state=torch.get_rng_state();mps_state=torch.mps.get_rng_state() if device=='mps' else None
    torch.manual_seed(509);model.eval();rng=random.Random(510);total=0.;sums={k:0. for k in ['loss','state_crps','token_nll']}
    for group in ds.batches(cfg):
        x,c,y,cut,lengths=ds.collate(group,cfg,rng,device);x=(x-mean)/std
        loss,parts=rollout_loss(model,bridge,x,c,y,cut,lengths,mean,std,4)
        n=len(cut);total+=n;sums['loss']+=float(loss)*n
        for k in parts:sums[k]+=float(parts[k])*n
    torch.set_rng_state(cpu_state)
    if device=='mps':torch.mps.set_rng_state(mps_state)
    model.train();return {k:v/total for k,v in sums.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--device',default='mps')
    p.add_argument('--bench',type=int,default=0);p.add_argument('--horizon',type=int,default=1);p.add_argument('--deadline',type=float,default=0.)
    a=p.parse_args();cfg=json.loads((ROOT/'experiments/fgn_state_v1/config_crps_only.json').read_text())
    assert a.variant in cfg['variants'];seed=int(a.variant.rsplit('s',1)[1])
    torch.set_num_threads(1);torch.manual_seed(seed);random.seed(seed)
    model=StateFGN(cfg,a.variant).to(a.device);bridge=FrozenRNNT(OUT/'rnnt_bridge.npz').to(a.device).eval()
    stats=json.loads((OUT/'cache/stats.json').read_text());mean=torch.tensor(stats['mean'],device=a.device);std=torch.tensor(stats['std'],device=a.device)
    train_rows,dev_rows=selections(cfg);train=PackedSet(train_rows);dev=PackedSet(dev_rows)
    out=OUT/a.variant
    if a.bench:out=out/f'bench_h{a.horizon}'
    out.mkdir(parents=True,exist_ok=True);log=(out/'log.jsonl').open('a',buffering=1)
    def emit(v):line=json.dumps(v);print(line,flush=True);log.write(line+'\n')
    initial=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.98),weight_decay=cfg['weight_decay'])
    spe=len(train.batches(cfg));total_steps=spe*cfg['epochs'];step=0;first_epoch=0;best=float('inf')
    if (out/'last.pt').exists() and not a.bench:
        ck=torch.load(out/'last.pt',map_location=a.device,weights_only=False)
        assert ck['cfg']==cfg and ck['variant']==a.variant
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first_epoch=ck['epoch']+1;step=ck['step'];best=ck['best']
    emit(dict(event='start',variant=a.variant,params=sum(t.numel() for t in model.parameters()),initial_state_sha256=initial,
              epochs=cfg['epochs'],steps_per_epoch=spe,device=a.device,cfg=cfg))
    for epoch in range(first_epoch,cfg['epochs']):
        rng=random.Random(7000+epoch);torch.manual_seed(seed*1000+epoch);start=time.time();agg=[]
        horizon=a.horizon if a.bench else horizon_for(epoch,cfg,a.variant)
        for group in train.batches(cfg,8100+epoch):
            x,c,y,cut,lengths=train.collate(group,cfg,rng,a.device);x=(x-mean)/std
            factor=min(1.,(step+1)/cfg['warmup_steps']) if step<cfg['warmup_steps'] else .1+.9*.5*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/max(1,total_steps-cfg['warmup_steps'])))
            for g in opt.param_groups:g['lr']=cfg['learning_rate']*factor
            opt.zero_grad(set_to_none=True);loss,parts=rollout_loss(model,bridge,x,c,y,cut,lengths,mean,std,horizon)
            if not torch.isfinite(loss):emit(dict(event='abort',reason='nonfinite loss',step=step));raise RuntimeError('nonfinite loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            if not torch.isfinite(gn):raise RuntimeError('nonfinite gradient')
            opt.step();step+=1;agg.append([float(loss.detach()),float(parts['state_crps']),float(parts['token_nll'])])
            if step%100==0:
                torch.mps.synchronize() if a.device=='mps' else None
                recent=torch.tensor(agg[-100:]).mean(0).tolist()
                emit(dict(event='train',epoch=epoch,step=step,horizon=horizon,loss=recent[0],state_crps=recent[1],token_nll=recent[2],
                          epoch_seconds=time.time()-start,memory_gb=torch.mps.driver_allocated_memory()/2**30 if a.device=='mps' else None))
            if a.bench and step>=a.bench:
                torch.mps.synchronize() if a.device=='mps' else None
                emit(dict(event='bench_done',steps=step,horizon=horizon,seconds=time.time()-start));return
        ev=evaluate(model,bridge,dev,cfg,mean,std,a.device);best=min(best,ev['loss'])
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),cfg=cfg,variant=a.variant,epoch=epoch,step=step,best=best,validation=ev,
                initial_state_sha256=initial,source_sha256={n:digest(ROOT/'experiments/fgn_state_v1'/n) for n in ['model.py','data.py','train_crps_only.py','config_crps_only.json']})
        torch.save(ck,out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
        if ev['loss']<=best:torch.save(ck,out/'best.pt')
        emit(dict(event='eval',epoch=epoch,step=step,horizon=horizon,validation=ev,epoch_seconds=time.time()-start))
        if a.deadline and time.time()>a.deadline:
            emit(dict(event='budget_stop',completed_epochs=epoch+1));return
    emit(dict(event='done',epochs=cfg['epochs'],best_validation=best))


if __name__=='__main__':main()
