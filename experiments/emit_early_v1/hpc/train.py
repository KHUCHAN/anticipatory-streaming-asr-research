"""Fine-tunes the 20M transducer with the pruned RNN-T loss and a k2 delay penalty. Usage: train.py TAG DELAY_PENALTY MINUTES [LR]
Environment: TRAIN_SPLITS (comma list), SA=1 (SpecAugment), SC=1 (train on 160/320ms chunks only), DECAY=1 (linear lr decay to 10%), MAX_FRAMES."""
import json,math,os,random,sys,time
import numpy as np,torch,k2,sentencepiece as spm
from model_def import ASR,load

TAG=sys.argv[1];DP=float(sys.argv[2]);MINUTES=float(sys.argv[3]);LR=float(sys.argv[4]) if len(sys.argv)>4 else 0.002
MAXF=int(os.environ.get('MAX_FRAMES','50000'));SA=os.environ.get('SA','0')=='1';SC=os.environ.get('SC','0')=='1';DECAY=os.environ.get('DECAY','0')=='1';EMA=0.999;SEED=6137;PAD=math.log(1e-10);dev=torch.device('cuda')


def loss_fn(model,x,x_lens,ys,dp):
    enc,enc_lens=model.encoder(x,x_lens);N=len(ys);U=max(len(y) for y in ys)
    y_pad=torch.zeros(N,U,dtype=torch.int64,device=dev)
    for i,y in enumerate(ys):y_pad[i,:len(y)]=torch.tensor(y)
    dec=model.decoder(torch.cat([torch.zeros(N,1,dtype=torch.int64,device=dev),y_pad],1))
    boundary=torch.zeros(N,4,dtype=torch.int64,device=dev);boundary[:,2]=torch.tensor([len(y) for y in ys]);boundary[:,3]=enc_lens
    simple,(px,py)=k2.rnnt_loss_smoothed(lm=model.simple_lm_proj(dec).float(),am=model.simple_am_proj(enc).float(),symbols=y_pad,termination_symbol=0,lm_only_scale=0.25,am_only_scale=0.0,
        boundary=boundary,reduction='sum',delay_penalty=dp,return_grad=True)
    ranges=k2.get_rnnt_prune_ranges(px_grad=px,py_grad=py,boundary=boundary,s_range=5)
    am,lm=k2.do_rnnt_pruning(am=model.joiner.encoder_proj(enc),lm=model.joiner.decoder_proj(dec),ranges=ranges)
    pruned=k2.rnnt_loss_pruned(logits=model.joiner(am,lm,project_input=False).float(),symbols=y_pad,ranges=ranges,termination_symbol=0,boundary=boundary,reduction='sum',delay_penalty=dp)
    return 0.5*simple+pruned,float(simple),float(pruned),int(enc_lens.sum())


def spec_augment(x,lens,rng):
    """Two frequency masks (<=27 bins) and up to ten time masks (<=100 frames, at most 20% of the utterance), filled with the utterance mean."""
    for i,n in enumerate(lens):
        mean=float(x[i,:n].mean())
        for _ in range(2):
            w=rng.randint(0,27);f=rng.randint(0,80-w);x[i,:n,f:f+w]=mean
        budget=int(0.2*n)
        for _ in range(10):
            w=min(rng.randint(0,100),budget);budget-=w;t=rng.randint(0,max(n-w,0));x[i,t:t+w]=mean
    return x


def main():
    torch.manual_seed(SEED);rng=random.Random(SEED);out=ASR/'ckpt'/TAG;out.mkdir(parents=True,exist_ok=True);log=open(out/'train.log','a')
    splits=os.environ.get('TRAIN_SPLITS','train-clean-100').split(',');banks=[np.memmap(ASR/f'data/fbank_{s}.f16',np.float16,'r').reshape(-1,80) for s in splits];index=[]
    for n,s in enumerate(splits):index+=[dict(r,bank=n) for r in json.load(open(ASR/f'data/index_{s}.json'))]
    sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'));rows=sorted(index,key=lambda r:r['feature_frames']);tokens=[sp.encode(r['text']) for r in rows]
    batches=[];cur=[];width=0
    for i,r in enumerate(rows):
        if cur and max(width,r['feature_frames'])*(len(cur)+1)>MAXF:batches.append(cur);cur=[];width=0
        cur.append(i);width=max(width,r['feature_frames'])
    batches.append(cur)
    model=load().to(dev).train()
    if SC:model.encoder.short_chunk_size=16;model.encoder.short_chunk_threshold=1.0  # training chunks of 160 or 320ms only, never full attention
    for m in model.modules():
        if hasattr(m,'batch_count'):m.batch_count=100000.0
    from optim import ScaledAdam
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=ScaledAdam(model.parameters(),lr=LR,clipping_scale=2.0,parameters_names=[[n for n,_ in model.named_parameters()]])
    print(json.dumps(dict(specaug=SA,streaming_chunks_only=SC,lr_decay=DECAY,ema=EMA,splits=splits,tag=TAG,delay_penalty=DP,minutes=MINUTES,lr=LR,utterances=len(rows),batches=len(batches),gpu=torch.cuda.get_device_name(0))),file=log,flush=True)
    start=time.time();step=0;epoch=0;acc=[0.0,0.0,0];saved=start
    while time.time()-start<60*MINUTES:
        order=list(range(len(batches)));rng.shuffle(order)
        for b in order:
            ids=batches[b];T=max(rows[i]['feature_frames'] for i in ids);x=torch.full((len(ids),T,80),PAD)
            for j,i in enumerate(ids):
                r=rows[i];x[j,:r['feature_frames']]=torch.from_numpy(np.array(banks[r['bank']][r['feature_offset']:r['feature_offset']+r['feature_frames']],np.float32))
            if SA:x=spec_augment(x,[rows[i]['feature_frames'] for i in ids],rng)
            loss,s,p,frames=loss_fn(model,x.to(dev),torch.tensor([rows[i]['feature_frames'] for i in ids],device=dev),[tokens[i] for i in ids],DP)
            if not math.isfinite(float(loss)):print(json.dumps(dict(step=step,skipped='nonfinite')),file=log,flush=True);opt.zero_grad(set_to_none=True);continue
            if DECAY:
                for g in opt.param_groups:g['lr']=LR*(1-0.9*min(1.0,(time.time()-start)/(60*MINUTES)))  # linear decay to 10% over the time budget
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();step+=1
            with torch.no_grad():
                for k,v in model.state_dict().items():
                    if v.dtype.is_floating_point:ema[k].mul_(EMA).add_(v.detach(),alpha=1-EMA)
                    else:ema[k].copy_(v)
            acc[0]+=s;acc[1]+=p;acc[2]+=frames
            if step%100==0:
                print(json.dumps(dict(step=step,epoch=epoch,minutes=round((time.time()-start)/60,1),simple=acc[0]/acc[2],pruned=acc[1]/acc[2])),file=log,flush=True);acc=[0.0,0.0,0]
            if time.time()-saved>1200:saved=time.time();torch.save(dict(model=model.state_dict(),ema=ema,step=step),out/'last.pt')
            if time.time()-start>=60*MINUTES:break
        epoch+=1
    torch.save(dict(model=model.state_dict(),ema=ema,step=step,delay_penalty=DP,minutes=MINUTES,lr=LR),out/'last.pt');print(json.dumps(dict(done=step,epochs=epoch)),file=log,flush=True);print('done',TAG,step)


if __name__=='__main__':main()
