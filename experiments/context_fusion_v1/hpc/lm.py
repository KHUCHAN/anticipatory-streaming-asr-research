"""Trains the LSTM token LM on $ASR/data/lm_tokens.u16. Usage: lm.py TAG MINUTES
Keeps the weights with the lowest loss on held-out training sentences in $ASR/ckpt/TAG/lm.pt and reports perplexity, next-token accuracy and confident-prediction shares on dev transcripts.
A fifth of every batch is drawn from the 960h transcripts, the rest from all sentences."""
import json,math,os,sys,time
import numpy as np,torch,sentencepiece as spm
from model_def import ASR,LM

TAG=sys.argv[1];MINUTES=float(sys.argv[2]);LR=float(os.environ.get('LM_LR','0.002'));B=int(os.environ.get('LM_BATCH','256'));EVERY=int(os.environ.get('LM_EVERY','2000'));L=128;SEED=6137
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad(seqs):
    w=max(len(s) for s in seqs);x=torch.zeros(len(seqs),w,dtype=torch.int64);y=torch.full((len(seqs),w-1),-100)
    for j,s in enumerate(seqs):x[j,:len(s)]=torch.as_tensor(s);y[j,:len(s)-1]=torch.as_tensor(s[1:])
    return x[:,:-1].to(dev),y.to(dev)


@torch.no_grad()
def score(model,seqs):
    """Mean negative log-likelihood per token, and per token: top probability, whether the top choice is right, the target."""
    model.eval();nll=0.0;conf=[];ok=[];tgt=[]
    for i in range(0,len(seqs),256):
        x,y=pad(seqs[i:i+256]);lp=model(x)[0].log_softmax(-1);m=y>=0;p,t=lp.max(-1)
        nll-=float(lp.gather(-1,y.clamp(min=0)[...,None])[...,0][m].sum());conf.append(p[m].exp().cpu());ok.append((t==y)[m].cpu());tgt.append(y[m].cpu())
    model.train();conf=torch.cat(conf);return nll/len(conf),conf,torch.cat(ok),torch.cat(tgt)


def main():
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED);out=ASR/'ckpt'/TAG;out.mkdir(parents=True,exist_ok=True);log=open(out/'lm.log','a')
    tok=np.fromfile(ASR/'data/lm_tokens.u16',np.uint16);starts=np.flatnonzero(tok==0);ends=np.append(starts[1:],len(tok));valid=np.flatnonzero(ends-starts>=2)
    nt=int((valid<json.load(open(ASR/'data/lm_meta.json'))['transcript_sentences']).sum());seq=lambda v:tok[starts[v]:min(ends[v],starts[v]+L+1)].astype(np.int64)
    held=rng.choice(len(valid),min(3000,len(valid)//20),replace=False);is_held=np.zeros(len(valid),bool);is_held[held]=True;held=[seq(v) for v in valid[held]]
    model=LM().to(dev).train();opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.0);torch.backends.cudnn.benchmark=True
    print(json.dumps(dict(tag=TAG,minutes=MINUTES,lr=LR,batch=B,tokens=len(tok),sentences=len(valid),transcript_sentences=nt,parameters=sum(p.numel() for p in model.parameters()))),file=log,flush=True)
    start=time.time();step=0;best=math.inf;acc=0.0;seen=0
    while time.time()-start<60*MINUTES:
        idx=np.concatenate([rng.integers(0,nt,B//5),rng.integers(0,len(valid),B-B//5)]);x,y=pad([seq(v) for v in valid[idx[~is_held[idx]]]])
        for g in opt.param_groups:g['lr']=LR*(1-0.9*min(1.0,(time.time()-start)/(60*MINUTES)))
        loss=torch.nn.functional.cross_entropy(model(x)[0].reshape(-1,500),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();step+=1;acc+=float(loss.detach());seen+=int((y>=0).sum())
        if step%EVERY==0:
            h=score(model,held)[0]
            if h<best:best=h;torch.save(dict(model=model.state_dict(),step=step,held_nll=h),out/'lm.pt')
            print(json.dumps(dict(step=step,minutes=round((time.time()-start)/60,1),train=acc/EVERY,held=h,tokens_seen=seen)),file=log,flush=True);acc=0.0
    h=score(model,held)[0]
    if h<best:best=h;torch.save(dict(model=model.state_dict(),step=step,held_nll=h),out/'lm.pt')
    model.load_state_dict(torch.load(out/'lm.pt',map_location=dev,weights_only=False)['model']);sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'))
    start_piece=torch.tensor([sp.id_to_piece(i).startswith('▁') for i in range(500)]);parts=['steps %d tokens %.0fM held ppl %.1f'%(step,seen/1e6,math.exp(best))]
    for s in ['dev-clean','dev-other']:
        rows=json.load(open(ASR/f'data/index_{s}.json'))
        if 'text' not in rows[0]:continue
        nll,conf,ok,tgt=score(model,[[0]+sp.encode(r['text'].upper())[:L] for r in rows]);ws=start_piece[tgt];text='%s ppl %.1f acc %.3f word-start acc %.3f'%(s,math.exp(nll),ok.float().mean(),ok[ws].float().mean())
        for c in (0.9,0.99):m=conf>=c;text+=' | p>=%g share %.3f acc %.3f'%(c,m.float().mean(),ok[m].float().mean() if m.any() else 0.0)
        parts.append(text)
    print('LM',TAG,'|',' || '.join(parts),flush=True)


if __name__=='__main__':main()
