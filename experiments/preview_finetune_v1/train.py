"""Fine-tunes a copy of the 20M encoder to read partly zero-filled chunks from native streaming state."""
import copy,json,math,os,random,sys,time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import torch
import torch.nn.functional as F

from shared import OUT,ROOT,config,role_rows
from torch_model import load

cfg=config();T=cfg['finetune'];W=cfg['feature_window'];H=cfg['feature_hop']
DEV=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BD=lambda i:2 if i//5 in (2,3,4) else 1  # batch dimension of each streaming state tensor


def chunks_of(row):return max(0,(row['feature_frames']-(W+1))//H+1)


def batch_features(fbank,rows,limit):
    L=min(min(chunks_of(r) for r in rows),limit);n=(L-1)*H+W
    return torch.from_numpy(np.stack([np.array(fbank[r['feature_offset']:r['feature_offset']+n],np.float32) for r in rows])),L


@torch.no_grad()
def teacher_pass(model,feats,L):
    """Native streaming pass. Returns per-chunk states, frozen log-probabilities and decoder inputs on the greedy path."""
    B=feats.shape[0];states=[s.repeat_interleave(B,BD(i)) for i,s in enumerate(model.encoder.get_init_state(DEV))];lens=torch.full((B,),W,device=DEV);kept=[];enc=[]
    for c in range(L):
        kept.append(states);out,_,states=model.encoder.streaming_forward(feats[:,c*H:c*H+W],lens,states);enc.append(model.joiner.encoder_proj(out))
    enc=torch.stack(enc);ctx=torch.zeros(B,2,dtype=torch.long,device=DEV);logp=[];dec=[]
    for c in range(L):
        for t in range(enc.shape[2]):
            d=model.joiner.decoder_proj(model.decoder(ctx,need_pad=False).squeeze(1));lp=model.joiner.output_linear(torch.tanh(enc[c,:,t]+d)).log_softmax(-1)
            tok=lp.argmax(-1);emit=(tok!=0)&(tok!=2);ctx=torch.where(emit[:,None],torch.stack([ctx[:,1],tok],1),ctx);logp.append(lp);dec.append(d)
    shape=(L,enc.shape[2],B,-1);logp=torch.stack(logp).view(*shape).transpose(1,2);dec=torch.stack(dec).view(*shape).transpose(1,2)
    return kept,logp,dec  # logp, dec: (L,B,8,*)


def pool(kept,feats,logp,dec,L):
    """Flattens chunk-major: example index = c*B+b."""
    states=[torch.cat([k[i] for k in kept],BD(i)) for i in range(len(kept[0]))]
    x=torch.stack([feats[:,c*H:c*H+W] for c in range(L)]).flatten(0,1)
    return states,x,logp.flatten(0,1),dec.flatten(0,1)


def student_logp(model,student,states,x,dec,index,k):
    xs=x[index].clone();mask=torch.arange(W,device=DEV)[None,:]>=k[:,None];xs[mask]=0.0
    st=[s.index_select(BD(i),index) for i,s in enumerate(states)];out,_,_=student['encoder'].streaming_forward(xs,torch.full((len(index),),W,device=DEV),st)
    return model.joiner.output_linear(torch.tanh(student['proj'](out)+dec[index])).log_softmax(-1)


@torch.no_grad()
def validate(model,student,fbank,rows):
    out={}
    for k in cfg['stage2']['ks']:
        kl=0.0;n=0;hit=0;temit=0;semit=0;sok=0
        for i in range(0,len(rows),8):
            feats,L=batch_features(fbank,rows[i:i+8],10**9);feats=feats.to(DEV)
            kept,lp,d=teacher_pass(model,feats,L);states,x,logp,dec=pool(kept,feats,lp,d,L);index=torch.arange(x.shape[0],device=DEV)
            for j in range(0,len(index),256):
                ix=index[j:j+256];s=student_logp(model,student,states,x,dec,ix,torch.full((len(ix),),k,device=DEV));t=logp[ix]
                kl+=float((t.exp()*(t-s)).sum());n+=t.shape[0]*t.shape[1];a=t.argmax(-1);b=s.argmax(-1);te=(a!=0)&(a!=2);se=(b!=0)&(b!=2)
                temit+=int(te.sum());semit+=int(se.sum());hit+=int((te&(a==b)).sum());sok+=int((se&(a==b)).sum())
        out[f'k{k}']=dict(kl=kl/n,recall=hit/max(temit,1),precision=sok/max(semit,1))
    return out


def main():
    minutes=float(sys.argv[1]) if len(sys.argv)>1 else T['minutes'];tag=sys.argv[2] if len(sys.argv)>2 else 'student'
    torch.manual_seed(cfg['seed']);rng=random.Random(cfg['seed']);OUT.mkdir(parents=True,exist_ok=True)
    a2cfg=json.loads((ROOT/cfg['a2_config']).read_text());cache=ROOT/a2cfg['input_cache'];index=json.loads((cache/'index.json').read_text())
    fbank=np.memmap(cache/'fbank.f16',np.float16,'r').reshape(-1,80)
    held={r['utterance_id'] for r in role_rows(cfg,'train')}
    rows=sorted([r for r in index if r['split']==cfg['stage2']['train_split'] and r['utterance_id'] not in held and chunks_of(r)>=4],key=chunks_of)
    val=role_rows(cfg,'selection')[:T['validation_utterances']];val=sorted(val,key=chunks_of)
    model=load().to(DEV)
    for p in model.parameters():p.requires_grad_(False)
    student=dict(encoder=copy.deepcopy(model.encoder),proj=copy.deepcopy(model.joiner.encoder_proj));params=[p for m in student.values() for p in m.parameters()]
    for p in params:p.requires_grad_(True)
    opt=torch.optim.AdamW(params,lr=T['lr'],weight_decay=0.0);B=T['batch_utterances'];M=T['examples_per_step']
    log=open(OUT/f'{tag}.log','a');first=validate(model,student,fbank,val);print(json.dumps(dict(step=0,val=first)),file=log,flush=True);print('start',len(rows),first,flush=True)
    start=time.time();step=0;last=start;running=0.0;count=0
    while (elapsed:=(time.time()-start)/60)<minutes:
        i=rng.randrange(0,len(rows)-B);feats,L=batch_features(fbank,rows[i:i+B],T['max_chunks']);feats=feats.to(DEV)
        kept,lp,d=teacher_pass(model,feats,L);states,x,logp,dec=pool(kept,feats,lp,d,L);order=torch.randperm(x.shape[0],device=DEV)
        for j in range(0,len(order)-M//2,M):
            ix=order[j:j+M];k=torch.randint(T['k_min'],T['k_max']+1,(len(ix),),device=DEV)
            s=student_logp(model,student,states,x,dec,ix,k);t=logp[ix];loss=(t.exp()*(t-s)).sum(-1).mean()
            lr=T['lr']*min(1,(step+1)/T['warmup_steps'])*(0.05+0.95*0.5*(1+math.cos(math.pi*min(1,elapsed/minutes))))
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,T['grad_clip']);opt.step();step+=1;running+=float(loss);count+=1
        if time.time()-last>600 or minutes<5 and step%20==0:
            last=time.time();v=validate(model,student,fbank,val);print(json.dumps(dict(step=step,minutes=round(elapsed,1),train=running/count,val=v)),file=log,flush=True);running=0.0;count=0
            torch.save({k:m.state_dict() for k,m in student.items()},OUT/f'{tag}.pt')
    v=validate(model,student,fbank,val);torch.save({k:m.state_dict() for k,m in student.items()},OUT/f'{tag}.pt')
    (OUT/f'{tag}.json').write_text(json.dumps(dict(steps=step,minutes=minutes,initial=first,final=v,train_utterances=len(rows)),indent=1));print('done',step,json.dumps(v),file=log,flush=True);print('done',step,v)


if __name__=='__main__':main()
