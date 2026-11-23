"""Stage 2 collector: preview passes with training-free and learned fillers, with per-token confidence features.

Usage: collect.py ROLE TAG FILLER[,FILLER...] [LIMIT]
Writes runs/preview_filler_v1/collect_ROLE_TAG.json. Preview passes never update native state.
"""
import json
import sys
import time

import numpy as np
import soundfile as sf
import torch

from shared import OUT,ROOT,config,features,role_rows,write
import inference


def decode(graph,encoder_out,context,pn,detail):
    ids=[];p=[];pb=[];mg=[];fr=[]
    for f,state in enumerate(encoder_out):
        logits=graph.joiner.run(None,{'encoder_out':np.ascontiguousarray(state[None]),'decoder_out':pn})[0][0]
        token=int(logits.argmax())
        if token in (0,2):continue
        if detail:
            q=np.exp(logits.astype(np.float64)-logits.max());q/=q.sum();top=np.sort(q)[-2:]
            p.append(round(float(q[token]),4));pb.append(round(float(q[0]),4));mg.append(round(float(top[1]-top[0]),4));fr.append(f)
        ids.append(token);context=np.concatenate([context[:,1:],np.array([[token]],np.int64)],1);pn=graph.decoder.run(None,{'y':context})[0]
    return dict(ids=ids,p=p,pb=pb,mg=mg,fr=fr),context,pn


def main():
    role,tag,names=sys.argv[1],sys.argv[2],sys.argv[3].split(',');limit=int(sys.argv[4]) if len(sys.argv)>4 else None
    cfg=config();s=cfg['stage2'];a2cfg=json.loads((ROOT/cfg['a2_config']).read_text());torch.set_num_threads(1)
    inf=inference.Inference(a2cfg);graph=inf.graph;W=cfg['feature_window'];H=cfg['feature_hop'];C=s['context_frames']
    stats=json.loads((ROOT/a2cfg['input_cache']/'stats.json').read_text());mean=np.array(stats['mean'],np.float32);std=np.array(stats['std'],np.float32)
    models={}
    if any(n=='det' for n in names):
        from predictor import load;models['det']=load('det')
    if any(n.startswith('fgn') for n in names):
        from predictor import load;models['fgn']=load('fgn')

    @torch.inference_mode()
    def fill(f,c,k,name,seed):
        x=np.ascontiguousarray(f[c*H:c*H+W]).copy();seen=c*H+k;missing=W-k
        if name=='zero':x[k:]=0.0
        elif name=='repeat':x[k:]=x[k-1]
        elif name=='mean':x[k:]=f[:seen].mean(0)
        elif name=='fade':
            w=(np.arange(1,missing+1,dtype=np.float32)/missing)[:,None];x[k:]=(1-w)*x[k-1]+w*f[:seen].mean(0)
        else:
            past=f[max(0,seen-C):seen]
            if len(past)<C:past=np.concatenate([np.repeat(past[:1],C-len(past),0),past])
            ctx=torch.from_numpy((past-mean)/std)[None]
            if name=='det':y=models['det'](ctx)[0]
            else:
                n=1 if name=='fgn1' else s['fgn_mean_samples'];g=torch.Generator().manual_seed(seed)
                y=models['fgn'](ctx.expand(n,-1,-1),torch.randn(n,models['fgn'].noise_dim,generator=g)).mean(0)
            x[k:]=y.numpy()[:missing]*std+mean
        return x

    selected=role_rows(cfg,role);selected=selected[:limit] if limit else selected;variants=[(n,k) for n in names for k in s['ks']];records=[]
    if role=='eval':inf.utterance(selected[0])
    for n,row in enumerate(selected):
        wave,sr=sf.read(ROOT/row['audio'],dtype='float32');assert sr==16000
        begin=time.perf_counter();f=features(np.concatenate([wave,np.zeros(10560,np.float32)]));feature_seconds=time.perf_counter()-begin
        count=max(0,(len(f)-(W+1))//H+1);states=graph.states();context=np.array([[-1]*(graph.context-1)+[0]],np.int64);pn=graph.decoder.run(None,{'y':context})[0];chunks=[]
        for c in range(count):
            previews={}
            for name,k in variants:
                begin=time.perf_counter();x=fill(f,c,k,name,1000003*n+97*c+k)[None]
                enc=graph.encoder.run(['encoder_out'],{'x':x,**states})[0][0];out,_,_=decode(graph,enc,context,pn,True)
                out['seconds']=time.perf_counter()-begin;previews[f'{name}|{k}']=out
            begin=time.perf_counter();values=graph.encoder.run(graph.outputs,{'x':np.ascontiguousarray(f[c*H:c*H+W])[None],**states})
            states=dict(zip(graph.names,values[1:]));out,context,pn=decode(graph,values[0][0],context,pn,False)
            chunks.append(dict(ids=out['ids'],seconds=time.perf_counter()-begin,previews=previews))
        records.append(dict(utterance_id=row['utterance_id'],split=row['split'],reference=row['text'],audio_seconds=len(wave)/sr,feature_seconds=feature_seconds,chunks=chunks))
        if n%50==0:print(role,tag,n,len(selected),flush=True)
    write(OUT/f'collect_{role}_{tag}.json',dict(variants=[f'{a}|{b}' for a,b in variants],tokens={str(k):v for k,v in graph.tokens.items()},records=records));print('done',role,tag,len(records))


if __name__=='__main__':main()
