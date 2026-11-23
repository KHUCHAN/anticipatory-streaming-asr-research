"""Preview passes on the 20M model: zero/repeat fill through the original ONNX encoder and zero fill through the fine-tuned copy (`ft`).

Usage: collect.py ROLE NAME[,NAME...] [LIMIT]. `ft0` runs the original weights through the PyTorch path as a pipeline check.
Preview passes never update native state; the native pass always uses the original ONNX encoder.
"""
import json
import sys
import time

import numpy as np
import soundfile as sf
import torch

from shared import OUT,ROOT,config,features,role_rows,write
import inference
from torch_model import load


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
    role,names=sys.argv[1],sys.argv[2].split(',');limit=int(sys.argv[3]) if len(sys.argv)>3 else None
    cfg=config();s=cfg['stage2'];a2cfg=json.loads((ROOT/cfg['a2_config']).read_text());torch.set_num_threads(1)
    inf=inference.Inference(a2cfg,small=True);graph=inf.graph;W=cfg['feature_window'];H=cfg['feature_hop'];assert graph.window==W and graph.hop==H
    net={}
    for name in names:
        if name.startswith('ft'):
            m=load()
            if name=='ft':
                state=torch.load(OUT/'student.pt',map_location='cpu');m.encoder.load_state_dict(state['encoder']);m.joiner.encoder_proj.load_state_dict(state['proj'])
            net[name]=m
    lens=torch.tensor([W])

    @torch.inference_mode()
    def torch_encoder(name,x,states):
        m=net[name];out,_,_=m.encoder.streaming_forward(torch.from_numpy(x),lens,[torch.from_numpy(states[n]) for n in graph.names]);return m.joiner.encoder_proj(out)[0].numpy()

    selected=role_rows(cfg,role);selected=selected[:limit] if limit else selected;variants=[(n,k) for n in names for k in s['ks']];records=[];same=0;total=0
    for n,row in enumerate(selected):
        wave,sr=sf.read(ROOT/row['audio'],dtype='float32');assert sr==16000
        begin=time.perf_counter();f=features(np.concatenate([wave,np.zeros(10560,np.float32)]));feature_seconds=time.perf_counter()-begin
        count=max(0,(len(f)-(W+1))//H+1);states=graph.states();context=np.zeros((1,graph.context),np.int64);pn=graph.decoder.run(None,{'y':context})[0];chunks=[]
        for c in range(count):
            previews={}
            for name,k in variants:
                begin=time.perf_counter();x=np.ascontiguousarray(f[c*H:c*H+W]).copy()[None]
                if name=='repeat':x[0,k:]=x[0,k-1]
                else:x[0,k:]=0.0
                enc=torch_encoder(name,x,states) if name in net else graph.encoder.run(['encoder_out'],{'x':x,**states})[0][0]
                out,_,_=decode(graph,enc,context,pn,True);out['seconds']=time.perf_counter()-begin;previews[f'{name}|{k}']=out
            for name,k in variants:
                if name in net:
                    p=previews[f'{name}|{k}'];p['torch_seconds']=p['seconds']
                    if f'zero|{k}' in previews:p['seconds']=previews[f'zero|{k}']['seconds'];total+=1;same+=p['ids']==previews[f'zero|{k}']['ids']
            begin=time.perf_counter();values=graph.encoder.run(graph.outputs,{'x':np.ascontiguousarray(f[c*H:c*H+W])[None],**states})
            states=dict(zip(graph.names,values[1:]));out,context,pn=decode(graph,values[0][0],context,pn,False)
            chunks.append(dict(ids=out['ids'],seconds=time.perf_counter()-begin,previews=previews))
        records.append(dict(utterance_id=row['utterance_id'],split=row['split'],reference=row['text'],audio_seconds=len(wave)/sr,feature_seconds=feature_seconds,chunks=chunks))
        if n%50==0:print(role,n,len(selected),flush=True)
    tag='all' if 'ft' in names else 'check'
    write(OUT/f'collect_{role}_{tag}.json',dict(variants=[f'{a}|{b}' for a,b in variants],tokens={str(k):v for k,v in graph.tokens.items()},records=records))
    print('done',role,len(records),'torch previews equal to zero-fill ONNX previews',same,total)


if __name__=='__main__':main()
