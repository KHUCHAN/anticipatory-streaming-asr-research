"""CTC forced alignment of the reference transcript; writes per-word start/end times in milliseconds."""
import json

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor

from shared import OUT,ROOT,config,role_rows,write


def viterbi(logp,labels,blank):
    """Best monotonic CTC path. Returns the frame span (first,last) of every label."""
    ext=[blank];[ext.extend([l,blank]) for l in labels];ext=np.array(ext);S=len(ext);T=len(logp)
    score=np.full(S,-np.inf);score[0]=logp[0,ext[0]];score[1]=logp[0,ext[1]]
    back=np.zeros((T,S),np.int8);skip=np.zeros(S,bool);skip[2:]=(ext[2:]!=blank)&(ext[2:]!=ext[:-2])
    for t in range(1,T):
        stay=score;one=np.concatenate([[-np.inf],score[:-1]]);two=np.where(skip,np.concatenate([[-np.inf,-np.inf],score[:-2]]),-np.inf)
        stack=np.stack([stay,one,two]);back[t]=stack.argmax(0);score=stack.max(0)+logp[t,ext]
    s=S-1 if score[S-1]>=score[S-2] else S-2;path=np.zeros(T,np.int64)
    for t in range(T-1,-1,-1):path[t]=s;s-=int(back[t,s])
    spans=[]
    for k in range(len(labels)):
        frames=np.nonzero(path==2*k+1)[0];spans.append((int(frames[0]),int(frames[-1])))
    return spans


def main():
    cfg=config();proc=Wav2Vec2Processor.from_pretrained(cfg['aligner']);model=Wav2Vec2ForCTC.from_pretrained(cfg['aligner']).eval()
    vocab=proc.tokenizer.get_vocab();blank=vocab['<pad>'];bar=vocab['|'];out={};ms=cfg['aligner_frame_ms']
    for n,row in enumerate(role_rows(cfg,'eval')+role_rows(cfg,'selection')):
        wave,sr=sf.read(ROOT/row['audio'],dtype='float32');assert sr==16000
        with torch.inference_mode():logp=torch.log_softmax(model(torch.from_numpy(wave)[None]).logits[0],-1).numpy()
        words=row['text'].split();labels=[];owner=[]
        for w,word in enumerate(words):
            if w:labels.append(bar);owner.append(-1)
            for ch in word:labels.append(vocab[ch]);owner.append(w)
        spans=viterbi(logp,labels,blank);start=[None]*len(words);end=[None]*len(words)
        for (a,b),w in zip(spans,owner):
            if w<0:continue
            if start[w] is None:start[w]=a*ms
            end[w]=(b+1)*ms
        out[row['utterance_id']]=dict(words=words,start_ms=start,end_ms=end,audio_ms=1000*len(wave)/sr)
        if n%50==0:print(n,flush=True)
    write(OUT/'alignments.json',out);print('done',len(out))


if __name__=='__main__':main()
