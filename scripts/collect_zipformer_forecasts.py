#!/usr/bin/env python3
"""Run OUR trained forecasters on causal native Zipformer prefixes.

All forecast inputs use arrived encoder frames and emitted native tokens only.
Reference matching is performed afterwards for offline gate labels/evaluation.
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import torch
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from avs.zipformer_forecaster import ZipformerForecaster,CachedSet,read_rows
from avs.stream import Node,Predictor

MODELS={};PIECES=[]
FEATURES=['p1','p2','entropy','continuation','native_last_logp','native_blank_gap_s','new_tokens','prefix_length']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def exact_pairs(reference,hypothesis):
    n,m=len(reference),len(hypothesis)
    d=np.zeros((n+1,m+1),np.int32);d[:,0]=np.arange(n+1);d[0]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(reference[i-1]!=hypothesis[j-1]))
    i,j=n,m;pairs=[]
    while i and j:
        if reference[i-1]==hypothesis[j-1] and d[i,j]==d[i-1,j-1]:
            pairs.append((i-1,j-1));i-=1;j-=1
        elif d[i,j]==d[i-1,j-1]+1:i-=1;j-=1
        elif d[i,j]==d[i-1,j]+1:i-=1
        else:j-=1
    return int(d[n,m]),pairs[::-1]


def token_word_indices(tokens):
    result=[];word=0;in_word=False
    for token in tokens:
        assigned=word
        for character in PIECES[token].replace('▁',' ') if token>=3 else '':
            if character.isspace():
                if in_word:word+=1;in_word=False
            else:in_word=True;assigned=word
        result.append(assigned if in_word else word)
    return result


def initialize():
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    global MODELS,PIECES
    for kind in ['full','band2']:
        state=torch.load(ROOT/f'runs/zipformer_ours/{kind}/best.pt',map_location='cpu',weights_only=False)
        model=ZipformerForecaster(state['cfg'],state['band']);model.load_state_dict(state['model']);model.eval()
        MODELS[kind]=model
    c=json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/c['model_dir']/'bpe.model'))
    PIECES=[sp.id_to_piece(i) for i in range(500)]


class Runtime:
    def __init__(self,model,cache_audio=True):
        self.model=model;self.p=Predictor(model.decoder,hist=model.band,release_cache=bool(model.band))
        self.cache_audio=cache_audio;self.memory_frames=0
        self.nodes=[Node(1,None)]
        self.p.set_memory(model.adapter(torch.zeros(0,512)))
        self.p.step(self.nodes,cut=0)

    def set_memory(self,encoder):
        if not self.cache_audio:
            self.p.set_memory(self.model.adapter(encoder));return
        assert len(encoder)>=self.memory_frames
        new=encoder[self.memory_frames:]
        if not len(new):return
        memory=self.model.adapter(new)[None];h=self.p.h;d=self.p.d
        split=lambda x:x.view(1,-1,h,d//h).transpose(1,2)
        for i,layer in enumerate(self.model.decoder.layers):
            old_k,old_v=self.p.mem_kv[i]
            k=split(layer.cross_att.k(memory));v=split(layer.cross_att.v(memory))
            self.p.mem_kv[i]=(torch.cat([old_k,k],2),torch.cat([old_v,v],2))
        self.memory_frames=len(encoder)

    def advance(self,encoder,new_tokens,native_logp,gap,prefix_length):
        begin=time.perf_counter();cut=len(encoder)
        self.set_memory(encoder)
        for token in new_tokens:
            node=Node(int(token),self.nodes[-1]);self.p.step([node],cut=cut);self.nodes.append(node)
            if self.model.band and len(self.nodes)>self.model.band+1:
                old=self.nodes[-self.model.band-2];old.kv=None;old.logp=None
        # Recondition the last token on the newly arrived audio even when the
        # native decoder emitted nothing. Older self-attention states stay cached.
        if not new_tokens:self.p.step([self.nodes[-1]],cut=cut)
        probs=self.nodes[-1].logp.exp();top=probs.topk(2);g=int(top.indices[0])
        entropy=float(-(probs*probs.clamp_min(1e-9).log()).sum())
        features=[float(top.values[0]),float(top.values[1]),entropy,float(not PIECES[g].startswith('▁')),
                  native_logp,min(5.,gap),min(12,len(new_tokens))/12,min(64,prefix_length)/64]
        seconds=time.perf_counter()-begin
        kept=self.nodes[-self.model.band-1:] if self.model.band else self.nodes
        text_bytes=sum(sum((k.numel()+v.numel())*k.element_size() for k,v in n.kv)+n.logp.numel()*n.logp.element_size() for n in kept if n.kv is not None)
        audio_bytes=sum((k.numel()+v.numel())*k.element_size() for k,v in self.p.mem_kv)
        return dict(g=g,features=features,seconds=seconds,text_bytes=text_bytes,audio_kv_bytes=audio_bytes)


def process(row):
    data=CachedSet([row]).load(0)
    native=data['ids'].tolist();gold=data['gold'].tolist()
    enc=torch.from_numpy(data['enc'].astype(np.float32))
    events={};elapsed={}
    order=['full','band2'] if sum(map(ord,row['utterance_id']))%2==0 else ['band2','full']
    for kind in order:
        model=MODELS[kind]
        start=time.perf_counter();runtime=Runtime(model);previous=0;records=[]
        for chunk,(count,available) in enumerate(zip(data['cuts'],data['available'])):
            count=int(count);end=(chunk+1)*8
            lp=float(data['logp'][count-1]) if count else 0.
            gap=(end-1-int(data['frames'][count-1]))*.04 if count else end*.04
            result=runtime.advance(enc[:end],native[previous:count],lp,gap,count)
            result.update(available=float(available),native_count=count,eligible=bool(available<float(data['duration']) and result['g']>=3))
            records.append(result);previous=count
        events[kind]=records;elapsed[kind]=time.perf_counter()-start
    # Future/reference information begins here; it never reaches Runtime.
    control=json.loads((ROOT/'runs/zipformer_ours/native_control'/(row['utterance_id']+'.json')).read_text())
    assert [e['native_count'] for e in control['events']]==data['cuts'].tolist()
    events['native']=control['events'];elapsed['native']=control['processing_seconds']
    _,matched=exact_pairs(gold,native)
    reference=str(data['reference']);hypothesis=str(data['hypothesis'])
    errors,word_pairs=exact_pairs(reference.split(),hypothesis.split())
    correct_words={j for _,j in word_pairs};token_words=token_word_indices(native)
    # Matching a fragment inside an otherwise wrong word is insufficient for a
    # positive confidence label. Require the whole native word to be correct.
    correct_native={j for _,j in matched if token_words[j] in correct_words}
    for records in events.values():
        for record in records:
            n=record['native_count'];g=record['g']
            record['label_keep']=int(n<len(native) and g==native[n])
            record['label_correct_and_keep']=int(record['label_keep'] and n in correct_native)
    return dict(utterance_id=row['utterance_id'],split=row['split'],speaker_id=row['speaker_id'],
                native=native,gold=gold,reference=reference,hypothesis=hypothesis,errors=errors,
                reference_words=len(reference.split()),correct_native_positions=sorted(correct_native),
                correct_final_word_positions=[j for _,j in word_pairs],audio_seconds=float(data['duration']),
                events=events,processing_seconds=elapsed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['dev','test'],required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();cfg=json.loads((ROOT/'configs/zipformer_ours.json').read_text())
    if a.part=='dev':
        rows=read_rows(ROOT/cfg['dev_protocol']/'gate_fit.jsonl')+read_rows(ROOT/cfg['dev_protocol']/'gate_calibration.jsonl')
    else:
        rows=sum([read_rows(ROOT/f'data/processed/manifests/{s}.jsonl') for s in cfg['evaluation_splits']],[])
    assert len({r['utterance_id'] for r in rows})==len(rows)
    if a.limit:rows=rows[:a.limit]
    out=ROOT/'runs/zipformer_ours'/f'events_{a.part}';out.mkdir(parents=True,exist_ok=True)
    provenance=dict(model_sha256={k:sha(ROOT/f'runs/zipformer_ours/{k}/best.pt') for k in ['full','band2']},
                    source_sha256={name:sha(ROOT/name) for name in ['scripts/collect_zipformer_forecasts.py','scripts/native_zipformer_forecast.py','avs/zipformer_forecaster.py','avs/stream.py','configs/zipformer_ours.json']})
    record=out.parent/f'events_{a.part}_provenance.json'
    if record.exists():assert json.loads(record.read_text())==provenance,'Collection inputs changed; preserve old events and start a clean collection'
    else:record.write_text(json.dumps(provenance,indent=2)+'\n')
    rows=[r for r in rows if not (out/(r['utterance_id']+'.json')).exists()]
    started=time.time();done=0
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as ex:
        futures=[ex.submit(process,row) for row in rows]
        for f in as_completed(futures):
            result=f.result();path=out/(result['utterance_id']+'.json');tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(result,separators=(',',':'))+'\n');tmp.replace(path);done+=1
            if done%100==0:print(json.dumps(dict(part=a.part,done=done,new_total=len(rows),seconds=round(time.time()-started,1))),flush=True)
    names={'full':FEATURES,'band2':FEATURES,'native':FEATURES+['log_native_nonblank_mass']}
    (out/'complete.json').write_text(json.dumps(dict(part=a.part,new_utterances=len(rows),limited=bool(a.limit),features=names,seconds=time.time()-started),indent=2)+'\n')


if __name__=='__main__':main()
