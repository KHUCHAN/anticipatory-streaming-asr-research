#!/usr/bin/env python3
"""Compare released and unreleased band-2 caches on real causal decoding."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from avs.data import FeatSet,FEATS
from avs.model import AVS,Decoder
from avs.stream import Predictor,Decoder3
from avs.text import Tokenizer


def models():
    ck=torch.load(ROOT/'runs/stage1/avg.pt',map_location='cpu')
    model=AVS(**ck['cfg']); model.load_state_dict(ck['model']); model.eval()
    pk=torch.load(ROOT/'runs/stage2_audio_b2/best.pt',map_location='cpu'); c=pk['cfg']
    decoder=Decoder(c['vocab'],c['d'],c['h'],c['ff'],c['dec_layers'],0.,cross=True)
    decoder.load_state_dict(pk['model']); decoder.eval()
    return model,decoder


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=100)
    ap.add_argument('--splits',nargs='+',default=['test-clean','test-other'])
    ap.add_argument('--chunks',type=int,nargs='+',default=[4,8,16])
    ap.add_argument('--out',default='runs/followups/cache.json')
    a=ap.parse_args()
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    tok=Tokenizer(FEATS/'spm.model'); model,decoder=models()
    output={'scope':'text KV+logp storage; audio cache reported separately; token objects retained', 'results':{}}
    for split in a.splits:
        ds=FeatSet(split,tok)
        idxs=random.Random(11).sample(range(len(ds)),a.n)
        for chunk in a.chunks:
            pred0=Predictor(decoder,2)
            pred1=Predictor(decoder,2,release_cache=True)
            decs=[Decoder3(p,beam=4,weight=.5,hold=0) for p in (pred0,pred1)]
            same=0; maxdiff=0.; summaries=[[],[]]; elapsed=[0.,0.]
            for i in idxs:
                mel,lens,_,_=ds.collate([i],'cpu')
                enc,el=model.encoder(mel,lens,chunk,64); enc=enc[0,:el[0]]
                logp=model.ctc(enc).log_softmax(-1)
                for dec in decs: dec.reset()
                ids=[[],[]]
                for start in range(0,len(enc),chunk):
                    end=min(start+chunk,len(enc))
                    for k,dec in enumerate(decs):
                        t=time.perf_counter(); ids[k]+=dec.advance(logp[start:end],enc[:end]); elapsed[k]+=time.perf_counter()-t
                        summaries[k].append(dec.state_usage())
                    if ids[0]!=ids[1]:
                        raise AssertionError(f'{split}/{i}/chunk{end}: changed committed tokens')
                    if decs[0].committed.logp is not None and decs[1].committed.logp is not None:
                        maxdiff=max(maxdiff,float((decs[0].committed.logp-decs[1].committed.logp).abs().max()))
                for k,dec in enumerate(decs): ids[k]+=dec.finish()
                if ids[0]!=ids[1]: raise AssertionError(f'{split}/{i}: changed final transcript')
                same+=1
            record={'utts':a.n,'identical_transcripts':same,'max_committed_logp_difference':maxdiff}
            for k,tag in enumerate(('retained','released')):
                record[tag]={'decode_seconds':round(elapsed[k],3),
                             **{f'peak_{key}':max(row[key] for row in summaries[k]) for key in summaries[k][0]}}
            output['results'][f'{split}/c{chunk}']=record
            print(json.dumps({'condition':f'{split}/c{chunk}',**record}),flush=True)
            target=ROOT/a.out; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(output,indent=2)+'\n')
    if any(r['max_committed_logp_difference']>1e-5 for r in output['results'].values()):
        raise AssertionError('Unexpected probability difference after releasing old states')


if __name__=='__main__': main()
