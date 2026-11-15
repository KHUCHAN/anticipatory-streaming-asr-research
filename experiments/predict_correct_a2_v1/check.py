#!/usr/bin/env python3
"""Preflight chunk-level causality, stride phase, real labels and tiny fitting."""
import argparse
import json
import time

import numpy as np
import torch

from common import ROOT,HERE,OUT,config,state_module,write
from data import datasets
from model import A2,Runtime
from objective import losses
from inference import Inference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tiny-fit',action='store_true');args=parser.parse_args()
    cfg=config();torch.set_num_threads(1);torch.manual_seed(cfg['seed']);model=A2(cfg).eval();train,dev,_=datasets(cfg)
    x=torch.randn(1,141,80);history=torch.zeros(1,model.audio.history,80);all_x=torch.cat([history,x],1)
    valid=torch.ones(all_x.shape[:2],dtype=torch.bool);valid[:,:model.audio.history]=False
    with torch.no_grad():
        full=model.sequence(all_x,valid,4);runtime=Runtime(model);stream=[]
        for lo,hi in [(0,45),(45,77),(77,109),(109,141)]:stream.append(runtime.advance(x[0,lo:hi]))
        error=float((full-torch.stack(stream)[None]).abs().max());assert error<2e-5,error
        changed=all_x.clone();changed[:,model.audio.history+45:]=torch.randn_like(changed[:,model.audio.history+45:])*5
        future=float((full[:,0]-model.sequence(changed,valid,4)[:,0]).abs().max());assert future==0.
        changed=all_x.clone();changed[:,model.audio.history+44]+=10
        impact=float((full[:,0,0]-model.sequence(changed,valid,4)[:,0,0]).abs().max());assert impact>1e-6
        group=[i for i in range(32) if any(train.start(i,e)==0 for e in range(1,37))]
        assert len(group)>20
        batch,starts=train.collate([0,1],1,'cpu');assert batch[-1][0,0].all()
    checks=dict(parameters=sum(p.numel() for p in model.parameters()),history_features=model.audio.history,
        batch_stream_max_difference=error,future_chunk_max_difference=future,last_arrived_frame_affects_first_output=impact,
        first_chunk_direct_supervision=True,test_used=False)
    assert checks['parameters']==9597696
    if args.tiny_fit:
        assert torch.backends.mps.is_available();torch.manual_seed(cfg['seed']);model=A2(cfg).to('mps').train()
        stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text());mean=torch.tensor(stats['mean'],device='mps');std=torch.tensor(stats['std'],device='mps')
        bridge=state_module.FrozenRNNT(ROOT/cfg['bridge']).to('mps');batch,_=train.collate(list(range(8)),1,'mps')
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
        initial=float(losses(model,bridge,batch,cfg,mean,std,False)[0].detach());started=time.time()
        for step in range(128):
            opt.zero_grad(set_to_none=True);mae,_=losses(model,bridge,batch,cfg,mean,std,False);mae.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        final=float(losses(model,bridge,batch,cfg,mean,std,False)[0].detach())
        assert final<initial*.8,(initial,final)
        checks['tiny_fit']=dict(steps=128,utterances=8,initial_mae=initial,final_mae=final,seconds=time.time()-started,
            memory_gib=torch.mps.driver_allocated_memory()/2**30,disposable_model_not_used_for_training=True)
    write(HERE/'artifacts/preflight.json',checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
