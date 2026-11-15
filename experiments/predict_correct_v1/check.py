#!/usr/bin/env python3
"""Check causal audio caching, corrected feedback, initialization and gradients."""
import torch

from common import config,write,HERE
from model import Corrector,Runtime
from data import datasets


def main():
    torch.set_num_threads(1);cfg=config();results=[];models={}
    for variant in cfg['variants']:
        torch.manual_seed(cfg['seed']);m=Corrector(cfg,variant).eval();models[variant]=m
        x=torch.randn(1,109,80)
        with torch.no_grad():
            offline=m.audio(x);caches=None;parts=[]
            for lo,hi in [(0,45),(45,77),(77,109)]:
                y,caches=m.audio.stream(x[:,lo:hi],caches);parts.append(y)
            delta=float((offline-torch.cat(parts,1)).abs().max())
            assert delta<2e-5,(variant,delta)
            changed=x.clone();changed[:,45:]=torch.randn_like(changed[:,45:])*10
            first=m.audio(changed)[:,:45]
            causal=float((first-offline[:,:45]).abs().max());assert causal==0.
            runtime=Runtime(m);output=runtime.advance(x[0,:45])
            assert torch.equal(runtime.current,output[None])
            if variant=='C':assert torch.equal(m.prior(runtime.previous,runtime.current,runtime.memory),output[None])
            if m.prior_projection is not None:m.prior_projection.weight.normal_(std=.01)
            prefix=torch.zeros(1,m.audio.history,80)
            full=torch.cat([prefix,x],1)
            mask=torch.ones(full.shape[:2],dtype=torch.bool);mask[:,:m.audio.history]=False
            batched=m.sequence(full,mask,3)
            runtime=Runtime(m);streamed=[]
            for lo,hi in [(0,45),(45,77),(77,109)]:streamed.append(runtime.advance(x[0,lo:hi]))
            recurrence_delta=float((batched-torch.stack(streamed)[None]).abs().max())
            assert recurrence_delta<5e-5,(variant,recurrence_delta)
            if m.prior_projection is not None:m.prior_projection.weight.zero_()
        results.append(dict(variant=variant,streaming_audio_max_difference=delta,sequence_vs_stream_recurrence_max_difference=recurrence_delta,future_audio_max_difference=causal,
            retained_state_is_own_corrected_output=True,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in m.parameters())))
    train,dev,_=datasets(cfg);batch=train.collate([0,1],0,'cpu')
    torch.manual_seed(15);x=torch.randn(1,45,80)
    reference=Runtime(models['A']).advance(x[0])
    for variant in cfg['variants']:
        out=Runtime(models[variant]).advance(x[0]);assert torch.equal(out,reference)
    from common import state_module,ROOT
    from objective import objective
    import json
    stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text());mean=torch.tensor(stats['mean']);std=torch.tensor(stats['std'])
    bridge=state_module.FrozenRNNT(ROOT/cfg['bridge'])
    for variant,m in models.items():
        loss,_=objective(m,bridge,batch,cfg,mean,std);loss.backward()
        assert float(m.head[-1].weight.grad.norm())>0
        if variant!='A':assert float(m.prior_projection.weight.grad.norm())>0
        if variant=='B':assert all(p.grad is None for p in m.predictor.parameters())
    starts0=train.collate(list(range(64)),0,'cpu')[-1];starts1=train.collate(list(range(64)),1,'cpu')[-1]
    assert starts0!=starts1
    write(HERE/'artifacts/initial_checks.json',dict(models=results,initial_audio_head_functions_identical=True,
        common_head_and_prior_adapter_gradients_nonzero=True,frozen_predictor_gradients_absent=True,
        training_cutoffs_change_each_epoch=True,test_used=False))
    print('INITIAL CORRECTOR CHECKS PASSED',results)


if __name__=='__main__':main()
