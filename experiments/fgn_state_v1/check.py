#!/usr/bin/env python3
"""Test the actual distinctions from the earlier token-only rollout."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from model import StateFGN,FrozenRNNT
from data import ROOT,OUT


def main():
    torch.set_num_threads(1);cfg=json.loads((ROOT/'experiments/fgn_state_v1/config.json').read_text())
    torch.manual_seed(37);model=StateFGN(cfg,'state_s37').eval()
    states=torch.randn(2,10,8,512,requires_grad=True);idx=torch.tensor([[0,3],[1,3]])
    memory=model.initial_memory(states,idx);z1=torch.randn(2,32);z2=torch.randn(2,32)
    first,m1=model.transition(states[:,2],states[:,3],memory,z1);first.retain_grad()
    second,m2=model.transition(states[:,3],first,m1,z2)
    second.square().mean().backward()
    assert first.grad is not None and first.grad.abs().max()>1e-8
    assert states.grad[:,:4].abs().max()>0
    assert states.grad[:,4:].abs().max()==0
    gradient=float(first.grad.norm())
    with torch.no_grad():
        a=model.initial_memory(states.detach(),idx)
        future=states.detach().clone();future[:,4:]=100*torch.randn_like(future[:,4:])
        b=model.initial_memory(future,idx);assert torch.equal(a,b)
        old=states.detach().clone();old[:,0]+=10
        c=model.initial_memory(old,idx);memory_effect=float((a-c).abs().max());assert memory_effect>1e-6
        reset=StateFGN(cfg,'reset_s37').eval();reset.load_state_dict(model.state_dict())
        r1=reset.initial_memory(states.detach(),idx);r2=reset.initial_memory(old,idx);assert torch.equal(r1,r2)
        p1,_=model.transition(states[:,2],states[:,3],a,z1)
        p2,_=model.transition(states[:,2],states[:,3],a,z2)
        assert (p1-p2).abs().max()>1e-6
        deterministic=StateFGN(cfg,'deterministic_s37').eval();deterministic.load_state_dict(model.state_dict())
        d1,_=deterministic.transition(states[:,2],states[:,3],a,z1)
        d2,_=deterministic.transition(states[:,2],states[:,3],a,z2);assert torch.equal(d1,d2)
        # First-cutoff handling must not read the last/future observed state.
        first_cut=torch.tensor([[0,0],[1,0]])
        assert torch.equal(model.initial_memory(states.detach(),first_cut),model.initial_memory(future,first_cut))
    base=json.loads((ROOT/'configs/zipformer_baseline.json').read_text());p=ROOT/base['model_dir']
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    dec=ort.InferenceSession(str(p/('decoder-'+base['model_suffix'])),sess_options=opt,providers=['CPUExecutionProvider'])
    join=ort.InferenceSession(str(p/('joiner-'+base['model_suffix'])),sess_options=opt,providers=['CPUExecutionProvider'])
    bridge=FrozenRNNT(OUT/'rnnt_bridge.npz')
    y=np.random.default_rng(31).integers(0,500,(32,2),dtype=np.int64);y[0]=[-1,0]
    ref=dec.run(None,{'y':y})[0];actual=bridge.prediction(torch.from_numpy(y)).numpy()
    x=np.random.default_rng(32).normal(size=(32,512)).astype(np.float32)
    jref=join.run(None,{'encoder_out':x,'decoder_out':ref})[0]
    value=torch.from_numpy(x).requires_grad_();jactual=bridge(value,torch.from_numpy(actual))
    join_diff=float(np.abs(jref-jactual.detach().numpy()).max());dec_diff=float(np.abs(ref-actual).max())
    assert dec_diff<1e-4 and join_diff<1e-4
    jactual.sum().backward();assert value.grad.abs().max()>0 and not list(bridge.parameters())
    result=dict(status='pass',continuous_two_step_backprop_gradient_norm=gradient,
                future_observed_state_gradient_max=0.,future_input_perturbation='exactly invariant',
                old_history_influence_on_persistent_memory=memory_effect,
                reset_control_old_history_invariance=True,shared_noise_changes_state=True,
                deterministic_control_noise_invariance=True,first_cutoff_future_invariance=True,
                frozen_decoder_max_absolute_difference=dec_diff,frozen_joiner_max_absolute_difference=join_diff,
                frozen_joiner_encoder_gradient=True,parameters=sum(p.numel() for p in model.parameters()))
    (OUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
