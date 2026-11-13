import copy,json,random
import torch
from common import OUT,config,datasets,write
from model import Hybrid,objective
from runtime import Runtime

def main():
    torch.set_num_threads(1);torch.manual_seed(37);cfg=config();ds,_,_=datasets(cfg)
    batch=ds.collate(ds.batches(cfg)[4][:2],random.Random(9),'cpu')
    model=Hybrid('small_fgn').eval();large=Hybrid('large_ar').eval();small=Hybrid('small_ar').eval()
    with torch.no_grad():
        a=small(batch)[0];b=large(batch)[0];identity=float((a-b).abs().max())
        assert identity<1e-5
        x=tuple(v.clone() for v in batch);states=x[0];cuts=x[2]
        for i,cut in enumerate(cuts):states[i,cut:]=100*torch.randn_like(states[i,cut:])
        torch.manual_seed(44);original=model(batch)[0]
        torch.manual_seed(44);perturbed=model(x)[0]
        future_diff=float((original-perturbed).abs().max());assert future_diff<1e-5
    future_batch=tuple(v.clone() for v in batch);future_batch[0].requires_grad_()
    logits,_,pred=model(future_batch,horizon=3);pred.retain_grad()
    logits.square().mean().backward();future_grad=0.
    for i,cut in enumerate(future_batch[2]):
        future_grad=max(future_grad,float(future_batch[0].grad[i,cut:].abs().max()))
    assert future_grad==0
    coupling=sum(float(p.grad.abs().sum()) for p in model.state.parameters() if p.grad is not None)
    assert coupling>0
    # Later losses must flow through a predicted state into the next step.
    model.zero_grad(set_to_none=True);captured=[]
    original_transition=model.state.transition
    def transition(*args):
        out=original_transition(*args);out[0].retain_grad();captured.append(out[0]);return out
    model.state.transition=transition
    pred=model.future(batch[0],batch[1],batch[2],3,2)
    pred[:,:,2].square().mean().backward();roll_grad=float(captured[0].grad.abs().sum());assert roll_grad>0
    # Confirmed AR states must survive speculative future audio unchanged.
    model.state.transition=original_transition
    pieces=['▁X']*500;runtime=Runtime(model,pieces,4)
    enc=batch[0][0].flatten(0,1)[:16].detach()
    with torch.inference_mode():
        runtime.advance(enc[:8],[15,20],-.5,0.,2,torch.Generator().manual_seed(42))
        old=[(k.clone(),v.clone()) for k,v in runtime.ar.p.mem_kv]
        assert all(k.shape[2]==9 for k,v in old)
        runtime.advance(enc,[22],-.5,0.,3,torch.Generator().manual_seed(42))
        assert all(k.shape[2]==17 for k,v in runtime.ar.p.mem_kv)
        for (k,v),(pk,pv) in zip(runtime.ar.p.mem_kv,old):assert torch.equal(k[:,:,:9],pk) and torch.equal(v[:,:,:9],pv)
        kept=[n for n in runtime.ar.nodes if n.kv is not None];assert len(kept)<=3
    write(OUT/'checks.json',dict(initial_large_identity_max_difference=identity,future_input_perturbation_max_difference=future_diff,
        future_input_gradient_max=future_grad,token_loss_gradient_into_fgn=coupling,later_state_gradient_through_earlier_prediction=roll_grad,
        confirmed_audio_cache_speculation_isolated=True,band2_live_text_nodes_at_most=3,passed=True))
    print((OUT/'checks.json').read_text(),flush=True)
if __name__=='__main__':main()
