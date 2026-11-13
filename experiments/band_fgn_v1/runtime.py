"""Arrived-only confirmed band2 cache; FGN audio exists only in temporary branches."""
import time
import torch
from common import ROOT,module
from avs.stream import Node

original=module('hybrid_original_collector',ROOT/'scripts/collect_zipformer_forecasts.py')
state_runtime=module('hybrid_state_runtime',ROOT/'experiments/fgn_state_v1/runtime.py')

class Runtime:
    def __init__(self,model,pieces,members=4):
        original.PIECES=pieces;self.model=model;self.pieces=pieces
        self.ar=original.Runtime(model.ar)
        self.members=members if model.variant.endswith('_fgn') else 1
        if model.hybrid:
            self.state=state_runtime.Runtime([model.state],None,model.state_mean,model.state_std,self.members,2,pieces)

    @torch.inference_mode()
    def advance(self,encoder,new_tokens,native_logp,gap,prefix_length,generator=None):
        begin=time.perf_counter()
        base=self.ar.advance(encoder,new_tokens,native_logp,gap,prefix_length)
        if not self.model.hybrid:
            base['features'] += [0.,0.,0.]
            base['seconds']=time.perf_counter()-begin
            return base
        self.state.observe(encoder[-8:])
        predicted=self.state.sample_states(generator)
        n,h,t,d=predicted.shape
        projected=self.model.ar.adapter(predicted.reshape(n,h*t,d))
        old=self.ar.p.mem_kv;new=[];heads=self.ar.p.h;dim=self.ar.p.d
        split=lambda x:x.view(n,-1,heads,dim//heads).transpose(1,2)
        for layer,(k,v) in zip(self.model.ar.decoder.layers,old):
            new.append((torch.cat([k.expand(n,-1,-1,-1),split(layer.cross_att.k(projected))],2),
                        torch.cat([v.expand(n,-1,-1,-1),split(layer.cross_att.v(projected))],2)))
        self.ar.p.mem_kv=new
        last=self.ar.nodes[-1];branches=[Node(last.tok,last.parent) for _ in range(n)]
        self.ar.p.step(branches,cut=len(encoder),spec=True)
        self.ar.p.mem_kv=old
        probabilities=torch.stack([node.logp.exp() for node in branches]);probs=probabilities.mean(0)
        top=probs.topk(2);g=int(top.indices[0]);entropy=float(-(probs*probs.clamp_min(1e-9).log()).sum())
        member_entropy=-(probabilities*probabilities.clamp_min(1e-9).log()).sum(-1).mean()
        features=[float(top.values[0]),float(top.values[1]),entropy,float(not self.pieces[g].startswith('▁')),
                  native_logp,min(5.,gap),min(12,len(new_tokens))/12,min(64,prefix_length)/64,
                  float((probabilities.argmax(-1)==g).float().mean()),max(0.,entropy-float(member_entropy)),
                  float(predicted.std(0,unbiased=False).mean())]
        state_bytes=sum(x.numel()*x.element_size() for x in [self.state.previous,self.state.current,*self.state.memories])
        return dict(g=g,features=features,seconds=time.perf_counter()-begin,text_bytes=base['text_bytes']+state_bytes,
                    audio_kv_bytes=base['audio_kv_bytes'],confirmed_state_bytes=base['text_bytes']+state_bytes,
                    speculative_state_bytes=sum((k.numel()+v.numel())*k.element_size() for k,v in new),
                    predicted_state_tensor_bytes=predicted.numel()*predicted.element_size())
