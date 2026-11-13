"""The actual hybrid: retain band2 token AR and jointly learn future audio states."""
import copy, math
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT,StateFGN,fair_crps_two
from avs.zipformer_forecaster import ZipformerForecaster
from avs.model import rope

def one_rope(x,positions):
    half=x.shape[-1]//2
    freq=1/(10000**(torch.arange(half,device=x.device,dtype=torch.float32)/half))
    angles=positions.float()[:,None]*freq
    c=angles.cos()[:,None,None];s=angles.sin()[:,None,None]
    a,b=x[...,:half],x[...,half:]
    return torch.cat([a*c-b*s,a*s+b*c],-1)

class Hybrid(nn.Module):
    def __init__(self,variant,initialize=True):
        super().__init__();self.variant=variant;self.hybrid=not variant.endswith('_ar')
        ck=torch.load(ROOT/'runs/zipformer_ours/band2/best.pt',map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(ck['cfg']);self.ar=ZipformerForecaster(cfg,2)
        if initialize:self.ar.load_state_dict(ck['model'])
        if variant.startswith('large'):
            for _ in range(8):
                layer=copy.deepcopy(self.ar.decoder.layers[-1])
                # Initially identity: retain the trained function exactly.
                for linear in [layer.self_att.o,layer.cross_att.o,layer.ff[-1]]:
                    nn.init.zeros_(linear.weight);nn.init.zeros_(linear.bias)
                self.ar.decoder.layers.append(layer)
        if self.hybrid:
            sc=torch.load(ROOT/'runs/fgn_state_v1/state_s37/best.pt',map_location='cpu',weights_only=False)
            self.state=StateFGN(sc['cfg'],'deterministic' if variant.endswith('_det') else 'state')
            if initialize:self.state.load_state_dict(sc['model'])
        stats=__import__('json').loads((ROOT/'runs/fgn_state_v1/cache/stats.json').read_text())
        self.register_buffer('state_mean',torch.tensor(stats['mean']))
        self.register_buffer('state_std',torch.tensor(stats['std']))

    def speculative_ar(self,tokens,last,arrived,allow,future):
        """Match deployment: preceding text KV sees arrived audio only.

        The temporary last-token copy sees predicted future audio. A batched
        full-prefix forward against future audio would contaminate old text KV
        and would be a different model from the causal runtime.
        """
        dec=self.ar.decoder;b,u=tokens.shape;n=future.shape[0];d=dec.emb.embedding_dim
        x=dec.drop(dec.emb(tokens));branch=x[torch.arange(b,device=x.device),last][None].expand(n,-1,-1).reshape(n*b,1,d)
        pos=torch.arange(u,device=x.device);ix=torch.arange(u,device=x.device)
        mask=((ix[None,:]<=ix[:,None])&(ix[None,:]>=ix[:,None]-2))[None,None]
        visible=torch.cat([torch.ones(b,1,1,device=x.device,dtype=torch.bool),allow],-1).unsqueeze(1)
        mem=torch.cat([dec.mem0.expand(b,1,-1),self.ar.adapter(arrived)],1)
        generated=self.ar.adapter(future.flatten(0,1))
        speculative_mem=torch.cat([mem.repeat(n,1,1),generated],1)
        speculative_allow=torch.cat([visible.repeat(n,1,1,1),torch.ones(n*b,1,1,generated.shape[1],device=x.device,dtype=torch.bool)],-1)
        past=(ix[None,:]<last[:,None])&(ix[None,:]>=last[:,None]-2)
        branch_allow=torch.cat([past,torch.ones(b,1,device=x.device,dtype=torch.bool)],-1).repeat(n,1)[:,None,None]
        for layer in dec.layers:
            y=layer.n1(x);att=layer.self_att;h=att.h;dh=d//h
            split=lambda z:z.reshape(-1,z.shape[1],h,dh).transpose(1,2)
            pk=rope(split(att.k(y)),pos);pv=split(att.v(y))
            by=layer.n1(branch)
            q=one_rope(split(att.q(by)),last.repeat(n));k=one_rope(split(att.k(by)),last.repeat(n));v=split(att.v(by))
            keys=torch.cat([pk.repeat(n,1,1,1),k],2);values=torch.cat([pv.repeat(n,1,1,1),v],2)
            out=F.scaled_dot_product_attention(q,keys,values,attn_mask=branch_allow,dropout_p=att.drop if self.training else 0.)
            branch=branch+layer.drop(att.o(out.transpose(1,2).reshape(n*b,1,d)))
            branch=branch+layer.drop(layer.cross_att(layer.n2(branch),speculative_mem,speculative_allow))
            branch=branch+layer.drop(layer.ff(layer.n3(branch)))
            x=x+layer.drop(att(y,y,mask,pos,pos))
            x=x+layer.drop(layer.cross_att(layer.n2(x),mem,visible))
            x=x+layer.drop(layer.ff(layer.n3(x)))
        return dec.out(dec.norm(branch))[:,0].reshape(n,b,-1)

    def future(self,states,lengths,cuts,horizon=2,members=2,generator=None):
        x=(states-self.state_mean)/self.state_std;b=len(x);idx=torch.arange(b,device=x.device)
        memories=self.state.observed_memory(x)
        current=x[idx,(cuts-1).clamp_min(0)]*(cuts>0)[:,None,None]
        previous=x[idx,(cuts-2).clamp_min(0)]*(cuts>1)[:,None,None]
        memory=memories[idx,(cuts-1).clamp_min(0)]*(cuts>0)[:,None]
        old=previous.repeat(members,1,1);cur=current.repeat(members,1,1);mem=memory.repeat(members,1)
        path=[]
        for h in range(horizon):
            z=torch.randn(b*members,self.state.cfg['noise_dim'],device=x.device,generator=generator)
            nxt,mem=self.state.transition(old,cur,mem,z);path.append(nxt);old,cur=cur,nxt
        return torch.stack(path,1).reshape(members,b,horizon,8,512)

    def forward(self,batch,horizon=2,generator=None):
        states,lengths,cuts,tokens,token_lengths,targets=batch;b=len(states)
        # Exact receptive-field optimization, not a smaller model context:
        # L band2 layers depend on at most 2L+1 raw input tokens. RoPE is
        # translation-invariant; no absolute text embeddings are used.
        window=2*len(self.ar.decoder.layers)+1
        if tokens.shape[1]>window:
            keep=token_lengths.clamp_max(window)
            positions=(token_lengths-keep)[:,None]+torch.arange(window,device=tokens.device)[None]
            tokens=tokens.gather(1,positions.clamp_max(tokens.shape[1]-1))
            token_lengths=keep
        last=token_lengths-1;idx=torch.arange(b,device=states.device)
        # The actual waveform/features after the cutoff are NEVER AR inputs.
        t=max(1,int(cuts.max())*8)
        arrived=states.flatten(1,2)[:,:t]
        allow=(torch.arange(t,device=states.device)[None,:]<cuts[:,None]*8)[:,None]
        if not self.hybrid:
            logits=self.ar(tokens,arrived,allow)[idx,last]
            return logits[None],states.new_zeros(()),None
        pred=self.future(states,lengths,cuts,horizon,max(2,2),generator)
        # Even if the curriculum is one step, only generated frames are appended.
        nf=min(horizon,2);future=(pred[:,:,:nf]*self.state_std+self.state_mean).flatten(2,3)
        logits=self.speculative_ar(tokens,last,arrived,allow,future)
        pos=cuts[:,None]+torch.arange(horizon,device=states.device)[None]
        valid=pos<lengths[:,None]
        target=states[idx[:,None],torch.minimum(pos,lengths[:,None]-1)]
        target=(target-self.state_mean)/self.state_std
        crps=fair_crps_two(pred[0],pred[1],target).mean((-1,-2))
        crps=(crps*valid).sum()/valid.sum().clamp_min(1)
        return logits,crps,pred

def objective(model,batch,cfg,horizon):
    logits,crps,pred=model(batch,horizon)
    targets=batch[-1]
    ce=torch.stack([F.cross_entropy(l,targets,label_smoothing=cfg['label_smoothing']) for l in logits]).mean()
    return ce+cfg['crps_weight']*crps,dict(ce=ce.detach(),crps=crps.detach())
