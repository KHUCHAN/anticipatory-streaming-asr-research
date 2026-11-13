"""Continuous recurrent-state FGN adaptation, separate from all prior models."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrozenRNNT(nn.Module):
    """Exact ONNX decoder/joiner arithmetic, with differentiable encoder input."""
    def __init__(self,path):
        super().__init__()
        with np.load(path) as data:
            for name in data.files:self.register_buffer(name,torch.from_numpy(data[name].copy()).float())

    def prediction(self,context):
        shape=context.shape[:-1]
        y=context.reshape(-1,2)
        embedded=F.embedding(y.clamp_min(0),self.embedding)*(y>=0).unsqueeze(-1)
        x=F.conv1d(embedded.transpose(1,2),self.conv,groups=128).squeeze(-1).relu()
        return F.linear(x,self.decoder_weight,self.decoder_bias).reshape(*shape,512)

    def forward(self,encoder,prediction):
        return F.linear(torch.tanh(encoder+prediction),self.joiner_weight,self.joiner_bias)


class ConditionalNorm(nn.Module):
    def __init__(self,d):
        super().__init__();self.norm=nn.LayerNorm(d);self.modulate=nn.Linear(d,2*d)
        nn.init.normal_(self.modulate.weight,std=.01);nn.init.zeros_(self.modulate.bias)
    def forward(self,x,noise_embedding):
        scale,shift=self.modulate(noise_embedding).chunk(2,-1)
        return self.norm(x)*(1+scale)+shift


class StateBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg['hidden_dim']
        self.n1=ConditionalNorm(d);self.n2=ConditionalNorm(d)
        self.attention=nn.MultiheadAttention(d,cfg['heads'],batch_first=True,dropout=0.)
        self.ff=nn.Sequential(nn.Linear(d,cfg['ff_dim']),nn.SiLU(),nn.Linear(cfg['ff_dim'],d))
    def forward(self,x,noise):
        y=self.n1(x,noise);x=x+self.attention(y,y,y,need_weights=False)[0]
        return x+self.ff(self.n2(x,noise))


class StateFGN(nn.Module):
    def __init__(self,cfg,variant):
        super().__init__();self.cfg=cfg;self.variant=variant
        d=cfg['hidden_dim'];m=cfg['memory_dim'];f=cfg['state_features'];t=cfg['state_frames']
        self.observation=nn.Sequential(nn.Linear(t*f,m),nn.LayerNorm(m),nn.SiLU())
        self.memory=nn.GRU(m,m,batch_first=True)
        self.inputs=nn.Linear(2*f,d);self.memory_projection=nn.Linear(m,d)
        self.positions=nn.Parameter(torch.randn(1,t,d)*.02)
        self.noise_encoder=nn.Sequential(nn.Linear(cfg['noise_dim'],d),nn.SiLU(),nn.Linear(d,d))
        self.blocks=nn.ModuleList(StateBlock(cfg) for _ in range(cfg['layers']))
        self.final_norm=ConditionalNorm(d);self.output=nn.Linear(d,f)
        nn.init.normal_(self.output.weight,std=.02);nn.init.zeros_(self.output.bias)

    def observed_memory(self,states):
        """Causal state sequence [B,T,8,512] -> [B,T,M]."""
        z=self.observation(states.flatten(-2))
        return self.memory(z)[0]

    def initial_memory(self,states,indices,all_memory=None):
        """states are a batch of utterances; indices contain (batch,cutoff)."""
        b,t=indices[:,0],indices[:,1]
        if self.variant.startswith('reset'):
            previous=states[b,(t-1).clamp_min(0)]* (t>0)[:,None,None]
            pair=torch.stack([previous,states[b,t]],1)
            two=self.observed_memory(pair)[:,-1]
            one=self.observed_memory(states[b,t][:,None])[:,-1]
            return torch.where((t>0)[:,None],two,one)
        if all_memory is None:all_memory=self.observed_memory(states)
        return all_memory[b,t]

    def transition(self,previous,current,memory,noise):
        """Sample a complete continuous next state and roll memory through it.

        No detach: losses at later forecast steps can reach earlier predicted
        states, earlier state transitions and the observed-history memory.
        """
        if self.variant.startswith('deterministic'):noise=torch.zeros_like(noise)
        e=self.noise_encoder(noise)[:,None]
        x=self.inputs(torch.cat([previous,current],-1))+self.positions+self.memory_projection(memory)[:,None]
        for block in self.blocks:x=block(x,e)
        predicted=current+self.output(self.final_norm(x,e))
        if self.variant.startswith('reset'):
            pair=torch.stack([current,predicted],1);updated=self.observed_memory(pair)[:,-1]
        else:
            z=self.observation(predicted.flatten(-2))[:,None]
            updated=self.memory(z,memory[None])[1][0]
        return predicted,updated


def fair_crps_two(a,b,target):
    return .5*((a-target).abs()+(b-target).abs()-(a-b).abs())


def rollout_loss(model,bridge,states,contexts,targets,indices,lengths,mean,std,horizon,generator=None):
    """Continuous self-generated rollout + task guidance; gold future is labels."""
    b,t=indices[:,0],indices[:,1]
    memory=model.initial_memory(states,indices)
    prior=states[b,(t-1).clamp_min(0)]*(t>0)[:,None,None]
    previous=[prior,prior];current=[states[b,t],states[b,t]];mem=[memory,memory]
    state_sum=states.new_zeros(());token_sum=states.new_zeros(());denom=0.
    for step in range(horizon):
        where=t+step+1;valid=where<lengths[b]
        safe=torch.minimum(where,lengths[b]-1)
        true=states[b,safe];target=targets[b,safe]
        with torch.no_grad():pn=bridge.prediction(contexts[b,safe])
        preds=[];token_logs=[]
        for member in range(2):
            noise=torch.randn(len(indices),model.cfg['noise_dim'],device=states.device,generator=generator)
            nxt,updated=model.transition(previous[member],current[member],mem[member],noise)
            logits=bridge(nxt*std+mean,pn)
            token_logs.append(logits.log_softmax(-1).gather(-1,target[...,None]).squeeze(-1))
            preds.append(nxt);previous[member],current[member],mem[member]=current[member],nxt,updated
        n=valid.sum().clamp_min(1)
        state_error=fair_crps_two(preds[0],preds[1],true).mean((-1,-2))
        state_sum=state_sum+(state_error*valid).sum()
        mixture=-torch.logsumexp(torch.stack(token_logs),0)+math.log(2)
        weights=torch.where(target==0,model.cfg['blank_token_weight'],1.)
        per_sample=(mixture*weights).sum(-1)/weights.sum(-1).clamp_min(1e-6)
        token_sum=token_sum+(per_sample*valid).sum();denom=denom+valid.sum()
    state_loss=state_sum/denom.clamp_min(1);token_loss=token_sum/denom.clamp_min(1)
    return state_loss+model.cfg['token_loss_weight']*token_loss,dict(state_crps=state_loss.detach(),token_nll=token_loss.detach())
