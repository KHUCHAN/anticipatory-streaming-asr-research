"""Shared causal audio model with optional self-generated prior conditioning."""
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT,state_module


class Block(nn.Module):
    def __init__(self,width,expansion,kernel,dilation):
        super().__init__();inner=width*expansion;self.history=(kernel-1)*dilation
        self.norm=nn.LayerNorm(width);self.expand=nn.Linear(width,inner)
        self.depth=nn.Conv1d(inner,inner,kernel,dilation=dilation,groups=inner)
        self.project=nn.Linear(inner,width)

    def forward(self,x,valid=None):
        z=F.silu(self.expand(self.norm(x)))
        if valid is not None:z=z*valid[...,None]
        z=self.depth(F.pad(z.transpose(1,2),(self.history,0))).transpose(1,2)
        x=x+self.project(F.silu(z))
        return x if valid is None else x*valid[...,None]

    def stream(self,x,cache=None):
        z=F.silu(self.expand(self.norm(x))).transpose(1,2)
        if cache is None:cache=z.new_zeros(len(z),z.shape[1],self.history)
        joined=torch.cat([cache,z],-1)
        y=self.depth(joined).transpose(1,2)
        return x+self.project(F.silu(y)),joined[:,:,-self.history:].clone()


class Audio(nn.Module):
    def __init__(self,cfg):
        super().__init__();w=cfg['audio_width']
        self.input=nn.Linear(80,w)
        self.blocks=nn.ModuleList(Block(w,cfg['audio_expansion'],cfg['audio_kernel'],d) for d in cfg['audio_dilations'])
        self.history=sum(b.history for b in self.blocks)
        self.norm=nn.LayerNorm(w)

    def forward(self,x,valid=None):
        x=self.input(x)
        if valid is not None:x=x*valid[...,None]
        for block in self.blocks:x=block(x,valid)
        return self.norm(x)

    def stream(self,x,caches=None):
        caches=[None]*len(self.blocks) if caches is None else caches
        x=self.input(x);updated=[]
        for block,cache in zip(self.blocks,caches):
            x,cache=block.stream(x,cache);updated.append(cache)
        return self.norm(x),updated


class Corrector(nn.Module):
    def __init__(self,cfg,variant):
        super().__init__();self.cfg=cfg;self.variant=variant
        self.audio=Audio(cfg);w=cfg['audio_width']
        # Construct common modules before the optional predictor, preserving
        # identical initial audio/head functions across variants.
        self.head=nn.Sequential(nn.Linear(w,cfg['head_hidden']),nn.SiLU(),nn.Linear(cfg['head_hidden'],512))
        self.prior_projection=nn.Linear(512,w,bias=False) if variant!='A' else None
        if self.prior_projection is not None:nn.init.zeros_(self.prior_projection.weight)
        if variant=='B':
            ck=torch.load(ROOT/cfg['predictor_checkpoint'],map_location='cpu',weights_only=False)
            self.predictor=state_module.StateFGN(ck['cfg'],'deterministic_s37')
            self.predictor.load_state_dict(ck['model'])
            self.predictor.requires_grad_(False);self.predictor.eval()
        else:self.predictor=None

    def train(self,mode=True):
        super().train(mode)
        if self.predictor is not None:self.predictor.eval()
        return self

    @torch.no_grad()
    def prior(self,previous,current,memory):
        if current is None:return None
        if self.variant=='C':return current
        if self.variant=='B':
            noise=current.new_zeros(len(current),self.predictor.cfg['noise_dim'])
            value,_=self.predictor.transition(previous,current,memory,noise)
            return value
        return None

    def correct(self,audio,prior):
        if prior is not None:audio=audio+self.prior_projection(prior)
        return self.head(audio)

    @torch.no_grad()
    def feedback(self,value,current,memory):
        value=value.detach()
        previous=torch.zeros_like(value) if current is None else current
        if self.predictor is not None:
            observation=self.predictor.observation(value.flatten(-2))[:,None]
            memory=self.predictor.memory(observation,None if memory is None else memory[None])[1][0]
        return previous,value,memory

    def sequence(self,features,feature_valid,chunks):
        h=self.audio(features,feature_valid)
        indices=self.audio.history+16+torch.arange(chunks,device=h.device)[:,None]*32+torch.arange(8,device=h.device)[None]*4
        selected=h[:,indices]
        previous=current=memory=None;outputs=[]
        for step in range(chunks):
            p=self.prior(previous,current,memory)
            corrected=self.correct(selected[:,step],p)
            outputs.append(corrected)
            previous,current,memory=self.feedback(corrected,current,memory)
        return torch.stack(outputs,1)


class Runtime:
    def __init__(self,model):
        self.model=model;self.caches=None;self.previous=self.current=self.memory=None

    @torch.inference_mode()
    def advance(self,new_features,clock=None):
        import time
        start=time.perf_counter()
        encoded,self.caches=self.model.audio.stream(new_features[None],self.caches)
        encoded=encoded[:,-29::4]
        after_audio=time.perf_counter()
        prior=self.model.prior(self.previous,self.current,self.memory)
        after_prior=time.perf_counter()
        output=self.model.correct(encoded,prior)
        after_correction=time.perf_counter()
        self.previous,self.current,self.memory=self.model.feedback(output,self.current,self.memory)
        end=time.perf_counter()
        if clock is not None:
            clock['audio']+=after_audio-start;clock['prediction']+=after_prior-after_audio
            clock['correction']+=after_correction-after_prior;clock['feedback']+=end-after_correction
        return output[0]
