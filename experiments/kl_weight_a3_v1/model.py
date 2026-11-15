"""Stride-aware causal audio stack with whole-arrived-chunk output mixing."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,width,expansion,kernel,dilation):
        super().__init__();inner=width*expansion;self.history=(kernel-1)*dilation
        self.norm=nn.LayerNorm(width);self.expand=nn.Linear(width,inner)
        self.depth=nn.Conv1d(inner,inner,kernel,dilation=dilation,groups=inner)
        self.project=nn.Linear(inner,width)

    def forward(self,x,valid):
        z=F.silu(self.expand(self.norm(x)))*valid[...,None]
        z=self.depth(F.pad(z.transpose(1,2),(self.history,0))).transpose(1,2)
        return (x+self.project(F.silu(z)))*valid[...,None]

    def stream(self,x,cache):
        z=F.silu(self.expand(self.norm(x))).transpose(1,2)
        if cache is None:cache=z.new_zeros(len(z),z.shape[1],self.history)
        joined=torch.cat([cache,z],-1)
        return x+self.project(F.silu(self.depth(joined).transpose(1,2))),joined[:,:,-self.history:].clone()


class Audio(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.stride=cfg['stride'];self.raw_history=cfg['kernel']-1
        self.front=nn.Conv1d(80,cfg['width'],cfg['kernel'],stride=self.stride)
        self.blocks=nn.ModuleList(Block(cfg['width'],cfg['expansion'],cfg['kernel'],d) for d in cfg['dilations'])
        self.norm=nn.LayerNorm(cfg['width'])
        self.history=self.raw_history+self.stride*sum(b.history for b in self.blocks)
        assert self.history%self.stride==0

    def forward(self,x,valid):
        x=self.front(F.pad(x.transpose(1,2),(self.raw_history,0))).transpose(1,2)
        valid=valid[:,::self.stride];x=x*valid[...,None]
        for block in self.blocks:x=block(x,valid)
        return self.norm(x)

    def stream(self,x,raw_cache=None,block_caches=None,seen=0):
        x=x.transpose(1,2)
        if raw_cache is None:raw_cache=x.new_zeros(len(x),80,self.raw_history)
        joined=torch.cat([raw_cache,x],-1);offset=(-seen)%self.stride
        z=self.front(joined[:,:,offset:]).transpose(1,2)
        caches=[]
        if block_caches is None:block_caches=[None]*len(self.blocks)
        for block,cache in zip(self.blocks,block_caches):z,cache=block.stream(z,cache);caches.append(cache)
        return self.norm(z),joined[:,:,-self.raw_history:].clone(),caches,seen+x.shape[-1]


class A2(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.audio=Audio(cfg)
        self.head=nn.Sequential(nn.Linear(8*cfg['width'],cfg['head_hidden']),nn.SiLU(),nn.Linear(cfg['head_hidden'],8*512))

    def sequence(self,features,valid,chunks):
        h=self.audio(features,valid)
        indices=self.audio.history//4+4+torch.arange(chunks,device=h.device)[:,None]*8+torch.arange(8,device=h.device)[None]
        return self.head(h[:,indices].flatten(-2)).reshape(len(h),chunks,8,512)


class Runtime:
    def __init__(self,model):self.model=model;self.raw_cache=None;self.caches=None;self.seen=0

    @torch.inference_mode()
    def advance(self,x,clock=None):
        import time
        start=time.perf_counter()
        h,self.raw_cache,self.caches,self.seen=self.model.audio.stream(x[None],self.raw_cache,self.caches,self.seen)
        h=h[:,-8:];mid=time.perf_counter()
        result=self.model.head(h.flatten(-2)).reshape(8,512)
        if clock is not None:clock['audio']+=mid-start;clock['head']+=time.perf_counter()-mid
        return result
