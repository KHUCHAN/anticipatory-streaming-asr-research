"""Bounded causal runtime: confirmed memory and speculative paths are separate."""
from collections import Counter
import time
import torch


def tensor_bytes(xs):return sum(x.numel()*x.element_size() for x in xs)


def summarize(paths,confidence,pieces,last_logp,gap,new_count,prefix_length):
    first=[p[0] if p else 0 for p in paths];counts=Counter(first)
    ranked=sorted(counts.items(),key=lambda v:(-v[1],v[0]));g,frequency=ranked[0];n=len(first)
    probs=torch.tensor(list(counts.values()),dtype=torch.float32)/n
    entropy=float(-(probs*probs.log()).sum())
    features=[frequency/n,ranked[1][1]/n if len(ranked)>1 else 0.,entropy,
              float(g>=3 and not pieces[g].startswith('▁')),last_logp,min(5.,gap),
              min(12,new_count)/12,min(64,prefix_length)/64]
    selected=torch.tensor([c for c,token in zip(confidence,first) if token==g])
    features+=selected.mean(0).tolist()
    return g,features,n


@torch.inference_mode()
def decode_states(states,bridge,context,first_only=False,return_confidence=False):
    """Free-running native greedy decoding; no future token contexts supplied."""
    n=len(states);states=states.flatten(1,2);ctx=torch.tensor(context,dtype=torch.long).expand(n,-1).clone()
    pn=bridge.prediction(ctx);sequences=[[] for _ in range(n)];confidence=[[0.,0.,0.] for _ in range(n)]
    for frame in range(states.shape[1]):
        logits=bridge(states[:,frame],pn);tokens=logits.argmax(-1)
        # Match the native rule: blank0 and unk2 do not advance the prediction
        # context; sos/eos1 does. A first token1 later makes the UI branch abstain.
        active=(tokens!=0)&(tokens!=2)
        if first_only:active=active & torch.tensor([not s for s in sequences])
        if active.any():
            for member in active.nonzero().flatten().tolist():
                if return_confidence and not sequences[member]:
                    probs=(logits[member]/2).softmax(-1);top=probs.topk(2).values.clamp(1e-6,1-1e-6)
                    lp=torch.log(top/(1-top));entropy=float(-(probs*probs.clamp_min(1e-9).log()).sum())
                    confidence[member]=[float(lp[0]),float(lp[1]),entropy]
                sequences[member].append(int(tokens[member]))
            ctx[active]=torch.stack([ctx[active,-1],tokens[active]],-1)
            pn=bridge.prediction(ctx)
        if first_only and all(sequences):break
    return (sequences,confidence) if return_confidence else sequences


class Runtime:
    def __init__(self,models,bridge,mean,std,members,steps,pieces):
        self.models=models;self.bridge=bridge;self.mean=mean;self.std=std
        self.members=members;self.steps=steps;self.pieces=pieces
        self.previous=None;self.current=None;self.memories=[None]*len(models)
        self.chunks=0

    @torch.inference_mode()
    def observe(self,arrived):
        x=((arrived-self.mean)/self.std).clone()[None]
        for j,model in enumerate(self.models):
            if model.variant.startswith('reset'):
                observed=torch.stack([self.current,x],1) if self.current is not None else x[:,None]
                self.memories[j]=model.observed_memory(observed)[:,-1]
            else:
                z=model.observation(x.flatten(-2))[:,None]
                old=self.memories[j]
                self.memories[j]=model.memory(z,None if old is None else old[None])[1][0]
        self.previous=self.current if self.current is not None else torch.zeros_like(x)
        self.current=x;self.chunks+=1

    @torch.inference_mode()
    def sample_states(self,generator=None,steps=None,members=None):
        h=self.steps if steps is None else steps;n=self.members if members is None else members
        counts=[n//len(self.models)+(j<n%len(self.models)) for j in range(len(self.models))]
        all_states=[]
        for j,(model,k) in enumerate(zip(self.models,counts)):
            if not k:continue
            prev=self.previous.expand(k,-1,-1);cur=self.current.expand(k,-1,-1);mem=self.memories[j].expand(k,-1)
            predictions=[]
            for _ in range(h):
                noise=torch.randn(k,model.cfg['noise_dim'],generator=generator)
                nxt,mem=model.transition(prev,cur,mem,noise)
                predictions.append(nxt*self.std+self.mean);prev,cur=cur,nxt
            all_states.append(torch.stack(predictions,1))
        return torch.cat(all_states,0)

    @torch.inference_mode()
    def advance(self,arrived,native_context,last_logp,gap,new_count,prefix_length,generator=None):
        start=time.perf_counter();self.observe(arrived)
        predictions=self.sample_states(generator)
        paths,confidence=decode_states(predictions,self.bridge,native_context,first_only=True,return_confidence=True)
        # Empty trajectories remain in the vote distribution, including its mode.
        # Equal probability-based confidence inputs for stochastic AND noiseless
        # controls. A deterministic one-path vote alone is always unanimous.
        g,features,n=summarize(paths,confidence,self.pieces,last_logp,gap,new_count,prefix_length)
        seconds=time.perf_counter()-start
        confirmed=tensor_bytes([self.previous,self.current,*self.memories])
        branch=n*(2*self.current.numel()+self.memories[0].numel())*4
        return dict(g=g,features=features,seconds=seconds,text_bytes=confirmed,audio_kv_bytes=0,
                    confirmed_state_bytes=confirmed,speculative_state_bytes=branch,
                    predicted_state_tensor_bytes=tensor_bytes([predictions]),members=n)


class PersistenceRuntime:
    """No learned predictor: repeat the last arrived encoder chunk as future.

    Tests whether merely replaying already observed acoustic states, with the
    updated confirmed token prefix, explains the benefit attributed to a model.
    """
    def __init__(self,bridge,steps,pieces):self.bridge=bridge;self.steps=steps;self.pieces=pieces;self.current=None

    def observe(self,arrived):self.current=arrived.clone()

    def sample_states(self,generator=None,steps=None,members=None):
        h=self.steps if steps is None else steps;n=1 if members is None else members
        return self.current[None,None].expand(n,h,-1,-1).clone()

    @torch.inference_mode()
    def advance(self,arrived,native_context,last_logp,gap,new_count,prefix_length,generator=None):
        start=time.perf_counter();self.observe(arrived);predictions=self.sample_states(generator)
        paths,confidence=decode_states(predictions,self.bridge,native_context,first_only=True,return_confidence=True)
        g,features,n=summarize(paths,confidence,self.pieces,last_logp,gap,new_count,prefix_length)
        return dict(g=g,features=features,seconds=time.perf_counter()-start,text_bytes=tensor_bytes([self.current]),audio_kv_bytes=0,
                    confirmed_state_bytes=tensor_bytes([self.current]),speculative_state_bytes=0,
                    predicted_state_tensor_bytes=tensor_bytes([predictions]),members=n)
