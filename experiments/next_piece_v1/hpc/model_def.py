"""20M streaming Zipformer transducer from upstream icefall sources in $ASR/src, an LSTM token LM, an optional long-context decoder branch and an optional next-piece head."""
import contextlib,os,sys,types
from pathlib import Path
import torch

ASR=Path(os.environ.get('ASR','.'));SRC=ASR/'src';W=39;H=32


def _stubs():
    def make_pad_mask(lengths,max_len=0):
        n=max(max_len,int(lengths.max()));return torch.arange(n,device=lengths.device)[None,:]>=lengths[:,None]

    def subsequent_chunk_mask(size,chunk_size,num_left_chunks=-1,device=torch.device('cpu')):
        i=torch.arange(size,device=device);c=i//chunk_size;hi=(c+1)*chunk_size
        lo=torch.zeros_like(i) if num_left_chunks<0 else ((c-num_left_chunks)*chunk_size).clamp(min=0)
        return (i[None,:]>=lo[:,None])&(i[None,:]<hi[:,None])

    @contextlib.contextmanager
    def torch_autocast(*a,**k):yield

    icefall=types.ModuleType('icefall');utils=types.ModuleType('icefall.utils')
    utils.make_pad_mask=make_pad_mask;utils.subsequent_chunk_mask=subsequent_chunk_mask;utils.torch_autocast=torch_autocast;icefall.utils=utils
    interface=types.ModuleType('encoder_interface');interface.EncoderInterface=torch.nn.Module
    lhotse=types.ModuleType('lhotse');lu=types.ModuleType('lhotse.utils');lu.fix_random_seed=lambda s:torch.manual_seed(s);lhotse.utils=lu
    for name,mod in [('icefall',icefall),('icefall.utils',utils),('encoder_interface',interface),('lhotse',lhotse),('lhotse.utils',lu)]:sys.modules.setdefault(name,mod)
    if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))


class LM(torch.nn.Module):
    """LSTM language model over the 500 BPE pieces. Token 0 (blank) starts every sequence."""
    def __init__(self,dim=512,hidden=768,layers=2,drop=0.1):
        super().__init__();self.emb=torch.nn.Embedding(500,dim);self.rnn=torch.nn.LSTM(dim,hidden,layers,batch_first=True,dropout=drop);self.drop=torch.nn.Dropout(drop);self.out=torch.nn.Linear(hidden,500)

    def hidden(self,y,state=None):
        h,state=self.rnn(self.drop(self.emb(y)),state);return h,state

    def forward(self,y,state=None):
        h,state=self.hidden(y,state);return self.out(self.drop(h)),state


class Transducer(torch.nn.Module):
    def __init__(self,chunk=16,ctx=False,nx=False):
        super().__init__();_stubs()
        from decoder import Decoder
        from joiner import Joiner
        from zipformer import Zipformer
        five=lambda v:(v,)*5
        self.encoder=Zipformer(num_features=80,encoder_dims=five(256),attention_dim=five(192),encoder_unmasked_dims=five(192),zipformer_downsampling_factors=(1,2,4,8,2),
            nhead=five(8),feedforward_dim=five(768),num_encoder_layers=five(2),cnn_module_kernels=five(31),num_left_chunks=64//chunk,decode_chunk_size=chunk)
        self.decoder=Decoder(vocab_size=500,decoder_dim=512,blank_id=0,context_size=2)
        self.joiner=Joiner(encoder_dim=256,decoder_dim=512,joiner_dim=512,vocab_size=500)
        self.simple_am_proj=torch.nn.Linear(256,500);self.simple_lm_proj=torch.nn.Linear(512,500);self.ctx=ctx;self.nx=nx
        if ctx:  # long-context branch: LSTM over the whole token history, added to the 2-token decoder output through a projection that starts at zero
            self.ctx_lm=LM();self.ctx_proj=torch.nn.Linear(768,512);torch.nn.init.zeros_(self.ctx_proj.weight);torch.nn.init.zeros_(self.ctx_proj.bias)

        if nx:  # next-piece head: its own projections of the encoder frame and of the decoder output, then 500 classes (class 0: no further piece)
            self.nx_enc=torch.nn.Linear(256,512);self.nx_dec=torch.nn.Linear(512,512);self.nx_out=torch.nn.Linear(512,500)

    def nx_logits(self,enc,dec):
        """Next-piece logits from encoder frames (..., 256) and decoder outputs (..., 512) of matching shape."""
        return self.nx_out(torch.tanh(self.nx_enc(enc)+self.nx_dec(dec)))

    def dec_out(self,y):
        """Decoder output (N,U,512) for token histories y (N,U) whose first column is blank."""
        d=self.decoder(y)
        if self.ctx:d=d+self.ctx_proj(self.ctx_lm.hidden(y)[0])
        return d


def load(path=None,chunk=16,key='model',ctx_lm=None,nx=False):
    """chunk: decode chunk in 50Hz frames (16 = 320ms); left context stays 64 frames.
    ctx_lm: path of a trained LM (or 'random'); adds the long-context decoder branch, initialised from that LM, to weights that do not have one.
    nx: adds a randomly initialised next-piece head to weights that do not have one."""
    state=torch.load(path or SRC/'pretrained.pt',map_location='cpu',weights_only=False);state=state.get(key,state);has='ctx_proj.weight' in state;hasn='nx_out.weight' in state
    model=Transducer(chunk,ctx=has or ctx_lm is not None,nx=hasn or nx);r=model.load_state_dict(state,strict=False);miss=r.missing_keys
    assert not r.unexpected_keys and all(k.startswith(('ctx_','nx_')) for k in miss) and not (has and any(k.startswith('ctx_') for k in miss)) and not (hasn and any(k.startswith('nx_') for k in miss)),r
    if ctx_lm is not None and not has and str(ctx_lm)!='random':model.ctx_lm.load_state_dict(torch.load(ctx_lm,map_location='cpu',weights_only=False)['model'])
    return model


def load_lm(path):
    lm=LM();lm.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);return lm
