"""20M streaming Zipformer transducer from upstream icefall sources in $ASR/src (no icefall/lhotse install needed)."""
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


class Transducer(torch.nn.Module):
    def __init__(self,chunk=16):
        super().__init__();_stubs()
        from decoder import Decoder
        from joiner import Joiner
        from zipformer import Zipformer
        five=lambda v:(v,)*5
        self.encoder=Zipformer(num_features=80,encoder_dims=five(256),attention_dim=five(192),encoder_unmasked_dims=five(192),zipformer_downsampling_factors=(1,2,4,8,2),
            nhead=five(8),feedforward_dim=five(768),num_encoder_layers=five(2),cnn_module_kernels=five(31),num_left_chunks=64//chunk,decode_chunk_size=chunk)
        self.decoder=Decoder(vocab_size=500,decoder_dim=512,blank_id=0,context_size=2)
        self.joiner=Joiner(encoder_dim=256,decoder_dim=512,joiner_dim=512,vocab_size=500)
        self.simple_am_proj=torch.nn.Linear(256,500);self.simple_lm_proj=torch.nn.Linear(512,500)


def load(path=None,chunk=16,key='model'):
    """chunk: decode chunk in 50Hz frames (16 = 320ms, 8 = 160ms); left context stays 64 frames."""
    model=Transducer(chunk);state=torch.load(path or SRC/'pretrained.pt',map_location='cpu',weights_only=False);state=state.get(key,state)
    model.load_state_dict(state,strict=True);return model
