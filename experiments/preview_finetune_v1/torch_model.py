"""PyTorch copy of the public 20M streaming Zipformer transducer, built from the upstream icefall sources kept in .local/20m_torch."""
import contextlib
import sys
import types
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'.local/20m_torch'


def _stubs():
    def make_pad_mask(lengths,max_len=0):
        n=max(max_len,int(lengths.max()));return torch.arange(n,device=lengths.device)[None,:]>=lengths[:,None]

    def subsequent_chunk_mask(size,chunk_size,num_left_chunks=-1,device=torch.device('cpu')):
        ret=torch.zeros(size,size,device=device,dtype=torch.bool)
        for i in range(size):
            start=0 if num_left_chunks<0 else max((i//chunk_size-num_left_chunks)*chunk_size,0)
            ret[i,start:min((i//chunk_size+1)*chunk_size,size)]=True
        return ret

    @contextlib.contextmanager
    def torch_autocast(*a,**k):yield

    icefall=types.ModuleType('icefall');utils=types.ModuleType('icefall.utils')
    utils.make_pad_mask=make_pad_mask;utils.subsequent_chunk_mask=subsequent_chunk_mask;utils.torch_autocast=torch_autocast;icefall.utils=utils
    interface=types.ModuleType('encoder_interface');interface.EncoderInterface=torch.nn.Module
    sys.modules.setdefault('icefall',icefall);sys.modules.setdefault('icefall.utils',utils);sys.modules.setdefault('encoder_interface',interface)
    if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))


class Transducer(torch.nn.Module):
    def __init__(self):
        super().__init__();_stubs()
        from decoder import Decoder
        from joiner import Joiner
        from zipformer import Zipformer
        five=lambda v:(v,)*5
        self.encoder=Zipformer(num_features=80,encoder_dims=five(256),attention_dim=five(192),encoder_unmasked_dims=five(192),zipformer_downsampling_factors=(1,2,4,8,2),
            nhead=five(8),feedforward_dim=five(768),num_encoder_layers=five(2),cnn_module_kernels=five(31),num_left_chunks=4,decode_chunk_size=16)
        self.decoder=Decoder(vocab_size=500,decoder_dim=512,blank_id=0,context_size=2)
        self.joiner=Joiner(encoder_dim=256,decoder_dim=512,joiner_dim=512,vocab_size=500)


def load(path=SRC/'pretrained.pt'):
    model=Transducer();state=torch.load(path,map_location='cpu',weights_only=False);state=state.get('model',state)
    missing,unexpected=model.load_state_dict(state,strict=False);model.load_report=dict(missing=list(missing),unexpected=list(unexpected))
    return model.eval()
