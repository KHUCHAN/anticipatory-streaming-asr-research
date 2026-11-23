"""Checks that the PyTorch copy reproduces the exported ONNX encoder on real chunks."""
import json,sys
import numpy as np,onnxruntime as ort,torch
from torch_model import ROOT,load
m=load();print('missing',len(m.load_report['missing']),m.load_report['missing'][:5],'unexpected',len(m.load_report['unexpected']),m.load_report['unexpected'][:5])
d=ROOT/'checkpoints/baselines/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17';s=ort.InferenceSession(str(d/'encoder-epoch-99-avg-1.onnx'),providers=['CPUExecutionProvider'])
ins=s.get_inputs();print('onnx inputs',len(ins),[(i.name,i.shape) for i in ins[:3]],[o.name for o in s.get_outputs()][:2])
st=m.encoder.get_init_state();print('torch states',len(st),[tuple(x.shape) for x in st[:8]])
C=ROOT/'runs/predict_correct_v1/cache';idx=json.loads((C/'index.json').read_text());rows=idx['rows'] if isinstance(idx,dict) and 'rows' in idx else idx
print('index type',type(idx).__name__,list(idx.keys())[:8] if isinstance(idx,dict) else rows[0])
F=np.memmap(C/'fbank.f16',np.float16,'r').reshape(-1,80);r=rows[3];f=torch.from_numpy(np.array(F[r['feature_offset']:r['feature_offset']+r['feature_frames']],np.float32))
names=[i.name for i in ins][1:];so={n:x.numpy() for n,x in zip(names,st)};worst=0
with torch.no_grad():
    for c in range(8):
        x=f[c*32:c*32+39][None];out,_,st=m.encoder.streaming_forward(x,torch.tensor([39]),st);out=m.joiner.encoder_proj(out)
        res=s.run(None,{'x':x.numpy(),**so});so=dict(zip(names,res[1:]));worst=max(worst,float(np.abs(res[0]-out.numpy()).max()))
        if c==0:print('out shape',res[0].shape,out.shape,'scale',float(np.abs(res[0]).mean()))
print('max abs diff over 8 chunks',worst,'state diff',max(float(np.abs(so[n]-x.numpy()).max()) for n,x in zip(names,st)))
