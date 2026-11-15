"""Equivalent FP32 ONNX first/step graphs; no quantization or weight changes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from common import ROOT,HERE,OUT,config,sha,write
from model import A2,Runtime


class ExportStep(torch.nn.Module):
    def __init__(self,model,first):super().__init__();self.model=model;self.first=first

    def forward(self,features,*state):
        h,raw,caches,_=self.model.audio.stream(features,None if self.first else state[0],None if self.first else list(state[1:]),0 if self.first else 45)
        value=self.model.head(h[:,-8:].flatten(-2)).reshape(1,8,512)
        return (value,raw,*caches)


class OnnxModel:
    def __init__(self,directory):
        option=ort.SessionOptions();option.intra_op_num_threads=1;option.inter_op_num_threads=1
        self.first=ort.InferenceSession(str(directory/'first.onnx'),sess_options=option,providers=['CPUExecutionProvider'])
        self.step=ort.InferenceSession(str(directory/'step.onnx'),sess_options=option,providers=['CPUExecutionProvider'])
        self.state_names=[x.name for x in self.step.get_inputs()][1:]


class OnnxRuntime:
    def __init__(self,model):self.model=model;self.state=None

    def advance(self,features):
        values={'features':np.ascontiguousarray(features[None])}
        if self.state is None:output=self.model.first.run(None,values)
        else:output=self.model.step.run(None,{**values,**self.state})
        self.state=dict(zip(self.model.state_names,output[1:]));return output[0][0]


def export(model,cfg,name,digest):
    directory=OUT/'onnx'/name/digest[:12];directory.mkdir(parents=True,exist_ok=True)
    output_names=['normalized_states','new_raw']+[f'new_cache_{i}' for i in range(len(model.audio.blocks))]
    input_names=['features','raw']+[f'cache_{i}' for i in range(len(model.audio.blocks))]
    if not (directory/'metadata.json').exists():
        # Isolated packages leave the active training interpreter untouched.
        sys.path.insert(0,str(ROOT/'.local/a2-export-deps'))
        first=ExportStep(model,True).eval();step=ExportStep(model,False).eval()
        x=torch.zeros(1,45,80)
        with torch.no_grad():states=first(x)[1:]
        for filename,wrapper,args,names in [('first.onnx',first,(x,),['features']),('step.onnx',step,(torch.zeros(1,32,80),*states),input_names)]:
            torch.onnx.export(wrapper,args,str(directory/filename),input_names=names,output_names=output_names,
                opset_version=17,dynamo=False,export_params=True,do_constant_folding=True,external_data=False)
        onnx_model=OnnxModel(directory);runtime=OnnxRuntime(onnx_model);reference=Runtime(model)
        generator=torch.Generator().manual_seed(7237);differences=[]
        for i in range(12):
            features=torch.randn(45 if i==0 else 32,80,generator=generator)
            with torch.no_grad():a=reference.advance(features).numpy()
            b=runtime.advance(features.numpy());differences.append(float(np.max(np.abs(a-b))))
        assert max(differences)<2e-4,differences
        record=dict(source_model_sha256=digest,model_source_sha256=sha(HERE/'model.py'),
            backend_source_sha256=sha(Path(__file__)),onnx_version=importlib.metadata.version('onnx'),
            files={n:sha(directory/n) for n in ['first.onnx','step.onnx']},
            fp32=True,quantized=False,weights_changed=False,random_stream_max_differences=differences,
            phase_contract='First graph consumes 45 frames at stride phase 0; step graph consumes 32 frames at stride phase 1, preserved after every subsequent hop.')
        write(directory/'metadata.json',record)
    record=json.loads((directory/'metadata.json').read_text());assert record['source_model_sha256']==digest
    for n,h in record['files'].items():assert sha(directory/n)==h,n
    return directory,record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['initial','mae','kl'],default='initial');args=parser.parse_args()
    cfg=config();torch.set_num_threads(1);torch.manual_seed(cfg['seed']);model=A2(cfg).eval()
    digest=f'initial-seed-{cfg["seed"]}'
    if args.phase!='initial':
        path=OUT/args.phase/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);digest=sha(path)
    directory,record=export(model,cfg,args.phase,digest)
    write(HERE/'artifacts'/f'onnx_{args.phase}.json',dict(directory=str(directory.relative_to(ROOT)),**record))
    print(json.dumps(dict(directory=str(directory.relative_to(ROOT)),maximum_difference=max(record['random_stream_max_differences']))))


if __name__=='__main__':main()
