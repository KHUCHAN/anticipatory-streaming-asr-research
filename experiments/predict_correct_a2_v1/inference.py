"""Native-decoder replay with complete frontend/decoder cost accounting."""
import json
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from common import ROOT,features,word_error
from model import Runtime


class Graph:
    def __init__(self,directory,suffix,load_encoder=True):
        option=ort.SessionOptions();option.intra_op_num_threads=1;option.inter_op_num_threads=1
        self.encoder=ort.InferenceSession(str(directory/f'encoder-{suffix}'),sess_options=option,providers=['CPUExecutionProvider']) if load_encoder else None
        self.decoder,self.joiner=[ort.InferenceSession(str(directory/f'{name}-{suffix}'),sess_options=option,providers=['CPUExecutionProvider']) for name in ['decoder','joiner']]
        self.window=self.encoder.get_inputs()[0].shape[1] if load_encoder else 45
        meta=self.encoder.get_modelmeta().custom_metadata_map if load_encoder else {};self.hop=int(meta.get('decode_chunk_len','32'))
        self.names=[v.name for v in self.encoder.get_inputs()][1:] if load_encoder else []
        self.outputs=['encoder_out']+['new_'+n for n in self.names]
        self.tokens={int(line.rsplit(' ',1)[1]):line.rsplit(' ',1)[0] for line in (directory/'tokens.txt').read_text().splitlines()}
        self.context=int(self.decoder.get_modelmeta().custom_metadata_map.get('context_size','2'))
        self.directory=directory

    def states(self):
        return {v.name:np.zeros([1 if isinstance(d,str) else d for d in v.shape],dtype=np.int64 if v.type=='tensor(int64)' else np.float32) for v in self.encoder.get_inputs()[1:]}

    def text(self,ids):return ' '.join(''.join(self.tokens[i] for i in ids).replace('▁',' ').upper().split())


class Decoder:
    def __init__(self,graph):
        self.graph=graph;self.ids=[];self.context=np.array([[-1]*(graph.context-1)+[0]],np.int64)
        self.pn=graph.decoder.run(None,{'y':self.context})[0]

    def advance(self,states):
        for state in states:
            logits=self.graph.joiner.run(None,{'encoder_out':np.ascontiguousarray(state[None]),'decoder_out':self.pn})[0]
            token=int(logits.argmax(-1)[0])
            if token not in (0,2):
                self.ids.append(token);self.context=np.concatenate([self.context[:,1:],np.array([[token]],np.int64)],1)
                self.pn=self.graph.decoder.run(None,{'y':self.context})[0]


class Inference:
    def __init__(self,cfg,model=None,small=False):
        self.cfg=cfg;self.model=model
        teacher=json.loads((ROOT/cfg['teacher_config']).read_text())
        directory=ROOT/(cfg['small_model'] if small else teacher['model_dir'])
        suffix='epoch-99-avg-1.onnx' if small else teacher['model_suffix']
        self.graph=Graph(directory,suffix,load_encoder=model is None)
        self.window=45 if model is not None else self.graph.window
        self.hop=32 if model is not None else self.graph.hop
        stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text());self.mean=torch.tensor(stats['mean']);self.std=torch.tensor(stats['std'])
        fs=json.loads((ROOT/cfg['input_cache']/'stats.json').read_text());self.fmean=np.array(fs['mean'],np.float32);self.fstd=np.array(fs['std'],np.float32)
        n=json.loads((ROOT/cfg['encoder_cache']/'complete.json').read_text())['frames']
        self.targets=np.memmap(ROOT/cfg['encoder_cache']/'encoder.f16',dtype=np.float16,mode='r',shape=(n,512))
        self.small=small

    @torch.inference_mode()
    def utterance(self,row):
        io=time.perf_counter();wave,sr=sf.read(ROOT/row['audio'],dtype='float32');io=time.perf_counter()-io;assert sr==16000
        start=time.perf_counter();f=features(np.concatenate([wave,np.zeros(10560,np.float32)]));feature_seconds=time.perf_counter()-start
        count=max(0,(len(f)-(self.window+1))//self.hop+1)
        decoder=Decoder(self.graph);runtime=Runtime(self.model) if self.model is not None else None
        states=None if runtime else self.graph.states();clock={k:0.0 for k in ['audio','head','decoder']}
        times=[];metric=0.;mae_sum=0.;mae_count=0;output_frames=0
        for chunk in range(count):
            begin=time.perf_counter()
            if runtime is None:
                x=np.ascontiguousarray(f[chunk*self.hop:chunk*self.hop+self.window][None]);a=time.perf_counter()
                values=self.graph.encoder.run(self.graph.outputs,{'x':x,**states});clock['audio']+=time.perf_counter()-a
                states=dict(zip(self.graph.names,values[1:]));raw=values[0][0]
            else:
                lo=0 if chunk==0 else 32*(chunk-1)+45;hi=32*chunk+45
                x=torch.from_numpy(np.ascontiguousarray((f[lo:hi]-self.fmean)/self.fstd))
                raw=(runtime.advance(x,clock)*self.std+self.mean).numpy()
            output_frames+=len(raw);a=time.perf_counter();decoder.advance(raw);clock['decoder']+=time.perf_counter()-a
            times.append(time.perf_counter()-begin)
            if not self.small:
                a=time.perf_counter();target=np.array(self.targets[row['offset']+chunk*8:row['offset']+(chunk+1)*8],np.float32)
                mae_sum+=float(np.abs((raw-target)/self.std.numpy()).sum());mae_count+=raw.size;metric+=time.perf_counter()-a
        hypothesis=self.graph.text(decoder.ids);elapsed=time.perf_counter()-start-metric
        errors,words=word_error(row['text'],hypothesis)
        return dict(utterance_id=row['utterance_id'],speaker_id=row['speaker_id'],split=row['split'],reference=row['text'],hypothesis=hypothesis,
            token_ids=decoder.ids,errors=errors,reference_words=words,audio_seconds=len(wave)/sr,chunks=count,
            output_frames=output_frames,window=self.window,hop=self.hop,feature_seconds=feature_seconds,
            components=clock,total_seconds=elapsed,audio_io_seconds=io,total_with_audio_io_seconds=elapsed+io,
            other_overhead_seconds=elapsed-feature_seconds-sum(clock.values()),chunk_seconds=times,
            mae_sum=mae_sum,mae_count=mae_count)


def corpus(infer,rows):
    records=[infer.utterance(row) for row in rows]
    return dict(wer_percent=100*sum(x['errors'] for x in records)/sum(x['reference_words'] for x in records),
        errors=sum(x['errors'] for x in records),reference_words=sum(x['reference_words'] for x in records),utterances=len(records)),records
