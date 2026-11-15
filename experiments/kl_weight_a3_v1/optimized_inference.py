"""Complete A2 cost with verified ONNX audio/head graphs and native decoder."""
import json
import time

import numpy as np
import soundfile as sf

from common import ROOT,features,word_error
from inference import Graph,Decoder
from onnx_backend import OnnxModel,OnnxRuntime


class OptimizedInference:
    def __init__(self,cfg,directory):
        self.model=OnnxModel(directory)
        teacher=json.loads((ROOT/cfg['teacher_config']).read_text())
        self.graph=Graph(ROOT/teacher['model_dir'],teacher['model_suffix'],load_encoder=False)
        stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text());self.mean=np.array(stats['mean'],np.float32);self.std=np.array(stats['std'],np.float32)
        stats=json.loads((ROOT/cfg['input_cache']/'stats.json').read_text());self.fmean=np.array(stats['mean'],np.float32);self.fstd=np.array(stats['std'],np.float32)
        n=json.loads((ROOT/cfg['encoder_cache']/'complete.json').read_text())['frames']
        self.targets=np.memmap(ROOT/cfg['encoder_cache']/'encoder.f16',dtype=np.float16,mode='r',shape=(n,512))

    def utterance(self,row):
        io=time.perf_counter();wave,sr=sf.read(ROOT/row['audio'],dtype='float32');io=time.perf_counter()-io;assert sr==16000
        start=time.perf_counter();f=features(np.concatenate([wave,np.zeros(10560,np.float32)]));feature_seconds=time.perf_counter()-start
        count=max(0,(len(f)-46)//32+1);runtime=OnnxRuntime(self.model);decoder=Decoder(self.graph)
        clock={k:0. for k in ['audio','head','decoder']};times=[];metric=0.;mae_sum=0.;mae_count=0
        for chunk in range(count):
            begin=time.perf_counter();lo=0 if chunk==0 else 32*(chunk-1)+45;hi=32*chunk+45
            x=np.ascontiguousarray((f[lo:hi]-self.fmean)/self.fstd)
            a=time.perf_counter();normalized=runtime.advance(x);clock['audio']+=time.perf_counter()-a
            raw=normalized*self.std+self.mean;a=time.perf_counter();decoder.advance(raw);clock['decoder']+=time.perf_counter()-a
            times.append(time.perf_counter()-begin)
            a=time.perf_counter();target=np.array(self.targets[row['offset']+chunk*8:row['offset']+(chunk+1)*8],np.float32)
            mae_sum+=float(np.abs((raw-target)/self.std).sum());mae_count+=raw.size;metric+=time.perf_counter()-a
        hypothesis=self.graph.text(decoder.ids);elapsed=time.perf_counter()-start-metric;errors,words=word_error(row['text'],hypothesis)
        return dict(utterance_id=row['utterance_id'],speaker_id=row['speaker_id'],split=row['split'],reference=row['text'],hypothesis=hypothesis,
            token_ids=decoder.ids,errors=errors,reference_words=words,audio_seconds=len(wave)/sr,chunks=count,output_frames=count*8,
            window=45,hop=32,feature_seconds=feature_seconds,components=clock,total_seconds=elapsed,audio_io_seconds=io,
            total_with_audio_io_seconds=elapsed+io,other_overhead_seconds=elapsed-feature_seconds-sum(clock.values()),chunk_seconds=times,
            mae_sum=mae_sum,mae_count=mae_count)
