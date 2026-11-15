"""Closed-loop inference and complete CPU compute accounting from raw audio."""
import json
import time

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch

from common import ROOT,OUT,state_module,module,word_error
from model import Runtime
from prepare import features


class Decoder:
    def __init__(self,bridge):
        self.bridge=bridge;self.context=torch.tensor([[-1,0]],dtype=torch.long)
        self.pn=bridge.prediction(self.context);self.ids=[]

    def advance(self,states):
        for state in states:
            logits=self.bridge(state[None],self.pn);token=int(logits.argmax(-1))
            if token not in (0,2):
                self.ids.append(token);self.context=torch.tensor([[int(self.context[0,-1]),token]])
                self.pn=self.bridge.prediction(self.context)


class Inference:
    def __init__(self,cfg,model=None,baseline=False):
        self.cfg=cfg;self.model=model;self.baseline=baseline
        self.bridge=state_module.FrozenRNNT(ROOT/cfg['bridge']).eval()
        bcfg=json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
        self.sp=spm.SentencePieceProcessor(model_file=str(ROOT/bcfg['model_dir']/'bpe.model'))
        stat=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text())
        self.mean=torch.tensor(stat['mean']);self.std=torch.tensor(stat['std'])
        fs=json.loads((OUT/'cache/stats.json').read_text())
        self.fmean=np.array(fs['mean'],np.float32);self.fstd=np.array(fs['std'],np.float32)
        n=json.loads((ROOT/cfg['encoder_cache']/'complete.json').read_text())['frames']
        self.targets=np.memmap(ROOT/cfg['encoder_cache']/'encoder.f16',dtype=np.float16,mode='r',shape=(n,512))
        if baseline:
            adapter=module('corrector_frozen_adapter','scripts/zipformer_features.py')
            self.full=adapter.FrozenZipformer()

    @torch.inference_mode()
    def utterance(self,row):
        io=time.perf_counter();wave,sr=sf.read(ROOT/row['audio'],dtype='float32');io=time.perf_counter()-io
        assert sr==16000
        start=time.perf_counter();f=features(np.concatenate([wave,np.zeros(10560,np.float32)]))
        feature_seconds=time.perf_counter()-start
        assert max(0,(len(f)-46)//32+1)==row['nchunks']
        clock={key:0. for key in ['audio','prediction','correction','feedback','decoder']}
        decoder=Decoder(self.bridge);runtime=Runtime(self.model) if not self.baseline else None
        states=self.full.states(1) if self.baseline else None
        times=[];metric_seconds=0.;mae_sum=0.;mae_count=0
        for chunk in range(row['nchunks']):
            begin=time.perf_counter()
            if self.baseline:
                x=np.ascontiguousarray(f[32*chunk:32*chunk+45][None])
                a=time.perf_counter();values=self.full.encoder.run(self.full.outputs,{'x':x,**states})
                states=dict(zip(self.full.names,values[1:]));clock['audio']+=time.perf_counter()-a
                raw=torch.from_numpy(values[0][0])
            else:
                # Each fbank frame enters the causal conv exactly once. First
                # chunk consumes 45; following chunks consume 32 new frames.
                lo=0 if chunk==0 else 32*(chunk-1)+45;hi=32*chunk+45
                x=torch.from_numpy(np.ascontiguousarray((f[lo:hi]-self.fmean)/self.fstd))
                raw=runtime.advance(x,clock)*self.std+self.mean
            a=time.perf_counter();decoder.advance(raw);clock['decoder']+=time.perf_counter()-a
            times.append(time.perf_counter()-begin)
            a=time.perf_counter()
            target=torch.from_numpy(np.array(self.targets[row['offset']+chunk*8:row['offset']+(chunk+1)*8],np.float32))
            mae_sum+=float(((raw-target)/self.std).abs().sum());mae_count+=raw.numel()
            metric_seconds+=time.perf_counter()-a
        hypothesis=self.sp.decode(decoder.ids)
        elapsed=time.perf_counter()-start-metric_seconds
        errors,words=word_error(row['text'],hypothesis)
        accounted=feature_seconds+sum(clock.values())
        return dict(utterance_id=row['utterance_id'],speaker_id=row['speaker_id'],split=row['split'],
            errors=errors,reference_words=words,hypothesis=hypothesis,reference=row['text'],token_ids=decoder.ids,
            audio_seconds=len(wave)/sr,chunks=row['nchunks'],mae=mae_sum/mae_count,
            mae_sum=mae_sum,mae_count=mae_count,total_seconds=elapsed,audio_io_seconds=io,
            total_with_audio_io_seconds=elapsed+io,feature_seconds=feature_seconds,
            components=clock,other_overhead_seconds=elapsed-accounted,chunk_seconds=times)


def corpus(infer,rows,emit=None):
    results=[];started=time.time()
    for i,row in enumerate(rows,1):
        results.append(infer.utterance(row))
        if emit and i%100==0:emit(dict(event='decode_progress',done=i,total=len(rows),seconds=time.time()-started))
    errors=sum(r['errors'] for r in results);words=sum(r['reference_words'] for r in results)
    return dict(wer_percent=100*errors/words,errors=errors,reference_words=words,utterances=len(rows)),results
