"""Frozen official Zipformer ONNX adapter; exposes causal encoder outputs.

Same fbank, [-1, blank] initial prediction context, unk-as-blank, strict
ready condition and one-symbol-per-frame greedy search as sherpa-onnx.
Feature extraction may run ahead offline; the encoder sees only its 45-frame
window. Every output carries the time at which that window became available.
"""
from pathlib import Path
import json
import math
import numpy as np
import onnxruntime as ort
import kaldi_native_fbank as knf
import sentencepiece as spm
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


class FrozenZipformer:
    def __init__(self):
        self.config = json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
        p = ROOT/self.config['model_dir']; suffix=self.config['model_suffix']
        opt=ort.SessionOptions(); opt.intra_op_num_threads=1; opt.inter_op_num_threads=1
        self.encoder,self.decoder,self.joiner=[ort.InferenceSession(str(p/f'{n}-{suffix}'),sess_options=opt,providers=['CPUExecutionProvider']) for n in ['encoder','decoder','joiner']]
        self.names=[x.name for x in self.encoder.get_inputs()][1:]
        self.outputs=['encoder_out']+['new_'+n for n in self.names]
        self.shapes=[x.shape for x in self.encoder.get_inputs()][1:]
        self.types=[x.type for x in self.encoder.get_inputs()][1:]
        self.sp=spm.SentencePieceProcessor(model_file=str(p/'bpe.model'))
        self.pieces=[self.sp.id_to_piece(i) for i in range(500)]

    def features(self,wave):
        o=knf.FbankOptions();o.frame_opts.dither=0;o.frame_opts.snip_edges=False
        o.mel_opts.num_bins=80;o.mel_opts.low_freq=20;o.mel_opts.high_freq=-400
        f=knf.OnlineFbank(o)
        # normalize_samples=true in the baseline: leave float samples in [-1,1].
        f.accept_waveform(16000,wave); f.input_finished()
        return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])

    def states(self,batch):
        return {name:np.zeros([batch if isinstance(d,str) else d for d in shape],dtype=np.int64 if typ=='tensor(int64)' else np.float32)
                for name,shape,typ in zip(self.names,self.shapes,self.types)}

    @staticmethod
    def availability(start,duration):
        # Strict readiness needs frame start+45 (46 frames), whose right edge
        # is (start+45)*10ms + 5ms + 12.5ms. Waveform packets are 20ms.
        required=(start+45)*.01+.0175
        if required<=duration:
            return min(duration,math.ceil((required-1e-9)/.02)*.02)
        return min(duration+.66,duration+math.ceil((required-duration-1e-9)/.02)*.02)

    def batch(self,rows):
        batch=len(rows); feats=[]; durations=[]
        for row in rows:
            x,sr=sf.read(ROOT/row['audio'],dtype='float32')
            assert sr==16000 and x.ndim==1 and len(x)==row['num_samples']
            durations.append(len(x)/sr)
            feats.append(self.features(np.concatenate([x,np.zeros(10560,np.float32)])))
        counts=[max(0,(len(f)-46)//32+1) for f in feats]
        assert min(counts)>0
        states=self.states(batch)
        context=np.tile(np.array([[-1,0]],np.int64),(batch,1))
        dec=self.decoder.run(None,{'y':context})[0]
        encs=[[] for _ in rows]; ids=[[] for _ in rows]; frames=[[] for _ in rows]
        logps=[[] for _ in rows]; cuts=[[] for _ in rows]; avail=[[] for _ in rows]
        for chunk in range(max(counts)):
            start=32*chunk
            x=np.stack([f[start:start+45] if chunk<count else np.zeros((45,80),np.float32) for f,count in zip(feats,counts)])
            values=self.encoder.run(self.outputs,{'x':x,**states})
            enc=values[0]; states=dict(zip(self.names,values[1:])); assert enc.shape[1]==8
            for i,count in enumerate(counts):
                if chunk<count:
                    encs[i].append(enc[i].astype(np.float16));avail[i].append(self.availability(start,durations[i]))
            for t in range(8):
                logits=self.joiner.run(None,{'encoder_out':np.ascontiguousarray(enc[:,t]),'decoder_out':dec})[0]
                y=logits.argmax(1)
                changed=False
                for i,token in enumerate(y):
                    if chunk<counts[i] and token not in (0,2):
                        token=int(token);ids[i].append(token);frames[i].append(chunk*8+t)
                        z=logits[i]/2;z=z-z.max();logps[i].append(float(z[token]-np.log(np.exp(z).sum())))
                        context[i]=[context[i,-1],token];changed=True
                if changed:
                    dec=self.decoder.run(None,{'y':context})[0]
            for i,count in enumerate(counts):
                if chunk<count:cuts[i].append(len(ids[i]))
        return [dict(enc=np.concatenate(encs[i]),ids=np.array(ids[i],np.int32),frames=np.array(frames[i],np.int32),
                     logp=np.array(logps[i],np.float32),cuts=np.array(cuts[i],np.int32),available=np.array(avail[i],np.float64),
                     gold=np.array(self.sp.encode(row['text']),np.int32),duration=np.array(durations[i]),
                     hypothesis=np.array(self.sp.decode(ids[i])),reference=np.array(row['text']))
                for i,row in enumerate(rows)]


def align_gold(gold,ids,frames,total_frames):
    """Offline monotonic timing labels from matching BPE blocks, not audio input.

    This is an approximate alignment, not RNN-T forced alignment or phonetic
    word ends. Unmatched reference token times are interpolated between anchors.
    """
    import difflib
    a=[];b=[]
    for block in difflib.SequenceMatcher(a=gold.tolist(),b=ids.tolist(),autojunk=False).get_matching_blocks():
        for offset in range(block.size):a.append(block.a+offset);b.append(int(frames[block.b+offset]))
    if a:
        timing=np.rint(np.interp(np.arange(len(gold)),a,b)).astype(np.int32)
    else:
        timing=np.linspace(0,max(0,total_frames-1),len(gold)).astype(np.int32)
    return np.maximum.accumulate(timing),len(a)/max(1,len(gold))
