"""Streaming replay of the frozen teacher with isolated preview passes on partly filled chunks.

For every chunk, each preview pass sees the first k real feature frames of the 45-frame window and a filler for the rest.
Preview passes read the native encoder state and decoder context but never write them; the native pass is unchanged.
"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

from shared import OUT,ROOT,config,features,rows,write
import inference  # frozen A2 adapter (Graph, Inference)


def decode(graph,encoder_out,context,pn):
    """Greedy, at most one symbol per frame. Returns emitted ids and the advanced decoder state."""
    ids=[]
    for state in encoder_out:
        token=int(graph.joiner.run(None,{'encoder_out':np.ascontiguousarray(state[None]),'decoder_out':pn})[0].argmax(-1)[0])
        if token not in (0,2):
            ids.append(token);context=np.concatenate([context[:,1:],np.array([[token]],np.int64)],1);pn=graph.decoder.run(None,{'y':context})[0]
    return ids,context,pn


def fill(window,k,kind,running_mean):
    x=window.copy()
    if kind=='zero':x[k:]=0.0
    elif kind=='repeat':x[k:]=x[k-1]
    elif kind=='mean':x[k:]=running_mean
    else:raise ValueError(kind)
    return x


def main():
    cfg=config();a2cfg=json.loads((ROOT/cfg['a2_config']).read_text());limit=int(sys.argv[1]) if len(sys.argv)>1 else None
    inf=inference.Inference(a2cfg);graph=inf.graph;W=cfg['feature_window'];H=cfg['feature_hop'];assert (graph.window,graph.hop)==(W,H)
    option=ort.SessionOptions();option.intra_op_num_threads=1;option.inter_op_num_threads=1
    path=[p for p in graph.directory.glob('encoder-*.onnx') if '.int8.' in p.name][0]
    int8=ort.InferenceSession(str(path),sess_options=option,providers=['CPUExecutionProvider']);assert [v.name for v in int8.get_inputs()][1:]==graph.names
    variants=[(f,k,'fp32') for f in cfg['fillers'] for k in cfg['preview_real_frames']]+[(cfg['int8_preview_filler'],k,'int8') for k in cfg['preview_real_frames']]
    selected=rows(cfg)[:limit] if limit else rows(cfg);records=[]
    inf.utterance(selected[0])  # warmup, discarded
    for n,row in enumerate(selected):
        wave,sr=sf.read(ROOT/row['audio'],dtype='float32');assert sr==16000
        begin=time.perf_counter();f=features(np.concatenate([wave,np.zeros(10560,np.float32)]));feature_seconds=time.perf_counter()-begin
        count=max(0,(len(f)-(W+1))//H+1);states=graph.states();context=np.array([[-1]*(graph.context-1)+[0]],np.int64);pn=graph.decoder.run(None,{'y':context})[0]
        chunks=[];total=np.zeros(f.shape[1],np.float64)
        for c in range(count):
            window=np.ascontiguousarray(f[c*H:c*H+W]);previews={}
            for filler,k,precision in variants:
                seen=c*H+k;mean=(f[:seen].mean(0)).astype(np.float32)
                begin=time.perf_counter();x=fill(window,k,filler,mean)[None]
                session=int8 if precision=='int8' else graph.encoder
                enc=session.run(['encoder_out'],{'x':x,**states})[0][0];ids,_,_=decode(graph,enc,context,pn)
                previews[f'{filler}|{k}|{precision}']=dict(ids=ids,seconds=time.perf_counter()-begin)
            begin=time.perf_counter();values=graph.encoder.run(graph.outputs,{'x':window[None],**states})
            states=dict(zip(graph.names,values[1:]));ids,context,pn=decode(graph,values[0][0],context,pn)
            chunks.append(dict(ids=ids,seconds=time.perf_counter()-begin,previews=previews))
        native=inf.utterance(row)  # independent native pass through the frozen adapter
        final=[i for ch in chunks for i in ch['ids']]
        records.append(dict(utterance_id=row['utterance_id'],split=row['split'],reference=row['text'],audio_seconds=len(wave)/sr,
            feature_seconds=feature_seconds,chunks=chunks,final_ids=final,native_ids=native['token_ids'],native_equal=final==native['token_ids'],
            hypothesis=graph.text(final)))
        if n%20==0:print(n,len(selected),flush=True)
    vocabulary={str(k):v for k,v in graph.tokens.items()}
    write(OUT/('replay.json' if not limit else 'replay_smoke.json'),dict(variants=[f'{f}|{k}|{p}' for f,k,p in variants],tokens=vocabulary,records=records))
    print('done',len(records),'native_equal',sum(r['native_equal'] for r in records))


if __name__=='__main__':main()
