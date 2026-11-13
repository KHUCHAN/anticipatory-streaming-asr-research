#!/usr/bin/env python3
"""Cheap control: native PN + joiner on last arrived frame, suppress blank.

No added model weights or training. Conditional nonblank probabilities and
the original nonblank mass are exposed to a dev-fitted gate. PN and joiner
are called separately per chunk and their actual CPU time is charged.
"""
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/zipformer_ours'


def main():
    c=json.loads((ROOT/'configs/zipformer_baseline.json').read_text());cfg=json.loads((ROOT/'configs/zipformer_ours.json').read_text())
    p=ROOT/c['model_dir'];opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    decoder=ort.InferenceSession(str(p/('decoder-'+c['model_suffix'])),sess_options=opt,providers=['CPUExecutionProvider'])
    joiner=ort.InferenceSession(str(p/('joiner-'+c['model_suffix'])),sess_options=opt,providers=['CPUExecutionProvider'])
    sp=spm.SentencePieceProcessor(model_file=str(p/'bpe.model'));pieces=[sp.id_to_piece(i) for i in range(500)]
    rows=[]
    for name in ['gate_fit','gate_calibration']:
        rows += [json.loads(l) for l in (ROOT/cfg['dev_protocol']/(name+'.jsonl')).open()]
    for split in cfg['evaluation_splits']:
        rows += [json.loads(l) for l in (ROOT/f'data/processed/manifests/{split}.jsonl').open()]
    directory=OUT/'native_control';directory.mkdir(parents=True,exist_ok=True)
    done=0
    for row in rows:
        target=directory/(row['utterance_id']+'.json')
        if target.exists():continue
        cache=ROOT/cfg['features']/row['split']/(row['utterance_id']+'.npz')
        last=0
        while not cache.exists():
            if time.time()-last>300:print('WAIT CACHE',row['split'],row['utterance_id'],flush=True);last=time.time()
            time.sleep(5)
        with np.load(cache) as loaded:data={k:loaded[k] for k in loaded.files}
        native=data['ids'].tolist();events=[];previous=0;started=time.perf_counter()
        for chunk,(count,available) in enumerate(zip(data['cuts'],data['available'])):
            count=int(count);begin=time.perf_counter()
            context=([-1,0]+native[:count])[-2:]
            d=decoder.run(None,{'y':np.array([context],np.int64)})[0]
            x=data['enc'][(chunk+1)*8-1:(chunk+1)*8].astype(np.float32)
            logits=joiner.run(None,{'encoder_out':x,'decoder_out':d})[0][0]
            z=logits-logits.max();allprob=np.exp(z);allprob/=allprob.sum()
            probability=allprob[3:]/max(float(allprob[3:].sum()),1e-30)
            order=probability.argsort();g=int(order[-1])+3
            top1=float(probability[order[-1]]);top2=float(probability[order[-2]])
            entropy=float(-(probability*np.log(np.maximum(probability,1e-30))).sum())
            lp=float(data['logp'][count-1]) if count else 0.
            gap=((chunk+1)*8-1-int(data['frames'][count-1]))*.04 if count else ((chunk+1)*8)*.04
            features=[top1,top2,entropy,float(not pieces[g].startswith('▁')),lp,min(5.,gap),min(12,count-previous)/12,min(64,count)/64,
                      float(np.log(max(float(allprob[3:].sum()),1e-30)))]
            elapsed=time.perf_counter()-begin
            events.append(dict(g=g,features=features,seconds=elapsed,text_bytes=0,audio_kv_bytes=0,
                               available=float(available),native_count=count,eligible=bool(available<float(data['duration']))))
            previous=count
        result=dict(utterance_id=row['utterance_id'],events=events,processing_seconds=time.perf_counter()-started,
                    method='existing native PN and joiner; conditional nonblank argmax, extra gate feature=log(nonblank mass); no added neural weights or retained text/audio KV')
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,separators=(',',':'))+'\n');temporary.replace(target)
        done+=1
        if done%200==0:print('NATIVE CONTROL',done,'/',len(rows),flush=True)
    (OUT/'native_control_complete.json').write_text(json.dumps(dict(utterances=len(rows),added_model_parameters=0,finished_unix=time.time()),indent=2)+'\n')


if __name__=='__main__':main()
