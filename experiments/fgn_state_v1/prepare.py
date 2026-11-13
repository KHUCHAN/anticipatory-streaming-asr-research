#!/usr/bin/env python3
"""Pack existing immutable caches and export a frozen differentiable bridge."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/fgn_state_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'cache';cache.mkdir(exist_ok=True)
    base=json.loads((ROOT/'configs/zipformer_baseline.json').read_text());p=ROOT/base['model_dir']
    decoder=onnx.load(p/('decoder-'+base['model_suffix']));joiner=onnx.load(p/('joiner-'+base['model_suffix']))
    arrays={v.name:numpy_helper.to_array(v).copy() for v in decoder.graph.initializer}
    arrays.update({v.name:numpy_helper.to_array(v).copy() for v in joiner.graph.initializer})
    weights={'embedding':arrays['decoder.embedding.weight'],'conv':arrays['decoder.conv.weight'],
             'decoder_weight':arrays['decoder_proj.weight'],'decoder_bias':arrays['decoder_proj.bias'],
             'joiner_weight':arrays['output_linear.weight'],'joiner_bias':arrays['output_linear.bias']}
    np.savez(OUT/'rnnt_bridge.npz',**weights)
    if (cache/'complete.json').exists():print('Packed cache already complete.',flush=True);return
    records=[];nframes=0;started=time.time()
    for split in ['train-clean-100','dev-clean','dev-other','test-clean','test-other']:
        for line in (ROOT/f'data/processed/manifests/{split}.jsonl').open():
            row=json.loads(line);row['offset']=nframes
            with np.load(ROOT/f"data/processed/zipformer_forecasts/{split}/{row['utterance_id']}.npz") as v:
                row['nframes']=len(v['enc']);row['nchunks']=len(v['cuts'])
            nframes+=row['nframes'];records.append(row)
        print('Indexed',split,'frames',nframes,flush=True)
    enc=np.memmap(cache/'encoder.f16',dtype=np.float16,mode='w+',shape=(nframes,512))
    ctx=np.memmap(cache/'contexts.i16',dtype=np.int16,mode='w+',shape=(nframes,2))
    tgt=np.memmap(cache/'tokens.i16',dtype=np.int16,mode='w+',shape=(nframes,))
    sums=np.zeros(512,np.float64);squares=np.zeros(512,np.float64);count=0
    for i,row in enumerate(records):
        with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{row['utterance_id']}.npz") as v:
            a=row['offset'];z=a+row['nframes'];x=v['enc'];enc[a:z]=x
            tokens=np.zeros(row['nframes'],np.int16);tokens[v['frames']]=v['ids'];tgt[a:z]=tokens
            context=np.empty((row['nframes'],2),np.int16);last=[-1,0]
            for t,token in enumerate(tokens):
                context[t]=last
                if token:last=[last[-1],int(token)]
            ctx[a:z]=context
            if row['split']=='train-clean-100':
                values=x.astype(np.float64);sums+=values.sum(0);squares+=(values*values).sum(0);count+=len(values)
        if (i+1)%2000==0:print('Packed',i+1,'/',len(records),'seconds',round(time.time()-started),flush=True)
    enc.flush();ctx.flush();tgt.flush()
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,.0025))
    stats=dict(mean=mean.tolist(),std=std.tolist(),train_frames=count,source='train-clean-100 only')
    (cache/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    (cache/'index.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
    (cache/'complete.json').write_text(json.dumps(dict(utterances=len(records),frames=nframes,seconds=time.time()-started),indent=2)+'\n')
    print('PACK COMPLETE',len(records),'utterances',nframes,'frames',flush=True)


if __name__=='__main__':main()
