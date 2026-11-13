import argparse,hashlib,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
import sentencepiece as spm
from common import ROOT,OUT,HERE,config,sha,write,rows_for
from model import Hybrid
from runtime import Runtime

KINDS=config()['variants'];MODELS={};PIECES=[];CFG=None

def initialize():
    global PIECES,CFG
    torch.set_num_threads(1);torch.set_grad_enabled(False);CFG=config()
    for variant in KINDS:
        ck=torch.load(OUT/variant/'best.pt',map_location='cpu',weights_only=False)
        model=Hybrid(variant,initialize=False);model.load_state_dict(ck['model']);MODELS[variant]=model.eval()
    bc=json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/bc['model_dir']/'bpe.model'))
    PIECES=[sp.id_to_piece(i) for i in range(500)]

def process(task):
    part,row=task;uid=row['utterance_id']
    with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{uid}.npz") as v:
        enc=torch.from_numpy(v['enc'].astype(np.float32));cuts=v['cuts'].tolist();available=v['available'].tolist()
        logp=v['logp'].tolist();frames=v['frames'].tolist();duration=float(v['duration'])
    old=json.loads((ROOT/f'runs/zipformer_ours/events_{part}/{uid}.json').read_text())
    assert cuts==[e['native_count'] for e in old['events']['full']]
    native=old['native'];correct=set(old['correct_native_positions']);order=KINDS.copy()
    np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)).shuffle(order)
    for kind in order:
        runtime=Runtime(MODELS[kind],PIECES,CFG['inference_members']);records=[];previous=0;started=time.perf_counter()
        generator=torch.Generator().manual_seed(int(hashlib.sha256((uid+'band_fgn').encode()).hexdigest()[:12],16))
        for chunk,(count,clock) in enumerate(zip(cuts,available)):
            end=(chunk+1)*8
            gap=(end-1-frames[count-1])*.04 if count else end*.04
            result=runtime.advance(enc[:end],native[previous:count],logp[count-1] if count else 0.,gap,count,generator)
            result.update(available=clock,native_count=count,eligible=bool(clock<duration and result['g']>=3),
                label_keep=int(count<len(native) and result['g']==native[count]),
                label_correct_and_keep=int(count<len(native) and result['g']==native[count] and count in correct))
            records.append(result);previous=count
        old['events'][kind]=records;old['processing_seconds'][kind]=time.perf_counter()-started
    return old

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',required=True,choices=['dev','test']);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=config()
    if a.part=='test' and not a.smoke:assert (OUT/'gates.json').exists(),'Calibrate dev BEFORE test collection'
    rows=rows_for(a.part,cfg);expected={r['utterance_id'] for r in rows}
    if a.limit:rows=rows[:a.limit]
    name=('smoke_' if a.smoke else 'events_')+a.part;out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    dependencies=['avs/model.py','avs/stream.py','avs/zipformer_forecaster.py','scripts/collect_zipformer_forecasts.py',
                  'experiments/fgn_state_v1/model.py','experiments/fgn_state_v1/runtime.py']
    provenance=dict(model_sha256={v:sha(OUT/v/'best.pt') for v in KINDS},
                    source_sha256={n:sha(HERE/n) for n in ['model.py','runtime.py','collect.py','config.json']},
                    dependency_sha256={n:sha(ROOT/n) for n in dependencies})
    if a.part=='test' and not a.smoke:assert json.loads((OUT/'gates.json').read_text())['collection_provenance']==provenance
    pp=OUT/(name+'_provenance.json')
    if pp.exists():assert json.loads(pp.read_text())==provenance
    else:write(pp,provenance)
    rows=[r for r in rows if not (out/(r['utterance_id']+'.json')).exists()];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as ex:
        fs=[ex.submit(process,(a.part,row)) for row in rows]
        for i,f in enumerate(as_completed(fs),1):
            r=f.result();tmp=out/(r['utterance_id']+'.tmp');tmp.write_text(json.dumps(r,separators=(',',':'))+'\n');tmp.replace(tmp.with_suffix('.json'))
            if i%50==0:print(json.dumps(dict(part=a.part,done=i,new_total=len(rows),seconds=time.time()-start)),flush=True)
    present={p.stem for p in out.glob('*.json') if p.name!='complete.json'}
    complete=dict(part=a.part,utterances=len(present),expected=len(expected),complete=present==expected,seconds=time.time()-start)
    write(out/'complete.json',complete);print(json.dumps(complete),flush=True)
if __name__=='__main__':main()
