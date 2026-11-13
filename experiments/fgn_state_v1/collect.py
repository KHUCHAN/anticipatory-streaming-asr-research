"""Collect new causal forecasts, retaining unchanged native evaluation labels."""
import argparse,hashlib,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
import sentencepiece as spm
from data import ROOT,OUT
from model import StateFGN,FrozenRNNT
from runtime import Runtime,PersistenceRuntime

KINDS=['state','state38','ensemble','reset','deterministic','single_step','crps_only','persistence']
MODELS={};BRIDGE=None;MEAN=None;STD=None;PIECES=[];CFG=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    global BRIDGE,MEAN,STD,PIECES,CFG
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    CFG=json.loads((ROOT/'experiments/fgn_state_v1/config.json').read_text())
    for variant in CFG['variants']+['crps_only_s37']:
        ck=torch.load(OUT/variant/'best.pt',map_location='cpu',weights_only=False)
        model=StateFGN(ck['cfg'],variant);model.load_state_dict(ck['model']);model.eval();MODELS[variant]=model
    BRIDGE=FrozenRNNT(OUT/'rnnt_bridge.npz').eval()
    stats=json.loads((OUT/'cache/stats.json').read_text());MEAN=torch.tensor(stats['mean']);STD=torch.tensor(stats['std'])
    base=json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/base['model_dir']/'bpe.model'))
    PIECES=[sp.id_to_piece(i) for i in range(500)]


def make_runtime(kind):
    if kind=='persistence':return PersistenceRuntime(BRIDGE,CFG['inference_rollout_steps'],PIECES)
    variants={'state':['state_s37'],'state38':['state_s38'],'ensemble':['state_s37','state_s38'],'reset':['reset_s37'],
              'deterministic':['deterministic_s37'],'single_step':['single_step_s37'],'crps_only':['crps_only_s37']}[kind]
    members=1 if kind=='deterministic' else CFG['inference_members']
    return Runtime([MODELS[v] for v in variants],BRIDGE,MEAN,STD,members,CFG['inference_rollout_steps'],PIECES)


def process(task):
    part,row=task;uid=row['utterance_id']
    with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{uid}.npz") as v:
        enc=torch.from_numpy(v['enc'].astype(np.float32));cuts=v['cuts'].tolist();available=v['available'].tolist()
        logp=v['logp'].tolist();frames=v['frames'].tolist();duration=float(v['duration'])
    old=json.loads((ROOT/f'runs/zipformer_ours/events_{part}/{uid}.json').read_text())
    assert cuts==[e['native_count'] for e in old['events']['full']]
    native=old['native'];correct=set(old['correct_native_positions']);events={};processing={}
    order=KINDS.copy();rng=np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:8],16));rng.shuffle(order)
    for kind in order:
        runtime=make_runtime(kind);records=[];previous=0
        # Matched noise draws across seed-37 ablations, independent of worker order.
        generator=torch.Generator().manual_seed(int(hashlib.sha256((uid+'forecast').encode()).hexdigest()[:12],16))
        started=time.perf_counter()
        for chunk,(count,clock) in enumerate(zip(cuts,available)):
            last=[-1,0]
            if count==1:last=[0,native[0]]
            elif count>=2:last=native[count-2:count]
            gap=((chunk+1)*8-1-frames[count-1])*.04 if count else (chunk+1)*.32
            result=runtime.advance(enc[chunk*8:(chunk+1)*8],last,logp[count-1] if count else 0.,gap,count-previous,count,generator)
            result.update(available=clock,native_count=count,eligible=bool(clock<duration and result['g']>=3))
            result['label_keep']=int(count<len(native) and result['g']==native[count])
            result['label_correct_and_keep']=int(result['label_keep'] and count in correct)
            records.append(result);previous=count
        events[kind]=records;processing[kind]=time.perf_counter()-started
    # Preserve controls exactly, including their measured timings and gate inputs.
    old['events'].update(events);old['processing_seconds'].update(processing)
    return old


def rows_for(part,cfg):
    if part=='dev':paths=[ROOT/cfg['gate_protocol']/f'{n}.jsonl' for n in ['gate_fit','gate_calibration']]
    else:paths=[ROOT/f'data/processed/manifests/{s}.jsonl' for s in cfg['evaluation_splits']]
    return [json.loads(line) for path in paths for line in path.open()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',required=True,choices=['dev','test'])
    p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();cfg=json.loads((ROOT/'experiments/fgn_state_v1/config.json').read_text())
    if a.part=='test' and not a.smoke:assert (OUT/'gates.json').exists(),'Freeze dev thresholds before test collection'
    rows=rows_for(a.part,cfg);all_ids={r['utterance_id'] for r in rows}
    if a.limit:rows=rows[:a.limit]
    out=OUT/(('smoke_' if a.smoke else 'events_')+a.part);out.mkdir(exist_ok=True)
    provenance=dict(model_sha256={v:sha(OUT/v/'best.pt') for v in cfg['variants']+['crps_only_s37']},
                    frozen_bridge_sha256=sha(OUT/'rnnt_bridge.npz'),normalization_sha256=sha(OUT/'cache/stats.json'),
                    backbone_config_sha256=sha(ROOT/'configs/zipformer_baseline.json'),
                    source_sha256={n:sha(ROOT/'experiments/fgn_state_v1'/n) for n in ['model.py','runtime.py','collect.py','config.json','config_crps_only.json']})
    if a.part=='test' and not a.smoke:
        learned=json.loads((OUT/'gates.json').read_text())
        assert learned['collection_provenance']==provenance,'Dev/test models, runtime or normalization changed'
    pp=out.parent/(out.name+'_provenance.json')
    if pp.exists():assert json.loads(pp.read_text())==provenance,'Changed collection inputs; start a separate output'
    else:pp.write_text(json.dumps(provenance,indent=2)+'\n')
    rows=[r for r in rows if not (out/(r['utterance_id']+'.json')).exists()];start=time.time();done=0
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as ex:
        futures=[ex.submit(process,(a.part,row)) for row in rows]
        for f in as_completed(futures):
            result=f.result();path=out/(result['utterance_id']+'.json');tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(result,separators=(',',':'))+'\n');tmp.replace(path);done+=1
            if done%50==0:print(json.dumps(dict(part=a.part,done=done,total=len(rows),seconds=round(time.time()-start,1))),flush=True)
    present={p.stem for p in out.glob('*.json') if p.name!='complete.json'}
    result=dict(part=a.part,utterances=len(present),expected=len(all_ids),complete=present==all_ids,
                limited=bool(a.limit),smoke=a.smoke,seconds=time.time()-start,kinds=KINDS)
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
