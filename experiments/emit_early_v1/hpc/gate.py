"""Display gate for zero-fill previews. A gradient-boosted classifier scores each preview token; display stops at the first token under a threshold.
The gate is fitted and its thresholds are chosen on dev utterances outside the fixed 400-utterance evaluation subset. Usage: gate.py TAG [CHECKPOINT [KEY]]"""
import base64,gzip,json,os,random,sys
import numpy as np,torch,sentencepiece as spm
from sklearn.ensemble import HistGradientBoostingClassifier
import evaluate as E
from model_def import ASR,load

TRAIN=int(os.environ.get('GATE_TRAIN','400'));SELECT=int(os.environ.get('GATE_SELECT','150'));TARGETS=[1.0,2.0,5.0];SEED=6137
POINTS={16:[7,11,15,19,23,27,31,35],8:[7,11,15,19]}
# (name, preview points, feature level, display policies); hold=True keeps text on screen until a pass contradicts it
STUDIES={16:[('k7',[7],0,[False,True]),('k7+free',[7],1,[True]),('k7_23+free',[7,23],1,[True]),('k7_15_23_31',[7,15,23,31],0,[True]),('k7_15_23_31+free',[7,15,23,31],1,[True]),('every40+free',POINTS[16],1,[True])],
    8:[('k7',[7],0,[False,True]),('k7_15+free',[7,15],1,[True]),('every40+free',POINTS[8],1,[True])]}


def features(rec,piece,ks,level=0):
    """One row per preview token: probability, blank probability, margin, encoder frame, index, preview length, running minimum probability, word start, preview point.
    level 1 adds entropy, blank probabilities after the token, tokens left, piece length and frames since the last committed token.
    level 2 adds agreement with a last-frame-repeat preview of the same point (one more encoder pass per chunk)."""
    X=[];y=[];where=[];done=0;gap=8;final=[t for ch in rec['chunks'] for t in ch['ids']]
    for c,ch in enumerate(rec['chunks']):
        for k in ks:
            p=ch['previews'][str(k)];low=1.0;other=ch['previews'].get(f'{k}r');same=True
            for n,t in enumerate(p['ids']):
                low=min(low,p['p'][n]);row=[p['p'][n],p['pb'][n],p['mg'][n],p['fr'][n],n,len(p['ids']),low,float(piece[t].startswith('▁')),k]
                if level>=1:after=p['bl'][p['fr'][n]+1:] or [1.0];row+=[p['en'][n],sum(after)/len(after),min(after),len(p['ids'])-n-1,len(piece[t]),gap]
                if level>=2:ok=n<len(other['ids']) and other['ids'][n]==t;same=same and ok;row+=[float(ok),float(same),other['p'][n] if ok else 0.0,len(other['ids'])-len(p['ids'])]
                X.append(row);y.append(int(done+n<len(final) and final[done+n]==t));where.append((c,k))
        done+=len(ch['ids']);gap=7-ch['fr'][-1] if ch['ids'] else min(gap+8,64)
    return X,y,where


def keeps(rec,score,where,thr,ks):
    keep={k:[0]*len(rec['chunks']) for k in ks};dead=set()
    for s,w in zip(score,where):
        if w in dead or w[1] not in keep:continue
        if s>=thr:keep[w[1]][w[0]]+=1
        else:dead.add(w)
    return keep


def run(chunk):
    E.H=2*chunk;E.W=E.H+7;E.KS=POINTS[chunk];E.REPEAT=[];tag=sys.argv[1]
    model=load(sys.argv[2] if len(sys.argv)>2 else None,chunk,sys.argv[3] if len(sys.argv)>3 else 'model').to(E.dev).eval();sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'))
    align=json.loads(gzip.decompress(base64.b64decode((ASR/'code/align_eval.b64').read_text())));piece=[sp.id_to_piece(i) for i in range(500)]
    words=lambda ids:''.join(piece[i] for i in ids).replace('▁',' ').upper().split();rows={};sets=dict(train=[],select=[],eval=list(align));rng=random.Random(SEED)
    for s in ['dev-clean','dev-other']:
        fb=np.memmap(ASR/f'data/fbank_{s}.f16',np.float16,'r').reshape(-1,80);index=json.load(open(ASR/f'data/index_{s}.json'))
        for r in index:rows[r['utterance_id']]=(r,fb,s)
        rest=sorted(r['utterance_id'] for r in index if r['utterance_id'] not in align);rng.shuffle(rest);sets['train']+=rest[:TRAIN];sets['select']+=rest[TRAIN:TRAIN+SELECT]

    def record(uid):
        r,fb,s=rows[uid];f=torch.from_numpy(np.array(fb[r['feature_offset']:r['feature_offset']+r['feature_frames']],np.float32)).to(E.dev)
        return dict(utterance_id=uid,split=s,audio_seconds=r['num_samples']/16000,chunks=E.replay(model,f))

    data={name:[record(u) for u in ids] for name,ids in sets.items()};summary={}

    def measure(name,feats,score,thr,ks,split=None,hold=False):
        delay=[];rew=0;fin=0
        for r,f,s in zip(data[name],feats[name],score[name]):
            if split and r['split']!=split:continue
            ref=align.get(r['utterance_id'])
            if ref is None:own=words([t for ch in r['chunks'] for t in ch['ids']]);ref=dict(words=own,end_ms=[0.0]*len(own))  # selection needs rewrites only
            d,_,_,fw,rr,_=E.simulate(r,ref,words,ks,None,None if thr is None else keeps(r,s,f[2],thr,ks),hold);delay+=d;rew+=rr;fin+=fw
        return np.array(delay),100*rew/fin

    for name,ks,level,holds in [('none',[],0,[False])]+STUDIES[chunk]:
        feats={n:[features(r,piece,ks,level) for r in recs] for n,recs in data.items()};score=None
        if ks:
            X=np.array([x for f in feats['train'] for x in f[0]]);y=np.array([v for f in feats['train'] for v in f[1]])
            clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,random_state=SEED).fit(X,y)
            score={n:[clf.predict_proba(np.array(f[0]))[:,1] if f[0] else np.zeros(0) for f in feats[n]] for n in ('select','eval')}
            grid=sorted(set(np.quantile(np.concatenate(score['select']),np.linspace(0,1,81)).round(5).tolist()))+[1.01]  # 1.01 shows nothing
        for hold in holds:
            policies=[('raw',None,None)]
            if ks:
                curve=[(t,measure('select',feats,score,t,ks,None,hold)[1]) for t in grid]
                for target in TARGETS:
                    best=None
                    for t,r in reversed(curve):
                        if r>target:break
                        best=t
                    policies.append((f'R{target:g}',best,score))
                policies.append(('oracle',0.5,{n:[np.array(f[1]) for f in feats[n]] for n in ('select','eval')}))  # stops at the first token that the final pass changes
            res={};label=name+('/hold' if hold else '')
            for policy,thr,sc in policies:
                res[policy]=dict(threshold=thr)
                for s in ['dev-clean','dev-other']:
                    d,r=measure('eval',feats,sc or feats,thr,ks,s,hold);res[policy][s]=dict(delay_mean=float(d.mean()),delay_p50=float(np.percentile(d,50)),delay_p90=float(np.percentile(d,90)),rewrites_per_100=r)
            summary[label]=res
            print('GATE',tag,f'chunk{chunk*20}ms',label,'|',' | '.join('%s %.0f/%.1f %.0f/%.1f'%(k,v['dev-clean']['delay_mean'],v['dev-clean']['rewrites_per_100'],v['dev-other']['delay_mean'],v['dev-other']['rewrites_per_100']) for k,v in res.items()),flush=True)
    out=ASR/'ckpt'/tag;out.mkdir(parents=True,exist_ok=True);json.dump(summary,open(out/f'gate_c{chunk}.json','w'),indent=1)


if __name__=='__main__':
    for chunk in (16,8):run(chunk)
