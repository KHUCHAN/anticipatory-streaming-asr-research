"""Display gate for zero-fill previews (eight per 320ms chunk, hold display), with and without an external LM. Usage: gate.py TAG [CHECKPOINT [KEY]]
A gradient-boosted classifier scores each preview token; display stops at the first token under a threshold. The gate is fitted and all thresholds are chosen on dev utterances outside the fixed
400-utterance evaluation subset. Studies: `free` (acoustic features only), `lm` (plus LM features of each token), `lm+wc` (plus LM word completion on the display, joint-probability threshold tau)."""
import base64,gzip,json,os,random,sys
import numpy as np,torch,sentencepiece as spm
from sklearn.ensemble import HistGradientBoostingClassifier
import evaluate as E
from model_def import ASR,load

TRAIN=int(os.environ.get('GATE_TRAIN','400'));SELECT=int(os.environ.get('GATE_SELECT','150'));TARGETS=[1.0,2.0];SEED=6137;POINTS=[7,11,15,19,23,27,31,35];TAUS=[0.95,0.9,0.8,0.7,0.5]
STUDIES=[('every40+free',1,None),('every40+lm',3,None),('every40+lm+wc',3,TAUS)]  # (name, feature level, completion thresholds); level 3 needs the external LM


def features(rec,piece,ks,level):
    """One row per preview token: probability, blank probability, margin, encoder frame, index, preview length, running minimum probability, word start, preview point,
    entropy, blank probabilities after the token, tokens left, piece length, frames since the last committed token.
    level 3 adds the LM log-probability of the token, the LM's top probability, whether the token is the LM's first choice and whether the LM changed it."""
    X=[];y=[];where=[];done=0;gap=8;final=[t for ch in rec['chunks'] for t in ch['ids']]
    for c,ch in enumerate(rec['chunks']):
        for k in ks:
            p=ch['previews'][str(k)];low=1.0
            for n,t in enumerate(p['ids']):
                low=min(low,p['p'][n]);after=p['bl'][p['fr'][n]+1:] or [1.0]
                row=[p['p'][n],p['pb'][n],p['mg'][n],p['fr'][n],n,len(p['ids']),low,float(piece[t].startswith('▁')),k,p['en'][n],sum(after)/len(after),min(after),len(p['ids'])-n-1,len(piece[t]),gap]
                if level>=3:row+=[p['lm'][n],p['lt'][n],p['l1'][n],p['sw'][n]]
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


def run():
    E.setup();E.KS=POINTS;tag=sys.argv[1];ks=POINTS
    model=load(sys.argv[2] if len(sys.argv)>2 else None,16,sys.argv[3] if len(sys.argv)>3 else 'model').to(E.dev).eval();sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'))
    align=json.loads(gzip.decompress(base64.b64decode((ASR/'code/align_eval.b64').read_text())));piece=[sp.id_to_piece(i) for i in range(500)]
    words=lambda ids:''.join(piece[i] for i in ids).replace('▁',' ').upper().split();rows={};sets=dict(train=[],select=[],eval=list(align));rng=random.Random(SEED)
    for s in ['dev-clean','dev-other']:
        fb=np.memmap(ASR/f'data/fbank_{s}.f16',np.float16,'r').reshape(-1,80);index=json.load(open(ASR/f'data/index_{s}.json'))
        for r in index:rows[r['utterance_id']]=(r,fb,s)
        rest=sorted(r['utterance_id'] for r in index if r['utterance_id'] not in align);rng.shuffle(rest);sets['train']+=rest[:TRAIN];sets['select']+=rest[TRAIN:TRAIN+SELECT]

    def record(uid):
        r,fb,s=rows[uid];f=torch.from_numpy(np.array(fb[r['feature_offset']:r['feature_offset']+r['feature_frames']],np.float32)).to(E.dev)
        return dict(utterance_id=uid,split=s,audio_seconds=r['num_samples']/16000,chunks=E.replay(model,f))

    data={name:[record(u) for u in ids] for name,ids in sets.items()};summary={};mode='lam=%g/%g'%(E.LAM,E.LAMN)

    def measure(name,feats,score,thr,split=None,tau=None,shown=ks):
        """Delays and rewrites per 100 words. Selection utterances have no alignment: their own final words stand in as the reference and the delays are display times."""
        delay=[];rew=0;fin=0
        for r,f,s in zip(data[name],feats[name],score[name]):
            if split and r['split']!=split:continue
            ref=align.get(r['utterance_id'])
            if ref is None:own=words([t for ch in r['chunks'] for t in ch['ids']]);ref=dict(words=own,end_ms=[0.0]*len(own))
            d,_,_,fw,rr,_=E.simulate(r,ref,words,shown,None,None if thr is None else keeps(r,s,f[2],thr,ks),True,tau,piece);delay+=d;rew+=rr;fin+=fw
        return np.array(delay),100*rew/fin

    def report(label,policies,feats):
        res={}
        for policy,thr,sc,tau,shown in policies:
            res[policy]=dict(threshold=thr,tau=tau)
            for s in ['dev-clean','dev-other']:
                d,r=measure('eval',feats,sc or feats,thr,s,tau,shown);res[policy][s]=dict(delay_mean=float(d.mean()),delay_p50=float(np.percentile(d,50)),delay_p90=float(np.percentile(d,90)),rewrites_per_100=r)
        summary[label]=res
        print('GATE',tag,mode,label,'|',' | '.join('%s %.0f/%.1f %.0f/%.1f'%(k,v['dev-clean']['delay_mean'],v['dev-clean']['rewrites_per_100'],v['dev-other']['delay_mean'],v['dev-other']['rewrites_per_100']) for k,v in res.items()),flush=True)

    for name,level,taus in STUDIES:
        if level>=3 and E.LMX is None:continue
        feats={n:[features(r,piece,ks,level) for r in recs] for n,recs in data.items()}
        X=np.array([x for f in feats['train'] for x in f[0]]);y=np.array([v for f in feats['train'] for v in f[1]])
        clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,random_state=SEED).fit(X,y)
        score={n:[clf.predict_proba(np.array(f[0]))[:,1] if f[0] else np.zeros(0) for f in feats[n]] for n in ('select','eval')}
        grid=sorted(set(np.quantile(np.concatenate(score['select']),np.linspace(0,1,41 if taus else 81)).round(5).tolist()))+[1.01]  # 1.01 shows no preview token
        policies=[('none',None,None,None,[]),('raw',None,None,None,ks)] if name==STUDIES[0][0] else [];curves={}
        for tau in [None]+(taus or []):
            curves[tau]=[]
            for t in grid:d,r=measure('select',feats,score,t,None,tau);curves[tau].append((t,float(d.mean()),r))
        for target in TARGETS:
            best=None  # per tau: the smallest threshold such that it and all higher thresholds meet the target; across taus: the earliest display on the selection utterances
            for tau,curve in curves.items():
                pick=None
                for t,d,r in reversed(curve):
                    if r>target:break
                    pick=(d,t,tau)
                if pick and (best is None or pick[0]<best[0]):best=pick
            policies.append((f'R{target:g}'+(f'@{best[2]:g}' if best[2] is not None else ''),best[1],score,best[2],ks))
        policies.append(('oracle',0.5,{n:[np.array(f[1]) for f in feats[n]] for n in ('select','eval')},None,ks))  # shows exactly the preview tokens that equal the final output
        report(name+'/hold',policies,feats)
    out=ASR/'ckpt'/tag;out.mkdir(parents=True,exist_ok=True);json.dump(summary,open(out/('gate_l%g_%g.json'%(E.LAM,E.LAMN)),'w'),indent=1)


if __name__=='__main__':run()
