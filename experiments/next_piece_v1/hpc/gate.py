"""Display gate for zero-fill previews (eight per 320ms chunk, hold display), with and without the next-piece head. Usage: gate.py TAG [CHECKPOINT [KEY]]
A gradient-boosted classifier scores each preview token; display stops at the first token under a threshold. The gate is fitted and all thresholds are chosen on dev utterances outside the fixed
400-utterance evaluation subset. Studies: `free` (transducer preview tokens, acoustic features only), `hq` (plus the head's probability of each token),
`ant` (the pieces the head anticipates after each preview pass are shown as further preview tokens). Also prints how often the next piece is named correctly before the transducer emits it."""
import base64,gzip,json,os,random,sys
import numpy as np,torch,sentencepiece as spm
from sklearn.ensemble import HistGradientBoostingClassifier
import evaluate as E
from model_def import ASR,load

TRAIN=int(os.environ.get('GATE_TRAIN','400'));SELECT=int(os.environ.get('GATE_SELECT','150'));TARGETS=[1.0,2.0];SEED=6137;POINTS=[7,11,15,19,23,27,31,35]
STUDIES=[('every40+free',1,False),('every40+hq',2,False),('every40+ant',3,True)]  # (name, feature level, anticipated pieces shown); levels 2 and 3 need the head


def extend(rec):
    """Copy of a record in which every preview also lists the pieces the head anticipates (after the transducer's preview tokens) as further preview tokens; ai numbers them from 1."""
    chunks=[]
    for ch in rec['chunks']:
        pv={}
        for k,p in ch['previews'].items():
            a=[r for r in p['ant'] if r[0]!=0];n=len(p['ids'])
            pv[k]=dict(p,ids=p['ids']+[r[0] for r in a],p=p['p']+[r[1] for r in a],pb=p['pb']+[r[4] for r in a],mg=p['mg']+[r[2] for r in a],en=p['en']+[r[3] for r in a],
                fr=p['fr']+[p['ja']]*len(a),hq=p['hq']+[r[1] for r in a],ai=[0]*n+list(range(1,len(a)+1)))
        chunks.append(dict(ch,previews=pv))
    return dict(rec,chunks=chunks)


def features(rec,piece,ks,level):
    """One row per preview token: probability, blank probability, margin, encoder frame, index, preview length, running minimum probability, word start, preview point,
    entropy, blank probabilities after the token, tokens left, piece length, frames since the last committed token.
    level 2 adds the head's probability of the token; level 3 also the token's rank among the anticipated pieces (0 for a transducer token)."""
    X=[];y=[];where=[];done=0;gap=8;final=[t for ch in rec['chunks'] for t in ch['ids']]
    for c,ch in enumerate(rec['chunks']):
        for k in ks:
            p=ch['previews'][str(k)];low=1.0
            for n,t in enumerate(p['ids']):
                low=min(low,p['p'][n]);after=p['bl'][p['fr'][n]+1:] or [1.0]
                row=[p['p'][n],p['pb'][n],p['mg'][n],p['fr'][n],n,len(p['ids']),low,float(piece[t].startswith('▁')),k,p['en'][n],sum(after)/len(after),min(after),len(p['ids'])-n-1,len(piece[t]),gap]
                if level>=2:row+=[p['hq'][n]]
                if level>=3:row+=[p['ai'][n]]
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


def next_piece(records,piece):
    """Share of positions at which the next piece of the final output is named correctly, by the head / by the transducer's best non-blank piece / by the external LM.
    native: after each chunk's native pass (last frame, all audio real), by how many 40ms frames later the transducer emits that piece and by whether it starts a word.
    preview: after each preview pass whose own tokens are all correct, by preview point."""
    tab={}

    def add(group,key,row,truth):
        a=tab.setdefault(group,{}).setdefault(key,[0,0,0,0]);a[0]+=1;a[1]+=row[0]==truth;a[2]+=row[6]==truth;a[3]+=row[7]==truth

    for rec in records:
        final=[(t,8*c+j) for c,ch in enumerate(rec['chunks']) for t,j in zip(ch['ids'],ch['fr'])];done=0
        for c,ch in enumerate(rec['chunks']):
            for k,p in ch['previews'].items():
                n=done+len(p['ids'])
                if n<len(final) and p['ids']==[t for t,_ in final[done:n]]:add('preview','k%02d'%int(k),p['ant'][0],final[n][0])
            done+=len(ch['ids'])
            if done<len(final):
                truth,pos=final[done];lead=pos-(8*c+7);row=ch['nx'][0]
                add('native','all',row,truth);add('native','lead %s'%('1-2' if lead<=2 else '3-4' if lead<=4 else '5-8' if lead<=8 else '9+'),row,truth);add('native','word start' if piece[truth].startswith('▁') else 'inside word',row,truth)
    return {g:{k:dict(n=v[0],head=v[1]/v[0],transducer=v[2]/v[0],lm=v[3]/v[0]) for k,v in sorted(d.items())} for g,d in tab.items()}


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

    plain={name:[record(u) for u in ids] for name,ids in sets.items()};summary={};view={}

    def measure(name,feats,score,thr,split=None,shown=ks):
        """Delays and rewrites per 100 words. Selection utterances have no alignment: their own final words stand in as the reference and the delays are display times."""
        delay=[];rew=0;fin=0
        for r,f,s in zip(view[name],feats[name],score[name]):
            if split and r['split']!=split:continue
            ref=align.get(r['utterance_id'])
            if ref is None:own=words([t for ch in r['chunks'] for t in ch['ids']]);ref=dict(words=own,end_ms=[0.0]*len(own))
            d,_,_,fw,rr,_=E.simulate(r,ref,words,shown,None,None if thr is None else keeps(r,s,f[2],thr,ks),True);delay+=d;rew+=rr;fin+=fw
        return np.array(delay),100*rew/fin

    def report(label,policies,feats):
        res={}
        for policy,thr,sc,shown in policies:
            res[policy]=dict(threshold=thr)
            for s in ['dev-clean','dev-other']:
                d,r=measure('eval',feats,sc or feats,thr,s,shown);res[policy][s]=dict(delay_mean=float(d.mean()),delay_p50=float(np.percentile(d,50)),delay_p90=float(np.percentile(d,90)),rewrites_per_100=r)
        summary[label]=res
        print('GATE',tag,label,'|',' | '.join('%s %.0f/%.1f %.0f/%.1f'%(k,v['dev-clean']['delay_mean'],v['dev-clean']['rewrites_per_100'],v['dev-other']['delay_mean'],v['dev-other']['rewrites_per_100']) for k,v in res.items()),flush=True)

    if model.nx:
        summary['next_piece']=next_piece([r for recs in plain.values() for r in recs],piece)
        for g,d in summary['next_piece'].items():print('NEXT',tag,g,'(head/transducer/lm, n) |',' | '.join('%s %.3f/%.3f/%.3f %d'%(k,v['head'],v['transducer'],v['lm'],v['n']) for k,v in d.items()),flush=True)
    for name,level,ant in STUDIES:
        if level>=2 and not model.nx:continue
        view.clear();view.update({n:[extend(r) for r in recs] for n,recs in plain.items()} if ant else plain)
        feats={n:[features(r,piece,ks,level) for r in recs] for n,recs in view.items()}
        X=np.array([x for f in feats['train'] for x in f[0]]);y=np.array([v for f in feats['train'] for v in f[1]])
        clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,random_state=SEED).fit(X,y)
        score={n:[clf.predict_proba(np.array(f[0]))[:,1] if f[0] else np.zeros(0) for f in feats[n]] for n in ('select','eval')}
        grid=sorted(set(np.quantile(np.concatenate(score['select']),np.linspace(0,1,81)).round(5).tolist()))+[1.01]  # 1.01 shows no preview token
        policies=[('none',None,None,[]),('raw',None,None,ks)] if name==STUDIES[0][0] else [('raw',None,None,ks)] if ant else []
        curve=[(t,*measure('select',feats,score,t)) for t in grid];curve=[(t,float(d.mean()),r) for t,d,r in curve]
        for target in TARGETS:
            pick=None  # the smallest threshold such that it and all higher thresholds meet the rewrite target on the selection utterances
            for t,d,r in reversed(curve):
                if r>target:break
                pick=t
            policies.append((f'R{target:g}',pick,score,ks))
        policies.append(('oracle',0.5,{n:[np.array(f[1]) for f in feats[n]] for n in ('select','eval')},ks))  # shows exactly the preview tokens that equal the final output
        report(name+'/hold',policies,feats)
    out=ASR/'ckpt'/tag;out.mkdir(parents=True,exist_ok=True);json.dump(summary,open(out/'gate.json','w'),indent=1)


if __name__=='__main__':run()
