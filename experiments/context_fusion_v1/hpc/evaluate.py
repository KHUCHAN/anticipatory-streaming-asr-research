"""Streaming replay (320ms chunks) with zero-fill previews, an optional external LM and an optional long-context decoder branch.
Usage: evaluate.py TAG [CHECKPOINT [KEY]] (KEY: model or ema). Prints WER, emission lag and display delay on the fixed 400-utterance dev subset.
Environment: LMX (path of the external LM), LAM / LAMN (LM weight when a preview / native pass picks a token), COMP (LM continuation tokens stored per display point)."""
import base64,gzip,json,os,sys
import numpy as np,torch,sentencepiece as spm
from model_def import ASR,load,load_lm

PASS_MS=5.0;H=32;W=39;KS=[7];SHIFT=10.0;OFFSET=17.5;dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LMX=None;LAM=float(os.environ.get('LAM','0'));LAMN=float(os.environ.get('LAMN','0'));COMP=int(os.environ.get('COMP','0'))
BD=lambda i:2 if i//5 in (2,3,4) else 1


def setup():
    """Loads the external LM named by $LMX, if the file exists."""
    global LMX
    path=os.environ.get('LMX','')
    if path and os.path.exists(path):LMX=load_lm(path).to(dev).eval()


def edit_pairs(a,b):
    n=len(a);m=len(b);d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):d[i][0]=i
    for j in range(m+1):d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))
    i=n;j=m;pairs=[]
    while i>0 and j>0:
        if d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]==b[j-1]:pairs.append((j-1,i-1))
            i-=1;j-=1
        elif d[i][j]==d[i-1][j]+1:i-=1
        else:j-=1
    return d[n][m],pairs[::-1]


class Dec:
    """Decoder-side search state: the 2-token context, the model's own long-context LSTM state (if it has the branch) and the external LM state (if one is loaded).
    A state is a dict: c (context), own, ext (LSTM states), out (projected decoder output), lp (external LM log-probabilities of the next token)."""
    def __init__(self,model,lm):self.m=model;self.lm=lm;self.cache={}

    def start(self):return self.push(dict(c=(0,0),own=None,ext=None),0,True)

    def push(self,st,t,first=False):
        c=st['c'] if first else (st['c'][1],t);tok=torch.tensor([[t]],device=dev);own=st['own'];ext=st['ext'];lp=None
        if c not in self.cache:self.cache[c]=self.m.decoder(torch.tensor([c],device=dev),need_pad=False).squeeze(1)
        d=self.cache[c]
        if self.m.ctx:h,own=self.m.ctx_lm.hidden(tok,own);d=d+self.m.ctx_proj(h[:,0])
        if self.lm is not None:o,ext=self.lm(tok,ext);lp=o[0,0].log_softmax(-1)
        return dict(c=c,own=own,ext=ext,out=self.m.joiner.decoder_proj(d),lp=lp)


def complete(lm,sts,m):
    """Greedy LM continuation of m tokens after each state: [[token, probability], ...] per state."""
    h=torch.cat([s['ext'][0] for s in sts],1);c=torch.cat([s['ext'][1] for s in sts],1);lp=torch.stack([s['lp'] for s in sts]);out=[[] for _ in sts]
    for _ in range(m):
        lp=lp.clone();lp[:,:3]=-1e9;p,t=lp.max(-1);p=p.exp().tolist();ts=t.tolist()
        for i in range(len(sts)):out[i].append([ts[i],round(p[i],4)])
        o,(h,c)=lm(t[:,None],(h,c));lp=o[:,0].log_softmax(-1)
    return out


def greedy(model,D,enc,st,lam):
    """One pass over the encoder frames of a chunk. The transducer decides when to emit; with lam > 0 the token is the best non-blank one under log p + lam * LM log p."""
    ids=[];fr=[];rows=[];bl=[];sts=[st]
    for j in range(enc.shape[0]):
        lp=model.joiner.output_linear(torch.tanh(enc[j:j+1]+st['out'])).log_softmax(-1)[0];bl.append(round(float(lp[0].exp()),4));t=int(lp.argmax())
        if t in (0,2):continue
        first=t
        if lam and st['lp'] is not None:s=lp+lam*st['lp'];s[:3]=-1e9;t=int(s.argmax())
        top=lp.topk(2).values.exp();row=[round(float(lp[t].exp()),4),bl[-1],round(float(top[0]-top[1]),4),round(float(-(lp.exp()*lp).sum()),4)]
        if st['lp'] is not None:lm=st['lp'].clone();lm[:3]=-1e9;best=int(lm.argmax());row+=[round(float(lm[t]),3),round(float(lm[best].exp()),4),float(best==t),float(first!=t)]
        ids.append(t);fr.append(j);rows.append(row);st=D.push(st,t);sts.append(st)
    info=dict(ids=ids,fr=fr,p=[r[0] for r in rows],pb=[r[1] for r in rows],mg=[r[2] for r in rows],en=[r[3] for r in rows],bl=bl)
    if D.lm is not None:
        info.update(lm=[r[4] for r in rows],lt=[r[5] for r in rows],l1=[r[6] for r in rows],sw=[r[7] for r in rows])
        if COMP:info['comp']=complete(D.lm,sts,COMP)
    return ids,st,info


@torch.no_grad()
def replay(model,f):
    """Native pass plus zero-fill previews at KS for every chunk; previews share the native state and never change it.
    With an external LM every token carries its LM log-probability (lm), the LM's top probability (lt), whether it is the LM's first choice (l1) and whether the LM changed it (sw);
    comp holds the LM's greedy continuation after every preview prefix and after the native pass."""
    count=max(0,(len(f)-(W+1))//H+1);B=len(KS)+1;states=[s.repeat_interleave(B,BD(i)) for i,s in enumerate(model.encoder.get_init_state(dev))]
    lens=torch.full((B,),W,device=dev);first=torch.zeros(B,dtype=torch.long,device=dev);D=Dec(model,LMX);st=D.start();chunks=[]
    for c in range(count):
        x=f[c*H:c*H+W][None].repeat(B,1,1)
        for b,k in enumerate(KS):x[b+1,k:]=0.0
        out,_,new=model.encoder.streaming_forward(x,lens,states);enc=model.joiner.encoder_proj(out)
        states=[s.index_select(BD(i),first) for i,s in enumerate(new)]
        previews={str(k):greedy(model,D,enc[b+1],st,LAM)[2] for b,k in enumerate(KS)}
        ids,st,info=greedy(model,D,enc[0],st,LAMN);chunks.append(dict(ids=ids,fr=info['fr'],comp=info['comp'][-1] if 'comp' in info else None,previews=previews))
    return chunks


def extra(comp,tau,piece):
    """Whole words from an LM continuation [[token, probability], ...] while the joint probability stays at or above tau. A word counts as whole when the next predicted piece starts a new word."""
    if tau is None or not comp:return []
    keep=0;prob=1.0
    for m in range(len(comp)-1):
        prob*=comp[m][1]
        if prob<tau:break
        if piece[comp[m+1][0]].startswith('▁'):keep=m+1
    return [t for t,_ in comp[:keep]]


def simulate(rec,ref,words,ks,frame_cut=None,keep=None,hold=False,tau=None,piece=None):
    """keep: optional {k: preview tokens shown per chunk} from a display gate; a gated preview that is a prefix of what is already shown does not shrink the display.
    frame_cut: show only preview tokens from encoder frames below it.
    hold: text on screen is replaced only when a pass contradicts it; a pass whose output is a prefix of the screen leaves it. The last pass always shows the final result.
    tau: also show the LM continuation (whole words) of what a pass displays while its joint probability is at least tau."""
    audio=1000*rec['audio_seconds'];free=0.0;committed=[];events=[];screen=[];last=len(rec['chunks'])-1
    for c,ch in enumerate(rec['chunks']):
        shown=[];more=[]
        for k in ks:
            p=ch['previews'][str(k)];n=len(p['ids']) if frame_cut is None else sum(j<frame_cut for j in p['fr'])
            if keep is not None:n=keep[k][c]
            ids=p['ids'][:n]
            if keep is None or shown[:len(ids)]!=ids:shown=ids;more=extra(p['comp'][n],tau,piece) if tau is not None else []
            cand=committed+shown+more
            if not hold or screen[:len(cand)]!=cand:screen=cand
            free=max(min((c*H+k-1)*SHIFT+OFFSET,audio),free)+PASS_MS;events.append((free,words(screen)))
        free=max(min((c*H+W-1)*SHIFT+OFFSET,audio),free)+PASS_MS;committed=committed+ch['ids']
        cand=committed if c==last or tau is None else committed+extra(ch['comp'],tau,piece)
        if not hold or c==last or screen[:len(cand)]!=cand:screen=cand
        events.append((free,words(screen)))
    final=words(committed);errors,pairs=edit_pairs(ref['words'],final);stable=[None]*len(final)
    for i in range(len(final)):
        for t,shown in reversed(events):
            if i<len(shown) and shown[i]==final[i]:stable[i]=t
            else:break
    rewrites=0
    for (_,a),(_,b) in zip(events[:-1],events[1:]):rewrites+=sum(1 for i,w in enumerate(a) if i>=len(b) or not b[i].startswith(w))
    return [stable[i]-ref['end_ms'][j] for i,j in pairs],errors,len(ref['words']),len(final),rewrites,pairs


def run():
    setup();tag=sys.argv[1];model=load(sys.argv[2] if len(sys.argv)>2 else None,16,sys.argv[3] if len(sys.argv)>3 else 'model').to(dev).eval();sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'))
    align=json.loads(gzip.decompress(base64.b64decode((ASR/'code/align_eval.b64').read_text())));piece=[sp.id_to_piece(i) for i in range(500)]
    words=lambda ids:''.join(piece[i] for i in ids).replace('▁',' ').upper().split();rows={}
    for s in ['dev-clean','dev-other']:
        fb=np.memmap(ASR/f'data/fbank_{s}.f16',np.float16,'r').reshape(-1,80)
        for r in json.load(open(ASR/f'data/index_{s}.json')):rows[r['utterance_id']]=(r,fb)
    records=[]
    for uid,ref in align.items():
        r,fb=rows[uid];f=torch.from_numpy(np.array(fb[r['feature_offset']:r['feature_offset']+r['feature_frames']],np.float32)).to(dev)
        records.append(dict(utterance_id=uid,split=ref['split'],audio_seconds=r['num_samples']/16000,chunks=replay(model,f)))
    out=ASR/'ckpt'/tag;out.mkdir(parents=True,exist_ok=True);summary={}
    for s in ['dev-clean','dev-other']:
        own=[r for r in records if r['split']==s];res={}
        for name,ks in [('none',[]),('k7',[7])]:
            delay=[];err=0;refw=0;finw=0;rew=0
            for rec in own:
                d,e,rw,fw,rr,_=simulate(rec,align[rec['utterance_id']],words,ks);delay+=d;err+=e;refw+=rw;finw+=fw;rew+=rr
            d=np.array(delay);res[name]=dict(delay_mean=float(d.mean()),delay_p50=float(np.percentile(d,50)),delay_p90=float(np.percentile(d,90)),rewrites_per_100=100*rew/finw,wer=100*err/refw)
        lag=[]
        for rec in own:
            ref=align[rec['utterance_id']];ids=[];slot=[]
            for c,ch in enumerate(rec['chunks']):ids+=ch['ids'];slot+=[(c*H+4*j+3)*SHIFT+OFFSET for j in ch['fr']]
            last={};w=-1
            for i,t in enumerate(ids):
                if piece[t].startswith('▁'):w+=1
                last[max(w,0)]=i
            _,pairs=edit_pairs(ref['words'],words(ids));lag+=[slot[last[hi]]-ref['end_ms'][ri] for hi,ri in pairs if hi in last]
        lag=np.array(lag);res['lag']=dict(mean=float(lag.mean()),p10=float(np.percentile(lag,10)),p50=float(np.percentile(lag,50)),p90=float(np.percentile(lag,90)))
        summary[s]=res
    json.dump(summary,open(out/'eval_c16.json','w'),indent=1);a=summary['dev-clean'];b=summary['dev-other']
    line=lambda k:'%s %.0f/%.1f %.0f/%.1f'%(k,a[k]['delay_mean'],a[k]['rewrites_per_100'],b[k]['delay_mean'],b[k]['rewrites_per_100'])
    print('EVAL',tag,'lamn=%g'%LAMN,'WER %.2f/%.2f lag %.0f/%.0f |'%(a['none']['wer'],b['none']['wer'],a['lag']['mean'],b['lag']['mean']),' | '.join(line(k) for k in a if k!='lag'),flush=True)


if __name__=='__main__':run()
