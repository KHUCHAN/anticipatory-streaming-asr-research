"""Streaming replay on the fixed 400-utterance dev subset: WER, emission lag, display delay with and without zero-fill previews.
Usage: evaluate.py TAG [CHECKPOINT [KEY]] (KEY: model or ema). Writes $ASR/ckpt/TAG/eval.json (summary) and records.json (per-chunk tokens)."""
import base64,gzip,json,sys
import numpy as np,torch,sentencepiece as spm
from model_def import ASR,load

PASS_MS=5.0;H=32;W=39;KS=[7,15,23];REPEAT=[];SHIFT=10.0;OFFSET=17.5;dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BD=lambda i:2 if i//5 in (2,3,4) else 1


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


@torch.no_grad()
def replay(model,f):
    """Native pass plus zero-fill previews at KS (and last-frame-repeat previews at REPEAT, stored as '<k>r') for every chunk; previews share the native state and never change it."""
    count=max(0,(len(f)-(W+1))//H+1);B=len(KS)+len(REPEAT)+1;states=[s.repeat_interleave(B,BD(i)) for i,s in enumerate(model.encoder.get_init_state(dev))]
    lens=torch.full((B,),W,device=dev);first=torch.zeros(B,dtype=torch.long,device=dev);ctx=[0,0];cache={};chunks=[]

    def dec(c):
        if c not in cache:cache[c]=model.joiner.decoder_proj(model.decoder(torch.tensor([c],device=dev),need_pad=False).squeeze(1))
        return cache[c]

    def greedy(enc,c):
        ids=[];fr=[];pr=[];bl=[]
        for j in range(enc.shape[0]):
            lp=model.joiner.output_linear(torch.tanh(enc[j:j+1]+dec(c))).log_softmax(-1)[0];t=int(lp.argmax());bl.append(round(float(lp[0].exp()),4))
            if t in (0,2):continue
            top=lp.topk(2).values.exp();ids.append(t);fr.append(j);c=(c[1],t)
            pr.append((round(float(top[0]),4),bl[-1],round(float(top[0]-top[1]),4),round(float(-(lp.exp()*lp).sum()),4)))
        return ids,fr,c,dict(ids=ids,fr=fr,p=[v[0] for v in pr],pb=[v[1] for v in pr],mg=[v[2] for v in pr],en=[v[3] for v in pr],bl=bl)

    ctx=(0,0)
    for c in range(count):
        x=f[c*H:c*H+W][None].repeat(B,1,1)
        for b,k in enumerate(KS):x[b+1,k:]=0.0
        for b,k in enumerate(REPEAT):x[len(KS)+1+b,k:]=x[0,k-1]
        out,_,new=model.encoder.streaming_forward(x,lens,states);enc=model.joiner.encoder_proj(out)
        states=[s.index_select(BD(i),first) for i,s in enumerate(new)]
        previews={}
        for b,k in enumerate(KS):previews[str(k)]=greedy(enc[b+1],ctx)[3]
        for b,k in enumerate(REPEAT):previews[f'{k}r']=greedy(enc[len(KS)+1+b],ctx)[3]
        ids,fr,ctx,_=greedy(enc[0],ctx);chunks.append(dict(ids=ids,fr=fr,previews=previews))
    return chunks


def simulate(rec,ref,words,ks,frame_cut=None,keep=None,hold=False):
    """keep: optional {k: preview tokens shown per chunk} from a display gate; a gated preview that is a prefix of what is already shown does not shrink the display.
    frame_cut: show only preview tokens from encoder frames below it.
    hold: text on screen is replaced only when a pass contradicts it; a pass whose output is a prefix of the screen leaves it. The last pass always shows the final result."""
    audio=1000*rec['audio_seconds'];free=0.0;committed=[];events=[];screen=[];last=len(rec['chunks'])-1
    for c,ch in enumerate(rec['chunks']):
        shown=[]
        for k in ks:
            p=ch['previews'][str(k)];n=len(p['ids']) if frame_cut is None else sum(j<frame_cut for j in p['fr'])
            if keep is not None:n=keep[k][c]
            ids=p['ids'][:n]
            if keep is None or shown[:len(ids)]!=ids:shown=ids
            if not hold or screen[:len(committed)+len(shown)]!=committed+shown:screen=committed+shown
            free=max(min((c*H+k-1)*SHIFT+OFFSET,audio),free)+PASS_MS;events.append((free,words(screen)))
        free=max(min((c*H+W-1)*SHIFT+OFFSET,audio),free)+PASS_MS;committed=committed+ch['ids']
        if not hold or c==last or screen[:len(committed)]!=committed:screen=committed
        events.append((free,words(screen)))
    final=words(committed);errors,pairs=edit_pairs(ref['words'],final);stable=[None]*len(final)
    for i in range(len(final)):
        for t,shown in reversed(events):
            if i<len(shown) and shown[i]==final[i]:stable[i]=t
            else:break
    rewrites=0
    for (_,a),(_,b) in zip(events[:-1],events[1:]):rewrites+=sum(1 for i,w in enumerate(a) if i>=len(b) or not b[i].startswith(w))
    return [stable[i]-ref['end_ms'][j] for i,j in pairs],errors,len(ref['words']),len(final),rewrites,pairs


def main():
    for chunk in (16,8):run(chunk)


def run(chunk):
    global H,W,KS
    H=2*chunk;W=H+7;KS=[7,15,23] if chunk==16 else [7]
    tag=sys.argv[1];model=load(sys.argv[2] if len(sys.argv)>2 else None,chunk,sys.argv[3] if len(sys.argv)>3 else 'model').to(dev).eval();sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'))
    align=json.loads(gzip.decompress(base64.b64decode((ASR/'code/align_eval.b64').read_text())));piece=[sp.id_to_piece(i) for i in range(500)]
    words=lambda ids:''.join(piece[i] for i in ids).replace('▁',' ').upper().split();rows={}
    for s in ['dev-clean','dev-other']:
        fb=np.memmap(ASR/f'data/fbank_{s}.f16',np.float16,'r').reshape(-1,80)
        for r in json.load(open(ASR/f'data/index_{s}.json')):rows[r['utterance_id']]=(r,fb)
    records=[]
    for uid,ref in align.items():
        r,fb=rows[uid];f=torch.from_numpy(np.array(fb[r['feature_offset']:r['feature_offset']+r['feature_frames']],np.float32)).to(dev)
        records.append(dict(utterance_id=uid,split=ref['split'],audio_seconds=r['num_samples']/16000,chunks=replay(model,f)))
    out=ASR/'ckpt'/tag;out.mkdir(parents=True,exist_ok=True);json.dump(records,open(out/f'records_c{chunk}.json','w'));summary={}
    for s in ['dev-clean','dev-other']:
        own=[r for r in records if r['split']==s];res={}
        policies=[('none',[],None)]+[(f'k{k}',[k],None) for k in KS]+([(f'k7_cut{c}',[7],c) for c in (3,4,5,6)]+[('k7_23',[7,23],None)] if chunk==16 else [(f'k7_cut{c}',[7],c) for c in (2,3)])
        for name,ks,cut in policies:
            delay=[];err=0;refw=0;finw=0;rew=0
            for rec in own:
                d,e,rw,fw,rr,_=simulate(rec,align[rec['utterance_id']],words,ks,cut);delay+=d;err+=e;refw+=rw;finw+=fw;rew+=rr
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
    json.dump(summary,open(out/f'eval_c{chunk}.json','w'),indent=1);a=summary['dev-clean'];b=summary['dev-other']
    line=lambda k:'%s %.0f/%.1f %.0f/%.1f'%(k,a[k]['delay_mean'],a[k]['rewrites_per_100'],b[k]['delay_mean'],b[k]['rewrites_per_100'])
    print('EVAL',tag,f'chunk{chunk*20}ms','WER %.2f/%.2f lag %.0f/%.0f |'%(a['none']['wer'],b['none']['wer'],a['lag']['mean'],b['lag']['mean']),' | '.join(line(k) for k in a if k!='lag'),flush=True)


if __name__=='__main__':main()
