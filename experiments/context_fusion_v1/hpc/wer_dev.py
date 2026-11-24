"""Native streaming WER (320ms chunks, greedy) on all of dev-clean and dev-other, with a paired utterance bootstrap against the public model (its per-utterance errors in ckpt/base/wer_dev.json).
Usage: wer_dev.py TAG [CHECKPOINT [KEY]]. With $LMX and LAMN > 0 the external LM picks among non-blank tokens whenever the transducer emits."""
import json,sys
import numpy as np,torch,sentencepiece as spm
import evaluate as E
from model_def import ASR,load


def run(tag,checkpoint,key):
    E.setup();E.KS=[];E.COMP=0;base=ASR/'ckpt/base/wer_dev.json';model=load(checkpoint,16,key).to(E.dev).eval()
    if E.LAMN==0:E.LMX=None
    sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'));piece=[sp.id_to_piece(i) for i in range(500)];words=lambda ids:''.join(piece[i] for i in ids).replace('▁',' ').upper().split();out={}
    for s in ['dev-clean','dev-other']:
        fb=np.memmap(ASR/f'data/fbank_{s}.f16',np.float16,'r').reshape(-1,80);rows=[]
        for r in json.load(open(ASR/f'data/index_{s}.json')):
            f=torch.from_numpy(np.array(fb[r['feature_offset']:r['feature_offset']+r['feature_frames']],np.float32)).to(E.dev)
            ref=r['text'].upper().split();rows.append((r['utterance_id'],E.edit_pairs(ref,words([t for ch in E.replay(model,f) for t in ch['ids']]))[0],len(ref)))
        out[s]=rows
    path=ASR/'ckpt'/tag;path.mkdir(parents=True,exist_ok=True);json.dump(out,open(path/('wer_dev_n%g.json'%E.LAMN),'w'));parts=[]
    for s,rows in out.items():
        e=np.array([r[1] for r in rows]);n=np.array([r[2] for r in rows]);text='%s %.2f (%d utterances)'%(s,100*e.sum()/n.sum(),len(rows))
        if base.exists():
            b=np.array([r[1] for r in json.load(open(base))[s]]);idx=np.random.default_rng(6137).integers(0,len(e),(2000,len(e)));d=100*(e[idx].sum(1)-b[idx].sum(1))/n[idx].sum(1)
            text+=' vs public %+.2f [%+.2f, %+.2f]'%(100*(e.sum()-b.sum())/n.sum(),*np.percentile(d,[2.5,97.5]))
        parts.append(text)
    print('WERDEV',tag,'lamn=%g'%E.LAMN,'|',' | '.join(parts),flush=True)


if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None,sys.argv[3] if len(sys.argv)>3 else 'model')
