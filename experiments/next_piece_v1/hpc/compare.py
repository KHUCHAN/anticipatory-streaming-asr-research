"""Paired utterance bootstrap between two full-dev WER results. Usage: compare.py FILE_A FILE_B (paths under $ASR/ckpt, as written by wer_dev.py). Prints A minus B."""
import json,sys
import numpy as np
from model_def import ASR

a=json.load(open(ASR/'ckpt'/sys.argv[1]));b=json.load(open(ASR/'ckpt'/sys.argv[2]));parts=[]
for s in a:
    e=np.array([r[1] for r in a[s]]);f=np.array([r[1] for r in b[s]]);n=np.array([r[2] for r in a[s]]);idx=np.random.default_rng(6137).integers(0,len(e),(2000,len(e)));d=100*(e[idx].sum(1)-f[idx].sum(1))/n[idx].sum(1)
    parts.append('%s %.2f vs %.2f: %+.2f [%+.2f, %+.2f]'%(s,100*e.sum()/n.sum(),100*f.sum()/n.sum(),100*(e.sum()-f.sum())/n.sum(),*np.percentile(d,[2.5,97.5])))
print('COMPARE',sys.argv[1],'minus',sys.argv[2],'|',' | '.join(parts),flush=True)
