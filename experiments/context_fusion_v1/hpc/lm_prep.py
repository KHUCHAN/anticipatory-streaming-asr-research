"""Token data for the LM. Usage: lm_prep.py THREADS [CORPUS_TXT_GZ]
Writes $ASR/data/lm_tokens.u16 (BPE ids, every sentence starts with 0) and lm_meta.json. The 960h training transcripts come first; the LibriSpeech LM corpus follows if its file is given."""
import gzip,itertools,json,os,sys
from pathlib import Path
import numpy as np,sentencepiece as spm

ASR=Path(os.environ.get('ASR','.'))


def main():
    threads=int(sys.argv[1]);sp=spm.SentencePieceProcessor(model_file=str(ASR/'src/bpe.model'));out=open(ASR/'data/lm_tokens.u16','wb');meta=dict(transcript_sentences=0,corpus_sentences=0,tokens=0)

    def write(lines,key):
        try:ids=sp.encode(lines,num_threads=threads)
        except TypeError:ids=sp.encode(lines)
        n=sum(map(len,ids))+len(ids);np.fromiter(itertools.chain.from_iterable(itertools.chain((0,),x) for x in ids),np.uint16,n).tofile(out);meta[key]+=len(ids);meta['tokens']+=n

    for s in ['train-clean-100','train-clean-360','train-other-500']:write([r['text'].upper() for r in json.load(open(ASR/f'data/index_{s}.json'))],'transcript_sentences')
    if len(sys.argv)>2:
        buf=[]
        with gzip.open(sys.argv[2],'rt') as f:
            for line in f:
                line=line.strip()
                if line:buf.append(line)
                if len(buf)>=500000:write(buf,'corpus_sentences');buf=[];print(meta,flush=True)
        if buf:write(buf,'corpus_sentences')
    out.close();json.dump(meta,open(ASR/'data/lm_meta.json','w'));print('LMPREP',meta,flush=True)


if __name__=='__main__':main()
