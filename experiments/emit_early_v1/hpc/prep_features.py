"""80-bin fbank for one LibriSpeech split, identical to experiments/predict_correct_v1/prepare.py (kaldi-native-fbank, 10560 zero tail)."""
import json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf


def features(wave):
    o=knf.FbankOptions();o.frame_opts.dither=0;o.frame_opts.snip_edges=False
    o.mel_opts.num_bins=80;o.mel_opts.low_freq=20;o.mel_opts.high_freq=-400
    fb=knf.OnlineFbank(o);fb.accept_waveform(16000,wave);fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])


def one(item):
    uid,path=item;wave,sr=sf.read(path,dtype='float32');assert sr==16000
    return uid,len(wave),features(np.concatenate([wave,np.zeros(10560,np.float32)])).astype(np.float16)


def main():
    root=Path(sys.argv[1]);split=sys.argv[2];workers=int(sys.argv[3]);items=[];text={}
    for trans in sorted((root/'LibriSpeech'/split).glob('*/*/*.trans.txt')):
        for line in trans.read_text().splitlines():
            uid,t=line.split(' ',1);text[uid]=t;items.append((uid,str(trans.parent/f'{uid}.flac')))
    items.sort();index=[];offset=0
    with open(root/f'fbank_{split}.f16','wb') as out,ProcessPoolExecutor(workers) as pool:
        for uid,n,f in pool.map(one,items,chunksize=16):
            out.write(f.tobytes());index.append(dict(utterance_id=uid,split=split,text=text[uid],num_samples=n,feature_offset=offset,feature_frames=len(f)));offset+=len(f)
    json.dump(index,open(root/f'index_{split}.json','w'));print(split,len(index),offset,flush=True)


if __name__=='__main__':main()
