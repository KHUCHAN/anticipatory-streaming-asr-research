#!/usr/bin/env python3
"""Synthetic 5/10-minute waveform streams, incremental encoder, cache accounting.

Whole test utterances are concatenated with 300 ms silence. Five-minute streams
are prefixes of the ten-minute streams: these are stress tests, not independent
long-form recordings. Timing references are old per-utterance CTC alignments.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from avs.data import FeatSet,FEATS
from avs.features import logmel
from avs.model import Decoder,sub_len
from avs.stream import Predictor,Decoder3
from avs.text import Tokenizer
from check_bounded_cache import models
from analyze_final import words_with_time,match_words,pct


def decoder_from(path):
    ck=torch.load(ROOT/path,map_location='cpu'); c=ck['cfg']
    d=Decoder(c['vocab'],c['d'],c['h'],c['ff'],c['dec_layers'],0.,cross=True)
    d.load_state_dict(ck['model']); return d.eval()


def decode(model,decoder,mel,ref_ids,ref_t,*,hist,release,chunk=8,sample=True):
    pred=Predictor(decoder,hist,release_cache=release)
    dec=Decoder3(pred,beam=4,weight=.5,hold=0)
    dec.reset(); cache=model.encoder.stream_init()
    total=int(sub_len(len(mel))); encs=[]; ids=[]; times=[]; steps=[]; snapshots=[]; free=0.
    for start in range(0,total,chunk):
        end=min(start+chunk,total)
        arrival=(160*(4*(end-1)+6)+400)/16000
        t=time.perf_counter()
        y=model.encoder.stream_chunk(mel[None,4*start:4*(end-1)+7],cache,64)[0]
        encs.append(y); enc=torch.cat(encs,0)
        new=dec.advance(model.ctc(y).log_softmax(-1),enc)
        dt=time.perf_counter()-t
        free=max(arrival,free)+dt
        if end==total: new+=dec.finish()
        ids+=new; times += [free]*len(new); steps.append((arrival,dt))
        if sample and (start==0 or start%(chunk*32)==0 or end==total):
            snapshots.append({'audio_s':round(arrival,3),**dec.state_usage(),
                              'encoder_cache_bytes':model.encoder.stream_state_bytes(cache),
                              'retained_encoder_output_bytes':enc.numel()*enc.element_size()})
    hyp=words_with_time(ids,times); ref=words_with_time(ref_ids,ref_t)
    errors,pairs=match_words([w for w,_ in ref],[w for w,_ in hyp])
    latency=[1000*(hyp[j][1]-ref[i][1]) for i,j in pairs]
    percentile=lambda xs,q:round(pct(xs,q),3) if xs else None
    record={'wer':round(100*errors/max(1,len(ref)),3),'errors':errors,'reference_words':len(ref),
            'commit_p50_ms':percentile(latency,.5),'commit_p95_ms':percentile(latency,.95),
            'rtf':round(sum(dt for _,dt in steps)/arrival,5),
            'step_p50_ms':percentile([dt*1000 for _,dt in steps],.5),
            'step_p95_ms':percentile([dt*1000 for _,dt in steps],.95),
            'first_minute_step_p50_ms':percentile([dt*1000 for tm,dt in steps if tm<=60],.5),
            'last_minute_step_p50_ms':percentile([dt*1000 for tm,dt in steps if tm>=arrival-60],.5),
            'snapshots':snapshots,'output_tokens':len(ids)}
    if snapshots:
        record['peak_text_storage_bytes']=max(r['text_storage_bytes'] for r in snapshots)
        record['final_audio_storage_bytes']=snapshots[-1]['audio_storage_bytes']
        record['final_token_metadata_nodes']=snapshots[-1]['token_nodes']
    del dec,pred,encs
    gc.collect()
    return record,ids


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--minutes',nargs='+',type=int,default=[5,10])
    ap.add_argument('--out',default='runs/followups/long_stream.json'); a=ap.parse_args()
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    tok=Tokenizer(FEATS/'spm.model'); model,band=models(); full=decoder_from('runs/stage2_audio/best.pt')
    st=json.loads((FEATS/'stats.json').read_text()); mean=np.asarray(st['mean'],np.float32); std=np.asarray(st['std'],np.float32)
    output={'scope':'waveform concatenation + precomputed causal mel; actual incremental encoder and search on simulated arrivals; concurrent seed training',
            'silence_seconds':.3,'reference_timing':'per-utterance CTC alignment offsets; not independent phonetic timestamps',
            'state_scope':'text tensor backing storages, audio cross-attention, encoder outputs/cache and token metadata counted separately',
            'results':{}}
    for split in ('test-clean','test-other'):
        ds=FeatSet(split,tok)
        metadata={r['utterance_id']:r for r in (json.loads(l) for l in (ROOT/f'data/processed/manifests/{split}.jsonl').open())}
        align=torch.load(ROOT/f'runs/stage1/align/{split}.pt',map_location='cpu')['full']
        eligible=sorted((i for i in range(len(ds)) if align[i] is not None),key=lambda i:tuple(map(int,ds.ids[i].split('-'))))
        waves=[]; selected=[]; cumulative=0
        for i in eligible:
            wav,sr=sf.read(ROOT/metadata[ds.ids[i]]['audio'],dtype='float32')
            if sr!=16000: raise ValueError('Expected 16 kHz audio')
            selected.append((i,cumulative,len(wav))); waves.append(wav)
            cumulative+=len(wav)
            if cumulative>=max(a.minutes)*60*16000: break
            waves.append(np.zeros(4800,np.float32)); cumulative+=4800
        whole=np.concatenate(waves)
        for minutes in a.minutes:
            used=[item for item in selected if item[1]<minutes*60*16000]
            end=used[-1][1]+used[-1][2]
            mel=torch.from_numpy((logmel(whole[:end])-mean)/std)
            ref_ids=[]; ref_t=[]
            for i,offset,_ in used:
                ref_ids+=ds.tokens[i]
                ref_t += [offset/16000+(160*(4*int(f)+3)+200)/16000 for f in align[i]]
            group={'audio_seconds':end/16000,'source_utterances':[ds.ids[i] for i,_,_ in used],
                   'speaker_count':len({ds.ids[i].split('-')[0] for i,_,_ in used}),'models':{}}
            outputs={}
            for name,decoder,hist,release in [('full',full,0,False),('band2_retained',band,2,False),('band2_released',band,2,True)]:
                record,ids=decode(model,decoder,mel,ref_ids,ref_t,hist=hist,release=release)
                group['models'][name]=record; outputs[name]=ids
                print(json.dumps({'split':split,'minutes':minutes,'model':name,
                                  **{k:v for k,v in record.items() if k!='snapshots'}}),flush=True)
            group['band_cache_transcripts_identical']=outputs['band2_retained']==outputs['band2_released']
            if not group['band_cache_transcripts_identical']: raise AssertionError('Released cache changed long-stream transcript')
            # Same source utterances, same frontend and decoder; reset at each source boundary.
            for name,decoder,hist,release in [('full',full,0,False),('band2_released',band,2,True)]:
                errors=words=0
                for i,offset,n in used:
                    single=torch.from_numpy((logmel(whole[offset:offset+n])-mean)/std)
                    times=[(160*(4*int(f)+3)+200)/16000 for f in align[i]]
                    record,_=decode(model,decoder,single,ds.tokens[i],times,hist=hist,release=release,sample=False)
                    errors+=record['errors']; words+=record['reference_words']
                group['models'][name]['utterance_reset_wer']=round(100*errors/words,3)
            output['results'][f'{split}/{minutes}min']=group
            target=ROOT/a.out; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(output,indent=2)+'\n')
    print('LONG STREAM EVALUATION COMPLETE',flush=True)


if __name__=='__main__': main()
