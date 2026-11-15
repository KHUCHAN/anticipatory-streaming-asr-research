#!/usr/bin/env python3
"""Independent full-corpus native audit after a small-model reproduction gap."""
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'runs/predict_correct_a2_v1'


def word_error(reference,hypothesis):
    a=reference.split();b=hypothesis.split();old=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,old[j]+1,old[j-1]+(x!=y)))
        old=new
    return old[-1],len(a)


def main():
    cfg=json.loads((HERE/'config.json').read_text());folder=ROOT/cfg['small_model']
    recognizer=sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(folder/'tokens.txt'),
        encoder=str(folder/'encoder-epoch-99-avg-1.onnx'),decoder=str(folder/'decoder-epoch-99-avg-1.onnx'),joiner=str(folder/'joiner-epoch-99-avg-1.onnx'),
        num_threads=1,sample_rate=16000,feature_dim=80,low_freq=20,high_freq=-400,dither=0,snip_edges=False,
        normalize_samples=True,enable_endpoint_detection=False,decoding_method='greedy_search',provider='cpu')
    token_ids={line.rsplit(' ',1)[0].replace('▁',' '):int(line.rsplit(' ',1)[1]) for line in (folder/'tokens.txt').read_text().splitlines()}
    def decode(path):
        wave,sr=sf.read(path,dtype='float32');stream=recognizer.create_stream();wave=np.concatenate([wave,np.zeros(10560,np.float32)]);chunks=0
        for lo in range(0,len(wave),320):
            stream.accept_waveform(sr,wave[lo:lo+320])
            while recognizer.is_ready(stream):recognizer.decode_stream(stream);chunks+=1
        stream.input_finished()
        while recognizer.is_ready(stream):recognizer.decode_stream(stream);chunks+=1
        value=recognizer.get_result_all(stream)
        return ' '.join(value.text.upper().split()),[token_ids[x] for x in value.tokens],chunks
    demo,_,_=decode(folder/'test_wavs/0.wav')
    # Exact documented upstream demo output; not used to tune the model.
    expected='THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BRAFFLELS'
    assert demo==expected
    selected=set(json.loads((ROOT/cfg['protocol']).read_text())['evaluation_ids'])
    rows=sorted([r for r in json.loads((ROOT/cfg['input_cache']/'index.json').read_text()) if r['utterance_id'] in selected],key=lambda r:(r['split'],r['utterance_id']))
    original={r['utterance_id']:r for r in json.loads((OUT/'evaluation/small_repeat0.json').read_text())['records']}
    records=[];started=time.time()
    for i,row in enumerate(rows,1):
        text,ids,chunks=decode(ROOT/row['audio']);errors,words=word_error(row['text'],text);old=original[row['utterance_id']]
        records.append(dict(utterance_id=row['utterance_id'],split=row['split'],text=text,errors=errors,words=words,
            native_tokens_match=ids==old['token_ids'],native_chunks_match=chunks==old['chunks'],
            text_matches_primary=text==old['hypothesis'],primary_errors=old['errors']))
        if i%100==0:print(json.dumps(dict(done=i,total=len(rows),seconds=time.time()-started)),flush=True)
    (OUT/'small_native_audit_records.json').write_text(json.dumps(records,indent=2)+'\n')
    scores={}
    for split in ['dev-clean','dev-other']:
        own=[r for r in records if r['split']==split];scores[split]=dict(wer_percent=100*sum(r['errors'] for r in own)/sum(r['words'] for r in own),utterances=len(own))
    report=dict(status='completed',sherpa_onnx_version=importlib.metadata.version('sherpa-onnx'),scores=scores,
        upstream_demo_exact_match=True,all_native_tokens_match=all(r['native_tokens_match'] for r in records),all_native_chunk_counts_match=all(r['native_chunks_match'] for r in records),
        text_difference_count=sum(not r['text_matches_primary'] for r in records),
        native_minus_primary_errors={split:sum(r['errors']-r['primary_errors'] for r in records if r['split']==split) for split in scores},
        interpretation='Token/chunk sequences agree with the independent native engine. Native text formatting has a small WER difference, recorded separately. The source card reports different test splits and incompletely matched publication conditions; its substantially lower reported level is not validated by this protocol. The gap remains unresolved.',
        extra_checks='Public-demo waveform scaling and high-frequency settings were checked in separate local diagnostics; changing to PCM-scale inputs produced empty demo transcripts. Longer tail padding did not change the public demo. Prefix-silence probes did not consistently repair selection-role outputs. These diagnostic variants do not replace the frozen primary results.',
        training_changed=False,test_splits_evaluated=False)
    (HERE/'artifacts/small_reproduction_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
