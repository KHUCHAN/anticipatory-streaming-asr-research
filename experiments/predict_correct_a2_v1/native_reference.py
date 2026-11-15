#!/usr/bin/env python3
"""Independent sherpa-onnx token reference; run in the baseline environment."""
import json
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    cfg=json.loads((HERE/'config.json').read_text());teacher=json.loads((ROOT/cfg['teacher_config']).read_text())
    rows=json.loads((ROOT/cfg['input_cache']/'index.json').read_text());roles=json.loads((ROOT/cfg['protocol']).read_text())
    selected=set(roles['selection_ids']);chosen=[]
    for split in ['dev-clean','dev-other']:
        chosen+=sorted([r for r in rows if r['utterance_id'] in selected and r['split']==split],key=lambda r:r['utterance_id'])[:2]
    records=[]
    for name in ['teacher','small']:
        directory=ROOT/(cfg['small_model'] if name=='small' else teacher['model_dir'])
        suffix='epoch-99-avg-1.onnx' if name=='small' else teacher['model_suffix']
        recognizer=sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(directory/'tokens.txt'),
            encoder=str(directory/f'encoder-{suffix}'),decoder=str(directory/f'decoder-{suffix}'),joiner=str(directory/f'joiner-{suffix}'),
            num_threads=1,sample_rate=16000,feature_dim=80,low_freq=20,high_freq=-400,dither=0,snip_edges=False,
            normalize_samples=True,enable_endpoint_detection=False,decoding_method='greedy_search',provider='cpu')
        ids={line.rsplit(' ',1)[0].replace('▁',' '):int(line.rsplit(' ',1)[1]) for line in (directory/'tokens.txt').read_text().splitlines()}
        for row in chosen:
            wave,sr=sf.read(ROOT/row['audio'],dtype='float32');stream=recognizer.create_stream();chunks=0
            wave=np.concatenate([wave,np.zeros(10560,np.float32)])
            for lo in range(0,len(wave),320):
                stream.accept_waveform(sr,wave[lo:lo+320])
                while recognizer.is_ready(stream):recognizer.decode_stream(stream);chunks+=1
            stream.input_finished()
            while recognizer.is_ready(stream):recognizer.decode_stream(stream);chunks+=1
            result=recognizer.get_result_all(stream)
            records.append(dict(model=name,utterance_id=row['utterance_id'],tokens=[ids[x] for x in result.tokens],
                text=' '.join(result.text.upper().split()),chunks=chunks))
    target=ROOT/'runs/predict_correct_a2_v1/native_reference.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(records=records,test_used=False),indent=2)+'\n')
    print('Native reference completed:',len(records),'decodes.')


if __name__=='__main__':main()
