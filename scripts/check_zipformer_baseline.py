#!/usr/bin/env python3
"""Check real streaming packet invariance, suffix causality, and WER counts."""
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from analyze_zipformer_baseline import metric_checks
from eval_zipformer_baseline import ROOT, atomic_json, make_recognizer, sha256, transcribe, word_errors


def scalar_distance(a,b):
    old = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current = [i]
        for j,y in enumerate(b,1):
            current.append(min(old[j]+1,current[-1]+1,old[j-1]+(x!=y)))
        old = current
    return old[-1]


def main():
    config = json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    recognizer = make_recognizer(config)
    manifest = ROOT/'data/processed/manifests/dev-clean.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()[:10]]
    packet_checks, causal_checks = [], []
    output = ROOT/'runs/baseline_zipformer_checks'
    output.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        a = transcribe(row,recognizer,config,20)
        b = transcribe(row,recognizer,config,320)
        c = transcribe(row,recognizer,config,60000)
        for field in ['hypothesis','tokens','token_timestamps']:
            assert a[field] == b[field] == c[field], (row['utterance_id'],field)
        packet_checks.append(dict(id=row['utterance_id'],packet_20_320_full_token_timestamp_parity=True))
        if i < 3:
            waveform,sr = sf.read(ROOT/row['audio'],dtype='float32')
            cutoff = 1.5
            assert len(waveform)>int(cutoff*sr)
            rng = np.random.default_rng(117+i)
            waveform[int(cutoff*sr):] = rng.normal(0,.03,len(waveform)-int(cutoff*sr)).astype(np.float32)
            temporary = output/f'changed_suffix_{i}.wav'
            sf.write(temporary,waveform,sr,subtype='FLOAT')
            changed = dict(row,audio=str(temporary.relative_to(ROOT)))
            try:
                d = transcribe(changed,recognizer,config,20)
            finally:
                temporary.unlink(missing_ok=True)
            earlier = lambda r:[e for e in r['events'] if e['available_s']<=cutoff]
            assert earlier(a) == earlier(d), row['utterance_id']
            causal_checks.append(dict(id=row['utterance_id'],cutoff_s=cutoff,
                                      unchanged_prefix_events=len(earlier(a)),future_noise_suffix_invariant=True))
    rng = random.Random(29)
    for _ in range(500):
        a = rng.choices(['A','B','C'],k=rng.randrange(12))
        b = rng.choices(['A','B','C'],k=rng.randrange(12))
        result = word_errors(' '.join(a),' '.join(b))
        assert result['errors'] == scalar_distance(a,b)
        assert result['errors'] == result['substitutions']+result['deletions']+result['insertions']
    record = dict(packet_parity=packet_checks, suffix_causality=causal_checks,
                  word_edit_distance_cases=500,display_metric_checks=metric_checks(),
                  config_sha256=sha256(ROOT/'configs/zipformer_baseline.json'),
                  manifest_sha256=sha256(manifest),script_sha256=sha256(__file__))
    atomic_json(output/'streaming_checks.json',record)
    print('PASS: 10 packet-size checks, 3 changed-future checks, 500 WER checks, display metric checks.')


if __name__ == '__main__':
    main()
