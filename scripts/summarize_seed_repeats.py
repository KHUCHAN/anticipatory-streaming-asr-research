#!/usr/bin/env python3
"""Paired seed/utterance summaries; no equivalence claim from a nonsignificant CI."""
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/followups/seeds'


def load(seed,model,split):
    if seed==2:
        source=ROOT/(f'runs/stage3/main/{split}.json' if model=='full' else f'runs/stage3/variants/{split}_b2.json')
    else:
        source=OUT/f'{model}_s{seed}_{split}.json'
    data=json.loads(source.read_text())
    if data['args']['hold']!=0: raise AssertionError('Hold policy differs from original comparison')
    return data


def main():
    result={'seeds':[2,3,4],'scope':'same frozen acoustic model; warm-started predictors; training RNG/data order/cutoffs vary in paired fashion',
            'ci_note':'95% paired utterance bootstrap within each seed. Seed mean/SD descriptive with only three training seeds; nonsignificance is not equivalence.',
            'results':{}}
    rng=np.random.default_rng(112026)
    for split in ('dev-clean','test-clean','test-other'):
        pairs={seed:(load(seed,'full',split),load(seed,'band2',split)) for seed in result['seeds']}
        for chunk in (4,8,16):
            records=[]
            for seed,(full,band) in pairs.items():
                a=full['per_utt'][f'B1a/c{chunk}']; b=band['per_utt'][f'H1a/c{chunk}']
                if [r['utt'] for r in a]!=[r['utt'] for r in b] or [r['nref'] for r in a]!=[r['nref'] for r in b]:
                    raise AssertionError('Paired utterance IDs or reference lengths differ')
                n=np.asarray([r['nref'] for r in a]); ea=np.asarray([r['err'] for r in a]); eb=np.asarray([r['err'] for r in b])
                idx=rng.integers(len(a),size=(2000,len(a)))
                boot=100*((eb-ea)[idx].sum(1)/n[idx].sum(1))
                wa=100*ea.sum()/n.sum(); wb=100*eb.sum()/n.sum()
                records.append({'seed':seed,'full_wer':round(float(wa),4),'band2_wer':round(float(wb),4),
                                'difference_pp':round(float(wb-wa),4),
                                'paired_utterance_ci95_pp':np.quantile(boot,[.025,.975]).round(4).tolist()})
            differences=[r['difference_pp'] for r in records]
            summary={'per_seed':records,
                     'full_mean_wer':statistics.mean(r['full_wer'] for r in records),
                     'band2_mean_wer':statistics.mean(r['band2_wer'] for r in records),
                     'mean_difference_pp':statistics.mean(differences),
                     'sd_difference_pp':statistics.stdev(differences),
                     'difference_range_pp':[min(differences),max(differences)]}
            result['results'][f'{split}/c{chunk}']=summary
            print(json.dumps({'condition':f'{split}/c{chunk}',**summary}),flush=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
