#!/usr/bin/env python3
"""Reproduce the independent native token/chunk consistency audit."""
import json

from common import ROOT,HERE,OUT,config,write
from inference import Inference


def main():
    cfg=config();rows={r['utterance_id']:r for r in json.loads((ROOT/cfg['input_cache']/'index.json').read_text())}
    reference=json.loads((OUT/'native_reference.json').read_text())['records'];checks=[]
    for name in ['teacher','small']:
        infer=Inference(cfg,small=name=='small')
        for row in reference:
            if row['model']!=name:continue
            result=infer.utterance(rows[row['utterance_id']]);same=result['token_ids']==row['tokens'];same_chunks=result['chunks']==row['chunks']
            assert same and same_chunks,(name,row['utterance_id'])
            checks.append(dict(model=name,utterance_id=row['utterance_id'],native_tokens_match=same,native_chunks_match=same_chunks,
                window=result['window'],chunks=result['chunks'],rtf=result['total_seconds']/result['audio_seconds']))
    write(HERE/'artifacts/native_checks.json',dict(checks=checks,all_match=True,test_used=False))
    print('All eight native token/chunk comparisons passed.')


if __name__=='__main__':main()
