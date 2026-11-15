#!/usr/bin/env python3
"""Verify training provenance, checkpoint selection and held-out decode coverage."""
import json
import math
import torch
from common import ROOT,HERE,OUT,config,sha,write


def main():
    cfg=config();provenance=json.loads((HERE/'artifacts/provenance.json').read_text())
    for name,h in provenance['sha256'].items():assert sha(ROOT/name)==h,name
    baseline=json.loads((HERE/'artifacts/baseline_provenance.json').read_text())
    for name,h in baseline['sha256'].items():assert sha(ROOT/name)==h,name
    protocol=json.loads((OUT/'protocol.json').read_text());select=set(protocol['selection_ids']);evaluation=set(protocol['evaluation_ids'])
    assert len(select)==128 and len(evaluation)==800 and not select&evaluation
    rows=json.loads((OUT/'cache/index.json').read_text());mapping={r['utterance_id']:r for r in rows}
    assert not {mapping[x]['speaker_id'] for x in select}&{mapping[x]['speaker_id'] for x in evaluation}
    assert sum(r['split']=='train-clean-100' for r in rows)==28539
    checked=[]
    for variant in cfg['variants']:
        log=[json.loads(s) for s in (OUT/variant/'log.jsonl').read_text().splitlines()]
        history=[r for r in log if r['event']=='eval'];assert [r['epoch'] for r in history]==list(range(1,cfg['epochs']+1))
        last=torch.load(OUT/variant/'last.pt',map_location='cpu',weights_only=False)
        best=torch.load(OUT/variant/'best.pt',map_location='cpu',weights_only=False)
        assert last['cfg']==best['cfg']==cfg
        assert last['epoch']==cfg['epochs'] and last['step']==446*cfg['epochs']
        optimum=min(history,key=lambda r:r['selection_wer']['wer_percent'])
        assert best['epoch']==optimum['epoch'] and best['record']==optimum
        assert last['record']==history[-1]
        assert best['source_sha256']==last['source_sha256']
        for name,h in last['source_sha256'].items():assert sha(ROOT/name)==h,name
        if variant=='B':
            original=torch.load(ROOT/cfg['predictor_checkpoint'],map_location='cpu',weights_only=False)['model']
            assert all(torch.equal(last['model']['predictor.'+k],v) for k,v in original.items())
        checked.append(dict(variant=variant,epochs=last['epoch'],best_epoch=best['epoch'],best_selection_wer=best['record']['selection_wer']['wer_percent'],best_sha256=sha(OUT/variant/'best.pt')))
        del last,best
    results=json.loads((OUT/'results.json').read_text())
    plan=json.loads((HERE/'artifacts/evaluation_plan.json').read_text())
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    assert results['evaluation_source_sha256']==sha(HERE/'evaluate.py')
    corpus_checked=[]
    for variant in ['full_encoder',*cfg['variants']]:
        first=None
        repeat_totals={split:[] for split in ['dev-clean','dev-other']}
        for repeat in range(cfg['timing_repetitions']):
            data=json.loads((OUT/'evaluation'/f'{variant}_repeat{repeat}.json').read_text())
            assert data['cfg']==cfg and {r['utterance_id'] for r in data['records']}==evaluation
            assert len(data['records'])==len(evaluation)
            if variant!='full_encoder':assert data['model_sha256']==sha(OUT/variant/'best.pt')
            signature=[(r['utterance_id'],r['reference'],r['hypothesis'],r['token_ids'],r['errors'],r['reference_words']) for r in data['records']]
            if first is None:first=signature
            else:assert signature==first
            for row in data['records']:
                assert row['reference_words']>0 and row['total_seconds']>0
                assert len(row['chunk_seconds'])==row['chunks']
                assert abs(row['total_seconds']-(row['feature_seconds']+sum(row['components'].values())+row['other_overhead_seconds']))<1e-8
            for split in repeat_totals:
                records=[r for r in data['records'] if r['split']==split]
                aggregate=results['primary'][f'{variant}/{split}']
                assert len(records)==cfg['evaluation_per_split']==aggregate['utterances']
                errors=sum(r['errors'] for r in records);words=sum(r['reference_words'] for r in records)
                assert errors==aggregate['errors'] and words==aggregate['reference_words']
                assert math.isclose(100*errors/words,aggregate['wer_percent'],abs_tol=1e-10)
                chunks=sum(r['chunks'] for r in records);audio=sum(r['audio_seconds'] for r in records)
                seconds=sum(r['total_seconds'] for r in records)
                repeat_totals[split].append((seconds,seconds/audio,1000*seconds/chunks))
        for split,values in repeat_totals.items():
            aggregate=results['primary'][f'{variant}/{split}']
            for column,key in enumerate(['total_inference_seconds','rtf','total_ms_per_chunk']):
                assert math.isclose(sum(v[column] for v in values)/len(values),aggregate[key],abs_tol=1e-10)
            corpus_checked.append(f'{variant}/{split}')
    assert results['all_repeated_transcripts_match'] and not results['test_used']
    checks=json.loads((HERE/'artifacts/initial_checks.json').read_text())
    assert checks['training_cutoffs_change_each_epoch'] and checks['frozen_predictor_gradients_absent']
    value=dict(status='verified',models=checked,original_predictor_unchanged=True,
        train_utterances=28539,selection_utterances=128,evaluation_utterances=800,
        selection_evaluation_speakers_disjoint=True,all_timing_repeats_covered=True,
        timing_components_sum_to_total=True,raw_corpus_aggregates_verified=corpus_checked,
        repeat_transcripts_independently_checked=True,initial_causality_and_streaming_checks=checks,test_used=False)
    write(HERE/'artifacts/verification.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
