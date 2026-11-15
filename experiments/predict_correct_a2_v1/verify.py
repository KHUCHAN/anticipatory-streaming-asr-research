#!/usr/bin/env python3
"""Verify immutable inputs, phase selection, native paths and all raw scores."""
import json
import math

import torch

from common import ROOT,HERE,OUT,config,sha,write


def main():
    cfg=config();provenance=json.loads((HERE/'artifacts/provenance.json').read_text())
    for name,h in provenance['sha256'].items():assert sha(ROOT/name)==h,name
    plan=json.loads((HERE/'artifacts/evaluation_plan.json').read_text())
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    roles=json.loads((ROOT/cfg['protocol']).read_text());selection=set(roles['selection_ids']);evaluation=set(roles['evaluation_ids'])
    rows=json.loads((ROOT/cfg['input_cache']/'index.json').read_text());mapping={r['utterance_id']:r for r in rows}
    assert len(selection)==128 and len(evaluation)==800 and not selection&evaluation
    assert not {mapping[i]['speaker_id'] for i in selection}&{mapping[i]['speaker_id'] for i in evaluation}
    phases=[]
    for phase in ['mae','kl']:
        path=OUT/phase/'best.pt'
        if not path.exists():continue
        history=[json.loads(s) for s in (OUT/phase/'log.jsonl').read_text().splitlines()];history=[r for r in history if r['event']=='eval']
        assert [r['epoch'] for r in history]==list(range(1,cfg[phase+'_epochs']+1))
        last=torch.load(OUT/phase/'last.pt',map_location='cpu',weights_only=False);best=torch.load(path,map_location='cpu',weights_only=False)
        assert best['cfg']==last['cfg']==cfg and last['step']==446*cfg[phase+'_epochs']
        optimum=min(history,key=lambda r:r['validation']['mae'] if phase=='mae' else r['selection']['dev-clean']['wer_percent'])
        assert best['record']==optimum and last['record']==history[-1]
        for name,h in best['source_sha256'].items():assert sha(ROOT/name)==h,name
        assert best['source_sha256']==last['source_sha256']
        if phase=='kl':
            calibration=json.loads((OUT/'kl_calibration.json').read_text());assert calibration['source_mae_sha256']==sha(OUT/'mae/best.pt')
            assert all(r['gradient']['mae_gradient_norm']>0 and r['gradient']['kl_gradient_norm']>0 for r in history)
        exported=json.loads((HERE/'artifacts'/f'onnx_{phase}.json').read_text());assert exported['source_model_sha256']==sha(path)
        assert exported['model_source_sha256']==sha(HERE/'model.py') and not exported['quantized'] and not exported['weights_changed']
        assert max(exported['random_stream_max_differences'])<2e-4
        for name,h in exported['files'].items():assert sha(ROOT/exported['directory']/name)==h,name
        phases.append(dict(phase=phase,epochs=last['epoch'],selected_epoch=best['epoch'],model_sha256=sha(path)))
        del best,last
    result=json.loads((OUT/'results.json').read_text());assert result['evaluation_source_sha256']==sha(HERE/'evaluate.py')
    chosen=json.loads((OUT/'candidate_selection.json').read_text())
    assert chosen['selected']==min(chosen['model_selection'],key=lambda k:chosen['model_selection'][k]['dev_clean_wer'])==result['selected_candidate']
    corpus=[]
    for name in ['teacher','small']+[p['phase'] for p in phases]:
        signature=None;time_sums={split:[] for split in ['dev-clean','dev-other']}
        for repeat in range(cfg['timing_repetitions']):
            raw=json.loads((OUT/'evaluation'/f'{name}_repeat{repeat}.json').read_text())
            assert raw['cfg']==cfg and {r['utterance_id'] for r in raw['records']}==evaluation and len(raw['records'])==800
            current=[(r['utterance_id'],r['reference'],r['hypothesis'],r['token_ids'],r['errors'],r['reference_words']) for r in raw['records']]
            if signature is None:signature=current
            else:assert signature==current
            for row in raw['records']:
                assert row['total_seconds']>0 and row['reference_words']>0 and len(row['chunk_seconds'])==row['chunks']
                assert math.isclose(row['total_seconds'],row['feature_seconds']+sum(row['components'].values())+row['other_overhead_seconds'],abs_tol=1e-8)
            for split in time_sums:
                own=[r for r in raw['records'] if r['split']==split];aggregate=result['primary'][f'{name}/{split}']
                assert len(own)==400 and sum(r['errors'] for r in own)==aggregate['errors']
                words=sum(r['reference_words'] for r in own);assert words==aggregate['reference_words']
                assert math.isclose(100*aggregate['errors']/words,aggregate['wer_percent'],abs_tol=1e-10)
                seconds=sum(r['total_seconds'] for r in own);audio=sum(r['audio_seconds'] for r in own);chunks=sum(r['chunks'] for r in own)
                time_sums[split].append((seconds,seconds/audio,1000*seconds/chunks))
        for split,times in time_sums.items():
            for column,key in enumerate(['total_seconds','rtf','total_ms_per_chunk']):
                assert math.isclose(sum(v[column] for v in times)/len(times),result['primary'][f'{name}/{split}'][key],abs_tol=1e-10)
            corpus.append(f'{name}/{split}')
    assert result['all_repeated_transcripts_match'] and result['test_used'] is False
    assert json.loads((HERE/'artifacts/native_checks.json').read_text())['all_match']
    verified=dict(status='verified',phases=phases,corpora_verified=corpus,original_inputs_unchanged=True,
        selected_before_heldout_decoding=True,speaker_disjoint_roles=True,all_repeated_transcripts_match=True,
        complete_runtime_sums_verified=True,fp32_onnx_exports_verified=True,selection_utterances=128,evaluation_utterances=800,test_used=False)
    write(HERE/'artifacts/verification.json',verified);print(json.dumps(verified,indent=2))


if __name__=='__main__':main()
