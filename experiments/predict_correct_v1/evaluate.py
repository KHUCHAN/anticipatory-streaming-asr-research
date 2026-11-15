#!/usr/bin/env python3
"""Evaluate final frozen checkpoints and the full encoder on identical dev audio."""
from collections import defaultdict
import json
import time

import numpy as np
import torch

from common import ROOT,HERE,OUT,config,write,sha
from data import datasets
from model import Corrector
from inference import Inference


def summarize(rows):
    words=sum(r['reference_words'] for r in rows);errors=sum(r['errors'] for r in rows)
    audio=sum(r['audio_seconds'] for r in rows);chunks=sum(r['chunks'] for r in rows)
    total=sum(r['total_seconds'] for r in rows);samples=[t*1000 for r in rows for t in r['chunk_seconds']]
    return dict(utterances=len(rows),wer_percent=100*errors/words,errors=errors,reference_words=words,
        audio_seconds=audio,chunks=chunks,total_inference_seconds=total,rtf=total/audio,
        total_ms_per_chunk=1000*total/chunks,total_with_audio_io_rtf=sum(r['total_with_audio_io_seconds'] for r in rows)/audio,
        feature_ms_per_chunk=1000*sum(r['feature_seconds'] for r in rows)/chunks,
        components_ms_per_chunk={k:1000*sum(r['components'][k] for r in rows)/chunks for k in rows[0]['components']},
        other_overhead_ms_per_chunk=1000*sum(r['other_overhead_seconds'] for r in rows)/chunks,
        compute_chunk_p50_ms=float(np.quantile(samples,.5)),compute_chunk_p95_ms=float(np.quantile(samples,.95)),
        state_mae=sum(r['mae_sum'] for r in rows)/sum(r['mae_count'] for r in rows))


def bootstrap(a,b,cfg):
    assert [r['utterance_id'] for r in a]==[r['utterance_id'] for r in b]
    delta=np.array([x['errors']-y['errors'] for x,y in zip(a,b)])
    nref=np.array([r['reference_words'] for r in a]);rng=np.random.default_rng(9173)
    indices=rng.integers(0,len(a),(cfg['bootstrap_replicates'],len(a)))
    values=100*delta[indices].sum(1)/nref[indices].sum(1)
    return dict(wer_difference_pp=100*delta.sum()/nref.sum(),ci95_pp=np.quantile(values,[.025,.975]).tolist(),
                replicates=cfg['bootstrap_replicates'],unit='paired utterance; does not measure training-seed variance')


def main():
    torch.set_num_threads(1);cfg=config();_,dev,evaluation=datasets(cfg)
    plan=json.loads((HERE/'artifacts/evaluation_plan.json').read_text())
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    rows=sorted(evaluation,key=lambda row:(row['split'],row['utterance_id']))
    log=(OUT/'evaluation.jsonl').open('a',buffering=1)
    def emit(value):print(json.dumps(value),flush=True);log.write(json.dumps(value)+'\n')
    models={};digests={}
    for variant in cfg['variants']:
        torch.manual_seed(cfg['seed']);model=Corrector(cfg,variant).eval()
        checkpoint=torch.load(OUT/variant/'best.pt',map_location='cpu',weights_only=False)
        assert checkpoint['cfg']==cfg;model.load_state_dict(checkpoint['model']);models[variant]=model
        digests[variant]=sha(OUT/variant/'best.pt')
    metrics={};all_records={}
    for repetition in range(cfg['timing_repetitions']):
        order=['full_encoder','A','B','C'] if repetition%2==0 else ['C','B','A','full_encoder']
        for variant in order:
            result_path=OUT/'evaluation'/f'{variant}_repeat{repetition}.json'
            if result_path.exists():
                result=json.loads(result_path.read_text());assert result['model_sha256']==digests.get(variant,'frozen official baseline')
                assert result['cfg']==cfg
                assert [r['utterance_id'] for r in result['records']]==[r['utterance_id'] for r in rows]
            else:
                infer=Inference(cfg,models.get(variant),baseline=variant=='full_encoder')
                for row in dev.rows[:2]:infer.utterance(row)
                records=[];started=time.time();emit(dict(event='evaluation_start',variant=variant,repetition=repetition))
                for i,row in enumerate(rows,1):
                    records.append(infer.utterance(row))
                    if i%100==0:emit(dict(event='evaluation_progress',variant=variant,repetition=repetition,done=i,total=len(rows),seconds=time.time()-started))
                result=dict(variant=variant,repetition=repetition,records=records,
                    model_sha256=digests.get(variant,'frozen official baseline'),cfg=cfg)
                write(result_path,result);del infer
            all_records[(variant,repetition)]=result['records']
            for split in ['dev-clean','dev-other']:
                metrics[f'{variant}/{repetition}/{split}']=summarize([r for r in result['records'] if r['split']==split])
            emit(dict(event='evaluation_complete',variant=variant,repetition=repetition,
                scores={s:metrics[f'{variant}/{repetition}/{s}'] for s in ['dev-clean','dev-other']}))
    for variant in ['full_encoder',*cfg['variants']]:
        a=all_records[(variant,0)]
        for repetition in range(1,cfg['timing_repetitions']):
            b=all_records[(variant,repetition)]
            assert [(r['utterance_id'],r['token_ids']) for r in a]==[(r['utterance_id'],r['token_ids']) for r in b]
    primary={};paired={}
    for variant in ['full_encoder',*cfg['variants']]:
        for split in ['dev-clean','dev-other']:
            values=[metrics[f'{variant}/{i}/{split}'] for i in range(cfg['timing_repetitions'])]
            score=dict(values[0])
            for key in ['rtf','total_ms_per_chunk','total_inference_seconds','total_with_audio_io_rtf','feature_ms_per_chunk','other_overhead_ms_per_chunk','compute_chunk_p50_ms','compute_chunk_p95_ms']:
                score[key]=sum(v[key] for v in values)/len(values)
            score['components_ms_per_chunk']={k:sum(v['components_ms_per_chunk'][k] for v in values)/len(values) for k in values[0]['components_ms_per_chunk']}
            score['timing_repeat_total_ms_per_chunk']=[v['total_ms_per_chunk'] for v in values]
            primary[f'{variant}/{split}']=score
    for split in ['dev-clean','dev-other']:
        for a,b in [('B','A'),('B','C'),('C','A')]:
            own=[r for r in all_records[(a,0)] if r['split']==split]
            other=[r for r in all_records[(b,0)] if r['split']==split]
            paired[f'{split}/{a}-minus-{b}']=bootstrap(own,other,cfg)
    decision={};operating_points={}
    for split in ['dev-clean','dev-other']:
        p=paired[f'{split}/B-minus-A'];b=primary[f'B/{split}'];a=primary[f'A/{split}'];teacher=primary[f'full_encoder/{split}']
        decision[split]=dict(b_wer_better_than_a=p['wer_difference_pp']<0,
            b_wer_ci_excludes_zero_in_beneficial_direction=p['ci95_pp'][1]<0,
            b_observed_improvement_meets_0_5pp=p['wer_difference_pp']<=-cfg['practical_wer_improvement_pp'],
            b_total_time_lower_than_full_encoder=b['rtf']<teacher['rtf'],
            b_wer_delta_vs_full_encoder_pp=b['wer_percent']-teacher['wer_percent'],
            b_meets_teacher_wer_tolerance=b['wer_percent']-teacher['wer_percent']<=cfg['max_wer_degradation_vs_teacher_pp'],
            b_meets_minimum_compute_saving=b['rtf']<=teacher['rtf']*(1-cfg['minimum_compute_saving_fraction']),
            b_total_time_relative_to_a=b['rtf']/a['rtf'],
            b_speedup_vs_full_encoder=teacher['rtf']/b['rtf'],
            note='Different total parameter/runtime budgets: this is a measured frontier comparison, not an iso-compute superiority claim. Check quality loss versus the full encoder separately.')
        for variant in cfg['variants']:
            own=primary[f'{variant}/{split}']
            operating_points[f'{variant}/{split}']=dict(
                wer_degradation_vs_full_encoder_pp=own['wer_percent']-teacher['wer_percent'],
                speedup_vs_full_encoder=teacher['rtf']/own['rtf'],
                meets_wer_tolerance=own['wer_percent']-teacher['wer_percent']<=cfg['max_wer_degradation_vs_teacher_pp'],
                meets_compute_saving=own['rtf']<=teacher['rtf']*(1-cfg['minimum_compute_saving_fraction']))
    result=dict(status='evaluation completed',primary=primary,paired_comparisons=paired,decisions=decision,
        operating_points=operating_points,evaluation_source_sha256=sha(HERE/'evaluate.py'),
        model_sha256=digests,all_repeated_transcripts_match=True,cfg=cfg,test_used=False,
        timing_scope='CPU batch-1 one-thread streaming replay from resident raw waveform. Includes fbank, normalization, conv caches, FGN prediction, correction, corrected-state memory update, token decoding and Python/array conversion overhead. Reference metrics and file IO excluded from primary runtime; IO-inclusive RTF reported separately. No microphone wait time or UI latency.',
        limitations='Teacher-forced distillation contexts are labels only. Final decoding is free-running with own corrected feedback. Single seed, no adaptive fallback or teacher-cache recovery, no total-parameter matching. Timing repeat order reversed; two repeats do not establish hardware-wide uncertainty.')
    write(OUT/'results.json',result);write(HERE/'artifacts/results.json',result)
    emit(dict(event='all_evaluation_complete',test_used=False))


if __name__=='__main__':main()
