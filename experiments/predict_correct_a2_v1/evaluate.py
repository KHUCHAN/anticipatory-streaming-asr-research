#!/usr/bin/env python3
"""Frozen held-out WER/cost collection for both phases and native controls."""
import json
import time

import numpy as np
import torch

from common import ROOT,HERE,OUT,config,sha,write
from data import datasets
from inference import Inference
from model import A2
from onnx_backend import export
from optimized_inference import OptimizedInference


def summarize(rows):
    words=sum(r['reference_words'] for r in rows);errors=sum(r['errors'] for r in rows)
    audio=sum(r['audio_seconds'] for r in rows);chunks=sum(r['chunks'] for r in rows);total=sum(r['total_seconds'] for r in rows)
    metric_count=sum(r['mae_count'] for r in rows)
    return dict(utterances=len(rows),wer_percent=100*errors/words,errors=errors,reference_words=words,
        audio_seconds=audio,chunks=chunks,output_frames=sum(r['output_frames'] for r in rows),
        feature_window=rows[0]['window'],feature_hop=rows[0]['hop'],rtf=total/audio,total_seconds=total,
        total_ms_per_chunk=1000*total/chunks,rtf_with_audio_io=sum(r['total_with_audio_io_seconds'] for r in rows)/audio,
        feature_ms_per_chunk=1000*sum(r['feature_seconds'] for r in rows)/chunks,
        components_ms_per_chunk={k:1000*sum(r['components'][k] for r in rows)/chunks for k in ['audio','head','decoder']},
        other_overhead_ms_per_chunk=1000*sum(r['other_overhead_seconds'] for r in rows)/chunks,
        state_mae=sum(r['mae_sum'] for r in rows)/metric_count if metric_count else None,
        compute_chunk_p50_ms=float(np.quantile([s*1000 for r in rows for s in r['chunk_seconds']],.5)),
        compute_chunk_p95_ms=float(np.quantile([s*1000 for r in rows for s in r['chunk_seconds']],.95)))


def bootstrap(a,b,cfg):
    assert [r['utterance_id'] for r in a]==[r['utterance_id'] for r in b]
    delta=np.array([x['errors']-y['errors'] for x,y in zip(a,b)],np.float64);words=np.array([r['reference_words'] for r in a])
    rng=np.random.default_rng(9237);indices=rng.integers(0,len(a),(cfg['bootstrap_replicates'],len(a)))
    draws=100*delta[indices].sum(1)/words[indices].sum(1)
    return dict(wer_difference_pp=float(100*delta.sum()/words.sum()),ci95_pp=np.quantile(draws,[.025,.975]).tolist(),unit='paired utterance, conditional on the selected single-seed checkpoints')


def main():
    torch.set_num_threads(1);cfg=config();_,dev,evaluation=datasets(cfg)
    plan=json.loads((HERE/'artifacts/evaluation_plan.json').read_text())
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    rows=sorted(evaluation,key=lambda r:(r['split'],r['utterance_id']))
    output=OUT/'evaluation';output.mkdir(exist_ok=True);log=(OUT/'evaluation.jsonl').open('a',buffering=1)
    def emit(value):print(json.dumps(value),flush=True);log.write(json.dumps(value)+'\n')
    models={};digests={};selection={}
    for phase in ['mae','kl']:
        path=OUT/phase/'best.pt'
        if not path.exists():continue
        ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['cfg']==cfg
        model=A2(cfg).eval();model.load_state_dict(ck['model']);models[phase]=model;digests[phase]=sha(path)
        selection[phase]=dict(epoch=ck['epoch'],dev_clean_wer=ck['record']['selection']['dev-clean']['wer_percent'])
    chosen=min(models,key=lambda p:selection[p]['dev_clean_wer'])
    write(OUT/'candidate_selection.json',dict(selected=chosen,model_selection=selection,rule=cfg['checkpoint_selection'],chosen_before_heldout_decoding=True))
    exports={};backend_checks={}
    check_rows=[]
    for split in ['dev-clean','dev-other']:check_rows+=sorted([r for r in dev.rows if r['split']==split],key=lambda r:r['utterance_id'])[:4]
    for phase,model in models.items():
        directory,metadata=export(model,cfg,phase,digests[phase]);exports[phase]=directory
        write(HERE/'artifacts'/f'onnx_{phase}.json',dict(directory=str(directory.relative_to(ROOT)),**metadata))
        original=Inference(cfg,model);optimized=OptimizedInference(cfg,directory);checks=[]
        for row in check_rows:
            a=original.utterance(row);b=optimized.utterance(row)
            assert a['chunks']==b['chunks']
            checks.append(dict(utterance_id=row['utterance_id'],tokens_match=a['token_ids']==b['token_ids'],torch_errors=a['errors'],onnx_errors=b['errors']))
        backend_checks[phase]=dict(actual_audio_checks=checks,all_checked_tokens_match=all(r['tokens_match'] for r in checks),
            random_stream_maximum_difference=max(metadata['random_stream_max_differences']))
        del original,optimized
    write(HERE/'artifacts/backend_checks.json',backend_checks)
    order=['teacher','small',*models];records={};summaries={}
    for repeat in range(cfg['timing_repetitions']):
        for name in order if repeat%2==0 else list(reversed(order)):
            path=output/f'{name}_repeat{repeat}.json';digest=digests.get(name,'public native checkpoint')
            if path.exists():
                result=json.loads(path.read_text());assert result['cfg']==cfg and result['model_sha256']==digest
                assert [r['utterance_id'] for r in result['records']]==[r['utterance_id'] for r in rows]
            else:
                infer=OptimizedInference(cfg,exports[name]) if name in exports else Inference(cfg,small=name=='small')
                for row in dev.rows[:2]:infer.utterance(row)
                values=[];started=time.time();emit(dict(event='evaluation_start',model=name,repeat=repeat))
                for i,row in enumerate(rows,1):
                    values.append(infer.utterance(row))
                    if i%100==0:emit(dict(event='evaluation_progress',model=name,repeat=repeat,done=i,total=len(rows),seconds=time.time()-started))
                result=dict(cfg=cfg,model=name,model_sha256=digest,repeat=repeat,records=values);write(path,result);del infer
            records[(name,repeat)]=result['records']
            for split in ['dev-clean','dev-other']:summaries[(name,repeat,split)]=summarize([r for r in result['records'] if r['split']==split])
            emit(dict(event='evaluation_complete',model=name,repeat=repeat,scores={s:summaries[(name,repeat,s)] for s in ['dev-clean','dev-other']}))
    for name in order:
        initial=[(r['utterance_id'],r['token_ids']) for r in records[(name,0)]]
        for repeat in range(1,cfg['timing_repetitions']):assert initial==[(r['utterance_id'],r['token_ids']) for r in records[(name,repeat)]]
    primary={}
    for name in order:
        for split in ['dev-clean','dev-other']:
            own=[summaries[(name,i,split)] for i in range(cfg['timing_repetitions'])];value=dict(own[0])
            for key in ['rtf','total_seconds','total_ms_per_chunk','rtf_with_audio_io','feature_ms_per_chunk','other_overhead_ms_per_chunk','compute_chunk_p50_ms','compute_chunk_p95_ms']:
                value[key]=sum(r[key] for r in own)/len(own)
            value['components_ms_per_chunk']={key:sum(r['components_ms_per_chunk'][key] for r in own)/len(own) for key in ['audio','head','decoder']}
            value['timing_repeat_ms_per_chunk']=[r['total_ms_per_chunk'] for r in own];primary[f'{name}/{split}']=value
    previous=json.loads((ROOT/cfg['previous_cost_results']).read_text());qualification={};paired={}
    for name in models:
        score=primary[f'{name}/dev-clean'];teacher=primary['teacher/dev-clean'];old=previous['primary']['full_encoder/dev-clean']
        qualification[name]=dict(wer_below_10=score['wer_percent']<cfg['qualification_wer_clean_below'],
            same_run_cost_fraction=score['rtf']/teacher['rtf'],previous_cost_fraction=score['rtf']/old['rtf'],
            same_run_cost_at_most_half=score['rtf']<=cfg['qualification_cost_fraction_at_most']*teacher['rtf'],
            previous_cost_at_most_half=score['rtf']<=cfg['qualification_cost_fraction_at_most']*old['rtf'])
        q=qualification[name];q['passes_both_cost_denominators_and_wer']=q['wer_below_10'] and q['same_run_cost_at_most_half'] and q['previous_cost_at_most_half']
        for split in ['dev-clean','dev-other']:
            a=[r for r in records[(name,0)] if r['split']==split]
            for other in ['teacher','small']+(['mae'] if name=='kl' else []):
                b=[r for r in records[(other,0)] if r['split']==split];paired[f'{split}/{name}-minus-{other}']=bootstrap(a,b,cfg)
    result=dict(status='evaluation completed',selected_candidate=chosen,model_selection=selection,primary=primary,
        qualification=qualification,paired_comparisons=paired,model_sha256=digests,
        backend_checks=backend_checks,
        all_repeated_transcripts_match=True,evaluation_source_sha256=sha(HERE/'evaluate.py'),test_used=False,
        timing_scope='CPU batch 1, one thread. Resident raw waveform to free-running native transcription; fbank, normalization, caches, A2 head, decoder, conversion/Python overhead included. Model load and reference metrics excluded; file IO additionally reported. No microphone waiting/display latency.',
        decoder_scope='Each recognizer uses its own native ONNX prediction network/joiner and tokens. A2 and teacher share native decoder weights; A2 loads/runs no teacher encoder. A2 frontend/head use FP32 ONNX first/step exports; training/selection used the numerically checked PyTorch implementation.',
        limitations='Single seed; reused project dev roles, no test split. Small recognizer uses native 39-frame window versus 45 for teacher/A2; all hop 32 frames. All measured neural paths use ONNX CPU. Two timing repetitions do not establish hardware-wide uncertainty.',cfg=cfg)
    write(OUT/'results.json',result);write(HERE/'artifacts/results.json',result);emit(dict(event='all_evaluation_complete',selected=chosen))


if __name__=='__main__':main()
