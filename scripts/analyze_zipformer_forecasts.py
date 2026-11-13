#!/usr/bin/env python3
"""Fit our gate on dev-fit, calibrate on disjoint dev-cal, then evaluate test.

No reference or future native tokens enter a forecast/gate input. References
are used for supervised gate targets, calibration error counts and scoring.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from avs.zipformer_forecaster import read_rows

OUT=ROOT/'runs/zipformer_ours'
KINDS=['native','full','band2']


def transform(features):
    x=np.asarray(features,dtype=np.float64).copy()
    for j in [0,1]:
        p=np.clip(x[...,j],1e-5,1-1e-5);x[...,j]=np.log(p/(1-p))
    return x


def fit_gate(rows,kind):
    examples=[e for r in rows for e in r['events'][kind] if e['eligible']]
    if not examples:
        dim=9 if kind=='native' else 8
        return dict(mu=[0.]*dim,sd=[1.]*dim,w=[0.]*dim+[-60.],examples=0,positive_fraction=0.)
    x=transform([e['features'] for e in examples]);y=np.array([e['label_correct_and_keep'] for e in examples],np.float64)
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-5)
    xt=torch.tensor((x-mu)/sd,dtype=torch.float64);yt=torch.tensor(y)
    w=torch.zeros(x.shape[1]+1,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([w],max_iter=150,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(xt@w[:-1]+w[-1],yt)+.001*(w[:-1]**2).sum()
        loss.backward();return loss
    optimizer.step(closure)
    return dict(mu=mu.tolist(),sd=sd.tolist(),w=w.detach().tolist(),examples=len(examples),positive_fraction=float(y.mean()))


def add_scores(rows,gates):
    for row in rows:
        for kind in KINDS:
            events=row['events'][kind];x=transform([e['features'] for e in events]);g=gates[kind]
            z=((x-np.array(g['mu']))/np.array(g['sd']))@np.array(g['w'][:-1])+g['w'][-1]
            probs=1/(1+np.exp(-np.clip(z,-60,60)))
            for event,p in zip(events,probs):event['scores']={'raw':event['features'][0],'gate':float(p)}


def common_prefix(a,b):
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]:i+=1
    return i


def policy(row,kind,method,threshold,pieces=None,with_timeline=False,gate_seconds=0.):
    native=row['native'];correct=set(row['correct_native_positions']);pending=None;prev_count=0
    attempts=wrong=accepted=correct_accepted=retractions=0;visible=[];free=0.
    for event in row['events'][kind]:
        n=event['native_count'];base_time=event['available']
        actual_time=max(base_time,free)
        # Native tokens can appear immediately at this boundary before the next
        # forecast finishes. Preserve a pending token until verification says otherwise.
        if pending is not None and n>pending[1]:
            if native[pending[1]]==pending[0]:
                accepted+=1;correct_accepted+=int(pending[1] in correct)
            else:retractions+=1
            pending=None
        if with_timeline:
            ids=native[:n]+([pending[0]] if pending else [])
            visible.append((actual_time,render(ids,pieces)))
        finish=actual_time+event['seconds']+(gate_seconds if method=='gate' else 0.)
        free=finish
        want=event['eligible'] and event['scores'][method]>=threshold
        g=event['g']
        if pending and (not want or pending[0]!=g or pending[1]!=n):
            retractions+=1;pending=None
        if want and pending is None:
            pending=(g,n,finish);attempts+=1
            wrong+=int(not (n<len(native) and g==native[n] and n in correct))
        if with_timeline:
            ids=native[:n]+([pending[0]] if pending else [])
            visible.append((finish,render(ids,pieces)))
        prev_count=n
    if pending is not None:retractions+=1
    if with_timeline:visible.append((max(row['audio_seconds']+.66,free),row['hypothesis'].split()))
    return dict(attempts=attempts,wrong=wrong,accepted=accepted,correct_accepted=correct_accepted,
                retractions=retractions,timeline=visible)


def render(ids,pieces):return ''.join(pieces[i] for i in ids if i>=3).replace('▁',' ').split()


def baseline_timeline(row,pieces):
    result=[(e['available'],render(row['native'][:e['native_count']],pieces)) for e in row['events']['full']]
    result.append((row['audio_seconds']+.66,row['hypothesis'].split()))
    return result


def stable_times(timeline,final_words):
    values=[None]*len(final_words);suffix_min=len(final_words)
    for clock,words in reversed(timeline):
        suffix_min=min(suffix_min,common_prefix(words,final_words))
        for i in range(suffix_min):values[i]=clock
    assert all(x is not None for x in values)
    return values


def screen_rewrites(timeline):
    old=[];n=0
    for _,words in timeline:n+=len(old)-common_prefix(old,words);old=words
    return n


def character_edits(timeline):
    old='';events=deleted=0
    for _,words in timeline:
        text=' '.join(words)
        if not text.startswith(old):
            n=common_prefix(old,text);events+=1;deleted+=len(old)-n
        old=text
    return events,deleted


def counts(rows,kind,method,threshold):
    total={k:0 for k in ['attempts','wrong','accepted','correct_accepted','retractions']}
    for row in rows:
        stat=policy(row,kind,method,threshold)
        for k in total:total[k]+=stat[k]
    denominator=sum(len(r['native']) for r in rows)
    total['wrong_per_100_native_tokens']=100*total['wrong']/max(1,denominator)
    total['correct_accepted_per_100_native_tokens']=100*total['correct_accepted']/max(1,denominator)
    return total


def choose_threshold(rows,kind,method,budget):
    scores=[e['scores'][method] for r in rows for e in r['events'][kind] if e['eligible']]
    grid=sorted(set((np.quantile(scores,np.linspace(0,1,101)).tolist() if scores else [])+[1.000001]))
    curve=[]
    for th in grid:
        stat=counts(rows,kind,method,th);curve.append(dict(threshold=th,**stat))
    valid=[s for s in curve if s['wrong_per_100_native_tokens']<=budget]
    chosen=max(valid,key=lambda s:(s['correct_accepted'],-s['wrong'],-s['threshold']))
    return dict(chosen=chosen,curve=curve)


def load_events(part,ids=None):
    result=[]
    for path in sorted((OUT/f'events_{part}').glob('*.json')):
        if path.name=='complete.json':continue
        if ids is not None and path.stem not in ids:continue
        result.append(json.loads(path.read_text()))
    if ids is not None:assert {r['utterance_id'] for r in result}==ids
    return result


def calibration(cfg):
    protocol=ROOT/cfg['dev_protocol']
    fit_rows=read_rows(protocol/'gate_fit.jsonl');cal_rows=read_rows(protocol/'gate_calibration.jsonl')
    fit_ids={r['utterance_id'] for r in fit_rows};cal_ids={r['utterance_id'] for r in cal_rows}
    assert fit_ids.isdisjoint(cal_ids)
    assert {r['speaker_id'] for r in fit_rows}.isdisjoint({r['speaker_id'] for r in cal_rows})
    fit=load_events('dev',fit_ids);cal=load_events('dev',cal_ids)
    gates={kind:fit_gate(fit,kind) for kind in KINDS}
    add_scores(cal,gates);operating={}
    for kind in KINDS:
        for method in ['raw','gate']:
            key=kind+'/'+method;operating[key]=choose_threshold(cal,kind,method,cfg['calibration_false_forecasts_per_100_native_tokens'])
            print(key,json.dumps(operating[key]['chosen']),flush=True)
    collection=json.loads((OUT/'events_dev_provenance.json').read_text())
    result=dict(gates=gates,operating=operating,fit_ids=sorted(fit_ids),calibration_ids=sorted(cal_ids),model_sha256=collection['model_sha256'],
                scope='Fit weights on gate_fit only; thresholds on disjoint gate_calibration only; no test read.',
                label='next native token equals forecast, matches gold BPE alignment, and belongs to a final native word correctly aligned to the reference')
    (OUT/'gates.json').write_text(json.dumps(result,indent=2)+'\n')


def bootstrap(records,repetitions=1000,by_speaker=False):
    if by_speaker:
        groups={}
        for row in records:
            g=groups.setdefault(row['speaker_id'],dict(gain_sum=0.,n_words=0))
            g['gain_sum']+=row['gain_sum'];g['n_words']+=row['n_words']
        records=list(groups.values())
    a=np.array([r['gain_sum'] for r in records]);b=np.array([r['n_words'] for r in records])
    rng=np.random.default_rng(351);values=[]
    for _ in range(repetitions):
        idx=rng.integers(0,len(a),len(a));values.append(a[idx].sum()/max(1,b[idx].sum()))
    return np.percentile(values,[2.5,97.5]).tolist()


def evaluate(cfg):
    assert (OUT/'gates.json').exists(),'Calibrate on dev before reading test'
    learned=json.loads((OUT/'gates.json').read_text());rows=load_events('test');add_scores(rows,learned['gates'])
    assert learned['model_sha256']==json.loads((OUT/'events_test_provenance.json').read_text())['model_sha256'],'Different predictors between dev gate and test'
    expected=sum([read_rows(ROOT/f'data/processed/manifests/{s}.jsonl') for s in cfg['evaluation_splits']],[])
    assert {r['utterance_id'] for r in rows}=={r['utterance_id'] for r in expected}
    base=json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/base['model_dir']/'bpe.model'));pieces=[sp.id_to_piece(i) for i in range(500)]
    # Time a live-size NumPy gate call, rather than treating the gate as free.
    sample=rows[0]['events']['full'][0]['features'];g=learned['gates']['full'];times=[]
    for _ in range(1000):
        start=time.perf_counter();x=(transform(sample)-np.array(g['mu']))/np.array(g['sd']);float(x@np.array(g['w'][:-1])+g['w'][-1]);times.append(time.perf_counter()-start)
    existing=OUT/'results.json'
    gate_seconds=json.loads(existing.read_text())['gate_call_median_seconds'] if existing.exists() else float(np.median(times))
    results={};per_records={}
    for split in cfg['evaluation_splits']:
        selected=[r for r in rows if r['split']==split];nref=sum(r['reference_words'] for r in selected);errs=sum(r['errors'] for r in selected)
        native_tokens=sum(len(r['native']) for r in selected);audio=sum(r['audio_seconds'] for r in selected)
        base_stats={}
        for row in selected:
            timeline=baseline_timeline(row,pieces);words=row['hypothesis'].split()
            base_stats[row['utterance_id']]=(stable_times(timeline,words),screen_rewrites(timeline))
        for kind in KINDS:
            for method in ['raw','gate']:
                th=learned['operating'][kind+'/'+method]['chosen']['threshold']
                records=[];gains=[];total={k:0 for k in ['attempts','wrong','accepted','correct_accepted','retractions']}
                rewrite=base_rewrite=0;deletion_events=deleted_characters=0;extra_seconds=0.;nchunks=0;text_peak=audio_peak=0
                for row in selected:
                    s=policy(row,kind,method,th,pieces,True,gate_seconds)
                    actual=stable_times(s['timeline'],row['hypothesis'].split());baseline,bw=base_stats[row['utterance_id']]
                    gain=[1000*(baseline[i]-actual[i]) for i in row['correct_final_word_positions']]
                    gains+=gain;records.append(dict(id=row['utterance_id'],speaker_id=row['speaker_id'],gain_sum=sum(gain),n_words=len(gain)))
                    for key in total:total[key]+=s[key]
                    rewrite+=screen_rewrites(s['timeline']);base_rewrite+=bw
                    de,dc=character_edits(s['timeline']);deletion_events+=de;deleted_characters+=dc
                    events=row['events'][kind];nchunks+=len(events)
                    extra_seconds+=sum(e['seconds']+(gate_seconds if method=='gate' else 0) for e in events)
                    text_peak=max(text_peak,max(e['text_bytes'] for e in events));audio_peak=max(audio_peak,max(e['audio_kv_bytes'] for e in events))
                key=f'{split}/{kind}/{method}';per_records[key]=records
                results[key]=dict(utterances=len(selected),wer_percent=100*errs/nref,threshold=th,
                                  correct_final_words=len(gains),stable_mean_gain_ms=float(np.mean(gains)),
                                  stable_mean_gain_ci95_ms=bootstrap(records),median_per_word_gain_ms=float(np.median(gains)),
                                  stable_mean_gain_speaker_ci95_ms=bootstrap(records,by_speaker=True),
                                  correct_words_advanced_percent=100*np.mean(np.array(gains)>1),
                                  wrong_forecasts_per_100_native_tokens=100*total['wrong']/native_tokens,
                                  retracted_per_100_native_tokens=100*total['retractions']/native_tokens,
                                  accepted_reference_correct_per_100_native_tokens=100*total['correct_accepted']/native_tokens,
                                  reference_correct_and_kept_precision=100*total['correct_accepted']/max(1,total['attempts']),
                                  screen_rewritten_suffix_per_100_final_words=100*rewrite/sum(len(r['hypothesis'].split()) for r in selected),
                                  baseline_screen_rewritten_suffix_per_100_final_words=100*base_rewrite/sum(len(r['hypothesis'].split()) for r in selected),
                                  character_deletion_events_per_100_final_words=100*deletion_events/sum(len(r['hypothesis'].split()) for r in selected),
                                  deleted_characters_per_100_final_characters=100*deleted_characters/sum(len(r['hypothesis']) for r in selected),
                                  character_deletion_events=deletion_events,deleted_characters=deleted_characters,
                                  extra_forecaster_rtf=extra_seconds/audio,extra_ms_per_chunk=1000*extra_seconds/nchunks,
                                  instrumented_runtime_rtf=(sum(r['processing_seconds'][kind] for r in selected)+(gate_seconds*nchunks if method=='gate' else 0))/audio,
                                  measured_text_state_peak_bytes=text_peak,measured_audio_state_peak_bytes=audio_peak,**total)
                print(key,json.dumps(results[key]),flush=True)
    between={}
    for split in cfg['evaluation_splits']:
        for method in ['raw','gate']:
            full=per_records[f'{split}/full/{method}'];band=per_records[f'{split}/band2/{method}'];delta=[]
            for f,b in zip(full,band):
                assert f['id']==b['id'] and f['n_words']==b['n_words']
                delta.append(dict(gain_sum=b['gain_sum']-f['gain_sum'],n_words=f['n_words'],speaker_id=f['speaker_id']))
            between[f'{split}/{method}']=dict(band_minus_full_gain_ms=sum(r['gain_sum'] for r in delta)/max(1,sum(r['n_words'] for r in delta)),ci95_ms=bootstrap(delta))
    against_native={}
    for split in cfg['evaluation_splits']:
        for kind in ['full','band2']:
            for method in ['raw','gate']:
                baseline=per_records[f'{split}/native/{method}'];own=per_records[f'{split}/{kind}/{method}'];delta=[]
                for n,o in zip(baseline,own):
                    assert n['id']==o['id'] and n['n_words']==o['n_words']
                    delta.append(dict(gain_sum=o['gain_sum']-n['gain_sum'],n_words=o['n_words'],speaker_id=o['speaker_id']))
                against_native[f'{split}/{kind}/{method}']=dict(own_minus_native_control_gain_ms=sum(r['gain_sum'] for r in delta)/max(1,sum(r['n_words'] for r in delta)),ci95_ms=bootstrap(delta),speaker_ci95_ms=bootstrap(delta,by_speaker=True))
    summary=dict(results=results,between_variants=between,against_native_control=against_native,gate_call_median_seconds=gate_seconds,
                 scope='Our trained predictors on frozen Zipformer, full test, dev-only weights and thresholds. Native final transcripts unchanged by construction.',
                 timing='Paired stable-word display gain over eager native decoder on exactly matched reference words; simulated native arrivals plus measured added forecaster/gate CPU work. Common encoder compute and actual UI/network timing are not measured here. Word timestamps are not independent phonetic alignments.',
                 labels='Reference-correct token alignment and native acceptance are both required. Gate calibration budget concerns false introduced forecasts per 100 native final tokens, not final WER.',
                 bootstrap='1000 paired utterance replicates plus a speaker-cluster robustness CI; one training seed and one frozen acoustic checkpoint.')
    (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'paired_word_gains.json').write_text(json.dumps(per_records)+'\n')


def metric_checks():
    pieces=['','', '', '▁CA','T','▁SAT','X']
    row=dict(native=[3,4,5],correct_native_positions=[0,1,2],audio_seconds=1.,hypothesis='CAT SAT',events={'full':[
        dict(available=.32,native_count=1,g=4,seconds=.01,eligible=True,scores={'raw':.99}),
        dict(available=.64,native_count=2,g=5,seconds=.01,eligible=True,scores={'raw':.99}),
        dict(available=.96,native_count=3,g=6,seconds=.01,eligible=True,scores={'raw':.99})]})
    s=policy(row,'full','raw',.9,pieces,True)
    assert s['attempts']==3 and s['wrong']==1 and s['accepted']==2 and s['retractions']==1
    stable=stable_times(s['timeline'],['CAT','SAT']);assert np.allclose(stable,[.33,1.66])
    # No future transcript is shown at the first partial-word event.
    assert s['timeline'][0][1]==['CA']
    # A wrong appended token destroys temporary apparent word stability; a
    # retrospective metric must wait until it is really removed again.
    assert stable_times([(.1,['CAT']),(.2,['CATX']),(.3,['CAT'])],['CAT'])==[.3]
    assert character_edits([(.1,['CA']),(.2,['CAT']),(.3,['CAT','SAT'])])==(0,0)
    assert character_edits([(.1,['CATX']),(.2,['CAT'])])==(1,1)
    return 'pass'


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','evaluate','check']);a=p.parse_args()
    torch.set_num_threads(1);cfg=json.loads((ROOT/'configs/zipformer_ours.json').read_text())
    assert metric_checks()=='pass'
    if a.stage=='calibrate':calibration(cfg)
    elif a.stage=='evaluate':evaluate(cfg)
    else:print('Causal replay, reference-correct error accounting and stable-word metric checks: PASS')


if __name__=='__main__':main()
