"""Dev-only calibration and paired replay against the unchanged old controls."""
import argparse,copy,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
import sentencepiece as spm
from common import ROOT,OUT
from collect import KINDS,rows_for

spec=importlib.util.spec_from_file_location('old_analysis',ROOT/'scripts/analyze_zipformer_forecasts.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
helpers.OUT=OUT
CONTROLS=['native','full','band2'];ALL=CONTROLS+KINDS
CALIBRATION_BUDGET_GRID=[.1,.25,.5,1.,2.]


def load(part,ids=None):
    rows=[]
    for path in sorted((OUT/f'events_{part}').glob('*.json')):
        if path.name=='complete.json' or (ids is not None and path.stem not in ids):continue
        rows.append(json.loads(path.read_text()))
    if ids is not None:assert {r['utterance_id'] for r in rows}==ids
    return rows


def calibrate(cfg):
    protocol=ROOT/cfg['gate_protocol']
    groups={key:[json.loads(s) for s in (protocol/f'{key}.jsonl').open()] for key in ['gate_fit','gate_calibration','model_selection']}
    fitids={r['utterance_id'] for r in groups['gate_fit']};calids={r['utterance_id'] for r in groups['gate_calibration']}
    assert fitids.isdisjoint(calids)
    assert {r['speaker_id'] for r in groups['gate_fit']}.isdisjoint({r['speaker_id'] for r in groups['gate_calibration']})
    assert (fitids|calids).isdisjoint({r['utterance_id'] for r in groups['model_selection']})
    old=json.loads((ROOT/'runs/zipformer_ours/gates.json').read_text())
    gates=copy.deepcopy(old['gates']);operating=copy.deepcopy(old['operating'])
    fit=load('dev',fitids);cal=load('dev',calids)
    for kind in KINDS:
        if any(e['eligible'] for r in fit for e in r['events'][kind]):gates[kind]=helpers.fit_gate(fit,kind)
        else:gates[kind]=dict(mu=[0.]*11,sd=[1.]*11,w=[0.]*11+[-60.],examples=0,positive_fraction=0.)
    helpers.KINDS=KINDS;helpers.add_scores(cal,gates)
    for kind in KINDS:
        for method in ['raw','gate']:
            key=kind+'/'+method
            operating[key]=helpers.choose_threshold(cal,kind,method,cfg['gate_false_forecasts_per_100_tokens'])
            print(key,json.dumps(operating[key]['chosen']),flush=True)
    # Reuse old control weights AND thresholds. Nothing from test is consulted.
    operating_grid={}
    for kind in ALL:
        curve=operating[kind+'/gate']['curve'];points=[]
        for budget in CALIBRATION_BUDGET_GRID:
            valid=[r for r in curve if r['wrong_per_100_native_tokens']<=budget]
            selected=max(valid,key=lambda s:(s['correct_accepted'],-s['wrong'],-s['threshold']))
            points.append(dict(dev_budget=budget,threshold=selected['threshold']))
        operating_grid[kind]=points
    result=dict(gates=gates,operating=operating,operating_grid=operating_grid,fit_ids=sorted(fitids),calibration_ids=sorted(calids),
                model_sha256=json.loads((OUT/'events_dev_provenance.json').read_text())['model_sha256'],
                collection_provenance=json.loads((OUT/'events_dev_provenance.json').read_text()),
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                created_unix=time.time(),controls='Native, full and band2 gates and thresholds reused exactly from the earlier frozen study.',
                scope='New gates fit only on gate_fit; thresholds fixed only on disjoint gate_calibration before test collection.')
    (OUT/'gates.json').write_text(json.dumps(result,indent=2)+'\n')


def evaluate(cfg):
    learned=json.loads((OUT/'gates.json').read_text());rows=load('test')
    assert {r['utterance_id'] for r in rows}=={r['utterance_id'] for r in rows_for('test',cfg)}
    assert learned['model_sha256']==json.loads((OUT/'events_test_provenance.json').read_text())['model_sha256']
    assert learned['collection_provenance']==json.loads((OUT/'events_test_provenance.json').read_text())
    helpers.KINDS=ALL;helpers.add_scores(rows,learned['gates'])
    base=json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/base['model_dir']/'bpe.model'));pieces=[sp.id_to_piece(i) for i in range(500)]
    oldresult=json.loads((ROOT/'runs/zipformer_ours/results.json').read_text())
    gate_seconds=oldresult['gate_call_median_seconds']
    results={};per={};baselines={}
    for row in rows:baselines[row['utterance_id']]=helpers.stable_times(helpers.baseline_timeline(row,pieces),row['hypothesis'].split())
    for split in cfg['evaluation_splits']:
        selected=[r for r in rows if r['split']==split]
        native_tokens=sum(len(r['native']) for r in selected);finalwords=sum(len(r['hypothesis'].split()) for r in selected)
        nref=sum(r['reference_words'] for r in selected);errors=sum(r['errors'] for r in selected);audio=sum(r['audio_seconds'] for r in selected)
        for kind in ALL:
            for method in ['raw','gate']:
                threshold=learned['operating'][kind+'/'+method]['chosen']['threshold']
                records=[];gains=[];counts={k:0 for k in ['attempts','wrong','accepted','correct_accepted','retractions']}
                extra=0.;chunks=0;rewritten=0;deletions=0;confirmed=branch=predicted=audio_projection=retained=0
                for row in selected:
                    p=helpers.policy(row,kind,method,threshold,pieces,True,gate_seconds)
                    actual=helpers.stable_times(p['timeline'],row['hypothesis'].split());baseline=baselines[row['utterance_id']]
                    gain=[1000*(baseline[i]-actual[i]) for i in row['correct_final_word_positions']]
                    gains+=gain;records.append(dict(id=row['utterance_id'],speaker_id=row['speaker_id'],gain_sum=sum(gain),n_words=len(gain),
                                                   wrong=p['wrong'],native_tokens=len(row['native'])))
                    for k in counts:counts[k]+=p[k]
                    rewritten+=helpers.screen_rewrites(p['timeline']);deletions+=helpers.character_edits(p['timeline'])[1]
                    events=row['events'][kind];chunks+=len(events)
                    extra+=sum(e['seconds']+(gate_seconds if method=='gate' else 0) for e in events)
                    confirmed=max(confirmed,max(e.get('confirmed_state_bytes',e['text_bytes']) for e in events))
                    audio_projection=max(audio_projection,max(e['audio_kv_bytes'] for e in events))
                    retained=max(retained,max(e.get('confirmed_state_bytes',e['text_bytes'])+e['audio_kv_bytes'] for e in events))
                    branch=max(branch,max(e.get('speculative_state_bytes',0) for e in events))
                    predicted=max(predicted,max(e.get('predicted_state_tensor_bytes',0) for e in events))
                key=f'{split}/{kind}/{method}';per[key]=records
                results[key]=dict(utterances=len(selected),wer_percent=100*errors/nref,threshold=threshold,
                    correct_final_words=len(gains),stable_mean_gain_ms=float(np.mean(gains)),
                    stable_mean_gain_ci95_ms=helpers.bootstrap(records),stable_mean_gain_speaker_ci95_ms=helpers.bootstrap(records,by_speaker=True),
                    median_per_word_gain_ms=float(np.median(gains)),correct_words_advanced_percent=100*np.mean(np.array(gains)>1),
                    wrong_forecasts_per_100_native_tokens=100*counts['wrong']/native_tokens,
                    retracted_per_100_native_tokens=100*counts['retractions']/native_tokens,
                    accepted_reference_correct_per_100_native_tokens=100*counts['correct_accepted']/native_tokens,
                    reference_correct_and_kept_precision=100*counts['correct_accepted']/max(1,counts['attempts']),
                    screen_rewritten_suffix_per_100_final_words=100*rewritten/finalwords,
                    deleted_characters_per_100_final_characters=100*deletions/sum(len(r['hypothesis']) for r in selected),
                    extra_forecaster_rtf=extra/audio,extra_ms_per_chunk=1000*extra/chunks,
                    confirmed_state_peak_bytes=confirmed,audio_projection_cache_peak_bytes=audio_projection,
                    retained_forecaster_context_peak_bytes=retained,speculative_state_peak_bytes=branch,predicted_state_tensor_peak_bytes=predicted,**counts)
                if kind in KINDS:
                    diagnostics=np.array([e['features'][8:11] for r in selected for e in r['events'][kind]])
                    results[key]['mean_branch_agreement']=float(diagnostics[:,0].mean())
                    results[key]['mean_predictive_mutual_information']=float(diagnostics[:,1].mean())
                    results[key]['mean_future_feature_spread']=float(diagnostics[:,2].mean())
                print(key,json.dumps(results[key]),flush=True)
    comparisons={}
    for split in cfg['evaluation_splits']:
        for kind in KINDS:
            for reference in CONTROLS+['small_ar','small_det','small_fgn','large_ar','large_det']:
                if kind==reference:continue
                a=per[f'{split}/{kind}/gate'];b=per[f'{split}/{reference}/gate'];delta=[];riskdelta=[]
                for own,other in zip(a,b):
                    assert own['id']==other['id'] and own['n_words']==other['n_words']
                    delta.append(dict(speaker_id=own['speaker_id'],gain_sum=own['gain_sum']-other['gain_sum'],n_words=own['n_words']))
                    riskdelta.append(dict(speaker_id=own['speaker_id'],gain_sum=100*(own['wrong']-other['wrong']),n_words=own['native_tokens']))
                comparisons[f'{split}/{kind}-minus-{reference}']=dict(delta_gain_ms=sum(r['gain_sum'] for r in delta)/sum(r['n_words'] for r in delta),
                    ci95_ms=helpers.bootstrap(delta),speaker_ci95_ms=helpers.bootstrap(delta,by_speaker=True),
                    delta_false_forecasts_per_100_native_tokens=sum(r['gain_sum'] for r in riskdelta)/sum(r['n_words'] for r in riskdelta),
                    false_forecast_delta_ci95=helpers.bootstrap(riskdelta),false_forecast_delta_speaker_ci95=helpers.bootstrap(riskdelta,by_speaker=True))
    controlcheck={}
    for split in cfg['evaluation_splits']:
        for kind in CONTROLS:
            for method in ['raw','gate']:
                key=f'{split}/{kind}/{method}';old=oldresult['results'][key];new=results[key]
                delta=abs(old['stable_mean_gain_ms']-new['stable_mean_gain_ms'])
                assert delta<1e-8,(key,delta)
                assert old['wrong_forecasts_per_100_native_tokens']==new['wrong_forecasts_per_100_native_tokens']
                controlcheck[key]='exactly reproduced'
    curves={}
    for split in cfg['evaluation_splits']:
        selected=[r for r in rows if r['split']==split];native_tokens=sum(len(r['native']) for r in selected)
        for kind in ALL:
            points=[]
            for point in learned['operating_grid'][kind]:
                gainsum=0.;nwords=wrong=attempts=retractions=0
                for row in selected:
                    p=helpers.policy(row,kind,'gate',point['threshold'],pieces,True,gate_seconds)
                    times=helpers.stable_times(p['timeline'],row['hypothesis'].split());baseline=baselines[row['utterance_id']]
                    gainsum+=sum(1000*(baseline[i]-times[i]) for i in row['correct_final_word_positions'])
                    nwords+=len(row['correct_final_word_positions']);wrong+=p['wrong'];attempts+=p['attempts'];retractions+=p['retractions']
                points.append(dict(**point,stable_mean_gain_ms=gainsum/nwords,
                                   wrong_forecasts_per_100_native_tokens=100*wrong/native_tokens,
                                   retracted_per_100_native_tokens=100*retractions/native_tokens,attempts=attempts))
            curves[f'{split}/{kind}']=points
    result=dict(results=results,paired_comparisons=comparisons,old_control_reproduction=controlcheck,risk_gain_operating_points=curves,
                gate_call_median_seconds=gate_seconds,scope='Fixed native final outputs; provisional single-token display only. Final WER unchanged by construction.',
                timing='Paired stable-word display gain on exactly reference-matched final words; simulated native arrivals plus measured additional CPU forecaster time, including state updates and speculative decoding. No independent phonetic timestamps, actual UI or live encoder resource contention measured.',
                uncertainty='1000 paired utterance and speaker-cluster bootstrap replicates. One continuation seed37 for all variants. These intervals do not estimate training-seed variance.',
                memory='Live tensor byte counts for band2 text, FGN state and growing arrived-audio KV; excludes backbone, weights, allocator/temporary activations and offline evaluation cache. No claim of bounded whole-system memory.',
                risk='Thresholds meet 0.5 false introduced forecasts/100 native tokens on dev calibration only. Test rates may differ; compare gain AND actual test error rates.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'paired_word_gains.json').write_text(json.dumps(per)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','evaluate','check']);a=p.parse_args()
    torch.set_num_threads(1);cfg=json.loads((ROOT/'experiments/band_fgn_v1/config.json').read_text())
    assert helpers.metric_checks()=='pass'
    if a.stage=='calibrate':calibrate(cfg)
    elif a.stage=='evaluate':evaluate(cfg)
    else:print('Replay and reference-correct labels: PASS')
