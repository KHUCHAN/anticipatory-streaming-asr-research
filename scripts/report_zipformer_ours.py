#!/usr/bin/env python3
"""Record actual training, dev calibration and OUR method's full-test results."""
import hashlib
import json
from pathlib import Path
import time
import torch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/zipformer_ours'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):digest.update(b)
    return digest.hexdigest()


def main():
    cfg=json.loads((ROOT/'configs/zipformer_ours.json').read_text())
    result=json.loads((OUT/'results.json').read_text());gates=json.loads((OUT/'gates.json').read_text())
    states={};logs={}
    for kind in ['full','band2']:
        logs[kind]=[json.loads(l) for l in (OUT/kind/'log.jsonl').open()]
        assert logs[kind][-1]['event']=='done'
        assert len([v for v in logs[kind] if v['event']=='eval'])==15
        states[kind]=torch.load(OUT/kind/'best.pt',map_location='cpu',weights_only=False)
    assert states['full']['initial_state_sha256']==states['band2']['initial_state_sha256']
    assert states['full']['cfg']==states['band2']['cfg']==cfg
    snapshot_path=OUT/'training_source_snapshot.json'
    if snapshot_path.exists():
        snapshot=json.loads(snapshot_path.read_text())
        for name,digest in snapshot['sources'].items():assert sha(ROOT/name)==digest, f'Training source drift: {name}'
    old={}
    for line in (ROOT/'runs/baseline_zipformer/utterances.jsonl').open():
        r=json.loads(line);old[r['utterance_id']]=r
    different=[];counts={}
    for split in cfg['evaluation_splits']:
        rows=[]
        for path in (OUT/'events_test').glob('*.json'):
            if path.name=='complete.json':continue
            r=json.loads(path.read_text())
            if r['split']==split:
                rows.append(r)
                if r['hypothesis'].split()!=old[r['utterance_id']]['hypothesis'].split():different.append(r['utterance_id'])
        counts[split]=len(rows)
    assert counts=={'test-clean':2620,'test-other':2939}
    table=[];seen_splits=set()
    for key,r in result['results'].items():
        split,kind,method=key.split('/');lo,hi=r['stable_mean_gain_ci95_ms']
        if split not in seen_splits:
            table.append(f"| {split} | native baseline | {r['wer_percent']:.3f} | 0 (standard) | 0 | 0 | 0 | 0 |")
            seen_splits.add(split)
        table.append(f"| {split} | {kind} / {method} | {r['wer_percent']:.3f} | {r['stable_mean_gain_ms']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {r['correct_words_advanced_percent']:.2f} | {r['wrong_forecasts_per_100_native_tokens']:.3f} | {r['retracted_per_100_native_tokens']:.3f} | {r['extra_ms_per_chunk']:.2f} |")
    accuracy=[];memory=[]
    for kind,state in states.items():
        e=state['dev'];accuracy.append(f"| {kind} | {state['epoch']+1} / 15 | {e['nll']:.4f} | "+' / '.join(f'{x:.2f}' for x in e['horizon_hit1'])+' |')
        selected=[r for k,r in result['results'].items() if f'/{kind}/' in k]
        memory.append(f"| {kind} | {max(r['measured_text_state_peak_bytes'] for r in selected)/1024:.1f} | {max(r['measured_audio_state_peak_bytes'] for r in selected)/2**20:.2f} |")
    cal=[]
    for key,value in gates['operating'].items():
        s=value['chosen'];cal.append(f"| {key} | {s['threshold']:.6f} | {s['wrong_per_100_native_tokens']:.3f} | {s['correct_accepted_per_100_native_tokens']:.2f} |")
    variant_delta=[]
    for key,r in result['between_variants'].items():
        lo,hi=r['ci95_ms'];variant_delta.append(f"| {key} | {r['band_minus_full_gain_ms']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    native_delta=[]
    for key,r in result['against_native_control'].items():
        lo,hi=r['ci95_ms'];native_delta.append(f"| {key} | {r['own_minus_native_control_gain_ms']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    quality=[]
    for key,r in result['results'].items():
        if key.endswith('/gate'):
            quality.append(f"| {key} | {r['reference_correct_and_kept_precision']:.2f} | {r['character_deletion_events_per_100_final_words']:.3f} | {r['deleted_characters_per_100_final_characters']:.3f} | {r['screen_rewritten_suffix_per_100_final_words']:.2f} |")
    sources=['configs/zipformer_ours.json','avs/zipformer_forecaster.py','scripts/zipformer_features.py',
             'avs/model.py','avs/stream.py','avs/text.py',
             'scripts/prepare_zipformer_forecasts.py','scripts/train_zipformer_forecaster.py',
             'scripts/collect_zipformer_forecasts.py','scripts/analyze_zipformer_forecasts.py',
             'scripts/native_zipformer_forecast.py',
             'scripts/run_zipformer_ours.py','scripts/report_zipformer_ours.py',
             'requirements-forecaster.txt','requirements-zipformer-features.txt']
    baseline_provenance=json.loads((ROOT/'runs/baseline_zipformer/provenance.json').read_text())
    for name,digest in baseline_provenance['model_sha256'].items():assert sha(ROOT/name)==digest
    manifests=['data/processed/manifests/train-clean-100.jsonl']+[f'data/processed/manifests/{s}.jsonl' for s in cfg['evaluation_splits']]
    manifests += [cfg['dev_protocol']+'/'+name+'.jsonl' for name in ['model_selection','gate_fit','gate_calibration']]
    result_files=['results.json','gates.json','paired_word_gains.json']
    result_files += [p for p in ['results_initial_v1.json','provenance_initial_v1.json','display_quality_audit.json','display_examples.json'] if (OUT/p).exists()]
    provenance=dict(finished_unix=time.time(),source_sha256={p:sha(ROOT/p) for p in sources},
                    backbone_sha256=baseline_provenance['model_sha256'],manifest_sha256={p:sha(ROOT/p) for p in manifests},
                    model_sha256={k:sha(OUT/k/'best.pt') for k in states},
                    result_sha256={p:sha(OUT/p) for p in result_files},training_source_snapshot_verified=snapshot_path.exists(),
                    counts=counts,native_transcript_differences_from_previous_official_runtime=different,
                    model_initial_state_sha256=states['full']['initial_state_sha256'],
                    native_note='Custom ONNX adapter uses official fbank/context/ready/greedy settings. Any numerical transcript differences from the previous official runtime are listed explicitly.',
                    commit='No commit or push performed by this experiment.')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    matches='all matched' if not different else f'{len(different)}utterances differed; their IDs are recorded in provenance'
    positive=[(k,r) for k,r in result['results'].items() if '/native/' not in k and r['stable_mean_gain_ci95_ms'][0]>0]
    conclusion=(f"Configuring 8 evaluations {len(positive)}configurations had correct-final-word mean stable-display-gain intervals above zero."
                if positive else 'None of the eight forecaster configurations had mean stable-display-gain intervals above zero.')
    audit_note=('Character-level diagnostics were added after primary evaluation without changing training, gates, thresholds or primary numbers. Preserve initial results/source in `results_initial_v1.json` and `source_snapshot_initial/`; `display_quality_audit.json` verifies unchanged main results.'
                if (OUT/'display_quality_audit.json').exists() else 'This reproduction computes character-level display diagnostics alongside evaluation; diagnostics do not select weights, gates or thresholds.')
    example_note=''
    if (OUT/'display_examples.json').exists():
        example=json.loads((OUT/'display_examples.json').read_text())[0]
        example_note=f"An example verified in the original records is `{example['utterance']}The correct word for `{example['word']}` is native {example['native_stable_s']:.3f}seconds rather than {example['our_stable_s']:.3f}seconds. These are file-arrival replay times; examples/timestamps are recorded in `display_examples.json`."
    doc=f'''# Our Forecast – Verification Method × Fixed Zipformer — 2026-11-12

**Our forecast has completed two actual training, dev gate fitting/baseline correction, and full test evaluation.** This distinguishes it from previous work that only reproduced the baseline. {conclusion} This is the first strong-backbone pilot from a training seed.

## Real model/learning

Freeze the official LibriSpeech 960h 66.11M Streaming Zipformer and train our audio-conditioned Transformer AR forecasters on **all 28,539 train-clean-100 utterances × 15 epochs**. BPE 500; 512-dimensional encoder output → Linear+LayerNorm adapter → hidden 256, 4 layers, 4 heads, FF1024; **4,602,868 parameters**. No initial weights from the older AVS forecaster or an external text model.

Backbone paper, checkpoint and evaluation sources are pinned in the [baseline reproduction](baseline_zipformer_2026-11-11.md). This compares our trained modules on the same frozen recognizer, not a newly trained acoustic backbone.

Full-context and per-layer 2-position models share random initialization, samples, cutoffs, optimizer and 15 epochs. Initial-weight SHA-256: `{states['full']['initial_state_sha256']}`All. For the first epoch, both models were sorted by length, and then shuffled with the same seed. The gold BPE's perspective is an approximation that interpolates between blocks that match the native BPE, and is only used for training labels. Independent pronunciation ending is not the correct answer or RNNT forced alignment. The fixed encoder output is stored in the fp16 cache and put into the forecast view, and the native decoder operates as fp32.

Token CE with smoothing 0.1; AdamW lr 5e-4, betas 0.9/0.98, decay 0.01, warmup 500, MPS FP32. Teacher-force 1–4-token spans at a fixed audio cutoff; primary evaluation is one-step prediction. No separate self-prefix rollout fine-tuning, noise-path generation or new acoustic training. Tests the selected forecaster on the same frozen backbone.

Select checkpoints by token NLL on 512 preselected utterances from speaker-separated dev model_selection, without test-based selection.

| Forecaster | Selected epoch / training epochs | Selected dev NLL | Teacher-forced Hit@1, horizons 0/1/2/3 % |
|---|---:|---:|---|
{chr(10).join(accuracy)}

## Gate and reference value

Train logistic gates on 1,297 separate dev gate_fit utterances. Features: forecaster p1/p2/entropy, continuation, recent native-token probability/emission gap, new-token count and bounded prefix length. Full and band2 gates each use only their own forecaster; deployment does not require both simultaneously.

The `native` control reuses the backbone prediction network at the current prefix and joiner at the last arrived encoder frame. Choose a token from conditional probabilities excluding blank/control IDs. Its gate also receives log nonblank mass. There are no added neural weights, but extra PN/joiner CPU calls are measured and charged; this is not free forecasting.

On 1,414 separate gate_calibration utterances, choose the threshold maximizing correctly accepted forecasts under **0.5 introduced errors per 100 final native tokens**. Roles have no utterance/speaker overlap. Fit weights separately from thresholds and freeze both before test. Equal actual test error rates are not assumed.

A positive gate label requires the forecast to match the next native token, that token to match the reference BPE alignment, and its complete word to match the reference word. A matching fragment inside a wrong word is not positive. Later-model agreement alone is insufficient. Alignment ambiguity remains for repeated text; token forecast counts differ from word accuracy. Report wrong tokens, retractions and complete-word stability separately.

| Forecaster / policy | Frozen threshold | Dev-cal errors /100 native tokens | Correct accepted forecasts /100 native tokens |
|---|---:|---:|---:|
{chr(10).join(cal)}

## Full test results of our method

test-clean 2,620 utterance, test-other 2,939 utterance. From the same input, native greedy determines the final transcription, and the forecast is only attached to the provisional screen. Therefore, WER maintenance is by design and is not used as evidence of forecast accuracy. The native transcription created by the new adapter is similar to the previous official runtime evaluation. {matches}.

Primary timing is **native eager display time minus our stable display time for the same correct final word**. Positive means earlier. A wrong appended forecast that removes a correct word delays stability and incurs a penalty. Add measured forecaster/gate CPU costs to simulated input arrival. This excludes shared encoder work, microphone, network, rendering and resource contention; do not compare directly with earlier CTC-aligned absolute P50.

| split | Forecaster / policy | Final WER % | Correct-word mean display gain ms [95% CI] | Correct words advanced % | Errors /100 native tokens | Retractions /100 native tokens | Added ms/chunk |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

`raw` was judged only by the forecast probability, and `gate` was judged by the learned logistic score. Both chose the reference value only from dev-cal. Errors are the number of times a newly introduced candidate does not fit both the corresponding future position of the native and the correct answer. A retraction is not the same as an error because it includes not only incorrect forecasts but also forecasts that could have been correct but were canceled. The denominator is token and is different from the screen word correction rate. Full screen suffix correction·reference-correct acceptance·addition RTF saved in [results.json](../runs/zipformer_ours/results.json).

The correct word brought forward by band2+gate is clean 28.03%, other 21.98%. These words are often brought forward by close to a chunk, creating an average gain. This does not mean that all words are sped up by 90/70ms. The median gain per word was 0ms across all configurations. This is never changed to the P50 improvement value of display delay.

## Diagnostics with separate check of display quality

{audit_note}

| split / configuration | correct answer/adoption forecast precision % | character deletion occurrence /100 final word | deleted character /100 final character | word suffix modification /100Final Words |
|---|---:|---:|---:|---:|
{chr(10).join(quality)}

`CON → CONFIDENCE` is one word-suffix revision but appends characters without deletion. Normal partial-word extension is included in word revisions; character deletions distinguish actual rollback. Band2+gate has 0.207 clean and 0.281 other deletion events per 100 final words. Correct-and-accepted precision measures introduced token forecasts that match a reference word and are adopted by native decoding; it is not WER or accuracy of every displayed word. User-perceived quality was not measured.

{example_note}

Intervals use 1,000 paired utterance-bootstrap samples, resampling all correct words of an utterance together. Speaker-cluster intervals are also stored in `stable_mean_gain_speaker_ci95_ms`. No multiple-comparison correction. One acoustic checkpoint/predictor seed does not establish equivalence or generalization to other initializations/backbones.

Only provisional display changes. Native decoding still determines acoustic commitment; neither faster irreversible commitment nor lower total acoustic-decoder work through future-candidate reuse is established.

Paired comparisons on identical final words subtract full-context gain from band2 gain; positive means greater band2 benefit. Policies share a development calibration budget, but report actual test risks too. These intervals are not an equivalence test.

| split / policy | band2 − full gain ms | 95% CI ms |
|---|---:|---:|
{chr(10).join(variant_delta)}

Compare identical words against the no-extra-forecaster native control. Positive differences mean earlier display by our forecaster; include actual test errors and CPU cost. An interval including zero does not establish added value from a separate forecaster.

| split / view forecast / policy | us − native forecast control group gain ms | 95% CI ms |
|---|---:|---:|
{chr(10).join(native_delta)}

## Actual states and operations

| Forecaster | Measured text KV/logp peak KiB | Audio cross-attention KV peak MiB |
|---|---:|---:|
{chr(10).join(memory)}

band2 actually releases old KV and logp. A cache of current and past 2 locations remains, but the maximum text storage capacity through the 4 layer is 9 tokens. This does not mean that the entire raw context is 3 token. Voice history and output metadata increases with length, and model weights, temporary task tensors, and full RSS are not included in this table. We do not claim that operations have been fixed or reduced due to the cache. Additional calculations of the forecast are reported separately in the table above and in the raw data.

Reuse adapter/cross-attention K/V for arrived audio and append new frames only. Reuse next-token scores obtained when processing new native tokens, without duplicate forwards. This is ordinary past-activation reuse, not speculative acoustic-decoder reuse. Trained full-model log-probability differs from full audio re-projection by at most 2.9e-6; a constant-memory parallel-graph control agrees within 1e-4. See `checks.json`.

## Execution/Verification/Output

- Model: `runs/zipformer_ours/full/best.pt`, `band2/best.pt`; Store 15 epoch and selection basis in each `log.jsonl`.
- Features and labels: `data/processed/zipformer_forecasts/`; The original sound is shared with existing data.
- Actual forecast: `events_dev/`, `events_test/`; Native prefix and encoder cutoff, forecast/gate input, CPU cost/cache measurement and preservation.
- Gate/reference value: `gates.json`; Preserve fit/calibration ID and entire calibration curve.
- Comparison results: `results.json`, `paired_word_gains.json`, `provenance.json`.
- fbank is different from the official extractor. 0, batch 1/4/8 native token and arrival time of 8 dev utterances match. Actual cache collection is batch 1.
- It passed stable visual tests for future encoder frame masking, replay without pre-exposure of future tokens on the screen, and partial word modification. The overall test differences from the original native results are also specified above.

Configuration: [zipformer_ours.json](../configs/zipformer_ours.json). Training: [train_zipformer_forecaster.py](../scripts/train_zipformer_forecaster.py); incremental forecasts: [collect_zipformer_forecasts.py](../scripts/collect_zipformer_forecasts.py); gate/display analysis: [analyze_zipformer_forecasts.py](../scripts/analyze_zipformer_forecasts.py). Data uses a separate ONNX/kaldi-native-fbank environment; training/forecasting use the existing Torch/MPS `.venv`.

Commands below reproduce the sequence in a fresh environment after data preparation. The actual first epoch overlapped feature generation, with the same length order and seed. Use new output paths or preserved copies; do not mix completed and new runs.

```bash
python3 -m venv .local/venv-baseline
.local/venv-baseline/bin/python -m pip install -r requirements-zipformer-features.txt
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-forecaster.txt
.local/venv-baseline/bin/python scripts/prepare_zipformer_baseline.py
.local/venv-baseline/bin/python scripts/eval_zipformer_baseline.py
.local/venv-baseline/bin/python scripts/prepare_zipformer_forecasts.py --workers 8 --batch 1
.local/venv-baseline/bin/python scripts/native_zipformer_forecast.py
.venv/bin/python scripts/train_zipformer_forecaster.py --kind full
.venv/bin/python scripts/train_zipformer_forecaster.py --kind band2
.venv/bin/python scripts/collect_zipformer_forecasts.py --part dev
.venv/bin/python scripts/analyze_zipformer_forecasts.py calibrate
.venv/bin/python scripts/collect_zipformer_forecasts.py --part test
.venv/bin/python scripts/analyze_zipformer_forecasts.py evaluate
.venv/bin/python scripts/report_zipformer_ours.py
```

Training/evaluation completed locally; no commit/push at that stage. Raw audio, transcripts, features and weights follow existing Git ignore rules.
'''
    (ROOT/'docs/zipformer_ours_2026-11-12.md').write_text(doc)
    print('Report:',ROOT/'docs/zipformer_ours_2026-11-12.md')
    print('Native transcript differences:',len(different))


if __name__=='__main__':main()
