#!/usr/bin/env python3
"""Build the baseline report from completed evaluation artifacts."""
import json
from pathlib import Path

from eval_zipformer_baseline import ROOT, atomic_json, sha256


def main():
    run = ROOT/'runs/baseline_zipformer'
    result = json.loads((run/'summary.json').read_text())
    assert result['status'] == 'complete'
    display = json.loads((run/'display_controls.json').read_text())
    provenance = json.loads((run/'provenance.json').read_text())
    sources = json.loads((run/'upstream_sources.json').read_text())
    smoke = json.loads((ROOT/'runs/baseline_zipformer_smoke/summary.json').read_text())
    checks = json.loads((ROOT/'runs/baseline_zipformer_checks/streaming_checks.json').read_text())
    assert len(checks['packet_parity']) == 10 and len(checks['suffix_causality']) == 3
    assert checks['word_edit_distance_cases'] == 500
    assert display['input_sha256'] == result['utterances_sha256'] == sha256(run/'utterances.jsonl')
    assert provenance['script_sha256'] == sha256(run/'eval_zipformer_baseline.executed.py')
    expected_counts = {'test-clean':2620, 'test-other':2939}
    rows = [json.loads(line) for line in (run/'utterances.jsonl').open()]
    assert len({row['utterance_id'] for row in rows}) == len(rows)
    split_rows = {}
    for split,count in expected_counts.items():
        manifest = ROOT/f'data/processed/manifests/{split}.jsonl'
        original = [json.loads(line) for line in manifest.open()]
        actual = [row for row in rows if row['split']==split]
        assert len(actual)==len(original)==count
        assert {row['utterance_id'] for row in original} == {row['utterance_id'] for row in actual}
        split_rows[split] = actual
    table = '\n'.join(
        f"| {split} | {s['utterances']:,} | {s['audio_hours']:.3f} | {s['published_wer_percent']:.2f} | {s['wer_percent']:.3f} | {s['difference_percentage_points']:+.3f} | {'Passed' if s['within_prefixed_margin'] else 'Difference requires investigation'} |"
        for split,s in result['splits'].items())
    displays = []
    names = {'eager_subwords':'Immediate partial-word display', 'word_boundary':'Wait for the next word boundary',
             'confirm_two_updates':'Agreement across two consecutive updates'}
    for split in expected_counts:
        for policy in names:
            r = display['results'][f'{split}/{policy}']
            displays.append(f"| {split} | {names[policy]} | {r['stable_delay_relative_to_native_last_token_mean_ms']:.1f} | {r['stable_delay_relative_to_native_last_token_p50_ms']:.1f} | {r['stable_delay_relative_to_native_last_token_p95_ms']:.1f} | {r['rewritten_suffixes_per_100_final_words']:.2f} |")
    c = provenance['config']
    total = result['total']
    revision = sources['upstream_revisions']['icefall_commit']
    first_ready = sorted({round(r['events'][0]['available_s'],4) for r in rows if r['audio_seconds']>.66})
    hashes = {
        'source_sha256':{name:sha256(ROOT/name) for name in [
            'scripts/prepare_zipformer_baseline.py','scripts/eval_zipformer_baseline.py',
            'scripts/analyze_zipformer_baseline.py','scripts/report_zipformer_baseline.py',
            'scripts/check_zipformer_baseline.py',
            'configs/zipformer_baseline.json','requirements-baseline.txt']},
        'result_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [
            run/'summary.json',run/'display_controls.json',run/'provenance.json',
            run/'model_preparation.json',run/'upstream_sources.json',
            run/'eval_zipformer_baseline.executed.py',
            ROOT/'runs/baseline_zipformer_checks/streaming_checks.json']},
        'notes':['Native public-checkpoint evaluation completed; no new acoustic model or forecaster training. No git commit or push performed by this experiment.']}
    atomic_json(run/'verification.json',hashes)
    text = f'''# Streaming Zipformer baseline reproduction — 2026-11-11

After the four follow-ups, the first external baseline was actually run. This reproduces **evaluation of public LibriSpeech 960h weights**, not new local 960h training or our forecaster's effect. Existing 100h AVS results, models and tokenizer are preserved.

## Thesis and implementation

- Paper: [Zipformer: A faster and better encoder for automatic speech recognition, ICLR 2024](https://arxiv.org/abs/2310.11230).
- Direct reproduction target: [icefall official normal-scale streaming result table](https://github.com/k2-fsa/icefall/blob/{revision}/egs/librispeech/ASR/RESULTS.md), 66,110,931 parameters, 320ms chunk, chunk-wise greedy. This does not mean that all tables in the paper or the latest SOTA have been reproduced.
- [Original public checkpoint](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17) and [Official ONNX conversion and distribution description](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html).
- FP32 encoder·decoder·joiner was used. The INT8 file was not used. Distribution archive SHA-256: `639e25b578e9e997131402199419c13a941f8e4e198e2da1ce57dbf5cf401282`.
- The upstream document specifies the connection between the original model and the ONNX distribution. We did not perform a numerical consistency test directly comparing the original PyTorch and ONNX. The export file name `epoch-99-avg-1` is a distribution alias for a prepared checkpoint and is not evidence of local 99 epoch learning. The original condition of the comparison result table is `epoch 30 / avg 8`.

## frozen execution conditions

Settings are in [configs/zipformer_baseline.json](../configs/zipformer_baseline.json). Before full test, predeclare **absolute WER difference within 0.10 percentage points** as the initial reproduction tolerance. No test-based adjustment of weights, tokenizer, search, features or margin.

| Item | Executed value |
|---|---|
| Audio / features | Original 16kHz mono FLAC, 80-bin fbank, dither 0, low/high frequency 20/−400, snip_edges=false |
| tokenizer / decoder context | official BPE 500 / recent 2 token prediction network |
| Precision / runtime | FP32 / sherpa-onnx {provenance['dependencies']['sherpa-onnx']}, CPUExecutionProvider |
| Search | Greedy, no external LM, no endpoint reset |
| Input delivery | Audio arrives in 20ms packets; actual incremental decoding whenever ready |
| encoder chunk | icefall 16 unit = 320ms; ONNX feature `decode_chunk_len=32`, input T=45 |
| Ending | Sequentially supply 0.66 seconds of silence after real audio, then input_finished |
| Evaluation | Full test-clean, 2,620 utterances, plus test-other, 2,939 utterances |
| Device | Mac M5 Pro 24GB, Python {provenance['python']}, arm64, CPU worker {c['workers']}workers × 1 thread |

320ms is the encoder chunk length that advances the input. Since there is an input window for the 45 feature frame and a wait for feature extraction, this is not interpreted as the user's word delay. In utterances with sufficient original sound length, the first decode-ready input time is {first_ready}seconds. Input timing is simulated, not an actual microphone, network or display clock.

dev-clean I checked loading, recognition, and recording with the first 30 utterance, and WER {smoke['splits']['dev-clean']['wer_percent']:.3f}It was %. This small sample was not used to estimate model performance or select settings. In a separate dev 10 utterance, even if the original sound was delivered as the entire 20ms/320ms/ packet, the final token and native timestamps were all the same. Even if the voice was changed to noise after 1.5 seconds of 3 utterance, the output, score, and time arrived before that were the same. The WER distance calculation was compared with 500 independent calculations including empty input and repeated words. The execution code is [check_zipformer_baseline.py](../scripts/check_zipformer_baseline.py).

## Full test results

WER is the sum of word editing errors / the sum of correct words for all utterances. In addition to capitalizing and organizing spaces, no number or punctuation substitution or sample exclusion was done.

| split | utterance | speech time | formula WER % | measurement WER % | difference %p | pre-accepted range |
|---|---:|---:|---:|---:|---:|---|
{table}

The difference in +0.030%p compared to other's official table remains and the cause has not been specified. Passing the tolerance range is not expressed as a numerical exact match. No direct numerical comparison with the original PyTorch or test-based configuration exploration was performed.

Original outputs: [summary.json](../runs/baseline_zipformer/summary.json), [utterances.jsonl](../runs/baseline_zipformer/utterances.jsonl). Preserve transcripts, edit errors, tokens, logp, native timestamps and chunk arrivals. All {total['utterances']:,}utterance IDs exactly match the original manifests, with zero duplicates or omissions.

Total execution wall time was {total['observed_wall_seconds']:.1f}seconds; summed worker processing time / audio time was {total['summed_worker_processing_rtf']:.4f}, wall time / total voice time is {total['observed_wall_rtf']:.4f}with four concurrent CPU workers; do not treat the latter throughput ratio as single-stream RTF. The former includes feature extraction, neural decoding and event recording, excluding file loading/initialization. The corresponding single-worker 30-dev-utterance smoke RTF was {smoke['splits']['dev-clean']['summed_worker_processing_rtf']:.4f}Different input, runtime and model size prevent attributing the difference from AVS RTF to architecture alone.

## Contrast default display policy

This baseline has no extra future-token forecaster. Apply three causal display policies to the same native stream: immediate partial words; hide the last word until another space; or common-word-prefix agreement across two distinct native updates. Final callback displays the same final transcript.

The timing below is **stable display time of a word in the final hypothesis minus its last native token-frame time**. Native timestamps are not independently verified phonetic word ends; do not compare this with CTC-aligned stable latency or perceived end-to-end latency. Compute/rendering time is excluded. A final incorrect word can still count as stable; WER measures accuracy separately.

| split | policy | average ms | P50 ms | P95 ms | rewritten suffix /100Final Words |
|---|---|---:|---:|---:|---:|
{chr(10).join(displays)}

Rewriting accumulates words removed after the common prefix with the previous screen. It includes growth of an unfinished word and is not a fraction of unique incorrect words. Greedy subword prefixes can remain unchanged while the displayed final word changes. Waiting for a word boundary avoids this category but adds latency. This table does not establish that our forecaster reduces baseline latency.

Code: [analyze_zipformer_baseline.py](../scripts/analyze_zipformer_baseline.py). Small causality checks for partial-word/final-callback future leakage passed. Original record: [display_controls.json](../runs/baseline_zipformer/display_controls.json).

## Compare current decision with next method

1. Completed the entire test evaluation and native streaming output recording of the public checkpoint. The official WER was closely reproduced in the split that passed the above pre-allowed range. Distinguish from reproducing a recipe learned from the original.
2. Above the strong 960h baseline, the additional effect of our forecast/gate is **not yet verified**. It is not assumed that the displayed gain of 40–80ms of the existing 100h will be maintained.
3. The existing AVS tokenizer ID system and the 4.47M forecaster cannot be connected as is. The new adapter must match the BPE 500 and 512 dimension encoder outputs of this baseline and learn the full context/layer-specific historical 2 location model from the same frozen baseline with the same initialization and budget. The encoder output of the current archive is a 512-dimensional expression projected toward the joiner, so it is not treated as the same as the existing AVS encoder expression.
4. Forecast learning uses only the original train sound and transcription. Maintain the prepared dev model_selection / gate_fit / gate_calibration roles for each speaker and do not learn the gate or select the threshold through test. Since we have already seen the results of the reproducible test, we do not use the subsequent test as a completely blind set.
5. Native immediate display is included in the control group. Verify whether the forecast is advanced at the same final WER and the same misindicated/corrected level. Reducing native output latency, prediction ahead of actual pronunciation, and computation reuse are reported as separate metrics.
6. This baseline is not an experiment that verified the FastEmit effect. 114M FastConformer scratch and 960h new learning have not started.

## Jaehyun and File

Commands assume original audio and dev-clean/test-clean/test-other manifests in `data/processed/manifests/`, prepared using the [dataset guide](../data/README.md). Baseline preparation does not redownload speech or mix training splits into evaluation.

```bash
python3 -m venv .local/venv-baseline
.local/venv-baseline/bin/python -m pip install -r requirements-baseline.txt
.local/venv-baseline/bin/python scripts/prepare_zipformer_baseline.py
.local/venv-baseline/bin/python scripts/eval_zipformer_baseline.py --splits dev-clean --limit 30 --workers 1 --out runs/baseline_zipformer_smoke
.local/venv-baseline/bin/python scripts/check_zipformer_baseline.py
caffeinate -i .local/venv-baseline/bin/python scripts/eval_zipformer_baseline.py
.local/venv-baseline/bin/python scripts/analyze_zipformer_baseline.py
.local/venv-baseline/bin/python scripts/report_zipformer_baseline.py
```

`caffeinate` is for sleep prevention in macOS and is omitted in other OSs. Re-evaluation uses the new `--out` path. The resume of incomplete output reuses the saved utterance only if the configuration·manifest·script·model hash is the same. Completed execution checks the input, model, and record hash and preserves the existing results and original wall time. After the evaluation, the added protection logic and the original execution code were distinguished, and the code at the time of the actual evaluation was stored in `runs/baseline_zipformer/eval_zipformer_baseline.executed.py` and compared with the script hash of the original provenance.

Weights: `checkpoints/baselines/`. Original outputs/hashes/environment: `runs/baseline_zipformer/`. Smoke and packet checks: `runs/baseline_zipformer_smoke/`, `runs/baseline_zipformer_checks/`. Source tables/snapshots: `.local/baseline_sources/`. Git includes publishable code, configuration and documentation; audio, transcripts, weights, raw logs and environments remain ignored. No commit/push was performed at that experiment's completion.

Execution provenance: [provenance.json](../runs/baseline_zipformer/provenance.json), [upstream_sources.json](../runs/baseline_zipformer/upstream_sources.json), [final hash checks](../runs/baseline_zipformer/verification.json).
'''
    destination = ROOT/'docs/baseline_zipformer_2026-11-11.md'
    destination.write_text(text)
    print(destination)


if __name__ == '__main__':
    main()
