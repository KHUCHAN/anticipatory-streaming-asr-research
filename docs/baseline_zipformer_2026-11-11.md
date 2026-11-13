# Streaming Zipformer baseline reproduction — 2026-11-11

After the four follow-ups, the first external baseline was actually run. This reproduces **evaluation of public LibriSpeech 960h weights**, not new local 960h training or our forecaster's effect. Existing 100h AVS results, models and tokenizer are preserved.

## Thesis and implementation

- Paper: [Zipformer: A faster and better encoder for automatic speech recognition, ICLR 2024](https://arxiv.org/abs/2310.11230).
- Direct reproduction target: [icefall official normal-scale streaming result table](https://github.com/k2-fsa/icefall/blob/179b5eef823e8c2c38a7972f4762e22c5dd23973/egs/librispeech/ASR/RESULTS.md), 66,110,931 parameters, 320ms chunk, chunk-wise greedy. This does not mean that all tables in the paper or the latest SOTA have been reproduced.
- [Original public checkpoint](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17) and [Official ONNX conversion and distribution description](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html).
- FP32 encoder·decoder·joiner was used. The INT8 file was not used. Distribution archive SHA-256: `639e25b578e9e997131402199419c13a941f8e4e198e2da1ce57dbf5cf401282`.
- The upstream document specifies the connection between the original model and the ONNX distribution. We did not perform a numerical consistency test directly comparing the original PyTorch and ONNX. The export file name `epoch-99-avg-1` is a distribution alias for a prepared checkpoint and is not evidence of local 99 epoch learning. The original condition of the comparison result table is `epoch 30 / avg 8`.

## frozen execution conditions

Settings are in [configs/zipformer_baseline.json](../configs/zipformer_baseline.json). Before full test, predeclare **absolute WER difference within 0.10 percentage points** as the initial reproduction tolerance. No test-based adjustment of weights, tokenizer, search, features or margin.

| Item | Executed value |
|---|---|
| Audio / features | Original 16kHz mono FLAC, 80-bin fbank, dither 0, low/high frequency 20/−400, snip_edges=false |
| tokenizer / decoder context | official BPE 500 / recent 2 token prediction network |
| Precision / runtime | FP32 / sherpa-onnx 2.0.1, CPUExecutionProvider |
| Search | Greedy, no external LM, no endpoint reset |
| Input delivery | Audio arrives in 20ms packets; actual incremental decoding whenever ready |
| encoder chunk | icefall 16 unit = 320ms; ONNX feature `decode_chunk_len=32`, input T=45 |
| Ending | Sequentially supply 0.66 seconds of silence after real audio, then input_finished |
| Evaluation | Full test-clean, 2,620 utterances, plus test-other, 2,939 utterances |
| Device | Mac M5 Pro 24GB, Python 3.13.2, arm64, 4 CPU workers × 1 thread |

320ms is the encoder chunk length that advances the input. Since there is an input window for the 45 feature frame and a wait for feature extraction, this is not interpreted as the user's word delay. In utterances with sufficient original sound length, the first decode-ready input time was [0.48] seconds. The input delivery clock is simulated and not the actual microphone, network, or screen clock.

dev-clean I checked loading, recognition, and recording with the first 30 utterance, and the WER was 3.461%. This small sample was not used to estimate model performance or select settings. In a separate dev 10 utterance, even if the original sound was delivered as the entire 20ms/320ms/ packet, the final token and native timestamps were all the same. Even if the voice was changed to noise after 1.5 seconds of 3 utterance, the output, score, and time arrived before that were the same. The WER distance calculation was compared with 500 independent calculations including empty input and repeated words. The execution code is [check_zipformer_baseline.py](../scripts/check_zipformer_baseline.py).

## Full test results

WER is the sum of word editing errors / the sum of correct words for all utterances. In addition to capitalizing and organizing spaces, no number or punctuation substitution or sample exclusion was done.

| split | utterance | speech time | formula WER % | measurement WER % | difference %p | pre-accepted range |
|---|---:|---:|---:|---:|---:|---|
| test-clean | 2,620 | 5.403 | 3.06 | 3.058 | -0.002 | Passed |
| test-other | 2,939 | 5.342 | 7.79 | 7.820 | +0.030 | Passed |

The difference in +0.030%p compared to other's official table remains and the cause has not been specified. Passing the tolerance range is not expressed as a numerical exact match. No direct numerical comparison with the original PyTorch or test-based configuration exploration was performed.

Original outputs: `runs/baseline_zipformer/summary.json` (local-only), `runs/baseline_zipformer/utterances.jsonl` (local-only). Preserve transcripts, edit errors, tokens, logp, native timestamps and chunk arrivals. All 5,559 IDs exactly match the original manifests, with zero duplicates/omissions.

Wall time: 409.2 seconds. Sum of worker processing time/audio time: 0.0417; wall time/audio time: 0.0106. Four CPU workers run concurrently; the latter is throughput, not single-stream RTF. The former includes feature extraction, neural decoding and event logging but excludes loading/initialization. A 30-dev-utterance single-worker smoke gave RTF 0.0300. Differences from older AVS involve input, runtime and size and do not isolate architecture speed.

## Contrast default display policy

This baseline has no extra future-token forecaster. Apply three causal display policies to the same native stream: immediate partial words; hide the last word until another space; or common-word-prefix agreement across two distinct native updates. Final callback displays the same final transcript.

The timing below is **stable display time of a word in the final hypothesis minus its last native token-frame time**. Native timestamps are not independently verified phonetic word ends; do not compare this with CTC-aligned stable latency or perceived end-to-end latency. Compute/rendering time is excluded. A final incorrect word can still count as stable; WER measures accuracy separately.

| split | policy | average ms | P50 ms | P95 ms | rewritten suffix /100Final Words |
|---|---|---:|---:|---:|---:|
| test-clean | Display partial words immediately | 346.8 | 360.0 | 480.0 | 29.88 |
| test-clean | Wait until next word boundary | 600.5 | 600.0 | 880.0 | 0.00 |
| test-clean | Two updates in a row match | 665.7 | 680.0 | 800.0 | 0.03 |
| test-other | Display partial words immediately | 353.2 | 360.0 | 480.0 | 25.45 |
| test-other | Wait until next word boundary | 611.9 | 600.0 | 920.0 | 0.00 |
| test-other | Two updates in a row match | 672.2 | 680.0 | 800.0 | 0.06 |

Rewriting accumulates words removed after the common prefix with the previous screen. It includes growth of an unfinished word and is not a fraction of unique incorrect words. Greedy subword prefixes can remain unchanged while the displayed final word changes. Waiting for a word boundary avoids this category but adds latency. This table does not establish that our forecaster reduces baseline latency.

Code: [analyze_zipformer_baseline.py](../scripts/analyze_zipformer_baseline.py). Small causality checks for partial-word/final-callback future leakage passed. Original record: `runs/baseline_zipformer/display_controls.json` (local-only).

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

Execution provenance: `runs/baseline_zipformer/provenance.json` (local-only), `runs/baseline_zipformer/upstream_sources.json` (local-only), `runs/baseline_zipformer/verification.json` (local-only).
