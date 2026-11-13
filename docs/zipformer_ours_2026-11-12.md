# Our Forecast – Verification Method × Fixed Zipformer — 2026-11-12

**Our forecast has completed two actual training, dev gate fitting/baseline correction, and full test evaluation.** This distinguishes it from previous work that only reproduced the baseline. Among the 8 evaluation configurations, the average stable display gain CI of the correct final word in 8 was greater than 0. This is the first strong-backbone pilot from a training seed.

## Real model/learning

Freeze the official LibriSpeech 960h 66.11M Streaming Zipformer and train our audio-conditioned Transformer AR forecasters on **all 28,539 train-clean-100 utterances × 15 epochs**. BPE 500; 512-dimensional encoder output → Linear+LayerNorm adapter → hidden 256, 4 layers, 4 heads, FF1024; **4,602,868 parameters**. No initial weights from the older AVS forecaster or an external text model.

Backbone paper, checkpoint and evaluation sources are pinned in the [baseline reproduction](baseline_zipformer_2026-11-11.md). This compares our trained modules on the same frozen recognizer, not a newly trained acoustic backbone.

Full-context and layer-band2 share random initialization, samples, cutoffs, optimizer and 15 epochs. Initial-weight SHA-256: `c1731f07ce03a7677424e192a7bbbe0929cc58bb600612a6004a8a39d389655f`. Both first epochs use length order, followed by same-seed shuffles. Gold-BPE timing is interpolated between matching native-BPE blocks for training labels only, not independent phonetic ends or RNNT forced alignment. Cached encoder features are fp16; native decoding is fp32.

Token CE with smoothing 0.1; AdamW lr 5e-4, betas 0.9/0.98, decay 0.01, warmup 500, MPS FP32. Teacher-force 1–4-token spans at a fixed audio cutoff; primary evaluation is one-step prediction. No separate self-prefix rollout fine-tuning, noise-path generation or new acoustic training. Tests the selected forecaster on the same frozen backbone.

Select checkpoints by token NLL on 512 preselected utterances from speaker-separated dev model_selection, without test-based selection.

| Forecaster | Selected epoch / training epochs | Selected dev NLL | Teacher-forced Hit@1, horizons 0/1/2/3 % |
|---|---:|---:|---|
| full | 15 / 15 | 1.9169 | 79.40 / 54.41 / 38.40 / 33.31 |
| band2 | 15 / 15 | 1.9150 | 79.51 / 53.94 / 38.37 / 33.72 |

## Gate and reference value

Train logistic gates on 1,297 separate dev gate_fit utterances. Features: forecaster p1/p2/entropy, continuation, recent native-token probability/emission gap, new-token count and bounded prefix length. Full and band2 gates each use only their own forecaster; deployment does not require both simultaneously.

The `native` control reuses the backbone prediction network at the current prefix and joiner at the last arrived encoder frame. Choose a token from conditional probabilities excluding blank/control IDs. Its gate also receives log nonblank mass. There are no added neural weights, but extra PN/joiner CPU calls are measured and charged; this is not free forecasting.

On 1,414 separate gate_calibration utterances, choose the threshold maximizing correctly accepted forecasts under **0.5 introduced errors per 100 final native tokens**. Roles have no utterance/speaker overlap. Fit weights separately from thresholds and freeze both before test. Equal actual test error rates are not assumed.

A positive gate label requires the forecast to match the next native token, that token to match the reference BPE alignment, and its complete word to match the reference word. A matching fragment inside a wrong word is not positive. Later-model agreement alone is insufficient. Alignment ambiguity remains for repeated text; token forecast counts differ from word accuracy. Report wrong tokens, retractions and complete-word stability separately.

| Forecaster / policy | Frozen threshold | Dev-cal errors /100 native tokens | Correct accepted forecasts /100 native tokens |
|---|---:|---:|---:|
| native/raw | 0.998259 | 0.482 | 14.43 |
| native/gate | 0.946639 | 0.480 | 14.97 |
| full/raw | 0.935151 | 0.484 | 16.53 |
| full/gate | 0.959938 | 0.491 | 19.10 |
| band2/raw | 0.934705 | 0.489 | 15.99 |
| band2/gate | 0.960881 | 0.482 | 18.07 |

## Full test results of our method

test-clean 2,620 utterance, test-other 2,939 utterance. From the same input, native greedy determines the final transcription, and the forecast is only attached to the provisional screen. Therefore, WER maintenance is by design and is not used as evidence of forecast accuracy. The native transcription produced by the new adapter was all the same as the previous official runtime evaluation.

Primary timing is **native eager display time minus our stable display time for the same correct final word**. Positive means earlier. A wrong appended forecast that removes a correct word delays stability and incurs a penalty. Add measured forecaster/gate CPU costs to simulated input arrival. This excludes shared encoder work, microphone, network, rendering and resource contention; do not compare directly with earlier CTC-aligned absolute P50.

| split | Forecaster / policy | Final WER % | Correct-word mean display gain ms [95% CI] | Correct words advanced % | Errors /100 native tokens | Retractions /100 native tokens | Added ms/chunk |
|---|---|---:|---:|---:|---:|---:|---:|
| test-clean | native baseline | 3.058 | 0 (baseline) | 0 | 0 | 0 | 0 |
| test-clean | native / raw | 3.058 | +55.87 [+54.67, +57.05] | 17.47 | 0.347 | 0.062 | 0.17 |
| test-clean | native / gate | 3.058 | +59.27 [+58.13, +60.33] | 18.54 | 0.335 | 0.036 | 0.17 |
| test-clean | full / raw | 3.058 | +78.75 [+77.37, +79.94] | 24.62 | 0.378 | 0.084 | 1.54 |
| test-clean | full / gate | 3.058 | +94.31 [+92.89, +95.52] | 29.50 | 0.397 | 0.118 | 1.55 |
| test-clean | band2 / raw | 3.058 | +75.92 [+74.64, +77.28] | 23.74 | 0.375 | 0.095 | 1.42 |
| test-clean | band2 / gate | 3.058 | +89.58 [+88.19, +90.86] | 28.03 | 0.375 | 0.122 | 1.43 |
| test-other | native baseline | 7.820 | 0 (baseline) | 0 | 0 | 0 | 0 |
| test-other | native / raw | 7.820 | +47.67 [+46.52, +48.78] | 14.92 | 0.532 | 0.077 | 0.07 |
| test-other | native / gate | 7.820 | +53.74 [+52.56, +54.83] | 16.82 | 0.507 | 0.058 | 0.07 |
| test-other | full / raw | 7.820 | +63.28 [+61.95, +64.42] | 19.76 | 0.569 | 0.118 | 1.50 |
| test-other | full / gate | 7.820 | +75.31 [+73.92, +76.57] | 23.53 | 0.583 | 0.171 | 1.50 |
| test-other | band2 / raw | 7.820 | +59.76 [+58.49, +60.89] | 18.66 | 0.570 | 0.114 | 1.39 |
| test-other | band2 / gate | 7.820 | +70.28 [+68.83, +71.54] | 21.98 | 0.537 | 0.159 | 1.40 |

`raw` was judged only by the forecast probability, and `gate` was judged by the learned logistic score. Both chose the reference value only from dev-cal. Errors are the number of times a newly introduced candidate does not fit both the corresponding future position of the native and the correct answer. A retraction is not the same as an error because it includes not only incorrect forecasts but also forecasts that could have been correct but were canceled. The denominator is token and is different from the screen word correction rate. Full screen suffix correction·reference-correct acceptance·addition RTF saved in `runs/zipformer_ours/results.json` (local-only).

The correct word brought forward by band2+gate is clean 28.03%, other 21.98%. These words are often brought forward by close to a chunk, creating an average gain. This does not mean that all words are sped up by 90/70ms. The median gain per word was 0ms across all configurations. This is never changed to the P50 improvement value of display delay.

## Diagnostics with separate check of display quality

Character-level diagnostics were added after primary evaluation without changing training, gates, thresholds or primary numbers. Preserve initial results/source in `results_initial_v1.json` and `source_snapshot_initial/`; `display_quality_audit.json` verifies unchanged main results.

| split / configuration | correct answer/adoption forecast precision % | character deletion occurrence /100 final word | deleted character /100 final character | word suffix modification /100Final Words |
|---|---:|---:|---:|---:|
| test-clean/native/gate | 98.05 | 0.068 | 0.022 | 46.11 |
| test-clean/full/gate | 98.07 | 0.209 | 0.100 | 45.85 |
| test-clean/band2/gate | 98.09 | 0.207 | 0.097 | 45.33 |
| test-other/native/gate | 96.56 | 0.105 | 0.040 | 37.96 |
| test-other/full/gate | 96.39 | 0.304 | 0.145 | 36.48 |
| test-other/band2/gate | 96.40 | 0.281 | 0.137 | 35.73 |

`CON → CONFIDENCE` is one word-suffix revision but appends characters without deletion. Normal partial-word extension is included in word revisions; character deletions distinguish actual rollback. Band2+gate has 0.207 clean and 0.281 other deletion events per 100 final words. Correct-and-accepted precision measures introduced token forecasts that match a reference word and are adopted by native decoding; it is not WER or accuracy of every displayed word. User-perceived quality was not measured.

For utterance `1089-134686-0000`, correct word `HE` is stable at 0.801 seconds rather than native 1.120 seconds. These are file-arrival replay times; examples/timestamps are in `display_examples.json`.

Intervals use 1,000 paired utterance-bootstrap samples, resampling all correct words of an utterance together. Speaker-cluster intervals are also stored in `stable_mean_gain_speaker_ci95_ms`. No multiple-comparison correction. One acoustic checkpoint/predictor seed does not establish equivalence or generalization to other initializations/backbones.

Only provisional display changes. Native decoding still determines acoustic commitment; neither faster irreversible commitment nor lower total acoustic-decoder work through future-candidate reuse is established.

Paired comparisons on identical final words subtract full-context gain from band2 gain; positive means greater band2 benefit. Policies share a development calibration budget, but report actual test risks too. These intervals are not an equivalence test.

| split / policy | band2 − full gain ms | 95% CI ms |
|---|---:|---:|
| test-clean/raw | -2.83 | [-3.69, -1.92] |
| test-clean/gate | -4.73 | [-5.57, -3.87] |
| test-other/raw | -3.52 | [-4.33, -2.69] |
| test-other/gate | -5.03 | [-5.92, -4.15] |

Compare identical words against the no-extra-forecaster native control. Positive differences mean earlier display by our forecaster; include actual test errors and CPU cost. An interval including zero does not establish added value from a separate forecaster.

| split / view forecast / policy | us − native forecast control group gain ms | 95% CI ms |
|---|---:|---:|
| test-clean/full/raw | +22.88 | [+21.53, +24.19] |
| test-clean/full/gate | +35.05 | [+33.67, +36.33] |
| test-clean/band2/raw | +20.05 | [+18.67, +21.49] |
| test-clean/band2/gate | +30.32 | [+28.88, +31.70] |
| test-other/full/raw | +15.61 | [+14.39, +16.85] |
| test-other/full/gate | +21.57 | [+20.17, +22.77] |
| test-other/band2/raw | +12.09 | [+10.88, +13.36] |
| test-other/band2/gate | +16.54 | [+15.20, +17.82] |

## Actual states and operations

| Forecaster | Measured text KV/logp peak KiB | Audio cross-attention KV peak MiB |
|---|---:|---:|
| full | 2249.4 | 6.88 |
| band2 | 29.9 | 6.88 |

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
