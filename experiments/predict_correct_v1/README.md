# Fbank + causal conv prediction-correction pilot

Status: completed and verified on 2026-11-14. All three paths trained 16 epochs; 800 dev utterances were decoded with two timing repetitions. [Results](../../docs/predict_correct_pilot_2026-11-14.md): B improves over A on clean, but B-over-C benefit is not established; every cheap path loses full-encoder accuracy. No useful encoder-skipping acceleration is demonstrated. Original raw logs and the reporting-only serialization recovery remain in `runs/predict_correct_v1/`.

The [interpretation review](../../docs/predict_correct_review_2026-11-14.md) leaves general FGN value unresolved in a viable student. Its post-hoc loss-gradient audit is separate from the original training/evaluation; the proposed A2 architecture and small-recognizer evaluation have not run. Original training sources, checkpoints and decode records remain unchanged.

The user authorized three ablations with WER and **complete inference cost** as outcomes:

| Path | Current audio | Additional input |
|---|---|---|
| A | 80-bin fbank + small causal conv | None |
| B | Same audio conv | One frozen deterministic FGN prediction based only on corrected history |
| C | Same audio conv | Previous corrected 8-by-512 state |

The audio network has width 128 and six residual depthwise-convolution blocks, kernel 5, dilations 1/2/4/8/16/32, per-frame channel normalization and SiLU. Its receptive field is bounded at 253 fbank frames. Each chunk consumes 32 new frames after the first 45; outputs use the last eight stride-4 positions, never audio beyond the baseline's available 45-frame window. A/B/C share the same initial audio/head function. B/C prior projections start at zero so the prior can be ignored.

The output features are strictly causal at their sampled positions (16, 20, ..., 44 in the initial window). The full encoder can use its permitted within-chunk context differently. Equal chunk-boundary audio availability therefore does not mean equal per-output receptive fields. This pilot tests this particular cheap causal architecture, rather than all audio correctors.

The prior is projected into the audio hidden representation before a dense output head. This is conditional state reconstruction; it neither forces an explicit forecast-plus-residual output nor selectively recomputes only changed state coordinates.

Frontend parameters are A 745,600, B 7,135,104 (including 6,323,968 frozen FGN parameters), C 811,136. B has greater total capacity/cost and an additional previously trained module; no matched-lifetime-training, matched-total-parameter or iso-compute superiority is claimed. The shared 16-epoch budget covers the new correction training only. Actual inference cost is charged in the WER/runtime frontier comparison.

## Training and state contract

Train all 28,539 train-clean-100 utterances for 16 epochs. One newly randomized six-chunk window per utterance per epoch; two audio/corrected-only burn-in chunks followed by four supervised chunks. Convolution receives its bounded raw-audio prefix. No teacher-state initialization: B/C corrected memory starts at zero at each training window. Feedback is detached between training chunks; full-utterance evaluation carries it throughout the utterance.

Minimize train-normalized encoder-state MAE plus frozen-native-distribution KL (temperature 2, teacher blank-frame weight 0.1). Cached teacher token contexts and encoder states are labels only; they are never model inputs. B retains its existing deterministic checkpoint without updates, executes one forecast step/branch and updates confirmed GRU memory from the **corrected** output. C also retains only corrected outputs. First-chunk prior is zero. Neither B nor C receives a true encoder state at inference.

AdamW learning rate 6e-4, 200-step warmup then cosine schedule with 10% floor, weight decay 0.001, gradient clip 5, batch 64, 10GiB GPU driver-allocation guard. These are a fixed pilot recipe, not a demonstrated optimum. No further same-fixed-case double-descent experiment is included.

## Evaluation and decision rules

Select checkpoints by actual free-running corpus WER on 64 dev-clean + 64 dev-other utterances from the existing model-selection speaker role. Final evaluation uses **400 dev-clean + 400 dev-other**, sampled before training from a separate speaker-disjoint dev role. No test splits. Fixed native decoder generates its own token context; final transcripts can change, so WER is actually measured rather than held fixed.

Run the full encoder baseline and all three cheap paths on identical raw waveforms. CPU batch 1, one thread, two timing repetitions with reversed path order and excluded warmups. Count fbank, normalization, conv caches, prediction, correction, corrected-state memory, native token decoding and glue/copy overhead. Input IO is additionally reported separately. Metric/label computation and one-time model loading are excluded from primary inference time. This is streaming replay compute, not live microphone latency.

Report WER, paired-utterance bootstrap 95% intervals (2,000 replicates), normalized state MAE, complete RTF/ms per chunk, compute P50/P95 and components. A practical B-minus-A gain of at least 0.5 percentage points and beneficial paired CI are positive evidence, not proof of iso-compute superiority. For an encoder-replacement operating point, also require at most 0.5pp WER degradation versus the full encoder and at least 10% total compute savings. Failure to reach significance is not proof of zero effect. Training-seed uncertainty is not measured.

The primary pilot replaces encoder outputs on **every** chunk. It does not implement adaptive fallback or teacher-cache recovery, and it does not execute a full encoder behind the scenes. Even a B-versus-A WER win is insufficient to claim practical acceleration if full-encoder accuracy is not retained.

## Checks and commands

Initial checks cover exact audio streaming, future-feature invariance, training-sequence versus runtime recurrence, own corrected C states, initial-function equality, nonzero gradients into prior projections and unchanged frozen FGN weights. The native frozen bridge reproduces cached baseline tokens in an initial raw-audio audit.

```bash
# Fbank extraction can use the baseline environment; full runner needs MPS.
.local/venv-baseline/bin/python experiments/predict_correct_v1/prepare.py --workers 4
.venv/bin/python experiments/predict_correct_v1/check.py
.venv/bin/python experiments/predict_correct_v1/run.py
.venv/bin/python experiments/predict_correct_v1/report.py
```

`run.py` uses `finalize.py` to serialize NumPy aggregate scalars without changing the frozen evaluator. The initial run finished all eight timed decodes before encountering a NumPy-bool JSON error. Recovery used `finalize.py --finish-existing`, which forbids inference, verifies all raw-record hashes remain unchanged, then completes verification/reporting. Training and timed inference were not repeated.

Use `requirements-corrector.txt`. Local raw outputs/checkpoints and per-utterance transcripts remain under ignored `runs/predict_correct_v1/`; public source, protocol and aggregate results live here. The frozen public teacher's 960h training and the prior FGN checkpoint training are separate from this local 100h student training.
