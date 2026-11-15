# Interpretation review and audio-only qualification plan — 2026-11-14

Follow-up result, 2026-11-15: the user subsequently authorized A2 implementation and training. The [completed A2 report](predict_correct_a2_2026-11-15.md) records 42.30/61.92% clean/other WER at 1.191/1.187ms per chunk: cost qualifies, accuracy does not. The small-model published-level reproduction gap remains unresolved. The review below preserves the earlier proposal; use the [executed protocol](../experiments/predict_correct_a2_v1/README.md) for fixed settings rather than interpreting historical proposal language as current run status.

The completed [A/B/C pilot](predict_correct_pilot_2026-11-14.md) does not settle the general value of FGN prediction. It establishes that this small audio reconstruction recipe cannot replace the teacher encoder at useful accuracy. Keep the reported numbers and paired intervals, but treat the failed audio path as the first problem to resolve. No new training is launched by this review.

## What the existing results support

The trained paths have normalized state MAE of 0.859–0.862 on dev-clean and 0.875–0.879 on dev-other, worse than the training-mean constant baseline (0.793/0.799). Their WER of 69–80% is far from the teacher's 2.93/8.82%. Thus, the current student did not learn an adequate replacement representation. This does not mean it learned no signal: actual WER and teacher-distribution KL improve during training, and global MAE does not measure every decoder-relevant direction equally.

B improves over A on clean by 2.062pp, conditional on the selected checkpoints and sampled utterances. The paired 95% interval is −3.254 to −0.914pp. Both B-minus-C intervals include zero. These are valid fixed-checkpoint utterance-bootstrap results; they do not include initialization, training-seed or checkpoint-selection uncertainty. Adjacent-epoch WER variation motivates repeated training, but does not by itself invalidate those conditional intervals or estimate independent seed variance.

The proper current conclusion is **FGN usefulness remains unresolved in a viable cheap recognizer**. No additional benefit over previous-corrected-state reuse was established in this weak student. A weak audio path also feeds inaccurate, unfamiliar states into a frozen predictor originally trained on teacher states. It is not established that this is the condition in which a prediction prior must help most.

## Loss balance: measured gradients, not only scalar values

The original objective is MAE + KL with both weights set to one. A's fixed-window validation values move from MAE 0.8861 / KL 7.2848 at epoch 1 to MAE 0.8822 / KL 4.1631 at epoch 16. This supports investigating loss balance, but the scalar sizes alone do not determine update pressure.

A new **post-hoc diagnostic**, with zero parameter updates, isolates each original loss on A's best (epoch 15) and last (epoch 16) checkpoints. It uses 16 fixed model-selection utterances, eight per split, in four batches. The median global KL-to-MAE parameter-gradient norm ratio is **27.06 / 27.44**, with batch ranges 23.42–49.86 / 26.76–50.36. The MAE gradients are nonzero; median gradient cosines are approximately +0.38. These snapshots support disproportionate local KL pressure. They do not reconstruct the gradients or Adam moments throughout training, and do not prove a causal diagnosis of the full run.

[Diagnostic code](../scripts/audit_corrector_loss_gradients.py) · [Numeric evidence](../experiments/predict_correct_v1/artifacts/loss_gradient_audit.json). Gradient-based loss balancing is motivated by [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html); that algorithm was not used in the completed pilot.

State MAE/CRPS improvements must not be described as a percentage of acoustic information recovered. In particular, the earlier phrase “less than 10% of state information” is unsupported: relative error reduction is not mutual information. A mean is the squared-error-optimal constant, whereas a median is the absolute-error-optimal constant; a training-only median baseline should also be recorded in an MAE-focused follow-up.

## Confirmed architectural disadvantages

1. The old student samples output j from feature position 16+4j. Although all eight outputs are released together, earlier outputs cannot use the later available frames. A chunk head can let all eight outputs use the whole arrived window through frame 44 without moving the release boundary. This is causal with respect to arrived audio, even though it is not causal at each internal frame index.
2. The residual conv stack runs at 100Hz. Early stride 4 can move most processing to 25Hz and redistribute the budget toward width/depth and chunk mixing. Speed must be profiled: parameter count alone does not establish compute cost.
3. The fixed two-chunk burn-in excludes the first two chunks from all direct training losses. The next data sampler must directly supervise utterance starts as well as later windows, and its sequence/runtime outputs must match.

The existing 2.6ms audio-conv versus 8.6–8.7ms teacher-encoder measurement describes two implementations: PyTorch CPU versus an optimized ONNX export. It is a useful measured operating point, not a lower bound on every cheap corrector or proof that a small Zipformer is the only possible design. Optimized-backend comparisons and complete inference cost are required.

## Proposed A-only qualification

The [proposed configuration](../configs/predict_correct_a2_proposal.json) uses an initial causal stride-4 conv, twelve width-256 low-rate blocks, and a chunk head that consumes the final eight low-rate representations to produce all eight 512-dimensional states. The analytical count is approximately 9.60M parameters; there is no implemented/profiled/trained A2 model yet.

Before a full run, verify chunk-level future-input isolation, batch/stream agreement, stride phase at the 45-frame first window and subsequent 32-frame hops, frame/target alignment, and fit on a fixed tiny subset. Profile batch-1 CPU cost before spending the training budget. Fix phase lengths, maximum steps/time, checkpoint rules and stop checks before launch.

Train representation MAE first. Require improvement over a training-only constant baseline and inspect representation quality before adding KL. Retain the MAE-only checkpoint, then add KL with explicit weighted gradient-norm/cosine monitoring. Do not infer healthy loss balance from matching scalar magnitudes.

The suggested entry gate is **dev-clean WER below 10% and total inference cost at most 50% of the same-run teacher pipeline**. Also show encoder-only cost separately. The original measurements correspond to about 4.60ms/chunk for half the complete clean pipeline, versus 4.32ms for half the teacher encoder alone; these are different denominators. Remeasure both in the new run. Passing this gate would qualify A for further study, not retain the teacher's 2.93% WER or guarantee room for B's additional forecast cost.

Use model-selection data for development, freeze policy before a separate evaluation role, and keep test data untouched. The existing dev roles have already been used in this project and are not a new external test. If the fixed architecture/budget fails, stop that recipe and reassess the budget or representation target; do not claim that all direct fbank correction is impossible.

Only after A qualifies should B/C be revisited with 2–3 seeds, the same new correction-training budget and common data/initialization. Keep C on its own previous corrected states. Report B's extra parameters, pretraining and actual runtime, paired utterance intervals within seed, and variability across independently trained seeds. A prior benefit and useful accuracy-preserving encoder acceleration remain separate questions.

## Small streaming recognizer control

An available candidate is the English **20M streaming Zipformer** [ONNX model](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17), linked from the [official small-model catalog](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/small-online-models.html). Its [source model card](https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small) reports greedy test WER 3.94/9.79%; these are published reference values, not results measured here. No small-model weights were downloaded or evaluated during this review.

Before comparing it, inspect its own graph, tokenizer and native decoder, effective chunk/lookahead and training scope. Decode the same evaluation audio with its native tokenizer/decoder and the same reference normalization. Reuse the cost-accounting definitions, not the 66M model's fixed weights or assumed 320ms input signature. Count fbank, caches, decoder, conversions and IO consistently; compare at matched latency and quality where possible.

The supported display result remains band2+gate's mean correct-word display gain of 89.58/70.28ms over eager native output; its incremental gain over the inexpensive gate is 30.32/16.54ms and median gain is zero. This is provisional-display replay evidence, not inference acceleration or irreversible commitment. Further seed replication and a FastEmit-backbone check are proposed, not completed by this review.
