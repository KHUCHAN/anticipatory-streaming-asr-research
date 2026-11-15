# A2: audio-only qualification before renewed FGN comparisons

Status: launched 2026-11-14; training in progress. Training and evaluation remain local in `runs/predict_correct_a2_v1/`. This is a one-seed 100h distillation study, not a new 960h acoustic model or a B/C experiment. Final numerical qualification is not claimed before both phases and held-out evaluation finish.

## Fixed model and data

A2 has **9,597,696 parameters**. A causal kernel-5 stride-4 frontend maps 80-bin fbank to width 256, followed by twelve low-rate residual depthwise-conv blocks (expansion 2, dilations 1/2/4/8 repeated three times, channel LayerNorm, SiLU). A 2048→1024→4096 chunk head consumes all eight final low-rate representations and produces eight 512-dimensional teacher-compatible states. Every output can use the last arrived frame; no later chunk is visible. Bounded raw-audio history is 724 feature frames. The first window consumes 45 features; subsequent hops consume 32 new features with the preserved stride phase.

Reuse immutable training-only feature normalization and encoder-state labels from the previous pilot. All 28,539 train-clean-100 utterances are visited each epoch. Eight-chunk windows sample beginnings (1/8), endings (1/8) and random locations (6/8), with all existing chunks directly supervised and no excluded burn-in. Caches and teacher contexts are loss labels only. There is no recurrent corrected-state prior in A2.

Checkpoint selection uses the previous speaker-separated model-selection role (64 clean + 64 other). Final evaluation uses a separate 400 dev-clean + 400 dev-other role. These dev roles have already been used in this project; they are not a new independent external test. Test splits remain untouched.

## Fixed training budget

- **MAE:** 12 epochs, AdamW learning rate 6e-4, batch 64, 200-step warmup/cosine schedule, weight decay 0.001, gradient clip 5. Select the lowest representation MAE and retain that checkpoint permanently.
- **Representation gate:** the best MAE checkpoint must beat the train-mean constant on the same fixed model-selection windows. If it fails, record the failure and skip KL.
- **KL:** 24 additional epochs, start from the MAE-best model with a fresh AdamW optimizer at learning rate 3e-4. Add native-distribution KL at temperature 2 and blank weight 0.1. Calibrate its coefficient once from the median MAE/KL gradient-norm ratio in the shared audio trunk on 16 training utterances, target ratio 1 and coefficient clipped to [0.001,1]. Ramp the coefficient over two epochs, then keep it fixed. Log separate losses and weighted gradient ratios each epoch.
- Select the KL checkpoint by actual free-running dev-clean WER in the selection role. Choose between MAE-best and KL-best with that same role before final evaluation. No early stop or additional epochs based on final-evaluation WER. GPU driver-allocation guard: 12GiB; nonfinite losses/gradients abort.

The disposable tiny-fit test reduced MAE 0.8404→0.0249 in 128 steps on eight utterances; its weights were discarded. Batch-64 GPU preflight used about 3.18GiB. Initial checks cover chunk-level future isolation, batch/stream equality, stride phase and use of the last arrived frame.

## Complete-cost qualification and native controls

The entry gate is **dev-clean WER <10% and full inference cost ≤50%** of both the same-run teacher pipeline and its previously recorded pipeline. Passing is not teacher-level WER parity. Report encoder/frontend-only and complete costs separately, including fbank, normalization, caches, output head, native decoder and conversion/Python overhead; file IO is additionally reported. CPU batch 1, one thread, two repetitions with reversed order; no microphone-wait or display-latency claim.

Controls are the existing 66M public teacher and the official [English small streaming Zipformer export](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17). Each uses its own native ONNX encoder/decoder/joiner and token table; A2 shares the teacher's native prediction network/joiner. Small-model input is 39 frames versus 45 for teacher/A2, with the same 32-frame hop and eight output frames. Its native availability window is 60ms shorter; do not claim exactly matched lookahead. Eight independent sherpa-onnx decodes match the custom native adapter's tokens and chunk counts.

An initial untrained PyTorch CPU profile was too slow (about 10.8ms/chunk). The same FP32 A2 computation was exported as first/step ONNX graphs without changed weights or quantization. A twelve-chunk random-stream check differed by at most about 1e-6. The untrained ONNX profile was about 1.3ms/chunk, but both early profiles ran during GPU training and are not final qualification. Final inference uses the checked ONNX path for A2 and both controls; original PyTorch training/selection sources remain frozen. Exports and further real-audio token checks are recorded for each final checkpoint before held-out decoding.

## Run and inspect

```bash
.venv/bin/python -m pip install --target .local/a2-export-deps -r requirements-a2-export.txt
.venv/bin/python experiments/predict_correct_a2_v1/prepare_small.py
.local/venv-baseline/bin/python experiments/predict_correct_a2_v1/native_reference.py
.venv/bin/python experiments/predict_correct_a2_v1/check_native.py
.venv/bin/python experiments/predict_correct_a2_v1/check.py --tiny-fit
.venv/bin/python experiments/predict_correct_a2_v1/run.py
```

Training needs approved MPS access in the desktop sandbox. The native-reference audit uses the existing independent baseline environment. Feature/teacher caches and role manifests must first be prepared by the documented preceding studies. Input hashes, checkpoint hashes, epoch logs and raw transcripts remain preserved; public artifacts contain aggregate evidence only. The ONNX exporter dependencies are isolated from the active training environment.
