# Fine-tuned preview encoder on the 20M streaming Zipformer

Status: completed on 2026-11-16. Negative result. A copy of the encoder fine-tuned to read partly zero-filled chunks does not beat the unmodified encoder with the same zero fill: delay and rewrites are within a few points of each other at every preview point, and slightly worse after gating. The best policy on this model remains zero fill plus a display gate (458/463ms -> 222/234ms mean stable delay at 1.3/1.8 rewrites per 100 words, dev-clean/dev-other).

Follows [preview_filler_v1](../preview_filler_v1/README.md), which found that predicting future acoustic features does not help. This run tests the alternative: train the encoder itself for partial chunks, with a token-level target.

## What was run

- **Native path**: the public 20M model (39-frame window, 32-frame hop), unmodified ONNX. Final output and WER are unchanged by construction.
- **Preview encoder (`ft`)**: a copy of the encoder and encoder projection, initialised from the public PyTorch weights (`desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small`). The PyTorch copy matches the ONNX encoder to 5e-6 on real chunks. It reads native streaming state and a chunk whose frames from `k` on are 0.0.
- **Loss**: KL from the frozen model's full-chunk joiner distribution to the preview distribution, on the frozen greedy path, 8 frames per chunk; `k` uniform in 7..38.
- **Training**: 150 minutes on an Apple GPU, AdamW 1e-4, 27,739 train-clean-100 utterances (the 800 gate-training utterances are held out).
- **Controls**: `zero` and `repeat` fill through the original encoder. Gates, thresholds, metrics, utterances (200 dev-clean + 200 dev-other) and alignments are those of preview_filler_v1 stage 2. Test unused.
- **Preview points**: `k` = 7, 15, 23 real frames; `k7` is the moment the previous chunk's native pass becomes possible.

## Results

Mean stable delay (ms) / rewrites per 100 words, dev-clean then dev-other. No preview: 458 / 0.0 and 463 / 0.0, WER 3.66/10.62%, RTF 0.015.

| Schedule | Preview | Ungated | Gate R5 | Gate R2 |
|---|---|---|---|---|
| `k7` | zero | 168/10.7, 179/16.5 | 191/3.6, 203/5.2 | 222/1.3, 234/1.8 |
| `k7` | repeat | 252/2.5, 255/5.3 | 252/2.3, 255/4.8 | 256/1.3, 266/2.0 |
| `k7` | ft | 176/10.2, 192/14.1 | 198/3.4, 216/5.3 | 226/1.4, 242/2.4 |
| `k15` | zero | 229/10.0, 241/14.0 | 239/4.6, 250/6.4 | 262/1.1, 272/2.3 |
| `k15` | ft | 236/8.0, 249/11.6 | 243/4.2, 258/6.4 | 258/1.6, 272/3.0 |
| `k23` | zero | 304/7.3, 311/9.2 | 308/3.8, 316/4.9 | 318/1.1, 325/2.2 |
| `k23` | ft | 307/6.6, 318/10.1 | 314/2.8, 325/4.1 | 323/1.1, 334/2.3 |
| `k7_23` | zero | 161/15.4, 171/22.2 | 207/3.8, 217/4.8 | 257/1.4, 266/2.3 |
| `k7_23` | ft | 168/14.5, 185/20.9 | 210/4.2, 227/5.7 | 274/2.8, 291/2.5 |

- Fine-tuning lowered the teacher-forced validation KL (k7 0.207 -> 0.134, k15 0.121 -> 0.083, k23 0.057 -> 0.055) and raised precision (0.76 -> 0.81 at k7) while lowering recall (0.82 -> 0.77). In free-running display this nets out to slightly fewer rewrites and slightly more delay than plain zero fill.
- 86% of `ft` preview token sequences on the evaluation set are identical to the original encoder's zero-fill preview.
- Zero fill with gate R2 is 30/21ms faster than ungated `repeat` at `k7` (95% intervals [-33, -25] / [-26, -17]) with fewer rewrites. One preview pass doubles RTF to 0.030.
- Full numbers: [`artifacts/results.json`](artifacts/results.json).

## Limits

- One training run, one learning rate, one loss; no RNN-T loss and no end-to-end training of the native path. A model trained from the start with short or simulated right context (CUSIDE-style, delay penalty) is not tested here.
- `ft` previews run in PyTorch and are charged the measured ONNX time of the zero-fill preview of the same chunk.
- Reference times come from our CTC forced alignment. Dev role reused; one timing repetition on one CPU thread.

## Reproduce

```bash
cd experiments/preview_finetune_v1
../../.local/venv-fgn-state/bin/python check_torch.py   # PyTorch copy vs ONNX
../../.local/venv-fgn-state/bin/python train.py         # runs/preview_finetune_v1/student.pt
./run_chain.sh                                          # collect.py for train/selection/eval, then evaluate.py
```

Requires the upstream sources and `pretrained.pt` in `.local/20m_torch/` and the alignments from preview_filler_v1.
