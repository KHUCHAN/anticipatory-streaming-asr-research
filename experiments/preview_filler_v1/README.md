# Training-free preview fillers on the frozen streaming teacher

Status: stages 1 and 2 completed on 2026-11-16. One preview pass per chunk lowers the mean reference-anchored stable word delay from 542/551ms to 251-329/264-337ms on dev-clean/dev-other, with unchanged final WER and roughly twice the complete compute. The fillers trade delay against rewrites: zero fill is fastest and rewrites most; last-frame repeat is slower and rewrites least. In stage 2, a display gate on the zero-fill preview gives 303/314ms at 1.4/2.5 rewrites per 100 words; learned future-feature fillers, deterministic or FGN-style, do not beat it.

This is the training-free control required by the [prior-work review](../../docs/prior_work_predict_refine_2026-11-16.md). Early provisional display is established by ZeroPrompt; nothing here is claimed as a new method.

## What was run

The public 66M streaming Zipformer (320ms chunks, 45-frame window, 32-frame hop) is replayed chunk by chunk. Before the native pass of a chunk, a preview pass sees only the first `k` real feature frames; the remaining frames are filled. Preview passes read the native encoder state and decoder context and never write them.

| Item | Setting |
|---|---|
| Utterances | 200 dev-clean + 200 dev-other, fixed-seed subset of the existing 800-utterance dev role; test unused |
| Fillers | `zero` (features set to 0.0), `repeat` (last real frame), `mean` (causal running mean of the utterance) |
| Preview points | `k` = 13, 21, 29, 37 real frames; `k13` is the moment the previous chunk's native pass becomes possible |
| Schedules | none, `k29`, `k13`, `k21_29_37`, `k13_21_29_37`; declared in `config.json` before the run |
| Low-cost variant | `repeat` previews through the published int8 encoder; native pass stays fp32 |
| Reference times | Word end from our CTC forced alignment of the reference text (`facebook/wav2vec2-base-960h`, 20ms frames) |
| Clock | Audio arrival of the last needed frame plus measured compute of each pass, queued on one CPU thread |

## Definitions

- **Stable delay**: earliest display time after which the word at that position never changes, minus the aligned end of the matching reference word. Only words that the final hypothesis gets right are included; `coverage` is their share of reference words. Final output is identical across policies, so coverage is too.
- **Rewrites per 100 final words**: over consecutive displays, a shown word that disappears or is replaced by something it is not a prefix of. A partial word growing into its completion is not a rewrite.
- **RTF**: feature extraction plus every encoder/decoder pass, divided by audio duration.

## Results

Mean stable delay with a 95% utterance-bootstrap interval for the change against no preview (2000 replicates). Full percentiles are in [`artifacts/results.json`](artifacts/results.json).

| Policy | Passes per chunk | dev-clean delay (ms) | dev-other delay (ms) | Change vs none, clean (ms) | Rewrites /100 words (clean / other) | RTF (clean) |
|---|---:|---:|---:|---|---:|---:|
| none | 1 | 542 | 551 | | 0.00 / 0.00 | 0.033 |
| `k29` repeat | 2 | 399 | 410 | -143 [-144, -141] | 0.91 / 1.99 | 0.064 |
| `k13` repeat | 2 | 329 | 337 | -213 [-218, -208] | 1.48 / 4.33 | 0.064 |
| `k13` mean | 2 | 264 | 287 | -277 [-281, -273] | 9.97 / 10.37 | 0.064 |
| `k13` zero | 2 | 251 | 264 | -291 [-293, -289] | 18.31 / 20.19 | 0.064 |
| `k13_21_29_37` repeat | 5 | 282 | 293 | -259 [-262, -257] | 4.08 / 9.31 | 0.157 |
| `k13_21_29_37` repeat, int8 previews | 5 | 280 | 291 | -261 [-264, -258] | 4.29 / 10.17 | 0.132 |
| `k13_21_29_37` zero | 5 | 246 | 259 | | 27.32 / 31.02 | 0.158 |

- Final WER is 2.99/7.99% for every policy, and all 400 token sequences equal an independent native pass through the frozen adapter.
- p90 delay falls from 707ms to 398-468ms (clean) with one `k13` pass.
- At the same schedule, `repeat` is 73-78ms slower than `zero` at `k13` and rewrites 5-12 times less.
- int8 previews leave delay and rewrites essentially unchanged and lower RTF by 10-16%.

## Stage 2: learned fillers and display gates

Completed on 2026-11-16 on the same 400 utterances. Settings were written to the `stage2` block of `config.json` before the run.

- **Learned fillers**: a 4-layer transformer predicts the next 32 feature frames from the last 64. `det` is trained with L1; `fgn` is an FGN-style stochastic version (noise through conditional layer norm, fair CRPS). `fgn1` fills with one sample, `fgnm` with the mean of 8. Each was trained for 75 minutes on train-clean-100.
- **`fade`**: last-frame repeat blended linearly into the causal mean.
- **Gate**: a gradient-boosted classifier per filler and preview point scores each preview token (probability, blank probability, margin, frame, position, word start); display stops at the first token under the threshold. Gates are trained on 800 train-clean-100 utterances; thresholds are chosen on the 128-utterance selection role for a rewrite target of 5 (`R5`) or 2 (`R2`) per 100 words.
- **`zero&repeat`**: show only the common prefix of the zero and repeat previews (two preview passes).

Mean stable delay (ms) / rewrites per 100 words at the `k13` schedule; change against ungated `repeat` with a 95% utterance-bootstrap interval (1000 replicates).

| Filler, display | dev-clean | dev-other | Delay change vs repeat, clean / other (ms) | RTF (clean) |
|---|---:|---:|---|---:|
| repeat, ungated | 327 / 1.5 | 336 / 4.3 | | 0.059 |
| zero, ungated | 249 / 18.3 | 263 / 20.2 | | 0.059 |
| zero, gate R5 | 281 / 3.6 | 290 / 5.4 | [-50, -43] / [-50, -42] | 0.059 |
| zero, gate R2 | 303 / 1.4 | 314 / 2.5 | [-28, -21] / [-26, -18] | 0.059 |
| mean, gate R5 | 276 / 4.6 | 297 / 5.7 | [-56, -46] / [-44, -35] | 0.059 |
| fade, ungated | 317 / 2.1 | 328 / 4.9 | | 0.059 |
| det, ungated | 301 / 3.3 | 313 / 6.0 | [-29, -24] / [-26, -20] | 0.064 |
| det, gate R2 | 315 / 1.2 | 329 / 2.4 | [-15, -9] / [-10, -4] | 0.064 |
| fgn1, ungated | 297 / 3.8 | 305 / 7.9 | [-33, -27] / [-34, -28] | 0.064 |
| fgnm, ungated | 316 / 1.8 | 323 / 5.8 | [-13, -9] / [-16, -10] | 0.088 |
| zero&repeat | 334 / 0.7 | 343 / 1.7 | [6, 7] / [7, 8] | 0.088 |

- A gate on the zero-fill preview dominates ungated `repeat`: at `R2` it is 22-24ms faster with equal or fewer rewrites, at no extra encoder cost.
- Learned fillers land between `repeat` and `zero` and do not beat gated `zero` at matched rewrites. The stochastic filler is not better than the deterministic one; averaging 8 samples moves it back toward `repeat` at higher cost.
- Future features are weakly predictable: `det` reaches MAE 0.535 against 0.694 for last-frame repeat, and from 0.18 on the first quarter of the horizon to 0.67 on the last (context-mean baseline 0.78).
- Two previews per chunk (`k13_29`) with ungated `repeat` give 293/304ms at 2.4/6.0 rewrites; gating does not improve on that schedule. Full table: [`artifacts/stage2_results.json`](artifacts/stage2_results.json).
- Final WER is 2.99/7.99% for every policy.

Reproduce: `./run_stage2.sh` (about 3.8 hours with an Apple GPU).

## Limits

- Reference times come from a CTC aligner trained on LibriSpeech. CTC spikes can sit before the true acoustic end of a word, so absolute delays may be biased; comparisons between policies share the bias.
- Whether the public checkpoint was trained with a delay penalty or FastEmit is not recorded in the local export script. A backbone trained to emit early may leave less to gain.
- One timing repetition, one Apple CPU thread. The compute clock excludes feature extraction, which is included in RTF only.
- The dev role has been reused across experiments. No test split was used.
- `zero` follows ZeroPrompt's filler value but not its exact layout: the original appends a whole zero chunk after a complete chunk and was evaluated on chunk-masked Conformer encoders.

## Reproduce

```bash
cd experiments/preview_filler_v1
../../.local/venv-fgn-state/bin/python align.py      # runs/preview_filler_v1/alignments.json
../../.local/venv-fgn-state/bin/python replay.py     # runs/preview_filler_v1/replay.json
../../.local/venv-fgn-state/bin/python evaluate.py   # artifacts/results.json
```

Requires the prepared feature cache index, protocol file and teacher checkpoint used by `predict_correct_a2_v1`. Per-utterance replay records and alignments stay in the ignored `runs/` directory.
