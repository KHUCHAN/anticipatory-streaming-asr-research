# An explicit next-piece loss on the 20M streaming Zipformer

Status: completed on 2026-11-24. A loss that makes the model name the next BPE piece at every frame trains as intended, but it does not make text appear earlier at the same WER. On the frozen penalty-0.01 model the display at one rewrite per 100 words moves by 3/1ms with the anticipated pieces and by 7/5ms with the head's probability as a gate feature. Trained jointly, the loss acts like a second delay penalty: words are stable 36-62ms earlier without previews and the full-dev WER rises by 0.30-0.37/0.80-0.85 points. All five hypotheses are refuted.

Follows [context_fusion_v1](../context_fusion_v1/README.md). There the long-context decoder branch was trained with the transducer loss and the delay penalty only. No loss asked the model for the next piece before the transducer emits it. This experiment adds that loss and tests whether its predictions speed up the gated display.

Pairs of numbers are dev-clean / dev-other throughout.

## Question

Does a loss that makes the model name the next BPE piece at every frame, from the audio so far and the token history, give an earlier display at the same rewrite rate?

## Hypotheses

Fixed before any GPU run: [`config.json`](config.json).

| | Hypothesis | Outcome |
|---|---|---|
| H1 | On `np_w02_dp010` the anticipated pieces lower the R1 delay by at least 10ms on both dev sets (`ant` against `free`) | Refuted. 7/2ms (63/98 to 56/96ms) |
| H2 | The same gain appears on the frozen penalty-0.01 baseline (`np_probe_dp010`) | Refuted. 3/1ms (62/102 to 59/101ms) |
| H3 | The joint arms with weight 0.02 have at least 10ms less word delay without previews than their matched baselines, at a full-dev WER that is not higher | Refuted on WER. The delay falls by 62/61ms at penalty 0.005 and 36/39ms at 0.01, but the WER rises by 0.30/0.80 and 0.37/0.85 points, with intervals above zero |
| H4 | After a native pass the head names the next piece at least 10 points more often than the LM and than the transducer's best non-blank piece (`np_w02_dp010`) | Refuted on the transducer. 58.7% against 21.3% for the LM (+37.4) and 51.6% for the transducer (+7.1) |
| H5 | With the long-context branch the head is at least 5 points more accurate at word starts | Refuted. 52.6% against 52.5% |

## Design

- **Head**: a classifier on one encoder frame and the decoder output (which includes the long-context branch when the model has one). Own projections to 512, tanh, 500 classes; class 0 means that nothing follows. 0.65M parameters.
- **Loss**: at frame `t` and state `u` (pieces emitted so far), cross-entropy against piece `u+1`, weighted by the probability that the alignment leaves frame `t` in state `u`. That probability is the blank-arc occupation of the simple lattice, which the pruned RNN-T loss already computes. Each frame carries total weight 1. The transducer loss gives no token target at these points; it trains blank there.
- **Display**: after every zero-fill preview pass the head proposes up to two further pieces. They are appended to the preview tokens and go through the same display gate (`ant`). `hq` instead gives the gate the head's probability of each transducer preview token. `free` is the gate of the earlier experiments.
- **Arms**: four 480-minute fine-tuning runs with the loss (weights 0.02 and 0.1, penalties 0.005 and 0.01, one with the long-context branch), and two 90-minute runs that train the head alone on frozen 480-minute models.
- **Baselines**: the 480-minute models of emit_early_v1 (`r4_dp005_sc_decay`, `r4_dp010_sc_decay`) and context_fusion_v1 (`cf_ctx480_dp010`) without the loss.
- **R1, R2**: gate thresholds chosen on 300 selection utterances for at most one or two rewrites per 100 words. Delays are mean stable word delays on the fixed 200 + 200 subset.

## Results

**Recognizer, 480 minutes.** WER on all of dev-clean and dev-other; the difference is against the matched baseline with a paired bootstrap interval. Lag and delay without previews on the subset.

| Arm | Penalty | Loss weight | Steps | WER (%) | Against matched baseline | Emission lag (ms) | Delay without previews (ms) |
|---|---|---|---|---|---|---|---|
| `r4_dp005_sc_decay` | 0.005 | – | 42,300-43,000 | 3.76 / 10.04 | | 122 / 145 | 346 / 376 |
| `np_w02_dp005` | 0.005 | 0.02 | 35,200 | 4.06 / 10.84 | +0.30 [+0.18, +0.43] / +0.80 [+0.61, +0.99] | 47 / 73 | 284 / 315 |
| `r4_dp010_sc_decay` | 0.01 | – | 42,300-43,000 | 4.05 / 10.50 | | 32 / 63 | 274 / 308 |
| `np_w02_dp010` | 0.01 | 0.02 | 35,100 | 4.42 / 11.35 | +0.37 [+0.24, +0.50] / +0.85 [+0.65, +1.05] | -13 / 16 | 238 / 269 |
| `np_w10_dp010` | 0.01 | 0.1 | 34,700 | 5.46 / 12.82 | +1.41 [+1.24, +1.57] / +2.32 [+2.06, +2.56] | -60 / -38 | 203 / 227 |
| `cf_ctx480_dp010` (long context) | 0.01 | – | 31,100-31,200 | 4.05 / 10.50 | | 25 / 53 | 267 / 298 |
| `np_ctx_w02_dp010` (long context) | 0.01 | 0.02 | 35,600 | 4.65 / 11.39 | +0.60 [+0.45, +0.75] / +0.88 [+0.67, +1.09] | -29 / -3 | 227 / 255 |

**Display.** Cells are delay (ms) / rewrites per 100 words for dev-clean, then dev-other. The two probes are the frozen baselines `r4_dp010_sc_decay` and `cf_ctx480_dp010`, so their `free` column is the baseline display and their WER is unchanged.

| Model | R1 `free` | R1 `hq` | R1 `ant` | R2 `free` | R2 `hq` | R2 `ant` | Oracle `free` | Oracle `ant` |
|---|---|---|---|---|---|---|---|---|
| `np_probe_dp010` (frozen) | 62 / 0.8, 102 / 1.2 | 55 / 0.7, 97 / 1.0 | 59 / 0.8, 101 / 0.9 | 40 / 1.7, 76 / 2.5 | 36 / 1.9, 71 / 2.5 | 38 / 1.5, 77 / 2.0 | 10 / 30 | -72 / -50 |
| `np_probe_ctx_dp010` (frozen) | 55 / 1.0, 92 / 1.5 | 51 / 1.1, 89 / 1.4 | 55 / 0.8, 95 / 1.2 | 31 / 2.1, 65 / 2.9 | 30 / 1.9, 64 / 2.6 | 34 / 1.6, 71 / 1.9 | -5 / 12 | -99 / -76 |
| `np_w02_dp005` | 66 / 1.1, 103 / 1.7 | 64 / 1.0, 101 / 1.4 | 56 / 1.2, 96 / 1.6 | 45 / 2.2, 77 / 3.5 | 42 / 2.2, 74 / 3.4 | 34 / 2.1, 71 / 2.7 | 19 / 37 | -95 / -73 |
| `np_w02_dp010` | 63 / 0.9, 98 / 0.9 | 58 / 1.0, 97 / 0.8 | 56 / 0.9, 96 / 1.0 | 36 / 2.2, 69 / 2.6 | 33 / 2.1, 66 / 2.5 | 28 / 2.0, 64 / 2.5 | -11 / 6 | -111 / -91 |
| `np_w10_dp010` | 55 / 0.8, 86 / 1.2 | 47 / 1.0, 81 / 1.2 | 46 / 1.0, 79 / 1.0 | 29 / 1.6, 60 / 2.1 | 23 / 1.6, 54 / 2.1 | 24 / 1.9, 55 / 2.1 | -39 / -24 | -135 / -115 |
| `np_ctx_w02_dp010` | 50 / 1.0, 85 / 1.7 | 52 / 0.7, 90 / 0.8 | 55 / 0.5, 93 / 0.9 | 25 / 2.0, 58 / 2.9 | 22 / 2.0, 55 / 2.7 | 24 / 1.5, 59 / 2.2 | -32 / -16 | -137 / -114 |

Shown without a gate, the anticipated pieces cause 590-740 rewrites per 100 words.

**Next-piece accuracy after a native pass** (last frame of a chunk, all audio real, no lookahead). Cells are head / transducer's best non-blank piece / external LM, in percent, over 28,100-28,500 positions on 1,500 dev utterances. Lead is the time until the transducer emits that piece. On `np_w02_dp010`, 64% of positions have a lead of 40-80ms, 15% 120-160ms, 7% 200-320ms and 15% 360ms or more.

| Model | All | Lead 40-80ms | Lead 120-160ms | Lead 200-320ms | Lead 360ms+ | Word start | Inside word |
|---|---|---|---|---|---|---|---|
| `np_probe_dp010` (frozen) | 54.4 / 46.5 / 22.3 | 77.0 / 71.1 / 25.6 | 37.9 / 18.4 / 26.2 | 12.9 / 5.4 / 15.1 | 6.4 / 3.3 / 8.9 | 46.8 / 38.0 / 13.4 | 88.7 / 85.1 / 63.0 |
| `np_probe_ctx_dp010` (frozen) | 55.3 / 46.9 / 21.3 | 77.9 / 72.3 / 24.6 | 40.7 / 19.7 / 24.5 | 15.9 / 6.0 / 14.4 | 7.9 / 3.7 / 8.8 | 48.5 / 38.7 / 13.2 | 89.0 / 87.4 / 60.9 |
| `np_w02_dp005` | 65.7 / 58.3 / 23.0 | 87.1 / 85.1 / 25.7 | 60.5 / 37.9 / 29.0 | 25.5 / 12.4 / 15.6 | 10.8 / 4.3 / 9.0 | 59.7 / 51.2 / 13.6 | 91.8 / 89.4 / 64.1 |
| `np_w02_dp010` | 58.7 / 51.6 / 21.3 | 78.9 / 74.4 / 24.4 | 39.3 / 19.9 / 24.2 | 15.8 / 6.5 / 14.0 | 9.0 / 3.9 / 8.3 | 52.5 / 44.4 / 13.0 | 88.3 / 86.4 / 61.3 |
| `np_w10_dp010` | 57.6 / 53.1 / 20.8 | 74.5 / 71.7 / 24.0 | 32.4 / 20.4 / 20.2 | 16.7 / 8.5 / 11.5 | 10.1 / 4.0 / 9.0 | 51.8 / 46.1 / 12.7 | 86.3 / 87.3 / 60.6 |
| `np_ctx_w02_dp010` | 58.3 / 51.0 / 20.3 | 77.5 / 73.1 / 23.7 | 38.8 / 19.0 / 21.0 | 16.8 / 6.4 / 11.6 | 11.0 / 3.9 / 8.7 | 52.6 / 44.3 / 12.8 | 87.8 / 86.3 / 59.4 |

After a zero-fill preview pass whose own tokens are all correct, the head on `np_w02_dp010` names the next piece 18% of the time at the first preview point of a chunk and 57% at the last, 280ms later (transducer's best non-blank piece: 8% and 51%; LM: 20-21% throughout).

Raw numbers: [`artifacts/`](artifacts) (`gate.json`, `eval.json`, `train.json`, `lines.txt`).

## Findings

- **The loss trains.** The head reaches 54-66% after a native pass, 4-8 points above the transducer's own best non-blank piece and 32-43 points above the LM.
- **The head is right when the piece is about to be emitted anyway.** At a lead of 40-80ms it is right 74-87% of the time, and the transducer's best guess is already right 71-85% of the time. At 200-320ms it is right 13-26% of the time and at 360ms or more 6-11%, which is the level of the text-only LM (12-16% and 8-9%). The piece cannot be named before its sound has started.
- **Token context adds nothing to the audio.** The long-context branch changes the accuracy at word starts by 0.1 points in the joint arms and 1.7 points in the probes.
- **The display does not move at the same WER.** On the frozen baseline, `ant` gains 3/1ms at R1 and 2/-1ms at R2; `hq` gains 7/5ms at R1 and 4/5ms at R2. The same baseline scored 59/98ms at R1 with the gate code of context_fusion_v1 and 62/102ms here, so the gate fit alone moves the R1 delay by 3-4ms.
- **Correct anticipations exist, but the gate cannot pick them.** With `ant` the oracle display is 80-114ms earlier than without. At one rewrite per 100 words the gate keeps at most 10ms of that: it has to pass tokens that are right about 99 times in 100, and after a preview the head is right 13-64% of the time.
- **Trained jointly, the loss is a delay penalty with a worse trade-off.** The emission lag falls to -13/16ms at penalty 0.01, so tokens come out before the word has ended, and the WER rises. `np_w02_dp005` (4.06/10.84%, 284/315ms without previews) is no better in WER and 10/7ms later than the plain penalty-0.01 baseline (4.05/10.50%, 274/308ms). The likely cause is that the head's gradient makes encoder frames carry a guess of the upcoming piece, and the joiner reads the same frames. This was not tested separately.
- **The WER loss is not a step-count effect.** The joint arms ran 34,700-35,600 steps against 42,300-43,000 for the baselines, but the baseline of emit_early_v1 that trained for only 240 minutes at penalty 0.01 (`960_dp010_sc`) has 4.02/10.63%.
- **The joint arms show the largest display gains, at a WER nobody would accept for them.** `np_w10_dp010` with `ant` reaches 46/79ms at R1, 16/23ms earlier than the baseline display, for 1.41/2.32 WER points.

## Conclusion

Asking the model for the next piece explicitly does not open a way to show text earlier. The head learns what can be learned: it beats the transducer's own guess by 4-8 points, and nearly all of that is for pieces whose sound has already begun. For pieces further away it is no better than a text-only LM, with or without long token context. At the same WER the gated display changes by 1-7ms, where the gate fit alone moves it by 3-4ms.

The best display on this model remains the one of emit_early_v1: gated zero-fill previews on the delay-penalised model, 62/102ms at one rewrite per 100 words for 4.05/10.50% WER (penalty 0.01), or 90/123ms for 3.76/10.04% (penalty 0.005). The gap to the oracle display (10/30ms at penalty 0.01) is a gap in telling right previews from wrong ones, not in predicting ahead.

## Limits

- The head is trained on ordinary chunked passes, where a frame sees real audio up to the end of its chunk. On a zero-fill preview the rest of the chunk is zero. A head trained on preview passes was not tried.
- The loss weights were guesses. Weight 0.02 already costs WER; smaller weights and a head whose gradient stops before the encoder in joint training were not tried. The frozen probes cover the second case for 90 minutes only.
- One run per arm, one seed. Wall-clock budgets do not equalise step counts.
- For `np_ctx_w02_dp010` the R1 threshold of `free` lands at 1.0/1.7 rewrites and those of `hq` and `ant` at 0.5-0.9, so its R1 columns compare unequal rewrite rates.
- Baseline lag, delay and WER come from emit_early_v1 and context_fusion_v1, which ran on other days. The probe on `r4_dp010_sc_decay` reproduces its lag, delay without previews and oracle exactly.
- Test sets were not used.

## Reproduce

Needs the workspace of emit_early_v1 on the cluster (features, upstream sources, checkpoints of the baselines, the LM of context_fusion_v1).

```bash
python pack.py job.sh > job.packed.sh   # self-contained Slurm script; fill the @MARKERS@ as in config.json
```

Markers for `np_w02_dp010`: `TAG=np_w02_dp010 GPU=v100 TIME=11:30:00 NXW=0.02 DP=0.01 MIN=480 ANT=2 LMX=ckpt/cf_lm/lm.pt GATE=np_w02_dp010_ema GCKPT=ckpt/np_w02_dp010/last.pt GKEY=ema WERDEV=np_w02_dp010_ema CMP="np_w02_dp010_ema/wer_dev_n0.json r4_dp010_sc_decay_ema/wer_dev.json"`. The probes add `FREEZE=1 NXW=1 INIT=ckpt/r4_dp010_sc_decay/last.pt INITKEY=ema MIN=90 GPU=p100`. The ctx arm adds `CTX=1 CTXLM=ckpt/cf_lm/lm.pt`.
