# Token context for early display: LM fusion and a long-context decoder branch

Status: completed on 2026-11-23. An LSTM token LM combined with the audio that has already arrived does not make text appear earlier without retraining: LM features in the display gate change the delay at one rewrite per 100 words by 1.5/3.0ms on average over 22 settings, inside the selection noise. Showing the LM's word completions gains 8/7ms at two rewrites per 100 words and nothing at one. Shallow fusion in the native pass at weight 0.15 lowers the full-dev WER by 0.06-0.10/0.09-0.13 points on four of the five checkpoints without a branch. A long-context decoder branch initialised from the LM (design D) emits earlier at the same penalty and WER: after 480 minutes of training its words are stable 7-16ms earlier without previews than those of the matched baselines. On the gated display that is 1-7ms at one rewrite per 100 words and 7-15ms at two.

Nothing here is claimed as a new method. Shallow fusion and LM-initialised prediction networks are standard. The delay penalty, the zero-fill previews, the display gate and the evaluation come from [emit_early_v1](../emit_early_v1/README.md).

## Hypotheses

The 20M transducer picks each token from the current encoder frame and the last two BPE pieces. In emit_early_v1 the gated display stays behind its oracle and every delay-penalized model loses WER. The question was whether longer token context, combined with partial audio, closes either gap. H1 to H6 were written to [`config.json`](config.json) before any GPU run, H7 before any result. H8 was added after the D/E results.

| | Hypothesis | Outcome |
|---|---|---|
| H1 | A lowers the R1 delay by at least a third of the gap to the oracle on the penalty-0.005 model | Refuted. The gap is 29/42ms and the gain is 1/2ms. Over all 22 settings the change is -1.5/-3.0ms with a spread of 3.5/4.2ms |
| H2 | B lowers the rewrites of ungated previews by at least 20% and lowers the R1 delay | Refuted. Fusion in preview passes alone raises ungated rewrites (6.1 to 7.8 on dev-clean, public model, weight 0.5). Fusion in both passes lowers dev-other rewrites by 3-12% and does not lower dev-clean rewrites. The R1 delay does not fall |
| H3 | B in the native pass lowers the full-dev WER of the penalty-0.005 model | Supported, small. At weight 0.15 the WER falls by 0.10 [0.05, 0.15] / 0.11 [0.02, 0.19] points, against 0.13/0.22 points lost to the penalty |
| H4 | C lowers the R1 delay by at least 10ms | Refuted at R1 (-1.3/-2.3ms on average). At R2 the average gain is 8.3/7.4ms, largest on the public model (13/10ms) and smallest on the 480-minute penalty-0.01 model (3/4ms) |
| H5 | D loses less WER than its matched baseline at the same penalty, at no more lag | Refuted as stated. The WER differences are -0.07/+0.05 points at penalty 0.005 and +0.07/+0.11 at 0.01, with intervals that include zero. The lag is 19-22ms lower |
| H6 | E does not beat D | Supported. E has a higher WER than D (+0.20 [0.09, 0.31] / +0.13 [-0.04, 0.29]) and than the baseline. It emits 14/13ms earlier than D with many more wrong previews |
| H7 | Without a penalty D has a lower full-dev WER than its matched baseline | Supported on dev-clean (-0.15 [-0.24, -0.05]); on dev-other the interval includes zero (-0.14 [-0.30, +0.02]). D is not different from the public model (-0.08/+0.00) |
| H8 | At 480 minutes D still has at least 10ms less word delay without previews than its matched baseline, at a WER that is not higher | Met at penalties 0.005 (11/16ms) and 0.0075 (11/12ms), missed at 0.01 (7/10ms). The WER is not higher at any penalty. The advantage is smaller than after 240 minutes (16-20ms), so part of that was convergence speed |

## What was run

| Item | Setting |
|---|---|
| Recognizer | Public 20M streaming Zipformer transducer of emit_early_v1, BPE 500, greedy decoding, 320ms chunks |
| LM | LSTM over the 500 BPE pieces (embedding 512, 2 x 768, 9.3M parameters). Trained for 150 minutes on one V100 on the 960h transcripts (281,241 sentences) plus the LibriSpeech LM corpus (40.4M sentences, 1.64B tokens in total); 554M tokens seen. A fifth of every batch comes from the transcripts. dev and test text unused |
| A, gate | LM features of each preview token added to the display gate: LM log-probability, the LM's top probability, whether the token is the LM's first choice, whether fusion changed it. Recognizer unchanged |
| B, choice | When the transducer emits, the token is the best non-blank one under `log p + lam x LM log p`. The transducer alone decides when to emit. `lam` applies to preview passes and `lamn` to the native pass; only `lamn` changes the final output |
| C, completion | The display also shows the LM's greedy continuation, in whole words, of what a pass displays, while its joint probability is at least `tau` (0.95, 0.9, 0.8, 0.7 or 0.5, chosen with the gate threshold) |
| D, branch | An LSTM over the whole token history, initialised from the LM, is added to the 2-token decoder output through a projection that starts at zero. The model is then fine-tuned. This adds 9.7M parameters |
| E, frozen branch | D with the LSTM frozen; the projection and the original model train |
| Checkpoints for A, B, C | The public model and the 480-minute models `r4_dp005_sc_decay` and `r4_dp010_sc_decay` of emit_early_v1, with weights (`lam`, `lamn`) of (0, 0), (0.25, 0), (0.5, 0), (0.25, 0.25), (0.5, 0.5) |
| Training arms | The `sc` + `decay` recipe of emit_early_v1: 960h, SpecAugment, ScaledAdam at lr 0.0005 decaying linearly to 10%, 160/320ms chunks, averaged weights, 240 minutes on one V100 (baselines 18,600-19,200 steps, D and E 20,800). Baseline and D at penalties 0, 0.005 and 0.01; E at 0.005. The branch uses AdamW (projection lr 0.001, LSTM lr 0.0001) |
| Follow-up at 480 minutes | D at penalties 0.005, 0.0075 and 0.01 for 480 minutes (31,100-31,200 steps), against the 480-minute baselines `r4_dp005_sc_decay`, `r4_dp0075_sc_decay` and `r4_dp010_sc_decay` of emit_early_v1 (42,300-43,000 steps). Added after the D/E results |
| Exploratory additions | Native-pass fusion weights 0.1, 0.15 and 0.35 on the two penalized checkpoints, and weight 0.15 on every arm trained here, after the A/B/C results were seen. The weight was not selected on held-out data; the whole sweep is reported |
| Evaluation | As in emit_early_v1: eight zero-fill previews per 320ms chunk, `hold` display, gate fitted on 800 dev utterances and thresholds chosen on 300 others, delay and rewrites on the fixed 200 + 200 subset, WER on all of dev-clean (2,703 utterances) and dev-other (2,864) with a paired utterance bootstrap (2,000 replicates, 95% interval). Every encoder pass is charged 5.0ms; LM steps are not charged. Test unused |
| Compute | USC CARC Discovery, one GPU per job |

## Definitions

Stable delay, rewrites per 100 words, emission lag, gate, R1, R2, `hold` and oracle are defined in [emit_early_v1](../emit_early_v1/README.md). Values are given as dev-clean/dev-other.

- **`free`, `lm`, `lm+wc`**: the gate without LM features, with them, and with them plus word completion on the display.
- **Selection noise**: R1 and R2 thresholds are chosen on 300 utterances. The same gate at the same target moves by a few ms between settings that should not differ; the R1 change from `free` to `lm` has a standard deviation of 3.5/4.2ms over the 22 settings.
- **Matched baseline**: the arm with the same recipe, penalty and training time but without the branch.

## Results

All settings are in [`artifacts/tables.md`](artifacts/tables.md).

**LM.** It rarely knows how a new word starts.

| Split | Perplexity | Next-token accuracy | Accuracy on the first piece of a word | Tokens with top probability >= 0.9 (accuracy) |
|---|---|---|---|---|
| dev-clean | 14.5 | 0.409 | 0.187 | 16.1% (0.970) |
| dev-other | 15.3 | 0.402 | 0.181 | 16.1% (0.969) |

**A, B, C without fusion in the decoder (`lam` = `lamn` = 0).** Mean stable delay in ms / rewrites per 100 words.

| Checkpoint | No preview | Ungated | R1 `free` | R1 `lm` | R2 `free` | R2 `lm+wc` | Oracle |
|---|---|---|---|---|---|---|---|
| Public | 458 / 464 | 130 / 6.1, 140 / 21.9 | 147 / 0.8, 168 / 1.6 | 146 / 0.7, 165 / 1.6 | 138 / 1.3, 155 / 3.0 | 125 / 1.1, 145 / 2.9 | 129 / 136 |
| Penalty 0.005 | 346 / 376 | 60 / 10.4, 82 / 26.2 | 87 / 1.1, 120 / 1.3 | 86 / 0.8, 118 / 1.2 | 73 / 1.7, 103 / 2.4 | 68 / 1.6, 98 / 2.5 | 58 / 78 |
| Penalty 0.01 | 274 / 308 | 14 / 19.9, 36 / 38.8 | 59 / 1.0, 98 / 1.3 | 58 / 0.8, 96 / 1.0 | 43 / 1.5, 79 / 2.3 | 40 / 1.5, 75 / 2.1 | 10 / 30 |

Average change against `free` over all 22 checkpoint and weight settings:

| Policy | Delay change (ms) | Spread (ms) | Rewrite change |
|---|---|---|---|
| R1 `lm` | -1.5 / -3.0 | 3.5 / 4.2 | -0.07 / -0.05 |
| R1 `lm+wc` | -1.3 / -2.3 | 4.5 / 4.8 | -0.12 / -0.06 |
| R2 `lm` | -1.1 / -2.2 | 1.5 / 2.1 | -0.02 / +0.04 |
| R2 `lm+wc` | -8.3 / -7.4 | 5.5 / 4.5 | -0.17 / -0.06 |

**B in the native pass: full-dev WER, fused minus unfused, same checkpoint.**

| Checkpoint (training minutes) | Unfused WER (%) | 0.1 | 0.15 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|---|---|
| Penalty 0.005 (480) | 3.76 / 10.04 | -0.07 [-0.11, -0.03] / -0.10 [-0.17, -0.04] | -0.10 [-0.15, -0.05] / -0.11 [-0.19, -0.02] | -0.06 [-0.13, -0.00] / -0.15 [-0.25, -0.04] | -0.06 [-0.14, +0.01] / -0.13 [-0.26, +0.00] | +0.03 [-0.06, +0.12] / -0.03 [-0.18, +0.12] |
| Penalty 0.01 (480) | 4.05 / 10.50 | -0.02 [-0.06, +0.02] / -0.04 [-0.11, +0.03] | -0.02 [-0.07, +0.03] / -0.07 [-0.16, +0.01] | -0.01 [-0.08, +0.06] / -0.04 [-0.14, +0.07] | -0.01 [-0.09, +0.07] / -0.05 [-0.18, +0.08] | +0.07 [-0.02, +0.17] / +0.08 [-0.07, +0.23] |
| Baseline, penalty 0 (240) | 3.70 / 9.95 | | -0.06 [-0.11, -0.01] / -0.13 [-0.21, -0.04] | | | |
| Baseline, penalty 0.005 (240) | 3.75 / 10.00 | | -0.07 [-0.12, -0.02] / -0.11 [-0.20, -0.02] | | | |
| Baseline, penalty 0.01 (240) | 4.01 / 10.47 | | -0.07 [-0.12, -0.02] / -0.09 [-0.17, -0.02] | | | |
| D, penalty 0 (240) | 3.55 / 9.82 | | -0.02 [-0.07, +0.03] / -0.01 [-0.10, +0.08] | | | |
| D, penalty 0.005 (240) | 3.68 / 10.05 | | +0.01 [-0.04, +0.06] / +0.00 [-0.09, +0.09] | | | |
| D, penalty 0.01 (240) | 4.08 / 10.58 | | +0.00 [-0.07, +0.08] / -0.04 [-0.13, +0.05] | | | |
| E, penalty 0.005 (240) | 3.87 / 10.18 | | +0.07 [+0.01, +0.13] / +0.05 [-0.04, +0.14] | | | |

On the public model the WER is 3.64/9.73% at weight 0.25 (+0.01 [-0.05, +0.08] / -0.08 [-0.18, +0.03] against 3.63/9.81%) and 3.77/9.81% at 0.5.

**D and E against matched baselines, 240 minutes.** Display columns are on the 400-utterance subset with the `free` gate.

| Penalty | Arm | Full-dev WER (%) | WER, arm minus baseline | Lag (ms) | No preview (ms) | Ungated | R1 | R2 | Oracle (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline | 3.70 / 9.95 | | 285 / 283 | 504 / 504 | 174 / 4.7, 180 / 18.6 | 184 / 1.1, 196 / 1.7 | 179 / 1.5, 189 / 2.7 | 173 / 176 |
| 0 | D | 3.55 / 9.82 | -0.15 [-0.24, -0.05] / -0.14 [-0.30, +0.02] | 268 / 267 | 487 / 490 | 148 / 6.9, 157 / 18.8 | 162 / 0.8, 181 / 1.6 | 154 / 1.6, 169 / 3.3 | 147 / 154 |
| 0.005 | Baseline | 3.75 / 10.00 | | 140 / 159 | 360 / 386 | 66 / 10.7, 88 / 25.0 | 86 / 1.1, 119 / 1.6 | 76 / 2.0, 106 / 3.1 | 64 / 84 |
| 0.005 | D | 3.68 / 10.05 | -0.07 [-0.17, +0.04] / +0.05 [-0.11, +0.20] | 121 / 139 | 343 / 366 | 48 / 14.7, 66 / 31.1 | 84 / 0.9, 114 / 1.5 | 66 / 1.7, 91 / 2.8 | 46 / 60 |
| 0.005 | E | 3.87 / 10.18 | +0.13 [+0.01, +0.25] / +0.18 [-0.02, +0.37] | 107 / 126 | 333 / 355 | 27 / 22.7, 43 / 40.8 | 80 / 0.8, 111 / 1.4 | 56 / 1.9, 82 / 2.5 | 24 / 36 |
| 0.01 | Baseline | 4.01 / 10.47 | | 59 / 86 | 292 / 323 | 22 / 18.0, 44 / 35.1 | 66 / 1.0, 104 / 1.2 | 45 / 2.1, 78 / 2.8 | 19 / 38 |
| 0.01 | D | 4.08 / 10.58 | +0.07 [-0.05, +0.18] / +0.11 [-0.07, +0.28] | 38 / 64 | 276 / 306 | 3 / 27.1, 24 / 46.4 | 57 / 0.9, 92 / 1.4 | 36 / 2.0, 68 / 2.8 | -0 / 16 |

**D against matched baselines, 480 minutes.** The baselines are those of emit_early_v1, scored with the gate code of this experiment; their lag is taken from emit_early_v1.

| Penalty | Arm | Steps | Full-dev WER (%) | WER, D minus baseline | Lag (ms) | No preview (ms) | Ungated | R1 | R2 | Oracle (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | Baseline | 42,284 | 3.76 / 10.04 | | 122 / 145 | 346 / 376 | 60 / 10.4, 82 / 26.2 | 87 / 1.1, 120 / 1.3 | 73 / 1.7, 103 / 2.4 | 58 / 78 |
| 0.005 | D | 31,220 | 3.72 / 9.95 | -0.04 [-0.14, +0.07] / -0.09 [-0.25, +0.08] | 110 / 131 | 335 / 360 | 43 / 15.8, 62 / 31.6 | 86 / 0.4, 119 / 1.1 | 61 / 1.7, 88 / 2.6 | 41 / 56 |
| 0.0075 | Baseline | 42,517 | 3.94 / 10.18 | | 70 / 97 | 304 / 335 | 32 / 15.3, 55 / 31.8 | 72 / 0.9, 109 / 1.2 | 50 / 2.1, 82 / 2.6 | 29 / 50 |
| 0.0075 | D | 31,106 | 3.83 / 10.28 | -0.11 [-0.22, -0.00] / +0.10 [-0.07, +0.26] | 61 / 86 | 293 / 323 | 17 / 21.2, 38 / 38.6 | 67 / 0.7, 102 / 1.3 | 42 / 1.6, 75 / 2.7 | 13 / 32 |
| 0.01 | Baseline | 42,958 | 4.05 / 10.50 | | 32 / 63 | 274 / 308 | 14 / 19.9, 36 / 38.8 | 59 / 1.0, 98 / 1.3 | 43 / 1.5, 79 / 2.3 | 10 / 30 |
| 0.01 | D | 31,207 | 4.05 / 10.50 | -0.00 [-0.13, +0.12] / +0.01 [-0.17, +0.18] | 25 / 53 | 267 / 298 | -1 / 28.2, 20 / 47.0 | 54 / 1.1, 92 / 1.5 | 31 / 2.0, 65 / 2.8 | -5 / 12 |

## Findings

- **The LM cannot tell what the next word is.** It gets the first piece of a word right 19% of the time. Its confident predictions (16% of tokens at probability 0.9 or more) are mostly continuations inside a word, and the next previews, 40ms apart, deliver those tokens anyway. This reading fits the null results of gate features (A) and of word completion at one rewrite per 100 words (C); it was not tested directly.
- **Word completion helps only when two rewrites per 100 words are accepted.** The gain is 8/7ms on average. It is largest on the public model (13/10ms) and smallest on the 480-minute penalty-0.01 model (3/4ms).
- **Fusion in preview passes makes previews less stable.** The preview and the final pass then disagree more often. Fusion in both passes lowers dev-other ungated rewrites by 3-12%, which the gate does not turn into earlier display.
- **Shallow fusion at a small weight lowers the WER a little.** At 0.15 the gain is 0.06-0.10/0.09-0.13 points on four of five checkpoints, with intervals that exclude zero. On the penalty-0.005 model it recovers 0.10 of the 0.13 points lost on dev-clean and 0.11 of 0.22 on dev-other. The 480-minute penalty-0.01 model gains nothing; the 240-minute model at the same penalty does. The cause is not established. Weight 0.5 is too high everywhere.
- **D makes the recognizer emit earlier at the same penalty and training time.** The lag falls by 17/16ms at penalty 0, 19/20ms at 0.005 and 21/22ms at 0.01. The WER difference against the matched baseline is inside the bootstrap interval at both penalties above zero.
- **The gate keeps little of that gain.** D moves the oracle display 18-26ms earlier, but its early previews are wrong more often (ungated rewrites 10.7 to 14.7 on dev-clean at penalty 0.005, 18.0 to 27.1 at 0.01). At one rewrite per 100 words the display is 2/5ms earlier at penalty 0.005 and 9/12ms at 0.01; the first is inside the selection noise. At penalty 0 it is 22/15ms earlier, but still later than the public model with the same gate (147/168ms).
- **The frozen LM is not enough.** E emits earliest and has the earliest oracle (24/36ms), but its WER is higher than the baseline's and its ungated previews are rewritten 22.7/40.8 times per 100 words. Adapting the LSTM is what keeps D's WER.
- **D and shallow fusion do not add up.** Fusion at 0.15 changes none of the D arms. Without a penalty, D (3.55/9.82%) and the fused baseline (3.64/9.82%) are close.
- **Part of the 240-minute advantage was convergence speed, and part remains.** D after 240 minutes already matches the 480-minute baselines in lag, delay and WER (penalty 0.005: 343/366ms without previews and 3.68/10.05% against 346/376ms and 3.76/10.04%). After 480 minutes D is still ahead of them: words are stable 11/16ms earlier without previews at penalty 0.005, 11/12ms at 0.0075 and 7/10ms at 0.01, at a WER the bootstrap cannot separate. The 240-minute gaps were 16-20ms.
- **On the display the remaining advantage shows at two rewrites per 100 words.** At R2, D after 480 minutes is 12/15, 8/7 and 12/14ms earlier at the three penalties. At R1 it is 1/1, 5/7 and 5/6ms earlier, close to the selection noise. At 0.005 the R1 threshold of D lands at 0.4/1.1 rewrites against 1.1/1.3 for the baseline, so that comparison is at unequal rewrites. The oracle is 15-22ms earlier at every penalty, so the gate again keeps only part of what the recognizer offers.

## Conclusion

Token context does not open a new way to show text earlier on this model. Without retraining, an LM adds 1-3ms at one rewrite per 100 words, and the likely reason is that the next word is not predictable from text while the rest of a word arrives with the next previews. Built into the decoder and fine-tuned, the same LM makes the recognizer emit about 10ms earlier at equal WER after 480 minutes (7-16ms). The display gate passes on 1-7ms of that at one rewrite per 100 words and 7-15ms at two. The earliest display in this experiment is D at penalty 0.01 after 480 minutes: 54/92ms at 1.1/1.5 rewrites with a WER of 4.05/10.50%, against 59/98ms for its baseline. These gains are small next to those of emit_early_v1, where previews, the gate and a penalty of 0.005 take the delay from 458/464ms to 87/120ms. The other gain is accuracy: shallow fusion at weight 0.15, or the branch without a penalty, lowers the WER by about 0.1 points.

## Limits

- One training run per arm and one seed. The bootstrap intervals cover the choice of utterances, not the variation between training runs.
- Delay, rewrites and lag come from 400 utterances; gate thresholds from 300. Differences under about 5ms at R1 are not resolved.
- The baselines have 9.7M fewer parameters than D and E. No arm adds the same capacity without LM initialisation, so the gain of D is not attributed to LM knowledge.
- Training budgets are wall-clock, so step counts are not matched. At 240 minutes D and E completed 20,800 steps and the baselines 18,600-19,200. At 480 minutes D completed 31,100-31,200 steps and the baselines 42,300-43,000: the three D jobs ran at 52-67 steps a minute for the first three quarters of the run, against 82-91 for the baseline, for a reason that was not established. The 480-minute comparison therefore gives D fewer updates.
- The 480-minute baselines come from emit_early_v1. The recipe is the same and the training code differs only in the branch, but they ran on another day.
- The LM runs once per emitted token. Its cost is not charged in the delay and was not measured.
- The fusion weight was not chosen on held-out data. The whole sweep is reported and no single weight is claimed as tuned.
- The LM corpus is LibriSpeech text, the same domain as the evaluation. Greedy decoding only. Test sets unused.

## Reproduce

Everything runs as Slurm jobs. `pack.py` inlines the files in `hpc/` into one self-contained script. The workspace, features and public weights come from `setup.sh` and `prep.sh` of emit_early_v1.

```bash
cd experiments/context_fusion_v1
python pack.py lm_prep.sh > lm_prep.sbatch   # LM corpus and token file (CPU)
python pack.py job.sh > job.sbatch           # every other job; set the @MARKER@ fields
sbatch --account=<your account> job.sbatch
```

Markers in `job.sh`, with `0` or `none` for unused steps:

- LM: `TAG=cf_lm LMTAG=cf_lm LMMIN=150 GPU=v100`.
- A, B, C on a checkpoint: `TAG=cf_dp005_l25n25 GATE=cf_dp005_l25n25 GCKPT=ckpt/r4_dp005_sc_decay/last.pt GKEY=ema LMX=ckpt/cf_lm/lm.pt LAM=0.25 LAMN=0.25 WERDEV=cf_dp005_l25n25 GPU=p100`.
- Baseline: `TAG=cf_base_dp005 MIN=240 DP=0.005 GPU=v100`, then the gate and WER steps as above with `GCKPT=ckpt/cf_base_dp005/last.pt`.
- D: `TAG=cf_ctx_dp005 CTX=1 CTXLM=ckpt/cf_lm/lm.pt MIN=240 DP=0.005 GPU=v100 GATE=cf_ctx_dp005_ema GCKPT=ckpt/cf_ctx_dp005/last.pt GKEY=ema LMX=ckpt/cf_lm/lm.pt WERDEV=cf_ctx_dp005_ema`. E uses `CTX=2`.
- Paired comparison: `CMP="A/wer_dev_n0.json B/wer_dev_n0.json"`.

Other markers: `TIME` (Slurm limit; replay, gate and full-dev WER need about two and a half hours after training), `MAXF=50000`, `GTRAIN=400`, and `DEP` (job IDs for `afterok`; remove the `--dependency` line when there are none).

Aggregates copied from the cluster are in [`artifacts/gate.json`](artifacts/gate.json), [`artifacts/eval.json`](artifacts/eval.json), [`artifacts/lines.txt`](artifacts/lines.txt) (result lines of every job log) and [`artifacts/tables.md`](artifacts/tables.md). Checkpoints, features and per-utterance records stay in the cluster workspace and are not in the repository.
