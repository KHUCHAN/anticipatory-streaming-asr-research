# Earlier emission: delay-penalized fine-tuning and gated zero-fill previews

Status: completed on 2026-11-22 (four rounds). On the public 20M streaming Zipformer at 320ms chunks, the mean stable word delay is 458/464ms on dev-clean/dev-other. Gated zero-fill previews with a hold-until-contradicted display bring it to 147/168ms at 0.8/1.6 rewrites per 100 words, with no training and unchanged WER. Fine-tuning on 960h with a delay penalty of 0.005 and the same display reaches 80-90/117-123ms at 0.8-1.0/1.1-1.8 rewrites, for a WER cost of 0.13-0.20/0.22-0.40 points on the full dev sets. No penalized model keeps the public WER; the smallest loss is 0.13/0.22 points.

Nothing here is claimed as a new method. The delay penalty is the one in k2, zero-fill previews follow ZeroPrompt, and the display gate comes from [preview_filler_v1](../preview_filler_v1/README.md). See the [prior-work review](../../docs/prior_work_predict_refine_2026-11-16.md).

## Hypotheses

On the public model the native display delay splits into an emission lag of 238/236ms after the aligned word end and about 220/228ms of chunk wait and lookahead. Zero-fill previews already emit most tokens correctly in encoder frames that contain no real audio, so the lag looked like a learned habit that training could remove. The hypotheses were written to [`config.json`](config.json) before any GPU run.

| | Hypothesis | Outcome |
|---|---|---|
| H1 | A delay penalty lowers the emission lag by at least 100ms at a WER cost of at most 0.5 points on the 400-utterance subset | Holds for penalty 0.005 on 960h: lag falls by 143/114ms; WER rises by 0.29/0.20 points on the subset and 0.17/0.40 on the full dev sets. The loss is small, but its interval excludes zero |
| H2 | Previews and the penalty consume the same lag, so the preview gain shrinks as the penalty grows | Holds, and they still stack. The gain of one ungated preview is 284/281ms at penalty 0, 218/232ms at 0.005 and 180/198ms at 0.01 |
| H0 | Fine-tuning with penalty 0 leaves lag and WER unchanged | WER is unchanged (3.65/9.89 against 3.63/9.81%). The lag is not: it rises from 238/236 to 275/273ms after 240 minutes and to 296/293ms after 480 minutes with `sc`. The cause is not established |

## What was run

| Item | Setting |
|---|---|
| Model | Public 20M streaming Zipformer transducer (`desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small`, `exp/pretrained.pt`), BPE 500, greedy decoding |
| Loss | 0.5 x smoothed simple RNN-T + pruned RNN-T (prune range 5, lm scale 0.25), `delay_penalty` passed to both |
| Optimizer | ScaledAdam, clipping scale 2.0, fp32, at most 50,000 feature frames per batch, a fixed wall-clock budget per arm on one V100 |
| Round 1 | lr 0.002, train-clean-100, 60 minutes, penalties 0 to 0.05 |
| Round 2 | lr 0.0002 to 0.001, averaged weights (EMA 0.999), SpecAugment |
| Round 3 | 960h, SpecAugment, lr 0.0005, 240 minutes, penalties 0 to 0.02; `sc` trains on 160/320ms attention chunks only; display gate and `hold` display policy |
| Round 4 | WER on the full dev sets; gate with `hold` on the 960h models; 480-minute arms with `sc`, penalties 0 to 0.01; `decay` lowers the lr linearly to 10% |
| Decoding | 320ms chunks (39-frame window, 32-frame hop); 160ms chunks as a secondary setting |
| Previews | Zero fill: an extra encoder pass on the first `k` real feature frames of the next chunk, the rest set to 0.0. Previews read the native encoder state and decoder context and never write them. `k` = 7, or 7/15/23/31, or every 40ms (7, 11, ..., 35) |
| Evaluation | Delay, rewrites and lag on the fixed 200 dev-clean + 200 dev-other subset and alignments of `preview_filler_v1`; WER also on all of dev-clean (2,703 utterances) and dev-other (2,864); test unused |
| Clock | Audio arrival of the last needed frame plus 5.0ms charged per encoder pass, queued on one worker |
| Compute | USC CARC Discovery (Slurm), one GPU per job |

## Definitions

- **Stable delay** and **rewrites per 100 final words**: as in [preview_filler_v1](../preview_filler_v1/README.md). Stable delay is the earliest display time after which a word never changes, minus the aligned end of the matching reference word, over words the final hypothesis gets right.
- **Emission lag**: position of the encoder frame in which the native pass emits the last token of a word, minus the aligned end of the word. It excludes chunk wait and compute.
- **Gate**: a gradient-boosted classifier scores each preview token; display stops at the first token under a threshold. Inputs are the token probability, blank probability, margin, frame, position, preview length, running minimum probability, word start and preview point, plus free extras from the same pass (entropy, blank probabilities after the token, tokens left, piece length, frames since the last committed token). It is fitted on 800 dev utterances and its threshold is chosen on 300 others, all outside the evaluation subset.
- **R1, R2, R5**: thresholds chosen for 1, 2 and 5 rewrites per 100 words on the selection utterances.
- **`hold`**: text on screen is replaced only when a pass contradicts it. A pass whose output is a prefix of the screen leaves the screen unchanged, and the last pass always shows the final result.
- **Oracle**: a gate that shows exactly the preview tokens equal to the final output.

## Results

Values are dev-clean / dev-other. WER and its change against the public model (95% paired utterance bootstrap, 2,000 replicates) are on the full dev sets; lag, delay and rewrites are on the 400-utterance subset. The last column uses eight gated previews per 320ms chunk with `hold` at R1, shown as delay / rewrites for dev-clean, dev-other.

| Model (960h) | Minutes | Penalty | WER (%) | WER change | Lag (ms) | Delay, no preview (ms) | Gated previews, `hold`, R1 |
|---|---:|---:|---|---|---|---|---|
| public | | | 3.63 / 9.81 | | 238 / 236 | 458 / 464 | 147 / 0.8, 168 / 1.6 |
| `960_dp0` | 240 | 0 | 3.65 / 9.89 | +0.02 [-0.08, +0.12] / +0.07 [-0.10, +0.24] | 275 / 273 | 493 / 496 | 177 / 0.9, 195 / 1.7 |
| `r4_dp0_sc_decay` | 480 | 0 | 3.61 / 9.83 | -0.02 [-0.13, +0.09] / +0.02 [-0.16, +0.19] | 296 / 293 | 513 / 511 | 197 / 0.9, 212 / 1.6 |
| `960_dp005_sc` | 240 | 0.005 | 3.76 / 10.21 | +0.13 [+0.01, +0.24] / +0.39 [+0.21, +0.59] | 128 / 148 | 351 / 379 | 89 / 1.0, 121 / 1.1 |
| `960_dp005` | 240 | 0.005 | 3.80 / 10.22 | +0.17 [+0.05, +0.28] / +0.40 [+0.22, +0.60] | 95 / 122 | 328 / 361 | 80 / 0.9, 117 / 1.8 |
| `r4_dp005_sc` | 480 | 0.005 | 3.83 / 10.22 | +0.20 [+0.09, +0.31] / +0.40 [+0.21, +0.60] | 112 / 135 | 339 / 369 | 88 / 0.8, 123 / 1.2 |
| `r4_dp005_sc_decay` | 480 | 0.005 | 3.76 / 10.04 | +0.13 [+0.02, +0.24] / +0.22 [+0.04, +0.41] | 122 / 145 | 346 / 376 | 90 / 0.9, 123 / 1.3 |
| `r4_dp0075_sc_decay` | 480 | 0.0075 | 3.94 / 10.18 | +0.31 [+0.19, +0.44] / +0.36 [+0.17, +0.56] | 70 / 97 | 304 / 335 | 72 / 0.9, 109 / 1.2 |
| `960_dp010_sc` | 240 | 0.01 | 4.02 / 10.63 | +0.39 [+0.26, +0.52] / +0.82 [+0.61, +1.03] | 41 / 72 | 281 / 315 | 67 / 0.6, 107 / 1.2 |
| `960_dp010` | 240 | 0.01 | 4.26 / 11.00 | +0.62 [+0.48, +0.77] / +1.19 [+0.98, +1.41] | 0 / 35 | 255 / 292 | 64 / 0.7, 106 / 1.5 |
| `r4_dp010_sc_decay` | 480 | 0.01 | 4.05 / 10.50 | +0.42 [+0.29, +0.54] / +0.68 [+0.48, +0.88] | 32 / 63 | 274 / 308 | 62 / 0.8, 102 / 1.1 |

In model names, `sc` means training on 160/320ms attention chunks only and `decay` a linear lr decay to 10%. All models use the averaged weights.

Per-round tables: [round 1](artifacts/round1.md), [round 2](artifacts/round2.md), [round 3](artifacts/round3.md), [round 4](artifacts/round4.md).

Findings:

- **The penalty trades WER for lag.** Every penalized model loses WER on the full dev sets, and the loss grows as the lag falls. Penalty 0.02 for 240 minutes on 960h and 0.05 for 60 minutes on train-clean-100 collapse (WER above 97%).
- **The recipe matters more than the penalty at first.** In round 1 (lr 0.002, train-clean-100, no augmentation, no averaging) the penalty-0 control alone lost 1.35/4.08 WER points. With lr 0.0005, SpecAugment, averaged weights and 960h, penalty-0 fine-tuning keeps the public WER.
- **Penalty-0 fine-tuning emits later than the public model** (lag 275-296 against 238ms), so it makes every display setting slower.
- **`sc` does not change the trade-off.** It gives the same WER at penalty 0.005 with more lag, and at 0.01 it sits on the line between the plain models.
- **Longer training does not remove the WER cost; lr decay reduces it on dev-other.** At penalty 0.005, 480 minutes at constant lr lose as much as 240 minutes (0.20/0.40 against 0.13/0.39 points). With lr decay the dev-other loss is 0.22 points at similar lag, and the decay arms at 0.0075 and 0.01 also lose less on dev-other than 240-minute arms with more lag. dev-clean does not change. One run per setting.
- **Display policy.** When every pass replaces the screen, a preview shows tokens that the next native pass has not yet committed; they disappear and come back. Even an oracle gate then has 7-14 rewrites per 100 words on the public model. `hold` removes this, and only with `hold` do more previews per chunk lower the delay.
- **The gate limits the stronger penalty.** At R1, penalty 0.01 is only 16-28/11-21ms faster than 0.005, although its oracle is 47-48ms faster.
- **The 400-utterance subset is too small for WER.** A train-clean-100 model that looked WER-neutral on it (3.64/10.90 against 3.66/10.62%) loses 0.25/1.14 points on the full sets.

## Conclusion

- Showing text earlier is mostly a display problem. Gated zero-fill previews with `hold` cut the delay of the unchanged public model to about a third (458/464 to 147/168ms).
- A delay penalty moves emission earlier and stacks with previews, but it is never free. The setting with the smallest loss (0.005 with lr decay) gives 90/123ms for 0.13/0.22 WER points.
- Penalties above 0.005 buy little extra delay at the same rewrite rate and cost more WER.

## Limits

- Delay, rewrites and lag come from 200 + 200 utterances and have no intervals. The R1 operating point moves by about 10ms between repeated gate runs, because the threshold is chosen on 300 utterances.
- The clock charges an assumed 5.0ms per encoder pass; device compute was not measured. Eight previews per chunk mean nine encoder passes per 320ms of audio.
- With `hold`, a wrong unconfirmed word stays on screen until a later pass contradicts it. Rewrites count it once; how long it stays was not measured.
- Recipe choices (lr, augmentation, penalties) were made on the dev subset, and the gate is fitted on other dev utterances. No test split was used, so all numbers are development numbers.
- One training run and one seed per arm. Arms have a wall-clock budget, not a fixed number of updates.
- Reference times come from a CTC aligner (`facebook/wav2vec2-base-960h`); absolute delays may be biased, and comparisons share the bias.
- Whether the public checkpoint was trained with a delay penalty is not known.
- LibriSpeech read speech and greedy decoding only.

## Reproduce

Everything runs as Slurm jobs. `pack.py` inlines the files in `hpc/` into one self-contained script.

```bash
cd experiments/emit_early_v1
python pack.py setup.sh > setup.sbatch    # environment, upstream sources, public weights, LibriSpeech, features
python pack.py prep.sh > prep.sbatch      # further training splits; set @NAME@
python pack.py train.sh > train.sbatch    # fine-tune, replay, gate, full-dev WER; set the @MARKER@ fields
sbatch --account=<your account> train.sbatch
```

Markers in `train.sh` for the `960_dp005` arm: `TAG=r3_960_dp005 DP=0.005 MIN=240 LR=0.0005 GPU=v100 SPLITS=train-clean-100,train-clean-360,train-other-500 SA=1 SC=0 DECAY=0 BASE=0 GTRAIN=400 GATE=r3_960_dp005_ema GCKPT=ckpt/r3_960_dp005/last.pt GKEY=ema WERDEV=r3_960_dp005_ema`. `TIME` is the Slurm limit; evaluation, gate and full-dev WER need about three hours after training. `MIN=0` skips training and runs only the gate and WER steps on an existing checkpoint. `hpc/align_eval.b64` holds the reference words and aligned word ends of the 400 evaluation utterances (gzip + base64 of the `preview_filler_v1` alignments), so that jobs are self-contained.

Aggregate summaries copied from the cluster are in [`artifacts/eval.json`](artifacts/eval.json) (WER, lag and ungated delay per model) and [`artifacts/gate.json`](artifacts/gate.json) (gated delay and rewrites). Checkpoints, features and per-utterance records stay in the cluster workspace and are not in the repository.
