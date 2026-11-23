# Round 3: 960h fine-tuning, display gate, display policy

Run on 2026-11-22 on the fixed 200 dev-clean + 200 dev-other subset. Values are dev-clean / dev-other. Delay is the mean stable word delay in ms; rewrites are per 100 final words.

## 3a. Fine-tuning on 960h

lr 0.0005, SpecAugment, 240 minutes on a V100 (about 3.3 epochs), averaged weights (`ema`), 320ms chunks. `sc` trains on 160/320ms attention chunks only.

| Arm | Penalty | WER (%) | Emission lag (ms) | Delay, no preview |
|---|---:|---|---|---|
| public model | | 3.66 / 10.62 | 238 / 236 | 458 / 464 |
| `960_dp0` | 0 | 3.53 / 10.25 | 275 / 273 | 493 / 496 |
| `960_dp005_sc` | 0.005 | 3.66 / 10.39 | 128 / 148 | 351 / 379 |
| `960_dp005` | 0.005 | 3.95 / 10.82 | 95 / 122 | 328 / 361 |
| `960_dp010_sc` | 0.01 | 4.05 / 11.08 | 41 / 72 | 281 / 315 |
| `960_dp010` | 0.01 | 4.23 / 11.50 | 0 / 35 | 255 / 292 |
| `960_dp020` | 0.02 | collapsed (WER 98%) | | |

60-minute train-clean-100 arms with SpecAugment for comparison: penalty 0.01 gives 4.08 / 11.82 at lag 36 / 65; penalty 0.005 with `sc` gives 3.77 / 10.62 at lag 147 / 162.

- WER rises as the lag falls below roughly 100ms, on both data sizes. 960h moves dev-other WER down by about 0.5 points at equal lag; dev-clean is within the noise of 200 utterances.
- Penalty 0.02 for 240 minutes drives the loss negative and the model collapses.
- `sc` gives higher lag and lower WER than the plain arm at the same penalty; whether it is a better trade-off needs the full dev sets.

## 3b. Display gate on the zero-fill preview

A gradient-boosted classifier scores each preview token; display stops at the first token under a threshold. The gate is fitted on 800 dev utterances and its threshold chosen on 300 others, all outside the evaluation subset. `R1`, `R2`, `R5` are thresholds chosen for 1, 2, 5 rewrites per 100 words on the selection utterances.

One preview per chunk at k=7, display replaced by every pass (the policy used so far):

| Model | WER (%) | No preview | Ungated | R1 | R2 |
|---|---|---|---|---|---|
| public | 3.66 / 10.62 | 458 / 464 | 170 / 10.7, 180 / 16.5 | 266 / 1.0, 287 / 0.9 | 239 / 2.0, 260 / 2.0 |
| `r2_lr5e4_dp005_sa` | 3.64 / 10.90 | 340 / 368 | 112 / 8.1, 133 / 15.7 | 174 / 1.0, 204 / 1.6 | 143 / 1.9, 173 / 2.9 |
| `r3_100_dp010_sa` ema | 4.08 / 11.82 | 277 / 310 | 84 / 12.0, 106 / 18.1 | 144 / 1.2, 174 / 1.4 | 122 / 2.3, 149 / 2.5 |

More previews per chunk (k=7,23 or k=7,15,23,31) do not lower the delay at R1 or R2 under this policy.

## 3c. Display policy: hold until contradicted

An oracle gate that shows only preview tokens equal to the final output still produced 7-14 rewrites per 100 words on the public model. Cause: the preview flushes tokens that the next native pass has not yet committed; the display then drops them and the following preview shows them again.

`hold` keeps text on screen until a pass contradicts it: a pass whose output is a prefix of the screen leaves the screen unchanged, and the last pass always shows the final result. Oracle rewrites become 0.

Delay / rewrites at R1 (clean, other), 320ms chunks:

| Model | k=7, replace | k=7, hold | k=7,15,23,31, hold | Oracle, 4 previews, hold |
|---|---|---|---|---|
| public | 266 / 1.0, 287 / 0.9 | 172 / 0.9, 190 / 2.0 | 152 / 0.9, 172 / 1.9 | 134 / 0.0, 141 / 0.0 |
| penalty-0 control (`r2_lr2e4_dp0` ema) | 315 / 1.2, 323 / 1.1 | 202 / 1.0, 221 / 1.5 | 218 / 0.6, 232 / 1.4 | 159 / 0.0, 164 / 0.0 |
| `r2_lr5e4_dp005_sa` | 174 / 1.0, 204 / 1.6 | 165 / 0.9, 193 / 1.3 | 105 / 0.8, 137 / 1.2 | 58 / 0.0, 79 / 0.0 |
| `r3_100_dp010_sa` ema | 144 / 1.2, 174 / 1.4 | 142 / 1.2, 171 / 1.4 | 78 / 0.8, 113 / 1.4 | 20 / 0.0, 40 / 0.0 |

- With `hold`, more previews help: the WER-neutral penalty model reaches 105 / 137ms at about one rewrite per 100 words, against 458 / 464ms for the public model without previews.
- The public model without any training reaches 152 / 172ms with four gated previews and `hold`.
- Extra gate features that cost nothing (entropy, blank probabilities after the token, tokens left, piece length, frames since the last committed token) lower the k=7 R1 delay by 3-19ms. Agreement with a second, repeat-filled preview adds nothing beyond them.
- With `hold`, a wrong unconfirmed word stays on screen until a later pass contradicts it. Rewrites count it once; how long it stays is not measured yet.

## Next (round 4)

Gate with `hold`, the free features and previews every 40ms on the 960h models; WER on all of dev-clean and dev-other with a paired bootstrap against the public model; 480-minute arms with `sc`, lr decay and penalties 0, 0.005, 0.0075, 0.01.
