# Round 4: full-dev WER, gate with `hold` on the 960h models, 480-minute arms

Run on 2026-11-22. Values are dev-clean / dev-other. Delay is the mean stable word delay in ms on the fixed 200 + 200 subset; rewrites are per 100 final words.

## 4a. WER on all of dev-clean and dev-other

Native streaming greedy decoding at 320ms chunks on 2,703 dev-clean and 2,864 dev-other utterances. The change against the public model has a 95% paired utterance-bootstrap interval (2,000 replicates).

| Model | Penalty | WER (%) | Change, dev-clean | Change, dev-other | Emission lag (ms) |
|---|---:|---|---|---|---|
| public | | 3.63 / 9.81 | | | 238 / 236 |
| `960_dp0` | 0 | 3.65 / 9.89 | +0.02 [-0.08, +0.12] | +0.07 [-0.10, +0.24] | 275 / 273 |
| `960_dp005_sc` | 0.005 | 3.76 / 10.21 | +0.13 [+0.01, +0.24] | +0.39 [+0.21, +0.59] | 128 / 148 |
| `960_dp005` | 0.005 | 3.80 / 10.22 | +0.17 [+0.05, +0.28] | +0.40 [+0.22, +0.60] | 95 / 122 |
| `960_dp010_sc` | 0.01 | 4.02 / 10.63 | +0.39 [+0.26, +0.52] | +0.82 [+0.61, +1.03] | 41 / 72 |
| `960_dp010` | 0.01 | 4.26 / 11.00 | +0.62 [+0.48, +0.77] | +1.19 [+0.98, +1.41] | 0 / 35 |
| `r2_lr5e4_dp005_sa` (train-clean-100, 60 min) | 0.005 | 3.88 / 10.95 | +0.25 [+0.14, +0.37] | +1.14 [+0.94, +1.34] | 113 / 135 |

All 960h models are the averaged weights after 240 minutes (round 3). Lag is from the 400-utterance subset.

- Penalty-0 fine-tuning on 960h keeps the public WER. Every penalized model loses WER, and the loss grows as the lag falls.
- The round-2 model that looked WER-neutral on the 400-utterance subset (3.64 / 10.90 against 3.66 / 10.62) is not: it loses 0.25 / 1.14 points on the full sets. On dev-other the subset understated the WER loss of every fine-tuned model by 0.2-0.9 points.
- 960h data matters mostly for dev-other: at penalty 0.005 the dev-other loss is 0.40 points with 960h and 1.14 with train-clean-100.
- `sc` does not change the trade-off. At penalty 0.005 it gives the same WER with 33 / 26ms more lag; at 0.01 its WER and lag sit on the line between the plain 0.005 and 0.01 models.

## 4b. Gate with `hold` on the 960h models

Same gate as round 3 (fitted on 800 dev utterances, threshold chosen on 300 others, all outside the evaluation subset). All columns use `hold` and the free extra features. `R1` and `R2` are thresholds chosen for 1 and 2 rewrites per 100 words on the selection utterances. Cells are delay / rewrites for dev-clean, dev-other. 320ms chunks.

| Model | No preview | 1 preview (k=7), R1 | 4 previews, R1 | Every 40ms (8 previews), R1 | Every 40ms, R2 | Every 40ms, oracle |
|---|---|---|---|---|---|---|
| public | 458, 464 | 170 / 0.9, 190 / 1.5 | 149 / 0.9, 168 / 1.8 | 147 / 0.8, 168 / 1.6 | 138 / 1.3, 155 / 2.9 | 129, 136 |
| `960_dp0` | 493, 496 | 195 / 1.1, 212 / 2.0 | 182 / 0.8, 199 / 1.4 | 177 / 0.9, 195 / 1.7 | 169 / 1.6, 183 / 3.4 | 163, 169 |
| `960_dp005_sc` | 351, 379 | 142 / 1.5, 172 / 1.6 | 92 / 1.0, 122 / 1.2 | 89 / 1.0, 121 / 1.1 | 74 / 1.9, 102 / 2.6 | 62, 80 |
| `960_dp005` | 328, 361 | 143 / 1.1, 170 / 1.3 | 87 / 1.0, 122 / 1.7 | 80 / 0.9, 117 / 1.8 | 61 / 1.9, 95 / 3.3 | 45, 66 |
| `960_dp010_sc` | 281, 315 | 132 / 1.1, 164 / 1.5 | 74 / 0.5, 111 / 1.0 | 67 / 0.6, 107 / 1.2 | 42 / 1.7, 76 / 2.5 | 14, 34 |
| `960_dp010` | 255, 292 | 132 / 1.0, 163 / 1.0 | 61 / 0.9, 100 / 1.7 | 64 / 0.7, 106 / 1.5 | 33 / 1.9, 70 / 3.3 | -2, 18 |
| `r2_lr5e4_dp005_sa` | 340, 368 | 146 / 1.3, 177 / 1.5 | 96 / 0.9, 131 / 1.2 | 95 / 0.7, 129 / 1.2 | 70 / 1.7, 102 / 2.6 | 50, 70 |

The oracle shows only preview tokens equal to the final output; with `hold` it has 0 rewrites.

- Without any training, the public model goes from 458 / 464ms to 147 / 168ms at about one rewrite per 100 words (dev-other 1.6), with unchanged WER.
- Penalty 0.005 on 960h reaches 80 / 117ms at the same rewrite level, for a WER cost of 0.17 / 0.40 points. Penalty 0.01 reaches 64 / 106ms for 0.62 / 1.19 points.
- The gate is what separates the models from their oracle: 35 / 51ms at penalty 0.005 and 66 / 88ms at 0.01. At R1 the stronger penalty gains only 16 / 11ms over the weaker one, although its ungated delay is 46 / 47ms lower.
- Four previews give most of the gain; eight change the R1 delay by 7ms or less in either direction.
- Penalty-0 fine-tuning makes the display slower than the public model at every setting (lag 275 against 238ms).
- Fitting the gate on 2,400 utterances instead of 800 changes nothing for `r2_lr5e4_dp005_sa` (every 40ms, R1: 95 / 0.7, 130 / 1.1).
- The R1 operating point moves between repeated gate runs on the same model (k=7 with `hold`, level-0 features: 165 / 0.9, 193 / 1.3 in round 3; 155 / 1.3, 181 / 1.6 here). The threshold is chosen on 300 utterances, so differences of about 10ms between settings are within this variation.

160ms chunks, every 40ms (4 previews), R1:

| Model | No preview | Gated, `hold` |
|---|---|---|
| public | 386, 388 | 224 / 0.4, 231 / 1.2 |
| `960_dp005` | 264, 297 | 115 / 0.6, 148 / 1.1 |
| `960_dp010` | 194, 230 | 63 / 1.1, 96 / 1.3 |

Smaller chunks do not beat 320ms chunks with previews, and their WER is higher (round 2).

## 4c. 480-minute arms

960h, SpecAugment, lr 0.0005, `sc`, averaged weights, 480 minutes on a V100 (about 6 epochs). `decay` lowers the lr linearly to 10% of its start value. WER is on the full dev sets; the last three columns use eight gated previews per 320ms chunk with `hold`.

| Arm | Penalty | WER (%) | Change, dev-clean | Change, dev-other | Lag (ms) | No preview | R1 | R2 | Oracle |
|---|---:|---|---|---|---|---|---|---|---|
| `r4_dp0_sc_decay` | 0 | 3.61 / 9.83 | -0.02 [-0.13, +0.09] | +0.02 [-0.16, +0.19] | 296 / 293 | 513, 511 | 197 / 0.9, 212 / 1.6 | 190 / 1.3, 201 / 2.7 | 183, 187 |
| `r4_dp005_sc` (no decay) | 0.005 | 3.83 / 10.22 | +0.20 [+0.09, +0.31] | +0.40 [+0.21, +0.60] | 112 / 135 | 339, 369 | 88 / 0.8, 123 / 1.2 | 69 / 1.6, 99 / 2.6 | 56, 75 |
| `r4_dp005_sc_decay` | 0.005 | 3.76 / 10.04 | +0.13 [+0.02, +0.24] | +0.22 [+0.04, +0.41] | 122 / 145 | 346, 376 | 90 / 0.9, 123 / 1.3 | 73 / 1.8, 103 / 2.7 | 58, 78 |
| `r4_dp0075_sc_decay` | 0.0075 | 3.94 / 10.18 | +0.31 [+0.19, +0.44] | +0.36 [+0.17, +0.56] | 70 / 97 | 304, 335 | 72 / 0.9, 109 / 1.2 | 50 / 2.1, 82 / 2.6 | 29, 50 |
| `r4_dp010_sc_decay` | 0.01 | 4.05 / 10.50 | +0.42 [+0.29, +0.54] | +0.68 [+0.48, +0.88] | 32 / 63 | 274, 308 | 62 / 0.8, 102 / 1.1 | 41 / 1.8, 76 / 2.4 | 10, 30 |

WER on the 400-utterance subset, in the same order: 3.51 / 10.16, 3.82 / 10.99, 3.77 / 10.62, 3.82 / 10.73, 4.08 / 10.99.

- Twice the training time at constant lr changes nothing: `r4_dp005_sc` loses 0.20 / 0.40 points, the 240-minute `960_dp005_sc` 0.13 / 0.39, and the gated delay is the same (88 / 123 against 89 / 121ms).
- lr decay lowers the dev-other loss. At penalty 0.005 it is 0.22 instead of 0.40 points at similar lag. `r4_dp0075_sc_decay` loses less on dev-other (0.36) than the 240-minute `960_dp005` (0.40) with 25ms less lag, and `r4_dp010_sc_decay` loses 0.68 against 0.82 for `960_dp010_sc` with 9ms less lag. dev-clean does not change. One run per setting.
- No penalized arm keeps the public WER. The smallest loss is 0.13 / 0.22 points (`r4_dp005_sc_decay`), with a gated delay of 90 / 123ms against 147 / 168ms for the public model.
- Penalty-0 fine-tuning keeps the WER and emits later still: lag 296 / 293ms after 480 minutes with `sc`, 275 / 273ms after 240 minutes without, 238 / 236ms for the public model.
- At R1, penalty 0.01 is 28 / 21ms faster than 0.005 for 0.29 / 0.46 more WER points; its oracle is 48 / 48ms faster.

160ms chunks, every 40ms, R1: `r4_dp005_sc_decay` 126 / 0.6, 158 / 1.1 (no preview 277, 308); `r4_dp010_sc_decay` 70 / 1.0, 101 / 0.7 (no preview 205, 236).
