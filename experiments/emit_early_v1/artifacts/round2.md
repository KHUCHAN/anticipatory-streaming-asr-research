# Round 2: learning rate, weight averaging, SpecAugment

Run on 2026-11-22. Each arm fine-tunes the public 20M model for 60 minutes on train-clean-100 (V100) and is replayed on the fixed 200 dev-clean + 200 dev-other subset. `ema` is the exponential moving average of the weights (0.999). Values are dev-clean / dev-other.

320ms chunks. Delay is the mean stable word delay in ms; `k7` is one zero-fill preview per chunk, shown as delay / rewrites per 100 words.

| Arm | lr | Penalty | WER (%) | Emission lag (ms) | Delay, no preview | `k7` clean | `k7` other |
|---|---:|---:|---|---|---|---|---|
| public model | | | 3.66 / 10.62 | 238 / 236 | 458 / 464 | 170 / 10.7 | 180 / 16.5 |
| `lr2e4_dp0` ema | 0.0002 | 0 | 3.66 / 10.71 | 272 / 269 | 488 / 490 | 204 / 17.4 | 208 / 21.5 |
| `lr5e4_dp0` ema | 0.0005 | 0 | 3.90 / 10.93 | 278 / 275 | 493 / 494 | 211 / 18.2 | 215 / 22.8 |
| `lr2e4_dp005` ema | 0.0002 | 0.005 | 3.95 / 11.39 | 84 / 107 | 314 / 342 | 103 / 8.3 | 123 / 13.9 |
| `lr5e4_dp005` ema | 0.0005 | 0.005 | 3.98 / 11.96 | 48 / 75 | 287 / 317 | 101 / 8.1 | 121 / 13.8 |
| `lr1e3_dp005` ema | 0.001 | 0.005 | 4.49 / 12.30 | 27 / 57 | 274 / 305 | 101 / 8.1 | 123 / 13.8 |
| `lr5e4_dp005_sa` (SpecAugment) | 0.0005 | 0.005 | 3.64 / 10.90 | 113 / 135 | 340 / 368 | 112 / 8.1 | 133 / 15.7 |
| `lr2e4_dp010` ema | 0.0002 | 0.01 | 4.16 / 12.13 | 8 / 39 | 257 / 289 | 86 / 10.3 | 109 / 15.3 |
| `lr5e4_dp010` ema | 0.0005 | 0.01 | 4.44 / 13.16 | -30 / 4 | 232 / 266 | 86 / 10.2 | 107 / 15.8 |
| `dp010_long` (round-1 recipe, 180 min) | 0.002 | 0.01 | 7.66 / 19.55 | -116 / -82 | 183 / 216 | 98 / 11.5 | 117 / 16.4 |

160ms chunks (same weights, smaller decode chunk).

| Arm | WER (%) | Delay, no preview | `k7` clean | `k7` other |
|---|---|---|---|---|
| `lr2e4_dp0` ema | 3.79 / 11.73 | 418 / 418 | 272 / 20.3 | 273 / 21.7 |
| `lr5e4_dp005_sa` ema | 3.95 / 11.70 | 276 / 305 | 137 / 7.0 | 166 / 13.6 |
| `lr2e4_dp005` ema | 4.26 / 12.10 | 252 / 280 | 115 / 8.3 | 144 / 13.3 |
| `lr2e4_dp010` ema | 4.65 / 13.36 | 195 / 227 | 78 / 7.8 | 102 / 13.1 |

## Reading

- The WER loss of round 1 came from the recipe, not from the penalty. With lr 0.0002 and averaged weights the penalty-0 control keeps the public WER (3.66 / 10.71).
- Penalty-0 fine-tuning still moves emission later (lag 238 to 272ms). The penalty effect measured against the control is therefore larger than against the public model.
- SpecAugment removes the WER cost of penalty 0.005 (3.64 / 10.90) while the lag falls by 125 / 101ms and the no-preview delay by 118 / 96ms against the public model. H1 holds on this subset. The lag falls more slowly with SpecAugment than without (113 against 48ms after 60 minutes).
- Higher lr and longer training on train-clean-100 without augmentation trade WER for lag (`lr1e3`, `dp010_long`).
- A preview still stacks on the penalty: `k7` reaches 101-112ms on dev-clean with 8 rewrites per 100 words, where the public model has 170ms with 10.7.

## Next (round 3)

960h data with SpecAugment for 240 minutes at penalties 0, 0.005, 0.01, 0.02; training on 160/320ms chunks only; a display gate on the preview of the fine-tuned model.
