# Round 1 (2026-11-22): delay penalty sweep, lr 0.002, 60 min on train-clean-100, V100

Fixed 200 dev-clean + 200 dev-other subset; values are dev-clean / dev-other. `lag` = emission slot minus aligned word end; `none` = native display delay; `k7` = one zero-fill preview at 7 real frames (delay / rewrites per 100 words). 320ms chunks unless noted.

| Arm | Penalty | WER % | lag ms | none ms | k7 | none, 160ms chunks | WER, 160ms chunks |
|---|---:|---|---|---|---|---|---|
| base (public weights) | - | 3.66 / 10.62 | 238 / 236 | 458 / 464 | 170/10.7, 180/16.5 | | |
| pilot (20 min) | 0.01 | 5.07 / 14.22 | -8 / 25 | 252 / 284 | 93/10.7, 114/16.2 | | |
| dp000 | 0 | 5.01 / 14.70 | 284 / 279 | 499 / 496 | 232/13.1, 235/14.9 | 429 / 426 | 5.07 / 15.53 |
| dp002 | 0.002 | 5.04 / 14.13 | 118 / 137 | 351 / 374 | 154/5.8, 175/9.8 | 285 / 307 | 5.72 / 15.99 |
| dp005 | 0.005 | 5.46 / 15.24 | 17 / 48 | 273 / 305 | 117/7.1, 140/13.1 | 213 / 245 | 6.05 / 16.61 |
| dp010 | 0.01 | 5.72 / 15.76 | -69 / -33 | 216 / 248 | 96/10.5, 117/15.1 | 160 / 191 | 6.50 / 16.99 |
| dp020 | 0.02 | 8.50 / 18.18 | -141 / -109 | 163 / 192 | 84/12.0, 102/17.0 | 112 / 139 | 8.73 / 18.81 |
| dp050 | 0.05 | 97 / 98 (collapsed) | | | | | |
| dp010_lr5 (lr 0.005) | 0.01 | 8.21 / 21.35 | -50 / -16 | 229 / 261 | 120/11.1, 142/16.3 | 170 / 200 | 9.02 / 23.41 |

Reading:
- The penalty moves emission earlier monotonically; 0.005 removes about 265ms of lag against the penalty-0 control.
- The penalty-0 control already loses 1.35 / 4.08 WER points against the public weights and emits later. The fine-tuning recipe (lr 0.002 from averaged weights, clean-100 only, no augmentation, no averaging) is the main source of WER loss, not the penalty. Higher lr is much worse.
- Against the control, penalty 0.002 costs no WER; 0.005 costs about 0.5 points.
- Preview gain (none minus k7) shrinks from 267ms (control) to 120ms (0.01): previews and the penalty consume the same lag, but still stack.
