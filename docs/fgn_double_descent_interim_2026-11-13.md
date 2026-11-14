# Epoch-wise double-descent interim assessment — 2026-11-13

Snapshot captured at 2026-11-14T06:55:05.313319+00:00. This is an incomplete-study report; the training runner remains active. 113 of 207 requested new epochs have completed across the three models. It does not stop or alter training.

| Generator blocks | Completed epoch / target | Train-probe NLL | Dev NLL | Best dev NLL (epoch) |
|---|---:|---:|---:|---:|
| 4 | 71 / 96 | 0.11487 | 1.28439 | 0.94858 (5) |
| 8 | 27 / 96 | 0.13215 | 1.12691 | 0.95087 (5) |
| 16 | 96 / 96 | 0.12724 | 1.45751 | 0.95043 (5) |

## Declared late windows

| Blocks | Window | Observed points / planned | Mean dev token NLL | Mean dev state CRPS |
|---|---|---:|---:|---:|
| 4 | 28–40 | 13 / 13 | 1.13345 | 0.50612 |
| 4 | 48–64 | 17 / 17 | 1.22994 | 0.50424 |
| 16 | 28–40 | 13 / 13 | 1.16924 | 0.51668 |
| 16 | 48–64 | 17 / 17 | 1.27908 | 0.51378 |
| 16 | 80–96 | 17 / 17 | 1.42482 | 0.50862 |

The completed 16-block run has increasing mean token NLL across the three declared windows, with no sustained second descent visible in that comparison. Its state CRPS recovers after an earlier peak. The other models are still incomplete; this is a metric-specific finding within the observed horizon, not proof of a general double-descent mechanism or its absence.

## State component and full-training fitting

- 4 blocks: dev CRPS peak 0.524470 at epoch 11; latest 0.504247.
  Full fixed-training arrived-audio errors: 70 at epoch 27 → 36 at epoch 64, out of 28318 cases. Zero-audio errors remain separately reported in the numeric snapshot.
- 8 blocks: dev CRPS peak 0.527305 at epoch 11; latest 0.513596.
- 16 blocks: dev CRPS peak 0.532853 at epoch 14; latest 0.508026.
  Full fixed-training arrived-audio errors: 68 at epoch 27 → 24 at epoch 96, out of 28318 cases. Zero-audio errors remain separately reported in the numeric snapshot.

The fixed training inputs contain contradictory no-audio targets; a conservative expected contribution of 193 hard errors is unavoidable under independent fresh noise. A single fixed-noise pass need not obey that expectation exactly. Full-training fitting and improving state loss do not establish better held-out token prediction. The study uses one seed and has no new final-ASR WER measurement.

[Protocol](../experiments/fgn_double_descent_v1/README.md) · [Numeric interim snapshot](../experiments/fgn_double_descent_v1/artifacts/interim.json) · [Current complete/incomplete curves](../experiments/fgn_double_descent_v1/artifacts/learning_curves.png)
