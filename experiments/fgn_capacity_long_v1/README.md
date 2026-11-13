# Longer FGN capacity training

Status: completed on 2026-11-13. All three models reached epoch 27, adding 48 epochs in 1.75 hours. Checkpoints, optimizer states, settings and original-input preservation passed verification. See the [completed report](../../docs/fgn_capacity_long_2026-11-13.md) and `artifacts/completion.json`; detailed local logs remain in `runs/fgn_capacity_long_v1/`.

The user requested more epochs after the first capacity study stopped on overfitting. Continue all three saved epoch-11 models for **16 additional epochs**, through epoch **27**, with overfit-based stopping disabled. The original study and its best/last checkpoints remain unchanged in `runs/fgn_capacity_v1/`; extended outputs have a separate namespace.

The model, data, fixed cases, optimizer state, learning rate, seed schedule and losses remain the same as the [original protocol](../fgn_capacity_v1/README.md). No activation-function or other architecture change is included. FGN uses SiLU in its feed-forward, observation and noise-embedding paths; 256 hidden dimensions, 1,024 FFN dimensions, four attention heads, 256-dimensional GRU memory, and 32-dimensional step noise. Generator block counts are 4 / 8 / 16. Two-step self-generated acoustic forecasts and two noise branches are trained with token CE (smoothing 0.1) + 0.1 state CRPS.

This is a depth comparison with fixed width and memory size. Per frame, the predicted change is `W h + b` with a 512-by-256 output matrix. Additional blocks do not increase that output projection's maximum rank. Whether this restriction matters for the acoustic data is unmeasured; these runs cannot establish that wider generators or larger memory would behave the same way.

Each continuation restores both model and AdamW state. Before updates, re-evaluate train/dev probes and require NLL agreement with the source checkpoint within 1e-5. Epochs 12–27 use the same global epoch-dependent batch/noise seeds and unchanged constant-after-warmup learning rate, 3e-5. Train one model at a time on MPS and retain the 10GiB driver-allocation guard. Nonfinite loss/gradient and resource failures still stop the run.

The diagnostic remains one fixed token/cutoff per each of 28,539 train-clean-100 utterances, with fixed 512-case train and model-selection dev probes. It is teacher-forced and single-seed. No test data is used. Lower token NLL does not establish improved final WER, better acoustic forecasts, a causal noise-regularization effect or lower inference cost.

This extension has no newly continued AR-only or noise-free control. It measures extra-epoch effects within the same stochastic hybrids, not a new FGN-versus-AR comparison.

The aggregate renderer is reused without editing the original experiment. Its full curves include inherited epochs 0–11; extension-specific comparisons separately identify the new epochs. Best and last checkpoints are distinct, and the original development-best checkpoint is copied before continuation.

```bash
.venv/bin/python experiments/fgn_capacity_long_v1/run.py
.venv/bin/python experiments/fgn_capacity_long_v1/report.py
.venv/bin/python experiments/fgn_capacity_long_v1/verify.py
# Only after all three models finish and pass verification:
.venv/bin/python experiments/fgn_capacity_long_v1/analyze.py
```

These commands require locally generated checkpoints and cached encoder features from the preceding studies. Raw weights, detailed logs and feature files remain local-only. Public aggregate figures and provenance are written to `artifacts/`.

## Post-hoc double-descent audit

The [review](../../docs/double_descent_review_2026-11-13.md) separates token-risk trends from the late decline in state CRPS. It does not classify or exclude double descent from a finite, single-seed record. No additional training is part of this audit; longer training and wider size sweeps remain proposals.

```bash
.venv/bin/python experiments/fgn_capacity_long_v1/audit_double_descent.py
```
