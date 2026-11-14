# FGN epoch-wise double-descent investigation

Status: running since 2026-11-13; live training status and completion are recorded in `runs/fgn_double_descent_v1/runner.jsonl`. The 19M model restored its epoch-27 model/optimizer states with zero probe-NLL difference and completed its first additional epoch. No long-horizon result is claimed before completion.

The user requested an experiment after the [double-descent review](../../docs/double_descent_review_2026-11-13.md). Continue all three existing generators from epoch 27 through **96**: **69 new epochs per model**, with no stopping based on worsening validation risk. Run one MPS model at a time, largest first (16 / 4 / 8 blocks). Preserve the original namespace and all previous best/last checkpoints.

## Fixed protocol

- Restore model and AdamW state from `runs/fgn_capacity_long_v1/layers_*/last.pt`. Reproduce saved train/dev probe NLL within 1e-5 before updates; optimizer global step starts at 11,097.
- Keep the original per-epoch batch/noise seeds, fixed cases, CE(smoothing 0.1) + 0.1 state CRPS, learning rate 3e-5, weight decay 0.01 and gradient clip 5. Retain the 10GiB MPS memory and nonfinite-value guards.
- Keep architecture unchanged: generator 4 / 8 / 16 blocks (6.32M / 10.54M / 18.96M), hidden/memory256, FFN1024, four heads, SiLU, noise32, two forecast steps and two stochastic branches. AR stays four layers and jointly trainable; backbone features remain frozen.
- Evaluate the same 512-case train/dev probes each epoch. No test use. This is teacher-forced, one fixed token/cutoff per training utterance, one seed and warm-start continuation.
- Evaluate **all 28,539 training cases** at epochs 27, 32, 48, 64, 80 and 96. Record hard errors, target-vs-best-other log-probability margins, unsmoothed NLL, CE and state CRPS. Report zero-arrived-audio cases separately. One fixed-noise evaluation is not a guarantee for every noise draw.
- Save separate best-NLL and last checkpoints plus snapshots at epochs 32, 48, 64, 80 and 96. Best selection includes the original history; operational overfit markers do not stop training.
- Predeclare descriptive late windows **28–40 / 48–64 / 80–96** for token-NLL and state-CRPS means. No retrospective stopping at an apparent recovery, and no significance claim from these window means.

## Interpretation

This checks whether validation risk later declines under the existing training recipe. It does not identify a mechanism from curve shape alone, guarantee that 96 epochs is sufficient, or exclude recovery beyond the observed range. A second decline need not beat the first minimum. Noise injected into the model is different from corrupted labels. Label smoothing does not itself rule out double descent.

The existing three depth settings do not constitute a broad model-wise interpolation sweep. Whole-set hard errors and margins help characterize fitting, but contradictory no-audio targets and stochastic predictions complicate interpolation. Width, memory, independent initializations and multiple seeds are outside this run. No new ASR WER, encoder acceleration, raw-audio corrector or causal FGN-noise effect is claimed.

An initial input audit finds 221 cases with identical complete BOS-only prefixes, zero arrived audio and conflicting next tokens. Their conservative contribution to expected full-training hard errors is at least 193 under fresh noise independent of targets. This is an expected-risk bound, not a bound for every finite noise draw. The 19M model's epoch-27 full pass has 261 errors: 193 in the zero-audio group and 68 among 28,318 arrived-audio cases. These measurements clarify the fitting regime; they do not establish a double-descent mechanism or change training.

## Commands

```bash
.venv/bin/python experiments/fgn_double_descent_v1/run.py
.venv/bin/python experiments/fgn_double_descent_v1/report.py
.venv/bin/python experiments/fgn_double_descent_v1/verify.py
.venv/bin/python experiments/fgn_double_descent_v1/audit_inputs.py
# After the runner finishes and verification passes:
.venv/bin/python experiments/fgn_double_descent_v1/analyze.py
```

The runner also generates final reports and verifies completed checkpoints automatically. Local prerequisite checkpoints and encoder caches come from preceding studies and are not bundled publicly. Raw checkpoints/per-utterance data stay local; public aggregate evidence lives in `artifacts/`.

## Interim status inspection

The [2026-11-13 interim assessment](../../docs/fgn_double_descent_interim_2026-11-13.md) captures the completed 19M epoch-96 run while the smaller models remain incomplete. Its token-risk window means rise while state CRPS recovers; these are separate metric-specific observations. This snapshot does not stop or change the active run.

```bash
.venv/bin/python experiments/fgn_double_descent_v1/interim.py
```
