# FGN generator capacity and overfitting diagnosis

Status: completed on 2026-11-12. All three capacities met the declared overfit rule at continuation epoch 11; total sequential training time was 1.04 hours. Best and last checkpoints passed verification. See the [completed results](../../docs/fgn_capacity_2026-11-12.md) and `artifacts/completion.json`; detailed logs remain local in `runs/fgn_capacity_v1/`.

The user requested larger models to determine whether limited capacity explains the remaining training loss. This study enlarges the **FGN generator**, unlike the earlier experiment that enlarged AR while keeping the generator fixed.

| FGN residual blocks | Generator parameters | Total AR+FGN parameters |
|---|---:|---:|
| 4 | 6,323,968 | 10,926,836 |
| 8 | 10,535,680 | 15,138,548 |
| 16 | 18,959,104 | 23,561,972 |

The AR remains four layers and is jointly trainable in every configuration. All runs start from the same trained small hybrid. Added generator blocks initially implement identity mappings, so all three initial functions are identical. Checks verify exact initial logits, nonzero gradients into the added blocks, and sampling independent of batch order.

Identity growth nests the smaller function class inside the larger one. Therefore the larger class's globally minimized empirical objective cannot be worse, but finite optimization need not reach that minimum. This is the capacity hypothesis under test, not a guarantee about trained validation performance.

## Controlled task

- Full train-clean-100 utterance pool: 28,539 utterances.
- One **fixed** token/cutoff case per utterance; this deliberately measures fitting a fixed training problem. It is not dense training on every token/cutoff, or evidence that the full corpus distribution can be memorized.
- Fixed 512-example training probe and 512 model-selection dev cases. The dev roles remain speaker-separated from gate fitting/calibration. No test evaluation or test-driven selection.
- Fixed two-step acoustic forecast horizon. No curriculum change during the new curves.
- The fixed cutoffs, probe size and horizon differ from the preceding 256-case/curriculum curves. Compare capacities within this study; do not attribute an absolute cross-study loss change to model size.
- Same objective as the existing hybrid: branch-mean token CE with smoothing 0.1 plus 0.1 acoustic-state fair CRPS.
- Reference text prefixes are teacher-forced, as in the prior training/probe task. These NLL curves are not recognized-prefix display acceptance or final transcription WER.
- Record online optimized loss and separately evaluated CE, CRPS, combined objective, unsmoothed token NLL and Hit@1. Training probes and dev use the same evaluation mode and fixed noise seeds.
- Online loss averages training batches while weights change; it is not the same measurement as the fixed 512-case evaluation-mode probe at the end of an epoch. The aggregate CSV and middle figure row retain both.
- AdamW, learning rate 3e-5, warmup 100 steps, weight decay 0.01, gradient clipping 5.0; batch at most 96 utterances / 3,000 padded chunks.
- Train one model at a time on MPS. Predeclared driver-allocation guard: 10GiB. The largest 24-step benchmark used about 4.89GiB and took 8.94 seconds; these short timings are planning observations, not full-epoch performance guarantees.

## Why training loss is not expected to be zero

With a 500-class vocabulary and label smoothing 0.1, even an unrestricted model minimizing smoothed CE has minimum CE about **0.94466 nats**, not zero. The unsmoothed NLL at that ideal smoothed solution is approximately 0.10514. These quantities are different from the unsmoothed training-probe NLL in the previous figure.

The state forecast is uncertain, and CRPS is not directly comparable in scale to CE. Larger capacity can reduce empirical fitting error, but the curve alone cannot distinguish capacity, optimization, sampling coverage and objective mismatch. The task here is to measure that distinction, not force an arbitrary zero.

A training-input audit finds 221 zero-arrived-audio cases with conflicting next-token targets for identical complete prefixes. Their conservative contribution to expected unsmoothed NLL is at least 0.03019 nats under fresh independent functional noise. This is an information-ambiguity bound, not a measured model loss or a bound on every finite Monte Carlo evaluation. It is separate from the smoothed-CE bound above; do not add them. See `artifacts/input_ambiguity.json`.

## Stopping and extension

Initial stage: at most 16 epochs per capacity. If no model satisfies the overfitting rule, extend in the same namespace to 32, 64 and at most 96 epochs. The learning-rate schedule is independent of the requested final epoch so resumed extensions preserve the original trajectory.

Operational overfitting is confirmed only after at least eight epochs when five successive evaluations have both:

1. Mixed-dev unsmoothed NLL at least 0.02 above its best checkpoint.
2. Fixed training-probe NLL at least 0.01 below its value at that checkpoint.

A plateau, a large absolute loss, or a rising validation curve without improving training does not satisfy this rule. Save the best development checkpoint and the last checkpoint separately. Finishing a stage or reaching the maximum does not mean overfitting was found. One seed and one fixed-case sampling design are diagnostic limitations.

## Scope boundary

This is continued training of the existing actual AR+FGN hybrid. The [raw-audio residual corrector and encoder-skipping plan](../../docs/predict_correct_acceleration_plan_2026-11-12.md) remains unimplemented. Lower loss here does not demonstrate lower final WER, useful audio correction, or actual inference acceleration. Prior completed runs are preserved.

## Commands

These commands require the local hybrid checkpoint, encoder-feature cache and role manifests produced by the preceding studies. Input hashes are recorded in `artifacts/provenance_initial.json`; weights and feature files are not distributed in the public repository.

```bash
# Run the staged study, resuming completed epochs safely.
.venv/bin/python experiments/fgn_capacity_v1/run.py

# One model or a short benchmark.
.venv/bin/python experiments/fgn_capacity_v1/study.py --layers 16 --epochs 16
.venv/bin/python experiments/fgn_capacity_v1/study.py --layers 16 --bench 24

# Update aggregate tables/plots from current logs without training.
.venv/bin/python experiments/fgn_capacity_v1/report.py

# Audit saved checkpoints and inputs; then write the final report only once
# the sequential runner has finished and every capacity passes verification.
.venv/bin/python experiments/fgn_capacity_v1/verify.py
.venv/bin/python experiments/fgn_capacity_v1/analyze.py
```

Public aggregate curves and summaries are written to `artifacts/`; weights, optimizer state and detailed logs remain under the ignored run directory.
