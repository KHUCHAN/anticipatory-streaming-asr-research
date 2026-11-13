# Layer-band2 AR + FGN hybrid

This is the explicitly requested combination, rather than another replacement
for the band2 AR. It preserves the trained 4-layer/256-wide token AR, its
per-layer current+past2 self-attention, and the arrived-audio cross-attention.
A continuously recurrent FGN predicts future 8x512 encoder-state chunks from
arrived observations. The AR can cross-attend to those **generated** features
as well as the actually arrived encoder frames. Token CE backpropagates through
the forecast audio into the FGN. Fair CRPS additionally supervises continuous
future states, with gradients retained through predicted-state feedback.

The Zipformer backbone is frozen. This is a 100-hour warm-start continuation
pilot, not a new 960-hour acoustic model or a direct replication of weather FGN.
The previously separate predictors and all their files remain unchanged.

## Predeclared comparisons

- Six variants, eight continuation epochs each, one seed37: AR alone,
  AR+deterministic state, AR+stochastic state, at each of two AR depths.
- Small AR retains the existing 4 layers (4.603M); large adds eight initially
  identity residual blocks (12 layers, 13.030M). Each hybrid adds 6.324M state
  parameters, totaling 10.927M and 19.354M. The added blocks initially preserve
  the old eval-mode function exactly. Larger depth also increases the effective
  text receptive field; this is a depth experiment, not a pure width comparison.
- Every AR starts from the **same** existing band2 checkpoint. Both noisy and
  deterministic hybrids start from the **same** trained state_s37 checkpoint.
  The deterministic variant holds functional noise at zero; the stochastic
  one samples a new32D noise shared across the forecast-state normalization
  layers/frames at each step. Noise is not added directly to token-AR weights.
  The state generator stays6.324M at both sizes: capacity is added to the AR.
  This does not directly test scaling the FGN generator itself.
- 28,539 train-clean-100 utterances per epoch. One random supervised token and
  a strictly earlier audio cutoff per utterance. All six share sample orders,
  token/cutoff draws, optimizer settings and learning rates. Hybrid branches use
  two samples; this matches optimizer steps, not FLOPs or supervised objectives.
- Joint objective: average branch token CE (label smoothing0.1) +0.1 fair CRPS
  on normalized continuous future encoder states. AR-only has token CE only.
  Both samples are supervised on the next reference token. The CRPS-versus-CE
  tradeoff is fixed in advance; whether this objective retains useful noise
  diversity is diagnosed from held-out feature spread and branch agreement.
- Rollouts H1 for epochs1–2, H2 for3–4, H4 for5–8. Token AR sees at most the
  first two generated chunks; the continuous state loss covers the whole rollout.
  No future observed feature is an input to the forecast or token AR.
- Select by un-smoothed next-token NLL on256 fixed speaker-partitioned
  model-selection utterances. Epoch0 is eligible; no forced degradation from
  requiring a newly trained checkpoint. Gate-fit1297 and calibration1414 are
  separate from model selection. Test is not used for architecture/thresholds.
- Record identical fixed256 train-probe utterances and dev-clean/dev-other
  separately every epoch. A train-versus-mixed-dev gap would confuse domain
  difficulty with overfitting; the primary gap uses **dev-clean minus train-clean**.
  A smaller gap alone is not proof of improvement if both losses get worse.
  No observed overfitting means prevention cannot be concluded.
- Native-prefix streaming inference, not gold prefixes. Confirmed text KV
  retains3 positions; prior text nodes never acquire future predicted audio.
  Only a temporary copy of the last token is evaluated against generated audio.
  Four stochastic branches vs one deterministic branch, two future chunks.
  Teacher-forced training builds preceding text states against arrived audio at
  the selected cutoff; deployment retains those states from historical native
  emission times. Both exclude generated audio from confirmed preceding states,
  but this does not establish exact teacher-forced/streaming state equality
  across changing acoustic cutoffs.
- All new gates receive the same11 features: original probability/prefix8 plus
  branch agreement, predictive entropy minus mean member entropy, and future
  feature spread. For AR-only the last3 are constant0. Separate model-specific
  gates are fit/calibrated with the same dev-only protocol and0.5 false forecasts
  per100 native BPE tokens budget. Actual test error rates may differ.
- Existing native/full/band2 gates, events and timings are preserved exactly.
  Final native transcript and WER stay fixed by construction. Timing is a
  simulated-arrival replay plus measured incremental CPU work, not a live UI.
- The hybrid retains growing audio cross-attention KV. It does **not** inherit
  the previous state-only model's33KiB total forecaster-context claim.
- One continuation seed and a limited depth range cannot establish a scaling
  law or general overfitting protection. Earlier public test sets have already
  been inspected across this research; this remains exploratory benchmarking.

The first completed small-AR control used full prefixes. Remaining variants use
an exact receptive-field computation optimization:2L+1 raw tokens for L band2
layers (9 at4 layers,25 at12). Eval-mode differences versus full-prefix computation
were below4e-6 in the recorded check. This preserves the architecture, but
training dropout-mask draws depend on tensor shape. Det/noisy hybrids use the
same optimized path. The brief uncompleted pre-optimization deterministic run
is archived under `runs/band_fgn_v1/throughput_pilot/`.

Independent variants may train concurrently on MPS when memory has headroom.
A per-variant file lease prevents duplicate writers. Per-model epoch times from
overlapping jobs are wall times under contention, not isolated GPU benchmarks.

## Reproduction

Use `.local/venv-fgn-state/bin/python` with scripts in this folder:

1. `prepare.py`: lightweight gold token/timing metadata, reusing existing audio
   features and packed encoder cache. Store pre-test plan and input hashes.
2. `check.py`: future-feature perturbation/gradient, cross-model coupling,
   differentiable rollout, identity expansion and confirmed-cache isolation.
3. `run.py --stage train`: all six training jobs sequentially on MPS.
4. `run.py --stage evaluate`: two-utterance smoke, full dev collection,
   calibration, then5,559 test utterances, paired analysis and report.

Files are under `runs/band_fgn_v1/`. The final report and small artifacts are
stored in this folder's `artifacts/` and `docs/band_fgn_hybrid_2026-11-12.md`.

## Original-paper interpretation

[FGN](https://arxiv.org/html/2506.10772v1) models uncertainty through learned
functional perturbations and independently trained models. It is not a theorem
that noise prevents overfitting. Section2.2.3 explicitly mentions reducing
geopotential loss weights to tame overfitting when scaling the model. AppendixA.4
reports improvements from model capacity/ensembling in weather, not proof that
our categorical ASR display task will benefit. Our det-versus-noisy comparison
tests an incremental generalization hypothesis for this particular hybrid.
