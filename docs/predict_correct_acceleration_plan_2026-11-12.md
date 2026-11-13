# Predict–correct encoder acceleration

Status: research direction selected on 2026-11-12; the residual corrector, computation gate and encoder-skipping runtime have **not** been implemented or trained. Existing display experiments remain completed results, not evidence of encoder acceleration.

## Goal

Predict the next acoustic representation from a bounded history. When the actual audio arrives, use a lightweight audio-conditioned module to correct the prediction. Run the full encoder only when the inexpensive path is insufficient. Measure the complete cost of prediction, correction, verification, fallback and state recovery at controlled recognition error.

The intended benefit has two parts:

1. Reduce actual inference computation by reusing a useful prediction.
2. Use a learned prior to help recognize ambiguous or noisy speech, while retaining the ability to recognize unexpected words.

A forecast is a provisional hypothesis. It must not restrict recognition to a fixed next word or candidate list. The corrector must be able to disregard the forecast, and the full recognition path must remain available.

## What already exists

| Component | Status | Evidence |
|---|---|---|
| Public 66.11M streaming Zipformer | Evaluated; frozen weights | [Baseline reproduction](baseline_zipformer_2026-11-11.md) |
| 4.60M layer-band2 AR forecaster and confidence gate | Trained on 100h; full test evaluation completed | [Strong-backbone experiment](zipformer_ours_2026-11-12.md) |
| 6.324M recurrent FGN acoustic-state predictor | Trained; independent and hybrid evaluations completed | [State pilot](fgn_state_pilot_2026-11-12.md), [hybrid study](band_fgn_hybrid_2026-11-12.md) |
| Raw-audio residual corrector | Proposed | No completed implementation or measurements |
| Gate selecting inexpensive versus full encoding | Proposed | Existing gates control display, not safe encoder omission |
| Encoder-cache correction and recovery | Required design work | Predicting encoder outputs does not reconstruct all internal caches |
| End-to-end compute acceleration | Not established | Existing forecasters add processing time |

The surviving display configuration is band2+gate. Its correct-word mean display gain over native eager output is 89.58ms on test-clean and 70.28ms on test-other. Its incremental gain over the inexpensive native-probability gate is 30.32/16.54ms. The median per-word gain is zero; final WER is fixed by design. Actual test forecast error rates differ between policies, so matched-risk Pareto superiority is not established.

## Measured bottleneck

A batch-1, one-thread CPU profile used 12 randomly selected dev-clean and 12 dev-other utterances, seed 37, with two warmup utterances excluded from timing. There were 431 chunks and 129.2300625 seconds of audio.

| Native neural component | Mean ms/chunk | Share of neural execution time |
|---|---:|---:|
| Encoder | 8.8421 | 98.3159% |
| Prediction network | 0.0751 | 0.8348% |
| Joiner | 0.0764 | 0.8493% |

This is a small adapter profile, not a live microphone/UI benchmark, mobile result, or isolated repeated performance estimate. File loading and feature extraction are included in wall time but excluded from the neural-component denominator. The original record is [profile_initial.json](../experiments/compute_profile_v1/artifacts/profile_initial.json); [profile_zipformer_compute.py](../scripts/profile_zipformer_compute.py) reproduces the measurement protocol.

Eliminating all prediction-network and joiner execution would save only about 1.68% of native neural time in this sample. Pure token speculation therefore has a small ceiling on this particular backbone unless it also reduces encoder work.

The previously quoted 1.3–1.4ms/chunk is the AR forecaster cost. It is **not** a measured cost for the proposed FGN-plus-corrector path. Existing stochastic hybrids are substantially more expensive; costs from different runs are not an isolated speed comparison.

## Proposed causal architecture

For chunk t, let x_t be newly arrived audio, s_(t-1) the retained state, and z*_t the full encoder's representation.

```text
retained state -> existing FGN-style predictor -> predicted representation
                                                        |
new audio -> inexpensive causal audio module -> residual corrector
                                                        |
                                      corrected representation + state
                                                        |
                                computation gate before full encoding
                                      /                         \
                          inexpensive path                full encoder
                                      \                         /
                                  recognition and state update
```

One candidate parameterization is:

\[
\widehat z_t=P_\theta(s_{t-1}),\qquad
u_t=A_\phi(x_{\leq t}),\qquad
\widetilde z_t=\widehat z_t+C_\psi(u_t,\widehat z_t,s_{t-1}).
\]

A gated mixture or replacement path is an alternative if additive correction anchors the model too strongly. The cheap audio path must obey the same chunk/lookahead contract as the baseline.

The current token forecaster consumes full encoder output. Its current-chunk gate cannot decide whether to avoid a full encoder execution that has already happened. The new decision must use prior state and inexpensive current audio features. The existing weights and bounded-context design are starting points, not a drop-in encoder replacement.

## Training objectives

The existing hybrid minimizes token cross-entropy plus 0.1 times acoustic-state CRPS. It does not train the audio-conditioned correction task above.

For the first correction pilot, separate the following objectives and report each component rather than interpreting a weighted sum as an absolute quality measure:

- **Corrected-representation loss:** compare the corrected output with full-encoder targets, using train-only normalization.
- **Recognition loss:** use reference transcripts so the cheap path can, in principle, correct a teacher error rather than merely copy it.
- **Token-distribution distillation:** optionally preserve useful behavior of the frozen teacher. This follows the general [knowledge-distillation approach](https://arxiv.org/abs/1503.02531), not a demonstrated result for this proposed architecture.
- **State and rollout consistency:** validate the state used by subsequent chunks, including any caches the cheap path replaces. Train with corrected-state feedback if that is the deployed recurrence.

An illustrative objective is:

\[
\mathcal L=\alpha\mathcal L_{\rm representation}
+\beta\mathcal L_{\rm reference}
+\gamma\mathcal L_{\rm distillation}
+\delta\mathcal L_{\rm state}.
\]

Weights, optimizer, learning rate and training duration are not finalized. Cost-aware gate fitting follows an adequate cheap path and uses development data only. A loss decrease alone does not establish lower WER or faster inference.

Past-only prediction has irreducible uncertainty about new speech. A larger predictor may reduce approximation error, but cannot reveal phonetic information that is absent from its inputs. The inexpensive current-audio path supplies that missing information.

## Prediction as a useful prior, and its failure modes

The user explicitly proposes that a well-trained forecast may correct ambiguous speech. This is a hypothesis to test alongside the risk of anchoring to an expected word.

Compare, under matched compute and training budgets:

| Condition | Desired behavior | Required comparison |
|---|---|---|
| Noisy or partially masked audio | Use context to recover the actual words | Cheap audio-only path versus prediction plus correction |
| Clean but surprising speech | Follow the audio and override the forecast | Unexpected names, numbers and topic changes |
| Incorrect prior or corrupted retained state | Detect disagreement and recover | Deliberately wrong forecasts and state perturbations |
| Long streams | Avoid accumulating state errors | Repeated cheap-path use and realistic resets |

Do not use a fixed candidate vocabulary that excludes an unexpected correct token. Measure errors where a correct audio-driven word is replaced by a plausible but wrong forecast. Report recovery delay and all replay/re-encoding costs. Reference transcripts, alignments and future teacher outputs may supervise training or offline evaluation, never an inference-time decision.

## Cost condition and correctness boundary

If q is the fraction of full encoder executions omitted, a simplified break-even condition is:

\[
q C_E>C_P+C_{\rm cheap\ audio}+C_{\rm correction}+C_{\rm gate}
+\mathbb E[C_{\rm recovery}].
\]

Adjust the accounting if the inexpensive path shares a prefix with full encoding, so shared work is counted once. Moving prediction into audio waiting time can reduce critical-path latency without reducing total work; measure both.

Token confidence and ensemble agreement do not certify encoder approximation error. For a fixed token context, a verified representation error bound epsilon and an L-Lipschitz logit map would preserve an argmax if its predicted top-two margin exceeds 2L epsilon. Neither a cheap certified bound nor its usefulness has been established here. A single-frame argmax condition also does not ensure future cache correctness or distribution-preserving stochastic decoding.

Approximate cache updates can accumulate error. Falling back to an old full-encoder cache may require replaying omitted audio. That recovery is not free. Output-only correction experiments are a first feasibility check, not proof of a working encoder-skipping recognizer.

## Scaling decision

In the completed hybrid study, AR grew from 4.60M to 13.03M while the FGN generator remained 6.324M. Mixed-development token NLL changed from 1.0687 to 1.0450 for the stochastic hybrids, while display gains worsened. This is not a direct capacity experiment on the FGN generator.

A subsequent [generator-depth diagnosis](fgn_capacity_2026-11-12.md) directly grew FGN to 6.32M / 10.54M / 18.96M with AR fixed at four layers. All met the overfit rule at continuation epoch 11. Train-probe NLL changed 0.2288 → 0.2228 from smallest to largest, while best dev NLL did not improve (0.9486 → 0.9504). This single-seed study uses one fixed case per training utterance; it favors examining cutoff coverage and the state/token loss balance before another size increase. It does not test the new corrector or settle capacity needs under broader training.

The proposed order is:

1. Reuse the approximately 6.3M state predictor and add a small audio-conditioned corrector. A 1–3M corrector is a candidate budget, not an implemented model or a measured speed estimate.
2. Establish correction quality, causal input availability and state consistency. Increase training cutoff coverage and inspect train/development curves before diagnosing insufficient capacity.
3. Compare approximately 6M versus 12M predictors and corrector capacity separately, holding data, training budget, noise settings and evaluation fixed.
4. Expand data or capacity when the larger configuration improves the error–compute frontier after prediction, gate and recovery costs.

Do not use a larger AR alone as evidence about FGN scaling. Noise and ensemble branches must pay for their own runtime cost. A large research teacher can be useful, but the deployed prediction/correction path must still satisfy the compute budget.

## Data and evaluation sequence

Start with existing LibriSpeech train-clean-100 and the stored speaker-disjoint development roles. Retain test-clean/test-other for final frozen-policy evaluation. Existing 960h data is available, but a new 960h acoustic model is not required to establish the first correction feasibility result.

Before running the new study, freeze model variants, training cutoffs, seed budget, checkpoint-selection rule, WER tolerance, gate-calibration policy and timing protocol. This document does not authorize a claim that those experiments have run.

Measure final WER, reference-based forecast errors, teacher disagreement, full-encoder usage, total RTF, per-chunk P50/P95 compute, recovery/replay time, peak memory and cache growth. Include full encoder, cheap audio-only, prediction-only and prediction-plus-correction controls. Calibrate on dev, report actual test risk, and use paired uncertainty estimates. Separate accuracy gains on noisy speech from failures on unexpected speech.

## Related-work boundaries

- [FGN](https://arxiv.org/abs/2506.10772) motivates recurrent continuous-state forecasting and probabilistic trajectories; it does not demonstrate this ASR corrector or guarantee resistance to overfitting.
- [EAGLE-ASR](https://arxiv.org/abs/2610.13506) accelerates autoregressive AED recognition with speculative verification. Its reported speedups do not establish a transducer encoder speedup here.
- [FSR](https://www.isca-archive.org/interspeech_2021/tian21_interspeech.html) predicts blank locations to skip transducer decoding work; it does not by itself remove the encoder bottleneck measured above.
- [Dynamic Encoder Transducer](https://www.isca-archive.org/interspeech_2021/shi21b_interspeech.html) studies encoder-depth choices and accuracy/latency trade-offs. It is a relevant comparison for conditional encoding, not evidence that the proposed policy works.

The potential contribution is a particular efficient correction and state-recovery method with measured gains. Prediction, residual learning, confidence gating and conditional computation are not individually claimed as new.

[Current conclusions](conclusion_and_next_steps.md) · [Research goal](goal_and_methodology.md) · [Repository map](../READ.MD)
