# Research goal and methodology

Initial scope: 2026-11-10. Current direction updated: 2026-11-12. This document distinguishes the user's objective, completed experiments and proposed implementation.

## Current selected goal

Predict the next acoustic representation from retained state. When actual speech arrives, correct the prediction using a lightweight audio-conditioned path. Use full encoding when necessary, with all verification, state maintenance and recovery costs included.

**Research question:** Can prediction plus inexpensive acoustic correction reduce total streaming ASR computation at controlled recognition error and bounded retained state, while recovering reliably from unexpected speech?

The user also proposes a positive role for prediction: a well-trained prior may improve recognition when audio is ambiguous or noisy. The model must still follow clear acoustic evidence when the speaker uses an unexpected word. These are complementary hypotheses, not established findings.

The [predict–correct acceleration plan](predict_correct_acceleration_plan_2026-11-12.md) defines the proposed architecture, objectives, capacity comparisons, causal gating and state-recovery requirements. Residual correction and encoder skipping are not yet implemented. The [current conclusions](conclusion_and_next_steps.md) and [experiment history](research_plan_3stage.md) provide completed evidence.

## Original predict–verify objective

Use observed speech and the recognized prefix to prepare possible continuations while waiting for new input. Verify those candidates using newly arrived audio; reuse valid work and recover from disagreement. Reduce committed-transcript latency and unnecessary sequential computation at matched recognition quality.

Three capabilities require separate evaluation:

1. Recognizing speech already received.
2. Predicting speech or text not yet observed.
3. Reusing a prediction to reduce verified inference cost or commitment latency.

An improvement in provisional display does not establish an improvement in the third capability.

## Preparation and implementation status

| Item | Status |
|---|---|
| Task | English streaming speech-to-text |
| Hardware used for local studies | Apple M5 Pro, 24GiB unified memory |
| Data | LibriSpeech training splits and all four dev/test splits downloaded and checked |
| Local training union | 281,241 utterances / 961.054 hours, excluding dev/test |
| Initial acoustic model | 17.28M chunk-causal Conformer with CTC/AED; trained on 100h |
| Strong public backbone | Frozen 66.11M Streaming Zipformer trained externally on 960h; evaluated locally |
| Trained prediction modules | Full/band2 AR and recurrent FGN state models; independent and actual hybrid comparisons |
| Completed primary benefit | Earlier provisional display; compute acceleration not demonstrated |
| Approximately 114M/960h acoustic model | Separate unexecuted recipe proposal |
| Current research extension | Inexpensive audio-conditioned correction and conditional encoding; proposed |

FGN provides a continuous-state forecasting and rollout inspiration. Neither functional noise nor a particular ensemble is mandatory for the predict–correct objective. Scaling and overfitting effects require their own controls.

The completed [generator-capacity diagnosis](fgn_capacity_2026-11-12.md) compares 6.32M / 10.54M / 18.96M from identical trained functions. All overfit the fixed-case task; added depth slightly improves fitting without an observed dev benefit in this seed. This is separate from training a raw-audio corrector or establishing a noise-regularization effect.

The [epoch-27 continuation](fgn_capacity_long_2026-11-13.md) restores optimizer/model states and adds 16 epochs to every size without overfit stopping. All final train-probe NLLs are about 0.131–0.132, while dev NLL worsens and the selected checkpoints remain at epoch 5. This confirms additional fitting within the fixed-case task, not a validation improvement or a test of width/memory growth. Actual SiLU settings are audited alongside the results.

## Causal procedure

1. Keep observations and confirmed state separate from speculative state.
2. Predict from information available at the current cutoff only.
3. Retain useful speculative work under an explicit state/branch budget.
4. Process newly arrived audio through the inexpensive observation path.
5. Correct or reject the prediction using that audio. Decide on expensive computation before performing it.
6. Update a consistent streaming state; charge fallback, replay and discarded work.
7. Generate the next prediction from the corrected state used in deployment.

The current token forecaster consumes full encoder output; its current implementation is not already step 4–6 of this proposal. Full-encoder cache recovery must be designed explicitly before claiming valid encoder omission.

## Output and cost definitions

- **Provisional output:** may be revised after more acoustic evidence.
- **Committed output:** satisfies the specified acoustic verification and commitment policy; false commitments are counted separately.
- **Reuse:** measured elimination or batching of target-model work, not merely generation of a candidate.
- **Critical-path reduction:** work moved before input arrival; total work may still increase.
- **Compute acceleration:** lower complete measured processing cost, including prediction, correction, gate, state updates and recovery.

## Hypotheses

| Hypothesis | Supporting result | Failure signal |
|---|---|---|
| Prediction saves computation | Complete matched-quality processing cost falls | Added prediction/recovery costs exceed avoided work |
| Audio-conditioned correction is inexpensive | Corrected representation and recognition recover at low cost | Accurate correction requires a path as expensive as full encoding |
| A prior helps ambiguous speech | Better reference WER than a matched-budget audio-only path | Only teacher agreement improves, or gains disappear under controls |
| Unexpected speech remains recoverable | Clear names, numbers and topic changes override a wrong forecast | Plausible prior replaces the actual words or errors persist |
| State remains usable | Long-stream quality and bounded state survive repeated cheap updates | Cache drift, uncharged replay or growing memory |
| Scaling is useful | A larger predictor/corrector improves the error–compute frontier | Lower loss produces no useful system improvement |

The initial Stage 1–3 H1–H4 and later G/H experiment labels retain their historical meanings in the experiment history. The hypotheses above describe the new extension and do not relabel completed tests.

## Baselines and controls

Historical AVS comparisons include CTC greedy/beam, text-only/audio-aware priors with and without anticipation, parallel-token FGN variants, short-context AR and display gates. Their detailed B0–B4 definitions and matched conditions remain in the history.

For the new extension, include full encoding, a cheap audio-only path, prediction alone and prediction plus correction. Compare predictor and corrector capacity separately. Charge ensemble branches, and distinguish the teacher's external training from local added training.

Keep tokenizer, audio preprocessing, chunk/lookahead, training budget, endpointing and evaluation normalization fixed where comparisons require it. A public model trained on 960h and a local module trained on 100h are different training scopes.

## Data and labels

Preserve train/dev/test utterance boundaries before sampling temporal cutoffs. Do not split prefixes of the same utterance across roles. Fit tokenizers and normalization on training data only.

Use full teacher states, future audio and alignments for offline supervision or evaluation only. They cannot enter an inference-time decision. Inspect normalization, centered operations, attention masks and streaming caches for future leakage.

Development data has stored speaker-disjoint roles for model selection, gate fitting and threshold calibration. Freeze policies before test evaluation. Report actual test risks instead of assuming a dev-calibrated threshold makes test error equal across methods.

## Measurements

| Metric | Definition and boundary |
|---|---|
| WER/CER | Final reference transcription error under fixed normalization |
| Stable provisional display | When a correct final word first appears and remains in the replayed output |
| Commit latency | Independent word-end reference to actual commitment event, with false commitments |
| Forecast acceptance and error | Reference correctness and native acceptance reported separately |
| RTF | Processing time / audio time; single-stream and throughput measurements distinguished |
| Component time | Encoder, prediction, correction, gate, decoder, fallback and recovery |
| Resources | Peak memory, retained cache/state, discarded work and branch count |
| Recovery | Time and computation after a wrong prior or state error |

Use hardware-appropriate synchronization, identical utterances and paired uncertainty estimates. Choose quality tolerances and repetition budgets before experiments. Forced-alignment timestamps are estimates; do not conflate them with independently measured phonetic boundaries.

The initial small CPU profile places approximately 98.3% of native neural time in the encoder. It motivates the current target but is not a live-device measurement or a general property of all ASR models.

## Prior work and claim boundaries

[SSR](../notes/anticipatory/P01-ssr.md), [Speculative Listening](../notes/anticipatory/P02-speculative-listening.md), [SpecASR](../notes/decoding/P03-specasr.md), [FGN](../notes/weather/P21-fgn.md) and conditional-encoder research cover related concepts. Prediction, residual learning, gating and speculative verification are not individually new claims.

Potential novelty must come from a concrete correction/state-recovery method and a measured quality–compute advantage over strong controls. The new plan has not established either. Existing provisional-display gains, withdrawn two-pass numbers and unsuccessful FGN extensions remain documented rather than being recast as compute acceleration.

[Literature map](literature_map.md) · [Claim audit](claim_audit.md) · [Data guide](../data/README.md) · [Current plan](predict_correct_acceleration_plan_2026-11-12.md)
