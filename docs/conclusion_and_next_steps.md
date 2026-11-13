# Current Conclusions and Next Experiments

Snapshot date: 2026-11-12, America/Los_Angeles. This is the current-conclusion entry point. Initial numbers were checked against saved outputs and code; four follow-ups and [our strong-Zipformer study](zipformer_ours_2026-11-12.md) were actually executed. [History](research_plan_3stage.md) preserves detailed results; [implementation review](results_review_2026-11-11.md) explains initial corrections.

## Selected direction: prediction plus acoustic correction

The user selected a new target on 2026-11-12: predict an acoustic representation, correct it with inexpensive processing of actual incoming audio, and avoid full encoder work when recognition quality permits. A learned prior may help ambiguous speech, but must yield to unexpected names, numbers or topic changes. Reference-based training and explicit recovery tests are required.

The [current acceleration plan](predict_correct_acceleration_plan_2026-11-12.md) records the architecture, losses, state/cache requirements, scaling sequence and comparisons. **The residual corrector, compute gate and encoder-skipping runtime are not yet implemented or trained.** Completed display results below do not establish this new hypothesis.

An initial 24-utterance development CPU profile measured encoder/prediction-network/joiner work as 8.8421/0.0751/0.0764ms per chunk; the encoder took 98.3159% of native neural time. A standalone reproduction again measured about 98.3%. See [profile records](../experiments/compute_profile_v1/README.md). This is a small batch-1 adapter measurement, not a live-device benchmark. The prior 1.3–1.4ms figure belongs to the AR forecaster, not the proposed FGN/correction path.

Start correction experiments with the existing approximately 6.3M state predictor and a small audio-conditioned corrector; compare their capacities separately. The completed large hybrid increased AR depth while leaving FGN-generator capacity unchanged. Lower NLL alone does not establish lower WER, less overfitting or faster inference.

## Completed experimental conclusions

**Actual AR+FGN joint training is complete:** [Six hybrid/control configurations](band_fgn_hybrid_2026-11-12.md) received 100h × 8 continuation epochs and full 5,559-utterance test evaluation. Small/large hybrids have 10.93M/19.35M parameters. Small stochastic gain was 88.46/70.18ms versus same-run AR's 89.63/70.30ms; large stochastic gain was 81.91/65.08ms. Added CPU cost increased approximately 1.3 → 6.6~6.8 → 11.9~12.2ms/chunk. Small-hybrid clean difference −1.17ms has an interval excluding zero; other −0.12ms includes zero. Actual test forecast risks differ.

Noise lowered the dev NLL by about 0.010/0.014 compared to the deterministic combination of the same size and also slightly reduced the train-dev gap. However, in both methods, the learning and validation losses went down together, and no pattern was identified that suppressed the obvious worsening of overfitting. This is a warm-start experiment with one seed, the target for expansion is AR, and the FGN generator is fixed at 6.324M. **It is not used as a direct verification of the scaling or general overfitting prevention effect of the FGN generator itself. The basic plan maintains the existing band2+gate.** Distinguish between the previous [separate continuous state model ](fgn_state_pilot_2026-11-12.md) and this actual combination.


**The demonstrated benefit is earlier provisional display in single-pass replay. A recurrent FGN predictor has not demonstrated better overall recognition or compute efficiency than full-context AR.**

**Our strong-backbone study is complete:** Freeze public 960h Zipformer 66.11M and train 4.60M full-context/band2 forecasters from the same random initialization for 100h × 15 epochs. Evaluate all 5,559 test utterances with recognized native prefixes. Fit gate weights on dev-fit and thresholds on separate dev-cal. Positive labels require native acceptance and reference BPE/complete-word correctness.

| configuration | clean average display gain ms | other average display gain ms | test false forecast /100native tokens, clean / other |
|---|---:|---:|---:|
| Native probability+gate, no added forecaster | 59.27 | 53.74 | 0.335 / 0.507 |
| Our full+gate | 94.31 | 75.31 | 0.397 / 0.583 |
| Our band2+gate | 89.58 | 70.28 | 0.375 / 0.537 |

Gains are **paired mean stable-display differences for the same correct final words** relative to native eager display. Simulated arrival includes measured forecaster/gate CPU cost, not live microphone/UI/contention. Band2 advances 28.03% of correct clean words and 21.98% of other; median per-word gain is 0ms. WER 3.058/7.820% is fixed by unchanged native transcripts, not improved recognition.

The additional average gain compared to the native probability + gate of band2+gate was **30.32ms / 16.54ms**, and both utterance and speaker bootstrap CI were greater than 0. However, since the actual test error rate is not the same as in the table above, it is not said that Pareto improvement has been proven for the same test error. band2 had a smaller average gain than 4.73/5.03ms full, and the CI of this difference excludes 0. The text KV/logp was reduced from the full maximum 2.20MiB to band2 29.9KiB, and the additional forecast cost was about 1.4ms/chunk. Audio KV/output metadata continues to increase.

Band2+gate character deletions occur 0.207 times per 100 final words on clean and 0.281 on other. Word-suffix revisions include normal extension such as `CON → CONFIDENCE`, so they are not an incorrect-word revision rate. This post-evaluation diagnostic preserves weights, thresholds and primary results. The conclusion is added forecast-display value on one strong backbone/seed; [the complete report](zipformer_ours_2026-11-12.md) contains numbers, intervals, precision and display diagnostics.

**Reproduced previous external baseline:** FP32·320ms chunk-wise greedy WER of public LibriSpeech 960h Streaming Zipformer 66.11M was 3.058/7.820% in the entire test (official 3.06/7.79%). We distinguish between that reproduction step and further experiments in our example above. The effects of 960h acoustic learning and FastEmit were not directly verified. The difference in WER from the existing 100h AVS is not interpreted as an effect of the proposed method. [Official Baseline Experiment Record](baseline_zipformer_2026-11-11.md)

**Latest follow-up verification:** The main candidate for short context is a model that references the current and previous 2 positions in each AR layer. Turning off the actual text KV/logp did not change the transcription in the 200 utterance × 3chunk and the four synthetic long-duration streams. The text storage held after a chunk of the entire beam was about 90–110KiB, in the short evaluation and about 70–90KiB in the long-term snapshot. Audio K/V and token metadata are incremented, thus distinguishing them from overall state fixation. A total of three seed comparisons were completed, including additional seeds 3·4. The seed average WER difference (limited by layer−total AR) for the 9 condition was −0.015…+0.131%p, and the test-clean 320ms was 14.534% vs. 14.537%. It is not proof of equivalence. [Check the entire table and limits in the next four experiments](followup_experiments_2026-11-11.md).

**Prior strict window comparison:** W1a, which learned the 3 token window with the same decoder and initial value as the full context AR, maintained about 85–96% of the WER improvement compared to AR's B0. In test-clean / 320 ms, WER is 14.75%, which is the difference between the full context 14.55% and +0.20%p. The failure of the initial slot-type short model (16.40%) below is not generalized to the failure of the entire short context AR. The cache increment calculation also newly confirmed that masked forward and transcription match under the 100 utterance × 3 condition.

Single-pass forecasting advances stable display in screen-event replay without changing the final transcript. A learned gate also allows more early tokens at comparable error levels. Temporary incorrect displays/retractions remain; unchanged final WER does not guarantee correct intermediate display or faster irreversible commitment.

Much current benefit recovers delayed emissions of already received audio. Evidence does not support long-horizon pre-utterance prediction that simultaneously reduces ASR compute and irreversible commitment latency.

## Judgment by hypothesis

| Research item | Current judgment | Analysis range |
|---|---|---|
| Reuse after pre-calculation | Unconfirmed gain of original goal | In the existing small model, the confirmation time of the same chunk is almost the same, and when additional prediction calculations are added, the total computation increases. |
| Single pass early display | Valid in screen event playback | Final WER maintained, stable P50 Approximately 40–80 ms reduced. Distinguish between provisional display and irreversible confirmation. |
| Gate writing independent model signal | Valid additional information Observation | Early indication increase in the same retraction rate curve. This is an effect of an independent model signal and has not been separately proven as an FGN-specific noise effect. |
| 2-pass P50 48 ms / P95 209 ms | Withdrawal | Was not reproduced in screen playback reflecting the actual display merge. |
| Initial FGN-style short-context model | Higher WER than AR | Better than no prior, worse than full-context AR; step computation did not decrease. |
| Same-architecture 3-token AR window | Approaches full-context benefit | Same initialization and 15 epochs preserve about 85–96% of the improvement over B0. WER is usually slightly higher; acceleration is not established. |
| Per-layer 2-position AR with actual KV release | Bounded text tensor cache verified | Released/unreleased transcripts and committed-node scores match. Four layers give at most 9 raw text tokens of receptive field; audio history, metadata and total work are not all bounded. |
| “3 token is enough” | Context expansion gain not observed in this structure | The figures of P=3 and P=32 are similar. Statistical equivalence and general sufficiency have not been proven. |
| Complete state of 24 bytes per hypothesis | Claim narrowed | Counts 3 int64 token IDs only, excluding total memory, logp, trie and audio state. |
| noise and rollout fine-tune | Additional gain unconfirmed in current settings | This is the result of fine learning of the 15 epoch and 3 epoch of the discrete token model. This is not a falsification of all FGN/continuous state models. |

## Representative figures from existing 17.28M·100h experiments

### Single pass and replay screen events of 2-pass: test-clean 300 utterance

| Configuration | Final WER % | Stable P50 ms | Stable P95 ms | Cumulative Rewrite/Final 100Word |
|---|---:|---:|---:|---:|
| single 320 ms | 14.24 | 208.7 | 368.0 | 0 |
| single 320 ms + forecast | 14.24 | 129.0 | 328.0 | 1.22 |
| single 640 ms | 12.78 | 370.5 | 654.5 | 0 |
| 160 → 640 ms | 12.78 | 127.2 | 327.5 | 21.21 |
| 160 + forecast → 640 ms | 12.78 | 86.8 | 296.4 | 34.64 |

Sources: `runs/final/summary2b.json` (local-only), [screen replay](../scripts/analyze_final2.py). Stability starts when the correct final word appears and remains. Rewriting accumulates suffix words after the common prefix, not unique revised words. Do not infer model improvement by comparing this 300-utterance WER with earlier 500-utterance WER. Excludes actual-device rendering/two-pass contention.

### Learning gate: test-clean 500 utterance / 320 ms

Compare interpolated evaluation-set threshold curves at 0.5 retractions per 100 tokens.

| Gate | Successful Early Mark /100Token | Average Delay ms |
|---|---:|---:|
| AR probability only | 13.50 | 153.99 |
| All signals | 15.93 | 144.04 |
| Excluding NAR/FGN signals | 14.47 | 147.98 |

Full gate improves mean latency over raw by 9.95 ms; removing independent-model signals still gives 6.01 ms. Their difference is 3.94 ms. After correction, most benefit cannot be attributed to FGN in this condition. Source: `runs/final/summary2.json` (local-only).

### Initial FGN short context model: test-clean 500 utterance / 320 ms

| Model | Final WER% |
|---|---:|
| no prior | 18.09 |
| P=3 Determinism | 16.40 |
| P=32 Determinism | 16.69 |
| P=3 noise model | 16.26 |
| P=3 determinism + rollout fine-tune | 17.29 |
| Full-context AR | 14.55 |

P=3 determinism obtains approximately 48% of the WER improvement compared to B0 obtained by AR under this condition. In all saved 12 conditions, the WER of P=3 is lower than B0, higher than B1a, and the directions of the two paired bootstrap CIs were also confirmed. Hypothetical 4 step times were about 1.3 ms for the short model and about 1.05 ms for the AR in single CPU measurements, so no computation acceleration was observed. Source: `runs/stage3/short/summary.json` (local-only), `runs/rollout_eval_lead2.json` (local-only).

### Recently added: 3 token window with the same AR structure and limited to initial values.

Comparison uses the same decoder, Stage 1 initialization, 15 epochs and prior weight 0.5. Each position is recomputed within a strict 3-token window so layers cannot expand text context. Each split has 500 utterances.

| split / chunk | B0 WER % | 3 Token AR WER % | Full Context AR WER % | Window − All, 95% CI %p |
|---|---:|---:|---:|---|
| test-clean / 160 ms | 20.00 | 16.73 | 16.50 | −0.06 … +0.54 |
| test-clean / 320 ms | 18.09 | 14.75 | 14.55 | −0.07 … +0.47 |
| test-clean / 640 ms | 16.86 | 13.65 | 13.09 | +0.30 … +0.82 |
| test-other / 320 ms | 42.04 | 37.21 | 36.82 | +0.03 … +0.73 |

The improvement of test-clean / 320 ms compared to B0 remains approximately 94.4% with the full context 3.54%p, 3 token window 3.34%p. Since B0 uses beam search in the CTC greedy, prior experiment, the overall improvement over B0 is not interpreted as a contribution to predictor deception. The contrast between window/full AR is the same navigation condition. Even if the confidence interval includes 0, it does not mean that the criteria for equivalence or a priori non-inferiority have been passed.

In test-clean / 320 ms, the critical-path predictor cost was window 2.066 ms/chunk and overall 1.601 ms/chunk. The overall RTF record is 0.0212 vs. 0.0233, but without the uncertainty of repeat measurements, it does not confirm meaningful computational acceleration. The implementation reported states 0.2 kB vs. 69.7 kB are also different calculation ranges using `window*8` and are not used as actual total memory reduction rates.

Sources: `runs/stage3/window/summary.json` (local-only), [WindowPredictor](../avs/stream.py), [training](../scripts/train_stage2.py). Supports potentially small accuracy loss from short context. Architecture versus initialization does not separately explain differences from the old slot model. One seed; 2/4/8-token windows were not yet compared at that stage.

### Recently added: Incremental compute time using cache

test-clean 100 utterance, calculated by updating the actual encoder cache in CPU 1 thread. In each condition, masked forward and transcription 100/100 were consistent. The same 100 utterance was evaluated under three conditions, and it is not 300 independent utterances.

| chunk | step P50 ms | step P95 ms | RTF | delayed start chunk number |
|---|---:|---:|---:|---:|
| 160 ms | 6.29 | 8.56 | 0.0413 | 1 / 4,440 |
| 320 ms | 7.49 | 10.79 | 0.0246 | 0 / 2,244 |
| 640 ms | 10.27 | 15.55 | 0.0169 | 0 / 1,146 |

Sources: `runs/final/paced.json` (local-only), [runner](../scripts/paced_stream.py). Timing includes encoder, CTC, AR beam search, forecast reading and added NAR signals, not complete logistic features, UI, audio features or parallel two-pass operation. Maximum 160 ms step was 164.42 ms, so not every step beats chunk duration. Supports feasibility, not a comparative total-compute speedup.

## Additional checks made during cleanup — P1–P4

`runs/final/prescale.json` (local-only) and `runs/final/prescale.md` (local-only) were additionally saved, so I checked them with the code. Learning/overall re-evaluation was not performed again in this arrangement.

1. **Source of gain:** test-clean / 320 Among the 3,420 correct answer/maintenance forecasts selected by the gate in ms, 60.4% was in the voice whose standard CTC sorting time had already arrived. The percentage far ahead of 80 ms was 3.7% among the corresponding success forecasts. CTC alignment is not an independent measure of the actual start and end of pronunciation, so it is not used as a proof of future prediction using raw speech. This situation has a great effect in recovering the existing output standby, and there is no guarantee that the 80 ms gain will be maintained even in the strong model.
2. **Correct label gate:** In the curve of the same mismark indicator 1 tokens/100 tokens, the early display of test-clean / 320 ms was the raw 15.56%, output match label 17.08%, correct/maintained label 17.23%. A separate logistic gate experiment of the correct answer label was performed, but the joint learning results of token CE and confidence BCE were not. Since withdrawn candidates are not re-scored for correct answers, the definition of this mismark indicator is also reported.
3. **Reference value transition:** If the reference value 0.91 selected with withdrawal 0.5 as the target in pooled dev half A was fixedly applied to test, test-clean withdrawal 0.45 / early display 15.69%, test-other withdrawal 0.49 / early display 10.42%. This table is different from the operating points interpolated again in the test. However, the gate weight is learned as dev half A+B, so dev half B is not a completely independent gate verification set. The new 960h protocol separates the three roles on a per-speaker basis.
4. **Relationship with acoustic model quality:** In the section where the ignition WER was low, the standard index for correct answer in the forecast was high. This is an observational correlation and is not causal proof that expanding the acoustic model causes forecast improvement.

Additional-check SHA-256: `fdfb614fda10637fe2ecc129b65c7c121e341ce8e4907473b664f32070240e03` for `runs/final/prescale.json`.

## Distinction between structure and learning style

- The original FGN generates the next state from the two most recent high-dimensional weather states and feeds back and accumulates the predictions. It is different from the parallel 6 token ASR variant.
- Zipformer also uses short token contexts and generates inferences cumulatively. A distinction is made between the answer prefix condition of basic RNN-T learning and rollout learning that feeds back the model creation prefix.
- In the short context model, the **lead condition**, which indicates how many tokens the current prefix is ​​ahead of the speech cutoff, was important. Failure of learning without this condition is not interpreted as a failure of the context length or the FGN as a whole.
- The initial rollout feeding back token argmax is not a continuous full path backpropagation through that selection. The cause of performance degradation cannot be determined based on this difference alone.
- ASR's past activation cache reduces redundant computation of observed speech. This is a different function from predicting future tokens that have not yet been verified.

## Priority for next experiment

1. **We have completed public baseline reproduction and the first strong backbone experiments of our forecaster.** The next priority verifications are iteration of the new predictor seed, delay/mis-display measurements including actual input/screen/resource contention, more limited voice history, and re-validation on the latency-tuned/FastEmit backbone. We do not use the current results as evidence for scratch acoustic learning, commit latency, or total computation savings.
2. **LibriSpeech 960h / approximately 114M cache-aware FastConformer is proposed as a common backbone.** User confirmation is the 960h main benchmark and model expansion direction. The specific 114M, learning rate, batch, and epoch are settings that have not yet been implemented.
3. **Fix the same backbone and compare the overall context AR with the 2 position-limited AR·gate for each layer.** The strict 3 token window is the control group. In the expanded model, the two models are compared with the same random initialization and learning budget, and adding common AR pre-training matches the budgets of the two models. Currently, subsequent seed iterations use existing decoder initial values, so the performance of the scratch setup is not proven. Noise·rollout is set as a separate control, and the additional value of confidence joint learning is confirmed based on the correct answer label gate.
4. **Measure again with the executing system.** Record processing time and memory, including encoder, forecast, gate, verification, and display, actual error exposure time, and confirmed events. The short model 24 byte calculation or small encoder RTF alone does not replace the overall system efficiency.

Model size, learning recipe, comparison conditions, and data division were recorded in [Data/Learning Settings: ](librispeech960_benchmark_plan.md) and [Settings: JSON](../configs/librispeech960_benchmark.json)]. Increasing the rollout as a main focus is not justified by the current results.

## Ready and Incomplete Items

| Item | Status |
|---|---|
| 100h Original sound + dev/test | Ready, checksum, and tally confirmed |
| Paper catalog | Initial 33 edition; Afterwards, additional papers are organized into separate links in related documents |
| 960h Primary Benchmark | User Adoption |
| Additional train-clean-360 / train-other-500 | Official MD5, extraction, audio/text and cross-split duplication checks complete |
| Full training manifest | `train-960.jsonl`: 281,241 utterances / 961.054 hours; excludes dev/test. New tokenizer, features and statistics are not prepared. |
| Data organization | List by dev role according to existing speaker assignment·NeMo format conversion·manifest SHA-256 Save report. NeMo actual loading and learning has not been verified |
| New dev protocol | Speaker-level model-selection / gate-fit / calibration assignments saved; utterance and speaker overlap 0 |
| Model augmentation learning | Not executed; CUDA environment/official recipe integration/memory/throughput measurement required |
| Four follow-ups | Actual text-KV release, synthetic 5/10-minute streams, three predictor seeds and two-pass stabilization completed. Distinct from strong-backbone/FastEmit or scratch validation. |
| Revealed 960h Streaming Zipformer Evaluation | Completed; Full test 5,559 ignition, WER 3.058/7.820% |
| Our full/band2 forecasters + gate | Each trained for 100h × 15 epochs; independent dev calibration and complete test evaluation finished. Display gains appear above; one seed. |
| Correct-and-kept-label logistic gate | P2 additional experiment saved |
| Joint token CE + confidence BCE training | Proposed |
| Incremental cached computation | `runs/final/paced.json`: matching transcripts and measured step times for 100 utterances × 3 conditions. Distinct from live UI/two-pass end-to-end evaluation. |
| Screen-event replay | Performed; separate from device UI and two-pass contention measurements |
| checkpoint selected | lead missing Evaluation issue recorded. Results from the last epoch and rollout initial checkpoint exceptions must be preserved and corrected in re-experiment |
| Storage / publication at this study's completion | Documents organized locally; original runs and models preserved. No Git commit or remote push was performed at that stage. |

## Map of documents and deliverables

| Material | Location |
|---|---|
| Current conclusion | This document |
| Goal and hypotheses | [goal_and_methodology.md](goal_and_methodology.md) |
| Step-by-step method and results | [research_plan_3stage.md](research_plan_3stage.md) |
| Modification/Retraction/Code Review | [results_review_2026-11-11.md](results_review_2026-11-11.md), [claim_audit.md](claim_audit.md) |
| Existing 17M Structure | [architecture.md](architecture.md) |
| 960h Data·Model·Learning Proposal | [librispeech960_benchmark_plan.md](librispeech960_benchmark_plan.md) |
| data readiness | [datasets registry](../data/datasets.json), [full inventory](../data/reports/librispeech_inventory.md), [new dev separate aggregate](../data/reports/librispeech960_protocol.json) |
| Paper/Additional Reading | [literature_map.md](literature_map.md), [33 edition index](../papers/INDEX.md) |
| File map and reproduction | [READ.MD](../READ.MD) |
