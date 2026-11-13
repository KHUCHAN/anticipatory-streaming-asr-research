# Experimental results and implementation review — 2026-11-11

[Current conclusions](conclusion_and_next_steps.md) consolidate findings and priorities. This document preserves numeric corrections, withdrawals, code review and earlier versions.

Review date: 2026-11-11, America/Los_Angeles. Existing documents, saved outputs and evaluation code were compared without retraining or rerunning the full evaluation. Numbers preserve original evidence; they do not establish actual simultaneous operation of the display system.

**Version Range:** Initial review targeted H1–H3, `runs/final/summary.json` and `scripts/analyze_final.py`. Since users have reported completion of calibration, screen event playback, and short context rollout, we have added the current verdict below. The initial result sheet and the then-current “not executed/not verified” statement are preserved as review history, and the current status takes precedence over subsequent calibration results.

**Subsequent benchmark decision:** On the same day, the user adopted LibriSpeech 960h as the primary benchmark. Proposals for data division and model/learning settings were recorded separately in [960h Benchmark Plan](librispeech960_benchmark_plan.md). The existing H1–H3 in this document is preserved as the result of the 100h model.

## Current judgment – ​​reflects subsequent correction results

**Latest collation added:** After the initial short context table in this section, the W1a results learned with the same AR decoder·initial values ​​and the actual cache increment calculation results were saved. W1a maintained approximately 85–96% of the WER improvement compared to B0 obtained by full-context AR. `runs/stage3/window/summary.json` and `runs/final/paced.json` were compared, and a new table and measurement range were added to [Comprehensive Conclusion](conclusion_and_next_steps.md). Do not confuse 16.40% of the slot-type model below with the latest 3 token AR result 14.75% (test-clean 320 ms).

**Additional P1–P4 check:** After the correction below, `runs/final/prescale.{md,json}` was also checked. Based on the CTC sorting time, about 60% among the success predictions was in the voice section that had already arrived, and a separate correct answer/maintenance label logistic gate and a test transition experiment with a fixed threshold value in dev were saved. Therefore, the “correctness head/fixed threshold evaluation not performed” below should be read as the scope of the joint learning head/existing curve evaluation at the time. The new gate and transition results and the remaining independent division/end-to-end measurements were separately recorded in [Additional Check of Comprehensive Conclusion](conclusion_and_next_steps.md).

Compared original `runs/final/summary2.json` (local-only), `runs/final/summary2b.json` (local-only), `runs/stage3/short/summary.json` (local-only) and `runs/rollout_eval_lead2.json` (local-only) with implementation. No retraining/evaluation in this documentation task. Full tables: [calibration and additional verification](research_plan_3stage.md#calibration-and-further-verification).

### Withdrawal, maintenance, scope limitation

- **Withdrawal:** The 2-pass P50 48.2 ms / P95 208.7 ms / edit distance 10.98 of the existing H3 are not used as the final figures to evaluate the actual display merge. Replace with the screen event playback results below. It does not mean that a single 160 ms + P50 of the forecast will be withdrawn until about 50 ms.
- **Maintenance:** A single pass forecast reduces stability display lag in screen playback without changing the final WER. This is distinguished from the time when the forecast is confirmed without modification, and simultaneous execution and rendering delays on actual devices have not yet been measured.
- **Gating Effect Reduction:** Additional useful information remains in the same retraction rate curve, but is not separately proven as a “FGN-specific effect” or a benefit from shared noise. The comparison signal is the candidate probability and consistency of the model without separate noise.
- **Short context:** In this structure, no gain was observed when increasing P=3 to P=32. This is not a result that proves general 3 token sufficiency or statistical equivalence of the two settings.
- **Cumulative prediction:** Short-context models are better than models without priors and worse than full-context ARs. Both paired bootstrap CI directions for the 12 condition were confirmed in the original JSON. No improvement was confirmed in the attempted rollout fine-tune and noise sampling.

### Same retraction rate curve — test-clean / 320 ms

Interpolated operating point at 0.5 retractions per 100 tokens on 500 evaluation utterances; NAR compute time is added to display time.

| Gate | Successful early displays /100 tokens | Mean latency ms | Decoder-agreement precision % | Reference-based measure % |
|---|---:|---:|---:|---:|
| AR probability only | 13.50 | 153.99 | 96.45 | 94.64 |
| Full gate | 15.93 | 144.04 | 96.98 | 95.16 |
| Excluding NAR/FGN signals | 14.47 | 147.98 | 96.69 | 94.74 |

Full-gate early display rises 2.43 percentage points, approximately 18% relative, with 9.95 ms mean latency improvement. This interpolates threshold curves on evaluation data; it does not show a dev-frozen threshold achieving 0.5 test retractions. The 960h protocol freezes thresholds on separate calibration dev before test.

Signal ablation improves mean latency by 6.01 ms over raw even without NAR/FGN; full versus ablated gate differs by 3.94 ms. After correction, test-clean at 320 ms does not support attributing most benefit to FGN. An added independent-model signal contributes within this limited scope.

Reference metric: `prec_ref = reference-correct committed tokens displayed early / (output-agreement successes + retractions)`. It combines correctness and retention; retracted forecasts are not rescored against the reference. Gate training still targets later-decoder agreement. Adding reference evaluation is distinct from training a correctness BCE head.

### Screen event playback — test-clean 300 utterance

| Configuration | Final WER % | Stable P50 ms | Stable P95 ms | Cumulative Rewrite/Final 100Word |
|---|---:|---:|---:|---:|
| single 640 ms | 12.78 | 370.5 | 654.5 | 0 |
| single 320 ms | 14.24 | 208.7 | 368.0 | 0 |
| single 320 ms + forecast | 14.24 | 129.0 | 328.0 | 1.22 |
| 160 → 640 ms | 12.78 | 127.2 | 327.5 | 21.21 |
| 160 + forecast → 640 ms | 12.78 | 86.8 | 296.4 | 34.64 |

`replay()` merges output/forecast events chronologically. Stability begins when the correct final word appears and remains. Evaluation changed from 500 to 300 utterances; WER changes such as 13.09→12.78 do not establish model improvement.

`rewritten_words_per_100` is the cumulative value of the number of suffix words erased or rewritten after the common prefix of adjacent screens, excluding cases where the last word is longer. It may include repeated modifications of the same word, and it is not the proportion of different words that are modified. Therefore, 21.21/100 is not expressed as “exactly the words of 21% were modified once.”

In test-other, 160 + forecast → 640 ms is WER 37.33%, stable P50 126.4 ms / P95 689.0 ms, cumulative rewrite 51.24/100 words. The resource contention and actual device delay of the two passes are still unverified.

### Short context model — test-clean / 320 ms

| Predictor | WER % | Interpretation |
|---|---:|---|
| None | 18.09 | B0 |
| P=3 Determinism | 16.40 | B0 vs. −1.69%p, Full Context AR vs. +1.85%p |
| P=32 Determinism | 16.69 | No context expansion gain observed in this structure |
| P=3 noise model | 16.26 | numerical difference compared to determinism model −0.14%p; Noise gain cannot be determined with this difference alone |
| P=3 Determinism + rollout fine-tune | 17.29 | Worse in attempted microlearning |
| Full-context AR | 14.55 | B1a |

Here P=3 deterministic retains about 48% of B1a's WER improvement over B0. Path hits use a separate protocol: 300 utterances, 8 cutoffs, 640 ms chunks and 16 candidates. Do not mix these with the decoding-WER conditions.

The 3-versus-32-context comparison alone cannot attribute the full-AR gap to architecture or training. Train context-restricted versions of the same AR under matched conditions, predeclare noninferiority and repeat seeds. Current short models use 15 epochs; rollout fine-tuning was evaluated for only 3 epochs.

**Memory interpretation correction:** Since `ShortPrior.bytes_per_node = P * 8`, 24/256 byte is the arithmetic size of P int64 token IDs. Currently, a separate `logp` vector remains in the node, and there are also prefix trie, Python object, shared speech expression, and temporary tensor. It is not a measure of real memory in the same range as AR's K/V+ probability magnitude. Short contexts can eliminate cumulative K/V, but do not claim that the entire system memory or all states per candidate are fixed to 24 bytes.

**Operational Issue Log:** User reported that the best checkpoint was selected incorrectly due to evaluation without lead input during training, and was evaluated again with the last epoch. The deterministic rollout has an exception, starting at epoch 13 best. It also reported that both rollout jobs were aborted using approximately 18GB each, so the batch was fixed and rerun. The learning process has not been redone or audited in this record. In the future, checkpoints will be selected again through modified dev evaluation and compared using the same initial checkpoint and learning amount.

Follow-up original SHA-256:

```text
70d46b4691a4743bceb07d85549a61b0a3e3e09c8c740002f726a21d662d56a4  runs/final/summary2.json
f521d25ac2a20b68863b390711536e25687e2e9e51067284679bcfd6f824a0c7  runs/final/summary2b.json
86f6d9cc857a4c36307c543f867e6dd89ea37da6242094e1d94eb68eab6a9222  runs/rollout_eval_lead2.json
7ee6cfc0e185b79905d8b869d0b9b93aa965441b730cfc69bbadad285e3144f8  runs/stage3/short/summary.json
```

## Initial H1–H3 review history below:

Tables and unexecuted items below are historical snapshots before subsequent corrections. The assessment above takes precedence; two-pass 48 ms remains only as a withdrawn analysis result.

## Where results are stored

| Material | Location | Role |
|---|---|---|
| Stage 1–3, F1–F3, G0·G2·H1–H3 | [research_plan_3stage.md](research_plan_3stage.md) | Overall methodology and results |
| H1–H3 Original summary | `runs/final/summary.json` (local-only), `runs/final/summary.md` (local-only) | 156 saved results and confidence intervals for each setting |
| Forecast/output/time for each boundary | `runs/final/gate_{dev-clean,dev-other}.pt`, `runs/final/eval_{dev-clean,dev-other,test-clean,test-other}.pt` | Event records for gates and evaluation; Local only |
| Event collection | [collect_events.py](../scripts/collect_events.py) | B1a decoding, AR forecast chains and added NAR signals |
| Gate/Post Evaluation | [analyze_final.py](../scripts/analyze_final.py) | Logistic regression gate, reference value selection, H1–H3 Aggregation |

`runs/` output and model weights are local data. The result table in this document records key values ​​so that readers who cannot download the file can check the conditions and figures. Saving a document is a separate task from Git commit/remote push.

## Check assessment scope

- All 4 evaluation files are dev-clean/dev-other/test-clean/test-other, each 500 utterance, seed 11, and each file contains an event of chunk 4/8/16 frame (160/320/640 ms). The 500 utterance of each split was evaluated in three chunks, and a separate 500 utterance was not added to each chunk.
- The files for the gate are dev-clean·dev-other, each 800 utterance, and seed 5. `exclude_seed=11` and `exclude_n=500` are saved.
- As a result of directly comparing the stored utterance IDs, the intersection of gate/evaluation utterances in all chunks of dev-clean and dev-other was **0**. This is a confirmation of utterance ID separation, and does not confirm speaker separation of gate and evaluation.
- Gate fitting and reference selection are performed on the same data for the gate. Afterwards, the standard value is fixed and applied to the separated evaluation data. It is not a structure with separate gate learning/compensation division.
- The existing summary reports 111,203 gate compliance boundaries. The confidence interval for the difference in average delay is the paired bootstrap 1,000 number of utterance units and does not imply reproduction in multiple learning seeds.

## Saved key results

### H1 — learning gate, test-clean / 320 ms

| Gate | Final WER % | Successful Early Mark/100 Token | Average Mark Delay ms | Withdrawal/100 Token | Output Match Precision % | 95% CI of average delay difference, ms compared to raw |
|---|---:|---:|---:|---:|---:|---|
| AR probability only | 14.55 | 13.46 | 154.0 | 0.48 | Based on 96.6 | |
| All signals | 14.55 | 18.07 | 136.0 | 0.75 | 96.0 | −19.5 … −16.5 |
| Excluding FGN/NAR signals | 14.55 | 14.63 | 147.6 | 0.50 | 96.7 | −7.7 … −5.1 |
| Excluding CTC signal | 14.55 | 18.08 | 135.9 | 0.73 | 96.1 | −19.6 … −16.5 |

Thresholds target **95% output-agreement precision** on gate data; actual evaluation precision differs. Raw 96.6% versus full-signal 96.0% accepts different risks and cannot directly be called a 34% improvement at equal precision. Additional curves at matched false-display/retraction rates are needed.

There are circumstances where additional information from the FGN/NAR signal may be useful. However, the model in question is a separate predictor, `noise='none'`, and the signal removal experiment alone does not prove the effect of FGN's unique probability modeling or shared noise. It should be compared with other independent models and reliability signals of the same computational budget.

### H2 — AR forecast chain, test-clean / 320 ms

| Depth/follow-on tokens basis | Successful early mark/100 tokens | Average mark delay ms | P95 ms | Withdrawal/100 tokens | Output matching precision % |
|---|---:|---:|---:|---:|---:|
| 1 Token | 18.07 | 136.0 | 327.4 | 0.75 | 96.0 |
| 3 token / 0.9 | 21.50 | 124.5 | 327.4 | 0.98 | 95.6 |

Mean latency falls another 11.5 ms and precision by 0.4 percentage points. Depth 3 is the number of forecast future tokens, not a 3-token history limit. `collect_events.py` uses `Predictor(d, 0)` and `pred.step()`; H2 is full-history AR recurrent generation.

### H3 — Post-joining of two pass outputs, test-clean

| Display / final pass | Final WER % | Display P50 ms | Mean ms | P95 ms | Final-string edit distance /100 words | Encoder ms / audio second |
|---|---:|---:|---:|---:|---:|---:|
| 640 ms / equal, no forecast | 13.09 | 370.5 | 369.4 | 653.3 | Not applicable | 6.5 |
| 640 ms + forecast / equal | 13.09 | 290.6 | 293.1 | 610.9 | Not applicable | 6.5 |
| 160 ms / 640 ms | 13.09 | 127.1 | 141.8 | 214.4 | 10.98 | 25.0 |
| 160 ms + forecast / 640 ms | 13.09 | 48.2 | 68.4 | 208.7 | 10.98 | 25.0 |

Single-pass forecast retractions have a different unit and calculation from the edit distance above. At 640 ms with forecasts, retractions are 0.36 per 100 tokens.

On test-other, 160 ms + forecast / 640 ms gives final WER 34.83%, P50 86.8 ms, mean 103.8 ms, P95 411.2 ms, final-string edit distance 24.15 per 100 words, and encoder work 25.2 ms per audio second.

Decomposing the test-clean P50 reduction attributes 243.4 ms to the small-chunk display pass and 78.9 ms to forecasting. P95 reductions are 438.9 ms and 5.7 ms. Most of the approximately 68% P95 reduction comes from the two-pass configuration, not FGN/forecasting alone.

## Limitations identified in metrics and implementation

### Distinguish between output match and actual correctness.

`labelled()` of `analyze_final.py` labels whether the forecast token is the same as the token that is first confirmed later. `simulate()`'s `ok`/`rev` are also matches/withdrawals between the forecast and the decoder output. It is not a correctness gate that learns or directly scores whether it matches the correct answer transcription. Even if you keep the same wrong answer from beginning to end, the output match is counted as a success.

- It is `precision = ok / (ok + rev)` and is different from the actual correct answer token precision.
- `early = ok / final number of output tokens`. This is not the entire display ratio, including forecasts that were incorrectly displayed in advance and then withdrawn.
- The final WER is compared to the correct transcript, but the forecast does not change the final transcript, so the structure is maintained. Maintaining WER itself does not guarantee the accuracy of temporary representations.
- Average · P50 · P95 delay is limited to words that exactly match the correct answer. The reference time is the time created from the full-context CTC alignment of the existing acoustic model, and is not an independent actual word end time or screen measurement value.

### H3 did not verify viable mark merge policy

`two_pass()` aligns the final words of the two passes with the correct transcription, and then selects the earlier value among the display times of the two passes for each word that the final pass answered correctly. Sorting of the correct answers can be used for evaluation, but this post-computation alone does not mean that a causal screen merging policy has been implemented and verified. You need to create an actual display stream and a modification event to make sure you get the same metrics.

`changed_words_per_100` is the edit distance between the final strings of two passes divided by the number of final pass words. This is different from the number of all screen modifications, including forecast withdrawals, the percentage of displayed words that were modified, and the time incorrect answers remained on the screen.

### Not all time and costs are included

- The encoder output is pre-computed with a causal mask for each chunk. The time table adds the separately measured `bench_encoder()` median and decoder time. It is not an end-to-end measurement that takes the actual audio and renders it on the screen.
- `collect_events.py` records the AR forecast chain time `tg` and the separate NAR model time `tn`. Only `tg` is added to the forecast time of `simulate()`, and `tn` and gate inference time required for the gate are not added.
- The next processing time of the collector, `free`, does not reflect the forecast chain and NAR calculations. The actual concurrent scheduling and resource contention of the two pass and forecast tasks were not measured.
- 25.0 ms/1 seconds of voice and approximately 3.8 times the **encoder cost estimate**. This does not mean that the entire RTF of the system, including decoder, gate, NAR predictor, and display merge, is 0.025.

## Original FGN, current variants, user suggestions

| Structure | Past conditions | Future creation method | Current confirmation status |
|---|---|---|---|
| Original weather FGN | Two recent high-dimensional weather states | Generate one next state and feed predictions back for AR rollout | Source reviewed; original model not reproduced here |
| Current ASR FGN variant | Last 8 tokens + last 64 encoder frames | Outputs 6 future token positions in parallel in one forward | Evaluated in F1–F3 and as G/H signals |
| Current H2 | Full text history + arrived audio | AR predictor sequentially generates up to 3 tokens | Post-hoc evaluation above completed |
| User suggestion | Recent 3 tokens/states inside and outside + arrived voice | Pushing the fixed context and putting back your own prediction to create cumulative | There is no corresponding learning comparison result in the existing H1–H3. New rollout-related code is subject to separate review |

[Original FGN §2.1·2.4](https://arxiv.org/html/2506.10772v1#S2) specifies the Markov decomposition of the two-point condition and the maximum 8-step rollout microlearning. One forward does not generate the entire time trajectory in parallel. The shared noise in the original paper is applied to the spatial and variable output structure at one point in time, and does not mean that the same noise is fixed at all future points in time.

F1–F3 did not establish added value for the parallel multi-token ASR variant/shared-noise setting. Do not generalize this to all original FGN, fixed-context AR rollout or multi-step prediction.

Earlier `hist=4/8/16` tests restricted attention at inference in a full-context-trained AR. On 500 dev-clean utterances at 320 ms, B3 WER was 14.66% for full context, 15.24% for 4 tokens, 14.98% for 8 and 14.79% for 16. This differs from training with short context. `hist` selects parent nodes but does not guarantee physical removal of old nodes/caches. Recurrent generation is compatible with fixed context; the chain rule does not prescribe context length.

## Connected prior research

| Paper | Relationship to this discussion |
|---|---|
| [Stateful Conformer with Cache-based Inference for Streaming ASR](https://arxiv.org/abs/2312.17279), 2023/2024 | Paper attached by user. Limited speech context, attention·convolution activation cache, and context matching for learning/inference are key. The two losses of CTC+RNNT are different from the two losses of token prediction+correctness discrimination. |
| [Less Is More](https://arxiv.org/abs/2012.06749), 2020 | Merge the search path with the trained RNN-T to use only 4 previous wordpieces. This is a direct comparison of limited context learning. |
| [RNN-Transducer with Stateless Prediction Network](https://research.google/pubs/rnn-transducer-with-stateless-prediction-network/), 2020 | A predictor that uses one token immediately before. We report similar recognition performance under the original wordpiece experimental conditions. |
| [Tied & Reduced RNN-T Decoder](https://arxiv.org/abs/2109.07513), 2021 | Limited token history, weight sharing with simple predictor. Reported decoder 23M→2M and WER retention under conditions including additional EMBR training. |
| [On the Prediction Network Architecture in RNN-T for ASR](https://www.isca-archive.org/interspeech_2022/albesano22_interspeech.html), 2022 | Comparison of predictor structures such as N-Concat. LibriSpeech 100 includes time experiments. |
| [Partial Rewriting for Multi-Stage ASR](https://arxiv.org/abs/2312.09463), 2023 | This is a prior study that merges fast intermediate output and slow high-quality output and reduces flicker. It is a comparison object to judge the novelty of H3. |
| [Cascaded Encoders](https://arxiv.org/abs/2010.14606), 2020 | A structure in which the output of a streaming encoder is reused by a subsequent encoder. This is distinguished from the current method of calculating the two chunk settings separately. |

Our encoder also caches K/V and convolution history. Stateful Conformer state consists of intermediate acoustic representations per layer, not a few words. Caching observed audio avoids duplicate work; predicting unverified future tokens is different. Different chunk contexts can produce different upper-layer activations, so identical weights do not guarantee all caches can be shared across passes.

## Next experiment item — not run/not verified on existing H1–H3

1. Correct/incorrect labels for candidate tokens are created by aligning them with the actual answer transcription, and evaluated separately whether or not the output is maintained. The combination of the existing token CE and the correct answer BCE is a proposal, and there are no implementation or learning results yet. Actual model prefixes and incorrect answer candidates are also included in the training materials.
2. Compare the coverage–latency curves of raw, learning gate, and independent model signals with the actual false display rate or output retraction rate set to the same level.
3. Implement a two-pass merger that does not look at the correct answer, and record mark, withdrawal, and replacement events and incorrect answer exposure time. Both passes and all gate operations are actually executed on the same resource to measure delay and total computation.
4. We compare the cumulative predictor learned from scratch with a fixed context such as the recent 3/4/8/16 token and the full context AR using the same model and data budget. Error accumulation during long rollouts and verification and recovery when actual voice arrives are also measured.
5. The user intent is to expand the model. Approximately 100M + English 1,000 hours is a discussed proposal and no scale/resource contract or learning implementation has been confirmed. We reaffirm the same evaluation in a stronger cache-aware streaming acoustic model.

## Original identifier at time of review

SHA-256 records identify the files at the start of this review; they were not modified during that documentation task. Later changes to results or evaluation code can make current hashes differ.

```text
cd4ff2b0ba3fd02e76a2fb09d3811551527abedf7bb25778b67a4b92904d4d41  runs/final/summary.json
93c6631a2940c28a5b3f1f75f3aba8877034f46c1ed3b7247280be14a203e1c3  runs/final/summary.md
87847e4cd7ba679a94f0089df306613bb570911d5c3ea279685d4b3f94094a74  scripts/analyze_final.py
d0eb8de4f2002576445f26fcbfa8ba4c97a84d3963dc8bf67270d579b587cf79  scripts/collect_events.py
a035c577002a496d342398865bd2a2de86e3a008deae67cc4cac74b8d166d5f8  avs/fgn.py
f573922febdb8cb3cc102fe6b52c9e010701c2e838b441370ac47c3ddaa07f45  avs/stream.py
```

Final review detected a subsequent change to `avs/fgn.py`, then SHA-256 `be63b3b9bf9a2f35448123447cfe34253390f0a0b239fb02114640f01766c9a0`. Existing results and four primary evaluation-script hashes matched the recorded versions. Review later changes with their associated follow-up results.
