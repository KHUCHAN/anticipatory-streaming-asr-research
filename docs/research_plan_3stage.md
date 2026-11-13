# 3Step Research Methodology — Anticipate–Verify Streaming ASR (AVS)

**Current conclusion entry point:** [Conclusion·Representative figures·Additional check·Next experiment](conclusion_and_next_steps.md). This document preserves the step-by-step history. Subsequent corrections take precedence over the initial table, and the existing 2-pass 48 ms have been withdrawn. `prescale`'s correct answer/maintenance label gate and reference value transition results were also connected to the overall conclusion.

Written: 2026-11-10. **Stage 1–3 and F1–F3 completed 2026-11-10; G0/G2/H1–H3 completed 2026-11-11; corrections and short-context recurrent prediction completed 2026-11-11 ([calibration and verification](#calibration-and-further-verification)); prescaling checks completed 2026-11-11 ([checks](#inspection-before-scale-up)). Results appear under each stage, [combined results](#comprehensive-results), [FGN extensions](#fgn-expansion-experiment-f1f3) and [forecast display](#forecast--verification-indicator-g0-g2-h1h3).**

**Added implementation review: 2026-11-11.** Resulting figures were checked against saved output. The delay of H1–H3 is a post-evaluation using event logs and is not an actual measurement of the actual screen or simultaneous execution of two passes. Gate precision is not the actual correctness, but the rate of agreement with the subsequent decoder output. [Results and Implementation Review Record](results_review_2026-11-11.md) records accurate indicators, confirmed data separation, omissions in time calculations, differences between original FGN and current variant, and unimplemented items.
Parent: [Research goal and methodology](goal_and_methodology.md) · Structure: [Architecture](architecture.md)

[The target document ](goal_and_methodology.md) calls for testing three capabilities separately: the ability to recognize speech that has already been received, the ability to predict words that have not yet been heard, and the ability to verify and reuse predictions to reduce commit latencys and computations. The three stages correspond one-to-one to this division. The output of the previous stage is frozen and then passed on to the next stage, so that the effects of one stage are not mixed with the values ​​of other stages.

| Steps | Questions | Learning object | Fixing things | Deliverables |
|---|---|---|---|---|
| Stage 1 Causal acoustic base | Future How accurately is recognition without voice | encoder + CTC + auxiliary decoder | Data·Feature·tokenizer | B0 Model, causality test, sorting |
| Stage 2 Leading predictor | Guess the next token path before the voice comes | text-only / audio-aware predictor | Stage 1 encoder·CTC | B2·B3 Predictor, Hit@K |
| Stage 3 predict–verify decoding | Does verification/reuse reduce latency and work? | None; decoding policy only | All Stage 1–2 weights | B0–B3 comparisons and latency/compute breakdown |

## Stage 0 — Data (complete)

- LibriSpeech train-clean-100 (28,539 utterance, 100.59 h) + dev/test 4 split. MD5 verification, speaker overlap between splits 0. → [inventory](../data/reports/librispeech_inventory.json)
- Features: 80 primary log-mel, 25 ms window / 10 ms hop, no centering. Normalization uses only the fixed average and standard deviation of the entire train (not using utterance unit statistics).
- Tokenizer: SentencePiece unigram 500 (blank, includes sos/eos). **train transcription only** used.
- Reproduction: `scripts/prepare_librispeech.py` → `scripts/prepare_features.py`

## Stage 1 — Causal Acoustic Base (B0)

**Purpose.** Create a streaming acoustic model to be shared by all comparison groups. It is learned and fixed only once so that differences in subsequent steps do not result from differences in the acoustic model.

**Method.** Learn the chunk-causal conformer encoder with CTC + attention decoder combination loss (0.3: 0.7). Randomize the execution mode per batch (dynamic chunk): 40 % is the full context (offline), 60 % is limited to chunks 4/8/16/32 frames (160–1280 ms) and left context 64 frames (2.56 s). Multiple delay conditions are evaluated with one model, and offline mode is used only to generate sort labels.

**Passing criteria (before moving on to the next step).**

1. Causality: Even if you change the mel frame after the chunk boundary, the encoder frame that has already appeared does not change (`scripts/check_causality.py`, re-execute with learned weights).
2. Convergence: dev WER decreases with each epoch and does not stop due to non-finite loss.
3. Records: dev/test WER (CTC greedy) by chunk size, number of parameters, training time.

**Output.** Last epoch average checkpoint, per-chunk WER table, per-mode CTC forced alignment (token release frame).

**Result (2026-11-10, completed).** 40 epoch (36,880 step), 4.9 time in M5 Pro. epoch 30–39 average checkpoint `runs/stage1/avg.pt`, 17.28 M parameters. Passed causality test with learned weights (past output difference 0, incremental streaming and mask forward difference maximum 2e-6, encoder state 1.74 MB). CTC greedy WER(%), without LM:

| split | offline | chunk 32 (1.28 s) | chunk 16 (640 ms) | chunk 8 (320 ms) | chunk 4 (160 ms) |
|---|---|---|---|---|---|
| dev-clean | 15.10 | 16.21 | 17.07 | 18.48 | 20.34 |
| dev-other | 34.91 | 36.77 | 37.93 | 39.61 | 41.67 |
| test-clean | 15.36 | 16.54 | 17.37 | 18.71 | 20.81 |
| test-other | 35.86 | 37.75 | 38.96 | 40.80 | 42.94 |

Absolute WER is worse than public recipes on the same data: small model, no speed perturbation/LM, CTC greedy. Subsequent experiments are relative comparisons with this model frozen.

## Stage 2 — Leading predictor (B2 text-only, B3 audio-aware)

**Purpose.** First measure the hypothesis H3 separately from the decoding policy: Does a predictor that looks at the audio to date guess the next token better than a predictor that only looks at the text?

**Label.** The emission frame for each token is obtained from the CTC forced alignment of the fixed Stage 1 model. The future text is only used as the correct answer label, and the future sound is not included in the input. The cutoff is created within the utterance and does not exceed the split.

**Learning example.** Divide the token string of the utterance into sections of length 1–4. The chunk boundary just before the first token of the section is emitted is set as a cutoff, and all tokens in the section see only the encoder frames up to the cutoff as cross-attention. The hth token in the interval is “predicted h tokens ahead of the speech.” Multiple cutoffs are learned in one forward step with one mask.

- **B2 (text-only):** Same structure decoder without cross-attention. Train Learn only with transcription.
- **B3 (audio-aware):** Initialize with Stage 1 decoder and learn with the above cutoff mask. The encoder·CTC is fixed.

Both predictors share layer count, hidden size and training text. B3 additionally has cross-attention parameters and audio input; report this difference.

**Evaluation (dev).** horizon h = Hit@1 / Hit@5, token NLL by 0–3. Compare B2 and B3 in pairs at the same utterance and same cutoff.

**Failure signal.** If the Hit@K of B3 is not different from B2, reject H3, and record that B3 of Stage 3 is expected to produce the same result as B2.

**Result (2026-11-10, complete).** Both predictors 15 epoch (13,830 step). dev Hit@1 / Hit@5 (%), h = number of tokens ahead of voice:

| split · chunk | h | B2 text-only | B3 audio-aware |
|---|---|---|---|
| dev-clean · 640 ms | 0 | 32.2 / 56.6 | 44.0 / 68.6 |
| | 1 | 33.5 / 56.4 | 36.8 / 59.3 |
| | 2 | 33.6 / 56.4 | 34.1 / 56.7 |
| | 3 | 33.9 / 57.8 | 34.2 / 57.8 |
| dev-clean · 320 ms | 0 | 32.2 / 56.6 | 54.9 / 78.4 |
| | 1 | 33.5 / 56.4 | 39.0 / 61.7 |
| dev-other · 640 ms | 0 | 30.7 / 54.3 | 41.1 / 65.2 |
| | 1 | 31.2 / 55.1 | 34.5 / 57.5 |

H3 is established **only up to one token**. The voice condition raises Hit@1 to 10–23 pt in the very next token (h=0), and 3–6 pt in h=1, but there is no difference from text-only in h≥2. Most of what the current voice tells you is "the token being pronounced now," and no information about the distant future is contained beyond the text context. Total number: `runs/stage2_{text,audio}/log.jsonl`.

## Stage 3 — Prediction-Verification streaming decoding (B0–B3)

**Purpose.** Verify H1 (commitment delay), H2 (computation reuse), H4 (limited state), and H5 (recovery) on the same acoustic model. There is no learning.

**Action.** Whenever a chunk arrives:

1. The encoder processes the new chunk and emits the CTC posterior (acoustic evidence).
2. CTC prefix beam search extends the hypothesis. The score of the new token is a combination of the CTC score and the predictor score, and if the predictor state is **pre-calculated, it is reused**, and if not, it is calculated in place (fallback).
3. Commit the common prefix of all hypotheses remaining in the beam. Confirmation cannot be reversed.
4. While waiting for the next chunk, the top K subsequent tokens for each living hypothesis are pre-deployed to depth H and the predictor state is stored in the trie. If the node number budget is exceeded, the lowest probability is discarded first.

| Comparison group | Predictor | Calculation point | Separating effect |
|---|---|---|---|
| B0 | None (CTC greedy) | — | Acoustic reference |
| B1 | text-only | After chunk arrival | language context binding |
| B1a | audio-aware | | predictor model difference after chunk arrival (control group of B3) |
| B2 | text-only | Waiting pre-calculation + reuse | Pre-calculation/reuse (same output as B1) |
| B3 | audio-aware | Wait-ahead computation + reuse | Additional effect of voice conditional prediction |

Since B1 and B2 have the same score, their WER must be the same. If it is different, it is an implementation error. This pair shows only the purely computational effect of reuse.

**measurement.**

- WER: Definitive transcription standard. We isolate the cost of early confirmation (false commit) compared to the WER of the same decoder that postpones the decision until the end of the utterance.
- Commit latency P50/P95: From token time (offline CTC sorting) to commit event. It is calculated only from words recognized as correct.
- Computational decomposition: encoder, critical-path predictor calls, preceding (pending) predictor calls, of which fraction is actually used (reuse rate) and fraction discarded.
- Status: maximum number of trie nodes and bytes, encoder cache size. Check whether it increases with utterance length (H4).
- Forecast quality: prefix acceptance length, lead time.
- Statistics: utterance unit paired bootstrap confidence interval.

**How ​​to compare.** For each chunk 4/8/16, rotate B0–B3 to place it in the WER-delay plane. “Whether delay has been reduced at the same WER level” is judged by this curve and is not combined into a single figure.

**Limitations to write down in advance**

- CTC-based verification is inherently lightweight. Predictor call savings may only account for a small portion of the overall delay, and most of the delay may be chunk latency. If the results come out that way, report them as is.
- A text-only predictor learned only from 100 time transcriptions (about 99 words) is weak. Not using an external LM corpus is a choice to control the training data.
- Since the reference time comes from the alignment of our own model, there is a bias in the absolute delay value. Differences between comparison groups are valid because they use the same standards.
- The delay is measured on one M5 Pro.

## Comprehensive results

Conditions: 500 random utterances/split, seed 11; beam 4; prior weight 0.5 chosen on 200 dev-clean utterances with a different seed; hold 0 commits the best path after each chunk; speculative K=2, H=2, node budget 32; CPU 1 thread with four splits concurrent. Raw outputs: `runs/stage3/main/*.json`; tables/intervals: `runs/stage3/summary.md`.

**WER % / commit latency P50 ms**

| split | chunk | B0 | B1 = B2 (text) | B1a = B3 (audio) | B1f |
|---|---|---|---|---|---|
| dev-clean | 160 ms | 20.48 / 126 | 17.76 / 127 | 16.85 / 127 | 16.66 |
| | 320 ms | 18.48 / 206 | 15.54 / 208 | 14.66 / 207 | 15.07 |
| | 640 ms | 17.14 / 368 | 14.08 / 372 | 13.67 / 371 | 13.67 |
| test-clean | 160 ms | 20.00 / 126 | 17.22 / 126 | 16.50 / 126 | 16.41 |
| | 320 ms | 18.09 / 207 | 14.88 / 208 | 14.55 / 208 | 14.59 |
| | 640 ms | 16.86 / 368 | 13.44 / 370 | 13.09 / 370 | 13.39 |
| dev-other | 160 ms | 41.54 / 127 | 36.88 / 127 | 37.30 / 127 | 37.16 |
| | 320 ms | 39.29 / 207 | 34.47 / 208 | 34.36 / 208 | 34.65 |
| | 640 ms | 37.60 / 368 | 31.98 / 372 | 32.17 / 372 | 33.14 |
| test-other | 160 ms | 43.89 / 126 | 39.80 / 126 | 39.17 / 126 | 39.68 |
| | 320 ms | 42.04 / 206 | 37.20 / 208 | 36.82 / 208 | 37.03 |
| | 640 ms | 40.05 / 368 | 34.87 / 373 | 34.83 / 374 | 35.03 |

**Operation effect of precomputation (dev-clean)**

| chunk | on-path predictor step/token: B1 → B2 | B1a → B3 | B3 Reuse rate | B3 Percent discarded during precalculation | Full RTF: B1a → B3 |
|---|---|---|---|---|---|
| 160 ms | 1.68 → 0.90 | 1.14 → 0.49 | 78 % | 79 % | 0.036 → 0.054 |
| 320 ms | 1.82 → 1.34 | 1.50 → 0.93 | 56 % | 70 % | 0.021 → 0.029 |
| 640 ms | 2.12 → 1.85 | 1.90 → 1.57 | 29 % | 65 % | 0.017 → 0.021 |

**Judgment by hypothesis**

- **H1 (confirmed delay): Conditionally established, but the cause is not pre-calculation.** The delay difference between comparison groups in the same chunk is within a few ms. Delay is determined by chunk latency. However, there is a shift in the WER-delay plane: B3 moves from chunk 160 ms (WER 16.85 %, P50 127 ms) to chunk 640 of B0. It produces a WER like ms(17.14 %, 368 ms). At the same accuracy, the commit latency was reduced by approximately 240 ms. This gain comes from combining predictor scores, and most of B1 (no precomputation) is also obtained.
- **H2 (computation reuse): Only true on the critical path.** Precomputation reduces predictor calls after chunk arrival by 15–57 % and does not change the output (B1 = B2 matches all conditions). However, the savings are not visible in latency at 0.3–0.6 ms per chunk, and 65–85 % of advance computation is discarded, increasing total operations (RTF) by 20–50 %. Even on the 1 thread CPU, the overall RTF is 0.06 or less, so the predictor is not a bottleneck.
- **H3 (negative conditional prediction): established in clean, indeterminate in other.** WER difference 95 % confidence interval of B1 → B1a is dev-clean [−1.38, −0.45]/[−1.26, −0.52]/[−0.72, −0.11] pt(160/320/640 ms), test-clean 160·640 ms does not include 0. dev-other contains all three chunks, test-other contains 0 from 320·640 ms.
- **H4 (limited state): established.** Encoder state 1.7 MB fixed, predictor trie up to 31 nodes (approximately 0.3 MB). If we limit the predictor token history to 8 dev-clean 320 ms, then B3 WER 14.66 → 14.98 %, if 16 %, then 14.79 %.
- **H5 (Recovery):** In hold 0, the confirmation is not reversed, so the cost of incorrect prediction is included in the WER. The difference between B1f (the predictor state is recalculated to the current voice every time) and B1a includes the confidence interval 0 in most conditions. There is no harm in reusing an old chunk.

**Implementation verification.** B1 = B2 matches all 12 conditions in utterance units. B1a = B3 matches 11 out of 12, and in test-clean 320 ms, 2 utterance is different from 500 utterance (no effect on WER value, cause undetermined).

**Interpretation and next steps.**

1. In this structure, “compute in advance and reuse” does not reduce delay. The verifier (CTC) and predictor are too light compared to the chunk interval. For reuse to make sense, either the predictor must be much heavier (large LM) or the policy must be one that touches the delay itself.
2. What works is to write the predictor as a score. Language/speech conditional prior probabilities make up for the acoustic loss of small chunks.
3. Audio conditioning helped only for h≤1, suggesting limited benefits from multiple paths; no additional benefit was established for this parallel multi-token ASR variant ([FGN extension](#fgn-expansion-experiment-f1f3)). This does not test every fixed-context recurrent AR predictor or the original FGN architecture.
4. The remaining candidates are **prediction-based adaptive chunks** placed out of scope. Since it is chunk waiting that determines the delay, only the policy of disconnecting chunks early to verify when there is a high-confidence prediction directly reduces algorithmic delay.
5. Absolute WER is high (test-clean 13–17 %). The generality of the conclusions needs to be rechecked in stronger acoustic models.

## FGN expansion experiment (F1–F3)

A parallel multi-token ASR variant using some design elements of FGN (Alet et al. 2025, P21) was created and evaluated in three experiments. It does not exactly reproduce the time progression of the original FGN. Codes: `avs/fgn.py`, `scripts/train_fgn.py`, `scripts/eval_fgn.py`, `scripts/run_stage3.py`(B4).

**Application of elements taken from FGN and ASR.** (1) 32-dimensional global noise is transmitted to all conditional layer-norms through a shared noise encoder. (2) In the ASR variant, the future 6 token position is output in parallel in one forward and members are also drawn in parallel. This time axis parallel generation is not the original FGN method. (3) The loss scores only the peripheral distribution for each location (2-member fair CRPS) and does not supervise the combination between locations. (4) The conditions are the immediately preceding 8 token and the preceding 64 encoder frame. (5) Epistemic uncertainty is expressed as an ensemble of independently learned models.

**Difference in time evolution.** The original FGN generates the state at the next point in time using the two most recent high-dimensional weather states, re-enters the forecast, and performs autoregressive rollout. One forward in the original text refers to the creation of the entire weather state at one point in time, and does not mean that the entire future time trajectory is created in parallel. The original paper also performs rollout microlearning. Therefore, the results of the current experiment are not used to judge the performance of the cumulative predictor learned with the recent 3~4 token/state. [Original paper §2.1·2.4](https://arxiv.org/html/2506.10772v1#S2)

**Different from FGN.** The FGN in the original paper is a model that predicts the next state based on the two most recent states and re-inputs the predictions to accumulate (§2.4: single-step learning followed by autoregressive rollout fine-tune of up to 8 steps, noise is newly extracted at each step). Here, “multiple points in space” of FGN were mapped to “positions of 6 future tokens” at once. Since the cumulative structure of the time axis has not been moved, the judgment of F1–F3 is for this parallel transformation. The cumulative structure was tested separately in [Calibration and Additional Verification](#calibration-and-further-verification). Pure CRPS collapses in softmax output (members become random one-hots and hits 0.1 %). So, the log score of the mean distribution 2-member was added as the weight 0.5 (still scoring only the marginal distribution). These weights were not tuned. The FGN main body was trained as float32 (float16 autocast emanates at about step 6,800).

**Model.** 7.91 M parameters, 4 layer, Stage 1 encoder fixed, 15 epoch (same budget as Stage 2 predictor). Four: shared noise seed 1·2, independent noise by location, no noise (cross-entropy). All four runs result in non-finite loss 0. The dev loss went down until the last epoch (no overfitting), and the increase after epoch 8 was small (2 token path +1.1 %p).

### F1 — Does the marginal distribution alone create a bonding structure?

All methods propose 16 paths at identical cutoffs. Report the percentage with at least one path matching the next n native tokens, using 640 ms chunks and 300 utterances × 8 cutoffs per split.

| method | dev-clean n=1 / 2 / 3 / 4 | test-clean n=1 / 2 / 3 / 4 | dev-other n=1 / 2 / 3 / 4 |
|---|---|---|---|
| AR audio-aware (B3 predictor), beam 16 | 67.7 / 37.7 / 21.2 / 12.9 | 70.1 / 37.7 / 22.5 / 14.4 | 61.8 / 35.1 / 21.1 / 13.0 |
| FGN, member 16 | 65.7 / 26.7 / 7.3 / 1.2 | 69.5 / 25.8 / 6.9 / 1.9 | 56.8 / 22.7 / 7.0 / 1.2 |
| FGN, mix members by position | 65.7 / 27.4 / 7.6 / 1.1 | 69.5 / 26.2 / 6.8 / 1.8 | 56.8 / 23.2 / 6.9 / 1.6 |
| AR text-only (B2 predictor), beam 16 | 27.0 / 12.1 / 6.3 / 3.9 | 27.5 / 11.1 / 6.0 / 3.3 | 27.2 / 11.5 / 6.7 / 4.0 |

- **Rejected.** The path hit does not change even if you mix the position of the member and remove the combination. Shared noise did not create consistency between locations.
- 1 token is close to AR by 0.6–5 %p, but it lags behind 2 token by 11–12 %p and 3 token by 14–16 %p. The same is true for chunk 320 ms (`runs/fgn_eval.json`).
- Path 16 creation time (CPU 1 thread): FGN 16 member × 6 token 4.9 ms, AR beam 16 × 4 token 6.9 ms.

### F2 — What each element of FGN gives

Same conditions, hit among 16 candidates (%), 640 ms chunks.

| Method | dev-clean n=1 / 2 / 3 | test-clean n=1 / 2 / 3 | dev-other n=1 / 2 / 3 | Distinct paths /16, dev-clean |
|---|---|---|---|---|
| No noise, top of marginal distribution product 16 | 55.5 / 21.9 / 6.4 | 57.7 / 22.0 / 6.4 | 46.3 / 18.4 / 6.2 | 7.4 |
| Independent noise per position | 66.7 / 26.9 / 7.1 | 69.2 / 27.5 / 7.4 | 57.5 / 23.2 / 7.1 | 15.1 |
| shared noise (FGN) | 65.7 / 26.7 / 7.3 | 69.5 / 25.8 / 6.9 | 56.8 / 22.7 / 7.0 | 14.3 |
| shared noise, 2- model ensemble (8+8) | 67.9 / 29.3 / 7.5 | 71.0 / 28.5 / 7.7 | 60.1 / 24.8 / 7.4 | 14.6 |

- Selecting candidates with noise is effective: 1 token +10–12 %p, 2 token +4–5 %p. However, if you look at just the first member, the model without noise is more accurate (dev-clean 1 token 49.9 vs. 46.5).
- There is no distinction between shared noise and independent noise. It is the same conclusion as F1.
- The model ensemble adds 1–3 %p across all three splits.

### F3 — FGN ensemble with decoding prior probability (B4)

For each B4: chunk boundary, ensemble forward once (member 32) to create a table for the next 6 token, and only search the table in the critical path. The token score is the average probability of members whose path matches the hypothesis. 500 ignition/split, seed 11, beam 4, hold 0, weight 0.1 (select from dev-clean 100 ignition seed 1 to 0.1/0.2/0.3/0.5). The values ​​of B0·B1a are the same ignition as in the main experiment of Stage 3. Confidence interval is utterance unit paired bootstrap 95 %.

| split | chunk | B0 WER | B4 WER | B4−B0 (95 % CI) | B1a WER | B4−B1a (95 % CI) |
|---|---|---|---|---|---|---|
| dev-clean | 160 ms | 20.48 | 20.03 | −0.72 … −0.18 | 16.85 | +2.67 … +3.69 |
| dev-clean | 320 ms | 18.48 | 18.46 | −0.28 … +0.25 | 14.66 | +3.35 … +4.29 |
| dev-clean | 640 ms | 17.14 | 16.95 | −0.43 … +0.06 | 13.67 | +2.83 … +3.77 |
| test-clean | 160 ms | 20.00 | 19.44 | −0.84 … −0.29 | 16.50 | +2.48 … +3.45 |
| test-clean | 320 ms | 18.09 | 17.68 | −0.66 … −0.15 | 14.55 | +2.66 … +3.63 |
| test-clean | 640 ms | 16.86 | 16.64 | −0.44 … +0.02 | 13.09 | +3.03 … +4.09 |
| dev-other | 320 ms | 39.29 | 38.73 | −0.88 … −0.23 | 34.36 | +3.78 … +5.00 |
| test-other | 320 ms | 42.04 | 41.39 | −0.97 … −0.36 | 36.82 | +3.97 … +5.15 |

Cost on dev-clean, 320 ms chunks:

| Method | Critical path prediction ms/chunk | Boundary precomputation ms/chunk | Delay P50 | RTF |
|---|---|---|---|---|
| B0 | 0 | 0 | 206.5 ms | 0.012 |
| B1a | 1.66 | 1.21 | 209.3 ms | 0.021 |
| B3 | 1.08 | 4.31 | 207.4 ms | 0.029 |
| B4 | 0.04 | 8.32 | 214.4 ms | 0.037 |

- B4 improves over B0 by 0.0–0.7 percentage points. Confidence intervals exclude zero in 9 of 12 conditions, but the effect is much smaller than the 3–5 percentage points from the AR prior (B1a).
- The critical path cost is 1/40 of B1a, but the boundary forward is 8.3 ms, so the total operation (RTF) and delay P50 (+8 ms) are worse than B1a.
- The 2- model ensemble did not help decoding (dev-clean 20.03/18.46/16.95 → 19.98/18.49/16.98).
- It is the same even if you use the no noise/independent noise model as prior (dev-clean 320 ms: 18.35, 18.36). No change in WER at member 64·128 opening and 100 ignition tuning.
- The rate of returning to the marginal distribution because the hypothesis does not match any member path is 33/50/72 % in chunk 160/320/640 ms.

### verdict

| Hypothesis | Judgment |
|---|---|
| F1: A consistent multi-token path is created only with shared noise + marginal distribution loss. | rejected (same as mixed control) |
| F2: Noise injection increases candidate diversity and hits | established (1 token +10–12 %p compared to no noise) |
| F2: Shared noise is better than independent noise | Rejected (no difference) |
| F2: Model ensemble increases hit | established (path hit +1–3 %p), no effect on decoding WER |
| F3: FGN prior probability provides the accuracy of AR prior probability at a lower cost | rejected (95 % CI of WER difference compared to B1a is +2.5 across 12 conditions... +5.7 %p less, the total calculation is also larger) |

**Interpretation.** While the weather field is continuous and the shared noise leads to spatially smooth perturbations, the token string is discrete and the distribution of positions n strongly depends on the choice of position n−1. Compared to AR, which puts this dependence directly as a condition, the loss that only scores the marginal distribution has no pressure to learn it. Here, the results of Stage 2 (the information provided by the voice is one token in advance) overlap, so there is no place to use the advantage of the structure of issuing 6 tokens at once.

**Limitations.** The loss is different from the original paper (log score auxiliary term, weight untuning), and CRPS was calculated only for member 2. The data is 100 time, the encoder is fixed, and the learning is 15 epoch. The dismissal of F1 is for this setting.

Outputs: `runs/fgn_{s1,s2,slot,none}/`, `runs/fgn_eval.json`, `runs/stage3/fgn/*.json`, `runs/stage3/fgn/summary.json`.

## Forecast – Verification indicator (G0, G2, H1–H3)

What remained from F1–F3 was “forecast one step ahead” and “concordance of independent models.” This is not written as a prior probability, but at **time of display**. We assume that we forecast the next token at a chunk boundary and, if confident, temporarily mark it before the next decoder verification. If the confirmed output is the same as the forecast, the previous displayed time is acknowledged, and if it is different, it is withdrawn. As a result, it cannot be assumed that the future word was predicted before the actual pronunciation began. Since the final transcription is not changed by the forecast, WER remains the same as the decoder (B1a). Code: `scripts/g0_early_commit.py`, `scripts/run_stage3.py --early`, `scripts/collect_events.py`, `scripts/analyze_final.py`.

**Nature of evaluation.** The table below is a post-mortem analysis of the saved event log. Precision is the rate of agreement with subsequent output, and delay is a model timetable limited to words that match the correct answer. Actual correctness, causal merging of two pass marks, and end-to-end delay including all calculations are subject to separate verification. [Detailed Review](results_review_2026-11-11.md)

Latency uses the Stage 3 definition: word display time minus full-context alignment time. Correct display before word completion can yield near-zero or negative latency.

### G0 — Upper limit of tokens that can be confirmed in advance

Fraction of tokens eligible for early commitment at 95% precision with a gold prefix: 300 dev-clean utterances, threshold chosen on dev-clean.

| chunk | AR audio-aware | FGN single (average probability) | No noise (forward 1 times) | AR text-only |
|---|---|---|---|---|
| 160 ms | 33.3 % | 28.8 % | 27.1 % | 4.9 % |
| 320 ms | 14.9 % | 14.8 % | 13.3 % | 2.5 % |
| 640 ms | 7.6 % | 7.2 % | 7.1 % | 1.2 % |

Average saving per token is 45–55 ms independently of chunk size. Only the next chunk's first token can be anticipated. The signal is acoustic; text-only yields 1–5%. A new G1 forecaster was not trained because predictor differences were small.

### G2 — Forecast in real decoder – verification

B1a reads the next token probability already calculated at the boundary (no additional network calls). Standard 0.85, 500ignition/split.

| split | chunk | WER | P50 Existing → Forecast | Average Existing → Forecast | P95 Existing → Forecast | In advance Indicator | Withdrawal/100 Token |
|---|---|---|---|---|---|---|---|
| dev-clean | 160 ms | 16.85 | 127 → 48 ms | 128 → 62 ms | 207 → 206 ms | 29.7 % | 1.13 |
| dev-clean | 320 ms | 14.66 | 208 → 131 ms | 211 → 143 ms | 367 → 328 ms | 15.5 % | 0.64 |
| dev-clean | 640 ms | 13.67 | 371 → 293 ms | 373 → 304 ms | 653 → 612 ms | 7.9 % | 0.37 |
| test-clean | 320 ms | 14.55 | 208 → 130 ms | 211 → 143 ms | 367 → 328 ms | 16.2 % | 0.73 |
| dev-other | 320 ms | 34.36 | 209 → 168 ms | 216 → 166 ms | 368 → 331 ms | 10.8 % | 1.04 |
| test-other | 320 ms | 36.82 | 210 → 168 ms | 219 → 165 ms | 369 → 332 ms | 11.0 % | 1.21 |

- A 320 ms chunk with forecasting (P50 131 ms, WER 14.66%) gives approximately the same P50 as a 160 ms chunk (P50 127 ms, WER 16.85%), with WER lower by 2.2 percentage points.
- The delay reduction is the same even with B0 + FGN forward 1 times (1.7–1.8 ms per boundary) without AR prior probability (dev-clean 320 ms: P50 206 → 126 ms, indicated in advance) 13.5 %, withdrawn 0.55). However, the WER is similar to that of B0 (18.48 %) and the RTF (0.016) is similar to B1a (0.018), so it is not a better compromise.
- Weakness: P95 hardly decreases. In other, the precision drops to 90–91 %. A withdrawal occurs.

### H1 — Learned confidence gate

Logistic regression combines 11 boundary signals, including AR top-two probabilities/entropy, last-frame CTC blank/token probabilities, deterministic-FGN agreement/probability and subword continuation. Labels measure agreement with later decoder output, not reference correctness. Gate and threshold targeting 95% output-agreement precision were fit on 800 dev-clean and 800 dev-other utterances, 111,203 boundaries, separate from the 500 evaluation utterances per split. Stored-ID checks on 2026-11-11 found zero overlap in each dev split. Related methods: ASR confidence estimation and model agreement.

| split | chunk | Gate | Early display | Mean latency | Retractions /100 | Precision | Mean latency difference versus raw, 95% CI |
|---|---|---|---|---|---|---|---|
| test-clean | 320 ms | AR probability only | 13.5 % | 154.0 ms | 0.48 | 96.6 % | — |
| test-clean | 320 ms | full signal | 18.1 % | 136.0 ms | 0.75 | 96.0 % | −19.5 … −16.5 ms |
| test-clean | 320 ms | Exclude FGN signal | 14.6 % | 147.6 ms | 0.50 | 96.7 % | −7.7 … −5.1 ms |
| test-clean | 320 ms | exclude CTC signal | 18.1 % | 135.9 ms | 0.73 | 96.1 % | −19.6 … −16.5 ms |
| test-clean | 160 ms | AR probability only → full signal | 29.2 → 35.4 % | 64.9 → 52.6 ms | 1.04 → 1.53 | 96.6 → 95.8 % | −13.3 … −11.2 ms |
| test-clean | 640 ms | AR probability only → full signal | 5.7 → 9.2 % | 320.9 → 293.1 ms | 0.13 → 0.36 | 97.8 → 96.3 % | −30.6 … −25.2 ms |
| dev-other | 320 ms | AR probability only → full signal | 8.7 → 12.2 % | 176.2 → 160.5 ms | 0.74 → 0.88 | 92.1 → 93.3 % | −17.8 … −13.8 ms |
| test-other | 320 ms | AR probability only → full signal | 9.0 → 12.3 % | 174.1 → 159.2 ms | 0.82 → 0.84 | 91.6 → 93.6 % | −16.9 … −13.1 ms |

- When choosing a baseline with the same output match precision goal, the number of successful early display tokens increases by 30–60 % and the average delay is 11–31 more ms. The actual evaluation precision is not the same (test-clean 320 ms: 96.6 → 96.0 %). The confidence interval of the posterior average delay difference in all 4 split × 3 chunks excludes 0.
- In the current ablation, if the candidate probability and match status signal of separate NAR/FGN models are subtracted together, the delay gain is reduced to about 1/3, and if the CTC signal is subtracted, the change is small. This is a situation where a separate model signal is useful, and it is not a result of isolating the effects of the token matching signal alone or the unique effects of FGN.
- The precision of other only increases slightly in the range of 92 → 93 %. It falls short of the target 95%.
- Cost: Approximately 2 ms is recorded per additional NAR/FGN forward boundary. This time and gate execution time were not fully reflected in the display timetable, and it was not measured whether it actually ended outside the critical path.

### H2 — forecast chain (2–3 token)

After the first token passes the gate, roll the AR predictor on its own forecast and provisionally display further tokens while probability exceeds 0.9 or 0.7. This uses full-history `Predictor(d, 0)`. Depth 3 is future output length, not three past context tokens. Distinct from a fixed-context-from-scratch FGN rollout study; related to speculative decoding.

| split | chunk | Depth / threshold | Early display | Mean latency | P95 | Retractions /100 | Precision |
|---|---|---|---|---|---|---|---|
| test-clean | 320 ms | 1 | 18.1 % | 136.0 ms | 327.4 ms | 0.75 | 96.0 % |
| test-clean | 320 ms | 3 / 0.9 | 21.5 % | 124.5 ms | 327.4 ms | 0.98 | 95.6 % |
| test-clean | 320 ms | 3 / 0.7 | 24.0 % | 114.7 ms | 327.4 ms | 1.74 | 93.2 % |
| test-clean | 160 ms | 3 / 0.9 | 40.5 % | 41.8 ms | 168.6 ms | 2.10 | 95.1 % |
| test-other | 320 ms | 3 / 0.9 | 14.4 % | 150.4 ms | 330.8 ms | 1.17 | 92.5 % |

At threshold 0.9, mean latency falls another 6–12 ms with precision lower by 0.4–1.1 percentage points. P95 is unchanged. Chains mainly extend subwords within a word.

### H3 — Post-evaluation of 2-pass indication

**Initial analysis values ​​withdrawn:** 2-pass P50 48 ms / P95 209 ms / edit distance 11 in this section are not currently used in results. [Subsequent screen playback ](#screen-play) is replaced by the result of 300 utterance, and this table is preserved as the initial analysis history. [The current review decision ](results_review_2026-11-11.md) takes precedence.

Evaluate the same encoder weights at small and large chunks, then combine output events post hoc; dynamic-chunk Stage 1 needs no extra training. The assumed display uses earlier small-pass words, including forecasts, and the large pass determines the final transcript. Current `two_pass()` aligns both passes with references and selects the earlier correct-word time; it is not a causal screen merger. Related ideas: two-pass/cascaded encoders and U2.

| split | display pass | final pass | final WER | post display P50 | average | P95 | Edit distance between two passes final string /100 word | encoder ms / voice 1 s |
|---|---|---|---|---|---|---|---|---|
| test-clean | 640 ms | equal | 13.09 | 370.5 ms | 369.4 ms | 653.3 ms | — | 6.5 |
| test-clean | 640 ms + forecast | equal | 13.09 | 290.6 ms | 293.1 ms | 610.9 ms | 0.36 (Token) | 6.5 |
| test-clean | 160 ms | 640 ms | 13.09 | 127.1 ms | 141.8 ms | 214.4 ms | 10.98 | 25.0 |
| test-clean | 160 ms + forecast | 640 ms | 13.09 | 48.2 ms | 68.4 ms | 208.7 ms | 10.98 | 25.0 |
| test-clean | 320 ms + forecast | 640 ms | 13.09 | 129.3 ms | 141.4 ms | 328.5 ms | 6.23 | 17.0 |
| test-other | 640 ms | equal | 34.83 | 373.3 ms | 380.5 ms | 656.0 ms | — | 6.5 |
| test-other | 160 ms + forecast | 640 ms | 34.83 | 86.8 ms | 103.8 ms | 411.2 ms | 24.15 | 25.2 |

- In post-calculation, while maintaining the WER of the final pass (13.09 %), the display delay becomes P50 371 → 48 ms, P95 653 → 209 ms. Most of the improvement in P95 comes from the 2-pass configuration, and the further reduction in forecast is 214.4 → 208.7 ms. The delay and screen quality of actual simultaneous execution have not been verified.
- The edit distance between the two final-pass strings is approximately 11 (clean) or 24 (other) per 100 final-pass words. This does not count all actual screen revisions or separately add forecast retractions. Estimated encoder work is 3.8 times baseline, about 25 ms per audio second; this is not a complete-system RTF of 0.025.

### Judgment and interpretation

| Weakness | Experiment | Results |
|---|---|---|
| Latency tied to chunk waiting | G2 predict–verify | Post-hoc display benefit: unchanged WER, P50 −40 … −80 ms, no added neural call |
| Confidence threshold shifts across conditions | H1 gate | Successful early display +30–60% at the same target precision. Actual evaluation precision differs; other remains near 93%. |
| Only the first token advances | H2 chain | Post-hoc mean −6 … −12 ms; P95 largely unchanged. Full-context AR rollout. |
| P95 does not decrease / small chunks are inaccurate | H3 2-pass | final WER maintained in post-evaluation, P95 −68 %. Actual merge/simultaneous execution and modification costs have not been verified |

Current useful signals are next-token forecasting and independently trained-model agreement, not an isolated FGN-specific effect. Parallel 6-token/shared-noise variants show no added benefit. This does not reject original fixed-state AR weather rollout or the user's short-context recurrent idea.

**Limitations.** (1) Display delays and withdrawals are not an actual measure of the user's perceived quality. (2) 2-pass assumed parallel execution and did not measure actual merging and resource contention. (3) The gate learned output matching from LibriSpeech dev. The actual correctness and correction performance in other domains have not been verified. (4) The acoustic model is weak (100 hours, test-clean 13–17 %). The cause of the small effect in other cannot be determined by the quality of the acoustic model, and re-confirmation is necessary in a stronger model. (5) Additional NAR/gate costs are not fully included in the displayed time. [Review History and Next Experiment](results_review_2026-11-11.md)

Outputs: `runs/g0_early_commit.json`, `runs/stage3/g2/*.json`, `runs/final/{gate,eval}_*.pt`, `runs/final/summary.{md,json}`.

## Calibration and further verification

External-review findings were confirmed: (1) H1 matched precision targets, not actual errors; (2) its precision was not reference correctness; (3) FGN time was omitted from display timestamps; (4) H3 did not run actual screen merging; (5) recurrent FGN forecasting had not yet been tested. Code: `scripts/analyze_final2.py`, `scripts/train_fgn.py --lead/--rollout/--split-ctx`, `scripts/eval_rollout.py`, `avs/fgn.py` (`ShortPrior`), `scripts/run_stage3.py` (S1a).

### Gate recomparison (same retraction rate)

Threshold curves were interpolated on the evaluation set at 0.5 retractions per 100 tokens. FGN forward time was added to display timestamps.

| split | chunk | gate | mark ahead | average delay | precision (by decoder) | precision (by correct answer) |
|---|---|---|---|---|---|---|
| test-clean | 160 ms | AR probability only | 20.4 % | 84.3 ms | 97.6 % | 95.3 % |
| test-clean | 160 ms | full signal | 25.1 % | 73.5 ms | 98.1 % | 96.0 % |
| test-clean | 160 ms | Exclude FGN signal | 22.6 % | 76.2 ms | 97.9 % | 95.6 % |
| test-clean | 320 ms | AR probability only | 13.5 % | 154.0 ms | 96.5 % | 94.6 % |
| test-clean | 320 ms | full signal | 15.9 % | 144.0 ms | 97.0 % | 95.2 % |
| test-clean | 320 ms | Exclude FGN signal | 14.5 % | 148.0 ms | 96.7 % | 94.7 % |
| test-other | 160 ms | AR probability only → full signal | 8.4 → 13.9 % | 115.5 → 99.7 ms | 94.5 → 96.7 % | 87.1 → 89.9 % |
| test-other | 320 ms | AR probability only → full signal | 6.8 → 10.3 % | 185.2 → 167.9 ms | 93.3 → 95.4 % | 86.1 → 87.8 % |

- The conclusion holds, but the gain is smaller than the H1 table. test-clean 320 ms, display ahead 13.5 → 15.9 %, average delay −10 ms. The worse the conditions, the greater the gain (+50–65 % displayed in advance in test-other).
- The correct answer standard precision is lower than the decoder standard by 2–3 %p in clean and 7–8 %p in other. Some of the pre-marked tokens are also tokens that the decoder found were incorrect.
- If the FGN forward time is reflected in the display time, the average delay increases by 0.2–0.9 ms.

### screen play

Chronologically merge both passes to replay the screen. Use slow-pass text for the audio region it has decoded, and fast-pass text/forecasts after that. Stability means the final correct word appears and remains; rewriting counts changed/deleted words, excluding extension of the last unfinished word. Evaluation: 300 utterances/split, full-signal gate, threshold 0.87. Single-pass stable P50 matches Stage 3 latency: 127 / 209 / 371 ms.

| split | configuration | final WER | stable P50 | stable average | stable P95 | rewrite /100Word |
|---|---|---|---|---|---|---|
| test-clean | single 640 ms | 12.78 | 370.5 ms | 374.0 ms | 654.5 ms | 0 |
| test-clean | single 640 ms + forecast | 12.78 | 290.5 ms | 298.2 ms | 612.4 ms | 0.45 |
| test-clean | single 320 ms + forecast | 14.24 | 129.0 ms | 140.2 ms | 328.0 ms | 1.22 |
| test-clean | single 160 ms + forecast | 16.53 | 49.6 ms | 60.8 ms | 206.3 ms | 3.78 |
| test-clean | 160 → 640 ms | 12.78 | 127.2 ms | 153.1 ms | 327.5 ms | 21.21 |
| test-clean | 160 + forecast → 640 ms | 12.78 | 86.8 ms | 103.0 ms | 296.4 ms | 34.64 |
| test-clean | 320 + forecast → 640 ms | 12.78 | 169.8 ms | 186.9 ms | 368.0 ms | 20.10 |
| test-other | single 640 ms | 37.33 | 409.7 ms | 430.9 ms | 734.6 ms | 0 |
| test-other | single 320 ms + forecast | 39.30 | 168.1 ms | 207.6 ms | 489.2 ms | 1.38 |
| test-other | 160 → 640 ms | 37.33 | 130.1 ms | 217.5 ms | 689.0 ms | 42.81 |
| test-other | 160 + forecast → 640 ms | 37.33 | 126.4 ms | 179.8 ms | 689.0 ms | 51.24 |

- The single pass + forecast also holds true on the actual screen. Stable P50 −80 ms, rewrite 100 per word 0.5–3.8.
- 2-pass maintains the final WER and reduces stable P50 to 371 → 127 ms, P95 to 655 → 328 ms, but out of the words 5 1 (2 of 5 in other cases) are reused. The post-calculation of H3 (P50 48 ms, P95 209 ms, edit distance 11) is not reproduced in screen playback.
- At 2-pass, the forecast reduces P50 to 40 ms further, but increases the rewrite from 21 → 35. This may have something to do with the simplicity of the seam rules, but no attempt has been made to merge better.
- The combination of fast display and slow correction itself has prior research. What this project can claim is **how much the forecast gate advances the mark at the same pullback level**.

### Short context cumulative predictions

The time axis structure of FGN has been moved: predict the next token using the finite recent state (recent P token + recent 64 encoder frame), and repeat by putting the prediction in context. Since the state of the hypothesis is only the most recent P token, it is constant regardless of the length of the utterance. Audio is encoded only once per boundary (`--split-ctx`). The model size and learning budget are the same as in the F experiment (7.91 M, 15 epoch).

**The learning method was decisive.** If we only train with examples where the last token of the context is always "where the speech ends", one step is equivalent to full-context AR (test-clean 1 tokens 69.3 % vs. 70.1 %), but breaks down in the second step (2 tokens 11.5 % vs. 37.7 %). The context 32 token model also breaks down the same way (7.3 %), so it is not a matter of context length. When we input as input how many of the context tokens are ahead of the voice (0–3) and learn that position as the correct answer (`--lead 4`), it was recovered. Below are the models.

Percentage of 16 paths matching the next n native tokens, 640 ms chunks, 300 utterances × 8 cutoffs, same sample as F1.

| method | dev-clean n=1 / 2 / 3 / 4 | test-clean n=1 / 2 / 3 / 4 | dev-other n=1 / 2 / 3 / 4 |
|---|---|---|---|
| Full-context AR, beam 16 | 67.7 / 37.7 / 21.2 / 12.9 | 70.1 / 37.7 / 22.5 / 14.4 | 61.8 / 35.1 / 21.1 / 13.0 |
| context 3 token deterministic, beam 16 | 62.0 / 29.7 / 14.8 / 7.4 | 65.6 / 30.4 / 14.7 / 8.1 | 54.4 / 26.6 / 14.0 / 7.7 |
| context 32 token deterministic, beam 16 | 60.5 / 28.7 / 14.9 / 7.4 | 66.0 / 30.2 / 14.6 / 8.4 | 53.9 / 26.5 / 14.0 / 7.3 |
| context 3 token FGN, cumulative sampling 16 times | 61.7 / 24.4 / 8.3 / 2.9 | 63.9 / 23.5 / 7.8 / 3.0 | 54.3 / 22.4 / 7.7 / 2.7 |
| context 3 token deterministic + rollout fine-tune | 53.9 / 24.1 / 10.5 / 3.8 | 54.9 / 24.2 / 10.2 / 4.3 | 46.2 / 22.0 / 9.7 / 3.9 |
| context 3 token FGN + rollout fine-tune | 61.2 / 24.0 / 7.8 / 2.2 | 62.2 / 23.4 / 7.9 / 2.6 | 53.5 / 22.0 / 7.1 / 2.6 |
| (F1) 6 Token Parallel FGN | 65.7 / 26.7 / 7.3 / 1.2 | 69.5 / 25.8 / 6.9 / 1.9 | 56.8 / 22.7 / 7.0 / 1.2 |

Used as a decoding prior: S1a, weight 0.3, 500 utterances/split, identical utterances to B0/B1a, paired-bootstrap 95% intervals.

| split | chunk | B0 | Context 3 token (S1a) | S1a−B0 (95 % CI) | B1a (Full Context AR) | S1a−B1a (95 % CI) |
|---|---|---|---|---|---|---|
| dev-clean | 160 ms | 20.48 | 19.16 | −1.69 … −0.92 | 16.85 | +1.89 … +2.73 |
| dev-clean | 320 ms | 18.48 | 17.08 | −1.77 … −1.03 | 14.66 | +2.06 … +2.80 |
| dev-clean | 640 ms | 17.14 | 15.62 | −1.89 … −1.16 | 13.67 | +1.55 … +2.38 |
| test-clean | 320 ms | 18.09 | 16.40 | −2.13 … −1.29 | 14.55 | +1.49 … +2.23 |
| dev-other | 320 ms | 39.29 | 37.07 | −2.72 … −1.74 | 34.36 | +2.16 … +3.28 |
| test-other | 320 ms | 42.04 | 39.25 | −3.29 … −2.35 | 36.82 | +1.95 … +2.89 |

| Variant (test-clean, 320 ms) | WER | Status indications for implementation (comparison ranges vary) |
|---|---|---|
| Context 3 Token Deterministic | 16.40 | 24 byte (constant) |
| Context 32 Token Deterministic | 16.69 | 256 byte (constant) |
| Context 3 token FGN, member 4 average | 16.26 | 24 byte (constant) |
| Context 3 token deterministic + rollout fine-tune | 17.29 | 24 byte (constant) |
| Full-context AR (B1a) | 14.55 | Accumulates 10.2 kB per token, proportional to utterance length |

- **In this structure, no benefit was observed when increasing the context from 3 to 32 token.** Path hit and WER values ​​are similar. This is not a result that proves statistical equivalence or general 3 token sufficiency.
- **Less than full-context AR.** Path hits are 4–9 % points lower per step, and in prior probability, you get about 35–50 % of the WER improvement that AR gives (significantly lower than B0 and significantly higher than B1a in all 12 conditions). Although structure and learning method are candidate causes of the gap, the influence of context length cannot be ruled out. Contrasts learned with short contexts in the same AR structure are needed.
- **Rollout fine-tune, which feeds back its own predictions, did not help.** Both path hits and WER worsen for the deterministic model, and there is no change for the FGN model. Tokens are discrete, so the gradient does not flow through the feedback path (the original paper is continuous, so it backpropagates throughout the rollout). Only the 3 epoch was attempted.
- **FGN-style cumulative sampling is worse than beam search** (2 token 23–24 % vs. 30 %). There is no difference from the deterministic model in terms of prior probability.
- **Computation did not decrease.** One step for 4 hypotheses takes 1.3 ms, similar to AR's 1.05 ms (CPU, 1 thread, isolated measurement). The short-context model does not accumulate K/V. However, 24 bytes, calculated as `P * 8`, counts only 3 int64 token IDs; it excludes logp, the prefix trie, shared audio state and temporary tensors. It is not a measurement of complete per-candidate state or system memory.
- As an early indicator signal, it is weaker than AR. test-clean 320 Withdraw from ms About 1 tokens/100 Tokens in advance Display 14.3 % (AR About 17.6 %, interpolated).

### updated judgment

| Claim | Status |
|---|---|
| Single pass forecast–verification: WER invariant, stable P50 −40 … −80 ms, rewrite 100 per word 0.5–3.8 | maintained in decoder events and screen playback. If you only read the existing AR score, there is no additional network call, but NAR/FGN calculation of the learning gate costs extra |
| Learned gate advances more at equal retraction rate | Retained with smaller effect: test-clean +10–23%, test-other +50–65% |
| 2-pass P50 48 ms / P95 209 ms | Not reproduced in screen playback. Actual: P50 87–127 ms, P95 296–328 ms, rewrite 21–35/100 words |
| Three context tokens suffice | No observed benefit from 32 tokens in this architecture; equivalence or general sufficiency not established |
| Short-context recurrent prediction replaces full-context AR | Retains about 35–50% of AR WER benefit with similar computation. Reduces cumulative K/V but does not establish complete state of 24 bytes. |
| FGN-style rollout learning and cumulative sampling benefit from discrete tokens | Rejected (in this setting) |

**Remaining limitations.** The seam rule for screen playback is the simplest. The cost of parallel execution of the two passes was not measured. The short context model only attempted one structure, 15 epoch, and did not distinguish whether the gap with AR was due to the structure or the amount of learning. The checkpoint selection was misaligned because the evaluation during learning was calculated without lead input (the best of the FGN model was epoch 6), and the above figure was calculated as the last epoch checkpoint (only the critical model during rollout fine-tune starts from epoch 13 best).

Outputs: `runs/final/summary2.{md,json}`, `runs/final/summary2b.{md,json}`, `runs/sl_*` with lead conditioning, `runs/sc_*` without it, `runs/rollout_eval*.json`, `runs/stage3/short/*.json`.

## Inspection before scale-up

Before spending money on Plan 960h ([Benchmark Plan](librispeech960_benchmark_plan.md)), I checked the risks I could see in my current setup. Codes: `scripts/train_stage2.py --window`, `avs/stream.py`(`WindowPredictor`), `scripts/run_stage3.py`(W1a), `scripts/prescale_checks.py`, `scripts/paced_stream.py`.

### Context 3 token AR predictor (same structure, same initial value)

The previous short-versus-full comparison was confounded: B1a used a decoder initialized by 40 Stage 1 epochs and teacher forcing at all positions, whereas short models differed structurally and started from scratch. Here use the same B1a decoder, initialization and 15 epochs, with each position restricted to its preceding 3 input tokens. Recompute windows per position so context cannot expand through layers. Evaluate the same 500 utterances/split and prior weight 0.5.

| split | chunk | B0 | Context 3 Token AR (W1a) | Full Context AR (B1a) | W1a−B1a (95 %CI) | B1a Hold during gain |
|---|---|---|---|---|---|---|
| dev-clean | 160 ms | 20.48 | 17.16 | 16.85 | 0.00 … +0.63 | 91 % |
| dev-clean | 320 ms | 18.48 | 15.18 | 14.66 | +0.27 … +0.79 | 86 % |
| dev-clean | 640 ms | 17.14 | 14.15 | 13.67 | +0.22 … +0.74 | 86 % |
| test-clean | 160 ms | 20.00 | 16.73 | 16.50 | −0.06 … +0.54 | 93 % |
| test-clean | 320 ms | 18.09 | 14.75 | 14.55 | −0.07 … +0.47 | 94 % |
| test-clean | 640 ms | 16.86 | 13.65 | 13.09 | +0.30 … +0.82 | 85 % |
| dev-other | 320 ms | 39.29 | 34.99 | 34.36 | +0.21 … +1.05 | 87 % |
| test-other | 320 ms | 42.04 | 37.21 | 36.82 | +0.03 … +0.73 | 93 % |

| (test-clean, 320 ms) | W1a | B1a |
|---|---|---|
| Predictor state per hypothesis | Token id 3, constant | 10.2 kB per token, proportional to utterance length |
| critical path prediction ms/chunk | 2.07 | 1.60 |
| RTF | 0.021 | 0.023 |
| Early display, threshold 0.85: displayed / retracted per 100 | 13.5% / 0.55 | 16.2% / 0.73 |

- **Context 3 tokens maintain 85–96 % of the WER gain provided by the full context AR.** The difference is 0.2–0.6 %p, and in 3 of the 12 conditions, the confidence interval includes 0.
- The gap seen in the previous section (maintaining only 35–50 % of the gain) came from the structure, initial values, and learning method, not the context length. It is correct to create the short context predictor of 960h (B) in the form of a window over the AR decoder rather than an FGN-type slot structure.
- Computation is not reduced (the window is recalculated every step). What is diminishing is the state.

### source of benefit

Where successfully forecast tokens fall relative to the chunk boundary under full-context alignment: test-clean, gate 0.87, one frame = 40 ms.

| chunk | frame before boundary 2 frame before | frame before boundary | next chunk first frame | +1 frame | +2 frame | +3 frame or higher |
|---|---|---|---|---|---|---|
| 160 ms | 1.4 % | 20.7 % | 39.3 % | 24.1 % | 10.5 % | 4.1 % |
| 320 ms | 1.3 % | 20.1 % | 39.0 % | 25.3 % | 10.6 % | 3.7 % |
| 640 ms | 1.1 % | 20.1 % | 39.2 % | 25.7 % | 10.2 % | 3.7 % |

- 96 % of tokens that the forecast hits are within ±2 frame (80 ms) of the boundary. Approximately 22 % are tokens that have already passed in the overall context alignment (emission delay of streaming CTC), and approximately 39 % are tokens that span boundaries. 120 The number of correct guesses by more than ms is 4 %. test-other has the same distribution.
- **Scale-up risk.** The backbone of the plan is a transducer that uses FastEmit and the frame is 80 ms. Backbones that have learned to accelerate release already capture some of this benefit. Immediately after learning the backbone and before learning the predictor, this table must be remeasured to check the remaining space.

### Gate label and reference value correction

Fraction displayed early at the same false-display rate: retractions plus committed incorrect tokens = 1.0 per 100 tokens.

| split | chunk | AR probability only | label = decoder match | label = correct answer |
|---|---|---|---|---|
| test-clean | 160 ms | 20.6 % | 24.4 % | 25.5 % |
| test-clean | 320 ms | 15.6 % | 17.1 % | 17.2 % |
| test-other | 160 ms | 7.3 % | 11.0 % | 11.6 % |
| test-other | 320 ms | 6.6 % | 8.9 % | 9.1 % |

Reference-based labels improve all 6 conditions, 2 splits × 3 chunks, by 0.1–1.2 percentage points. Use reference-based labels as planned.

Actual retraction rate (chunk 320 ms) when applying a reference value picked as "withdrawal 0.5/100 tokens" in the dev half.

| Calibration data | dev-clean (other half) | dev-other (other half) | test-clean | test-other |
|---|---|---|---|---|
| dev-clean only | 0.56 | 0.60 | 0.59 | 0.68 |
| dev-other only | 0.30 | 0.32 | 0.32 | 0.39 |
| Combined | 0.44 | 0.44 | 0.45 | 0.49 |

If corrected with clean alone, it exceeds the target by 12–36%, and if corrected combined, all four sets are within the target. The dev division of the plan (correction set using clean·other speakers together) is used as is.

### Forecast effectiveness depending on recognition quality

| Utterance WER | Number of utterances | Display in advance | Withdraw /100 | Correct answer standard precision |
|---|---|---|---|---|
| 0–5 % | 144 | 20.6 % | 0.33 | 98.3 % |
| 5–15 % | 204 | 18.8 % | 0.74 | 94.4 % |
| 15–30 % | 278 | 15.4 % | 0.88 | 90.6 % |
| 30–60 % | 269 | 12.0 % | 0.92 | 83.2 % |
| At least 60% | 105 | 7.7% | 0.83 | 64.3% |

Lower-WER utterances have more and more accurate early displays: combined test-clean/test-other, 320 ms chunks, threshold 0.87. This observation alone does not suggest the effect disappears with stronger acoustics; changed emission timing is a separate risk. It is not causal scaling evidence.

### Real Incremental Streaming

A loop that advances the encoder chunk by chunk with the cache (test-clean 100 utterance, CPU 1 thread, no other work). One step = encoder + CTC + AR prior probability beam search + forecast reading + FGN second opinion.

| chunk | transcription match (compared to masked forward) | encoder output maximum difference | step P50 | step P95 | step maximum | P95 / chunk length | RTF |
|---|---|---|---|---|---|---|---|
| 160 ms | 100 / 100 | 9.2e-5 | 6.3 ms | 8.6 ms | 164 ms | 5.4 % | 0.041 |
| 320 ms | 100 / 100 | 3.3e-5 | 7.5 ms | 10.8 ms | 37 ms | 3.4 % | 0.025 |
| 640 ms | 100 / 100 | 4.7e-5 | 10.3 ms | 15.6 ms | 37 ms | 2.4 % | 0.017 |

- The masked full-utterance forward and actual incremental streaming transcriptions used in all experiments so far are all 300 identical.
- One step is 2–5 % of the chunk length. In 160 ms, 1 chunks of 4,440 arrived before the previous step ended (maximum 164 ms, temporary delay).
- Simultaneous execution and screen merging of 2-pass are not included here.

### Predictor variant (based on AR decoder)

Six more variants were learned based on “AR decoder + short context”. All the same decoder (4.47 M)·15 epoch·same data as B1a, decoding weight 0.5, 500 utterance/split, confidence interval is paired bootstrap 95 % compared to B1a.

- **Strict window (W1a, window 1/2/3 token).** Each position is recalculated using only the N previous input tokens. The state of the hypothesis is N token ids.
- **Limited by layer (H1a, 1/2 tokens).** The self-attention of each layer only sees the current token and the N before it. The status is a layered key/value cache of the most recent N+1 tokens. Through the layers, the effective context expands to the 4N+1 token (4 layer).
- **No initial value.** Stage 1 Learn from the beginning without initializing the decoder.

WER (chunk 160 / 320 / 640 ms) and the percentage of the gain maintained by B1a over B0 (range of three chunks).

| Predictor | dev-clean | test-clean | test-other | Benefit retained |
|---|---|---|---|---|
| No prior (B0) | 20.48 / 18.48 / 17.14 | 20.00 / 18.09 / 16.86 | 43.89 / 42.04 / 40.05 | 0 % |
| Full-context AR (B1a) | 16.85 / 14.66 / 13.67 | 16.50 / 14.55 / 13.09 | 39.17 / 36.82 / 34.83 | 100% |
| Per-layer 2-token history | 16.91 / 14.74 / 13.71 | 16.47 / 14.43 / 13.32 | 39.26 / 36.90 / 34.92 | 94–103% |
| Per-layer 1-token history | 17.07 / 14.96 / 13.92 | 16.49 / 14.58 / 13.47 | 39.30 / 36.85 / 34.99 | 90–100% |
| Strict 3-token window | 17.16 / 15.18 / 14.15 | 16.73 / 14.75 / 13.65 | 39.70 / 37.21 / 35.33 | 85–94% |
| Strict 2-token window | 17.82 / 15.93 / 15.00 | 17.26 / 15.31 / 14.34 | 39.93 / 37.70 / 35.71 | 62–84% |
| strict window 1 token | 19.16 / 17.28 / 16.74 | 18.59 / 16.72 / 16.10 | 41.62 / 39.40 / 38.04 | 12–51 % |
| Full context, no initialization | 17.48 / 15.19 / 14.14 | 17.08 / 14.85 / 13.40 | 39.90 / 37.44 / 34.87 | 83–99% |
| strict window 3 token, no initial value | 17.87 / 16.16 / 14.98 | 17.75 / 15.62 / 14.25 | 40.17 / 37.75 / 35.88 | 61–82 % |

| (chunk 320 ms) | WER difference compared to B1a 95 % CI: dev-clean / test-clean / test-other | RTF (test-clean) | Early display criteria 0.85: pre-mark/withdraw/100 (test-clean) |
|---|---|---|---|
| Full-context AR (B1a) | — | 0.023 | 16.2% / 0.73 |
| Per-layer 2-token history | −0.12 … +0.29 / −0.34 … +0.10 / −0.20 … +0.35 | 0.023 | 15.6% / 0.75 |
| Per-layer 1-token history | +0.09 … +0.53 / −0.21 … +0.28 / −0.28 … +0.33 | 0.022 | 15.2% / 0.65 |
| Strict 3-token window | +0.27 … +0.79 / −0.07 … +0.47 / +0.03 … +0.73 | 0.021 | 13.5% / 0.55 |
| strict window 2 token | +0.95 … +1.58 / +0.43 … +1.10 / +0.47 … +1.28 | 0.029 | 12.3 % / 0.52 |
| Full context, no initialization | +0.26 … +0.82 / 0.00 … +0.60 / +0.25 … +0.98 | 0.025 | 11.6% / 0.54 |

- **The 2 token limit for each layer was close to the full context in most conditions of a single seed.** The confidence interval of the difference in 8 among the 9 conditions includes 0, but equivalence is not proven. test-clean 640ms was disadvantageous to +0.23%p, CI +0.03…+0.43%p. The early display rate at the same threshold as RTF is also a similar observation and is not an equivalence test at the same risk level. Per hypothesis, the text cache budget for the most recent location 3 is approximately 31kB. The actual release/backing-storage measurements follow [Four subsequent experiments](followup_experiments_2026-11-11.md).
- The 1 token limit for each floor is also generally 0.0–0.4%p different. The strict window showed a difference between 3 tokens, +0.2–0.6%p,, 2 tokens, and +0.8–1.3%p differences. The past 1/2 location limit for each layer of the 4 layer distinguishes between the number of cache locations and the raw context length because the self-attention text storage capacity is the maximum 5/9 token. The strict window location scoring budget is also different, so it is not possible to conclude that “5 tokens are generally sufficient.”
- The strict window recalculated the window every step, and the RTF of the window 2 token was 0.029. The short evaluation RTF of layer-by-layer constraints was similar to the full context and did not demonstrate speed equivalence or full computational constantization.
- Without an initial value, the full context becomes worse by +0.0–0.6 %p, and the strict window 3 token becomes worse by +0.5–1.0 %p. The early display signal also becomes weaker (pre-indication 16.2 → 11.6 %). Strict window learning only scores 16 positions per utterance (memory constraints), so there are fewer learning signals and thus seems to rely more on initial values.

**Correction.** In the preceding tables, the predictor status of B1a is written as "up to about 70 kB", which only counts the hypothetical nodes that are alive. The implementation continues to refer to the cache of ancestor tokens as the parent pointer, so the actual requirement is proportional to the length of the utterance. The current implementation of the per-layer limit model does not release old caches, so the above 31 kB is not a measurement but a calculation of the minimum amount needed.

### Will be reflected in scale-up

1. The short context predictor uses the past 2 location restriction for each layer in the AR decoder (H1a) as the main candidate. It is based on WER proximity and the possibility of actual text cache limitations and does not claim that the entire system memory is fixed. The strict window 3 token (W1a) is a separate control. FGN type slot structure·rollout fine-tune·noise is set to ablation. Since the 960h backbone does not have the same Transformer decoder initial value, matching the random initialization and learning budget of the two predictors, and adding common AR pre-training provides the same budget to both models.
2. Immediately after learning the backbone, we first measure the “Sources of Gain” table. The success or failure of forecast-verification is determined by whether there is room for boundary ±80 ms tokens in the FastEmit backbone.
3. The gate label is based on the correct answer, and the reference value correction is a set that combines clean and other.
4. Delays are reported based on screen playback (settlement time, rewrite).

**Limitations of this transformation comparison.** All are results from the 100 time CTC model, and the original transformation table in this section is seed one. Additional seed iterations and actual cache release of the entire AR/layer 2 model are recorded separately in [the next four experiments](followup_experiments_2026-11-11.md). Window 4·8 tokens and past 3 locations by floor were not measured.

Outputs: `runs/stage2_audio_{w1,w2,w3,w3_scratch,b1,b2,scratch}/`, `runs/stage3/window/*.json`, `runs/stage3/variants/*.json`, `runs/final/prescale.{md,json}`, `runs/final/paced.json`.

## What is not included in this scope

- **GNN.** Candidate trie is a graph structure, but message passing is not used. (FGN-style multipath generation was performed in the range: [FGN expansion experiment](#fgn-expansion-experiment-f1f3).)
- **Prediction-based adaptive chunk.** If there is a candidate with high confidence, a policy of partial verification and early confirmation before the chunk is full. Instead, the forecast was displayed in advance and verified with the next chunk, followed by 2-pass display ([Forecast – Verification display ](#forecast--verification-indicator-g0-g2-h1h3)).
- The above 100h experiment did not cover the transducer series, external pretrained comparison model, or other corpora. Subsequent public Zipformer evaluation reproductions are recorded in a separate follow-up section below.

## Follow-up four experiments — real cache, long stream, seed, and display stabilization

Completed actual KV release, 5/10-minute waveform streams, full-versus-band2 at new seeds 3/4, and two-pass display stabilization. Four new predictors trained for 15 epochs each; each decoded 3 splits × 500 utterances × 3 chunks. [Follow-up report](followup_experiments_2026-11-11.md) includes original seed 2 and raw evidence. A first run stopped at 14 epochs was fully rerun and excluded from aggregates.

The earlier fixed-state-at-31kB statement was a text-cache calculation. Actual release tests measure backing storage separately from audio K/V, encoder history and token metadata. They do not establish bounded complete system state. Follow-up documentation supersedes earlier single-seed and unreleased-cache limitations where explicitly tested.

### Summary of follow-up completion results

- Actual text KV/logp release: Transcription of utterance 200 × 3chunk and confirmed node score matched. The post-chunk text storage of the entire beam was approximately 90–110KiB and does not include model weights, audio KV, temporary workspace, or output metadata.
- Even in the four conditions of the synthetic 5/10 minute stream, the transcription of the 2 model for each unreleased/released layer was consistent. In 10 minute clean, the entire AR was reset WER 16.717% → continuous 20.365%, and 2 for each floor was 17.021% → 16.900%. The audio KV increased to about 120MiB, and the overall system stationary state was not achieved.
- The differences in WER among the three learning seeds are as follows. This is descriptive statistics for the seed average and does not prove equivalence.

| test split / chunk | Total AR average WER | Average WER by floor 2 | Average difference %p |
|---|---:|---:|---:|
| test-clean / 160ms | 16.472 | 16.485 | +0.013 |
| test-clean / 320ms | 14.534 | 14.537 | +0.003 |
| test-clean / 640ms | 13.134 | 13.262 | +0.128 |
| test-other / 160ms | 39.279 | 39.264 | -0.015 |
| test-other / 320ms | 36.839 | 36.916 | +0.077 |
| test-other / 640ms | 34.863 | 34.852 | -0.011 |

- If 2-pass is checked twice, the rewrite of test-clean is reduced to 21.21 → 15.66 pieces/100 words, but stable P50 is increased to 127.2 → 287.0ms. The overlap alignment chosen by dev reduced clean P95, but increased rewriting of other. Currently, there is insufficient evidence to promote 2-pass to the flagship.

## Public 960h Streaming Zipformer baseline — first external experiment

At the user's request, run the first external baseline from the paper/official implementation: public 66.11M FP32 Streaming Zipformer, BPE 500, 320ms chunk-wise greedy, no external LM or endpoint reset, complete original test sets.

| split | Utterance | Error/Correct Word | Formula WER % | Measurement WER % |
|---|---:|---:|---:|---:|
| test-clean | 2,620 | 1,608 / 52,576 | 3.06 | 3.058 |
| test-other | 2,939 | 4,093 / 52,343 | 7.79 | 7.820 |

Both splits meet the predeclared absolute 0.10-percentage-point tolerance. The +0.030-point other difference was not explained. Approximately 409 seconds for 10.745 hours uses four CPU workers: throughput, not single-stream RTF or display latency. Checks passed for tokens/timestamps across 20ms/320ms/full-input packetization on 10 dev utterances, unchanged earlier output under future-audio perturbation on 3 utterances, and 500 WER distances.

This evaluates a public checkpoint, not local 960h scratch training, FastEmit or our forecaster's added value. It records native outputs and causal display policies; native timestamps are not independent phonetic references or equivalent to prior CTC alignments. Existing forecasters were not directly attached because tokenizers and acoustic features differ. The next comparison trains matched full/bounded-context forecasters for this frozen baseline's BPE 500 and 512-dimensional output.

Conditions, sources, display policies, limitations and reproduction commands are in the [baseline report](baseline_zipformer_2026-11-11.md). Raw outputs: `runs/baseline_zipformer/`; weights: `checkpoints/baselines/`. Executed `configs/zipformer_baseline.json` differs from the scale-up proposal in `librispeech960_benchmark.json`.

## Our Forecast – Verification × Fixed Zipformer — 2026-11-12 Completed

After baseline reproduction, our modules were trained and evaluated. Freeze official 960h 66.11M Zipformer; train 4.60M full/band2 forecasters with BPE 500 and 512-dimensional audio adapters on all 28,539 train-clean-100 utterances for 15 epochs each. Initialization, order and budget match; older AVS initialization is not used. Fit gates on 1,297 separate dev-fit utterances and thresholds on 1,414 dev-cal utterances, then freeze them for all 5,559 test utterances. Inference uses recognized native prefixes, not a gold-prefix upper bound.

Gate positives require the next native token, reference BPE and complete word to agree. The inexpensive native-probability control follows the same protocol. Native decoding owns the final transcript, so unchanged WER 3.058/7.820% is by construction. All final transcripts match the previous official-runtime evaluation.

| configuration | test-clean average display gain ms | test-other average display gain ms | false forecast /100native tokens, clean / other |
|---|---:|---:|---:|
| native probability+gate | 59.27 | 53.74 | 0.335 / 0.507 |
| Our full+gate | 94.31 | 75.31 | 0.397 / 0.583 |
| Our band2+gate | 89.58 | 70.28 | 0.375 / 0.537 |

Gains pair identical correct final words relative to native eager display. Simulated arrival plus measured added CPU work is not independently aligned phonetic latency or live microphone/UI/contention. Incremental band2 gain over native probability+gate is 30.32/16.54ms; utterance/speaker bootstrap intervals are above zero. Actual test errors differ, so matched-test-error Pareto improvement is not established. Thresholds were not changed after observing results.

The average gain of band2 was smaller than 4.73/5.03ms full, and the difference CI excludes 0. Instead, text KV/logp decreased from the maximum 2.20MiB to 29.9KiB. The cost of an additional forecast is approximately 1.4ms/chunk. Voice history/metadata increases and the overall state is not fixed. The correct word preceded by band2+gate is 28.03/21.98%; The median gain per word is 0ms.

Post-hoc display diagnostics found 0.207/0.281 character-deletion events per 100 final words for band2+gate. Word-suffix revisions also include normal partial-word extension. Original results/source and primary metrics are preserved; the diagnostic did not change weights, gates or thresholds. One predictor seed; FastEmit, irreversible commitment, speculative work reuse and total compute savings remain untested.

The [strong-backbone report](zipformer_ours_2026-11-12.md) contains full tables, training, precision, intervals, display diagnostics and commands. Models: `runs/zipformer_ours/full/best.pt`, `band2/best.pt`; raw evidence/provenance are in the same experiment directory.


## Floor-limited AR + FGN actual combination and expansion — 2026-11-12 completed

After the independent StateFGN replacement study, retain band2 AR and append predicted future encoder features to cross-attention. Token CE backpropagates into FGN; train both with CRPS. Six variants each add 8 epochs on all 28,539 train-clean-100 utterances. Small AR/hybrid: 4.60/10.93M; large: 13.03/19.35M. Shared initial weights; large AR adds eight identity blocks. Native final transcripts stay fixed.

| Configuration | clean Average display gain ms | other Average display gain ms | Additional CPU ms/chunk, clean / other |
|---|---:|---:|---:|
| Small AR alone | 89.63 | 70.30 | 1.29 / 1.32 |
| Small hybrid, no noise | 86.53 | 68.51 | 4.51 / 4.67 |
| Small hybrid, noise | 88.46 | 70.18 | 6.59 / 6.81 |
| Large AR alone | 88.89 | 69.70 | 3.70 / 3.76 |
| Large hybrid, no noise | 83.21 | 66.29 | 8.48 / 8.71 |
| Large hybrid, noise | 81.91 | 65.08 | 11.87 / 12.23 |

Evaluate all 5,559 test utterances after independent dev gate-fit/calibration on 1,297/1,414 utterances. Small stochastic-minus-AR gains: clean −1.17ms, CI −2.02~−0.31; other −0.12ms, CI −0.97~+0.72. Large stochastic-minus-matched-AR gains: −6.99/−4.62ms. Equal dev error budgets do not ensure equal actual test risk, so latency alone does not prove Pareto superiority. Unchanged WER follows fixed transcripts.

Noise slightly lowered the dev NLL of the small/large combination to 1.0791 → 1.0687 / 1.0589 → 1.0450, but it did not lead to display performance. The train–dev gap was also slightly lowered, but both losses were improved together, so it could not be separately proven as an overfitting suppression effect. The expansion target is the AR4 → 12 layer, and the FGN generator is fixed at 6.324M. As a seed·8epoch warm-start pilot, it does not determine general FGN scaling or overfitting prevention.

Final verification checks 6×8 epochs, matched noisy/deterministic initialization, all 5,559 native outputs, reproduction of 12 prior operating points, model/inference-source hashes, and trained-large full-prefix versus exact receptive-field-crop difference 2.86e−6. **Existing band2+gate remains the default.** [Hybrid report](band_fgn_hybrid_2026-11-12.md) contains tables/errors/intervals/curves/architecture; originals remain in `runs/band_fgn_v1/`.


## Predict–correct compute-acceleration direction — 2026-11-12

The user selected acoustic-representation prediction followed by inexpensive correction from actual incoming audio as the next objective. A learned prior may improve ambiguous/noisy speech, but must permit unexpected words to override it. The [current plan](predict_correct_acceleration_plan_2026-11-12.md) records proposed losses, reference supervision, cache consistency, recovery costs and separate predictor/corrector scaling. These modules have not been implemented or trained.

A read-only development profile of 24 utterances and 431 chunks measured native encoder/prediction-network/joiner times of 8.8421/0.0751/0.0764ms per chunk. Encoder share was 98.3159%; a standalone reproduction again measured approximately 98.3%. [Protocol and original records](../experiments/compute_profile_v1/README.md). This is not a live microphone/UI benchmark. The approximately 1.3ms AR forecaster cost is not the unmeasured cost of the proposed FGN/corrector.

Existing model weights, full evaluations and chronological results are preserved. Public prose and report generators were converted to English; original runtime-source hashes describe their execution-time versions. This documentation change does not constitute another model-training run.
