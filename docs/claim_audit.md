# Review the claims in the attached conversation

Preserves initial claims and corrections. [Current conclusions](conclusion_and_next_steps.md) take precedence, including subsequent correct-and-kept gate/P1–P4 checks. Distinguish the original unproven objective from demonstrated single-pass display benefit.

Review date: 2026-11-10. Initial explanations were checked against primary sources rather than copied as established facts.

| Issue in the original explanation | Verified finding and documentation | Evidence |
|---|---|---|
| Is FGNN Google's weather model? | Google's FGN is a Functional Generative Network. Factor Graph NN and Finite Gaussian NN are separate architectures. | [P21](../notes/weather/P21-fgn.md), [P27](../notes/disambiguation/P27-factor-graph.md), [P28](../notes/disambiguation/P28-finite-gaussian.md) |
Is it faster than Transformer because it is | Chain rule? | Distinguish between the chain rule itself and modeling assumptions that limit the past context. Transformer also uses autoregressive decomposition. | [P21 Eq. (1)](https://arxiv.org/abs/2506.10772), [P05](https://proceedings.mlr.press/v202/leviathan23a.html) |
Is the state of | FGN dimension 32? The | 32 dimension is the injecting probability noise. The two wait states that are model inputs are high-dimensional arrays. | [P21 Original text](https://arxiv.org/abs/2506.10772) |
| Does one forward generate every future candidate and time? | It refers to generation for one weather step and ensemble member. Rollout and additional ensemble members still require computation. | [P21](../notes/weather/P21-fgn.md) |
| FGN is 8 times faster, so is ASR also faster? This is the result of comparison with diffusion sampling of | meteorological GenCast. This is not the basis for Whisper·ASR speed. | [P21 Section 3.3](https://arxiv.org/abs/2506.10772) |
The exact source of the speed figures of | ConMamba 2025 | 63–70% is the author's ASR comparison paper. Distinguished from the original ConMamba study. | [P19](https://www.isca-archive.org/interspeech_2025/zevallos25_interspeech.html), [P18](https://arxiv.org/abs/2407.09732) |
| SpecASR 3.04–3.79 times | Be sure to write down the Whisper path and the experiment settings using Llama/Vicuna simulated target. | [P03 PDF 5 page](https://arxiv.org/pdf/2507.18181v2) |
| Average 5 adoption of ParaASR | This is a specific MTP setting for WenetSpeech meeting, and the conditions were recorded along with H800 RTF. There were also some numerical discrepancies between the main text tables. | [P04 Tables 1–4](https://arxiv.org/abs/2607.29279) |
| Is Speculative Listening good at guessing future ASR words? | full-duplex output pre-calculation study. The ratio of output being equal to silence and actual input is not the accuracy of the next word. | [P02](https://arxiv.org/abs/2611.06982) |
| Does TDT skip voice processing entirely? Transducer decoding time axis proceeds according to | duration. An acoustic encoder is not always unnecessary. | [P11](https://proceedings.mlr.press/v202/xu23g.html) |
| Is Mamba always a fast streaming ASR? | Depends on length, operation kernel, and bidirectional/causal settings. CPU/MPS performance is subject to separate measurement. | [P18](https://arxiv.org/abs/2407.09732), [P20](https://www.isca-archive.org/interspeech_2025/moriya25_interspeech.html) |
| Can LM+ASR be combined with simple Bayesian multiplication? | Likelihood and posterior must be distinguished, and the language prior can be reflected redundantly. Combination and verification rules are unconfirmed. | [RNN-T formulation](https://arxiv.org/abs/1211.3711); Methodological review of this project |
| PyTorch 2.14 MPS CTC support | Forward/backward support was confirmed in the official announcement. The currently installed version is 2.12.0 and has not been upgraded. | [Official 2.14 announced](https://pytorch.org/blog/pytorch-2-14-release-blog/) |
| Common Voice 42,593Time·295Language | v27 scripted All recording statistics. English is not validated on the scale of learning time or spontaneous conversation data. | [Mozilla v27 Notice](https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490) |
If | 100 time·10–20M, is accuracy/learning time guaranteed? | This is only an initial scale proposal. Model size and learning time are not confirmed before actual structure, computation, and memory measurements. | [Preparation scope and decision items](goal_and_methodology.md) |
| Is the detailed structure of WeatherNext 3 in the FGN paper? The existence of the | official product page has been confirmed. P21 is a technical report of WN2 and does not assume that the open structure of WN3 is the same. | [Official Product Page](https://deepmind.google/science/weathernext/), [WN2 Official Store](https://github.com/google-deepmind/weathernext) |

Measure bounded state, memory banks, KV caches and candidate-tree caches separately. A bounded encoder does not establish bounded system memory if other components grow indefinitely.

## Add review of 2026-11-11 local experiment results

Compare saved `runs/final/summary.json`, event files and evaluation code. Preserve original numbers while limiting interpretation below. [Separate review](results_review_2026-11-11.md) contains main tables, original SHA-256 and next checks.

| Claim / issue | Verified finding |
|---|---|
| Are the results recorded? | H1–H3 156 Summary of settings JSON/Markdown and border-specific events are local. The learning/overall assessment was not re-run in this review. |
| Are gate data and dev evaluation separated? By comparing the | stored IDs, we confirmed the intersection of 0 utterances of each dev-clean/dev-other chunk. It is different from speaker separation verification. |
Did | H1 actually learn the correct answer? | `labelled()` then creates a label if the decoder confirmed output and forecast are the same. There are no implementation results yet regarding the correctness head/loss based on the actual correct answer transcription standard. |
| Has the early display increased at the same precision? The goal of | reference value selection is the same as 95%, but the actual test-clean 320 ms is raw 96.6%, full gate 96.0%. Distinguish from same-risk comparison. |
| 371→48 Is ms the confirmed delay actual measurement? | is a post-calculation of the displayed time. The final transcription is determined by the slow pass. It is not an end-to-end measurement of actual audio input, screen rendering, and simultaneous execution. |
| Did the final pass revise 11 of every 100 displayed words? | This is edit distance between final strings per 100 final-pass words, not all screen revision or forecast-retraction events. |
| 25 ms/voice 1 seconds is the total system cost? | is the estimated total of the encoder cost. The entire NAR, gate, decoder, merge, and resource contention are not included, and `tn` is not added to the forecast display time. |
Is | H2 a short context FGN rollout? The | entire history AR predictor was rolled forward up to 3 tokens. This is not a model learned with past 3 tokens. |
| Does original FGN generate every future time at once? | It generates the next state from two recent weather states, then feeds predictions back for rollout. This differs from the parallel 6-token ASR variant. |
Is the weak acoustic model the reason for the small effect in | other? | There is no possible explanation or cause isolation experiment. It is not confirmed that the acoustic model quality is the cause. |

### Reflection of subsequent correction results

[Calibration and additional verification](research_plan_3stage.md#calibration-and-further-verification) supersede this initial review. Use the [review's current assessment](results_review_2026-11-11.md) for current numbers and conclusions.

| Follow-up claim | Current interpretation |
|---|---|
| 2-pass's 48 ms·209 ms·amended 11 | withdrawn due to current results. The configuration of 300 speech screen playback includes P50 86.8 ms, P95 296.4 ms, and cumulative rewrite 34.64/100 words. |
| Gate improvement at the same retraction rate was confirmed by comparison by scanning and interpolating the reference value curve on the | evaluation set. It is distinguished from the test operation performance of the threshold fixed by dev. |
| Gate improvement Most of the FGN signals | Calibrated test-clean 320 ms The table shows the total gate −9.95 ms compared to raw, and −6.01 ms excluding NAR/FGN signals. It is limited to the additional contribution of separate model signals. |
| 3 tokens are sufficient. In short context structures such as |, no benefit from 32 token expansion was observed. The cause of the gap with equivalence, universal sufficiency, or AR has not been determined. |
| is the token ID formula for the state 24 byte | `P*8` per hypothesis. It does not include separate logp·trie·voice state·temporary tensor, so it is distinguished from the actual memory of the entire system. |
| No additional single pass calculations. | There are no additional network calls for forecasts that only use existing AR scores. There is additional computation in the learning gate that uses a separate NAR signal. |
