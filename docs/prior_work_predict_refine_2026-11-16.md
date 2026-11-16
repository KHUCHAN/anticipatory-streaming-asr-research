# Prior work: future context, provisional display and encoder acceleration

Reviewed 2026-11-15 to 2026-11-16. This targeted extension adds **17 papers, P34-P50**, to the initial 33-paper catalog, bringing that snapshot to 50. All 17 source PDFs were collected, with version identifiers, SHA-256 hashes and individual English notes. Core methods and selected experiments were read; selected tables were also checked visually. No author experiment was reproduced. This is a bounded related-work review, not an exhaustive search or a certification of novelty.

**Separate research-only extension:** the [LLM correction review](prior_work_llm_correction_2026-11-16.md) fills the candidate-selection/rescoring gap with P51-P60. The current catalog has 60 entries. That update also distinguishes fixed-list selection from free correction, audits public teacher training text, and records actual streaming-cost evidence. It does not launch the proposed experiments below. No unfilled comparison in this review should be read as proof that nobody has studied that broader topic.

**The basic ideas of filling unavailable future frames, showing provisional words and correcting them later already have close precedents.** ZeroPrompt is the direct comparison for training-free preview; CUSIDE/CUSIDE-T for learned future acoustic context; Fast-U2++ and Partial Rewriting for fast/slow output paths. These papers do not establish that our proposed prediction-plus-correction system can skip expensive current-chunk encoder work at useful accuracy.

[Paper folder and reading order](../notes/predict-refine/README.md) · [Current paper index](../papers/INDEX.md) · [Source/evidence manifest for this 17-paper extension](../references/reviews/predict_refine_2026-11-16.json)

## 1. Which problem each paper solves

| Family | Paper | Mechanism and closest comparison | Consequence for this project |
|---|---|---|---|
| Synthetic future, provisional text | [P34 ZeroPrompt](../notes/predict-refine/P34-zeroprompt.md), Interspeech 2023 | Append zero features; decode provisional future positions; refine after real audio arrives | Required zero-fill baseline; early display itself is established |
| Learned acoustic future | [P35 CUSIDE](../notes/predict-refine/P35-cuside.md), Interspeech 2022 | Jointly train future-feature simulation and ASR | Learned padding is established; joint versus frozen training matters |
| Learned acoustic future, transducer | [P36 CUSIDE-T](../notes/predict-refine/P36-cuside-t.md), ISCSLP 2024 | Future-feature simulator plus Conformer RNN-T | Direct transducer-family comparison; current encoder still runs |
| Multi-channel extension | [P40 CUSIDE-array](../notes/predict-refine/P40-cuside-array.md), ISCSLP 2024 | Streaming beamformer and simulated context in ASR backend | Noise/channel robustness needs its own evidence |
| Revise with arrived audio | [P39 TSCA/DRC](../notes/predict-refine/P39-tsca.md), Interspeech 2024 | Revisit trailing frames using the next real chunk; train right-context masks | Separate actual future prediction from delayed contextual refinement |
| Earlier transducer emission | [P37 FastEmit](../notes/predict-refine/P37-fastemit.md), ICASSP 2021 | Train label transitions to emit earlier | Necessary early-emission backbone/control; no future generator |
| Earlier emission through data augmentation | [P38 TrimTail](../notes/predict-refine/P38-trimtail.md), ICASSP 2023 | Trim training spectrogram tails while retaining labels | Inexpensive training control before adding a forecasting network |
| Explicit time penalty | [P41 Delay-penalized transducer](../notes/predict-refine/P41-delay-penalized.md), ICASSP 2023 | Time-dependent nonblank score penalty in training | Relevant k2-family latency control |
| Alignment-aware latency objective | [P42 Minimum latency training](../notes/predict-refine/P42-minimum-latency.md), Interspeech 2022 | Expected delay loss with reference alignments | Task-aware latency supervision, with extra alignment requirements |
| Fast/slow transcript merging | [P43 Partial Rewriting](../notes/predict-refine/P43-partial-rewriting.md), 2023 preprint | Align and merge two partial transcript streams | Direct prior for two-pass display; merger cost differs from both model costs |
| Display stability measurement | [P44 Streaming stability](../notes/predict-refine/P44-streaming-stability.md), Interspeech 2020 | Count unstable word occurrences and revised segments | Replay every displayed hypothesis; final edit distance misses repeated changes |
| Two-pass pretrained ASR | [P45 U2 Whisper](../notes/predict-refine/P45-whisper-two-pass.md), Interspeech 2025 | Causal CTC partials, attention rescoring and hybrid vocabulary | Existing two-pass adaptation alternative, with different pretraining/data |
| Semantic context distillation | [P46 SENS-ASR](../notes/predict-refine/P46-sens-asr.md), LREC 2026 | Distill sentence semantics into a context module from arrived encoder frames | Language-informed acoustic augmentation exists; not encoder omission |
| Future-aware self-distillation | [P47 Learning from Back Chunks](../notes/predict-refine/P47-back-chunks.md), Interspeech 2024 | Match earlier look-ahead representations to later chunks | Task-aware feature/probability distillation is established |
| Temporally aligned distillation | [P48 Delayed-KD](../notes/predict-refine/P48-delayed-kd.md), Interspeech 2025 | Minimize CTC KL over an allowed teacher/student delay range | Audit temporal alignment before assuming larger KL weight fixes a student |
| Shared fast/slow encoder paths | [P49 Fast-U2++](../notes/predict-refine/P49-fast-u2pp.md), 2022 preprint | Lower layers use small chunks; upper layers use larger chunks; distill lower path | Compare shared work against two independent encoders |
| Adaptive waiting | [P50 Selective Lookahead](../notes/predict-refine/P50-selective-lookahead.md), accepted SLT 2026 | Learn when to wait for more audio; bound lookahead and deferral | Adaptive confidence gating is also a direct recent prior |

ISCSLP/SLT acceptance above is recorded from the authors' arXiv metadata. P43 and P49 remain labeled as collected preprints because this review did not confirm a publisher record. P40 evidence uses collected v1, although arXiv also lists v2. Proceedings and preprint dates can differ; this is particularly relevant to TSCA, published in 2024 but posted to arXiv in 2025.

Existing [P01 SSR](../notes/anticipatory/P01-ssr.md) remains the closest text-continuation reference. [P02 Speculative Listening](../notes/anticipatory/P02-speculative-listening.md) concerns precomputation in a full-duplex speech-language model. Neither should be replaced by a general claim that all these papers solve the same task.

## 2. Corrections to the supplied summary

### ZeroPrompt: earlier display, with extra computation

The paper's LibriSpeech test-clean example at a 640ms chunk and a 640ms prompt reports first display **1136 to 662ms** and last display **7328 to 7098ms**. Final greedy/rescored WER remains **4.41/3.80%**, while greedy RTF rises **0.04826 to 0.07006**, approximately **45.2%**. These are author-reported values, not our reproduction. See Table 1, PDF p. 3. [Original paper](https://www.isca-archive.org/interspeech_2023/song23c_interspeech.html)

The final-output argument relies on protected chunk-causal attention and prompt-and-refine. A synthetic chunk must not contaminate the native state that later consumes real audio. A port to a different encoder, convolution cache or RNN-T decoder must verify that property. The paper does not establish exact output preservation for arbitrary repeat padding or Zipformer implementations. [Methods, Figures 1-3](https://arxiv.org/abs/2305.10649)

**Intermediate ZeroPrompt already inserts the prompt at hidden encoder layers** (Section 3.3, Table 2). Therefore, moving a preview from input features into hidden features is not by itself a new contribution. Its TDT metrics omit compute time; PER counts characters. Neither is directly our stable-word delay or word-rewrite rate.

### CUSIDE is not simply a different filler for ZeroPrompt

CUSIDE/CUSIDE-T use simulated future acoustic features as recognition context, jointly train the simulator with ASR, and retain the current chunk's encoder work. ZeroPrompt also exposes text decoded at the synthetic future positions as provisional output. The distinction is between improving current recognition context and explicitly displaying speculative continuation. Their source notes retain the Chinese CER, LM/rescoring and hardware conditions. [CUSIDE](https://www.isca-archive.org/interspeech_2022/an22_interspeech.html), [CUSIDE-T](https://arxiv.org/abs/2407.10255)

This prior work weakens a broad claim of novelty for learned future-frame filling. It does not determine whether our frozen FGN state forecast is useful, or whether a cheaper corrector can replace a strong encoder. Those are different experimental questions.

### FastEmit and later papers already target emission timing

FastEmit alters transducer training gradients; it is not an inference-time token prior or a KL coefficient. TrimTail and delay-penalized transducers provide other direct timing controls. A forecasting method should be compared against a backbone that has already been trained to emit early. The presence of FastEmit in a recipe alone does not determine how much forecast benefit remains. [FastEmit](https://arxiv.org/abs/2010.11148), [TrimTail](https://arxiv.org/abs/2211.00522), [Delay penalty](https://arxiv.org/abs/2211.00490)

The same caution applies to headline performance. Delayed-KD's **5.42% CER** is after utterance-level rescoring; its first-pass streaming value is **6.11%**, with 40ms chunks and an 80ms training alignment buffer. Those numbers do not mean that complete recognition finishes in 40ms. See Table 1, PDF p. 4. [Delayed-KD](https://www.isca-archive.org/interspeech_2025/li25u_interspeech.html)

The 2026 Selective Lookahead paper also illustrates a median/tail trade-off: its unbudgeted dynamic policy has a better median but worse p90 than one-chunk static lookahead. It reports extra decoding work and excludes compute time from emission delay. An adaptive gate is not automatically a runtime acceleration. [Selective Lookahead](https://arxiv.org/abs/2609.37611)

## 3. Definitions that must remain separate

| Quantity | What is counted | Do not substitute |
|---|---|---|
| Final WER / CER | Final string against a reference, with specified normalization and unit | Agreement with the model's own later output |
| First/last TDT in ZeroPrompt | First/last displayed token on an utterance timeline | Per-word acoustic-end-to-stable-display latency |
| FastEmit PR50/PR90 | Finalized last-token emission relative to aligned speech end | Average delay of every word |
| Chunk or right-context duration | Required audio buffering under a stated definition | Total wall-clock delay, including computation |
| Reference-correct display delay | When a correctly recognized word appears, anchored to aligned audio | Accuracy of all predicted words, unless errors/coverage are also counted |
| Stable display delay | Earliest output time after which the displayed item never changes | First appearance or irreversible commitment |
| Forecast withdrawal fraction | Fraction of forecasts later withdrawn, with an explicit denominator | ZeroPrompt character PER or UPWR |
| UPWR / UPSR | Repeated unstable word occurrences / revised segments | A bounded probability or only the final two strings' edit distance |
| Runtime / RTF | Full declared inference pipeline on the same device, backend and threading | A simulator's standalone cost or omitted main encoder work |

For comparison, report both the distribution of delays and the proportion of words included. Correct-only latency can omit difficult errors. Always include final errors, false previews, repeated revisions and coverage alongside that distribution. A consistently wrong but stable token is still wrong.

## 4. Implications for our architecture and next comparisons

These are project interpretations and proposed controls, not new experimental results.

| Proposed claim | Closest established prior | What would still need to be demonstrated here |
|---|---|---|
| Training-free provisional future display | ZeroPrompt | A useful improvement over zero/repeat padding at equal cost and revision risk |
| Learned future acoustic features | CUSIDE, CUSIDE-T | A matched benefit of the proposed forecast, including frozen versus joint-training controls |
| Hidden-space prompting | Intermediate ZeroPrompt | A mechanism or measured advantage beyond relocating the prompt |
| Fast display plus slow correction | Fast-U2++, Partial Rewriting, U2 Whisper | Better complete accuracy-delay-rewrite-cost trade-off under the same output protocol |
| Learned confidence-based waiting | Selective Lookahead | Advantage over existing calibrated triggers, with controlled tail delay and compute |
| Predict state, correct, skip current encoder | No evidence for our implementation follows from the papers above | Accuracy-retaining reduction in total inference cost versus a small encoder and audio-only/previous-corrected-state controls |

For an actual encoder-skipping design, count prediction, correction, gating, fallback, decoding and cache recovery. In a simplified per-chunk cost model, `C_new = C_predict + C_correct + C_gate + p_fallback*C_full + C_recovery + C_decode`. Compare with the same-run complete baseline, with shared work deducted only if it is actually shared. Displaying earlier while still running the full encoder is a different result.

A concrete next protocol, if separately authorized, would be:

1. **Protect and verify the native path.** Keep provisional caches isolated; compare token sequences and timestamps against native decoding after all real audio arrives. Check chunk boundaries, start/tail padding and long streams.
2. **Compare fillers fairly.** No preview, zero fill, last-frame repeat, simple mean/extrapolation and learned future prediction. Fix the backbone, available audio, preview schedule and actual compute budget. An oracle future is a labeled diagnostic, not a deployable result.
3. **Evaluate display and compute separately.** Reference-correct preview precision/coverage, stable-delay p50/p90/p95, all rewrite events, final WER, complete RTF, peak memory and fallback cost. Tune gates only on designated dev data.
4. **Use relevant controls.** FastEmit/delay-trained backbone, native confidence gate, a small streaming encoder and shared-layer fast/slow architecture where supported. For prediction-plus-correction retain A audio-only, B forecast and C previous corrected state with matched training/resource conditions.
5. **Audit distillation alignment before another weight sweep.** Delayed-KD motivates inspecting temporal correspondence, but its CTC solution is not a validated remedy for our RNN-T student. A2/A3 results remain unchanged by this literature review.

The current local evidence is recorded separately in [current conclusions](conclusion_and_next_steps.md), [20M reproduction correction](small_zipformer_reproduction_2026-11-15.md), and [A3 matched KL results](kl_weight_a3_2026-11-15.md). The supplied discussion's repeat-padded incomplete-chunk experiment and state-prediction R-squared of 0.46 were not located in the committed artifacts reviewed here; they are not imported as verified results. Reconstruction R-squared alone would not establish useful downstream accuracy or net compute savings anyway.

## 5. Collection, search and reproduction boundary

The review started from the four named studies, then followed full-text references, ISCA proceedings, arXiv records, ACL publisher metadata and official code repositories. Recent extensions include Delayed-KD (2025), SENS-ASR (2026) and Selective Lookahead (2026). Related work cited in these papers is broader than the 17 selected additions. Unread references remain candidates, not evidence for a novelty claim.

The [review manifest](../references/reviews/predict_refine_2026-11-16.json) records search queries, evidence locations, collected versions, later-version caveats, code availability and visual checks. The unrelated NLP paper also called ZeroPrompt and a similarly named game were excluded. No statement that no further 2024-2026 prior art exists is supported by this search.

Originals are stored locally under `papers/pdf/P34_zeroprompt.pdf` through the corresponding P50 file. The public repository contains links, hashes, notes and BibTeX; publisher PDFs, extracted full text and rendered source pages remain local. Retrieve the same catalog versions with:

```bash
python3 scripts/download_papers.py --category predict-refine
python3 scripts/validate_catalog.py --pdfs
```

Official implementation pointers are recorded as confirmed repository, toolkit only, or unconfirmed exact implementation. Finding a GitHub repository does not establish successful reproduction or checkpoint parity.
