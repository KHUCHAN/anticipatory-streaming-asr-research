# Thesis guidance and reading order

The [2026-11-16 prior-work review](prior_work_predict_refine_2026-11-16.md) adds 17 collected papers, P34-P50, with an [English folder guide](../notes/predict-refine/README.md). The catalog now has 50 entries. It distinguishes future-context simulation, provisional text, learned emission timing and actual encoder-compute reduction; it does not certify novelty or reproduce author results.

## key distinction

| Problem | Do you predict speech that has not actually arrived yet? | Priority Paper |
|---|---|---|
| Predict future utterance text | Yes | [P01 SSR](../notes/anticipatory/P01-ssr.md) |
| Provisional text from synthetic future features | Yes, with revision | [P34 ZeroPrompt](../notes/predict-refine/P34-zeroprompt.md) |
| Simulate missing right context for current recognition | Yes, acoustic context | [P35 CUSIDE](../notes/predict-refine/P35-cuside.md), [P36 CUSIDE-T](../notes/predict-refine/P36-cuside-t.md) |
| Train earlier emissions | No synthetic future required | [P37 FastEmit](../notes/predict-refine/P37-fastemit.md), [P38 TrimTail](../notes/predict-refine/P38-trimtail.md), [P41 Delay penalty](../notes/predict-refine/P41-delay-penalized.md) |
| Decide when to wait for real future audio | Wait rather than fill | [P50 Selective Lookahead](../notes/predict-refine/P50-selective-lookahead.md) |
| Precomputation during input waiting | Assume a next chunk; full-duplex SLM | [P02 Speculative Listening](../notes/anticipatory/P02-speculative-listening.md) |
| Accelerating decoding of received audio | Not established by that fact alone | [P03 SpecASR](../notes/decoding/P03-specasr.md), [P04 ParaASR](../notes/decoding/P04-paraasr.md) |
| Probabilistic draft verification | Usually preserves target distribution for fixed input | [P05](../notes/decoding/P05-speculative-decoding.md), [P06](../notes/decoding/P06-speculative-sampling.md) |
| Multiple candidate tree verification | itself is not a negative prediction | [P07 SpecInfer](../notes/decoding/P07-specinfer.md), [P08 Blockwise](../notes/decoding/P08-blockwise-parallel.md) |
| Bounded-state/context encoding | Efficiency of processing past input | [P15 Emformer](../notes/streaming/P15-emformer.md), [P16 Mamba](../notes/state-space/P16-mamba.md), [P20 LC-BiMamba](../notes/state-space/P20-lc-bimamba.md) |
| Functional probabilistic paths | Probabilistic weather-state forecasting | [P21 FGN](../notes/weather/P21-fgn.md) |

## Recommended Reading Order

1. **How ​​far the target has already been researched:** P01 → P02 → P03 → P04.
2. **Principle of adoption, rejection and recovery:** P05 → P06 → P07 → P08.
3. **Speech and text alignment:** P10 CTC → P09 RNN-T → P11 TDT.
4. **Streaming and memory constraints:** P15 → P18 Speech Slytherin → P20 → P30 Zipformer.
5. **Practical role of FGN:** P21 → P22 GraphCast → P23 GenCast → P29 WeatherNext Cyclones.
6. **Candidate graph and abbreviation confusion:** P25 → P24 → P26; P27·P28 is a separate study of the same name.
7. **Data source:** P31 LibriSpeech → P32 GigaSpeech → P33 Common Voice.
8. **New direct comparisons:** P34 → P35 → P36 → P37; then P49/P43/P44 for two-pass display, P48 for temporal distillation, and P50 for recent adaptive waiting. The new folder guide covers all 17 additions.

## Direct comparison data added after the experiment

Additional sources reviewed after the initial 33-paper catalog are linked below with their comparison role. P43/P44 have now been promoted to collected catalog entries; the other links remain supplementary, outside the current 50-entry count.

| Study | Compare Now |
|---|---|
| [Stateful Conformer with Cache-based Inference](https://arxiv.org/abs/2312.17279) | Strong cache-aware streaming backbone, learning/inference context matching, encoder delay definition |
| [Less Is More](https://arxiv.org/abs/2012.06749) | Merge paths with predictors learned from scratch with short token context |
| [RNN-Transducer with Stateless Prediction Network](https://research.google/pubs/rnn-transducer-with-stateless-prediction-network/) | Predictor and observation voice recognition using only the last token |
| [Tied & Reduced RNN-T Decoder](https://arxiv.org/abs/2109.07513) | Short context, simple predictor, and weight sharing. Classification of experimental conditions including EMBR |
| [On the Prediction Network Architecture in RNN-T](https://www.isca-archive.org/interspeech_2022/albesano22_interspeech.html) | Contrast of predictor structures on the same encoder, N-Concat |
| [Multi-Task Learning for End-to-End ASR Word and Utterance Confidence with Deletion Prediction](https://www.isca-archive.org/interspeech_2021/qiu21b_interspeech.html) | confidence estimation and multi-objective learning; Distinguish between output match and actual correctness |
| [P44 Quality and Stability of Streaming ASR](../notes/predict-refine/P44-streaming-stability.md) | Measuring the stability and correction cost of partial output |
| [P43 Partial Rewriting for Multi-Stage ASR](../notes/predict-refine/P43-partial-rewriting.md) | Merge of fast output/slow correction and flicker; a direct two-pass display comparison |

Distinguish original weather FGN, the current ASR variant, short-context rollout and display confidence. See [current conclusions](conclusion_and_next_steps.md) for supported findings/novelty gaps and [960h plan](librispeech960_benchmark_plan.md) for training comparisons.

## Questions to ask when comparing methods

Inspect observed audio, future context, output unit, acceptance/fallback rules, hardware, model size and idle-time use. Distinguish target-output preservation from empirical similarity in corpus WER when a paper says accuracy is unchanged.

Do not rank unrelated speedup factors together: weather versus diffusion, LLM decoding acceleration and streaming word-commit latency have different denominators.

[Full 50-paper catalog](../papers/INDEX.md) · [Research objectives and methodology](goal_and_methodology.md)
