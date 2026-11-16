# Anticipatory Streaming ASR Research

Research on predicting acoustic representations, correcting them with newly arrived speech, and reducing streaming ASR computation at controlled recognition error.

**Current direction:** predict an acoustic representation and correct it cheaply with arriving audio. The initial correction pilot and the A2/A3 student paths fail their recognition-accuracy gates. Useful accuracy-preserving encoder acceleration is not established for these student replacements. Adaptive encoder skipping and teacher-cache recovery remain proposed.

Start with the [predict–correct acceleration plan](docs/predict_correct_acceleration_plan_2026-11-12.md), [current conclusions](docs/conclusion_and_next_steps.md), and [repository map](READ.MD).

**Research-only LLM review added, 2026-11-16:** [ten papers on candidate selection, correction and fusion](docs/prior_work_llm_correction_2026-11-16.md), with an [English paper folder](notes/llm-correction/README.md). The catalog now has 60 entries. The review separates offline WER, streamable algorithms, actual streaming measurements and extra inference cost; it also checks the teacher's public training provenance. No new experiment or architecture change was executed.

**Future-context review added, 2026-11-16:** [17 papers and method comparisons](docs/prior_work_predict_refine_2026-11-16.md), including ZeroPrompt, CUSIDE/CUSIDE-T, FastEmit, Delayed-KD and 2026 Selective Lookahead. The [paper folder](notes/predict-refine/README.md) contains English notes, evidence locations and implementation status. This extension brought the initial 33-paper catalog to 50; original PDFs remain local-only. Provisional display, future-context simulation and actual encoder-compute reduction are distinguished explicitly. This literature review does not rerun or revise the A2/A3 results.

**20M reproduction corrected; A3 completed:** a [legacy decoder padding mismatch](docs/small_zipformer_reproduction_2026-11-15.md) caused the earlier small-model gap; corrected WER is **3.56/10.97%**, with all 800 independent native token/chunk checks passing. The [matched A3 results](docs/kl_weight_a3_2026-11-15.md) are A2 control **42.30/61.92%**, KL 4x **41.38/61.52%**, KL 16x **42.39/63.11%**. Both paired 95% intervals for 4x versus control include zero. Selected 4x costs **1.192/1.186ms per chunk**, or 13.18% of the same-run teacher, but fails the <10% accuracy gate. Same-run full/small costs are **9.045/9.042ms** and **4.289/4.293ms**. Only the KL coefficient changed; the initial checkpoint, seed, data/windows and 24-epoch budget are matched, and the first 100 control updates reproduce exactly. One seed, existing dev roles, no test split or 960h expansion.

**A2 qualification completed — cost passes, accuracy fails:** [9.60M audio-only results](docs/predict_correct_a2_2026-11-15.md), after 12 MAE epochs and 24 calibrated-KL epochs. Selected A2 has dev-clean/other WER **42.30/61.92%**, versus full Zipformer **2.93/8.82%**. Complete CPU cost is **1.191/1.187ms per chunk**, versus **8.942/8.935ms**: 13.32% of same-run teacher cost on clean, but WER remains above the <10% gate. The original small-model operating point is superseded by the correction above. One seed, existing development roles, no test evaluation or renewed FGN/B/C training.

**Prediction-correction pilot completed:** [48 correction-training epochs and 800 dev utterances](docs/predict_correct_pilot_2026-11-14.md), with corrected-only feedback. WER clean/other: A audio-only 71.40/80.38%, B frozen FGN 69.34/79.72%, C previous corrected state 70.08/79.75%, versus full encoder 2.93/8.82%. B costs 3.89/3.85ms per chunk, versus C 3.09/3.08ms and full encoder 9.19/9.28ms. B improves on A on clean, but its advantage over C is not established; both paired intervals include zero. No cheap path retains full-encoder accuracy, so useful encoder-skipping acceleration is not demonstrated.

**Interpretation review:** the failed audio student does not settle general FGN usefulness. A [post-hoc loss-gradient audit](docs/predict_correct_review_2026-11-14.md) finds KL-to-MAE gradient norm ratios near 27 at two original A checkpoints. The subsequent A2 study implements the proposed chunk-wide stride-4 design and staged loss calibration. Its state MAE improves below a constant baseline, but its WER still fails qualification; better representation loss alone is insufficient.

**Epoch-96 double-descent investigation completed:** [all three generators](docs/fgn_double_descent_2026-11-13.md) completed 69 additional epochs each, without early stopping, in 7.50 hours. Dev token NLL at epoch 27 → 96 was 1.1147 → 1.3571 / 1.1269 → 1.4322 / 1.1306 → 1.4575; best-dev checkpoints remain at epoch 5. Token NLL rises across all declared late windows, while state CRPS shows recovery. No sustained second token-risk descent was observed through epoch 96. The small generator's final state CRPS is slightly below its trained epoch-0 score; a double-descent mechanism, model-wise behavior and ASR-WER improvement are not established. All checkpoints and input-preservation checks passed.

**Longer training completed:** [16 extra epochs per generator](docs/fgn_capacity_long_2026-11-13.md) brought all three models to epoch 27. Train-probe NLL fell to 0.1313 / 0.1322 / 0.1319, while dev NLL rose to 1.1147 / 1.1269 / 1.1306; all best-dev checkpoints remain at epoch 5. Added epochs improve fitting without a validation benefit in this fixed-case, single-seed task. Actual checkpoint settings confirm SiLU, no FGN ReLU modules, and fixed 256-dimensional hidden/memory sizes. Original results and checkpoints are preserved.

The [double-descent review](docs/double_descent_review_2026-11-13.md) limits that conclusion to the measured range. Later recovery and other size regimes are not excluded. State CRPS already shows late recovery; no new training was run for this descriptive audit.

**FGN capacity diagnosis completed:** [6.32M / 10.54M / 18.96M generators](docs/fgn_capacity_2026-11-12.md) all reached the declared overfit rule after 11 continuation epochs. At that common budget, train-probe NLL was 0.2288 / 0.2269 / 0.2228; best dev NLL was 0.9486 / 0.9509 / 0.9504. This single-seed, fixed-case study shows limited fitting benefit from depth growth and no observed dev improvement. It does not evaluate the proposed residual corrector, final WER or inference acceleration.

## Completed evidence

| Study | Result | Interpretation |
|---|---|---|
| Public 66.11M Streaming Zipformer, full 5,559-utterance test | WER 3.058% clean / 7.820% other | Reproduction of a public 960h checkpoint; no new acoustic training |
| Our 4.60M layer-band2 AR + gate, trained on 100h | Correct-word mean display gain 89.58ms / 70.28ms over eager native output | Provisional display, not irreversible commitment or compute acceleration |
| Inexpensive native-probability gate | Display gain 59.27ms / 53.74ms | Our incremental gain is 30.32ms / 16.54ms; actual test error rates differ |
| Actual AR + FGN hybrid, six variants | Small stochastic hybrid 88.46ms / 70.18ms; large 81.91ms / 65.08ms | Did not outperform existing band2; additional computation increased |
| Native CPU component profile, 24 dev utterances | Encoder approximately 98.3% of neural time in two measurements | Encoder work is the primary acceleration target on this adapter |

The median per-word display gain is zero. Final WER in the display experiments is unchanged by construction. The previous two-pass claim of P50 48ms / P95 209ms was withdrawn after screen-event replay corrections. These are not live-device or mobile benchmarks.

## Read and reproduce

- [Research goal and methodology](docs/goal_and_methodology.md)
- [Complete experiment history](docs/research_plan_3stage.md)
- [Our method on a strong backbone](docs/zipformer_ours_2026-11-12.md)
- [FGN hybrid and scaling limitations](docs/band_fgn_hybrid_2026-11-12.md)
- [Experiment directories and public artifacts](experiments/README.md)
- [60-paper catalog](papers/INDEX.md), [literature map](docs/literature_map.md), and [BibTeX](references/references.bib); initial 33 plus 17 future-context and 10 LLM-related additions
- [Dataset guide](data/README.md) and [inventory](data/reports/librispeech_inventory.md)

LibriSpeech training and evaluation data has been prepared locally: 292,367 utterances / 982.308 hours in total, including a training-only union of 281,241 utterances / 961.054 hours. A new 960h acoustic model has not been trained. The approximately 114M model recipe remains a separate proposal.

Public documentation, paper notes, catalog text and report generators are in English. Original PDFs, speech, transcripts, feature caches, downloaded weights and per-utterance run outputs are local-only; source code and aggregate experimental evidence are versioned.

```bash
python3 scripts/validate_catalog.py
python3 scripts/check_public_repository.py
```

See [READ.MD](READ.MD) for preparation and execution commands. Snapshot date: 2026-11-16.
