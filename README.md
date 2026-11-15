# Anticipatory Streaming ASR Research

Research on predicting acoustic representations, correcting them with newly arrived speech, and reducing streaming ASR computation at controlled recognition error.

**Current direction:** predict an acoustic representation and correct it cheaply with arriving audio. Both the initial correction pilot and the A2 audio-only follow-up fail their recognition-accuracy gates. Useful accuracy-preserving encoder acceleration is not established. Adaptive encoder skipping and teacher-cache recovery remain proposed.

Start with the [predict–correct acceleration plan](docs/predict_correct_acceleration_plan_2026-11-12.md), [current conclusions](docs/conclusion_and_next_steps.md), and [repository map](READ.MD).

**A2 qualification completed — cost passes, accuracy fails:** [9.60M audio-only results](docs/predict_correct_a2_2026-11-15.md), after 12 MAE epochs and 24 calibrated-KL epochs. Selected A2 has dev-clean/other WER **42.30/61.92%**, versus full Zipformer **2.93/8.82%**. Complete CPU cost is **1.191/1.187ms per chunk**, versus **8.942/8.935ms**: 13.32% of same-run teacher cost on clean, but WER remains above the <10% gate. The small native Zipformer measures 23.71/32.35% at 4.252/4.240ms; its published performance level is not reproduced. Independent official-engine decoding matches all 800 token sequences, and the remaining publication gap is unresolved. One seed, existing development roles, no test evaluation or renewed FGN/B/C training.

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
- [Initial 33-paper catalog](papers/INDEX.md), [literature map](docs/literature_map.md), and [BibTeX](references/references.bib)
- [Dataset guide](data/README.md) and [inventory](data/reports/librispeech_inventory.md)

LibriSpeech training and evaluation data has been prepared locally: 292,367 utterances / 982.308 hours in total, including a training-only union of 281,241 utterances / 961.054 hours. A new 960h acoustic model has not been trained. The approximately 114M model recipe remains a separate proposal.

Public documentation, paper notes, catalog text and report generators are in English. Original PDFs, speech, transcripts, feature caches, downloaded weights and per-utterance run outputs are local-only; source code and aggregate experimental evidence are versioned.

```bash
python3 scripts/validate_catalog.py
python3 scripts/check_public_repository.py
```

See [READ.MD](READ.MD) for preparation and execution commands. Snapshot date: 2026-11-15.
