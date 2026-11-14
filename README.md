# Anticipatory Streaming ASR Research

Research on predicting acoustic representations, correcting them with newly arrived speech, and reducing streaming ASR computation at controlled recognition error.

**Current direction:** reuse the existing FGN-style state predictor and bounded-context AR work as a starting point for a lightweight audio-conditioned correction path. The residual corrector and encoder-skipping runtime are **proposed, not implemented**. A learned prior may help ambiguous speech, but must allow unexpected words to override the forecast.

Start with the [predict–correct acceleration plan](docs/predict_correct_acceleration_plan_2026-11-12.md), [current conclusions](docs/conclusion_and_next_steps.md), and [repository map](READ.MD).

**Epoch-wise double-descent experiment running:** the [declared long-horizon protocol](experiments/fgn_double_descent_v1/README.md) continues all three epoch-27 models through 96 without early stopping. It adds full-training hard-error/margin checks and preserved milestone checkpoints. No late-recovery result is claimed yet; this is not a broad model-wise size sweep.

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

See [READ.MD](READ.MD) for preparation and execution commands. Snapshot date: 2026-11-12.
