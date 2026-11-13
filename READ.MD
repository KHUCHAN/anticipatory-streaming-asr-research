# Repository guide and file map

Snapshot date: 2026-11-13. All public prose is maintained in English.

The selected goal is to predict an acoustic representation, correct it using newly arrived audio, and avoid sufficiently expensive encoder work at controlled recognition error. Existing display studies are completed; the residual-correction and encoder-skipping architecture is still a proposal.

## Reading order

| Document | Purpose |
|---|---|
| [Predict–correct acceleration plan](docs/predict_correct_acceleration_plan_2026-11-12.md) | Current user-selected goal, proposed architecture/losses, scaling order, noisy-speech benefit and unexpected-speech risk |
| [Current conclusions](docs/conclusion_and_next_steps.md) | Supported, withdrawn and unproven claims; representative results |
| [Research goal and methodology](docs/goal_and_methodology.md) | Original predict–verify objective, hypotheses, evaluation and causal-input rules |
| [Full experiment history](docs/research_plan_3stage.md) | Stage 1–3, F/G/H studies, measurement corrections and subsequent experiments |
| [Strong-backbone experiment](docs/zipformer_ours_2026-11-12.md) | Actual full/band2 training, independent dev calibration and full test evaluation |
| [FGN hybrid study](docs/band_fgn_hybrid_2026-11-12.md) | Actual joint AR+FGN continuation, two AR depths, costs and scaling limitations |
| [Independent FGN state study](docs/fgn_state_pilot_2026-11-12.md) | Recurrent continuous-state predictor and component ablations |
| [Compute profile](experiments/compute_profile_v1/README.md) | Reproducible encoder/prediction-network/joiner timing breakdown |
| [FGN generator capacity diagnosis](docs/fgn_capacity_2026-11-12.md) | Completed 6.32M / 10.54M / 18.96M comparison; all met the overfit rule at epoch 11; no test use |
| [Longer FGN training](docs/fgn_capacity_long_2026-11-13.md) | Completed epoch 11 → 27 continuation; fitting improves, dev NLL worsens; actual SiLU settings verified |
| [Implementation review](docs/results_review_2026-11-11.md) | Metric definitions, missing costs and corrected claims |
| [Four follow-up experiments](docs/followup_experiments_2026-11-11.md) | Real cache release, long streams, seed repetitions and display stabilization |
| [Public baseline reproduction](docs/baseline_zipformer_2026-11-11.md) | Frozen public 66.11M checkpoint, inference conditions and WER |
| [960h benchmark proposal](docs/librispeech960_benchmark_plan.md) | Prepared data, speaker-separated dev roles and unexecuted acoustic-training recipe |
| [Literature map](docs/literature_map.md) and [33-paper index](papers/INDEX.md) | Source relationships, notes, original links and reading priorities |
| [Dataset guide](data/README.md) and [inventory](data/reports/librispeech_inventory.md) | Local data organization and verified preparation status |

## File structure

```text
.
├── README.md                         # GitHub entry point and current status
├── READ.MD                           # This guide
├── .gitignore                        # Local data, PDFs, weights and runs
├── docs/
│   ├── predict_correct_acceleration_plan_2026-11-12.md
│   ├── goal_and_methodology.md
│   ├── conclusion_and_next_steps.md
│   ├── architecture.md               # Historical AVS plus later adapters
│   ├── research_plan_3stage.md        # Chronological evidence
│   ├── results_review_2026-11-11.md
│   ├── followup_experiments_2026-11-11.md
│   ├── baseline_zipformer_2026-11-11.md
│   ├── zipformer_ours_2026-11-12.md
│   ├── fgn_state_pilot_2026-11-12.md
│   ├── band_fgn_hybrid_2026-11-12.md
│   ├── fgn_capacity_2026-11-12.md
│   ├── fgn_capacity_long_2026-11-13.md
│   ├── librispeech960_benchmark_plan.md
│   ├── literature_map.md
│   ├── claim_audit.md
│   ├── datasets_and_mac.md
│   ├── resources.md
│   └── scope_and_provenance.md
├── notes/                            # 33 English paper notes
│   ├── anticipatory/                 # P01–P02
│   ├── decoding/                     # P03–P08
│   ├── streaming/                    # P09–P15, P30
│   ├── state-space/                  # P16–P20
│   ├── weather/                      # P21–P23, P29
│   ├── graph-asr/                    # P24–P26
│   ├── disambiguation/               # P27–P28
│   └── datasets/                     # P31–P33
├── papers/
│   ├── INDEX.md
│   └── pdf/README.md                 # Original PDFs are local-only
├── references/
│   ├── catalog.json                  # Curated English metadata and evidence
│   ├── catalog.csv
│   └── references.bib
├── configs/
│   ├── zipformer_baseline.json        # Executed baseline conditions
│   ├── zipformer_ours.json            # Executed forecaster/gate study
│   └── librispeech960_benchmark.json   # Proposed acoustic-training settings
├── data/
│   ├── README.md
│   ├── datasets.json
│   ├── reports/                      # Public aggregate verification reports
│   ├── raw/                          # Local-only audio, text and archives
│   └── processed/                    # Local-only manifests and features
├── avs/                              # Acoustic, AR/FGN and streaming code
├── experiments/
│   ├── README.md                     # Study index
│   ├── fgn_state_v1/                 # Independent recurrent-state study
│   │   └── artifacts/                # Public aggregate results and figures
│   ├── band_fgn_v1/                  # Actual AR+FGN hybrid study
│   │   └── artifacts/                # Public results, checks and curves
│   ├── compute_profile_v1/           # Read-only bottleneck measurement
│   │   └── artifacts/                # Initial and reproduced timings
│   ├── fgn_capacity_v1/              # Actual generator growth / fitting study
│   │   └── artifacts/                # Verified curves, completion and provenance
│   └── fgn_capacity_long_v1/         # Extra epochs beyond the original overfit stop
│       └── artifacts/                # Verified extended curves, settings and completion
├── scripts/                          # Preparation, training and analysis
├── requirements-baseline.txt
├── requirements-zipformer-features.txt
├── requirements-forecaster.txt
├── runs/                             # Local-only checkpoints and raw outputs
├── checkpoints/                      # Local-only downloaded model weights
└── .local/                           # Local environments and work records
```

Directory paths remain compatible with completed experiments. Local-only output paths appear as code references rather than broken public links. Existing source hashes describe the source versions used at execution time; later English documentation/report-template edits do not imply that model training was repeated.

## Status and data boundary

- **Prepared:** initial 33-paper source set, English notes/catalog/BibTeX; LibriSpeech training splits and all four dev/test splits.
- **Completed:** original AVS studies, follow-up controls, public Zipformer reproduction, actual full/band2 forecaster training, independent FGN and hybrid experiments, the small CPU component profile, the three-capacity fixed-case fitting diagnosis, and its 48-extra-epoch continuation/settings audit.
- **Proposed:** audio-conditioned residual correction, conditional encoder execution and cache recovery; new 960h acoustic training; capacity comparisons for the new corrector and broader training-cutoff coverage.
- **Not established:** lower total inference compute, lower irreversible commit latency, matched-test-risk Pareto superiority, live mobile performance, or a general FGN anti-overfitting effect.

The local corpus contains 292,367 utterances / 982.308 hours. The training-only union contains 281,241 utterances / 961.054 hours and excludes dev/test. Dataset preparation does not mean a 960h model was trained.

## Public documentation checks

```bash
python3 scripts/validate_catalog.py
python3 scripts/check_public_repository.py

# Also verify downloaded source PDFs when they are available locally.
python3 scripts/validate_catalog.py --pdfs
```

The public checker parses Python and JSON, checks English text and repository-local Markdown targets, and rejects accidental model/data assets or credential markers. These checks do not rerun model training or validate paper acceptance claims.

## Prepare source literature and data

```bash
python3 scripts/download_papers.py
python3 scripts/build_index.py
python3 scripts/prepare_librispeech.py

# Full 960h plus dev/test; reuse and verify existing downloads.
python3 scripts/prepare_librispeech.py --scope 960h --mirror elda --workers 2 --connections 16

# Existing corpus inventory and role-specific/NeMo manifests.
python3 scripts/prepare_librispeech.py --scope 960h --inventory-only
python3 scripts/organize_librispeech960.py
```

Paper indexing uses Python 3.9+; corpus preparation uses Python 3.12+ and `curl`. The inference/training environments have their own pinned requirements. Read the dataset guide before downloading hundreds of hours of audio.

## Strong-backbone reproduction entry points

```bash
python3 -m venv .local/venv-baseline
.local/venv-baseline/bin/python -m pip install -r requirements-zipformer-features.txt
.local/venv-baseline/bin/python scripts/prepare_zipformer_baseline.py
.local/venv-baseline/bin/python scripts/eval_zipformer_baseline.py

# Read-only batch-1 profiling; choose a fresh output file.
.local/venv-baseline/bin/python scripts/profile_zipformer_compute.py \
  --output experiments/compute_profile_v1/artifacts/profile_new.json
```

Full forecaster training, feature preparation, calibration and evaluation commands are in the [strong-backbone report](docs/zipformer_ours_2026-11-12.md). The [independent state study](experiments/fgn_state_v1/README.md) and [hybrid study](experiments/band_fgn_v1/README.md) document their own runners. Use new output directories rather than mixing new runs with preserved evidence.

## Script groups

| Task | Entry points |
|---|---|
| Literature | `build_index.py`, `download_papers.py`, `validate_catalog.py` |
| Public-file validation | `check_public_repository.py` |
| Data preparation | `prepare_librispeech.py`, `organize_librispeech960.py`, `prepare_features.py` |
| Original AVS training | `train_stage1.py`, `train_stage2.py`, `train_fgn.py`, `average.py` |
| Original decoding/analysis | `run_stage3.py`, `collect_events.py`, `analyze_stage3.py`, `analyze_final.py`, `analyze_final2.py` |
| Follow-up controls | `check_bounded_cache.py`, `long_stream_followup.py`, `repeat_predictors.py`, `stabilize_two_pass.py`, `prescale_checks.py` |
| Public baseline | `prepare_zipformer_baseline.py`, `eval_zipformer_baseline.py`, `check_zipformer_baseline.py` |
| Frozen features | `zipformer_features.py`, `prepare_zipformer_forecasts.py` |
| Our forecasters | `train_zipformer_forecaster.py`, `collect_zipformer_forecasts.py`, `native_zipformer_forecast.py` |
| Gate/evaluation | `analyze_zipformer_forecasts.py`, `run_zipformer_ours.py` |
| Reports | `report_followups.py`, `report_zipformer_baseline.py`, `report_zipformer_ours.py` |
| Component profiling | `profile_zipformer_compute.py` |

## Publication boundary

The Git repository contains documentation, source code, configuration, references and aggregate experimental artifacts. Original PDFs, raw speech/transcripts, feature caches, environments, downloaded weights, checkpoints and detailed per-utterance runs are excluded. Public model/corpus source links and preparation commands are retained for reproducibility.
