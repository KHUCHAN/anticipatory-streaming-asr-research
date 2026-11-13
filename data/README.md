# Experimental data location

We prepared three LibriSpeech learning splits and four dev/test sets up to the original sound, transcription, and speech manifest levels. The total is 292,367 utterance / 982.308 time, and the learning union is 281,241 utterance / 961.054 time. The final actual measurement status is summarized in [Verification Report ](reports/librispeech_inventory.md), and the data selection basis and access conditions are summarized in [Data·Mac Guide ](../docs/datasets_and_mac.md).

**LibriSpeech 960h** was selected as next week's benchmark for 2026-11-11. Additional train-clean-360 and train-other-500 have also completed the official MD5·decompression·voice/transcription response·split duplication check. [960h Data/Learning Setup Plan](../docs/librispeech960_benchmark_plan.md) and [Current Conclusion/Preparation Status](../docs/conclusion_and_next_steps.md)] are read separately.

```text
data/
├── README.md
├── datasets.json
├── reports/
│   ├── librispeech_inventory.json   # Audio, text, official checksums and counts
│   ├── librispeech_inventory.md     # Human-readable verification
│   ├── librispeech960_protocol.json # Speakers, utterances and hours by dev role
│   └── librispeech960_organization.json # Manifest hashes and NeMo conversion locations
├── raw/librispeech/                 # Archives, original FLAC and text; Git-excluded
└── processed/
    ├── manifests/ # 7 split JSONL + learning union train-960.jsonl; Except Git
    │ └── nemo/ # NeMo format for full split and train-960
    └── protocols/librispeech960_v1/ # Assignment of local ID of dev role for each speaker; Except Git
        ├── dev_partitions.json
        ├── manifests/ # native JSONL by dev role
        └── nemo/ # NeMo JSONL by dev role
```

Record corpus version, split, speaker/session and original utterance IDs, alignment model/version, normalization and download date in manifests. Split original utterances into train/dev/test before generating chunks.

Manifests record relative audio paths, transcripts, samples, duration, speakers and chapters. Historical 100h tokenizer/features/CTC alignment/weights were generated separately. The tree describes audio/manifests/protocols, not every feature file. Training-union, NeMo and dev-role lists are prepared with original splits/text preserved. New 960h tokenizer/features/statistics remain unprepared; do not count 100h outputs as 960h outputs.

## Purpose of Data List

| Location | Purpose |
|---|---|
| `raw/librispeech/archives/` | Official compressed original and transfer section |
| `raw/librispeech/LibriSpeech/` | FLAC/transcription that maintains the original split/speaker/chapter structure |
| `processed/manifests/<split>.jsonl` | 7 official split list; The voice path is relative to the storage root |
Learning union of | `processed/manifests/train-960.jsonl` | train-clean-100 → train-clean-360 → train-other-500; Excluding dev/test |
| `processed/manifests/nemo/` | Full split and learning union converted to NeMo fields `audio_filepath`, `duration`, `text` |
| `processed/protocols/librispeech960_v1/dev_partitions.json` | Assignment of dev roles for each existing seed/speaker; List of original IDs, excluding Git |
| `processed/protocols/librispeech960_v1/manifests/` | List by dev role; clean/other 6 each and 3 combined by role |
NeMo format | in the same dev role list as | `processed/protocols/librispeech960_v1/nemo/` |

The dev role is `model_selection` (model/learning settings selection), `gate_fit` (gate fitting), and `gate_calibration` (reference value correction). Speakers and utterances between roles are checked to ensure that they do not overlap. test-clean/test-other are not included in this dev role list or training union.

| dev role, clean/other combined | Number of utterances | Number of speakers | Time |
|---|---:|---:|---:|
| Select Model | 2,856 | 37 | 5.316 |
| Gate suitable | 1,297 | 18 | 2.604 |
| Gate reference value correction | 1,414 | 18 | 2.589 |

Preserve uppercase original transcripts without duration filtering, normalization or resampling. Original audio is 16kHz mono 16-bit FLAC. NeMo manifests use **absolute audio paths** on this machine; rerun organization at the destination after moving data/repository. Original manifests keep relative paths; individual transcripts/lists remain Git-excluded.

## Reproduction command and verification results

```bash
python3 scripts/prepare_librispeech.py --scope 960h --mirror elda --workers 2 --connections 16
python3 scripts/organize_librispeech960.py
```

Run both commands from the repository root. Organization does not redownload audio or change existing speaker assignments. A fresh checkout uses the configured seed and speaker counts to recreate the same roles.

- [Original sound, checksum, voice/transcription response, length verification](reports/librispeech_inventory.md)
- [Tagation of speakers, utterances, and time by dev role](reports/librispeech960_protocol.json)
- [Number per manifest, length, SHA-256, conversion location ](reports/librispeech960_organization.json)

This organization task excludes full-waveform decoding checks, new 960h tokenizer/features/statistics and expanded training. NeMo schema conversion prepares manifests; actual NeMo loading/training requires separate verification.

Conversion schema: [official NeMo ASR manifests](https://docs.nvidia.com/nemo/speech/latest/asr/datasets.html).

2026-11-11 In the local disk tally, `data/raw/librispeech/`, which combines the compressed original, decompressed audio, and transcription, is approximately 117GiB. The existing 100h feature model and new manifest space are not included in this value.

## Cache of our forecast experiments above fixed Zipformer — 2026-11-12

To `data/processed/zipformer_forecasts/`, train-clean-100 28,539 pieces, dev-clean 2,703 pieces, dev-other 2,864 pieces, test-clean 2,620 pieces, test-other 2,939 pieces, total **39,665** utterance cache has been prepared. The 512-dimensional output (fp16) of the formal fixed encoder, native token, emission frame, causal available time, reference BPE and approximate learning time labels are stored separately. It is approximately 11GiB and does not duplicate the original FLAC.

This cache supports 100h forecaster training and dev/test, not newly generated full-960h features, BPE 1024 or normalization. Use public-backbone BPE 500 and existing speaker-separated model-selection/gate-fit/calibration roles. Native fbank exactly matches the official extractor; inputs use only available encoder prefixes. Future gold/native information is offline supervision/evaluation only.

Preparation, training, gates, full evaluation and code locations are in [our experiment](../docs/zipformer_ours_2026-11-12.md). Models, features, transcripts and raw events follow existing ignore rules.


### Using data from AR + FGN combined pilot — 2026-11-12

`experiments/band_fgn_v1/` reused the existing Zipformer features and immutable packed encoder of `runs/fgn_state_v1/cache/`. Learning is train-clean-100 overall 28,539 utterance, separate model-selection256 utterance, gate-fit1,297 / calibration1,414 utterance, and overall test5,559 utterance. For the learning sample, 1 token positions per utterance are newly selected every epoch. 960h Distinguishes from new sound learning and full 960h feature extraction. Additional gold token/timing metadata was placed in `runs/band_fgn_v1/gold_metadata.json`, and the original sound, existing features, and manifest were not changed. [Results and limitations](../docs/band_fgn_hybrid_2026-11-12.md)
