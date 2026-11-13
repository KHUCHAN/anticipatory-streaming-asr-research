# Data Preparation and Mac Running Conditions

## preparation range

**2026-11-11 Range Change:** User chose LibriSpeech 960h as the main benchmark. Collection and verification of three training splits + four dev/test splits has been completed. The learning union is 281,241 utterance / 961.054 time. [960h follows data and learning setup plan ](librispeech960_benchmark_plan.md) and [new dev protocol aggregation ](../data/reports/librispeech960_protocol.json). Distinguish between the output of the existing 100h experiment and the preparation of the original sound/manifest of this 960h.

Initial preparation selected train-clean-100 and four dev/test splits; subsequent requests expanded to full 960h training data. This covers FLAC, transcripts and utterance manifests. New 960h tokenizer, feature caches, alignments and expanded training are separate work.

| Data | Official scale/characteristics | Role/status in this work |
|---|---|---|
| LibriSpeech | train 100/360/500 time split, 16 kHz English recitation | full 960h train + dev-clean/dev-other/test-clean/test-other actual preparation; The final result is [inventory](../data/reports/librispeech_inventory.md) |
| GigaSpeech | 10/250/1,000/2,500/10,000 time subset, English multi-domain | Preparation of literature and distribution path for expansion; Application/access consent/original music download not performed |
| Common Voice | v27 scripted All language 42,593 time, 295 language | language·For expansion to be reviewed after determining the validated subset; Original sound download not performed |
| AMI | About 100 hour meeting, multiple microphones, speakers and time annotation | Dialogue, overlap, intonation evaluation candidate; Not included in this time selection range 100 |

Sources: [OpenSLR](https://www.openslr.org/12), [GigaSpeech distribution](https://github.com/SpeechColab/GigaSpeech), [Mozilla v27 announcement](https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490), [AMI overview](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml).

The compressed size of train-clean-100 in the LibriSpeech official table is 6.3G. The total number of downloads, utterances, and lengths in this work use the actual measured values ​​of inventory. Distinguish between compressed size and disk usage after decompression. The original MP3·book text of the entire 960 time corpus is not automatically received.

## preparation script

Use Python 3.12+ and `curl`. Do not install the Python learning library.

```bash
python3 scripts/prepare_librispeech.py
python3 scripts/prepare_librispeech.py --scope 960h --mirror elda --workers 2 --connections 16
python3 scripts/prepare_librispeech.py --scope 960h --inventory-only
```

Downloads check HTTPS byte-range support and resume safely. Compare official `md5sum.txt` with reviewed checksums before extracting. Inspect FLAC headers for sample rate, channels, sample count and bit depth; verify exact transcript/audio ID correspondence and cross-split speaker separation. This is not decoding every waveform.

Default preparation retains 100h+dev/test. `--scope 960h` prepares three training and four dev/test splits. `--mirror` selects official OpenSLR primary/ELDA/TRMAL distributions and reuses completed byte ranges when switching mirrors. Full preparation combines only training splits in `data/processed/manifests/train-960.jsonl`, retaining original split fields. Use the same `--scope 960h` when rechecking.

Utterance manifests contain `dataset`, `split`, `utterance_id`, `speaker_id`, `chapter_id`, `audio`, `text`, `sample_rate`, `channels`, `num_samples` and `duration_seconds`. Use relative paths and preserve uppercase original transcripts. Raw audio/manifests are ignored; publish aggregate reports and reproduction scripts.

## What we found in M5 Pro 24GB

2026-11-10 local read inspection confirmed Apple M5 Pro, 24 GiB integrated memory, arm64, PyTorch 2.12.0, MPS built/available. No hardware identification numbers or account information were recorded. Package installation/upgrade and model execution were not performed.

[PyTorch 2.14 official announcement](https://pytorch.org/blog/pytorch-2-14-release-blog/) explains MPS's CTC Loss forward/backward support. Before actual adoption, small calculation verification and memory and time measurements are required in a separate environment. It does not assume the same support from installed 2.12.0. Support for CUDA-specific fused scan, transducer loss, and decoding extension is also subject to separate confirmation.

24 GiB is shared by the OS and CPU/GPU. Parameters alone do not determine learning possibilities or time required. Next, the cost of input length, activation, optimizer state, sorting loss, and fallback must be measured in the selected structure.

## external comparison model

[Zipformer 20M streaming ONNX official guide ](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html) and [model card ](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17) are public. This is an optional external reference and no weighting or inference was performed in this work.

Separate comparisons among models trained on the same 100 hours from external pretrained-model comparisons. ONNX CPU versus PyTorch MPS timings also reflect runtime backends and cannot isolate an algorithmic effect.

[Research objectives and methodology](goal_and_methodology.md) · [Data registry](../data/datasets.json)
