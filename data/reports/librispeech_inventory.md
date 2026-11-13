# LibriSpeech Data Preparation Verification Results

Verification run time: 2026-11-11T22:11:16+00:00 (UTC). Source: [OpenSLR SLR12](https://www.openslr.org/12), CC BY 4.0.

| Split | Number of utterances | Number of speakers | Voice length (hours) |
|---|---:|---:|---:|
| train-clean-100 | 28,539 | 251 | 100.591 |
| train-clean-360 | 104,014 | 921 | 363.606 |
| train-other-500 | 148,688 | 1166 | 496.858 |
| dev-clean | 2,703 | 40 | 5.388 |
| dev-other | 2,864 | 33 | 5.121 |
| test-clean | 2,620 | 40 | 5.403 |
| test-other | 2,939 | 33 | 5.342 |

Total **292,367 utterances / 982.308 hours**. Original archives occupy 61.358 GB.

- The collection file matches the formula MD5 of the selected 7 compressed files.
- We read the STREAMINFO of all FLACs and checked the number and length of 16 kHz·mono·16-bit, samples.
- All voices and transcription IDs correspond one-to-one, and there are no duplicate or empty transcriptions.
- There is no duplication of utterance ID and speaker duplication between splits.
- A decoding inspection of the entire waveform was not performed. The original payload was confirmed using the compressed file checksum.

Originals: `data/raw/librispeech/LibriSpeech/`. Utterance manifests: `data/processed/manifests/`.

This report verifies raw audio, transcripts and utterance metadata. Historical 100h runs remain separate; this collection did not generate new tokenizers/features/alignments or start model training.

[Full Machine Readable Report](librispeech_inventory.json) · [Data Guide](../../docs/datasets_and_mac.md)

Union of three training splits: **281,241 utterances / 961.054 hours / 2,338 speakers** in `data/processed/manifests/train-960.jsonl`, preserving original split fields and excluding dev/test.
